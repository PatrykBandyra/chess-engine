"""
Experiment 1 specific analysis: round-robin between 4 engine variants.

Consumes output of analyze_experiment.py (analysis_games.csv) and produces:

  - exp1_pair_significance.csv  — per-pair binomial test on White wins vs draws+losses
  - exp1_axis_summary.csv       — main effects on axes A (algorithm) and B (evaluator)
  - exp1_color_advantage.csv    — overall White vs Black win rate
  - exp1_round_robin_summary.txt — human-readable
  - plots/exp1_pair_significance.png — bar chart per pair with CI
  - plots/exp1_axis_a_effect.png — Minimax vs MCTS aggregate
  - plots/exp1_axis_b_effect.png — TRAD vs NN aggregate
  - plots/exp8_wdl_matrix.png    — 4x4 result matrix

The 4 variants:
  MINIMAX_TRAD, MINIMAX_NN, MCTS_TRAD, MCTS_NN

Axis A (algorithm): Minimax vs MCTS
Axis B (evaluator): TRAD    vs NN

Usage:
    python exp1_round_robin.py <experiment_dir>
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Variants in the round-robin
VARIANTS = ['minimax_trad', 'minimax_nn', 'mcts_trad', 'mcts_nn']


def algorithm_of(variant: str) -> str:
    """Returns 'MINIMAX' or 'MCTS' based on variant name."""
    return 'MINIMAX' if variant.startswith('minimax_') else 'MCTS'


def evaluator_of(variant: str) -> str:
    """Returns 'TRAD' or 'NN' based on variant name."""
    return 'TRAD' if variant.endswith('_trad') else 'NN'


def parse_label(label: str) -> tuple[str, str] | None:
    """Splits matchup label into (white_variant, black_variant)."""
    parts = label.split('_vs_')
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def aggregate_per_pair(games_df: pd.DataFrame) -> pd.DataFrame:
    """For each matchup label: W/D/L counts + variant-perspective score (handles swap)."""
    rows = []
    for label, group in games_df.groupby('label'):
        parsed = parse_label(label)
        if not parsed:
            continue
        white_var, black_var = parsed

        wv_wins = 0  # variant on white side (white_var) wins
        bv_wins = 0  # variant on black side (black_var) wins
        draws = 0
        total = len(group)

        for _, row in group.iterrows():
            file_name = str(row.get('file', ''))
            swapped = '_swap' in file_name
            result = row.get('result')
            # In a swapped game, the players exchange colors:
            #   - white_var actually plays black
            #   - black_var actually plays white
            if result == '1/2-1/2':
                draws += 1
            elif result == '1-0':
                # The variant that played white wins
                if swapped:
                    bv_wins += 1
                else:
                    wv_wins += 1
            elif result == '0-1':
                if swapped:
                    wv_wins += 1
                else:
                    bv_wins += 1

        score_wv = (wv_wins + 0.5 * draws) / total if total > 0 else 0.0
        rows.append({
            'label': label,
            'variant_a': white_var,
            'variant_b': black_var,
            'games': total,
            'a_wins': wv_wins,
            'draws': draws,
            'b_wins': bv_wins,
            'a_score': round(score_wv, 3),
        })
    return pd.DataFrame(rows)


def binomial_test_pair(a_wins: int, b_wins: int, draws: int) -> dict:
    """
    Tests H0: variant A and B are equal strength.
    Uses binomial test on decisive games (excludes draws).
    """
    from scipy.stats import binomtest

    decisive = a_wins + b_wins
    if decisive == 0:
        return {'decisive': 0, 'p_value': None, 'lower_ci': None, 'upper_ci': None,
                'note': 'no decisive games'}
    result = binomtest(a_wins, decisive, 0.5)
    ci = result.proportion_ci(confidence_level=0.95)
    return {
        'decisive': int(decisive),
        'a_win_rate_decisive': round(a_wins / decisive, 3),
        'p_value': round(float(result.pvalue), 4),
        'lower_ci': round(float(ci.low), 3),
        'upper_ci': round(float(ci.high), 3),
        'note': 'binomial test on decisive games',
    }


def axis_effect(pair_df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """
    Aggregates pair results along an axis.
    axis='algorithm' → MINIMAX vs MCTS (across both evaluators)
    axis='evaluator' → TRAD vs NN (across both algorithms)
    """
    rows = []
    if axis == 'algorithm':
        groups = {'MINIMAX_vs_MCTS': lambda v1, v2: (algorithm_of(v1) == 'MINIMAX' and algorithm_of(v2) == 'MCTS')
                                                      or (algorithm_of(v1) == 'MCTS' and algorithm_of(v2) == 'MINIMAX')}
        get_class = algorithm_of
    elif axis == 'evaluator':
        groups = {'TRAD_vs_NN': lambda v1, v2: (evaluator_of(v1) == 'TRAD' and evaluator_of(v2) == 'NN')
                                                 or (evaluator_of(v1) == 'NN' and evaluator_of(v2) == 'TRAD')}
        get_class = evaluator_of
    else:
        raise ValueError(f"Unknown axis: {axis}")

    # Aggregate: for each pair that has cross-axis variants, accumulate stats from
    # the perspective of class X (= MINIMAX or TRAD) vs class Y (= MCTS or NN).
    if axis == 'algorithm':
        class_x, class_y = 'MINIMAX', 'MCTS'
    else:
        class_x, class_y = 'TRAD', 'NN'

    x_wins = 0
    y_wins = 0
    draws = 0
    pairs_used = []

    for _, row in pair_df.iterrows():
        va, vb = row['variant_a'], row['variant_b']
        if get_class(va) == get_class(vb):
            continue  # same class — not a cross-axis comparison
        pairs_used.append(row['label'])

        # Determine which variant is class X
        if get_class(va) == class_x:
            x_wins += row['a_wins']
            y_wins += row['b_wins']
        else:
            x_wins += row['b_wins']
            y_wins += row['a_wins']
        draws += row['draws']

    total = x_wins + y_wins + draws
    if total == 0:
        return pd.DataFrame()

    rows.append({
        'axis': axis,
        f'{class_x}_wins': int(x_wins),
        'draws': int(draws),
        f'{class_y}_wins': int(y_wins),
        'total_games': int(total),
        f'{class_x}_score': round((x_wins + 0.5 * draws) / total, 3),
        'pairs_used': '; '.join(pairs_used),
    })
    return pd.DataFrame(rows)


def color_advantage(games_df: pd.DataFrame) -> pd.DataFrame:
    """Overall White wins vs Black wins vs draws across all games."""
    total = len(games_df)
    white = (games_df['result'] == '1-0').sum()
    draws = (games_df['result'] == '1/2-1/2').sum()
    black = (games_df['result'] == '0-1').sum()
    return pd.DataFrame([{
        'total_games': int(total),
        'white_wins': int(white),
        'draws': int(draws),
        'black_wins': int(black),
        'white_score': round((white + 0.5 * draws) / total, 3) if total > 0 else 0.0,
    }])


def plot_pair_significance(sig_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if sig_df.empty:
        return
    df = sig_df.dropna(subset=['a_win_rate_decisive']).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.3), 5))
    x = np.arange(len(df))
    rates = df['a_win_rate_decisive'].values
    lower = df['lower_ci'].values
    upper = df['upper_ci'].values
    yerr = np.array([rates - lower, upper - rates])
    colors = ['#4CAF50' if p < 0.05 else '#9E9E9E' for p in df['p_value']]

    ax.bar(x, rates, color=colors, alpha=0.7, edgecolor='black')
    ax.errorbar(x, rates, yerr=yerr, fmt='none', color='black', capsize=4)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='H0: equal strength')
    ax.set_xticks(x)
    labels = [f"{row['variant_a']}\nvs\n{row['variant_b']}\np={row['p_value']}"
              for _, row in df.iterrows()]
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Variant A win rate (decisive games)')
    ax.set_title('Pair significance (95% CI). Green = p<0.05')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_axis_effect(axis_df: pd.DataFrame, axis: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if axis_df.empty:
        return

    row = axis_df.iloc[0]
    if axis == 'algorithm':
        labels = ['MINIMAX', 'MCTS']
        wins = [row['MINIMAX_wins'], row['MCTS_wins']]
    else:
        labels = ['TRAD', 'NN']
        wins = [row['TRAD_wins'], row['NN_wins']]

    draws = row['draws']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels + ['Draws'], wins + [draws],
                  color=['#1976D2', '#D32F2F', '#9E9E9E'])
    ax.set_ylabel('Games')
    ax.set_title(f'Axis {axis.upper()} main effect (aggregated across other axis)')
    for bar, val in zip(bars, wins + [draws]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{int(val)}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wdl_matrix(pair_df: pd.DataFrame, out_path: Path) -> None:
    """4x4 heatmap of variant_a score against variant_b."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if pair_df.empty:
        return

    variants = sorted(set(pair_df['variant_a']) | set(pair_df['variant_b']))
    idx = {v: i for i, v in enumerate(variants)}
    n = len(variants)
    matrix = np.full((n, n), np.nan)

    for _, row in pair_df.iterrows():
        i = idx[row['variant_a']]
        j = idx[row['variant_b']]
        matrix[i, j] = row['a_score']
        # Mirror with complement
        matrix[j, i] = 1 - row['a_score']

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(variants, rotation=20, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(variants)
    ax.set_xlabel('Opponent')
    ax.set_ylabel('Variant')
    ax.set_title('Score matrix (row variant\'s score vs column variant)')
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='black' if 0.2 < val < 0.8 else 'white', fontsize=10)
    fig.colorbar(im, ax=ax, label='Variant score')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Exp 1 — Round-robin specific analysis')
    parser.add_argument('experiment_dir', type=str)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_dir():
        print(f"Error: {exp_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    games_csv = exp_dir / 'analysis_games.csv'
    if not games_csv.exists():
        print(f"Error: missing {games_csv}", file=sys.stderr)
        print("Run analyze_experiment.py first.", file=sys.stderr)
        sys.exit(1)

    games_df = pd.read_csv(games_csv)
    print(f"Loaded {len(games_df)} games from {games_csv.name}")

    # Per-pair aggregation
    pair_df = aggregate_per_pair(games_df)
    if pair_df.empty:
        print("No exp1 pairs found.", file=sys.stderr)
        sys.exit(1)

    # Significance tests
    sig_rows = []
    for _, row in pair_df.iterrows():
        sig = binomial_test_pair(row['a_wins'], row['b_wins'], row['draws'])
        sig_rows.append({**row.to_dict(), **sig})
    sig_df = pd.DataFrame(sig_rows)
    sig_csv = exp_dir / 'exp1_pair_significance.csv'
    sig_df.to_csv(sig_csv, index=False)
    print(f"Saved: {sig_csv.name}")

    print()
    print('--- Pair significance (binomial test on decisive games) ---')
    cols = ['variant_a', 'variant_b', 'games', 'a_wins', 'draws', 'b_wins',
            'a_score', 'a_win_rate_decisive', 'p_value', 'lower_ci', 'upper_ci']
    print(sig_df[[c for c in cols if c in sig_df.columns]].to_string(index=False))

    # Axis effects
    axis_a_df = axis_effect(pair_df, 'algorithm')
    axis_b_df = axis_effect(pair_df, 'evaluator')
    axis_combined = pd.concat([axis_a_df, axis_b_df], ignore_index=True)
    if not axis_combined.empty:
        axis_csv = exp_dir / 'exp1_axis_summary.csv'
        axis_combined.to_csv(axis_csv, index=False)
        print(f"\nSaved: {axis_csv.name}")
        print()
        print('--- Axis A (algorithm) and axis B (evaluator) main effects ---')
        print(axis_combined.to_string(index=False))

    # Color advantage
    color_df = color_advantage(games_df)
    color_csv = exp_dir / 'exp1_color_advantage.csv'
    color_df.to_csv(color_csv, index=False)
    print(f"\nSaved: {color_csv.name}")
    print()
    print('--- Color advantage (overall) ---')
    print(color_df.to_string(index=False))

    # Plots
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print('\nGenerating plots...')
    plot_pair_significance(sig_df, plots_dir / 'exp1_pair_significance.png')
    if not axis_a_df.empty:
        plot_axis_effect(axis_a_df, 'algorithm', plots_dir / 'exp1_axis_a_effect.png')
    if not axis_b_df.empty:
        plot_axis_effect(axis_b_df, 'evaluator', plots_dir / 'exp1_axis_b_effect.png')
    plot_wdl_matrix(pair_df, plots_dir / 'exp8_wdl_matrix.png')
    print(f"  Plots saved to: {plots_dir}")

    # Human-readable summary
    txt_path = exp_dir / 'exp1_round_robin_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Experiment 1 — Round-Robin between 4 engine variants\n')
        f.write('=' * 60 + '\n\n')
        f.write('Per-pair results (variant A = white_label, variant B = black_label):\n')
        f.write(sig_df.to_string(index=False) + '\n\n')
        if not axis_combined.empty:
            f.write('Axis main effects:\n')
            f.write(axis_combined.to_string(index=False) + '\n\n')
        f.write('Color advantage:\n')
        f.write(color_df.to_string(index=False) + '\n')
    print(f"  Saved: {txt_path.name}")

    print('\nDone.')


if __name__ == '__main__':
    main()
