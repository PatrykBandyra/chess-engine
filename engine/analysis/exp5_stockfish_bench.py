"""
Experiment 5 specific analysis: Stockfish benchmark.

Combines analysis_*.csv from analyze_experiment.py with stockfish_reval.csv
(if available) and produces per-variant performance summary.

Key outputs:
  - exp5_variant_summary.csv      — per-variant: score vs each SF skill, est. Elo, ACPL
  - exp5_variant_summary.txt      — human-readable
  - exp5_score_curve.png          — variant score vs SF Elo (one line per variant)
  - exp5_acpl_by_variant.png      — ACPL per variant (bar chart, if reval available)
  - exp5_acpl_by_phase.png        — ACPL stratified by game phase
  - exp5_blunder_rate.png         — blunder rate per variant

Usage:
    python exp5_stockfish_bench.py <experiment_dir>
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_RE = re.compile(r'^(?P<variant>.+)_vs_stockfish_sk(?P<skill>\d+)$')


def parse_label(label: str) -> dict | None:
    m = LABEL_RE.match(label)
    if not m:
        return None
    return {'variant': m.group('variant'), 'skill': int(m.group('skill'))}


def load_sf_skill_elo(path: Path) -> dict[int, float]:
    if not path.is_file():
        # Fallback default mapping
        return {0: 800, 3: 1100, 5: 1400, 8: 1700, 10: 2000, 13: 2400, 15: 2700, 20: 3500}
    df = pd.read_csv(path)
    return dict(zip(df['skill'].astype(int), df['approx_elo'].astype(float)))


def aggregate_variant_scores(games_df: pd.DataFrame, sf_elo: dict[int, float]) -> pd.DataFrame:
    """For each (variant, sf_skill), compute games / wins / draws / losses / score."""
    if games_df.empty:
        return pd.DataFrame()

    rows = []
    for label, group in games_df.groupby('label'):
        parsed = parse_label(label)
        if not parsed:
            continue

        # Determine who is the variant (in our matchups, white is the variant)
        # But with swap colors, the variant could be on either side.
        # We need to look at the file naming to know if it was swapped.
        # Without that info per-row in games_df, we assume the variant's
        # score is the matchup-level "white_score" if not swapped, else 1 - white_score.
        # The label encodes <variant>_vs_stockfish, so:
        #   variant plays white → its score = score for "1-0" results
        #   variant plays black → its score = score for "0-1" results
        # Since the analyze_experiment.py aggregates by label without splitting by swap,
        # we recompute here from games_df, which has 'file' field containing 'orig|swap'.

        wins_as_variant = 0
        draws = 0
        losses_as_variant = 0
        total = 0

        for _, row in group.iterrows():
            file_name = str(row.get('file', ''))
            is_swap = '_swap' in file_name
            result = row.get('result')
            total += 1
            if result == '1/2-1/2':
                draws += 1
            elif result == '1-0':
                # White wins. Variant plays white if not swapped.
                if is_swap:
                    losses_as_variant += 1
                else:
                    wins_as_variant += 1
            elif result == '0-1':
                if is_swap:
                    wins_as_variant += 1
                else:
                    losses_as_variant += 1

        score = (wins_as_variant + 0.5 * draws) / total if total > 0 else 0.0

        rows.append({
            'variant': parsed['variant'],
            'sf_skill': parsed['skill'],
            'sf_elo': sf_elo.get(parsed['skill'], np.nan),
            'games': total,
            'variant_wins': wins_as_variant,
            'draws': draws,
            'variant_losses': losses_as_variant,
            'variant_score': round(score, 3),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['variant', 'sf_skill']).reset_index(drop=True)


def interpolate_variant_elo(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each variant, estimate Elo by finding the SF skill level where
    variant scores 0.5, interpolating between the known SF Elo points.
    """
    rows = []
    for variant, group in scores_df.groupby('variant'):
        g = group.sort_values('sf_elo').reset_index(drop=True)
        scores = g['variant_score'].values
        elos = g['sf_elo'].values

        if len(g) < 2:
            est_elo = float(elos[0]) if len(g) == 1 else np.nan
        elif (scores > 0.5).all():
            # Variant beats even the strongest SF — Elo above max
            est_elo = float(elos[-1])
        elif (scores < 0.5).all():
            est_elo = float(elos[0])
        else:
            # Find bracket where score crosses 0.5
            est_elo = None
            for i in range(len(g) - 1):
                s1, s2 = scores[i], scores[i + 1]
                e1, e2 = elos[i], elos[i + 1]
                if (s1 - 0.5) * (s2 - 0.5) <= 0 and abs(s1 - s2) > 1e-6:
                    # Linear interpolation in (score, elo) space
                    t = (0.5 - s1) / (s2 - s1)
                    est_elo = float(e1 + t * (e2 - e1))
                    break
            if est_elo is None:
                est_elo = float(np.interp(0.5, scores, elos))

        rows.append({
            'variant': variant,
            'estimated_elo': round(est_elo, 0) if not np.isnan(est_elo) else np.nan,
            'matchups': len(g),
        })

    return pd.DataFrame(rows).sort_values('estimated_elo', ascending=False).reset_index(drop=True)


def merge_acpl(scores_df: pd.DataFrame, reval_df: pd.DataFrame) -> pd.DataFrame:
    """Merge reval ACPL into variant scores (variant-side ACPL only)."""
    if reval_df.empty:
        return scores_df

    rows = []
    for _, sc in scores_df.iterrows():
        # Find all reval rows for this matchup
        matchup_label = f"{sc['variant']}_vs_stockfish_sk{sc['sf_skill']}"
        m = reval_df[reval_df['label'] == matchup_label]

        if m.empty:
            sc = sc.copy()
            sc['acpl_variant'] = np.nan
            sc['blunders_variant'] = np.nan
            rows.append(sc)
            continue

        # For each game, variant ACPL depends on whether swapped
        variant_acpls = []
        variant_blunders = []
        variant_moves = []
        for _, row in m.iterrows():
            if row.get('swapped', False):
                variant_acpls.append(row['acpl_black'])
                variant_blunders.append(row['blunders_black'])
                variant_moves.append(row.get('moves_black', 0))
            else:
                variant_acpls.append(row['acpl_white'])
                variant_blunders.append(row['blunders_white'])
                variant_moves.append(row.get('moves_white', 0))

        sc = sc.copy()
        sc['acpl_variant'] = round(np.mean(variant_acpls), 2) if variant_acpls else np.nan
        sc['blunders_variant'] = round(np.mean(variant_blunders), 2) if variant_blunders else np.nan
        sc['avg_moves_variant'] = round(np.mean(variant_moves), 1) if variant_moves else np.nan
        rows.append(sc)

    return pd.DataFrame(rows)


def plot_score_curve(scores_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if scores_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for variant, group in scores_df.groupby('variant'):
        g = group.sort_values('sf_elo')
        ax.plot(g['sf_elo'], g['variant_score'], 'o-', label=variant,
                linewidth=2, markersize=8)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='break-even')
    ax.set_xlabel('Stockfish skill Elo (approximate)')
    ax.set_ylabel('Variant score (1=win, 0.5=draw, 0=loss)')
    ax.set_title('Variant performance vs Stockfish skill level')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_acpl_bars(scores_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'acpl_variant' not in scores_df.columns:
        return
    df = scores_df.dropna(subset=['acpl_variant'])
    if df.empty:
        return

    grouped = df.groupby('variant')['acpl_variant'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped.index, grouped.values, color='#1976D2')
    ax.set_ylabel('Average ACPL (centipawn loss per move)')
    ax.set_title('Variant ACPL (lower = closer to Stockfish optimal play)')
    plt.xticks(rotation=20, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_blunder_rate(scores_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'blunders_variant' not in scores_df.columns or 'avg_moves_variant' not in scores_df.columns:
        return
    df = scores_df.dropna(subset=['blunders_variant', 'avg_moves_variant'])
    if df.empty:
        return

    df = df.copy()
    df['blunder_rate'] = df['blunders_variant'] / df['avg_moves_variant'].replace(0, np.nan)
    grouped = df.groupby('variant')['blunder_rate'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped.index, grouped.values * 100, color='#D32F2F')
    ax.set_ylabel('Blunder rate (%)')
    ax.set_title('Blunder rate per variant (moves with CP loss > 100)')
    plt.xticks(rotation=20, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Exp 5 — Stockfish benchmark analysis')
    parser.add_argument('experiment_dir', type=str)
    parser.add_argument('--sf-elo-csv', type=str, default='',
                        help='SF skill->Elo mapping (default: experiments/exp5/_sf_skill_elo.csv)')
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_dir():
        print(f"Error: {exp_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    games_csv = exp_dir / 'analysis_games.csv'
    if not games_csv.exists():
        print(f"Error: missing {games_csv}", file=sys.stderr)
        print(f"  Run analyze_experiment.py first.", file=sys.stderr)
        sys.exit(1)

    # SF skill -> Elo mapping
    if args.sf_elo_csv:
        sf_elo_path = Path(args.sf_elo_csv)
    else:
        sf_elo_path = exp_dir.parent.parent / 'experiments' / 'exp5' / '_sf_skill_elo.csv'
    sf_elo = load_sf_skill_elo(sf_elo_path)
    print(f"Loaded SF skill->Elo mapping: {sf_elo}")

    # Load games
    games_df = pd.read_csv(games_csv)
    print(f"Loaded {len(games_df)} games from {games_csv.name}")

    # Aggregate
    print("Aggregating variant scores vs each SF skill level...")
    scores_df = aggregate_variant_scores(games_df, sf_elo)

    if scores_df.empty:
        print("No exp5 matchups found (labels must match <variant>_vs_stockfish_skN).")
        sys.exit(0)

    # Merge ACPL if reval available
    reval_csv = exp_dir / 'stockfish_reval.csv'
    if reval_csv.exists():
        print(f"Loading {reval_csv.name}...")
        reval_df = pd.read_csv(reval_csv)
        scores_df = merge_acpl(scores_df, reval_df)
    else:
        print(f"No {reval_csv.name} found — skipping ACPL analysis (run stockfish_reval.py to generate)")

    # Save summary
    out_csv = exp_dir / 'exp5_variant_summary.csv'
    scores_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv.name}")

    # Interpolated Elo
    elo_df = interpolate_variant_elo(scores_df)
    elo_csv = exp_dir / 'exp5_variant_elo.csv'
    elo_df.to_csv(elo_csv, index=False)
    print(f"  Saved: {elo_csv.name}")

    print()
    print('--- Per-variant performance ---')
    print(scores_df.to_string(index=False))
    print()
    print('--- Estimated Elo (interpolated from SF skill->Elo) ---')
    print(elo_df.to_string(index=False))
    print()

    # Plots
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print("Generating plots...")
    plot_score_curve(scores_df, plots_dir / 'exp5_score_curve.png')
    if 'acpl_variant' in scores_df.columns:
        plot_acpl_bars(scores_df, plots_dir / 'exp5_acpl_by_variant.png')
        plot_blunder_rate(scores_df, plots_dir / 'exp5_blunder_rate.png')
    print(f"  Plots saved to: {plots_dir}")

    # Human-readable summary
    txt_path = exp_dir / 'exp5_variant_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Experiment 5 — Stockfish benchmark\n')
        f.write('=' * 60 + '\n\n')
        f.write('Stockfish skill -> Elo mapping:\n')
        for k, v in sorted(sf_elo.items()):
            f.write(f'  skill {k:2d}: ~{int(v)} Elo\n')
        f.write('\n')
        f.write('Variant scores vs each SF skill:\n')
        f.write(scores_df.to_string(index=False) + '\n\n')
        f.write('Estimated variant Elo (interpolation at score=0.5):\n')
        f.write(elo_df.to_string(index=False) + '\n')
    print(f"  Saved: {txt_path.name}")

    print('\nDone.')


if __name__ == '__main__':
    main()
