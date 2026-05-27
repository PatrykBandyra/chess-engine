"""
Experiment 7 specific analysis: opening book impact.

Consumes output of analyze_experiment.py and the raw metrics JSONL files
(to count `from_book: true` markers) from an exp7 directory and produces:

  - exp7_summary.csv            — W/D/L + book metrics per condition
  - exp7_summary.txt            — human-readable
  - exp7_statistical_tests.txt  — chi-square and McNemar test results
  - plots/exp7_wdl_comparison.png    — bar chart book OFF vs ON
  - plots/exp7_book_exit_hist.png    — book exit move distribution
  - plots/exp7_opening_time.png      — opening phase time comparison

Usage:
    python exp7_book_impact.py <experiment_dir>
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION_RE = re.compile(r'^(?P<engine>minimax_trad_d\d+|mcts_trad)_book_(?P<book>on|off)$')


def parse_condition(label: str) -> dict | None:
    m = CONDITION_RE.match(label)
    if not m:
        return None
    return {'engine': m.group('engine'), 'book': m.group('book')}


def load_metrics_moves(path: Path) -> tuple[list[dict], dict | None]:
    moves = []
    summary = None
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('type') == 'game_summary':
                summary = rec
            else:
                moves.append(rec)
    return moves, summary


def parse_filename(stem: str) -> dict:
    s = stem.removeprefix('metrics_')
    m = re.match(r'^(?P<label>.+?)_g(?P<idx>\d+)_(?P<color>orig|swap)$', s)
    if not m:
        return {'label': s, 'game_idx': None, 'swapped': False}
    return {
        'label': m.group('label'),
        'game_idx': int(m.group('idx')),
        'swapped': m.group('color') == 'swap',
    }


def extract_book_metrics(exp_dir: Path) -> pd.DataFrame:
    """For each metrics file, compute book exit move and opening-phase times."""
    rows = []
    for mfile in sorted(exp_dir.glob('metrics_*.jsonl')):
        parsed_fn = parse_filename(mfile.stem)
        moves, summary = load_metrics_moves(mfile)
        if not moves:
            continue

        last_book_ply = 0
        book_time_white = 0.0
        book_time_black = 0.0
        # "Opening phase" defined as first 20 plies (10 moves per side)
        opening_phase_time_white = 0.0
        opening_phase_time_black = 0.0
        opening_phase_plies_counted = 0
        OPENING_PHASE_PLIES = 20

        for i, move in enumerate(moves):
            stats = move.get('algorithm_stats') or {}
            from_book = stats.get('from_book', False)
            side = move.get('side')
            time_s = move.get('time_s', 0) or 0

            if from_book:
                last_book_ply = i + 1
                if side == 'WHITE':
                    book_time_white += time_s
                elif side == 'BLACK':
                    book_time_black += time_s

            if i < OPENING_PHASE_PLIES:
                if side == 'WHITE':
                    opening_phase_time_white += time_s
                elif side == 'BLACK':
                    opening_phase_time_black += time_s
                opening_phase_plies_counted += 1

        result = (summary or {}).get('result', '?')
        rows.append({
            'file': mfile.name,
            'label': parsed_fn['label'],
            'game_idx': parsed_fn['game_idx'],
            'swapped': parsed_fn['swapped'],
            'result': result,
            'total_moves': (summary or {}).get('total_moves'),
            'last_book_ply': last_book_ply,
            'book_time_white': round(book_time_white, 3),
            'book_time_black': round(book_time_black, 3),
            'opening_phase_time_white': round(opening_phase_time_white, 3),
            'opening_phase_time_black': round(opening_phase_time_black, 3),
            'opening_phase_plies': opening_phase_plies_counted,
        })
    return pd.DataFrame(rows)


def aggregate_per_condition(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate W/D/L and book metrics per (engine, book) condition."""
    rows = []
    for label, group in metrics_df.groupby('label'):
        parsed = parse_condition(label)
        if not parsed:
            continue
        total = len(group)
        wins = (group['result'] == '1-0').sum()
        draws = (group['result'] == '1/2-1/2').sum()
        losses = (group['result'] == '0-1').sum()
        rows.append({
            'engine': parsed['engine'],
            'book': parsed['book'],
            'label': label,
            'games': total,
            'white_wins': int(wins),
            'draws': int(draws),
            'black_wins': int(losses),
            'white_score': round((wins + 0.5 * draws) / total, 3) if total > 0 else 0.0,
            'avg_total_moves': round(group['total_moves'].mean(), 1),
            'avg_last_book_ply': round(group['last_book_ply'].mean(), 2),
            'max_last_book_ply': int(group['last_book_ply'].max()),
            'avg_opening_time_white': round(group['opening_phase_time_white'].mean(), 3),
            'avg_opening_time_black': round(group['opening_phase_time_black'].mean(), 3),
        })
    return pd.DataFrame(rows).sort_values(['engine', 'book']).reset_index(drop=True)


def chi_square_2x3(off: dict, on: dict) -> dict:
    """
    Chi-square test on 2x3 contingency table (book × outcome).
    Returns dict with statistic, p-value, df.
    """
    from scipy.stats import chi2_contingency
    observed = np.array([
        [off['white_wins'], off['draws'], off['black_wins']],
        [on['white_wins'], on['draws'], on['black_wins']],
    ])
    if observed.sum() == 0:
        return {'chi2': None, 'p_value': None, 'df': None}
    chi2, p, dof, _ = chi2_contingency(observed)
    return {'chi2': round(float(chi2), 4), 'p_value': round(float(p), 4), 'df': int(dof)}


def mcnemar_paired(metrics_df: pd.DataFrame, engine: str) -> dict | None:
    """
    Paired McNemar test on win/non-win binary outcomes.

    Pairs games by (game_idx, swapped) — the same game index in book OFF and
    book ON is treated as a pair. Returns None if no pairs found.

    For each pair, define X = white wins in OFF, Y = white wins in ON.
    Discordant pairs:
        b = OFF win, ON loss
        c = OFF loss, ON win
    Test statistic = (b - c)^2 / (b + c), chi-square df=1.
    """
    off = metrics_df[metrics_df['label'] == f'{engine}_book_off'].copy()
    on = metrics_df[metrics_df['label'] == f'{engine}_book_on'].copy()
    if off.empty or on.empty:
        return None

    off = off.dropna(subset=['game_idx'])
    on = on.dropna(subset=['game_idx'])

    merged = off.merge(on, on=['game_idx', 'swapped'], suffixes=('_off', '_on'))
    if merged.empty:
        return None

    # Code outcome as binary: 1 = white wins, 0 = otherwise
    merged['outcome_off'] = (merged['result_off'] == '1-0').astype(int)
    merged['outcome_on'] = (merged['result_on'] == '1-0').astype(int)

    # Discordant pairs
    b = ((merged['outcome_off'] == 1) & (merged['outcome_on'] == 0)).sum()
    c = ((merged['outcome_off'] == 0) & (merged['outcome_on'] == 1)).sum()
    n_pairs = len(merged)

    if (b + c) == 0:
        return {'engine': engine, 'n_pairs': int(n_pairs), 'b': int(b), 'c': int(c),
                'statistic': 0.0, 'p_value': 1.0, 'note': 'no discordant pairs'}

    # Use exact binomial for small samples (b+c < 25)
    if (b + c) < 25:
        from scipy.stats import binomtest
        result = binomtest(int(b), int(b + c), 0.5)
        return {
            'engine': engine, 'n_pairs': int(n_pairs),
            'b_off_wins_on_loses': int(b), 'c_off_loses_on_wins': int(c),
            'statistic': None, 'p_value': round(float(result.pvalue), 4),
            'note': 'exact binomial (b+c<25)',
        }

    statistic = (b - c) ** 2 / (b + c)
    from scipy.stats import chi2
    p = 1 - chi2.cdf(statistic, df=1)
    return {
        'engine': engine, 'n_pairs': int(n_pairs),
        'b_off_wins_on_loses': int(b), 'c_off_loses_on_wins': int(c),
        'statistic': round(float(statistic), 4),
        'p_value': round(float(p), 4),
        'note': 'chi-square approximation',
    }


def plot_wdl_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if summary_df.empty:
        return

    engines = sorted(summary_df['engine'].unique())
    fig, axes = plt.subplots(1, len(engines), figsize=(6 * len(engines), 5), squeeze=False)
    for ax, engine in zip(axes[0], engines):
        sub = summary_df[summary_df['engine'] == engine]
        if sub.empty:
            continue
        x = np.arange(len(sub))
        w = 0.25
        ax.bar(x - w, sub['white_wins'], w, label='White wins', color='#4CAF50')
        ax.bar(x, sub['draws'], w, label='Draws', color='#9E9E9E')
        ax.bar(x + w, sub['black_wins'], w, label='Black wins', color='#F44336')
        ax.set_xticks(x)
        ax.set_xticklabels([f"book {b.upper()}" for b in sub['book']])
        ax.set_title(engine)
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle('W / D / L comparison: opening book OFF vs ON')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_book_exit_hist(metrics_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    on_only = metrics_df[metrics_df['label'].str.contains('_book_on')]
    if on_only.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in on_only.groupby('label'):
        ax.hist(group['last_book_ply'], bins=range(0, 25, 1), alpha=0.5,
                label=label, edgecolor='black')
    ax.set_xlabel('Last book ply per game (0 = no book moves)')
    ax.set_ylabel('Count')
    ax.set_title('Book exit distribution (book ON)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_opening_time(summary_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if summary_df.empty:
        return

    summary_df = summary_df.copy()
    summary_df['avg_opening_time'] = (
        summary_df['avg_opening_time_white'] + summary_df['avg_opening_time_black']
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    engines = sorted(summary_df['engine'].unique())
    x = np.arange(len(engines))
    w = 0.35

    off_vals = []
    on_vals = []
    for eng in engines:
        off = summary_df[(summary_df['engine'] == eng) & (summary_df['book'] == 'off')]
        on = summary_df[(summary_df['engine'] == eng) & (summary_df['book'] == 'on')]
        off_vals.append(off['avg_opening_time'].iloc[0] if not off.empty else 0)
        on_vals.append(on['avg_opening_time'].iloc[0] if not on.empty else 0)

    ax.bar(x - w / 2, off_vals, w, label='book OFF', color='#1976D2')
    ax.bar(x + w / 2, on_vals, w, label='book ON', color='#4CAF50')
    ax.set_xticks(x)
    ax.set_xticklabels(engines)
    ax.set_ylabel('Average time in first 20 plies (s)')
    ax.set_title('Time saved by opening book in opening phase')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Exp 7 — Opening book impact analysis')
    parser.add_argument('experiment_dir', type=str)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_dir():
        print(f"Error: {exp_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Loading metrics files from {exp_dir}...")
    metrics_df = extract_book_metrics(exp_dir)
    if metrics_df.empty:
        print("No metrics files found.", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(metrics_df)} games")

    raw_csv = exp_dir / 'exp7_raw_per_game.csv'
    metrics_df.to_csv(raw_csv, index=False)
    print(f"  Saved: {raw_csv.name}")

    # Aggregate per condition
    summary_df = aggregate_per_condition(metrics_df)
    if summary_df.empty:
        print("No valid exp7 conditions found in labels.", file=sys.stderr)
        sys.exit(1)

    sum_csv = exp_dir / 'exp7_summary.csv'
    summary_df.to_csv(sum_csv, index=False)
    print(f"  Saved: {sum_csv.name}")

    print()
    print('--- Per-condition summary ---')
    print(summary_df.to_string(index=False))

    # Statistical tests
    print()
    print('--- Statistical tests ---')
    test_results = []

    for engine in sorted(summary_df['engine'].unique()):
        off_row = summary_df[(summary_df['engine'] == engine) & (summary_df['book'] == 'off')]
        on_row = summary_df[(summary_df['engine'] == engine) & (summary_df['book'] == 'on')]
        if off_row.empty or on_row.empty:
            continue

        # Chi-square on 2x3 outcome table
        chi = chi_square_2x3(off_row.iloc[0].to_dict(), on_row.iloc[0].to_dict())
        chi['engine'] = engine
        chi['test'] = 'chi-square 2x3'
        test_results.append(chi)

        # Paired McNemar
        mc = mcnemar_paired(metrics_df, engine)
        if mc:
            mc['test'] = 'McNemar (paired)'
            test_results.append(mc)

    if test_results:
        tests_df = pd.DataFrame(test_results)
        cols = ['engine', 'test', 'p_value', 'chi2', 'statistic',
                'b_off_wins_on_loses', 'c_off_loses_on_wins', 'n_pairs', 'note', 'df']
        tests_df = tests_df[[c for c in cols if c in tests_df.columns]]
        tests_csv = exp_dir / 'exp7_statistical_tests.csv'
        tests_df.to_csv(tests_csv, index=False)
        print(f"  Saved: {tests_csv.name}")
        print(tests_df.to_string(index=False))

    # Plots
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print('\nGenerating plots...')
    plot_wdl_comparison(summary_df, plots_dir / 'exp7_wdl_comparison.png')
    plot_book_exit_hist(metrics_df, plots_dir / 'exp7_book_exit_hist.png')
    plot_opening_time(summary_df, plots_dir / 'exp7_opening_time.png')
    print(f"  Plots saved to: {plots_dir}")

    # Human-readable summary
    txt_path = exp_dir / 'exp7_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Experiment 7 — Opening book impact\n')
        f.write('=' * 60 + '\n\n')
        f.write('Per-condition summary:\n')
        f.write(summary_df.to_string(index=False) + '\n\n')
        if test_results:
            f.write('Statistical tests:\n')
            f.write(tests_df.to_string(index=False) + '\n')
    print(f"  Saved: {txt_path.name}")

    print('\nDone.')


if __name__ == '__main__':
    main()
