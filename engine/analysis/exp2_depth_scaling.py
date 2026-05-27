"""
Experiment 2 specific analysis: Minimax depth scaling.

Consumes output of analyze_experiment.py (analysis_moves.csv,
analysis_wdl.csv) from an exp2 directory and produces:

  - exp2_depth_summary.csv     — per-(evaluator, depth) aggregate metrics
  - exp2_depth_summary.txt     — human-readable table
  - exp2_elo_curve.png         — Elo per depth, separate line per evaluator
  - exp2_ebf_curve.png         — Effective Branching Factor vs depth
  - exp2_pruning_by_depth.png  — pruning technique usage per depth
  - exp2_time_curve.png        — avg time/move vs depth (log y-axis)
  - exp2_nodes_curve.png       — avg nodes searched vs depth (log y-axis)

Usage:
    python exp2_depth_scaling.py <experiment_dir>
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_RE = re.compile(r'^minimax_(?P<eval>trad|nn)_d(?P<depth>\d+)_vs_minimax_\1_d(?P<ref>\d+)$')


def parse_label(label: str) -> dict | None:
    m = LABEL_RE.match(label)
    if not m:
        return None
    return {
        'evaluator': m.group('eval').upper(),
        'depth': int(m.group('depth')),
        'ref_depth': int(m.group('ref')),
    }


def estimate_elo_per_group(games_df: pd.DataFrame, evaluator: str) -> pd.DataFrame:
    rows = []
    for _, row in games_df.iterrows():
        parsed = parse_label(row['label'])
        if not parsed or parsed['evaluator'] != evaluator:
            continue
        rows.append({
            'depth': parsed['depth'],
            'ref_depth': parsed['ref_depth'],
            'result': row['result'],
        })

    if not rows:
        return pd.DataFrame()

    sub = pd.DataFrame(rows)
    depths = sorted(set(sub['depth']) | set(sub['ref_depth']))
    n = len(depths)
    idx = {d: i for i, d in enumerate(depths)}

    results = []
    for _, r in sub.iterrows():
        score = {'1-0': 1.0, '0-1': 0.0}.get(r['result'], 0.5)
        results.append((r['depth'], r['ref_depth'], score))

    # Bradley-Terry iterative ML
    ratings = np.zeros(n)
    for _ in range(300):
        wins = np.zeros(n)
        played = np.zeros(n)
        expected = np.zeros(n)
        for wd, bd, score in results:
            wi, bi = idx[wd], idx[bd]
            diff = ratings[wi] - ratings[bi]
            exp_score = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
            wins[wi] += score
            wins[bi] += 1.0 - score
            played[wi] += 1
            played[bi] += 1
            expected[wi] += exp_score
            expected[bi] += 1.0 - exp_score
        for i in range(n):
            if played[i] > 0 and expected[i] > 0:
                ratings[i] += 16.0 * (wins[i] - expected[i]) / played[i]
        # Anchor: reference depth = 0
        ref_i = idx[sub['ref_depth'].iloc[0]]
        ratings -= ratings[ref_i]

    return pd.DataFrame([
        {'evaluator': evaluator, 'depth': d, 'elo': round(ratings[idx[d]], 1)}
        for d in depths
    ]).sort_values('depth').reset_index(drop=True)


def aggregate_depth_metrics(moves_df: pd.DataFrame) -> pd.DataFrame:
    if moves_df.empty:
        return pd.DataFrame()

    # Only consider moves of the "test" side (depth varies per matchup).
    # For each matchup, the side with varying depth is white (depth_white in config).
    # Since both players have the same matchup label, we filter by parsed label depth
    # and side: WHITE in our matchups always plays the varying depth.
    rows = []
    for _, row in moves_df.iterrows():
        if row.get('from_book', False):
            continue
        if row.get('side') != 'WHITE':
            continue
        parsed = parse_label(row.get('label', ''))
        if not parsed:
            continue
        rec = {
            'evaluator': parsed['evaluator'],
            'depth': parsed['depth'],
        }
        for col in ['time_s', 'nodes_searched', 'depth_completed',
                    'tt_hit_rate', 'tt_cutoff_rate', 'nmp_success_rate', 'ebf_mean',
                    'nmp_cutoffs', 'rfp_cutoffs', 'futility_prunes', 'lmp_prunes',
                    'see_prunes', 'check_extensions', 'qs_nodes', 'qs_max_depth']:
            if col in row and pd.notna(row[col]):
                rec[col] = row[col]
        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    grouped = df.groupby(['evaluator', 'depth']).agg(['mean', 'std', 'count'])
    grouped.columns = ['_'.join(col) for col in grouped.columns]
    return grouped.reset_index()


def plot_elo_curve(elo_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if elo_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for evaluator, group in elo_df.groupby('evaluator'):
        g = group.sort_values('depth')
        ax.plot(g['depth'], g['elo'], 'o-', label=evaluator, linewidth=2, markersize=8)

    ax.set_xlabel('Search depth')
    ax.set_ylabel('Elo (relative to d=4)')
    ax.set_title('Elo vs depth — Minimax depth scaling')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_curve(summary_df: pd.DataFrame, metric: str, ylabel: str,
                      title: str, out_path: Path, log_y: bool = False) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    col = f'{metric}_mean'
    if col not in summary_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for evaluator, group in summary_df.groupby('evaluator'):
        g = group.sort_values('depth')
        ax.plot(g['depth'], g[col], 'o-', label=evaluator, linewidth=2, markersize=8)

    ax.set_xlabel('Search depth')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pruning_by_depth(summary_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pruning_cols = ['nmp_cutoffs_mean', 'rfp_cutoffs_mean',
                    'futility_prunes_mean', 'lmp_prunes_mean', 'see_prunes_mean']
    available = [c for c in pruning_cols if c in summary_df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    evaluators = sorted(summary_df['evaluator'].unique())
    for ax, evaluator in zip(axes, evaluators):
        group = summary_df[summary_df['evaluator'] == evaluator].sort_values('depth')
        for col in available:
            label = col.replace('_mean', '').replace('_', ' ')
            ax.plot(group['depth'], group[col], 'o-', label=label, markersize=6)
        ax.set_xlabel('Search depth')
        ax.set_title(f'Pruning per move — {evaluator}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Average count per move')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Exp 2 — Minimax depth scaling analysis')
    parser.add_argument('experiment_dir', type=str)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_dir():
        print(f"Error: {exp_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    moves_csv = exp_dir / 'analysis_moves.csv'
    games_csv = exp_dir / 'analysis_games.csv'

    if not moves_csv.exists() or not games_csv.exists():
        print(f"Error: missing analysis CSVs in {exp_dir}", file=sys.stderr)
        print(f"  Run analyze_experiment.py first to generate them.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {moves_csv.name}, {games_csv.name}")
    moves_df = pd.read_csv(moves_csv)
    games_df = pd.read_csv(games_csv)

    # Aggregate per-depth metrics
    print("Aggregating per-(evaluator, depth) metrics...")
    summary_df = aggregate_depth_metrics(moves_df)

    if summary_df.empty:
        print("No exp2 matchups found (label format must be minimax_<eval>_d<N>_vs_minimax_<eval>_d<N>).")
        sys.exit(0)

    summary_path = exp_dir / 'exp2_depth_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path.name}")

    # Per-group Elo
    print("Estimating Elo per evaluator (reference = d=4)...")
    elo_frames = []
    for evaluator in ['TRAD', 'NN']:
        elo = estimate_elo_per_group(games_df, evaluator)
        if not elo.empty:
            elo_frames.append(elo)
    elo_df = pd.concat(elo_frames, ignore_index=True) if elo_frames else pd.DataFrame()

    if not elo_df.empty:
        elo_path = exp_dir / 'exp2_elo_per_depth.csv'
        elo_df.to_csv(elo_path, index=False)
        print(f"  Saved: {elo_path.name}")
        print()
        print('--- Elo per depth (relative to d=4) ---')
        print(elo_df.to_string(index=False))
        print()

    # Plots
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print("Generating plots...")
    if not elo_df.empty:
        plot_elo_curve(elo_df, plots_dir / 'exp2_elo_curve.png')
    plot_metric_curve(summary_df, 'ebf_mean', 'Effective Branching Factor',
                      'EBF vs depth', plots_dir / 'exp2_ebf_curve.png')
    plot_metric_curve(summary_df, 'time_s', 'Avg time per move (s)',
                      'Time per move vs depth', plots_dir / 'exp2_time_curve.png', log_y=True)
    plot_metric_curve(summary_df, 'nodes_searched', 'Avg nodes searched per move',
                      'Nodes searched vs depth', plots_dir / 'exp2_nodes_curve.png', log_y=True)
    plot_pruning_by_depth(summary_df, plots_dir / 'exp2_pruning_by_depth.png')
    print(f"  Plots saved to: {plots_dir}")

    # Human-readable summary
    txt_path = exp_dir / 'exp2_depth_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Experiment 2 — Minimax depth scaling\n')
        f.write('=' * 60 + '\n\n')
        if not elo_df.empty:
            f.write('Elo per depth (relative to d=4):\n')
            f.write(elo_df.to_string(index=False) + '\n\n')
        f.write('Per-(evaluator, depth) metrics:\n')
        f.write(summary_df.to_string(index=False) + '\n')
    print(f"  Saved: {txt_path.name}")

    print("\nDone.")


if __name__ == '__main__':
    main()
