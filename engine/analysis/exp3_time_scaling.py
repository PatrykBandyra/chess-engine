"""
Experiment 3 specific analysis: MCTS time budget scaling.

Consumes output of analyze_experiment.py (analysis_moves.csv,
analysis_games.csv) from an exp3 directory and produces:

  - exp3_time_summary.csv      — per-(evaluator, time) aggregate MCTS metrics
  - exp3_time_summary.txt      — human-readable table
  - exp3_elo_curve.png         — Elo vs log2(time), separate line per evaluator
  - exp3_elo_log_fit.csv       — log-linear fit parameters per evaluator
  - exp3_throughput_curve.png  — iterations/s vs time (TRAD vs NN)
  - exp3_tree_size_curve.png   — nodes_created vs time
  - exp3_max_depth_curve.png   — tree depth vs time
  - exp3_entropy_curve.png     — root visit entropy vs time

Usage:
    python exp3_time_scaling.py <experiment_dir>
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Matches labels like: mcts_trad_1s_vs_mcts_trad_20s, mcts_nn_40s_vs_mcts_nn_20s
LABEL_RE = re.compile(r'^mcts_(?P<eval>trad|nn)_(?P<time>\d+(?:\.\d+)?)s_vs_mcts_\1_(?P<ref>\d+(?:\.\d+)?)s$')


def parse_label(label: str) -> dict | None:
    m = LABEL_RE.match(label)
    if not m:
        return None
    return {
        'evaluator': m.group('eval').upper(),
        'time': float(m.group('time')),
        'ref_time': float(m.group('ref')),
    }


def estimate_elo_per_group(games_df: pd.DataFrame, evaluator: str) -> pd.DataFrame:
    rows = []
    for _, row in games_df.iterrows():
        parsed = parse_label(row['label'])
        if not parsed or parsed['evaluator'] != evaluator:
            continue
        rows.append({
            'time': parsed['time'],
            'ref_time': parsed['ref_time'],
            'result': row['result'],
        })

    if not rows:
        return pd.DataFrame()

    sub = pd.DataFrame(rows)
    times = sorted(set(sub['time']) | set(sub['ref_time']))
    n = len(times)
    idx = {t: i for i, t in enumerate(times)}

    results = []
    for _, r in sub.iterrows():
        score = {'1-0': 1.0, '0-1': 0.0}.get(r['result'], 0.5)
        results.append((r['time'], r['ref_time'], score))

    ratings = np.zeros(n)
    for _ in range(300):
        wins = np.zeros(n)
        played = np.zeros(n)
        expected = np.zeros(n)
        for wt, bt, score in results:
            wi, bi = idx[wt], idx[bt]
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
        ref_i = idx[sub['ref_time'].iloc[0]]
        ratings -= ratings[ref_i]

    return pd.DataFrame([
        {'evaluator': evaluator, 'time': t, 'elo': round(ratings[idx[t]], 1)}
        for t in times
    ]).sort_values('time').reset_index(drop=True)


def fit_log_linear(elo_df: pd.DataFrame) -> pd.DataFrame:
    """Fit Elo = a + b * log2(time) per evaluator."""
    rows = []
    for evaluator, group in elo_df.groupby('evaluator'):
        if len(group) < 2:
            continue
        x = np.log2(group['time'].values)
        y = group['elo'].values
        # least squares
        coeffs = np.polyfit(x, y, 1)
        b, a = coeffs[0], coeffs[1]
        # R^2
        y_pred = a + b * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rows.append({
            'evaluator': evaluator,
            'intercept_a': round(a, 2),
            'slope_b': round(b, 2),
            'elo_per_doubling': round(b, 2),
            'r_squared': round(r2, 4),
        })
    return pd.DataFrame(rows)


def aggregate_time_metrics(moves_df: pd.DataFrame) -> pd.DataFrame:
    if moves_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in moves_df.iterrows():
        if row.get('from_book', False):
            continue
        # Test side is WHITE in our matchups (varying time budget)
        if row.get('side') != 'WHITE':
            continue
        parsed = parse_label(row.get('label', ''))
        if not parsed:
            continue
        rec = {
            'evaluator': parsed['evaluator'],
            'time': parsed['time'],
        }
        for col in ['time_s', 'iterations', 'nodes_created', 'max_depth',
                    'eval_calls', 'eval_cache_hits', 'eval_cache_hit_rate',
                    'throughput_iter_per_s', 'throughput_eval_per_s',
                    'root_children_count', 'best_child_visits',
                    'root_visit_entropy', 'convergence_point',
                    'avg_backprop_depth', 'skipped_terminals', 'reused_visits']:
            if col in row and pd.notna(row[col]):
                rec[col] = row[col]
        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    grouped = df.groupby(['evaluator', 'time']).agg(['mean', 'std', 'count'])
    grouped.columns = ['_'.join(col) for col in grouped.columns]
    return grouped.reset_index()


def plot_elo_curve(elo_df: pd.DataFrame, fit_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if elo_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'TRAD': '#1976D2', 'NN': '#D32F2F'}
    for evaluator, group in elo_df.groupby('evaluator'):
        g = group.sort_values('time')
        c = colors.get(evaluator, 'black')
        ax.plot(g['time'], g['elo'], 'o-', label=f'{evaluator} (measured)',
                linewidth=2, markersize=8, color=c)

        # Plot log-linear fit if available
        fit_row = fit_df[fit_df['evaluator'] == evaluator]
        if not fit_row.empty:
            a = fit_row['intercept_a'].iloc[0]
            b = fit_row['slope_b'].iloc[0]
            r2 = fit_row['r_squared'].iloc[0]
            x_fit = np.linspace(g['time'].min(), g['time'].max(), 50)
            y_fit = a + b * np.log2(x_fit)
            ax.plot(x_fit, y_fit, '--', linewidth=1.2, color=c, alpha=0.6,
                    label=f'{evaluator} fit: +{b:.0f} Elo/doubling (R²={r2:.2f})')

    ax.set_xlabel('MCTS time budget (s)')
    ax.set_ylabel('Elo (relative to t=20s)')
    ax.set_title('Elo vs MCTS time — log-linear scaling')
    ax.set_xscale('log', base=2)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_curve(summary_df: pd.DataFrame, metric: str, ylabel: str,
                      title: str, out_path: Path, log_x: bool = True,
                      log_y: bool = False) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    col = f'{metric}_mean'
    if col not in summary_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'TRAD': '#1976D2', 'NN': '#D32F2F'}
    for evaluator, group in summary_df.groupby('evaluator'):
        g = group.sort_values('time')
        ax.plot(g['time'], g[col], 'o-', label=evaluator, linewidth=2,
                markersize=8, color=colors.get(evaluator, 'black'))

    ax.set_xlabel('MCTS time budget (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_x:
        ax.set_xscale('log', base=2)
    if log_y:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Exp 3 — MCTS time scaling analysis')
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

    # Aggregate per-time metrics
    print("Aggregating per-(evaluator, time) MCTS metrics...")
    summary_df = aggregate_time_metrics(moves_df)

    if summary_df.empty:
        print("No exp3 matchups found (label format must be mcts_<eval>_<T>s_vs_mcts_<eval>_<T>s).")
        sys.exit(0)

    summary_path = exp_dir / 'exp3_time_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path.name}")

    # Per-group Elo
    print("Estimating Elo per evaluator (reference = t=20s)...")
    elo_frames = []
    for evaluator in ['TRAD', 'NN']:
        elo = estimate_elo_per_group(games_df, evaluator)
        if not elo.empty:
            elo_frames.append(elo)
    elo_df = pd.concat(elo_frames, ignore_index=True) if elo_frames else pd.DataFrame()

    fit_df = pd.DataFrame()
    if not elo_df.empty:
        elo_path = exp_dir / 'exp3_elo_per_time.csv'
        elo_df.to_csv(elo_path, index=False)
        print(f"  Saved: {elo_path.name}")
        print()
        print('--- Elo per time budget (relative to t=20s) ---')
        print(elo_df.to_string(index=False))
        print()

        fit_df = fit_log_linear(elo_df)
        if not fit_df.empty:
            fit_path = exp_dir / 'exp3_elo_log_fit.csv'
            fit_df.to_csv(fit_path, index=False)
            print(f"  Saved: {fit_path.name}")
            print('--- Log-linear fit: Elo = a + b·log2(time) ---')
            print(fit_df.to_string(index=False))
            print()

    # Plots
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print("Generating plots...")
    if not elo_df.empty:
        plot_elo_curve(elo_df, fit_df, plots_dir / 'exp3_elo_curve.png')
    plot_metric_curve(summary_df, 'throughput_iter_per_s', 'Iterations / s',
                      'MCTS throughput (TRAD vs NN)', plots_dir / 'exp3_throughput_curve.png',
                      log_x=True, log_y=False)
    plot_metric_curve(summary_df, 'nodes_created', 'Avg nodes created per move',
                      'Tree size vs time', plots_dir / 'exp3_tree_size_curve.png',
                      log_x=True, log_y=True)
    plot_metric_curve(summary_df, 'max_depth', 'Max tree depth per move',
                      'Tree depth vs time', plots_dir / 'exp3_max_depth_curve.png',
                      log_x=True, log_y=False)
    plot_metric_curve(summary_df, 'root_visit_entropy', 'Root visit entropy',
                      'Search certainty (lower = more confident)',
                      plots_dir / 'exp3_entropy_curve.png', log_x=True, log_y=False)
    print(f"  Plots saved to: {plots_dir}")

    # Human-readable summary
    txt_path = exp_dir / 'exp3_time_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Experiment 3 — MCTS time budget scaling\n')
        f.write('=' * 60 + '\n\n')
        if not elo_df.empty:
            f.write('Elo per time budget (relative to t=20s):\n')
            f.write(elo_df.to_string(index=False) + '\n\n')
        if not fit_df.empty:
            f.write('Log-linear fit (Elo = a + b·log2(time)):\n')
            f.write(fit_df.to_string(index=False) + '\n\n')
        f.write('Per-(evaluator, time) MCTS metrics:\n')
        f.write(summary_df.to_string(index=False) + '\n')
    print(f"  Saved: {txt_path.name}")

    print("\nDone.")


if __name__ == '__main__':
    main()
