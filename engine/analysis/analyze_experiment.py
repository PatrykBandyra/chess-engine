"""
Post-hoc analysis of chess engine experiment results.

Reads a directory of JSONL metrics files produced by run_experiment.ps1,
computes derived metrics, aggregates W/D/L per matchup, and exports CSVs.

Usage:
    python analyze_experiment.py <experiment_dir>
    python analyze_experiment.py out/exp1_round_robin_20260527_120000
    python analyze_experiment.py out/exp1_round_robin_20260527_120000 --plots
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

def parse_metrics_file(path: Path) -> dict[str, Any]:
    moves = []
    summary = None
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get('type') == 'game_summary':
                summary = record
            else:
                moves.append(record)
    return {'moves': moves, 'summary': summary, 'file': path.name}


def load_experiment(experiment_dir: Path) -> list[dict[str, Any]]:
    games = []
    for p in sorted(experiment_dir.glob('metrics_*.jsonl')):
        game = parse_metrics_file(p)
        game['label'] = _extract_label(p.stem)
        games.append(game)
    return games


def _extract_label(stem: str) -> str:
    # metrics_<label>_g<N>_<orig|swap> -> <label>
    parts = stem.removeprefix('metrics_')
    # find last _g<digits>_ pattern
    idx = parts.rfind('_g')
    if idx >= 0:
        return parts[:idx]
    return parts


# ---------------------------------------------------------------------------
# Derived metrics — Minimax
# ---------------------------------------------------------------------------

def derive_minimax_metrics(stats: dict) -> dict:
    derived = {}
    tt_lookups = stats.get('tt_lookups', 0)
    tt_hits = stats.get('tt_hits', 0)
    tt_cutoffs = stats.get('tt_cutoffs', 0)
    nmp_attempts = stats.get('nmp_attempts', 0)
    nmp_cutoffs = stats.get('nmp_cutoffs', 0)

    derived['tt_hit_rate'] = round(tt_hits / tt_lookups, 4) if tt_lookups > 0 else 0.0
    derived['tt_cutoff_rate'] = round(tt_cutoffs / tt_hits, 4) if tt_hits > 0 else 0.0
    derived['nmp_success_rate'] = round(nmp_cutoffs / nmp_attempts, 4) if nmp_attempts > 0 else 0.0

    nodes_per_depth = stats.get('nodes_per_depth', [])
    ebf_values = []
    for i in range(1, len(nodes_per_depth)):
        if nodes_per_depth[i - 1] > 0:
            ebf_values.append(round(nodes_per_depth[i] / nodes_per_depth[i - 1], 2))
    derived['ebf_per_depth'] = ebf_values
    derived['ebf_mean'] = round(np.mean(ebf_values), 2) if ebf_values else 0.0

    return derived


# ---------------------------------------------------------------------------
# Derived metrics — MCTS
# ---------------------------------------------------------------------------

def derive_mcts_metrics(stats: dict, time_s: float) -> dict:
    derived = {}
    iterations = stats.get('iterations', 0)
    eval_calls = stats.get('eval_calls', 0)
    eval_cache_hits = stats.get('eval_cache_hits', 0)

    derived['throughput_iter_per_s'] = round(iterations / time_s, 1) if time_s > 0 else 0.0
    derived['throughput_eval_per_s'] = round(eval_calls / time_s, 1) if time_s > 0 else 0.0
    derived['eval_cache_hit_rate'] = round(eval_cache_hits / eval_calls, 4) if eval_calls > 0 else 0.0

    return derived


# ---------------------------------------------------------------------------
# Per-move DataFrame
# ---------------------------------------------------------------------------

def build_moves_dataframe(games: list[dict]) -> pd.DataFrame:
    rows = []
    for game in games:
        label = game['label']
        summary = game.get('summary', {}) or {}
        game_result = summary.get('result', '?')
        file_name = game['file']
        for move in game['moves']:
            stats = move.get('algorithm_stats', {}) or {}
            is_book = stats.get('from_book', False)

            row = {
                'file': file_name,
                'label': label,
                'game_result': game_result,
                'move_number': move.get('move_number'),
                'side': move.get('side'),
                'move': move.get('move'),
                'eval': move.get('eval'),
                'time_s': move.get('time_s'),
                'phase': move.get('phase'),
                'from_book': is_book,
            }

            if not is_book:
                # Minimax stats
                if 'nodes_searched' in stats:
                    row.update({
                        'nodes_searched': stats.get('nodes_searched'),
                        'depth_completed': stats.get('depth_completed'),
                        'tt_lookups': stats.get('tt_lookups'),
                        'tt_hits': stats.get('tt_hits'),
                        'tt_cutoffs': stats.get('tt_cutoffs'),
                        'nmp_attempts': stats.get('nmp_attempts'),
                        'nmp_cutoffs': stats.get('nmp_cutoffs'),
                        'rfp_cutoffs': stats.get('rfp_cutoffs'),
                        'futility_prunes': stats.get('futility_prunes'),
                        'lmp_prunes': stats.get('lmp_prunes'),
                        'see_prunes': stats.get('see_prunes'),
                        'check_extensions': stats.get('check_extensions'),
                        'aspiration_researches': stats.get('aspiration_researches'),
                        'qs_nodes': stats.get('qs_nodes'),
                        'qs_max_depth': stats.get('qs_max_depth'),
                        'see_calls': stats.get('see_calls'),
                        'tt_size': stats.get('tt_size'),
                    })
                    derived = derive_minimax_metrics(stats)
                    row.update({
                        'tt_hit_rate': derived['tt_hit_rate'],
                        'tt_cutoff_rate': derived['tt_cutoff_rate'],
                        'nmp_success_rate': derived['nmp_success_rate'],
                        'ebf_mean': derived['ebf_mean'],
                    })

                # MCTS stats
                if 'iterations' in stats:
                    row.update({
                        'iterations': stats.get('iterations'),
                        'nodes_created': stats.get('nodes_created'),
                        'max_depth': stats.get('max_depth'),
                        'eval_calls': stats.get('eval_calls'),
                        'eval_cache_hits': stats.get('eval_cache_hits'),
                        'skipped_terminals': stats.get('skipped_terminals'),
                        'reused_visits': stats.get('reused_visits'),
                        'root_children_count': stats.get('root_children_count'),
                        'best_child_visits': stats.get('best_child_visits'),
                        'root_visit_entropy': stats.get('root_visit_entropy'),
                        'convergence_point': stats.get('convergence_point'),
                        'avg_backprop_depth': stats.get('avg_backprop_depth'),
                        'c_puct': stats.get('c_puct'),
                    })
                    time_s = move.get('time_s', 0)
                    derived = derive_mcts_metrics(stats, time_s)
                    row.update({
                        'throughput_iter_per_s': derived['throughput_iter_per_s'],
                        'throughput_eval_per_s': derived['throughput_eval_per_s'],
                        'eval_cache_hit_rate': derived['eval_cache_hit_rate'],
                    })

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Game-level DataFrame
# ---------------------------------------------------------------------------

def build_games_dataframe(games: list[dict]) -> pd.DataFrame:
    rows = []
    for game in games:
        summary = game.get('summary')
        if not summary:
            continue
        rows.append({
            'file': game['file'],
            'label': game['label'],
            'result': summary.get('result'),
            'total_moves': summary.get('total_moves'),
            'termination': summary.get('termination'),
            'total_time_white': summary.get('total_time_white'),
            'total_time_black': summary.get('total_time_black'),
            'avg_time_white': summary.get('avg_time_white'),
            'avg_time_black': summary.get('avg_time_black'),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# W/D/L aggregation
# ---------------------------------------------------------------------------

def aggregate_wdl(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    def wdl(group):
        wins = (group['result'] == '1-0').sum()
        draws = (group['result'] == '1/2-1/2').sum()
        losses = (group['result'] == '0-1').sum()
        total = len(group)
        return pd.Series({
            'games': total,
            'white_wins': wins,
            'draws': draws,
            'black_wins': losses,
            'white_score': round((wins + 0.5 * draws) / total, 3) if total > 0 else 0.0,
            'avg_moves': round(group['total_moves'].mean(), 1),
            'avg_time_white': round(group['avg_time_white'].mean(), 3),
            'avg_time_black': round(group['avg_time_black'].mean(), 3),
        })

    return games_df.groupby('label').apply(wdl, include_groups=False).reset_index()


# ---------------------------------------------------------------------------
# Per-matchup metric summaries
# ---------------------------------------------------------------------------

def aggregate_move_metrics(moves_df: pd.DataFrame) -> pd.DataFrame:
    if moves_df.empty:
        return pd.DataFrame()

    non_book = moves_df[~moves_df['from_book']].copy()
    if non_book.empty:
        return pd.DataFrame()

    numeric_cols = non_book.select_dtypes(include=[np.number]).columns
    agg_cols = [c for c in numeric_cols if c not in ('move_number',)]

    grouped = non_book.groupby(['label', 'side'])[agg_cols]
    result = grouped.agg(['mean', 'std', 'median']).reset_index()
    result.columns = ['_'.join(col).rstrip('_') for col in result.columns]
    return result


# ---------------------------------------------------------------------------
# Elo estimation (simple Bradley-Terry maximum likelihood)
# ---------------------------------------------------------------------------

def estimate_elo(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates Elo ratings using the Bradley-Terry model via iterative
    maximum likelihood. Input: games_df with 'label' and 'result' columns.
    Labels encode matchups as '<white_type>_vs_<black_type>'.
    """
    if games_df.empty:
        return pd.DataFrame()

    players = set()
    results = []

    for _, row in games_df.iterrows():
        label = row['label']
        parts = label.split('_vs_')
        if len(parts) != 2:
            continue

        white_player = parts[0]
        black_player = parts[1]
        players.add(white_player)
        players.add(black_player)

        result_str = row['result']
        if result_str == '1-0':
            score = 1.0
        elif result_str == '0-1':
            score = 0.0
        else:
            score = 0.5

        results.append((white_player, black_player, score))

    if not results or len(players) < 2:
        return pd.DataFrame()

    players = sorted(players)
    n = len(players)
    idx = {p: i for i, p in enumerate(players)}

    # Bradley-Terry iterative fitting
    ratings = np.zeros(n)
    for _ in range(200):
        wins = np.zeros(n)
        games_played = np.zeros(n)
        expected = np.zeros(n)

        for wp, bp, score in results:
            wi, bi = idx[wp], idx[bp]
            diff = ratings[wi] - ratings[bi]
            exp_score = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

            wins[wi] += score
            wins[bi] += (1.0 - score)
            games_played[wi] += 1
            games_played[bi] += 1
            expected[wi] += exp_score
            expected[bi] += (1.0 - exp_score)

        for i in range(n):
            if games_played[i] > 0 and expected[i] > 0:
                ratings[i] += 16.0 * (wins[i] - expected[i]) / games_played[i]

        ratings -= ratings.mean()

    rows = []
    for p in players:
        i = idx[p]
        rows.append({
            'player': p,
            'elo': round(ratings[i], 1),
            'games': sum(1 for wp, bp, _ in results if wp == p or bp == p),
        })

    return pd.DataFrame(rows).sort_values('elo', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_plots(games_df: pd.DataFrame, moves_df: pd.DataFrame,
                   wdl_df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    _plot_wdl_bars(wdl_df, plots_dir)
    _plot_time_per_move(moves_df, plots_dir)
    _plot_eval_over_game(moves_df, plots_dir)

    if 'nodes_searched' in moves_df.columns:
        _plot_minimax_pruning(moves_df, plots_dir)

    if 'iterations' in moves_df.columns:
        _plot_mcts_throughput(moves_df, plots_dir)

    print(f"  Plots saved to: {plots_dir}")


def _plot_wdl_bars(wdl_df: pd.DataFrame, plots_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if wdl_df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(wdl_df) * 1.2), 5))
    labels = wdl_df['label']
    x = np.arange(len(labels))
    w = 0.25

    ax.bar(x - w, wdl_df['white_wins'], w, label='White wins', color='#4CAF50')
    ax.bar(x, wdl_df['draws'], w, label='Draws', color='#9E9E9E')
    ax.bar(x + w, wdl_df['black_wins'], w, label='Black wins', color='#F44336')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Count')
    ax.set_title('W / D / L per matchup')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / 'wdl_bars.png', dpi=150)
    plt.close(fig)


def _plot_time_per_move(moves_df: pd.DataFrame, plots_dir: Path) -> None:
    import matplotlib.pyplot as plt

    non_book = moves_df[~moves_df['from_book']].copy()
    if non_book.empty:
        return

    groups = non_book.groupby(['label', 'side'])['time_s']
    data_labels = []
    data_values = []
    for (label, side), g in groups:
        data_labels.append(f"{label}\n({side})")
        data_values.append(g.dropna().values)

    if not data_values:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(data_values) * 0.8), 5))
    ax.boxplot(data_values, tick_labels=data_labels, vert=True)
    ax.set_ylabel('Time per move (s)')
    ax.set_title('Move time distribution')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / 'time_per_move_boxplot.png', dpi=150)
    plt.close(fig)


def _plot_eval_over_game(moves_df: pd.DataFrame, plots_dir: Path) -> None:
    import matplotlib.pyplot as plt

    non_book = moves_df[~moves_df['from_book'] & moves_df['eval'].notna()].copy()
    if non_book.empty:
        return

    files = non_book['file'].unique()
    sample = files[:min(6, len(files))]

    fig, axes = plt.subplots(len(sample), 1, figsize=(10, 3 * len(sample)), squeeze=False)
    for i, f in enumerate(sample):
        ax = axes[i, 0]
        game_moves = non_book[non_book['file'] == f].sort_values('move_number')
        white = game_moves[game_moves['side'] == 'WHITE']
        black = game_moves[game_moves['side'] == 'BLACK']
        ax.plot(white['move_number'], white['eval'], 'o-', label='White eval', markersize=3, color='#1976D2')
        ax.plot(black['move_number'], black['eval'], 's-', label='Black eval', markersize=3, color='#D32F2F')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Eval')
        ax.set_title(f, fontsize=8)
        ax.legend(fontsize=7)

    axes[-1, 0].set_xlabel('Move number')
    fig.suptitle('Evaluation over game (sample)', fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / 'eval_over_game.png', dpi=150)
    plt.close(fig)


def _plot_minimax_pruning(moves_df: pd.DataFrame, plots_dir: Path) -> None:
    import matplotlib.pyplot as plt

    cols = ['tt_cutoffs', 'nmp_cutoffs', 'rfp_cutoffs', 'futility_prunes', 'lmp_prunes', 'see_prunes']
    available = [c for c in cols if c in moves_df.columns]
    if not available:
        return

    non_book = moves_df[~moves_df['from_book'] & moves_df['nodes_searched'].notna()].copy()
    if non_book.empty:
        return

    means = non_book.groupby('label')[available].mean()
    if means.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(means) * 1.5), 5))
    means.plot(kind='bar', ax=ax)
    ax.set_ylabel('Average count per move')
    ax.set_title('Pruning technique usage (Minimax)')
    ax.legend(fontsize=7, loc='upper right')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / 'minimax_pruning.png', dpi=150)
    plt.close(fig)


def _plot_mcts_throughput(moves_df: pd.DataFrame, plots_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if 'throughput_iter_per_s' not in moves_df.columns:
        return

    non_book = moves_df[~moves_df['from_book'] & moves_df['iterations'].notna()].copy()
    if non_book.empty:
        return

    groups = non_book.groupby('label')['throughput_iter_per_s']
    data_labels = []
    data_values = []
    for label, g in groups:
        vals = g.dropna().values
        if len(vals) > 0:
            data_labels.append(label)
            data_values.append(vals)

    if not data_values:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(data_values) * 1.2), 5))
    ax.boxplot(data_values, tick_labels=data_labels, vert=True)
    ax.set_ylabel('Iterations / second')
    ax.set_title('MCTS throughput')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / 'mcts_throughput.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyze chess engine experiment results')
    parser.add_argument('experiment_dir', type=str, help='Path to experiment output directory')
    parser.add_argument('--plots', action='store_true', help='Generate visualization plots')
    parser.add_argument('--elo', action='store_true', help='Estimate Elo ratings')
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.is_dir():
        print(f"Error: {experiment_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Loading experiment: {experiment_dir}")
    games = load_experiment(experiment_dir)
    print(f"  Found {len(games)} game files")

    if not games:
        print("  No metrics files found. Exiting.")
        sys.exit(0)

    # Build DataFrames
    print("Building DataFrames...")
    moves_df = build_moves_dataframe(games)
    games_df = build_games_dataframe(games)
    wdl_df = aggregate_wdl(games_df)
    metrics_summary = aggregate_move_metrics(moves_df)

    # Export CSVs
    moves_csv = experiment_dir / 'analysis_moves.csv'
    games_csv = experiment_dir / 'analysis_games.csv'
    wdl_csv = experiment_dir / 'analysis_wdl.csv'
    metrics_csv = experiment_dir / 'analysis_metrics_summary.csv'

    moves_df.to_csv(moves_csv, index=False)
    games_df.to_csv(games_csv, index=False)
    wdl_df.to_csv(wdl_csv, index=False)
    if not metrics_summary.empty:
        metrics_summary.to_csv(metrics_csv, index=False)

    print(f"  Saved: {moves_csv.name} ({len(moves_df)} rows)")
    print(f"  Saved: {games_csv.name} ({len(games_df)} rows)")
    print(f"  Saved: {wdl_csv.name} ({len(wdl_df)} rows)")
    if not metrics_summary.empty:
        print(f"  Saved: {metrics_csv.name}")

    # W/D/L summary
    print("\n--- W / D / L ---")
    if not wdl_df.empty:
        print(wdl_df.to_string(index=False))
    else:
        print("  No game summaries found.")

    # Elo estimation
    if args.elo:
        print("\n--- Elo estimation (Bradley-Terry ML) ---")
        elo_df = estimate_elo(games_df)
        if not elo_df.empty:
            print(elo_df.to_string(index=False))
            elo_csv = experiment_dir / 'analysis_elo.csv'
            elo_df.to_csv(elo_csv, index=False)
            print(f"  Saved: {elo_csv.name}")
        else:
            print("  Could not estimate Elo (labels must contain '_vs_' to identify players).")

    # Plots
    if args.plots:
        print("\nGenerating plots...")
        generate_plots(games_df, moves_df, wdl_df, experiment_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
