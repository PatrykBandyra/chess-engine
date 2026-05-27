"""
Aggregates results from all 4 variants of Experiment 6 (Tactical accuracy).

Reads exp6_variant<N>_<name>_<tag>.csv files from this directory and produces:
  - exp6_solve_rate.csv          — overall solve rate per variant
  - exp6_solve_by_theme.csv      — solve rate per (variant, theme)
  - exp6_summary.txt             — human-readable
  - plots/exp6_solve_rate_bars.png       — overall solve rate per variant
  - plots/exp6_solve_by_theme_heatmap.png — variant × theme heatmap
  - plots/exp6_solve_by_theme_radar.png  — radar chart (if matplotlib supports)
  - plots/exp6_minimax_depth_hist.png    — depth-to-solve for Minimax variants

Usage:
    python exp6_analyze.py
    python exp6_analyze.py --tag 20260527
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='',
                        help='Only consider files with this tag suffix')
    parser.add_argument('--exp6-dir', type=str, default='',
                        help='Default: this script\'s directory')
    args = parser.parse_args()

    exp6_dir = Path(args.exp6_dir) if args.exp6_dir else Path(__file__).resolve().parent

    pattern = f'exp6_variant*{"_" + args.tag if args.tag else ""}*.csv'
    files = sorted(exp6_dir.glob(pattern))

    if not files:
        print(f"No exp6_variant*.csv files found in {exp6_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} variant result files:")
    for f in files:
        print(f"  {f.name}")

    # Load and concatenate
    frames = []
    for f in files:
        df = pd.read_csv(f)
        # Extract variant name from filename: exp6_variant<N>_<NAME>_<TAG>.csv
        stem_parts = f.stem.split('_')
        # exp6, variant<N>, ...name..., tag
        if len(stem_parts) >= 4:
            variant_name = '_'.join(stem_parts[2:-1])
        else:
            variant_name = '_'.join(stem_parts[2:])
        df['variant'] = variant_name
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    print(f"Total rows: {len(full_df)}")

    # Overall solve rate per variant
    solve_rate = full_df.groupby('variant').agg(
        n=('solved', 'count'),
        solved=('solved', 'sum'),
        avg_time_s=('duration_s', 'mean'),
        timeouts=('timed_out', 'sum'),
    ).reset_index()
    solve_rate['solve_rate'] = (solve_rate['solved'] / solve_rate['n']).round(4)
    solve_rate = solve_rate.sort_values('solve_rate', ascending=False)

    rate_csv = exp6_dir / 'exp6_solve_rate.csv'
    solve_rate.to_csv(rate_csv, index=False)
    print(f"\nSaved: {rate_csv.name}")
    print()
    print('--- Overall solve rate ---')
    print(solve_rate.to_string(index=False))

    # Per-theme breakdown
    theme_df = full_df.groupby(['variant', 'theme']).agg(
        n=('solved', 'count'),
        solved=('solved', 'sum'),
    ).reset_index()
    theme_df['solve_rate'] = (theme_df['solved'] / theme_df['n']).round(4)

    theme_csv = exp6_dir / 'exp6_solve_by_theme.csv'
    theme_df.to_csv(theme_csv, index=False)
    print(f"\nSaved: {theme_csv.name}")

    # Minimax depth-to-solve (only for Minimax variants)
    minimax_df = full_df[
        full_df['variant'].str.contains('MINIMAX', case=False) &
        full_df['solved'] &
        full_df['min_depth_to_solve'].notna()
    ]
    depth_csv = None
    if not minimax_df.empty:
        depth_summary = minimax_df.groupby('variant')['min_depth_to_solve'].agg(
            ['mean', 'median', 'min', 'max', 'count']
        ).round(2).reset_index()
        depth_csv = exp6_dir / 'exp6_minimax_depth_to_solve.csv'
        depth_summary.to_csv(depth_csv, index=False)
        print(f"\nSaved: {depth_csv.name}")
        print()
        print('--- Minimax min depth to find solution ---')
        print(depth_summary.to_string(index=False))

    # Text summary
    txt_path = exp6_dir / 'exp6_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Experiment 6 — Tactical accuracy\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Variants tested: {len(files)}\n')
        f.write(f'Total puzzle attempts: {len(full_df)}\n\n')
        f.write('Overall solve rate:\n')
        f.write(solve_rate.to_string(index=False) + '\n\n')
        f.write('Solve rate per theme (top 30):\n')
        f.write(theme_df.sort_values(['variant', 'theme']).head(30).to_string(index=False) + '\n')
        if not minimax_df.empty:
            f.write('\nMinimax min depth to solve:\n')
            f.write(depth_summary.to_string(index=False) + '\n')
    print(f"Saved: {txt_path.name}")

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    plots_dir = exp6_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Overall solve rate bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(solve_rate['variant'], solve_rate['solve_rate'] * 100,
           color='#1976D2')
    ax.set_ylabel('Solve rate (%)')
    ax.set_title('Overall tactical solve rate per variant')
    plt.xticks(rotation=20, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(plots_dir / 'exp6_solve_rate_bars.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: plots/exp6_solve_rate_bars.png")

    # Heatmap variant × theme
    pivot = theme_df.pivot(index='theme', columns='variant', values='solve_rate')
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5),
                                         max(4, len(pivot.index) * 0.4)))
        im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=20, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title('Solve rate by theme × variant')
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                            color='black' if val > 0.5 else 'white', fontsize=7)
        fig.colorbar(im, ax=ax, label='Solve rate')
        fig.tight_layout()
        fig.savefig(plots_dir / 'exp6_solve_by_theme_heatmap.png', dpi=150)
        plt.close(fig)
        print(f"  Saved: plots/exp6_solve_by_theme_heatmap.png")

        # Radar chart (only if few enough themes)
        themes = list(pivot.index)
        if 3 <= len(themes) <= 20:
            angles = np.linspace(0, 2 * np.pi, len(themes), endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            for variant in pivot.columns:
                values = pivot[variant].fillna(0).tolist()
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=variant)
                ax.fill(angles, values, alpha=0.1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(themes, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title('Solve rate per theme (radar)', y=1.08)
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=8)
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(plots_dir / 'exp6_solve_by_theme_radar.png', dpi=150)
            plt.close(fig)
            print(f"  Saved: plots/exp6_solve_by_theme_radar.png")

    # Minimax depth-to-solve histogram
    if not minimax_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant, group in minimax_df.groupby('variant'):
            ax.hist(group['min_depth_to_solve'].dropna(), bins=range(1, 12),
                    alpha=0.5, label=variant, edgecolor='black')
        ax.set_xlabel('Minimum search depth at which solution was found')
        ax.set_ylabel('Count')
        ax.set_title('Depth-to-solve distribution (Minimax variants)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(plots_dir / 'exp6_minimax_depth_hist.png', dpi=150)
        plt.close(fig)
        print(f"  Saved: plots/exp6_minimax_depth_hist.png")

    print('\nDone.')


if __name__ == '__main__':
    main()
