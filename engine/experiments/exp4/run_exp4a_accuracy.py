"""
Experiment 4a — Static evaluation accuracy.

For each test position, queries:
  - BoardEvaluatorTrad (heuristic)
  - BoardEvaluatorNN   (Stockfish low-depth, as configured in the project)
  - Stockfish d=1      (baseline)
  - Stockfish d=20     (ground truth)

Computes Spearman ρ, MAE, RMSE against d=20. Stratifies by game phase.

Output:
  exp4a_evaluations.csv     — per-position evaluations and phase
  exp4a_accuracy_summary.csv — Spearman/MAE/RMSE per evaluator + phase band
  exp4a_accuracy_summary.txt — human-readable
  plots/exp4a_scatter_trad.png, exp4a_scatter_nn.png, exp4a_scatter_sfd1.png
  plots/exp4a_mae_by_phase.png

Usage:
    python run_exp4a_accuracy.py --stockfish <path>
    python run_exp4a_accuracy.py --stockfish <path> --positions test_positions.fen --nn-depth 10
"""

import argparse
import math
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pandas as pd

# Allow imports from engine/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse as _ap  # avoid name clash with our local args


def make_eval_args(stockfish_path: str, depth: int) -> _ap.Namespace:
    """Constructs an argparse.Namespace satisfying BoardEvaluator* constructors."""
    args = _ap.Namespace()
    args.depth_white_stockfish = depth
    args.depth_black_stockfish = depth
    args.skill_white = None
    args.skill_black = None
    args.stockfish_path = stockfish_path
    args.debug = False
    return args


def load_positions(path: Path) -> list[tuple[str, str]]:
    positions = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '#' in line:
            fen, tag = line.split('#', 1)
            positions.append((fen.strip(), tag.strip()))
        else:
            positions.append((line, ''))
    return positions


PHASE_WEIGHTS = {chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 5.0, chess.QUEEN: 9.0}


def game_phase(board: chess.Board) -> float:
    material = 0.0
    for pt, w in PHASE_WEIGHTS.items():
        material += w * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
    if material >= 67.0:
        return 1.0
    if material <= 20.0:
        return 0.0
    return (material - 20.0) / 47.0


def phase_band(phase: float) -> str:
    if phase > 0.8:
        return 'opening'
    if phase < 0.3:
        return 'endgame'
    return 'midgame'


def sf_eval_cp(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> float:
    """Returns evaluation in pawns (centipawns / 100) from White's perspective."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info['score'].white()
    if score.is_mate():
        m = score.mate()
        return 100.0 if (m or 0) > 0 else -100.0
    cp = score.score()
    return (cp or 0) / 100.0


def main():
    parser = argparse.ArgumentParser(description='Exp 4a — Static evaluation accuracy')
    parser.add_argument('--stockfish', type=str, required=True)
    parser.add_argument('--positions', type=str, default='',
                        help='Default: experiments/exp4/test_positions.fen')
    parser.add_argument('--nn-depth', type=int, default=10,
                        help='Depth for BoardEvaluatorNN (Stockfish under the hood)')
    parser.add_argument('--ground-truth-depth', type=int, default=20)
    parser.add_argument('--limit', type=int, default=0, help='Limit positions for testing')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    pos_file = Path(args.positions) if args.positions else script_dir / 'test_positions.fen'
    if not pos_file.is_file():
        print(f"Test positions not found: {pos_file}", file=sys.stderr)
        print(f"Run prepare_test_positions.py first.", file=sys.stderr)
        sys.exit(1)

    positions = load_positions(pos_file)
    if args.limit > 0:
        positions = positions[:args.limit]
    print(f"Loaded {len(positions)} test positions from {pos_file.name}")

    # Instantiate evaluators
    print(f"Initializing BoardEvaluatorTrad and BoardEvaluatorNN (depth={args.nn_depth})...")
    from board_evaluator_trad import BoardEvaluatorTrad
    from board_evaluator_nn import BoardEvaluatorNN

    eval_args = make_eval_args(args.stockfish, args.nn_depth)
    trad = BoardEvaluatorTrad(eval_args, chess.WHITE)
    nn = BoardEvaluatorNN(eval_args, chess.WHITE)

    # Ground-truth and baseline Stockfish
    print(f"Initializing Stockfish d=1 baseline and d={args.ground_truth_depth} ground truth...")
    engine_gt = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine_d1 = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        engine_gt.configure({'Threads': 1, 'Hash': 256})
    except chess.engine.EngineError:
        pass

    rows = []
    t0 = time.perf_counter()
    try:
        for i, (fen, tag) in enumerate(positions, 1):
            try:
                board = chess.Board(fen)
            except ValueError as e:
                print(f"  [{i}] invalid FEN: {fen} ({e})", file=sys.stderr)
                continue

            phase = game_phase(board)
            band = phase_band(phase)

            try:
                e_trad = trad.evaluate_board(board)
            except Exception as e:
                print(f"  [{i}] TRAD error: {e}", file=sys.stderr)
                e_trad = math.nan
            try:
                e_nn = nn.evaluate_board(board)
            except Exception as e:
                print(f"  [{i}] NN error: {e}", file=sys.stderr)
                e_nn = math.nan
            try:
                e_sf1 = sf_eval_cp(engine_d1, board, 1)
            except Exception as e:
                print(f"  [{i}] SF d=1 error: {e}", file=sys.stderr)
                e_sf1 = math.nan
            try:
                e_sf20 = sf_eval_cp(engine_gt, board, args.ground_truth_depth)
            except Exception as e:
                print(f"  [{i}] SF d={args.ground_truth_depth} error: {e}", file=sys.stderr)
                e_sf20 = math.nan

            # Make TRAD/NN consistent with SF convention (always from White's perspective)
            # BoardEvaluatorTrad/NN return eval from side-to-move perspective in some
            # implementations. Normalize to White's perspective:
            if board.turn == chess.BLACK:
                e_trad_white = -e_trad
                e_nn_white = -e_nn
            else:
                e_trad_white = e_trad
                e_nn_white = e_nn

            # Cap values to avoid mate-score noise dominating MAE
            def cap(v):
                if v is None or math.isnan(v) or math.isinf(v):
                    return math.nan
                return max(-50.0, min(50.0, v))

            rows.append({
                'idx': i,
                'fen': fen,
                'tag': tag,
                'phase': round(phase, 3),
                'phase_band': band,
                'eval_trad': cap(e_trad_white),
                'eval_nn': cap(e_nn_white),
                'eval_sf_d1': cap(e_sf1),
                'eval_sf_d20': cap(e_sf20),
            })

            if i % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  [{i}/{len(positions)}] elapsed {elapsed:.1f}s")

    finally:
        engine_gt.quit()
        engine_d1.quit()

    df = pd.DataFrame(rows)
    out_csv = script_dir / 'exp4a_evaluations.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name} ({len(df)} rows)")

    # Compute accuracy metrics
    from scipy.stats import spearmanr

    summary_rows = []
    bands = ['all', 'opening', 'midgame', 'endgame']
    evaluators = [('TRAD', 'eval_trad'), ('NN', 'eval_nn'), ('SF_d1', 'eval_sf_d1')]
    gt_col = 'eval_sf_d20'

    for band in bands:
        if band == 'all':
            sub = df
        else:
            sub = df[df['phase_band'] == band]
        if sub.empty:
            continue
        gt = sub[gt_col].dropna()
        if gt.empty:
            continue

        for name, col in evaluators:
            ev = sub[[col, gt_col]].dropna()
            if ev.empty or len(ev) < 2:
                continue
            x = ev[col].values
            y = ev[gt_col].values
            rho, _ = spearmanr(x, y)
            mae = float(np.mean(np.abs(x - y)))
            rmse = float(np.sqrt(np.mean((x - y) ** 2)))
            summary_rows.append({
                'evaluator': name,
                'phase_band': band,
                'n_positions': len(ev),
                'spearman_rho': round(float(rho), 4),
                'mae': round(mae, 3),
                'rmse': round(rmse, 3),
            })

    summary = pd.DataFrame(summary_rows)
    sum_csv = script_dir / 'exp4a_accuracy_summary.csv'
    summary.to_csv(sum_csv, index=False)
    print(f"Saved: {sum_csv.name}")

    print("\n--- Accuracy vs Stockfish d=" + str(args.ground_truth_depth) + " ---")
    print(summary.to_string(index=False))

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    plots_dir = script_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Scatter plots per evaluator
    for name, col in evaluators:
        sub = df[[col, gt_col, 'phase_band']].dropna()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 7))
        colors = {'opening': '#1976D2', 'midgame': '#9E9E9E', 'endgame': '#D32F2F'}
        for band in ['opening', 'midgame', 'endgame']:
            b = sub[sub['phase_band'] == band]
            ax.scatter(b[gt_col], b[col], alpha=0.5, label=band,
                       color=colors[band], s=20)
        lim = max(abs(sub[gt_col].min()), abs(sub[gt_col].max()),
                  abs(sub[col].min()), abs(sub[col].max()))
        ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, alpha=0.5, label='y=x')
        ax.set_xlabel(f'Stockfish d={args.ground_truth_depth} (pawns)')
        ax.set_ylabel(f'{name} eval (pawns)')
        ax.set_title(f'{name} vs Stockfish d={args.ground_truth_depth}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f'exp4a_scatter_{name.lower()}.png', dpi=150)
        plt.close(fig)

    # MAE by phase grouped bar chart
    if not summary.empty:
        phase_only = summary[summary['phase_band'] != 'all']
        if not phase_only.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            pivot = phase_only.pivot(index='phase_band', columns='evaluator', values='mae')
            pivot = pivot.reindex(['opening', 'midgame', 'endgame'])
            pivot.plot(kind='bar', ax=ax)
            ax.set_ylabel('MAE (pawns)')
            ax.set_title(f'Mean Absolute Error vs Stockfish d={args.ground_truth_depth} by phase')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=0)
            fig.tight_layout()
            fig.savefig(plots_dir / 'exp4a_mae_by_phase.png', dpi=150)
            plt.close(fig)

    # Text summary
    txt_path = script_dir / 'exp4a_accuracy_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Experiment 4a — Static evaluation accuracy\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Ground truth: Stockfish d={args.ground_truth_depth}\n')
        f.write(f'NN evaluator depth: {args.nn_depth}\n')
        f.write(f'Test positions: {len(df)}\n\n')
        f.write('Accuracy metrics:\n')
        f.write(summary.to_string(index=False) + '\n')
    print(f"Saved: {txt_path.name}")

    print(f"\nPlots saved to: {plots_dir}")
    print("Done.")


if __name__ == '__main__':
    main()
