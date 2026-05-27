"""
Experiment 4b — Move agreement with Stockfish d=20.

For each test position:
  1. Get Stockfish d=20 top-3 best moves (ground truth).
  2. For each of 4 engine variants (MINIMAX_TRAD/NN, MCTS_TRAD/NN),
     run main.py via subprocess with -i <fen> for a single move,
     extract the engine's chosen move.
  3. Record match (exact) and top-3 match per variant.

Output:
  exp4b_moves.csv               — per-position-and-variant: variant choice + SF top-3
  exp4b_move_agreement.csv      — per-variant: match rate, top-3 match rate, stratified by phase
  plots/exp4b_match_rate.png    — bar chart of match rates

Usage:
    python run_exp4b_move_agreement.py --stockfish <path>
    python run_exp4b_move_agreement.py --stockfish <path> --positions test_positions.fen \
        --minimax-depth 4 --mcts-time 1.0 --limit 20

Performance note: each position = 4 main.py invocations. With Python startup
(~1.5s) + search time (~1-10s), this is heavy. For 200 positions × 4 variants
= 800 runs × ~5s avg = ~1h.
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pandas as pd


VARIANTS = [
    {'name': 'MINIMAX_TRAD_d4', 'type': 'MINIMAX_TRAD', 'depth_arg': '-dw', 'time_arg': None},
    {'name': 'MINIMAX_NN_d3',   'type': 'MINIMAX_NN',   'depth_arg': '-dw', 'time_arg': None},
    {'name': 'MCTS_TRAD',       'type': 'MCTS_TRAD',    'depth_arg': None,  'time_arg': '-mtw'},
    {'name': 'MCTS_NN',         'type': 'MCTS_NN',      'depth_arg': None,  'time_arg': '-mtw'},
]


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
    material = sum(w * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
                   for pt, w in PHASE_WEIGHTS.items())
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


def get_sf_topk(engine: chess.engine.SimpleEngine, board: chess.Board,
                depth: int, k: int = 3) -> list[str]:
    """Returns up to k best UCI moves from Stockfish at given depth."""
    info_list = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=k)
    moves = []
    for info in info_list:
        pv = info.get('pv')
        if pv:
            moves.append(pv[0].uci())
    return moves


def run_variant_on_position(variant: dict, fen: str, engine_dir: Path,
                            python_exe: str, stockfish_path: str,
                            minimax_depth: int, mcts_time: float,
                            timeout: float = 120.0) -> str | None:
    """
    Runs main.py on the given FEN for a single move (we set the opponent to
    HUMAN with no input - the engine plays its move, then the game blocks).

    Workaround: use stockfish (very fast) as opponent at skill 0; engine plays
    first move, then result writes -o output. We extract first move from -g file.
    """
    # Write FEN to a temp file in out/
    out_dir = engine_dir / 'out'
    out_dir.mkdir(exist_ok=True)

    tag = f"_exp4b_{variant['name']}_{abs(hash(fen)) % 100000}"
    fen_file = f'_temp{tag}.fen'
    game_file = f'_temp{tag}_game.txt'
    log_file = f'_temp{tag}_log.txt'

    (out_dir / fen_file).write_text(fen, encoding='utf-8')

    # Determine which side the engine plays from the FEN's side-to-move
    board = chess.Board(fen)
    engine_is_white = (board.turn == chess.WHITE)

    cmd = [python_exe, 'main.py', '-m', 'B', '-i', fen_file,
           '-g', game_file, '-l', log_file,
           '-sp', stockfish_path,
           # Adjudicate quickly to avoid long games
           '-adj', '-adjt', '0.1', '-adjm', '5']

    # Set engine variant as the side-to-move
    if engine_is_white:
        cmd += ['-w', variant['type'], '-b', 'STOCKFISH', '-sb', '0', '-dbs', '1']
    else:
        cmd += ['-b', variant['type'], '-w', 'STOCKFISH', '-sw', '0', '-dws', '1']

    if variant['depth_arg']:
        # depth_white parameter is for the engine, regardless of color
        dw_value = str(minimax_depth)
        if engine_is_white:
            cmd += ['-dw', dw_value]
        else:
            cmd += ['-db', dw_value]
    if variant['time_arg']:
        mt_value = str(mcts_time)
        if engine_is_white:
            cmd += ['-mtw', mt_value]
        else:
            cmd += ['-mtb', mt_value]

    try:
        subprocess.run(cmd, cwd=str(engine_dir), timeout=timeout,
                       capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for {variant['name']}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ERROR running {variant['name']}: {e}", file=sys.stderr)
        return None

    # Parse the engine's first move from the game file
    # File format: "Result: ...\n\n1: <uci>\n2: <uci>\n..."
    game_path = out_dir / game_file
    if not game_path.exists():
        return None
    try:
        content = game_path.read_text(encoding='utf-8')
        for line in content.splitlines():
            m = re.match(r'^(\d+):\s*(\S+)\s*$', line.strip())
            if m:
                move_idx = int(m.group(1))
                # If engine plays white, move 1 is the engine's first.
                # If engine plays black, move 2 is the engine's first.
                if (engine_is_white and move_idx == 1) or \
                        (not engine_is_white and move_idx == 2):
                    return m.group(2)
    except Exception as e:
        print(f"  parse error: {e}", file=sys.stderr)
        return None
    finally:
        # Clean up temp files
        for f in [fen_file, game_file, log_file]:
            p = out_dir / f
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    return None


def main():
    parser = argparse.ArgumentParser(description='Exp 4b — Move agreement with Stockfish')
    parser.add_argument('--stockfish', type=str, required=True)
    parser.add_argument('--positions', type=str, default='')
    parser.add_argument('--minimax-depth', type=int, default=4)
    parser.add_argument('--minimax-nn-depth', type=int, default=3)
    parser.add_argument('--mcts-time', type=float, default=1.0)
    parser.add_argument('--ground-truth-depth', type=int, default=20)
    parser.add_argument('--python', type=str, default='python')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--variants', type=str, default='1,2,3,4',
                        help='Comma-separated variant indices (1=MINIMAX_TRAD, 2=MINIMAX_NN, 3=MCTS_TRAD, 4=MCTS_NN)')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    engine_dir = script_dir.parent.parent

    pos_file = Path(args.positions) if args.positions else script_dir / 'test_positions.fen'
    if not pos_file.is_file():
        print(f"Test positions not found: {pos_file}", file=sys.stderr)
        sys.exit(1)

    positions = load_positions(pos_file)
    if args.limit > 0:
        positions = positions[:args.limit]
    print(f"Loaded {len(positions)} positions")

    selected_idx = [int(x) - 1 for x in args.variants.split(',')]
    selected_variants = [VARIANTS[i] for i in selected_idx]
    print(f"Variants: {[v['name'] for v in selected_variants]}")

    # Get Stockfish d=20 top-3 moves for all positions
    print(f"Querying Stockfish d={args.ground_truth_depth} top-3 for {len(positions)} positions...")
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        engine.configure({'Threads': 1, 'Hash': 256})
    except chess.engine.EngineError:
        pass

    sf_topk = {}
    try:
        for i, (fen, _tag) in enumerate(positions, 1):
            try:
                board = chess.Board(fen)
                sf_topk[fen] = get_sf_topk(engine, board, args.ground_truth_depth, k=3)
            except Exception as e:
                print(f"  SF failed on position {i}: {e}", file=sys.stderr)
                sf_topk[fen] = []
            if i % 20 == 0:
                print(f"  SF analysis {i}/{len(positions)}")
    finally:
        engine.quit()

    # Run each variant on each position
    rows = []
    t0 = time.perf_counter()
    total_calls = len(positions) * len(selected_variants)
    call_idx = 0

    for variant in selected_variants:
        # Determine depth/time for this variant
        if variant['name'] == 'MINIMAX_NN_d3':
            mm_depth = args.minimax_nn_depth
        else:
            mm_depth = args.minimax_depth

        for fen, tag in positions:
            call_idx += 1
            board = chess.Board(fen)
            phase = game_phase(board)

            t_start = time.perf_counter()
            variant_move = run_variant_on_position(
                variant, fen, engine_dir, args.python, args.stockfish,
                mm_depth, args.mcts_time
            )
            duration = time.perf_counter() - t_start

            top3 = sf_topk.get(fen, [])
            sf_best = top3[0] if top3 else None
            match = (variant_move == sf_best) if variant_move and sf_best else False
            top3_match = (variant_move in top3) if variant_move else False

            rows.append({
                'variant': variant['name'],
                'fen': fen,
                'tag': tag,
                'phase': round(phase, 3),
                'phase_band': phase_band(phase),
                'variant_move': variant_move,
                'sf_best': sf_best,
                'sf_top3': '|'.join(top3),
                'match': match,
                'top3_match': top3_match,
                'duration_s': round(duration, 2),
            })

            elapsed = time.perf_counter() - t0
            pct = call_idx / total_calls * 100
            print(f"  [{call_idx}/{total_calls} {pct:.0f}%] {variant['name']:18s} "
                  f"-> {variant_move or 'FAIL':6s} (SF best: {sf_best}, "
                  f"{'MATCH' if match else 'top3' if top3_match else 'miss'}) "
                  f"[{duration:.1f}s, total {elapsed/60:.1f}min]")

    df = pd.DataFrame(rows)
    out_csv = script_dir / 'exp4b_moves.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name} ({len(df)} rows)")

    # Aggregate
    summary_rows = []
    for variant_name in df['variant'].unique():
        for band in ['all', 'opening', 'midgame', 'endgame']:
            if band == 'all':
                sub = df[df['variant'] == variant_name]
            else:
                sub = df[(df['variant'] == variant_name) & (df['phase_band'] == band)]
            if sub.empty:
                continue
            n = len(sub)
            matches = sub['match'].sum()
            top3 = sub['top3_match'].sum()
            successful = sub['variant_move'].notna().sum()
            summary_rows.append({
                'variant': variant_name,
                'phase_band': band,
                'n': n,
                'successful': successful,
                'match_rate': round(matches / n, 4) if n > 0 else 0,
                'top3_match_rate': round(top3 / n, 4) if n > 0 else 0,
            })

    summary = pd.DataFrame(summary_rows)
    summary_csv = script_dir / 'exp4b_move_agreement.csv'
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv.name}")

    print("\n--- Move agreement summary ---")
    print(summary.to_string(index=False))

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plots_dir = script_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    all_band = summary[summary['phase_band'] == 'all'].copy().sort_values('match_rate', ascending=False)
    if not all_band.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(all_band))
        w = 0.4
        ax.bar(x - w / 2, all_band['match_rate'] * 100, w, label='Exact match', color='#1976D2')
        ax.bar(x + w / 2, all_band['top3_match_rate'] * 100, w, label='Top-3 match', color='#4CAF50')
        ax.set_xticks(x)
        ax.set_xticklabels(all_band['variant'], rotation=20, ha='right')
        ax.set_ylabel('Agreement rate (%)')
        ax.set_title(f'Move agreement with Stockfish d={args.ground_truth_depth}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(plots_dir / 'exp4b_match_rate.png', dpi=150)
        plt.close(fig)
        print(f"Saved: plots/exp4b_match_rate.png")

    print("\nDone.")


if __name__ == '__main__':
    main()
