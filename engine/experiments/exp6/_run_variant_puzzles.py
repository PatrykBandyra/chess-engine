"""
Internal helper for run_exp6_variant.ps1.

For each puzzle in puzzles.json:
  - Runs main.py with -i <fen> (engine plays as side-to-move).
  - Extracts engine's first move from game file.
  - Compares to puzzle's best_moves_uci.
  - Records: solved (bool), move, time, parsing of log for min-depth-to-solve.

Output: CSV with one row per puzzle.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import chess
import pandas as pd


def run_variant_on_puzzle(variant_type: str, puzzle: dict, engine_dir: Path,
                          python_exe: str, stockfish_path: str,
                          minimax_depth: int, mcts_time: float,
                          puzzle_tag: str, timeout: float = 120.0) -> dict:
    """Runs main.py on one puzzle. Returns dict with engine_move, duration, log_path."""
    out_dir = engine_dir / 'out'
    out_dir.mkdir(exist_ok=True)

    fen = puzzle['fen']
    side = puzzle['side_to_move']
    engine_is_white = (side == 'w')

    safe_tag = f"_exp6_{puzzle_tag}_{abs(hash(fen)) % 100000}"
    fen_file = f'_temp{safe_tag}.fen'
    game_file = f'_temp{safe_tag}_game.txt'
    log_file = f'_temp{safe_tag}_log.txt'
    json_file = f'_temp{safe_tag}_metrics.jsonl'

    (out_dir / fen_file).write_text(fen, encoding='utf-8')

    cmd = [python_exe, 'main.py', '-m', 'B',
           '-i', fen_file,
           '-g', game_file,
           '-l', log_file,
           '-jl', json_file,
           '-sp', stockfish_path,
           '-adj', '-adjt', '0.1', '-adjm', '5']

    # Engine plays the puzzle side, Stockfish skill 0 plays opponent
    if engine_is_white:
        cmd += ['-w', variant_type, '-b', 'STOCKFISH', '-sb', '0', '-dbs', '1']
        if variant_type.startswith('MINIMAX'):
            cmd += ['-dw', str(minimax_depth)]
        if variant_type.startswith('MCTS'):
            cmd += ['-mtw', str(mcts_time)]
    else:
        cmd += ['-b', variant_type, '-w', 'STOCKFISH', '-sw', '0', '-dws', '1']
        if variant_type.startswith('MINIMAX'):
            cmd += ['-db', str(minimax_depth)]
        if variant_type.startswith('MCTS'):
            cmd += ['-mtb', str(mcts_time)]

    t0 = time.perf_counter()
    timed_out = False
    try:
        subprocess.run(cmd, cwd=str(engine_dir), timeout=timeout,
                       capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        timed_out = True
    duration = time.perf_counter() - t0

    # Parse engine move from game file
    engine_move_uci = None
    game_path = out_dir / game_file
    if game_path.exists():
        try:
            content = game_path.read_text(encoding='utf-8')
            for line in content.splitlines():
                m = re.match(r'^(\d+):\s*(\S+)\s*$', line.strip())
                if m:
                    move_idx = int(m.group(1))
                    if (engine_is_white and move_idx == 1) or \
                            (not engine_is_white and move_idx == 2):
                        engine_move_uci = m.group(2)
                        break
        except Exception as e:
            print(f"  parse_game error: {e}", file=sys.stderr)

    # For Minimax: parse log for ID iteration moves to find min depth where solution appears
    min_depth_to_solve = None
    log_path = out_dir / log_file
    if log_path.exists() and variant_type.startswith('MINIMAX'):
        try:
            log_content = log_path.read_text(encoding='utf-8')
            # Lines: "ID iteration depth=N; move: <uci>; value: <V>"
            for log_line in log_content.splitlines():
                m = re.search(r'ID iteration depth=(\d+);\s*move:\s*(\S+);', log_line)
                if m:
                    d = int(m.group(1))
                    mv = m.group(2)
                    if mv in puzzle['best_moves_uci']:
                        if min_depth_to_solve is None or d < min_depth_to_solve:
                            min_depth_to_solve = d
        except Exception:
            pass

    # Clean up temp files
    for f in [fen_file, game_file, log_file, json_file]:
        p = out_dir / f
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    solved = engine_move_uci in puzzle['best_moves_uci'] if engine_move_uci else False

    return {
        'engine_move': engine_move_uci,
        'solved': solved,
        'duration_s': round(duration, 2),
        'timed_out': timed_out,
        'min_depth_to_solve': min_depth_to_solve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--puzzles', required=True)
    parser.add_argument('--variant-name', required=True)
    parser.add_argument('--variant-type', required=True)
    parser.add_argument('--minimax-depth', type=int, default=4)
    parser.add_argument('--mcts-time', type=float, default=1.0)
    parser.add_argument('--stockfish-path', required=True)
    parser.add_argument('--engine-dir', required=True)
    parser.add_argument('--python', default='python')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    engine_dir = Path(args.engine_dir)
    puzzles = json.loads(Path(args.puzzles).read_text(encoding='utf-8-sig'))

    if args.limit > 0:
        puzzles = puzzles[:args.limit]

    print(f"Running {args.variant_name} on {len(puzzles)} puzzles")
    print(f"Output: {args.output}")
    print()

    rows = []
    t0 = time.perf_counter()
    solved_count = 0

    for i, puzzle in enumerate(puzzles, 1):
        print(f"  [{i}/{len(puzzles)}] {puzzle['id']:25s} ", end='', flush=True)

        result = run_variant_on_puzzle(
            args.variant_type, puzzle, engine_dir, args.python,
            args.stockfish_path, args.minimax_depth, args.mcts_time,
            puzzle_tag=f"v_{args.variant_name}_p{i}",
        )

        rows.append({
            'puzzle_id': puzzle['id'],
            'theme': puzzle['theme'],
            'fen': puzzle['fen'],
            'expected_uci': '|'.join(puzzle['best_moves_uci']),
            'expected_san': '|'.join(puzzle['best_moves_san']),
            'engine_move_uci': result['engine_move'],
            'solved': result['solved'],
            'duration_s': result['duration_s'],
            'timed_out': result['timed_out'],
            'min_depth_to_solve': result['min_depth_to_solve'],
        })

        if result['solved']:
            solved_count += 1
            status = 'SOLVED'
        elif result['timed_out']:
            status = 'TIMEOUT'
        else:
            status = 'miss'

        depth_info = f" d={result['min_depth_to_solve']}" if result['min_depth_to_solve'] else ''
        print(f"{status:8s} ({result['engine_move'] or 'no move'}, "
              f"expected {'/'.join(puzzle['best_moves_uci'])}){depth_info} [{result['duration_s']:.1f}s]")

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Solve rate: {solved_count}/{len(puzzles)} ({100 * solved_count / len(puzzles):.1f}%)")
    print(f"Total time: {elapsed / 60:.1f} min")
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
