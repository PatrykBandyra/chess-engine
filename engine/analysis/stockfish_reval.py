"""
Post-hoc Stockfish re-evaluation for played games.

For each move in each game, queries Stockfish at high depth to get the
'ground truth' eval and best move. Computes ACPL (average centipawn loss)
per side, blunder rate, and per-move evaluations.

This is essential for Experiment 5 (Stockfish benchmark) but works on any
metrics directory.

Output per game (one JSONL line in stockfish_reval.jsonl + summary CSV):
    {
        "file": "metrics_<label>_g<N>_<colortag>.jsonl",
        "label": "...",
        "result": "1-0",
        "acpl_white": 25.3,
        "acpl_black": 42.1,
        "blunders_white": 1,
        "blunders_black": 4,
        "moves_evaluated": 60
    }

Usage:
    python stockfish_reval.py <experiment_dir> --stockfish <path> [--depth 20] [--limit 5]
        --limit N          process only first N files (testing)
        --opening-set      path to openings FEN file (auto-detected if exp uses openings)

Performance note: at depth=20, ~2s/position. A 60-move game = ~120 positions
= ~4 minutes per game. For 640 games this is ~40 hours sequential.
Lower depth (e.g., 15) gives ~10x speedup with acceptable accuracy for ACPL.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import chess
import chess.engine
import pandas as pd


BLUNDER_CP_THRESHOLD = 100  # centipawn loss threshold for blunder classification
MATE_CP = 30000             # cap mate scores to avoid overflow in ACPL


def score_to_cp(score: chess.engine.PovScore, perspective: chess.Color) -> int:
    """Convert engine score to centipawns from given perspective."""
    rel = score.pov(perspective)
    if rel.is_mate():
        m = rel.mate()
        # Closer mate = larger magnitude
        return MATE_CP if m > 0 else -MATE_CP
    return rel.score()


def parse_filename(stem: str) -> dict:
    """Parse metrics_<label>_g<N>_<orig|swap>.jsonl filename to extract label, game idx, swapped."""
    # stem like: metrics_minimax_trad_d4_vs_stockfish_sk5_g3_orig
    s = stem.removeprefix('metrics_')
    m = re.match(r'^(?P<label>.+?)_g(?P<idx>\d+)_(?P<color>orig|swap)$', s)
    if not m:
        return {'label': s, 'game_idx': None, 'swapped': False}
    return {
        'label': m.group('label'),
        'game_idx': int(m.group('idx')),
        'swapped': m.group('color') == 'swap',
    }


def load_metrics_moves(path: Path) -> tuple[list[dict], dict | None]:
    """Returns (per-move records, game_summary or None)."""
    moves = []
    summary = None
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get('type') == 'game_summary':
                summary = rec
            else:
                moves.append(rec)
    return moves, summary


def load_openings(openings_file: Path) -> list[str]:
    """Load FENs from openings file (lines: FEN [# comment])."""
    if not openings_file.is_file():
        return []
    fens = []
    for line in openings_file.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        fen = line.split('#', 1)[0].strip()
        if fen:
            fens.append(fen)
    return fens


def reval_game(moves: list[dict], starting_fen: str | None,
               engine: chess.engine.SimpleEngine, depth: int,
               progress_prefix: str = '') -> dict:
    """Re-evaluates a single game. Returns summary dict."""
    if starting_fen:
        board = chess.Board(starting_fen)
    else:
        board = chess.Board()

    cp_loss_white = []
    cp_loss_black = []
    moves_data = []
    blunders_white = 0
    blunders_black = 0

    for i, move_rec in enumerate(moves):
        uci = move_rec.get('move')
        side = move_rec.get('side')
        if not uci:
            continue
        try:
            actual_move = chess.Move.from_uci(uci)
        except ValueError:
            continue

        if actual_move not in board.legal_moves:
            print(f"{progress_prefix}  Illegal move {uci} at position; aborting game reval", file=sys.stderr)
            break

        side_to_move = board.turn

        # Eval before move from side-to-move perspective
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
        except chess.engine.EngineError as e:
            print(f"{progress_prefix}  Engine error: {e}", file=sys.stderr)
            break

        best_move = info.get('pv', [None])[0]
        eval_before_cp = score_to_cp(info['score'], side_to_move)

        # Push the actual move
        board.push(actual_move)

        # Eval after move (from same side perspective = -eval from opponent)
        try:
            info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
        except chess.engine.EngineError as e:
            print(f"{progress_prefix}  Engine error: {e}", file=sys.stderr)
            break

        eval_after_cp = score_to_cp(info_after['score'], side_to_move)

        # Centipawn loss = (best possible - what we got)
        # Best move score is what the position evaluates to with best play.
        # After our move, the position is now opponent's turn; eval from our
        # perspective = eval_after_cp (already from side_to_move perspective).
        # If best_move was played, eval_after would equal eval_before (approximately).
        # The 'cost' of our move = eval_before - eval_after (in our favor).
        # If our move was the best, cost ≈ 0. If we made a blunder, cost > 0.
        cp_loss = max(0, eval_before_cp - eval_after_cp)

        # Cap CP loss at a reasonable maximum
        cp_loss = min(cp_loss, 1000)

        if side_to_move == chess.WHITE:
            cp_loss_white.append(cp_loss)
            if cp_loss > BLUNDER_CP_THRESHOLD:
                blunders_white += 1
        else:
            cp_loss_black.append(cp_loss)
            if cp_loss > BLUNDER_CP_THRESHOLD:
                blunders_black += 1

        moves_data.append({
            'move_number': move_rec.get('move_number'),
            'side': side,
            'uci': uci,
            'eval_before_cp': eval_before_cp,
            'eval_after_cp': eval_after_cp,
            'cp_loss': cp_loss,
            'best_move_sf': best_move.uci() if best_move else None,
            'is_blunder': cp_loss > BLUNDER_CP_THRESHOLD,
            'phase': move_rec.get('phase'),
        })

    def safe_mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    return {
        'acpl_white': safe_mean(cp_loss_white),
        'acpl_black': safe_mean(cp_loss_black),
        'blunders_white': blunders_white,
        'blunders_black': blunders_black,
        'moves_white': len(cp_loss_white),
        'moves_black': len(cp_loss_black),
        'moves_data': moves_data,
    }


def main():
    parser = argparse.ArgumentParser(description='Stockfish re-evaluation of played games')
    parser.add_argument('experiment_dir', type=str)
    parser.add_argument('--stockfish', type=str, required=True, help='Path to Stockfish binary')
    parser.add_argument('--depth', type=int, default=20, help='SF analysis depth (default 20)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of files (0=all)')
    parser.add_argument('--openings-file', type=str, default='',
                        help='Path to openings FEN file (default: experiments/openings_eco25.fen)')
    parser.add_argument('--per-move-csv', action='store_true',
                        help='Also output per-move CSV (large file)')
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_dir():
        print(f"Error: {exp_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    sf_path = Path(args.stockfish)
    if not sf_path.is_file():
        print(f"Error: Stockfish binary not found: {sf_path}", file=sys.stderr)
        sys.exit(1)

    # Load openings (for starting FEN of each game)
    if args.openings_file:
        openings_path = Path(args.openings_file)
    else:
        # Default location relative to experiment_dir
        openings_path = exp_dir.parent.parent / 'experiments' / 'openings_eco25.fen'

    openings = load_openings(openings_path) if openings_path.exists() else []
    if openings:
        print(f"Loaded {len(openings)} opening FENs from {openings_path}")
    else:
        print(f"No openings file found at {openings_path} — assuming standard start for all games")

    # Find metrics files
    metrics_files = sorted(exp_dir.glob('metrics_*.jsonl'))
    if args.limit > 0:
        metrics_files = metrics_files[:args.limit]

    print(f"Found {len(metrics_files)} metrics files to re-evaluate")
    print(f"Stockfish: {sf_path} (depth={args.depth})")

    # Open Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    try:
        engine.configure({'Threads': 1, 'Hash': 256})
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
        pass

    results = []
    per_move_rows = []

    try:
        for i, mfile in enumerate(metrics_files, 1):
            prefix = f"[{i}/{len(metrics_files)}]"
            print(f"{prefix} {mfile.name} ... ", end='', flush=True)

            parsed = parse_filename(mfile.stem)
            moves, summary = load_metrics_moves(mfile)

            if not moves:
                print('skip (no moves)')
                continue

            # Determine starting FEN
            starting_fen = None
            if openings and parsed['game_idx']:
                opening_idx = (parsed['game_idx'] - 1) % len(openings)
                starting_fen = openings[opening_idx]

            try:
                result = reval_game(moves, starting_fen, engine, args.depth, prefix)
            except Exception as e:
                print(f'ERROR: {e}')
                continue

            row = {
                'file': mfile.name,
                'label': parsed['label'],
                'game_idx': parsed['game_idx'],
                'swapped': parsed['swapped'],
                'result': (summary or {}).get('result', '?'),
                'termination': (summary or {}).get('termination', '?'),
                'total_moves': (summary or {}).get('total_moves'),
                'acpl_white': result['acpl_white'],
                'acpl_black': result['acpl_black'],
                'blunders_white': result['blunders_white'],
                'blunders_black': result['blunders_black'],
                'moves_white': result['moves_white'],
                'moves_black': result['moves_black'],
            }
            results.append(row)

            if args.per_move_csv:
                for m in result['moves_data']:
                    m_row = dict(m)
                    m_row['file'] = mfile.name
                    m_row['label'] = parsed['label']
                    per_move_rows.append(m_row)

            print(f"ACPL W={result['acpl_white']:.1f} B={result['acpl_black']:.1f}  "
                  f"blunders W={result['blunders_white']} B={result['blunders_black']}")

    finally:
        engine.quit()

    # Save outputs
    df = pd.DataFrame(results)
    out_csv = exp_dir / 'stockfish_reval.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name} ({len(df)} games)")

    if args.per_move_csv and per_move_rows:
        out_moves = exp_dir / 'stockfish_reval_moves.csv'
        pd.DataFrame(per_move_rows).to_csv(out_moves, index=False)
        print(f"Saved: {out_moves.name} ({len(per_move_rows)} moves)")

    # Summary
    if not df.empty:
        print("\n--- Summary ---")
        for col in ['acpl_white', 'acpl_black', 'blunders_white', 'blunders_black']:
            print(f"  {col:20s} mean={df[col].mean():7.2f}  median={df[col].median():7.2f}")

    print('\nDone.')


if __name__ == '__main__':
    main()
