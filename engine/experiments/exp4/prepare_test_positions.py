"""
Prepare 200 test positions for Experiment 4 (evaluation function comparison).

Strategy: starting from each of 25 ECO openings, plays random legal sequences
of moves and samples positions stratified by game phase:
  - 50 opening   (phase > 0.8)
  - 100 midgame  (phase 0.3..0.8)
  - 50 endgame   (phase < 0.3)

Phase metric matches BoardEvaluatorTrad.__get_game_phase (non-pawn material
weighted KNIGHT=3, BISHOP=3, ROOK=5, QUEEN=9; both sides combined).

Output:
  engine/experiments/exp4/test_positions.fen — one FEN per line, optional comment

Usage:
    python prepare_test_positions.py
    python prepare_test_positions.py --seed 123 --target-opening 60 --target-mid 90 --target-end 50
"""

import argparse
import random
import sys
from pathlib import Path

import chess


PHASE_WEIGHTS = {
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}
ENDGAME_THRESHOLD = 20.0
MIDDLEGAME_THRESHOLD = 67.0


def game_phase(board: chess.Board) -> float:
    """Returns float in [0, 1]. 1.0 = middlegame, 0.0 = endgame."""
    material = 0.0
    for piece_type, weight in PHASE_WEIGHTS.items():
        material += weight * (
            len(board.pieces(piece_type, chess.WHITE))
            + len(board.pieces(piece_type, chess.BLACK))
        )
    if material >= MIDDLEGAME_THRESHOLD:
        return 1.0
    if material <= ENDGAME_THRESHOLD:
        return 0.0
    return (material - ENDGAME_THRESHOLD) / (MIDDLEGAME_THRESHOLD - ENDGAME_THRESHOLD)


def load_openings(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '#' in line:
            fen, name = line.split('#', 1)
            out.append((fen.strip(), name.strip()))
        else:
            out.append((line, ''))
    return out


def random_play(start_fen: str, rng: random.Random, max_plies: int,
                sample_every: int) -> list[tuple[str, float, str]]:
    """Plays random legal moves from start_fen, sampling positions periodically."""
    board = chess.Board(start_fen)
    samples = []
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        board.push(move)
        if (ply + 1) % sample_every == 0:
            samples.append((board.fen(), game_phase(board), f'ply={ply+1}'))
    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate test positions for Exp 4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target-opening', type=int, default=50, help='Phase > 0.8')
    parser.add_argument('--target-mid', type=int, default=100, help='Phase 0.3..0.8')
    parser.add_argument('--target-end', type=int, default=50, help='Phase < 0.3')
    parser.add_argument('--openings-file', type=str, default='',
                        help='Default: experiments/openings_eco25.fen')
    parser.add_argument('--output', type=str, default='',
                        help='Default: experiments/exp4/test_positions.fen')
    parser.add_argument('--max-attempts', type=int, default=200,
                        help='Max random walks per opening')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    engine_dir = script_dir.parent.parent

    openings_file = Path(args.openings_file) if args.openings_file else \
        engine_dir / 'experiments' / 'openings_eco25.fen'
    output_file = Path(args.output) if args.output else script_dir / 'test_positions.fen'

    if not openings_file.is_file():
        print(f"Openings file not found: {openings_file}", file=sys.stderr)
        sys.exit(1)

    openings = load_openings(openings_file)
    print(f"Loaded {len(openings)} openings from {openings_file.name}")

    rng = random.Random(args.seed)

    opening_pool = []
    midgame_pool = []
    endgame_pool = []

    # Walk each opening multiple times with different random seeds
    for attempt in range(args.max_attempts):
        for fen, name in openings:
            samples = random_play(fen, rng, max_plies=200, sample_every=8)
            # Include the opening position itself
            samples.insert(0, (fen, game_phase(chess.Board(fen)), f'{name} start'))

            for sample_fen, phase, note in samples:
                tag = f'{name} {note} phase={phase:.2f}'.strip()
                if phase > 0.8:
                    opening_pool.append((sample_fen, tag))
                elif phase < 0.3:
                    endgame_pool.append((sample_fen, tag))
                else:
                    midgame_pool.append((sample_fen, tag))

        if (len(opening_pool) >= args.target_opening * 5
                and len(midgame_pool) >= args.target_mid * 5
                and len(endgame_pool) >= args.target_end * 5):
            break

    print(f"  Pool sizes: opening={len(opening_pool)}, mid={len(midgame_pool)}, end={len(endgame_pool)}")

    # Deduplicate by FEN and sample required counts
    def sample_unique(pool, target):
        seen = set()
        unique = []
        rng.shuffle(pool)
        for fen, tag in pool:
            if fen in seen:
                continue
            seen.add(fen)
            unique.append((fen, tag))
            if len(unique) >= target:
                break
        return unique

    opening_set = sample_unique(opening_pool, args.target_opening)
    midgame_set = sample_unique(midgame_pool, args.target_mid)
    endgame_set = sample_unique(endgame_pool, args.target_end)

    print(f"  Selected: opening={len(opening_set)}, mid={len(midgame_set)}, end={len(endgame_set)}")

    if (len(opening_set) < args.target_opening
            or len(midgame_set) < args.target_mid
            or len(endgame_set) < args.target_end):
        print(f"WARNING: did not reach target counts — increase --max-attempts",
              file=sys.stderr)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total = len(opening_set) + len(midgame_set) + len(endgame_set)
    lines = [f'# {total} test positions for Exp 4 — stratified by game phase',
             f'# Seed: {args.seed}',
             f'# Phase bands: opening > 0.8, midgame 0.3..0.8, endgame < 0.3',
             f'# Counts: opening={len(opening_set)}, mid={len(midgame_set)}, end={len(endgame_set)}',
             '']
    for fen, tag in opening_set + midgame_set + endgame_set:
        lines.append(f'{fen} # {tag}')
    output_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"\nSaved: {output_file} ({total} positions)")


if __name__ == '__main__':
    main()
