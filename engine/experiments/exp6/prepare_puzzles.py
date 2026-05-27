"""
Convert EPD (Extended Position Description) puzzles to a standardized JSON format.

Supported sources:
  - WAC (Win At Chess): 300 tactical puzzles. Format: `<FEN> bm <move>; id "WAC.NNN";`
  - STS (Strategic Test Suite): 1500 positions, 15 themes. Format:
    `<FEN> bm <move>; c0 "STS(v1.0) <theme>"; id "STS(v1.0) NNN.NNN";`
  - Bratko-Kopec: similar EPD format

Standardized output (puzzles.json):
[
  {
    "id": "WAC.001",
    "fen": "...",
    "best_moves_san": ["Re8+"],     # SAN as in source
    "best_moves_uci": ["d1e8"],     # converted to UCI for matching
    "side_to_move": "w",
    "theme": "WAC",                  # from STS c0 or fallback to set name
    "difficulty": null,              # if available
    "source_id": "WAC.001"
  }
]

Usage:
    python prepare_puzzles.py --input puzzles/sample_wac.epd
    python prepare_puzzles.py --input puzzles/wac.epd --set WAC
    python prepare_puzzles.py --input puzzles/STS1-STS15.epd --set STS

Where to get full sets (publicly available):
    WAC: https://www.chessprogramming.org/Test-Positions  (search "WAC.epd")
    STS: https://www.chessprogramming.org/Strategic_Test_Suite
"""

import argparse
import json
import re
import sys
from pathlib import Path

import chess


# EPD operation parser: matches `op_name value;` pairs
OP_RE = re.compile(r'(\w+)\s+("[^"]*"|[^;]+);')


def parse_epd_line(line: str) -> dict | None:
    """Parses one EPD line into a dict with fen + operations."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    # Find where the FEN ends — EPD FEN has 4 fields (no halfmove/fullmove counts).
    # The first 4 whitespace-separated tokens are the FEN.
    tokens = line.split()
    if len(tokens) < 5:
        return None
    fen_part = ' '.join(tokens[:4])
    ops_part = ' '.join(tokens[4:])

    # Full FEN needs halfmove + fullmove counts — default to 0 1
    fen = f'{fen_part} 0 1'

    ops = {}
    for m in OP_RE.finditer(ops_part):
        name = m.group(1)
        value = m.group(2).strip().strip('"')
        ops[name] = value

    return {'fen': fen, 'ops': ops}


def san_to_uci(fen: str, san_moves: list[str]) -> list[str]:
    board = chess.Board(fen)
    uci_moves = []
    for san in san_moves:
        try:
            move = board.parse_san(san)
            uci_moves.append(move.uci())
        except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            uci_moves.append(None)
    return uci_moves


def extract_theme(ops: dict, source_set: str) -> str:
    """Extracts theme info from EPD ops, falling back to source set name."""
    c0 = ops.get('c0', '')
    # STS format: c0 "STS(v1.0) <theme>"
    sts_m = re.match(r'STS\(v[\d.]+\)\s+(.+)', c0)
    if sts_m:
        return sts_m.group(1).strip()
    # Bratko-Kopec sometimes uses different conventions
    if c0:
        return c0
    return source_set


def parse_best_moves(bm_field: str) -> list[str]:
    """Returns list of SAN moves. EPD allows multiple acceptable moves."""
    if not bm_field:
        return []
    # Split on whitespace; some sets may use commas
    bm_field = bm_field.replace(',', ' ')
    return [m.strip() for m in bm_field.split() if m.strip()]


def main():
    parser = argparse.ArgumentParser(description='Convert EPD puzzles to standardized JSON')
    parser.add_argument('--input', type=str, required=True, help='Path to EPD file')
    parser.add_argument('--output', type=str, default='',
                        help='Default: <input_dir>/puzzles.json')
    parser.add_argument('--set', type=str, default='', help='Source set name (WAC, STS, etc.)')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else \
        Path(__file__).resolve().parent / 'puzzles.json'

    # Auto-detect source set from filename if not provided
    source_set = args.set or in_path.stem.upper()

    puzzles = []
    skipped = 0

    with open(in_path, 'r', encoding='utf-8-sig') as f:
        for line_idx, raw in enumerate(f, 1):
            if args.limit and len(puzzles) >= args.limit:
                break
            parsed = parse_epd_line(raw)
            if parsed is None:
                continue

            ops = parsed['ops']
            bm_san = parse_best_moves(ops.get('bm', ''))
            if not bm_san:
                skipped += 1
                continue

            try:
                bm_uci = san_to_uci(parsed['fen'], bm_san)
            except Exception as e:
                print(f"  line {line_idx}: SAN conversion failed: {e}", file=sys.stderr)
                skipped += 1
                continue

            # Reject puzzles where no SAN converted successfully
            valid_uci = [u for u in bm_uci if u]
            if not valid_uci:
                skipped += 1
                continue

            side_to_move = parsed['fen'].split()[1]
            puzzle_id = ops.get('id', f'{source_set}.{line_idx:04d}')

            puzzles.append({
                'id': puzzle_id,
                'fen': parsed['fen'],
                'best_moves_san': bm_san,
                'best_moves_uci': valid_uci,
                'side_to_move': side_to_move,
                'theme': extract_theme(ops, source_set),
                'difficulty': ops.get('dm') or ops.get('difficulty') or None,
                'source_id': puzzle_id,
            })

    out_path.write_text(json.dumps(puzzles, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f"Parsed {len(puzzles)} puzzles from {in_path.name}")
    if skipped:
        print(f"Skipped {skipped} lines (no bm or conversion error)")
    print(f"Saved: {out_path}")

    # Theme distribution
    theme_counts = {}
    for p in puzzles:
        theme_counts[p['theme']] = theme_counts.get(p['theme'], 0) + 1
    if theme_counts:
        print("\nTheme distribution:")
        for theme, count in sorted(theme_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {theme:50s} {count}")


if __name__ == '__main__':
    main()
