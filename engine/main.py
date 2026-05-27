import argparse
import logging

from constants import LOGGER, FORMATTER
from engine import Engine
from mode import Mode
from player_type import PlayerType


def depth(arg: str) -> int:
    try:
        value = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError('Depth must be an integer')
    if not (1 <= value <= 20):
        raise argparse.ArgumentTypeError('Depth must be an integer within a range [1, 20]')
    return value


def skill_level(arg: str) -> int:
    try:
        value = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError('Skill level must be an integer')
    if not (0 <= value <= 20):
        raise argparse.ArgumentTypeError('Skill level must be an integer within a range [0, 20]')
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Chess engine')
    parser.add_argument('-w', '--white', help='White player', type=str, choices=[p.value for p in PlayerType],
                        required=False)
    parser.add_argument('-b', '--black', help='Black player', type=str, choices=[p.value for p in PlayerType],
                        required=False)
    parser.add_argument('-m', '--mode', help='Graphical | background | settings', type=str,
                        choices=[m.value for m in Mode], required=False)
    parser.add_argument('-e', '--empty', help='Initial board state in settings mode is empty', action='store_true',
                        required=False)
    parser.add_argument('-i', '--input', help='Initial board state fen input file', type=str, required=False)
    parser.add_argument('-o', '--output', help='Final board state fen output file', type=str, required=False)
    parser.add_argument('-ob', '--opening_book', help='Use opening book', action='store_true')
    parser.add_argument('-obs', '--opening_book_strict',
                        help='Always pick the max-weight opening book entry (deterministic)',
                        action='store_true')
    parser.add_argument('-g', '--game', help='Game moves output txt file', type=str, required=False)
    parser.add_argument('-l', '--logs', help='Game logs file', type=str, required=False)
    parser.add_argument('-dw', '--depth_white', help='Depth for white', type=depth, required=False)
    parser.add_argument('-dws', '--depth_white_stockfish', help='Stockfish depth for white', type=depth, required=False)
    parser.add_argument('-sw', '--skill_white', help='Stockfish skill level for white', type=skill_level,
                        required=False)
    parser.add_argument('-db', '--depth_black', help='Depth for black', type=depth, required=False)
    parser.add_argument('-dbs', '--depth_black_stockfish', help='Stockfish depth for black', type=depth, required=False)
    parser.add_argument('-sb', '--skill_black', help='Stockfish skill level for black', type=skill_level,
                        required=False)
    parser.add_argument('-sp', '--stockfish_path', help='Path to Stockfish binary', type=str, required=False)
    parser.add_argument('-mtw', '--mcts_time_white', help='MCTS time budget in seconds for white', type=float,
                        default=20.0, required=False)
    parser.add_argument('-mtb', '--mcts_time_black', help='MCTS time budget in seconds for black', type=float,
                        default=20.0, required=False)
    parser.add_argument('-d', '--debug', help='Debug mode', action='store_true', required=False)
    parser.add_argument('-jl', '--json_log', help='JSON metrics log file', type=str, required=False)
    parser.add_argument('-adj', '--adjudicate', help='Enable eval-based draw adjudication', action='store_true',
                        required=False)
    parser.add_argument('-adjt', '--adjudicate_threshold', help='Eval threshold for adjudication (default 0.05)',
                        type=float, default=0.05, required=False)
    parser.add_argument('-adjm', '--adjudicate_moves', help='Consecutive full moves for adjudication (default 20)',
                        type=int, default=20, required=False)
    return parser.parse_args()


def add_logs_file(args: argparse.Namespace) -> None:
    if args.logs:
        file_handler = logging.FileHandler(f'out/{args.logs}', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FORMATTER)
        LOGGER.addHandler(file_handler)


def main() -> None:
    args = parse_args()
    add_logs_file(args)
    Engine(args)


if __name__ == '__main__':
    main()
