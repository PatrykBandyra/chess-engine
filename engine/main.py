import argparse

from engine import Engine
from mode import Mode
from player_type import PlayerType


def depth(arg: str) -> int:
    try:
        value = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError('Depth must be an integer')
    if not (0 <= value <= 20):
        raise argparse.ArgumentTypeError('Depth must be an integer within a range [1, 50]')
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
                        required=True)
    parser.add_argument('-b', '--black', help='Black player', type=str, choices=[p.value for p in PlayerType],
                        required=True)
    parser.add_argument('-m', '--mode', help='Graphical or background mode', type=str, choices=[m.value for m in Mode],
                        required=False)
    parser.add_argument('-dw', '--depth_white', help='Stockfish depth for white', type=depth, required=False)
    parser.add_argument('-sw', '--skill_white', help='Stockfish skill level for white', type=skill_level,
                        required=False)
    parser.add_argument('-db', '--depth_black', help='Stockfish depth for black', type=depth, required=False)
    parser.add_argument('-sb', '--skill_black', help='Stockfish skill level for black', type=skill_level,
                        required=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Engine(args)


if __name__ == '__main__':
    main()
