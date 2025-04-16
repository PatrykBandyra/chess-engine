import argparse

from engine import Engine
from mode import Mode
from player_type import PlayerType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Chess engine')
    parser.add_argument('-w', '--white', help='White player', type=str, choices=[p.value for p in PlayerType])
    parser.add_argument('-b', '--black', help='Black player', type=str, choices=[p.value for p in PlayerType])
    parser.add_argument('-m', '--mode', help='Graphical or background mode', type=str, choices=[m.value for m in Mode])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Engine(args)


if __name__ == '__main__':
    main()
