import chess

openings = [
    ("Italian Game",           "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6"),
    ("Ruy Lopez",              "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6"),
    ("Scotch Game",            "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8c5"),
    ("Four Knights",           "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 f1b5 f8b4"),
    ("Petrov Defense",         "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4"),
    ("Vienna Game",            "e2e4 e7e5 b1c3 g8f6 f1c4 f8c5 d2d3 d7d6"),
    ("Sicilian Najdorf",       "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6"),
    ("Sicilian Dragon",        "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g7g6"),
    ("Sicilian Scheveningen",  "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 e7e6"),
    ("Sicilian Alapin",        "e2e4 c7c5 c2c3 d7d5 e4d5 d8d5 d2d4 g8f6"),
    ("French Defense",         "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 e4e5 f6d7"),
    ("Caro-Kann",              "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5"),
    ("Pirc Defense",           "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f1c4 f8g7"),
    ("Scandinavian",           "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6"),
    ("QGD Classical",          "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7"),
    ("QGD Tartakower",         "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 g1f3 b7b6"),
    ("Slav Defense",           "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4"),
    ("Queens Gambit Accepted", "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6"),
    ("Kings Indian Classical", "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6"),
    ("Kings Indian Samisch",   "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 f2f3 d7d6"),
    ("Grunfeld Exchange",      "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5"),
    ("Nimzo-Indian",           "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8"),
    ("English Symmetrical",    "c2c4 c7c5 b1c3 b8c6 g1f3 g8f6 d2d4 c5d4"),
    ("Catalan",                "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7"),
    ("Reti Opening",           "g1f3 d7d5 c2c4 c7c6 b2b3 g8f6 c1b2 c8f5"),
]

for name, moves in openings:
    b = chess.Board()
    for m in moves.split():
        b.push_uci(m)
    print(f"{b.fen()} # {name}")
