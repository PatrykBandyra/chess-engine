import chess
import random
import time
from stockfish import Stockfish

# Path to your Stockfish binary (adjust if needed)
# STOCKFISH_PATH = r"../../stockfish_ai/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
STOCKFISH_PATH = r"../../stockfish_ai/stockfish/stockfish-ubuntu-x86-64-avx2"

def random_fen():
    board = chess.Board()
    # Play a random number of random legal moves from the starting position
    num_moves = random.randint(0, 60)
    for _ in range(num_moves):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    return board.fen()

def evaluate_fens_with_stockfish(fens):
    # stockfish = Stockfish(STOCKFISH_PATH, parameters={'Threads': 10})
    stockfish = Stockfish(STOCKFISH_PATH, parameters={'Threads': 20, 'Hash': 36000})
    total_time = 0.0
    for i, fen in enumerate(fens):
        start = time.perf_counter()
        stockfish.set_fen_position(fen)
        score = stockfish.get_evaluation()
        elapsed = time.perf_counter() - start
        total_time += elapsed
        # print(f"{i+1}: FEN: {fen}\n   Score: {score}   Time: {elapsed:.4f}s")
    avg_time = total_time / len(fens)
    print(f"\nAverage Stockfish evaluation time: {avg_time:.4f}s over {len(fens)} positions.")

if __name__ == "__main__":
    fens = [random_fen() for _ in range(1000)]
    print("\n--- Stockfish Evaluation ---\n")
    evaluate_fens_with_stockfish(fens)
