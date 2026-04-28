## Plan: Opis implementacji silnika szachowego + ulepszenia MCTS

Repozytorium to silnik szachowy w Pythonie z architekturą Strategy Pattern: interfejs `Player` z implementacjami Minimax (alfa-beta + TT + killer/history heuristics) i MCTS (UCT), dwoma strategiami ewaluacji (tradycyjna PST + heurystyki, oraz Stockfish jako "NN"), GUI w Pygame, i książką otwarć Polyglot. MCTS ma kilka istotnych słabości, które można zaadresować.

### Kroki – Opis implementacji (README / dokumentacja)

1. **Opisać architekturę ogólną** – wzorzec Strategy w [player.py](engine/player.py) z 6 typami graczy (`PlayerType`), tryby uruchomienia (`Mode`: GUI/background/settings), threading GUI w [engine.py](engine/engine.py).
2. **Opisać Minimax** – alfa-beta pruning z transposition table (Zobrist hashing), killer moves, history heuristic, MVV-LVA move ordering w [minimax.py](engine/minimax.py) + [order_moves_minimax.py](engine/order_moves_minimax.py).
3. **Opisać MCTS** – UCT (Upper Confidence Bound for Trees), fazy select→expand→simulate→backpropagate, heurystyczne sortowanie ruchów przy ekspansji w [mcts.py](engine/mcts.py).
4. **Opisać ewaluację tradycyjną** – materiał, Piece-Square Tables (mid/endgame z interpolacją fazy), struktura pionów, bezpieczeństwo króla, mobilność w [board_evaluator_trad.py](engine/board_evaluator_trad.py).
5. **Opisać sieć CNN** – `ChessEvaluationCNN` z Conditional BatchNorm w [network.py](lab/nn/network.py) – **uwaga: model nie jest zintegrowany z silnikiem**; klasa `BoardEvaluatorNN` w rzeczywistości odpytuje Stockfisha.

### Kroki – Ulepszenia MCTS

1. **Normalizacja wartości ewaluacji do [0, 1]** – obecnie `evaluate_board` zwraca nieograniczone wartości centipawnowe; UCB1 w `best_child()` wymaga wartości w stałym zakresie. Dodać sigmoid/tanh normalizację w `__simulate` w [mcts.py](engine/mcts.py), linia ~108.
2. **Zastąpić losowe rollouts ewaluacją statyczną** – `__simulate` używa `random.choice()` co daje bardzo szumne wyniki. Zamiast symulacji do `self.depth` losowych ruchów, zastosować bezpośrednią ewaluację pozycji po ekspansji (podejście "MCTS bez rolloutów" / "MCTS with evaluation cutoff"), eliminując potrzebę parametru `depth` jako limitu rolloutów.
3. **Dodać transposition table do MCTS** – w przeciwieństwie do Minimax, MCTS nie wykrywa transpozycji. Użyć słownika `{zobrist_hash → MCTSNode}` do współdzielenia węzłów między poddrzewami, co znacznie zmniejszy redundancję.
4. **Reuse drzewa między ruchami** – po wybraniu najlepszego ruchu w `__run_mcts`, zachować poddrzewo korzenia dla kolejnego wywołania, zamiast budować od zera (oszczędność ~50% iteracji).
5. **Konfigurowalna `SIMULATION_TIME`** – hardcoded `20.0s` w [mcts.py](engine/mcts.py) linia 70 – przenieść do argumentów CLI lub uzależnić od fazy gry.
6. **Naprawić `MCTSNN`** – klasa w [mcts_nn.py](engine/mcts_nn.py) dziedziczy po `Player` zamiast `MCTS`, więc **nie implementuje algorytmu MCTS** – brakuje `make_move`. Zmienić na dziedziczenie po `MCTS`.

### Dalsze rozważania

1. **Integracja prawdziwej sieci CNN** – model `ChessEvaluationCNN` jest wytrenowany w `lab/nn/`, ale nie używany. Czy zintegrować go jako `BoardEvaluatorNN` zamiast Stockfisha? To wymaga konwersji FEN→tensor i ładowania `.pth` w silniku.
2. **Quiescence Search w Minimax** – brak quiescence search oznacza efekt horyzontu; dodanie przeszukiwania bić/szachów na głębokości 0 znacząco poprawiłoby jakość Minimaxa.
3. **RAVE (Rapid Action Value Estimation)** – technika AMAF mogłaby przyspieszyć konwergencję MCTS w otwartych pozycjach, ale jest trudniejsza w implementacji i mniej popularna w nowoczesnych silnikach szachowych niż np. policy network guidance.

