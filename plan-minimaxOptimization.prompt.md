## Plan: Optymalizacja algorytmu Minimax z przycinaniem alfa-beta

Aktualna implementacja Minimax w [minimax.py](engine/minimax.py) stosuje przeszukiwanie alfa-beta z tablicą transpozycji (TT), killer moves i heurystyką historii. Poniższy plan adresuje 1 krytyczny bug w logice TT, 1 poprawkę logowania, 1 optymalizację wydajności searcha, 1 bug w ewaluatorze, 2 optymalizacje wydajności move orderingu, Iterative Deepening, oraz 5 nowych usprawnień algorytmu przeszukiwania (Quiescence Search, Check Extensions, Null Move Pruning, ograniczenie TT, Aspiration Windows).

### Kolejność wdrożenia: 1 → 2 → 3 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13

### Kroki

1. ✅ **Naprawić flagę TT dla minimizera — użyć `original_beta` zamiast `beta` (KRYTYCZNY BUG)**

   **Cel:** W [minimax.py](engine/minimax.py) (linie 200–205) logika ustalania flagi TT w gałęzi minimizera porównuje `min_eval` z wartością `beta`, która została zmodyfikowana w pętli przeszukiwania (`beta = min(beta, evaluation)` na linii 186). Powinna porównywać z `original_beta` (zapisanym na linii 115, ale nieużywanym w flagach TT minimizera). Analogicznie gałąź maksymizera (linie 164–168) poprawnie używa `original_alpha` — minimizer powinien być symetryczny.

   **Analiza techniczna:**

   - **Obecny stan i skutki buga:**
     - Po pętli minimizera `beta = min(original_beta, eval1, eval2, ...) = min(original_beta, min_eval)`. Gdy `min_eval < original_beta` (normalny przypadek bez fail-high), wówczas `beta = min_eval`, więc warunek `min_eval >= beta` jest **zawsze prawdziwy**. Flaga jest ustawiana na `'L'` (Lower bound) zamiast `'E'` (Exact).
     - Gdy następuje cutoff (`beta <= alpha`), `beta = min_eval` i `min_eval >= beta` nadal jest prawdziwe → flaga `'L'` zamiast `'U'` (Upper bound). Uszkodzone wpisy TT powodują, że przyszłe wyszukiwania traktują górne ograniczenia jako dolne, co może prowadzić do błędnych odcięć i suboptymalnej gry.

   - **Dowód na przykładzie (normalne przeszukanie):**
     - Węzeł A (min), alpha=5, original_beta=10. Brak modyfikacji TT.
     - Ruch 1: eval=7, min_eval=7, beta=min(10,7)=7. 7≤5? Nie.
     - Ruch 2: eval=8, min_eval=7, beta=7. 7≤5? Nie.
     - Przeszukano wszystkie ruchy. min_eval=7. Prawdziwa wartość JEST 7 (exact).
     - Obecny kod: `min_eval(7) >= beta(7)` → True → flag='L' (**BŁĄD**, powinno być 'E').
     - Po naprawie: `min_eval(7) >= original_beta(10)` → False. `min_eval(7) <= alpha(5)` → False → flag='E' ✓.

   - **Dowód na przykładzie (cutoff):**
     - Węzeł B (min), alpha=5, original_beta=10.
     - Ruch 1: eval=7, min_eval=7, beta=7. 7≤5? Nie.
     - Ruch 2: eval=4, min_eval=4, beta=4. 4≤5? TAK, CUTOFF.
     - min_eval=4, beta=4, alpha=5.
     - Obecny kod: `min_eval(4) >= beta(4)` → True → flag='L' (**BŁĄD**, powinno być 'U' — nie przeszukano wszystkich ruchów, prawdziwa wartość może być niższa).
     - Po naprawie: `min_eval(4) >= original_beta(10)` → False. `min_eval(4) <= alpha(5)` → True → flag='U' ✓.

   - **Zmiana — 1 linia w `minimax.py` (linia 202):**
     - Linia 202: `if min_eval >= beta:` → `if min_eval >= original_beta:`.
     - Linia 204 (`elif min_eval <= alpha:`) — bez zmian. `alpha` jest modyfikowane przez TT lookup, ale nie przez pętlę minimizera — poprawne zachowanie.

   - **Weryfikacja symetrii z maksymizerem:**
     - Maksymizer (linie 164–168): `max_evaluation <= original_alpha` → 'U'; `max_evaluation >= beta` → 'L'. Tutaj `beta` nie jest modyfikowane przez pętlę maksymalizera (modyfikowane jest `alpha`), więc użycie `beta` jest poprawne.
     - Minimizer (po naprawie): `min_eval >= original_beta` → 'L'; `min_eval <= alpha` → 'U'. Tutaj `alpha` nie jest modyfikowane przez pętlę minimizera (modyfikowane jest `beta`), więc użycie `alpha` jest poprawne.
     - Obie gałęzie są teraz symetryczne: każda porównuje wynik z **oryginalną** wartością granicy, którą sama modyfikuje, oraz z **bieżącą** wartością granicy przeciwnej (modyfikowaną tylko przez TT lookup).

   - **⚠ Wpływ na heurystykę historii (linia 194):**
     Warunek `if beta > alpha` (linia 194) w gałęzi minimizera używa zmodyfikowanego `beta`. Po cutoff-ie `beta <= alpha`, więc warunek jest fałszywy — poprawnie blokuje aktualizację historii po cutoff-ie (cutoff jest obsługiwany w pętli na linii 190). Bez cutoff-u `beta = min_eval` i `alpha` jest z TT — warunek sprawdza, czy min_eval jest nadal powyżej alpha, co jest sensowne. **Ta linia nie wymaga zmiany.**

   - **Skala wpływu:** Bug dotyczy **każdego** węzła minimizera w **każdej** grze. Przy typowej głębokości 4–6 i ~30 ruchach na pozycję, połowa z ~setek tysięcy węzłów TT ma błędne flagi. Naprawa powinna zauważalnie poprawić siłę gry, szczególnie na wyższych głębokościach, gdzie TT jest intensywniej wykorzystywana.

   - **Kolejność wdrożenia:**
     Krok 1 jest **niezależny** od kroków 2–4 i powinien być wdrożony **jako pierwszy** — jest to naprawa krytycznego buga, nie optymalizacja.

2. ✅ **Poprawić logowanie — dynamiczna nazwa klasy**

   **Cel:** Linie 90 i 94–96 w [minimax.py](engine/minimax.py) zawierają zahardcodowaną nazwę `'MINIMAX-TRAD'`, niezależnie od tego, czy gra `MinimaxTrad` (ewaluator tradycyjny z [minimax_trad.py](engine/minimax_trad.py)) czy `MinimaxNN` (ewaluator Stockfish z [minimax_nn.py](engine/minimax_nn.py)). Log jest mylący, gdy gra `MinimaxNN`.

   **Analiza techniczna:**

   - **Obecny stan:** Metoda `make_move` jest zdefiniowana w klasie bazowej `Minimax` i dziedziczona przez obie podklasy. Obie podklasy korzystają z tego samego kodu logowania, ale log zawsze wypisuje `'MINIMAX-TRAD'`.

   - **Zmiana — 2 miejsca w `minimax.py`:**
     1. Linia 90: `'MINIMAX-TRAD: No valid move found...'` → `f'{type(self).__name__}: No valid move found...'`.
     2. Linia 95: `'MINIMAX-TRAD; ...'` → `f'{type(self).__name__}; ...'`.
     - Wynik: `MinimaxTrad` lub `MinimaxNN` w logach — jednoznacznie identyfikuje wariant.

   - **Spójność z planem MCTS:** Analogiczna zmiana została wdrożona w kroku 10 planu MCTS (`plan-mctsOptimization.prompt.md`) dla klasy `MCTS`. Obie zmiany stosują ten sam wzorzec `type(self).__name__`.

   - **Kolejność wdrożenia:** Niezależna od pozostałych kroków. Może być wdrożona w dowolnym momencie.

3. ✅ **Dodać odcięcie alfa-beta na poziomie korzenia (root-level pruning)**

   **Cel:** W metodzie `make_move` (linie 71–85 w [minimax.py](engine/minimax.py)) pętla po ruchach aktualizuje `alpha`/`beta`, ale nigdy nie przerywa przeszukiwania, gdy `beta <= alpha`. Choć na poziomie korzenia beta startuje od `+inf` (więc cutoff niemal nigdy nie zachodzi), TO zachodzi, gdy znaleziono forsownego mata (wartość = `±math.inf`).

   **Analiza techniczna:**

   - **Scenariusz, w którym pruning zachodzi:**
     - Biały (maximizing): Ruch 1 zwraca `+inf` (mat). alpha = max(-inf, +inf) = +inf. beta(+inf) <= alpha(+inf) → True → break.
     - Czarny (minimizing): Ruch 1 zwraca `-inf` (mat). beta = min(+inf, -inf) = -inf. beta(-inf) <= alpha(-inf) → True → break.
     - Bez break: silnik przeszukuje **wszystkie** pozostałe ruchy mimo znalezienia mata w 1.

   - **Zmiana — dodać `if beta <= alpha: break` w obu gałęziach pętli (po liniach 80 i 85):**
     - Po linii 80 (`alpha = max(alpha, board_value)`): dodać `if beta <= alpha: break`.
     - Po linii 85 (`beta = min(beta, board_value)`): dodać `if beta <= alpha: break`.

   - **Wpływ na poprawność:** Żaden. Root-level cutoff jest poprawny z punktu widzenia algorytmu alfa-beta — jeśli znaleziono wartość = +inf (mat), żaden inny ruch nie da lepszego wyniku. Przerywamy przeszukiwanie identycznie jak w rekurencyjnym `__minimax_alphabeta` (linie 150, 187).

   - **Szacowany zysk:** Pomijalny w otwarciu/środkowej grze (beta = +inf, cutoff nie zachodzi). Potencjalnie znaczący w końcówkach z forsownym matem — oszczędza przeszukiwanie `n-1` ruchów, gdzie `n` to liczba legalnych ruchów (zwykle ~10–30 w końcówce). Przy głębokości 5 i ~20 ruchach: oszczędność ~19 × pełne poddrzewo głębokości 4.

   - **Kolejność wdrożenia:** Niezależna od kroków 1–2. Rekomendowana po kroku 1 (naprawa TT), bo poprawna TT zwiększa efektywność przeszukiwania.

4. ~~**Unikać redundantnego wywołania `board.is_game_over()` w `__minimax_alphabeta`**~~ (REVERT — optymalizacja nieistotna, zachowano spójność z `evaluate_board`)

   **Cel:** Linia 131 w [minimax.py](engine/minimax.py) wywołuje `board.is_game_over()` na **każdym** węźle wewnętrznym (nie-liściu). Metoda ta wewnętrznie generuje listę legalnych ruchów (O(n)), a następnie linia 134 generuje ją ponownie (`list(board.legal_moves)`). To podwójna praca na każdym węźle.

   **Analiza techniczna:**

   - **Co robi `board.is_game_over()`:** Wewnętrznie sprawdza: (a) brak legalnych ruchów (mat/pat), (b) regułę 75 posunięć, (c) pięciokrotne powtórzenie, (d) niewystarczający materiał. Punkt (a) wymaga generacji legalnych ruchów — to samo co `list(board.legal_moves)`.

   - **Proponowana zmiana — reorganizacja linii 131–135:**
     1. Warunek `depth == 0` nadal zwraca `self.evaluate_board(board)` — bez zmian.
     2. Przenieść generację ruchów **przed** sprawdzenie terminalne.
     3. Zamiast `board.is_game_over()`, użyć sprawdzenia opartego na wygenerowanych ruchach:
        - `if not legal_moves:` — wykrywa mat i pat.
        - Dla pata (`not board.is_check()`): zwrócić `0.0` (remis).
        - Dla mata (`board.is_check()`): zwrócić `-math.inf` jeśli `maximizing_player`, `+math.inf` w przeciwnym razie (przegrywający gracz dostaje najgorszą ocenę).
     4. Opcjonalnie zachować sprawdzenie `board.is_insufficient_material()` lub `board.can_claim_draw()` dla dodatkowych remisów (rzadkie, ale tanie).

   - **Uwaga o kolejności z TT:** Obecny kod wykonuje TT lookup **przed** sprawdzeniem terminalnym — to pozostaje bez zmian. TT może zawierać wpisy dla pozycji terminalnych z głębszych przeszukiwań, co jest poprawne.

   - **Nowa struktura (po TT lookup):**
     ```python
     if depth == 0:
         return self.evaluate_board(board)
     
     legal_moves = list(board.legal_moves)
     if not legal_moves:
         if board.is_check():
             return -math.inf if maximizing_player else math.inf  # mat
         return 0.0  # pat
     if board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
         return 0.0  # remis
     ```

   - **Wpływ na ewaluację mata/pata:** Obecny `evaluate_board` w [board_evaluator_trad.py](engine/board_evaluator_trad.py) obsługuje mat/pat wewnętrznie (sprawdza `board.is_checkmate()` itd. → kolejna redundancja). Po zmianie mat/pat jest wykrywany **zanim** dotrze do `evaluate_board`, więc ewaluator jest wywoływany tylko dla pozycji nieterminalnych na `depth == 0`.

   - **Szacowany zysk:** Eliminacja jednego wywołania `board.is_game_over()` na każdym węźle wewnętrznym. Przy głębokości 5 i ~30 ruchach: ~setki tysięcy węzłów → mierzalna oszczędność czasu.

   - **⚠ Wpływ na krok 1 (TT):** Żaden — reorganizacja sprawdzenia terminalnego nie wpływa na logikę TT (TT lookup następuje przed sprawdzeniem terminalnym i to się nie zmienia).

   - **Kolejność wdrożenia:** Niezależna od kroków 1–3, ale rekomendowana po nich (mniejsze ryzyko regresji po naprawieniu TT).

5. ✅ **Naprawić `__evaluate_mobility_and_activity` — mobilność liczona tylko dla jednej strony (BUG + OPTYMALIZACJA)**

   **Cel:** W [board_evaluator_trad.py](engine/board_evaluator_trad.py) (linie 336–344) metoda `__evaluate_mobility_and_activity` iteruje po figurach obu kolorów, ale `board.legal_moves` generuje ruchy **wyłącznie dla strony do ruchu** (`board.turn`). Dla figur koloru `!= board.turn` mobilność wynosi zawsze 0. Ewaluacja mobilności powinna uwzględniać **obie strony**, aby wynik wskazywał przewagę mobilności na szachownicy.

   **Analiza techniczna:**

   - **Obecny stan i skutki buga:**
     - Pętla `for color in [chess.WHITE, chess.BLACK]` iteruje po obu stronach, ale wewnętrzna pętla `for move in board.legal_moves` zwraca tylko ruchy strony `board.turn`.
     - Gdy `board.turn == chess.WHITE`: mobilność białych jest policzona poprawnie, mobilność czarnych = 0.
     - Gdy `board.turn == chess.BLACK`: mobilność czarnych jest policzona poprawnie, mobilność białych = 0.
     - Efekt: ewaluacja mobilności jest **jednostronna** — brakuje składnika jednej ze stron, więc wynik nie odzwierciedla rzeczywistej przewagi mobilności na planszy.

   - **Proponowana zmiana — `board.attacks(sq)` zamiast iteracji po `board.legal_moves`:**
     - `board.attacks(sq)` zwraca zbiór pól atakowanych przez figurę na `sq` (pseudo-legal, uwzględnia blokery dla figur ślizgowych). Działa dla **obu stron** niezależnie od `board.turn`.
     - Jest to standardowe przybliżenie mobilności w silnikach szachowych — szybsze niż legal moves i symetryczne.
     - Dodatkowa optymalizacja: eliminacja iteracji O(figury × ruchy) → bezpośredni dostęp O(1) per figurę (`board.attacks(sq)` korzysta z wewnętrznych tablic ataków).

   - **Nowa struktura:**
     ```python
     for color in [chess.WHITE, chess.BLACK]:
         color_sign = 1 if color == chess.WHITE else -1
         for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
             for sq in board.pieces(piece_type, color):
                 mobility = len(board.attacks(sq))
                 score += color_sign * mobility_weights[piece_type] * mobility
                 # Activity: ...
     ```

   - **Wpływ:** Naprawa buga (obie strony ewaluowane) + znacząca poprawa wydajności (eliminacja wielokrotnej iteracji po `board.legal_moves`).

   - **Kolejność wdrożenia:** Niezależna od kroków 1–3. Dotyczy wyłącznie `BoardEvaluatorTrad` — nie wpływa na `BoardEvaluatorNN` (Stockfish).

6. ✅ **Użyć `board.gives_check(move)` zamiast `push/pop + is_check` w move orderingu**

   **Cel:** W [order_moves_minimax.py](engine/order_moves_minimax.py) (linie 80–83) sprawdzenie, czy cichy ruch daje szacha, wykonywane jest przez kosztowne `board.push(move)` + `board.is_check()` + `board.pop()`. Biblioteka `python-chess` udostępnia `board.gives_check(move)`, które robi to samo **bez modyfikacji planszy**.

   **Analiza techniczna:**

   - **Obecny kod:**
     ```python
     board.push(move)
     is_check: bool = board.is_check()
     board.pop()
     ```
     Wykonywane dla **każdego cichego ruchu** w **każdym wywołaniu `order_moves`**. `push`/`pop` to operacje O(1), ale z dużą stałą (aktualizacja tablic bitowych, hash, stanu roszady itp.).

   - **Proponowana zmiana — 3 linie → 1 linia:**
     ```python
     is_check: bool = board.gives_check(move)
     ```
     `gives_check` analizuje ruch bez modyfikacji planszy — znacznie szybsze.

   - **Szacowany zysk:** Przy ~20 cichych ruchach na węzeł i setkach tysięcy węzłów, eliminujemy miliony operacji `push`/`pop`. To jedna z najdroższych operacji w move orderingu.

   - **Kolejność wdrożenia:** Niezależna od pozostałych kroków. Czysto mechaniczna zmiana.

7. ✅ **Sprawdzić szach także dla bić i promocji w move orderingu**

   **Cel:** W [order_moves_minimax.py](engine/order_moves_minimax.py) bonus za szacha (`move_ordering_check_bonus`) jest przyznawany **wyłącznie cichym ruchom** (linie 80–89, wewnątrz bloku `else`). Bicia i promocje, które jednocześnie dają szacha, nie otrzymują dodatkowego bonusu za szach.

   **Analiza techniczna:**

   - **Obecny stan:** Struktura `if promotion / elif capture / else quiet` sprawdza szach tylko w gałęzi `else` (cichy ruch). Bicie dające szacha ma ten sam priorytet co bicie bez szacha. Promocja dająca szacha ma ten sam priorytet co promocja bez szacha.

   - **Proponowana zmiana:**
     - Przenieść sprawdzenie `board.gives_check(move)` (z kroku 6) **przed** blok `if/elif/else`, tak aby bonus za szach był dodawany niezależnie od typu ruchu.
     - Alternatywnie: dodać `board.gives_check(move)` wewnątrz gałęzi `if is_promotion` i `elif is_capture`.

   - **Wpływ na siłę gry:** Bicia i promocje dające szacha są szczególnie niebezpieczne taktycznie. Priorytetyzowanie ich przed innymi biciami/promocjami poprawia kolejność ruchów, co zwiększa liczbę odcięć alfa-beta.

   - **Kolejność wdrożenia:** Po kroku 6 (zamiana na `gives_check`). Logicznie powiązane — oba dotyczą detekcji szacha w move orderingu.

8. ✅ **Wdrożyć Iterative Deepening (ID)**

   **Cel:** Wielokrotne przeszukiwanie drzewa z rosnącą głębokością (1, 2, …, max_depth), zachowując TT między iteracjami. Wpisy TT z płytszych iteracji seedują głębsze — poprawiając kolejność ruchów (PV move jako pierwszy) i efektywność pruning.

   **Analiza techniczna — podkroki:**

   - **8a. Dodać `best_move` do wpisów TT**
     - **Plik:** `minimax.py`, linie 175–176 i 212–213.
     - Przy każdym zapisie do TT dodać pole `'m': best_move` — ruch który dał najlepszą ewaluację.
     - `best_move` jest już śledzony w obu gałęziach (linia 141, 189) — wystarczy dołączyć do dict'a.
     - Pole `'m'` będzie kluczowe w 8d do priorytetyzacji PV move w sortowaniu.

   - **8b. Dodać atrybut `_current_max_depth` i naprawić obliczanie `current_ply`**
     - **Plik:** `minimax.py`, linia 118.
     - Zamienić `current_ply = self.depth - depth` na `current_ply = self._current_max_depth - depth`.
     - Dodać `self._current_max_depth = self.depth` w `__init__` (domyślnie = max depth).
     - **Problem:** Przy iteracji depth=1, `self.depth=5`, `depth=0` → `current_ply = 5 - 0 = 5`. Błędny ply — killer moves indeksowane złym slotem. Po naprawie: `current_ply = 1 - 0 = 1` ✓.

   - **8c. Dodać priorytetyzację TT move w sortowaniu ruchów**
     - **Plik:** `order_moves_minimax.py`, linia 24 i 45–85.
     - Dodać parametr `tt_move: chess.Move | None = None` do `order_moves()`.
     - Jeśli `move == tt_move`, przypisać bonus `2_000_000` (wyższy niż `promotion_bonus = 1_500_000`) — TT move zawsze pierwszy.
     - Stała `self.move_ordering_tt_move_bonus = 2_000_000` w `__init__`.
     - **⚠ Sygnatura abstrakcyjna:** Zmiana w `order_moves.py` (linia 19) wymaga dodania `tt_move` parametru. `OrderMovesMCTS` w `order_moves_mcts.py` też musi dodać parametr (choćby ignorowany: `tt_move=None`).

   - **8d. Przekazać TT move do `order_moves` w `__minimax_alphabeta`**
     - **Plik:** `minimax.py`, linia 140.
     - Pobrać TT move: `tt_move = tt_entry.get('m') if tt_entry else None`.
     - Przekazać: `order_moves(board, legal_moves, ply=current_ply, tt_move=tt_move)`.
     - `tt_entry` jest już pobrany w linii 123 — wystarczy wyciągnąć pole `'m'`.

   - **8e. Owinąć search w pętlę ID w `make_move()`**
     - **Plik:** `minimax.py`, linie 54–101.
     - **Nowa struktura:**
       ```python
       # NIE czyścić TT (usunąć linia 55: self.transposition_table = {})
       # Czyścić killer moves i history RAZ przed pętlą
       self.order_moves_minimax.killer_moves = [...]
       self.order_moves_minimax.history_heuristic_table = [...]

       internal_board = board.copy()
       legal_moves = list(internal_board.legal_moves)
       if not legal_moves:
           return

       best_move = None
       best_value = -math.inf if self.color == chess.WHITE else math.inf
       is_maximizing = self.color == chess.WHITE
       board_hash = self.hasher.hash_board(internal_board)

       for current_depth in range(1, self.depth + 1):
           self._current_max_depth = current_depth
           alpha = -math.inf
           beta = math.inf

           # TT move z poprzedniej iteracji dla root
           tt_entry = self.transposition_table.get(board_hash)
           tt_move = tt_entry.get('m') if tt_entry else None
           ordered_moves = self.order_moves_minimax.order_moves(
               internal_board, legal_moves, ply=0, tt_move=tt_move)

           iteration_best_move = None
           iteration_best_value = -math.inf if is_maximizing else math.inf

           for move in ordered_moves:
               internal_board.push(move)
               board_value = self.__minimax_alphabeta(
                   internal_board, current_depth - 1, alpha, beta, not is_maximizing)
               internal_board.pop()
               # ... (identyczna logika wyboru best_move jak obecna)

           best_move = iteration_best_move
           best_value = iteration_best_value

           # Early termination: mat znaleziony
           if abs(best_value) == math.inf:
               break

       board.push(best_move)
       ```
     - **Zachowanie TT między ruchami** — bezpieczne. TT lookup sprawdza hash pozycji (Zobrist). Stare wpisy są nadpisywane wg replacement policy (`depth >= tt_entry['d']`).
     - **Killer/history czyścić raz** przed pętlą ID (nie per iterację) — kumulacja daje lepszą informację.
     - **Koszt overhead:** Iteracje 1..(k-1) kosztują ~5% iteracji k (wykładniczy branching factor).

   - **8f. Rozszerzyć logowanie**
     - Dodać `depth: {self.depth}` do istniejącego logu `LOGGER.info`.
     - Opcjonalnie: `LOGGER.debug(f'ID iteration depth={current_depth}; move={best_move}; value={best_value:.2f}')` po każdej iteracji.

   **Kolejność podkroków:** 8a → 8b → 8c → 8d → 8e → 8f

   **Pliki do zmiany:**
   | Plik | Zmiana |
   |------|--------|
   | `minimax.py` | TT store z best_move (8a), `_current_max_depth` (8b), TT move do order_moves (8d), pętla ID (8e), logowanie (8f) |
   | `order_moves_minimax.py` | Parametr `tt_move`, bonus TT move (8c) |
   | `order_moves.py` | Sygnatura abstrakcyjna — dodać `tt_move` (8c) |
   | `order_moves_mcts.py` | Dodać `tt_move=None` parametr (8c) |

    **Wpływ na poprawność:** Żaden. `__minimax_alphabeta` działa identycznie — zmienia się tylko co go wywołuje (pętla zamiast jednorazowego wywołania) i kolejność ruchów (TT move priorytet). Wynik algorytmu jest taki sam, ale szybciej osiągnięty.

---

### Kolejność wdrożenia (nowe kroki): 9 → 10 → 11 → 12 → 13

9. ✅ **Dodać Quiescence Search (KRYTYCZNY BRAK)**

   **Cel:** Przy `depth == 0` silnik zwraca statyczną ewaluację bez uwzględnienia toczących się wymian figur. Powoduje to **horizon effect** — silnik nie widzi bić tuż za horyzontem przeszukiwania i popełnia elementarne błędy taktyczne (np. ocenia pozycję na +9 po zbiciu hetmanem pionka, nie widząc natychmiastowego odbicia hetmana).

   **Analiza techniczna:**

   - **Obecny stan (linia 131 w `minimax.py`):**
     ```python
     if depth == 0 or board.is_game_over():
         return self.evaluate_board(board)
     ```
     Brak jakiegokolwiek przeszukiwania bić po osiągnięciu `depth == 0`.

   - **Proponowana zmiana — nowa metoda `__quiescence_search`:**
     ```python
     def __quiescence_search(self, board: chess.Board, alpha: float, beta: float,
                             maximizing_player: bool) -> float:
         """
         Quiescence search: kontynuuje przeszukiwanie wyłącznie bić i promocji
         aż pozycja się 'uspokoi', eliminując horizon effect.
         """
         stand_pat = self.evaluate_board(board)

         if maximizing_player:
             if stand_pat >= beta:
                 return beta
             alpha = max(alpha, stand_pat)
         else:
             if stand_pat <= alpha:
                 return alpha
             beta = min(beta, stand_pat)

         # Generuj tylko bicia i promocje
         capture_moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion is not None]

         # Opcjonalnie order captures by MVV-LVA
         for move in capture_moves:
             board.push(move)
             evaluation = self.__quiescence_search(board, alpha, beta, not maximizing_player)
             board.pop()

             if maximizing_player:
                 if evaluation > alpha:
                     alpha = evaluation
                 if alpha >= beta:
                     return beta
             else:
                 if evaluation < beta:
                     beta = evaluation
                 if beta <= alpha:
                     return alpha

         return alpha if maximizing_player else beta
     ```

   - **Zmiana w `__minimax_alphabeta` (linia 131):**
     ```python
     if depth == 0:
         return self.__quiescence_search(board, alpha, beta, maximizing_player)
     if board.is_game_over():
         return self.evaluate_board(board)
     ```

   - **Opcjonalne ulepszenia:**
     * **Delta pruning:** Jeśli `stand_pat + max_possible_capture_gain < alpha` (maximizer), przytnij — żadne bicie nie poprawi sytuacji.
     * **Limit głębokości QS:** Dodać parametr `qs_depth` (np. max 8) aby uniknąć eksplozji w pozycjach z wieloma biciami.
     * **SEE (Static Exchange Evaluation):** Filtrować bicia — przeszukiwać tylko te z korzystną wymianą (SEE >= 0). Eliminuje przeszukiwanie oczywistych strat materiału.

   - **Wpływ na wydajność:** QS zwiększa liczbę węzłów (typowo 2-5x więcej ewaluacji), ale drastycznie poprawia jakość gry. Przy dobrym move orderingu bić (MVV-LVA) i delta pruning, koszt jest umiarkowany.

   - **Wpływ na siłę gry:** KRYTYCZNY. Bez QS silnik przegrywa figury na każdym poziomie trudności. To jedyna optymalizacja, która **fundamentalnie** zmienia jakość gry.

   - **Pliki do zmiany:**
     | Plik | Zmiana |
     |------|--------|
     | `minimax.py` | Nowa metoda `__quiescence_search`, zmiana warunku `depth == 0` |

10. ✅ **Dodać Check Extensions**

    **Cel:** Gdy pozycja jest w szachu po wykonaniu ruchu, rozszerzyć głębokość przeszukiwania o 1. Szach drastycznie redukuje liczbę legalnych ruchów (zwykle 1-3), więc koszt jest minimalny, a zysk duży — unikanie przegapienia forsownych matów i taktyk opartych na szachach.

    **Analiza techniczna:**

    - **Obecny stan:** Brak jakichkolwiek rozszerzeń przeszukiwania. Każda gałąź jest przeszukiwana do tej samej głębokości, niezależnie od taktycznej intensywności pozycji.

    - **Proponowana zmiana w `__minimax_alphabeta` (wewnątrz pętli po ruchach, po `board.push(move)`):**
      ```python
      for move in ordered_moves:
          board.push(move)
          extension = 1 if board.is_check() else 0
          evaluation = self.__minimax_alphabeta(board, depth - 1 + extension, alpha, beta, not maximizing_player)
          board.pop()
      ```

    - **Uwaga:** `board.is_check()` po `push` sprawdza czy ruch DAŁ szacha (strona do ruchu jest w szachu). Jest to informacja już częściowo dostępna z `gives_check` w move orderingu — potencjalnie można ją przekazać, aby uniknąć podwójnego obliczania.

    - **Ograniczenie głębokości:** Aby uniknąć eksplozji (series of checks), ograniczyć extensions:
      * Max 1 extension per branch: `if depth - 1 + extension > self._current_max_depth + MAX_EXTENSIONS: extension = 0`
      * Lub: globalny/per-branch counter extensions (np. max 3-4 dodatkowe ply per gałąź).

    - **Wpływ na siłę gry:** Znaczący. Pozwala silnikowi widzieć forsowne maty i taktyki szachowe głębiej niż nominalna głębokość.

    - **Pliki do zmiany:**
      | Plik | Zmiana |
      |------|--------|
      | `minimax.py` | Dodać extension logic w pętli `__minimax_alphabeta` (obie gałęzie: max i min) |

11. ✅ **Dodać Null Move Pruning**

    **Cel:** Jeśli nawet po "oddaniu ruchu" przeciwnikowi (skipping our turn) pozycja jest wciąż na tyle dobra, że powoduje beta cutoff, to gałąź można bezpiecznie przyciąć. Redukuje branching factor o ~30-40%.

    **Analiza techniczna:**

    - **Idea:** Przed pełnym przeszukiwaniem ruchów wykonaj "null move" (pass turn) i przeszukaj z redukowaną głębokością (`depth - 1 - R`, gdzie `R` = 2 lub 3). Jeśli wynik >= beta, odetnij.

    - **Proponowana zmiana w `__minimax_alphabeta` (po TT lookup, przed generacją ruchów):**
      ```python
      # Null Move Pruning
      R = 2  # Reduction factor
      if (depth >= 3
              and not board.is_check()
              and not is_zugzwang_risk(board)):  # Nie stosować w końcówkach pionkowych
          board.push(chess.Move.null())
          null_eval = self.__minimax_alphabeta(board, depth - 1 - R, alpha, beta, not maximizing_player)
          board.pop()

          if maximizing_player and null_eval >= beta:
              return beta
          elif not maximizing_player and null_eval <= alpha:
              return alpha
      ```

    - **Warunki bezpieczeństwa (kiedy NIE stosować):**
      - Pozycja jest w szachu (null move byłby nielegalny koncepcyjnie).
      - Ryzyko zugzwangu: końcówki z samymi pionkami lub bardzo mało materiału (tam oddanie ruchu naprawdę boli).
      - `depth < 3`: za płytki search po redukcji.
      - Poprzedni ruch był null move (nie dopuszczać dwóch null moves z rzędu).

    - **Heurystyka `is_zugzwang_risk`:**
      ```python
      def __is_zugzwang_risk(self, board: chess.Board) -> bool:
          """Prosta heurystyka: nie stosuj null move w końcówkach z małą ilością materiału."""
          side = board.turn
          # Ma przynajmniej jedną figurę oprócz króla i pionków
          return not bool(board.pieces(chess.KNIGHT, side) | board.pieces(chess.BISHOP, side) |
                          board.pieces(chess.ROOK, side) | board.pieces(chess.QUEEN, side))
      ```

    - **Wpływ na wydajność:** Redukcja ~30-40% przeszukiwanych węzłów. To jeden z najefektywniejszych technik pruning po alfa-beta.

    - **Wpływ na siłę gry:** Znaczący. Pozwala silnikowi unikać zbędnego przeszukiwania w pozycjach, gdzie oddanie ruchu przeciwnikowi nie zmienia wyniku.

    - **Pliki do zmiany:**
      | Plik | Zmiana |
      |------|--------|
      | `minimax.py` | Nowa metoda `__is_zugzwang_risk`, logika NMP w `__minimax_alphabeta` |

12. ✅ **Ograniczyć rozmiar tablicy transpozycji**

    **Cel:** `self.transposition_table = {}` rośnie bez limitu przez całą grę. W długich grach (40+ ruchów, głębokość 5-6) TT może zawierać miliony wpisów → degradacja wydajności (pamięć + GC pressure).

    **Analiza techniczna:**

    - **Obecny stan:** Dict bez limitu. TT nigdy nie jest czyszczona (`self.transposition_table = {}` usunięte w kroku 8e — TT preserved across moves).

    - **Proponowane rozwiązanie — replacement policy z limitem rozmiaru:**
      ```python
      TT_MAX_SIZE = 1_000_000  # ~50-100 MB w zależności od rozmiaru wpisu

      # W zapisie do TT:
      if len(self.transposition_table) >= self.TT_MAX_SIZE:
          # Opcja A: wyczyść starsze/płytsze wpisy (age-based)
          # Opcja B: wyczyść losowo ~25% wpisów
          # Opcja C: nadpisuj zawsze (current policy z limitem)
          pass
      ```
    - **Alternatywa — generational cleanup:** Dodać numer generacji (move counter). Przy każdym ruchu (`make_move`) inkrementować. Wpisy starsze niż N generacji mogą być nadpisywane bez sprawdzania głębokości.

    - **Prosta implementacja (rekomendowana):**
      * Dodać `self._tt_generation = 0` w `__init__`.
      * Inkrementować w `make_move()`.
      * Dodać pole `'g': self._tt_generation` do każdego wpisu TT.
      * Replacement policy: nadpisuj jeśli `depth >= tt_entry['d']` LUB `self._tt_generation - tt_entry['g'] >= 2`.
      * Opcjonalnie: cleanup starych wpisów co N ruchów.

    - **Wpływ:** Zapobiega degradacji wydajności w długich grach. Nie wpływa na siłę gry w typowych grach (<40 ruchów po opening book).

    - **Pliki do zmiany:**
      | Plik | Zmiana |
      |------|--------|
      | `minimax.py` | `TT_MAX_SIZE`, `_tt_generation`, replacement policy, opcjonalny cleanup |

13. ✅ **Dodać Aspiration Windows (z Iterative Deepening)**

    **Cel:** Mając wynik z poprzedniej iteracji ID, kolejna iteracja startuje z wąskim oknem `[prev_score - delta, prev_score + delta]` zamiast `[-inf, +inf]`. Węższe okno = więcej cutoffów = szybszy search.

    **Analiza techniczna:**

    - **Obecny stan (w pętli ID, linie 72-73 w `minimax.py`):**
      ```python
      alpha: float = -math.inf
      beta: float = math.inf
      ```
      Każda iteracja zaczyna z pełnym oknem — marnuje potencjał pruning.

    - **Proponowana zmiana:**
      ```python
      ASPIRATION_DELTA = 50  # W centipawnach (0.50 pawn)

      for current_depth in range(1, self.depth + 1):
          self._current_max_depth = current_depth

          if current_depth == 1:
              alpha = -math.inf
              beta = math.inf
          else:
              alpha = best_value - ASPIRATION_DELTA
              beta = best_value + ASPIRATION_DELTA

          # ... search ...

          # Jeśli fail-high lub fail-low: re-search z pełnym oknem
          if iteration_best_value <= alpha or iteration_best_value >= beta:
              alpha = -math.inf
              beta = math.inf
              # Re-search at same depth
              # (powtórzyć search z pełnym oknem)
              ...
      ```

    - **Obsługa fail-high/fail-low:**
      * Fail-low (`iteration_best_value <= alpha`): prawdziwa wartość jest niższa niż oczekiwano → widen alpha.
      * Fail-high (`iteration_best_value >= beta`): prawdziwa wartość jest wyższa → widen beta.
      * Strategia: przy fail → podwoić delta i re-search. Przy kolejnym fail → pełne okno.

    - **Wpływ na wydajność:** ~10-20% mniej węzłów w typowych pozycjach. Koszt re-searchów jest amortyzowany (rzadkie, <5% iteracji).

    - **Uwaga:** Aspiration windows działają najlepiej w połączeniu z dobrym move orderingiem i TT. Obecna implementacja (po krokach 1-8) jest dobrą bazą.

    - **Pliki do zmiany:**
      | Plik | Zmiana |
      |------|--------|
      | `minimax.py` | Logika aspiration windows w pętli ID w `make_move()` |


---

## Analiza po wdrożeniu kroków 1-13 (post-implementation review)

Pełny przegląd `minimax.py` po wdrożeniu wszystkich kroków planu. Wszystkie kroki 1-13 (z wyjątkiem zrevertowanego kroku 4) są poprawnie zaimplementowane. Poniżej zidentyfikowane **nowe bugi** i **istotne usprawnienia**, które nie były częścią pierwotnego planu.

### 14. ✅ **Null Move Pruning — brak ochrony przed podwójnym null move (BUG)**

**Cel:** Po wykonaniu null move, w rekurencji oddajemy ruch przeciwnikowi. Obecnie nic nie zapobiega temu, by przeciwnik również wykonał null move w swoim wywołaniu `__minimax_alphabeta`. Dwa null move pod rząd są semantycznie równoważne pasowaniu obu stron — pozycja jest niezmieniona, ale głębokość zredukowana o `2 * (1 + R) = 6` ply. To powoduje, że ewaluacja jest liczona na zupełnie innej (płytszej) głębokości niż gdyby kontynuować normalne przeszukiwanie, co prowadzi do błędnych odcięć.

**Analiza techniczna:**

- **Obecny stan w `__minimax_alphabeta` (linie 254-263):**
  ```python
  if (depth >= 3
          and not board.is_check()
          and not self.__is_zugzwang_risk(board)):
      R: int = 2
      board.push(chess.Move.null())
      null_eval = self.__minimax_alphabeta(board, depth - 1 - R, alpha, beta,
                                          not maximizing_player, extensions_left)
      board.pop()
  ```
  Brak parametru ani flagi blokującej kolejny null move w rekurencji.

- **Scenariusz buga:**
  - Maximizer w pozycji o depth=8, brak szacha, są figury → wykonuje null move.
  - Rekurencyjnie minimizer dostaje depth=5, brak szacha, ma figury → też wykonuje null move.
  - Rekurencyjnie maximizer dostaje depth=2, oblicza statycznie (lub QS).
  - Wynik = statyczna ewaluacja pozycji wyjściowej, ale "udajemy" że jest to wynik na depth=8.
  - Maximizer porównuje to z beta i może wykonać błędny cutoff.

- **Proponowana zmiana — dodać parametr `can_null: bool`:**
  ```python
  def __minimax_alphabeta(self, board, depth, alpha, beta,
                          maximizing_player, extensions_left, can_null: bool = True):
      # ...
      if (can_null and depth >= 3 and not board.is_check()
              and not self.__is_zugzwang_risk(board)):
          board.push(chess.Move.null())
          null_eval = self.__minimax_alphabeta(board, depth - 1 - R, alpha, beta,
                                              not maximizing_player, extensions_left,
                                              can_null=False)  # Zablokuj kolejny null
          board.pop()
  ```
  - Domyślnie `can_null=True` w wywołaniach z pętli ruchów (gracze grają normalne ruchy).
  - Po null move przekazujemy `can_null=False`, blokując konsekutywne null.

- **Wpływ na poprawność:** Eliminuje fałszywe cutoffs wynikające z double-null. Znacząca poprawa jakości NMP, szczególnie w pozycjach taktycznie złożonych.

- **Pliki do zmiany:**
  | Plik | Zmiana |
  |------|--------|
  | `minimax.py` | Dodać parametr `can_null` do `__minimax_alphabeta`, ustawić `can_null=False` po null move |

### 15. ✅ **Quiescence Search — brak obsługi szacha (BUG/USPRAWNIENIE)**

**Cel:** Gdy pozycja na wejściu do QS jest w szachu, obecny kod stosuje stand-pat (statyczna ewaluacja jako lower bound dla maximizera). To jest **fundamentalnie błędne** — strona w szachu nie może "spasować" (stand-pat zakłada że strona ma opcję nie wykonywania ruchu, ale w szachu to jest niemożliwe). Ponadto QS generuje tylko bicia i promocje — jeśli jedyną legalną odpowiedzią na szach jest cichy ruch (np. odejście królem), QS go pominie i wróci stand-pat zamiast prawidłowo zewaluować pozycję.

**Analiza techniczna:**

- **Obecny stan w `__quiescence_search` (linie 376-383):**
  ```python
  stand_pat: float = self.evaluate_board(board)
  if maximizing_player:
      if stand_pat >= beta:
          return beta
      ...
  ```
  Brak sprawdzenia `board.is_check()` przed stand-pat.

- **Scenariusz buga:**
  - Maximizer (białe) w QS, w szachu, jedyna obrona to cichy ruch królem.
  - `stand_pat = evaluate_board(board)` zwraca jakąś wartość (np. -2.5, bo materiał czarnych przewyższa).
  - `if stand_pat >= beta`: prawdopodobnie nie. Continue.
  - `if stand_pat > alpha: alpha = stand_pat`.
  - Pętla po noisy_moves: brak bić/promocji (jedyne legalne ruchy to cichy ruch królem) → pętla pusta.
  - Return `alpha` = stand_pat = -2.5.
  - **Prawda:** najlepszy ruch (cichy uciec królem) prowadzi do innej oceny, np. -1.0.
  - **Pomyłka:** silnik widzi pozycję jako -2.5, podczas gdy jest -1.0.

- **Proponowana zmiana — sprawdzenie szacha na wejściu:**
  ```python
  in_check: bool = board.is_check()
  
  if not in_check:
      stand_pat: float = self.evaluate_board(board)
      # ... istniejąca logika stand-pat
  
  if qs_depth >= self.QS_MAX_DEPTH:
      return self.evaluate_board(board) if in_check else stand_pat
  
  # Jeśli w szachu: generuj WSZYSTKIE legalne ruchy (check evasion)
  # Jeśli nie w szachu: tylko bicia i promocje
  if in_check:
      candidate_moves = list(board.legal_moves)
      if not candidate_moves:
          # Mat — strona do ruchu jest mata
          return -math.inf if maximizing_player else math.inf
  else:
      candidate_moves = [filtruj noisy moves jak wcześniej]
  ```

   - **Wpływ na siłę gry:** Znaczący. Bez tej poprawki silnik regularnie błędnie ocenia pozycje z szachem na granicy horyzontu, co prowadzi do utraty figur i przegrywania pozycji.

   - **Wpływ na wydajność:** Lekki wzrost kosztu QS (gdy w szachu, generujemy wszystkie ruchy zamiast tylko bić). Ale szachy są rzadkie statystycznie, więc średni narzut jest mały.

   - **Pliki do zmiany:**
     | Plik | Zmiana |
     |------|--------|
     | `minimax.py` | Dodać check evasion w `__quiescence_search` |

### 16. ✅ **Killer moves — kolizje slotów przy check extensions (DROBNY PROBLEM)**

**Cel:** `current_ply` jest obliczany jako `_current_max_depth - depth`, ale przy check extensions `depth` może utrzymywać tę samą wartość przez kilka rzeczywistych ply (każde extension daje `depth - 1 + 1 = depth`). To powoduje, że killer moves z różnych rzeczywistych ply mapują się do tego samego slotu w `killer_moves[ply]`, nadpisując się wzajemnie.

**Lokalizacja:** `engine/minimax.py`, linia 118.

**Problem:** Przy check extensions `current_ply` dla ply 0 może być równe 5 (przy głębokości 5), więc `self.killer_moves[0]` może być nadpisywane przez różne gałęzie rekurencji. To nie jest błąd krytyczny, bo killer move heuristic działa na zasadzie probabilistycznej — nawet jeśli nie jest idealnie dokładna, to i tak poprawia się w miarę gry.

**Proponowana zmiana:** Przekazywać prawdziwy `actual_ply` jako parametr (inkrementowany przy każdym wywołaniu rekurencyjnym) zamiast wyliczać go z `depth`. Wymaga rozszerzenia tablicy `killer_moves` do `self.depth + MAX_CHECK_EXTENSIONS + 1`.

**Priorytet:** Niski. Wpływ na siłę gry jest marginalny.

**Pliki do zmiany:**
| Plik | Zmiana |
|------|--------|
| `minimax.py` | Przekazywać `actual_ply` jako parametr `__minimax_alphabeta` |
| `order_moves_minimax.py` | Zwiększyć rozmiar `killer_moves` o `MAX_CHECK_EXTENSIONS` |

---

## Najnowsze addendum końcowe — finalny zakres domknięcia Minimax

Ta sekcja jest aktualnym, końcowym podsumowaniem tego, co jeszcze należy zrobić, żeby uznać temat minimaxa za zamknięty. Wcześniejsze sekcje 32–34 pozostają merytorycznie aktualne, ale poniższa lista porządkuje priorytety i dodaje ostatnie dwa punkty z audytu.

### Rekomendowana kolejność finalnego wdrożenia: 32 → 33 → 34 → 35 → 36

| Krok | Priorytet | Plik | Charakter | Status |
|------|-----------|------|-----------|--------|
| 32 | P1 | `engine/minimax.py` | TT replacement policy — nie nadpisywać świeższych wpisów starym snapshotem | ✅ Wdrożone |
| 33 | P2 | `engine/minimax.py` | Defensywny fallback `best_move` w inner nodes | ✅ Wdrożone |
| 34 | P2 | `engine/order_moves_minimax.py` | Spójne rozmiarowanie `killer_moves` pod `actual_ply` i check extensions | ✅ Wdrożone |
| 35 | P2 | `engine/minimax.py` | Guard limitu check extensions również na root | ✅ Wdrożone |
| 36 | P3 | `engine/minimax.py` | Opcjonalny reorder ruchów przy aspiration re-search | ✅ Wdrożone |

**Decyzja końcowa:** nie zmieniać teraz fail-hard semantyki NMP/QS/RFP na fail-soft. Obecne `return alpha` / `return beta` jest poprawne i bezpieczne; ewentualna zmiana wymagałaby osobnego audytu TT flag, aspiration windows i mate-distance scoringu. To nie jest potrzebne do domknięcia minimaxa.

### 32. ✅ **Re-fetch `current_tt_entry` przed zapisem do TT**

**Cel:** Replacement policy w `__minimax_alphabeta()` nie powinna używać snapshotu `tt_entry` pobranego na początku funkcji do decyzji o zapisie po zakończeniu pętli ruchów.

**Problem:** W trakcie rekurencji ta sama pozycja może zostać zapisana w TT przez transpozycję z większą głębokością lub nowszą generacją. Jeśli po powrocie bieżący node użyje starego `tt_entry`, może nadpisać świeższy wpis płytszym wynikiem.

**Docelowy schemat — w obu gałęziach, max i min:**

```python
current_tt_entry = self.transposition_table.get(board_hash)
if (not current_tt_entry
        or depth >= current_tt_entry['d']
        or self._tt_generation - current_tt_entry['g'] >= self.TT_MAX_AGE):
    self.transposition_table[board_hash] = {
        'v': self.__score_to_tt(value, actual_ply),
        'd': depth,
        'f': flag,
        'm': best_move,
        'g': self._tt_generation,
    }
```

**Uwaga:** Początkowy `tt_entry` nadal zostaje użyty do TT lookup i pobrania `tt_move`; re-fetch dotyczy tylko decyzji replacement tuż przed zapisem.

**Status:** ✅ Wdrożone. Priorytet P1 — najważniejsza pozostała poprawka TT.

### 33. ✅ **Defensywny fallback `best_move` w inner nodes**

**Cel:** Ujednolicić wybór `best_move` w `__minimax_alphabeta()` z logiką używaną w `__search_root()`.

**Problem:** Root ma defensywny warunek `best_move is None and value == current_best`, ale inner nodes nadal używają wyłącznie ścisłej nierówności. Po mate-distance scoringu realny wpływ jest mały, ale dla spójności i odporności na przyszłe zmiany warto to domknąć.

**Docelowy schemat:**

```python
# maximizer
if (evaluation > max_evaluation) or (best_move is None and evaluation == max_evaluation):
    max_evaluation = evaluation
    best_move = move

# minimizer
if (evaluation < min_eval) or (best_move is None and evaluation == min_eval):
    min_eval = evaluation
    best_move = move
```

**Status:** ✅ Wdrożone. Priorytet P2 — defensywność i spójność.

### 34. ✅ **Zsynchronizować początkowe rozmiarowanie `killer_moves` z `actual_ply`**

**Cel:** `OrderMovesMinimax.__init__()` powinien inicjalizować `killer_moves` rozmiarem zgodnym z faktycznym zakresem `actual_ply` po check extensions.

**Problem:** Produkcyjny flow w `Minimax.make_move()` nadpisuje rozmiar poprawnie, ale bezpośrednie użycie `OrderMovesMinimax` startuje z tablicą `self.depth + 1`, co jest niespójne z `self.depth + MAX_CHECK_EXTENSIONS + 1`.

**Docelowy schemat:**

```python
# Must stay in sync with Minimax.MAX_CHECK_EXTENSIONS.
MAX_CHECK_EXTENSIONS: int = 3

# Stores 2 moves per actual_ply, not old depth-derived ply.
self.killer_moves = [[None, None] for _ in range(self.depth + self.MAX_CHECK_EXTENSIONS + 1)]
```

**Status:** ✅ Wdrożone. Priorytet P2 — spójność i testowalność.

### 35. ✅ **Dodać guard limitu Check Extensions na poziomie root**

**Cel:** `__search_root()` powinien respektować `MAX_CHECK_EXTENSIONS` tak samo jak rekurencyjny `__minimax_alphabeta()`. Obecnie root zawsze daje `extension = 1`, jeśli ruch daje szacha, nawet gdy `MAX_CHECK_EXTENSIONS == 0`.

**Lokalizacja:** `engine/minimax.py`, metoda `__search_root()`:

```python
# obecnie
extension = 1 if board.is_check() else 0
effective_extensions = self.MAX_CHECK_EXTENSIONS - extension
```

**Docelowy schemat:**

```python
extensions_left = self.MAX_CHECK_EXTENSIONS
extension = 1 if extensions_left > 0 and board.is_check() else 0
effective_extensions = extensions_left - extension
```

**Wpływ na poprawność:** Brak zmiany przy domyślnej konfiguracji. Poprawia spójność i umożliwia faktyczne wyłączenie check extensions ustawieniem `MAX_CHECK_EXTENSIONS = 0`.

**Wpływ na wydajność:** Pomijalny.

**Pliki do zmiany:**
| Plik | Zmiana |
|------|--------|
| `minimax.py` | W `__search_root()` dodać guard `extensions_left > 0` analogicznie do `__minimax_alphabeta()` |

**Status:** ✅ Wdrożone. Priorytet niski — spójność konfiguracji i edge case.

### 36. ✅ **Opcjonalnie: przeliczyć move ordering przy aspiration re-search**

**Cel:** Po nieudanym narrow searchu aspiration window (`fail-low` / `fail-high`) pełny re-search powinien móc wykorzystać informacje zebrane przez pierwszą próbę: zaktualizowany TT move, killer moves i history heuristic.

**Lokalizacja:** `engine/minimax.py`, `make_move()`:

```python
ordered_moves = self.order_moves_minimax.order_moves(
    internal_board, legal_moves, ply=0, tt_move=tt_move)

iteration_best_move, iteration_best_value = self.__search_root(...)

if current_depth > 1 and (iteration_best_value <= alpha or iteration_best_value >= beta):
    alpha = -math.inf
    beta = math.inf
    iteration_best_move, iteration_best_value = self.__search_root(
        internal_board, ordered_moves, current_depth, alpha, beta, is_maximizing)
```

**Problem:** Re-search korzysta z tej samej listy `ordered_moves`, mimo że narrow search mógł zaktualizować root TT entry lub heurystyki. To jest poprawne semantycznie, bo lista legalnych ruchów się nie zmienia, ale suboptymalne wydajnościowo.

**Proponowana zmiana — wariant prosty:** Przed full-window re-search ponownie pobrać root TT move i przeliczyć ordering:

```python
if current_depth > 1 and (iteration_best_value <= alpha or iteration_best_value >= beta):
    alpha = -math.inf
    beta = math.inf

    tt_entry = self.transposition_table.get(board_hash)
    tt_move = tt_entry.get('m') if tt_entry else iteration_best_move
    ordered_moves = self.order_moves_minimax.order_moves(
        internal_board, legal_moves, ply=0, tt_move=tt_move)

    iteration_best_move, iteration_best_value = self.__search_root(
        internal_board, ordered_moves, current_depth, alpha, beta, is_maximizing)
```

**Uwaga:** Ten krok nie jest wymagany do poprawności. Jeżeli chcemy minimalnie domknąć minimax, wystarczy wdrożyć kroki 32–35. Krok 36 jest czystą optymalizacją aspiration windows i został wdrożony jako końcowy polish wydajnościowy.

**Wpływ na poprawność:** Żaden.

**Wpływ na wydajność:** Potencjalnie pozytywny przy fail-high/fail-low aspiration windows; koszt przeliczenia orderingu root jest mały względem full re-searchu.

**Pliki do zmiany:**
| Plik | Zmiana |
|------|--------|
| `minimax.py` | Przy aspiration full-window re-search ponownie wyznaczyć `tt_move` i `ordered_moves` |

**Status:** ✅ Wdrożone. Priorytet niski — optymalizacja, nie bug.

---

## EOF — aktualna finalna kolejność domknięcia Minimax

Najbardziej aktualny zakres końcowy to:

1. **32 / P1:** ✅ re-fetch `current_tt_entry` przed zapisem TT w `__minimax_alphabeta()`.
2. **33 / P2:** ✅ defensywny fallback `best_move` w inner nodes przy remisie wartości.
3. **34 / P2:** ✅ spójne rozmiarowanie `killer_moves` pod `actual_ply` i `MAX_CHECK_EXTENSIONS`.
4. **35 / P2:** ✅ root check extension guard — root respektuje `MAX_CHECK_EXTENSIONS`.
5. **36 / P3 opcjonalnie:** ✅ przeliczenie root move orderingu przy aspiration full-window re-search.

**Definition of Done:** po wdrożeniu 32–36 temat Minimax uznajemy za domknięty. Fail-hard NMP/QS/RFP pozostaje bez zmian jako świadoma decyzja projektowa.
