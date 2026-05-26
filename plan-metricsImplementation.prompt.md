# Plan implementacji metryk — zmodyfikowany po analizie kodu

## Kontekst

Praca magisterska wymaga zebrania szczegolowych metryk z silnika szachowego (Minimax i MCTS) do przeprowadzenia eksperymentow porownawczych (round-robin, skalowanie glebokosci/czasu, benchmark Stockfish, dokladnosc taktyczna). Oryginalny plan implementacji wymaga korekty po analizie aktualnego kodu — niektorych punktow nie trzeba implementowac od zera (MCTS ma juz bogate logowanie), a niektorych nie da sie zaimplementowac dokladnie jak w planie (np. `make_move()` zwraca `None`, nie ruch).

---

## Kluczowe obserwacje z analizy kodu

### 1. `make_move()` zwraca `None` — ruch pushowany wewnetrznie
Oryginalny plan zaklada `move = player.make_move(board)` w `engine.py`, ale faktycznie `make_move()` pushuje ruch bezposrednio na board i zwraca `None`. To wymaga innego podejscia do pomiaru czasu i logowania w engine.py — musimy odczytac ruch z `board.move_stack[-1]` po wywolaniu.

### 2. MCTS ma juz bogate logowanie (linie 206-213 mcts.py)
MCTS juz loguje: duration, move, mean_value, new_visits, total_visits, iterations, reused visits, policy, proven values, top-3 moves. Zamiast duplikowac to w `self.stats`, lepiej ujednolicic format i dodac brakujace metryki (nodes_created, max_depth, eval_calls, entropy, convergence).

### 3. Struktura TT cutoff w minimax.py (linie 391-400)
Sa DWA punkty return: flag 'E' (linia 394) i `beta <= alpha` po L/U (linia 400). Oba wymagaja `tt_cutoffs += 1`.

### 4. NMP i RFP maja po DWA cutoff points (max i min)
- NMP: linia 441 (max: `null_eval >= beta`) i 443 (min: `null_eval <= alpha`)  
- RFP: linia 421 (max: `static_eval - rfp_margin >= beta`) i 423 (min: `static_eval + rfp_margin <= alpha`)

### 5. MCTS `__backpropagate` uzywa `value = 1.0 - value` (complement flip)
Oryginalny plan proponuje `current = 1 - current` co jest tym samym, ale zmienna w kodzie nazywa sie `value`. Kod dodajacy depth tracking musi zachowac istniejaca logike.

### 6. Phase — evaluator wywolywany wiele razy w trakcie searchu
`evaluate_board()` wywoluje `__get_game_phase()` za kazdym razem. Aby dostac phase dla pozycji root (do logowania), lepiej obliczyc ja RAZ na poczatku `make_move()` zamiast polegac na `last_phase` z evaluatora (ktory bedzie nadpisywany przez kazde wywolanie w searchu).

### 7. Engine.py — obecne logowanie przez LOGGER, brak JSON
Trzeba dodac nowy argument CLI (`--json_log`) i JSONL logging obok istniejacego LOGGER. Nie zastepujemy obecnego logowania.

---

## Pliki do modyfikacji

| Plik | Zakres zmian |
|------|-------------|
| `engine/minimax.py` | Faza 0.1, Faza 1 (1.1-1.18), Faza 3 (3.1-3.2) |
| `engine/order_moves_minimax.py` | Krok 1.15 (killer stats) |
| `engine/mcts.py` | Faza 0.2, Faza 2 (2.1-2.12) |
| `engine/board_evaluator_trad.py` | Krok 3.2 (last_phase) |
| `engine/engine.py` | Faza 0.3 (JSON logging, timing, game summary) |
| `engine/main.py` | Nowy argument `--json_log` |

---

## Faza 0 — Infrastruktura

### 0.1 Slownik `self.stats` w Minimax (`engine/minimax.py`)

**Miejsce:** `make_move()`, po linii 124 (`internal_board = board.copy()`), PRZED petla ID (linia 136).

```python
self.stats = {
    'nodes_searched': 0,
    'tt_lookups': 0,
    'tt_hits': 0,
    'tt_cutoffs': 0,
    'nmp_cutoffs': 0,
    'nmp_attempts': 0,
    'rfp_cutoffs': 0,
    'futility_prunes': 0,
    'lmp_prunes': 0,
    'see_prunes': 0,
    'check_extensions': 0,
    'aspiration_researches': 0,
    'qs_nodes': 0,
    'qs_max_depth': 0,
    'see_calls': 0,
    'killer_hits': 0,
    'killer_checks': 0,
    'depth_completed': 0,
    'tt_size': 0,
    'pv_from_tt': 0,
    'nodes_per_depth': [],
}
self.order_moves_minimax.stats = self.stats
```

**Uwaga:** Dodajemy `see_prunes` (brak w oryginalnym planie — SEE pruning to osobny mechanizm od futility/LMP i powinien byc sledzony oddzielnie).

### 0.2 Slownik `self.stats` w MCTS (`engine/mcts.py`)

**Miejsce:** `__run_mcts()`, linia ~140, po `root = self.__get_or_create_root(board)`.

```python
self.stats = {
    'iterations': 0,
    'skipped_terminals': 0,
    'nodes_created': 0,
    'max_depth': 0,
    'eval_calls': 0,
    'eval_cache_hits': 0,
    'reused_visits': root.visits,
    'root_children_count': 0,
    'best_child_visits': 0,
    'root_visit_entropy': 0.0,
    'convergence_point': 1.0,
    'avg_backprop_depth': 0.0,
}
self._backprop_total_depth = 0
self._backprop_count = 0
```

**Dodatkowe vs oryginalny plan:** `eval_cache_hits` — MCTS uzywa eval cache (linie 354-356 w `__simulate`), warto wiedziec jaka czesc evaluacji pochodzi z cache.

### 0.3 JSON logging w `engine.py` + argument CLI

**Plik `engine/main.py`:** Dodac argument:
```python
parser.add_argument('-jl', '--json_log', help='JSON metrics log file', type=str, required=False)
```

**Plik `engine/engine.py`:** 

**WAZNA KOREKTA:** `make_move()` zwraca `None` i pushuje ruch na board. Dlatego:

```python
# W __run(), zamiast move = player.make_move():
t0 = time.perf_counter()
self.white_player.make_move(self.board, self.screen)
move_time = time.perf_counter() - t0
last_move = self.board.move_stack[-1] if self.board.move_stack else None

# Log JSON
if self.json_log_file and last_move:
    move_log = {
        'move_number': white_move_number,
        'side': 'WHITE',
        'move': last_move.uci(),
        'eval': getattr(self.white_player, 'last_eval', None),
        'time_s': round(move_time, 4),
        'phase': getattr(self.white_player, 'last_phase', None),
        'algorithm_stats': getattr(self.white_player, 'stats', {}),
    }
    self.json_log_file.write(json.dumps(move_log) + '\n')
    self.json_log_file.flush()
```

Analogicznie dla czarnych. Na koniec partii — game summary z termination_reason i result.

Inicjalizacja pliku JSON w `__init__`:
```python
self.json_log_file = open(f'out/{args.json_log}', 'w') if args.json_log else None
```
Zamkniecie w `__handle_game_over`.

Akumulatory czasu (`total_time_white`, `total_time_black`, `white_moves`, `black_moves`) inicjalizowane w `__run()` przed petla.

---

## Faza 1 — Metryki Minimax

Wszystkie wstawki w `engine/minimax.py`.

### 1.1 `nodes_searched`
**Miejsce:** `__minimax_alphabeta()`, pierwsza linia ciala (linia ~378, przed `if self.__is_drawn_position`).
```python
self.stats['nodes_searched'] += 1
```

### 1.2 `tt_lookups`
**Miejsce:** Przed linia 388 (`tt_entry = self.transposition_table.get(board_hash)`).
```python
self.stats['tt_lookups'] += 1
```

### 1.3 `tt_hits`
**Miejsce:** Wewnatrz warunku `if tt_entry and tt_entry['d'] >= depth:` (linia 391), na poczatku bloku.
```python
self.stats['tt_hits'] += 1
```

### 1.4 `tt_cutoffs`
**Miejsca (DWA):**
1. Przed `return tt_value` w galeziach 'E' (po linii 393)
2. Przed `return tt_value` w `if beta <= alpha:` (po linii 399)
```python
self.stats['tt_cutoffs'] += 1
```

### 1.5 `nmp_attempts`
**Miejsce:** Wewnatrz bloku NMP (po linii 430, w warunku if), PRZED `board.push(chess.Move.null())` (linia 435).
```python
self.stats['nmp_attempts'] += 1
```

### 1.6 `nmp_cutoffs`
**Miejsca (DWA):**
1. Przed `return beta` w `if maximizing_player and null_eval >= beta:` (linia 442)
2. Przed `return alpha` w `elif not maximizing_player and null_eval <= alpha:` (linia 444)
```python
self.stats['nmp_cutoffs'] += 1
```

### 1.7 `rfp_cutoffs`
**Miejsca (DWA):**
1. Przed `return beta` w maximizer RFP (linia 422)
2. Przed `return alpha` w minimizer RFP (linia 424)
```python
self.stats['rfp_cutoffs'] += 1
```

### 1.8 `futility_prunes`
**Miejsca (DWA):**
1. Maximizer: przed `continue` w futility (linia 481)
2. Minimizer: przed `continue` w futility (linia 578)
```python
self.stats['futility_prunes'] += 1
```

### 1.9 `lmp_prunes`
**Miejsca (DWA):**
1. Maximizer: przed `continue` w LMP (linia 468)
2. Minimizer: przed `continue` w LMP (linia 565)
```python
self.stats['lmp_prunes'] += 1
```

### 1.9b `see_prunes` (NOWE — brak w oryginalnym planie)
**Miejsca (DWA):**
1. Maximizer: przed `continue` w SEE pruning (linia 491)
2. Minimizer: przed `continue` w SEE pruning (linia 588)
```python
self.stats['see_prunes'] += 1
```
**Uzasadnienie:** SEE pruning to odrebny mechanizm od futility i LMP. Sledzenie go oddzielnie pozwala ocenic skutecznosc kazdej techniki pruning niezaleznie.

### 1.10 `check_extensions`
**Miejsca (DWA):**
1. Maximizer: po `extension = 1` (linia 496), wewnatrz warunku `if extensions_left > 0 and board.is_check()`
2. Minimizer: analogicznie (linia 593)
```python
if extension == 1:
    self.stats['check_extensions'] += 1
```

### 1.11 `aspiration_researches`
**Miejsce:** `make_move()`, wewnatrz warunku fail (linia 156), na poczatku bloku `if current_depth > 1 and (iteration_best_value <= alpha or iteration_best_value >= beta):`.
```python
self.stats['aspiration_researches'] += 1
```

### 1.12 `qs_nodes`
**Miejsce:** `__quiescence_search()`, pierwsza linia ciala (linia ~692).
```python
self.stats['qs_nodes'] += 1
```

### 1.13 `qs_max_depth`
**Miejsce:** Obok 1.12.
```python
if qs_depth > self.stats['qs_max_depth']:
    self.stats['qs_max_depth'] = qs_depth
```
**Uwaga:** Parametr `qs_depth` juz istnieje w sygnaturze `__quiescence_search()` (linia 669).

### 1.14 `see_calls`
**Miejsce:** `__static_exchange_evaluation()`, pierwsza linia ciala (linia ~805).
```python
self.stats['see_calls'] += 1
```

### 1.15 `killer_hits` i `killer_checks` (plik: `order_moves_minimax.py`)

**Zmiana w `__init__`:** Dodac `self.stats = None`.

**Zmiana w `order_moves()`:** W sekcji quiet moves (linia 94-100), zamiast obecnego kodu:
```python
# Obecny kod (linia 95):
is_killer: bool = move == killers[0] or move == killers[1]

# Nowy kod:
is_killer = False
for slot in killers:
    if slot is not None:
        if self.stats:
            self.stats['killer_checks'] += 1
        if move == slot:
            is_killer = True
            if self.stats:
                self.stats['killer_hits'] += 1
            break
```

**Uwaga:** Oryginalny plan sprawdzal oba sloty nawet po trafieniu. Zmodyfikowana wersja uzywa `break` po trafieniu — nie ma sensu sprawdzac drugiego slotu jesli pierwszy trafil.

### 1.16 `depth_completed`
**Miejsce:** `make_move()`, po `self.__store_root_tt_entry(...)` (linia 166-167).
```python
self.stats['depth_completed'] = current_depth
```

### 1.17 `tt_size`
**Miejsce:** `make_move()`, przed `return` / koniec metody (przed linia 181).
```python
self.stats['tt_size'] = len(self.transposition_table)
```

### 1.18 `pv_from_tt`
**Miejsce:** `make_move()`, wewnatrz petli ID, po `tt_move = tt_entry.get('m') if tt_entry else None` (linia 148), w warunku:
```python
if tt_move is not None:
    self.stats['pv_from_tt'] += 1
```

### 1.19 `nodes_per_depth` (z Fazy 4.1 oryginalu — przeniesione tutaj)
**Miejsca w `make_move()`:**
1. Na poczatku kazdej iteracji ID (po linii 136): `nodes_before = self.stats['nodes_searched']`
2. Na koncu kazdej iteracji ID (po linii 170): `self.stats['nodes_per_depth'].append(self.stats['nodes_searched'] - nodes_before)`

---

## Faza 2 — Metryki MCTS

Wszystkie wstawki w `engine/mcts.py`.

### 2.1 `iterations`
**Miejsce:** `__run_mcts()`, PO petli while (linia ~168, po `iterations += 1`):
```python
self.stats['iterations'] = iterations
```
Iterations jest juz zliczane w zmiennej lokalnej `iterations` w petli.

### 2.2 `skipped_terminals`
**Miejsce:** `__run_mcts()`, wewnatrz `if node.proven_value is not None and node.visits > 0:` (linia 155), PRZED `iterations += 1`:
```python
self.stats['skipped_terminals'] += 1
```

### 2.3 `nodes_created`
**Miejsce:** `__expand()`, PO `node.children.append(child_node)` (linia 330):
```python
self.stats['nodes_created'] += 1
```

### 2.4 `max_depth`
**Miejsce:** `__select()`, dodac tracker:
```python
def __select(self, node: MCTSNode) -> MCTSNode:
    depth = 0
    while not node.is_terminal and node.children:
        if node.untried_moves:
            self.__prepare_untried_moves(node)
            best_child = node.best_child(self.C_PUCT)
            if self.__best_untried_score(node) >= self.__puct_child_score(node, best_child):
                break  # wracamy do expand
            node = best_child
            depth += 1
        elif node.is_fully_expanded():
            node = node.best_child(self.C_PUCT)
            depth += 1
    if hasattr(self, 'stats') and depth > self.stats.get('max_depth', 0):
        self.stats['max_depth'] = depth
    return node
```

### 2.5 `eval_calls` i `eval_cache_hits`
**Miejsce:** `__simulate()`:
1. Na poczatku (linia ~352): `self.stats['eval_calls'] += 1`
2. W galezi cache hit (linia 354-356): `self.stats['eval_cache_hits'] += 1`

**Uwaga:** Oryginalny plan nie rozroznia cache hit vs miss. Dodanie `eval_cache_hits` pozwala obliczyc cache hit rate = `eval_cache_hits / eval_calls`.

### 2.6 `root_children_count`
**Miejsce:** `__run_mcts()`, PO petli while, przed select_root_move:
```python
self.stats['root_children_count'] = len(root.children)
```

### 2.7 `best_child_visits`
**Miejsce:** `__run_mcts()`, PO `best_child, policy = self.__select_root_move(root)`:
```python
self.stats['best_child_visits'] = best_child.visits
```

### 2.8 `root_visit_entropy`
**Miejsce:** `__run_mcts()`, obok 2.6-2.7:
```python
total_visits = sum(c.visits for c in root.children) or 1
entropy = 0.0
for c in root.children:
    if c.visits > 0:
        p = c.visits / total_visits
        entropy -= p * math.log(p)
self.stats['root_visit_entropy'] = round(entropy, 4)
```
**Uwaga:** `math` jest juz importowany w mcts.py (linia 3).

### 2.9 `convergence_point`
**Miejsce:** Wymaga zmiennych w petli `__run_mcts()`:

Po inicjalizacji stats (po 0.2):
```python
convergence_iteration = None
current_best_move = None
```

Na koncu kazdej iteracji (przed `iterations += 1`):
```python
if root.children:
    top_child = max(root.children, key=lambda c: c.visits)
    if top_child.move != current_best_move:
        current_best_move = top_child.move
        convergence_iteration = iterations
```

**KOREKTA vs oryginalny plan:** Porownujemy `top_child.move` zamiast obiektu `top_child` — obiekty MCTSNode nie sa por runtime stabline, ale move jest identyfikatorem ruchu. Uzywamy `!=` na ruchach, nie na obiektach.

PO petli:
```python
self.stats['convergence_point'] = round(
    (convergence_iteration or 0) / max(iterations, 1), 4
)
```

### 2.10 `reused_visits` — juz w kroku 0.2
Zainicjalizowane jako `root.visits`. Nie wymaga dodatkowych zmian.

### 2.11 `avg_backprop_depth`
**Miejsce:** `__backpropagate()`:
```python
def __backpropagate(self, node: MCTSNode, value: float) -> None:
    depth = 0
    while node is not None:
        node.visits += 1
        node.value += value
        value = 1.0 - value
        node = node.parent
        depth += 1
    self._backprop_total_depth += depth
    self._backprop_count += 1
```

PO petli w `__run_mcts()`:
```python
if self._backprop_count > 0:
    self.stats['avg_backprop_depth'] = round(
        self._backprop_total_depth / self._backprop_count, 2
    )
```

**Uwaga:** `_backprop_total_depth` i `_backprop_count` inicjalizowane w kroku 0.2 (nie uzywa hasattr jak oryginalny plan — czysciej).

### 2.12 `last_eval` w MCTS
**Miejsce:** `__run_mcts()`, PO wybraniu best_child (linia ~170):
```python
self.last_eval = best_child.value / max(best_child.visits, 1)
```
**Uwaga:** Ta wartosc jest w [0,1] (sigmoid MCTS), nie w centypionkach. Dokumentacja powinna to jasno wskazywac.

---

## Faza 3 — Metryki wspolne / engine-level

### 3.1 `last_eval` w Minimax (`engine/minimax.py`)
**Miejsce:** `make_move()`, po petli ID (po linii 179), przed `if best_move is not None:`:
```python
self.last_eval = best_value
```

### 3.2 `last_phase`

**Podejscie zmienione vs oryginalny plan.** Evaluator jest wywolywany wielokrotnie w searchu — `last_phase` z evaluatora bylby z ostatniego wezla QS, nie z pozycji root. Lepsze rozwiazanie:

**Minimax (`engine/minimax.py`, `make_move()`):**
```python
self.last_phase = self.board_evaluator.get_game_phase(internal_board) if hasattr(self, 'board_evaluator') else None
```
Wymaga uczynienia `__get_game_phase` publicznym (zmiana nazwy na `get_game_phase`) w `board_evaluator_trad.py`.

**MCTS (`engine/mcts.py`, `__run_mcts()`):**
```python
self.last_phase = self.board_evaluator.get_game_phase(board) if hasattr(self, 'board_evaluator') else None
```

**Plik `engine/board_evaluator_trad.py`:** Dodac publiczna metode wrapper:
```python
def get_game_phase(self, board: chess.Board) -> float:
    return self.__get_game_phase(board)
```

### 3.3 Czas na ruch
Zaimplementowany w kroku 0.3 (engine.py). Pomiar `time.perf_counter()` wokol `make_move`.

### 3.4 `termination_reason` w engine.py
**Miejsce:** `__handle_game_over()`, przed zapisaniem game summary:
```python
if self.board.is_checkmate():
    termination_reason = 'checkmate'
elif self.board.is_stalemate():
    termination_reason = 'stalemate'
elif self.board.is_seventyfive_moves():
    termination_reason = 'draw_75'
elif self.board.is_fivefold_repetition():
    termination_reason = 'draw_5fold'
elif self.board.is_insufficient_material():
    termination_reason = 'draw_insufficient'
elif self.board.can_claim_fifty_moves():
    termination_reason = 'draw_50_claim'
elif self.board.can_claim_threefold_repetition():
    termination_reason = 'draw_3fold_claim'
else:
    termination_reason = 'unknown'
```

**KOREKTA:** Oryginalny plan nie pokrywa 50-move claim ani threefold claim, a kod engine.py (`__is_game_over_or_draw_claim_available`) sprawdza `can_claim_threefold_repetition()` i `can_claim_fifty_moves()`. Dodane.

### 3.5 `game_result` w engine.py
```python
outcome = self.board.outcome(claim_draw=True)
if outcome is None:
    result = '1/2-1/2'
elif outcome.winner == chess.WHITE:
    result = '1-0'
elif outcome.winner == chess.BLACK:
    result = '0-1'
else:
    result = '1/2-1/2'
```

---

## Faza 4 — Metryki pochodne (post-hoc)

Nie wymagaja zmian w kodzie silnika. Obliczane z logow JSONL.

### 4.1 EBF (Effective Branching Factor)
Obliczane z `nodes_per_depth`: `EBF[d] = nodes_per_depth[d] / nodes_per_depth[d-1]`

### 4.2 TT hit rate
`tt_hits / tt_lookups`

### 4.3 TT cutoff rate
`tt_cutoffs / tt_hits`

### 4.4 ACPL (Average Centipawn Loss)
Osobny skrypt `scripts/stockfish_reval.py` — bez zmian vs oryginalny plan.

### 4.5 MCTS eval cache hit rate (NOWE)
`eval_cache_hits / eval_calls`

---

## Kolejnosc implementacji

1. **Faza 0** (infrastruktura) — najpierw, bo reszta od niej zalezy
2. **Faza 1** (minimax) i **Faza 2** (MCTS) — niezalezne, mozna rownolegle
3. **Faza 3** (engine-level) — po Fazach 1-2 (potrzebuje `self.stats`, `last_eval`, `last_phase`)
4. **Faza 4** (post-hoc) — na koncu, po zebraniu pierwszych logow

---

## Weryfikacja

1. **Test jednostkowy:** Uruchomic partie minimax vs minimax (d=3) i sprawdzic ze `self.stats` zawiera sensowne wartosci (nodes_searched > 0, tt_lookups > 0, depth_completed == 3)
2. **Test MCTS:** Uruchomic partie MCTS vs MCTS (t=5s) i sprawdzic stats (iterations > 0, nodes_created > 0, max_depth > 0)
3. **Test JSON output:** Uruchomic z `--json_log test.jsonl` i zweryfikowac ze plik zawiera poprawny JSONL z move_log i game_summary
4. **Test regresji:** Porownac wyniki partii (ruchy, ewalucje) przed i po zmianach — instrumentacja nie powinna zmieniac zachowania silnika
5. **Test wydajnosci:** Zmierzyc narzut instrumentacji — porownac czas partii z i bez stats. Oczekiwany narzut < 1%
