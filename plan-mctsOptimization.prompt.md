## Plan: Optymalizacja algorytmu MCTS

Aktualna implementacja MCTS w [mcts.py](engine/mcts.py) stosuje klasyczny wariant z losowymi rolloutami, co — jak opisano w pracy magisterskiej — nie sprawdza się w szachach. Poniższy plan adresuje 7 konkretnych optymalizacji, od naprawy bugów, przez eliminację losowych symulacji i normalizację wartości, po usprawnienia strukturalne (reuse drzewa, transpozycje, konfigurowalność czasu).

### Kolejność wdrożenia: 1 → 2 → 3+4+5 (razem) → 6 → 7

### Kroki

1. ✅ **Naprawić `MCTSNN`**

   **Cel:** Klasa `MCTSNN` w [mcts_nn.py](engine/mcts_nn.py) dziedziczy po `Player` zamiast `MCTS`, przez co nie posiada implementacji `make_move` i nie jest funkcjonalnym graczem MCTS. Zmienić dziedziczenie na `MCTS` (analogicznie do `MCTSTrad` w [mcts_trad.py](engine/mcts_trad.py)).

   **Analiza techniczna:**

   - **Obecny stan i skutki buga:**
     - `MCTSNN(Player)` — dziedziczy z `Player`, który deklaruje `make_move` jako `@abc.abstractmethod`.
     - `MCTSNN` **nie** implementuje `make_move` → klasa nie jest abstrakcyjna (brak `ABC` w `MCTSNN`, Python nie wymusza tego), ale wywołanie `make_move` wywoła `Player.make_move` → **`TypeError: Can't instantiate abstract class`** przy próbie utworzenia instancji w `Engine.__get_player()` (engine.py, linia 75).
     - Efekt: wybranie gracza `MCTS_NN` w konfiguracji gry powoduje natychmiastowy crash.

   - **Zmiana — 2 linie w `mcts_nn.py`:**
     1. Import: `from player import Player` → `from mcts import MCTS`.
     2. Deklaracja klasy: `class MCTSNN(Player):` → `class MCTSNN(MCTS):`.
     - Reszta pliku bez zmian — `__init__` z `super().__init__(args, color)` wywoła `MCTS.__init__` (zamiast `Player.__init__`), a `evaluate_board` spełni kontrakt abstrakcyjnej metody z `MCTS`.

   - **Efekt `super().__init__` po zmianie:**
     `MCTS.__init__` (mcts.py, linie 47–53) inicjalizuje:
     - `self.opening_book = OpeningBook(args, color)` — wymaga standardowych argumentów CLI; brak konfliktów.
     - `self.order_moves_mcts = OrderMovesMCTS(args, color)` — j.w.
     - `self.depth = args.depth_white / args.depth_black` — po wdrożeniu kroku 3 (eliminacja rolloutów) pole to stanie się martwe. Przed krokiem 3 jest używane jako limit rolloutów — poprawne zachowanie.
     - Po wdrożeniu kroku 7: `self.__root = None`, `self.__eval_cache = {}` — bez konfliktów.

   - **`BoardEvaluatorNN` — wymagania na argumenty CLI:**
     `BoardEvaluatorNN.__init__` (board_evaluator_nn.py, linie 13–24) odczytuje z `args`:
     - `args.depth_white_stockfish` / `args.depth_black_stockfish`
     - `args.skill_white` / `args.skill_black`
     - `args.stockfish_path`
     Te argumenty są specyficzne dla Stockfisha. Należy zweryfikować, czy `main.py` definiuje je w `argparse` — jeśli nie, `MCTSNN` crashuje przy tworzeniu `BoardEvaluatorNN`. Sprawdzono: `StockfishPlayer` i `MinimaxNN` również korzystają z tych argumentów, więc powinny być zdefiniowane w parserze. **Nie wymaga zmian.**

   - **Uwaga o nazewnictwie:**
     Nazwa `BoardEvaluatorNN` jest myląca — klasa nie korzysta z sieci neuronowej, lecz z silnika Stockfish jako ewaluatora. Nazwa `MCTSNN` sugeruje „MCTS z siecią neuronową", ale faktycznie oznacza „MCTS z ewaluacją Stockfisha". To jest problem nazewnictwa, nie funkcjonalności — nie wymaga zmiany w ramach tego planu optymalizacji, ale warto odnotować na przyszłość.

   - **⚠ Wpływ na krok 3 (eliminacja rolloutów):**
     Po naprawieniu dziedziczenia `MCTSNN` będzie korzystać z `MCTS.__simulate`. Eliminacja rolloutów (krok 3) wpłynie na `MCTSNN` — zamiast losowego rollout + ewaluacja Stockfishem, będzie bezpośrednia ewaluacja Stockfishem. Przy ~5–50ms na wywołanie Stockfisha, cache ewaluacji z kroku 6 staje się **krytyczny** dla wydajności `MCTSNN` (bez cache: ~400–4000 iteracji/20s; z cache i transpozycjami: potencjalnie kilkukrotnie więcej).

   - **⚠ Wpływ na krok 6 (cache ewaluacji):**
     Analiza kroku 6 już uwzględnia `BoardEvaluatorNN` jako przypadek, gdzie cache jest szczególnie korzystny. Po naprawieniu kroku 1 ta analiza staje się faktycznie obowiązująca (obecnie `MCTSNN` nie działa, więc cache dla NN był teoretyczny). Brak zmian wymaganych w treści kroku 6.

   - **Kolejność wdrożenia:**
     Krok 1 jest **niezależny** od kroków 2–7 i powinien być wdrożony **jako pierwszy** — jest to naprawa istniejącego buga, nie optymalizacja. Bez niego `MCTS_NN` w ogóle nie działa, co uniemożliwia testowanie kroków 2–7 z ewaluatorem Stockfisha.

2. ✅ **Przenieść `SIMULATION_TIME` do argumentów CLI**

   **Cel:** Zahardkodowana stała `SIMULATION_TIME = 20.0` (mcts.py, linia 67) kontroluje **całkowity budżet czasowy** pętli MCTS (`while time.perf_counter() < end_time`), czyli czas na wielokrotne wykonanie cyklu selekcja → ekspansja → ewaluacja → backpropagation. Wbrew nazwie sugerującej fazę „symulacji" (rollout), stała ta **nie jest powiązana z krokiem symulacji** eliminowanym w kroku 3 — ogranicza czas *całego* przeszukiwania drzewa dla danej pozycji. Po usunięciu rolloutów (krok 3) stała pozostaje potrzebna i pełni tę samą rolę.

   **Analiza techniczna:**

   - **Zmiana nazwy** — Po eliminacji rolloutów (krok 3) nazwa `SIMULATION_TIME` jest myląca. Zmienić na `SEARCH_TIME` lub `MCTS_TIME_BUDGET` — lepiej oddaje semantykę „czas przeszukiwania drzewa".

   - **Przeniesienie do CLI** — Dodać argument w `main.py` (`parse_args`), np. `--mcts-time-white` / `--mcts-time-black` (domyślnie `20.0`), analogicznie do istniejących par `--depth-white` / `--depth-black`. Odczyt w `MCTS.__init__`: `self.search_time: float = args.mcts_time_white if color == chess.WHITE else args.mcts_time_black`. Użycie w `__run_mcts`: `end_time = time.perf_counter() + self.search_time`.

   - **Osobne wartości dla białych i czarnych** — Umożliwia asymetryczne eksperymenty (np. MCTS z 10s vs MCTS z 30s), co jest przydatne do testowania siły gry w funkcji czasu.

3. ✅ **Zastąpić losowe rollouty ewaluacją statyczną**

   **Cel:** W `__simulate` w [mcts.py](engine/mcts.py) (linie 100–110) usunąć pętlę `random.choice()` i zwracać bezpośrednio `self.evaluate_board(node.board)` na nowo rozwiniętym węźle. To podejście AlphaZero ("MCTS bez rolloutów"), które opisano w pracy jako klucz do sukcesu w szachach.

   **Analiza techniczna:**

   - **Zmiana w `__simulate`** — Obecna metoda: (a) kopiuje planszę, (b) gra losowe ruchy do limitu `self.depth`, (c) ewaluuje wynikową pozycję. Po zmianie cała metoda redukuje się do jednolinijkowego `return self.evaluate_board(node.board)` — kopia planszy, pętla i `random.choice()` są zbędne.

   - **Obsługa pozycji terminalnych** — Obecny rollout naturalnie kończy się na pozycjach terminalnych (`is_game_over()`). Po usunięciu rolloutu konieczne jest jawne sprawdzenie stanu gry *przed* wywołaniem ewaluatora. Obie istniejące implementacje `evaluate_board` (`BoardEvaluatorTrad` linia 368–372 oraz `BoardEvaluatorNN`) już wewnętrznie obsługują mat/pat/remis, więc dodatkowa logika w `__simulate` nie jest wymagana. Warto jednak rozważyć, czy węzły terminalne powinny być w ogóle expandowane (optymalizacja w `__select`/`__expand`).

   - **Zakres wartości ewaluacji a formuła UCT** — `BoardEvaluatorTrad.evaluate_board()` zwraca wartości w zakresie (-∞, +∞): mat to `±math.inf`, typowa pozycja to ~±20 (w jednostkach wartości figur, np. pionek = 1.0). `BoardEvaluatorNN.evaluate_board()` zwraca centipawny Stockfisha / 100, czyli ~±15. Bezpośrednie użycie tych wartości w formule UCT (`value / visits + c * sqrt(...)`) powoduje, że człon eksploracji (~1.4 * sqrt(...) ≈ 0–5) jest zdominowany przez człon eksploatacji (~±20), co de facto wyłącza eksplorację. **To wymaga normalizacji (krok 4) jako ściśle powiązanej zmiany.**

   - **Wpływ na wydajność** — Bez rolloutu `evaluate_board` jest wywoływany raz na iterację MCTS (zamiast raz po `depth` losowych ruchach). Przy 20s budżecie czasu:
     - `BoardEvaluatorTrad`: pełna ewaluacja (materiał + PST + struktura pionów + bezpieczeństwo króla + mobilność) trwa ~0.1–0.5ms. Wąskie gardło to `__evaluate_mobility_and_activity` (iteruje po *wszystkich* `legal_moves` — O(n²) od liczby figur). Szacunkowo ~40k–200k iteracji na ruch. Eliminacja rolloutu *zwiększa* throughput, bo obecny rollout też wywołuje `evaluate_board` na końcu, ale dodatkowo wykonuje `depth` ruchów z kopiowaniem planszy.
     - `BoardEvaluatorNN` (Stockfish): ~5–50ms na wywołanie (zależnie od `depth_stockfish`). Przy 20s = ~400–4000 iteracji. To jest akceptowalne, ale marginalnie — warto rozważyć cache pozycji (transpozycje z kroku 6 będą tu podwójnie korzystne).

   - **Martwy kod po zmianie:**
     - `self.depth` w klasie `MCTS` (linia 53) — używany wyłącznie jako limit rolloutów. Po usunięciu rolloutów parametr staje się nieużywany w kontekście MCTS (nadal może być potrzebny w Minimax, ale to osobna hierarchia klas). Można go usunąć lub zachować na przyszłość (np. dla progressive deepening).
     - `import random` (linia 4) — staje się zbędny.
     - Kopia planszy `sim_board = node.board.copy(stack=False)` — zbędna, bo `evaluate_board` nie modyfikuje planszy.

   - **Kompatybilność z podklasami** — Zmiana dotyczy wyłącznie `MCTS.__simulate`, która jest prywatna (name-mangled). `MCTSTrad` i `MCTSNN` dziedziczą z `MCTS` i dostarczają jedynie `evaluate_board()` — nie nadpisują `__simulate`. Zmiana jest więc transparentna dla podklas. Uwaga: `MCTSNN` dziedziczy po `Player` (nie `MCTS`) — bug opisany w kroku 1 — więc ta zmiana go nie dotyczy dopóki krok 1 nie zostanie wdrożony. **Rekomendacja: wdrożyć krok 1 jako pierwszy (patrz analiza kolejności w kroku 1).**

4. ✅ **Normalizować wartość ewaluacji do [0, 1]**

   **Cel:** Sprowadzić wynik `evaluate_board` do stałego zakresu [0, 1], aby formuła UCT w `best_child()` poprawnie balansowała eksploatację i eksplorację.

   **Analiza techniczna:**

   - **Problem obecny** — `BoardEvaluatorTrad.evaluate_board()` zwraca wartości z perspektywy białych: dodatnie = przewaga białych, ujemne = przewaga czarnych. Zakres: mat = `±math.inf`, typowa pozycja ~±20 (pionek = 1.0, hetman = 9.0, plus PST/mobilność/etc.). `BoardEvaluatorNN` zwraca centipawny Stockfisha / 100, czyli ~±15. Formuła UCT w `best_child()` (linia 36): `value/visits + 1.4 * sqrt(2 * ln(N) / n_i)` — człon eksploracji daje wartości ~0–5, podczas gdy `value/visits` może wynosić ~±20. Eksploracja jest zdominowana.

   - **Wybór funkcji normalizacji** — Sigmoida `σ(x) = 1 / (1 + exp(-x/k))` z parametrem skalującym `k`:
     - Mapuje (-∞, +∞) → (0, 1), gdzie 0.5 = pozycja równa.
     - Mat (`±math.inf`) naturalnie mapuje się na 1.0 / 0.0 — bez specjalnego przypadku.
     - Remis (0.0) mapuje się na 0.5 — poprawna semantyka.
     - Parametr `k` kontroluje czułość: przy `k = 4.0` (dla skali gdzie pionek = 1.0) przewaga 1 pionka → ~0.56, przewaga hetmana (9.0) → ~0.90. Proponowana wartość w oryginalnym planie (`k = 400`) zakładała skalę centipawnową — tu skalą jest wartość figur, więc `k = 4.0` jest adekwatne dla `BoardEvaluatorTrad`, a `k = 4.0` również dla `BoardEvaluatorNN` (centipawny / 100 = ta sama skala).

   - **Gdzie umieścić normalizację** — Trzy opcje:
     1. **W `__simulate`** (rekomendowane) — normalizacja jako ostatni krok przed zwróceniem wartości. Izoluje logikę MCTS od ewaluatora; `evaluate_board` pozostaje uniwersalne (używane też przez Minimax). Implementacja: `raw = self.evaluate_board(node.board); v = 1 / (1 + math.exp(-raw / 4.0))`. Następnie konwersja perspektywy na potrzeby backpropagation (szczegóły w kroku 5): `return v if node.player == chess.BLACK else 1 - v`.
     2. W `evaluate_board` — naruszałoby kontrakt interfejsu `BoardEvaluator` i wymagało zmian w Minimax.
     3. W `best_child` — komplikowałoby formułę UCT i wymagało przechowywania surowych wartości.

   - **Obsługa `math.inf`** — `math.exp(-math.inf / 4.0)` = `math.exp(-inf)` = `0.0`, więc `1 / (1 + 0)` = `1.0`. `math.exp(math.inf / 4.0)` = `inf`, więc `1 / (1 + inf)` = `0.0`. Python obsługuje to poprawnie bez specjalnych przypadków.

   - **Stała `c_param`** — Po normalizacji `value/visits` mieści się w [0, 1]. Człon eksploracji `c * sqrt(2 * ln(N) / n_i)` przy `c = 1.4` i stosunku N/n_i = 100 daje ~1.4 * sqrt(2 * 4.6 / 1) ≈ 4.2 — nadal za dużo. Standardowa wartość `c = 1/sqrt(2) ≈ 0.707` (Kocsis & Szepesvári) lub nawet `c = sqrt(2) ≈ 1.414` działa, gdy w formule nie ma czynnika `2` pod pierwiastkiem. **Obecna formuła ma `2 *` w środku** (`sqrt(2 * log(N) / n_i)`), co jest niestandardowe — klasyczny UCT to `c * sqrt(ln(N) / n_i)`, a `c = sqrt(2)`. Obecna implementacja to `c * sqrt(2 * ln(N) / n_i)`, co jest równoważne `c' * sqrt(ln(N) / n_i)` z `c' = c * sqrt(2) = 1.4 * 1.414 ≈ 1.98`. Po normalizacji warto: albo (a) usunąć czynnik `2` i ustawić `c_param = sqrt(2)`, albo (b) zostawić formułę i dostroić `c_param` empirycznie (~0.5–1.0).

   **⚠ Wpływ na krok 5 (backpropagation):**
   Po normalizacji wartości do [0, 1] zmiana perspektywy gracza w backpropagation **nie może** używać negacji (`value = -value`), ponieważ `-0.8` jest poza zakresem [0, 1]. Zamiast tego należy użyć **komplementu**: `value = 1 - value`. Jeśli pozycja jest dobra dla jednego gracza (0.9), jest zła dla przeciwnika (0.1). To wymusza aktualizację planu kroku 5.

   **⚠ Wpływ na formułę UCT w `best_child()`:**
   Obecna formuła (linia 36) zawiera niestandardowy czynnik `2 *` pod pierwiastkiem. Wraz z normalizacją należy:
   - Zmienić formułę na standardową: `c * sqrt(ln(N) / n_i)` (usunąć `2 *`).
   - Ustawić domyślne `c_param = sqrt(2) ≈ 1.414` — standardowa wartość UCT dla znormalizowanych nagród w [0, 1].
   - Alternatywnie: zostawić czynnik `2 *` i obniżyć `c_param` do ~0.7 (matematycznie równoważne).
   Zmiana ta jest technicznie częścią kroku 4 (bez niej normalizacja nie przyniesie efektu).

5. ✅ **Naprawić backpropagation**

   **Cel:** Obecna logika (linie 112–119) dodaje/odejmuje surową wartość na podstawie globalnego `self.color`, zamiast odwracać perspektywę co poziom drzewa (jak w pseudokodzie z pracy: `result ← −result`). Po wdrożeniu normalizacji z kroku 4 (wartości w [0, 1]) odwrócenie perspektywy musi używać **komplementu** `value = 1 - value` (nie negacji `-value`).

   **Analiza techniczna:**

   - **Demonstracja buga w obecnej implementacji** — Przykład: silnik gra białymi (`self.color = WHITE`), `evaluate_board` zwraca `+5` (pozycja korzystna dla białych):
     - Backprop w węźle-dziecku (`node.player = BLACK`): warunek `BLACK != WHITE` → `node.value -= 5` → dziecko przechowuje `−5`.
     - Backprop w korzeniu (`node.player = WHITE`): warunek `WHITE == WHITE` → `node.value += 5` → korzeń przechowuje `+5`.
     - `best_child()` na korzeniu wybiera dziecko z **najwyższym** `child.value/child.visits = −5/1 = −5`. Im lepsza pozycja dla białych, tym niższa wartość dziecka → algorytm wybiera **najgorszy** ruch. **Bug potwierdzony.**
     - Przyczyna: wartość z perspektywy białych (`+5`) jest odejmowana w węźle czarnych, ale `best_child` zakłada, że `child.value` to wartość korzystna z perspektywy rodzica (białych). Konwencja jest niespójna.

   - **Wybór konwencji przechowywania wartości** — Istnieją dwie poprawne konwencje:

     **(A) Wartość z własnej perspektywy węzła** — każdy węzeł przechowuje sumę wartości z perspektywy swojego `node.player`. Backprop: `node.value += current_value; current_value = 1 - current_value`. Wymaga zmiany w `best_child()`: rodzic wybiera dziecko, które jest *najgorsze* dla przeciwnika, więc UCT staje się `(1 - child.value / child.visits) + c * sqrt(...)`.

     **(B) Wartość z perspektywy rodzica (rekomendowane)** — każdy węzeł przechowuje sumę wartości z perspektywy gracza, który *wybrał* ruch prowadzący do tego węzła (tj. `not node.player`). Backprop: `node.value += current_value; current_value = 1 - current_value`, ale wartość startowa jest odwrócona (patrz niżej). `best_child()` działa poprawnie bez zmian — rodzic wybiera dziecko z najwyższym `child.value/child.visits`, bo ta wartość jest już z perspektywy rodzica.

     **Rekomendacja: Konwencja (B)** — minimalizuje zmiany w kodzie (`best_child` pozostaje bez zmian), a semantyka `child.value/child.visits` = „win rate z perspektywy gracza, który wybrał ten ruch" jest intuicyjna i zgodna z literaturą.

   - **Przepływ perspektywy (konwencja B):**

     1. `__simulate` zwraca wartość `v ∈ [0, 1]` z perspektywy **białych** (po normalizacji sigmoidalnej z kroku 4, bo `evaluate_board` zawsze ewaluuje z perspektywy białych).
     2. Konwersja na perspektywę **gracza w ocenianym węźle** (`node.player`):
        - Jeśli `node.player == WHITE`: `v_node = v` (perspektywa zgodna).
        - Jeśli `node.player == BLACK`: `v_node = 1 - v` (odwrócenie).
     3. Konwersja na perspektywę **rodzica** (konwencja B): `v_parent = 1 - v_node`.
     4. Backpropagation od ocenianego węzła w górę:
        ```
        current = v_parent  # perspektywa rodzica liścia
        while node is not None:
            node.visits += 1
            node.value += current
            current = 1 - current  # odwrócenie na kolejny poziom
            node = node.parent
        ```
     5. Rezultat: liść przechowuje wartość z perspektywy swojego rodzica ✓, rodzic przechowuje wartość z perspektywy dziadka ✓, itd.

     **Uproszczenie:** Kroki 2–3 łącznie dają: jeśli `node.player == WHITE` → `v_parent = 1 - v`, jeśli `BLACK` → `v_parent = v`. Innymi słowy: `v_parent = v` gdy `node.player == BLACK` (bo rodzic jest biały), `v_parent = 1 - v` gdy `node.player == WHITE` (bo rodzic jest czarny). To eliminuje podwójną konwersję.

   - **Wpływ na `__simulate` (interakcja z krokiem 3 i 4):**
     Po wdrożeniu kroków 3–5 łącznie, metoda `__simulate` powinna:
     1. Wywołać `self.evaluate_board(node.board)` → surowa wartość z perspektywy białych.
     2. Znormalizować sigmoidą → `v ∈ [0, 1]` z perspektywy białych.
     3. Skonwertować na perspektywę rodzica: `return v if node.player == chess.BLACK else 1 - v`.
     Alternatywnie, krok 3 konwersji można przenieść do `__backpropagate` — ale umieszczenie go w `__simulate` daje czytelniejszy kontrakt: „`__simulate` zwraca wartość z perspektywy rodzica ocenianego węzła".

   - **Wpływ na `best_child()`:**
     Przy konwencji (B) formuła UCT **nie wymaga zmian** — `child.value / child.visits` jest już z perspektywy rodzica (gracza wybierającego), więc maksymalizacja jest poprawna.

   - **Wpływ na `most_visited_child()`:**
     Bez zmian — wybiera dziecko z największą liczbą odwiedzin, niezależnie od konwencji wartości.

   - **Pole `node.player`:**
     Nadal potrzebne — jest używane w konwersji perspektywy w backpropagation lub `__simulate`. Nie można go usunąć.

   - **Pole `self.color` w `MCTS`:**
     Obecne użycie `self.color` w `__backpropagate` (linia 115) zostaje **całkowicie usunięte**. `self.color` w klasie `MCTS` nadal jest używane w logowaniu (linia 81) i w `OpeningBook`/`OrderMovesMCTS`, więc pole samo w sobie pozostaje.

 6. ✅ **Dodać transposition table**

   **Cel:** Obecnie identyczne pozycje osiągane różnymi ścieżkami ruchów są traktowane jako osobne węzły w drzewie MCTS, co marnuje budżet iteracji na redundantne ewaluacje.

   **Analiza techniczna:**

   - **Dwa podejścia — cache ewaluacji vs współdzielenie węzłów (DAG):**

     **(A) Cache ewaluacji (rekomendowane)** — Słownik `{zobrist_hash → float}` przechowujący wyniki `evaluate_board`. Sprawdzany w `__simulate` przed wywołaniem ewaluatora. Jeśli pozycja jest w cache, zwraca zapisaną wartość (po normalizacji/konwersji perspektywy z kroków 4–5). Węzły w drzewie pozostają niezależne — struktura drzewa nie zmienia się.
       - **Zalety:** Prostota; brak zmian w `MCTSNode`, `__backpropagate`, `__select`, `__expand`. Kompatybilny z krokiem 7 (tree reuse — cache może przetrwać między ruchami). Szczególnie korzystny dla `BoardEvaluatorNN` (Stockfish ~5–50ms/call → cache eliminuje powtórne wywołania).
       - **Wady:** Nie eliminuje redundancji strukturalnej (dwa węzły dla tej samej pozycji nadal istnieją w drzewie, każdy z własnymi statystykami `visits`/`value`). Statystyki nie są współdzielone.

     **(B) Współdzielenie węzłów (DAG)** — Słownik `{zobrist_hash → MCTSNode}`. Przy `__expand`, jeśli pozycja istnieje w tablicy, zamiast tworzyć nowy węzeł, dołącza istniejący jako dziecko. Drzewo staje się skierowanym grafem acyklicznym (DAG).
       - **Problem krytyczny: `node.parent`** — Obecna struktura `MCTSNode` ma pole `self.parent` (jeden rodzic). Backpropagation (`__backpropagate`) idzie w górę po `node.parent`. Przy współdzieleniu węzeł ma **wiele rodziców** — backprop zaktualizowałby tylko jedną ścieżkę, pozostawiając pozostałe nieaktualne. Rozwiązania:
         - (B1) Zamienić `self.parent` na `self.parents: List[MCTSNode]` i propagować po wszystkich ścieżkach — złożoność backprop rośnie z O(głębokość) do O(liczba ścieżek), potencjalnie eksponencjalnie.
         - (B2) Przy `__expand` nie współdzielić samego węzła, lecz **kopiować statystyki** (`visits`, `value`) z istniejącego węzła do nowo tworzonego — unika problemu DAG, ale statystyki szybko się rozjeżdżają po dalszych iteracjach.
       - **Problem: interakcja z krokiem 7 (tree reuse)** — Przy DAG „przycięcie" drzewa do poddrzewa wymaga przetwarzania grafu zamiast prostego odnalezienia dziecka.
       - **Wniosek:** DAG-MCTS jest rzadko stosowany w praktyce (nawet AlphaZero/Leela Chess Zero nie używają transpozycji w drzewie MCTS). Złożoność implementacji nie jest proporcjonalna do zysku.

     **Rekomendacja: Podejście (A) — cache ewaluacji.**

   - **Implementacja cache ewaluacji:**

     1. **Haszowanie** — `chess.polyglot.zobrist_hash(board)` z biblioteki `python-chess` zwraca 64-bitowy int uwzględniający pozycje figur, kolor na ruchu, prawa roszady i en passant. Zweryfikowano dostępność: `import chess.polyglot; chess.polyglot.zobrist_hash(board)` → `int`.

     2. **Lokalizacja cache** — Pole `self.__eval_cache: dict[int, float]` w klasie `MCTS`. Inicjalizowane w `__run_mcts` (jeśli bez kroku 7) lub w `__init__` (jeśli z krokiem 7 — cache przetrwa między ruchami).

     3. **Użycie w `__simulate`** — Po wdrożeniu kroków 3–5:
        ```python
        def __simulate(self, node: MCTSNode) -> float:
            board_hash = chess.polyglot.zobrist_hash(node.board)
            if board_hash in self.__eval_cache:
                v = self.__eval_cache[board_hash]
            else:
                raw = self.evaluate_board(node.board)
                v = 1 / (1 + math.exp(-raw / 4.0))
                self.__eval_cache[board_hash] = v
            return v if node.player == chess.BLACK else 1 - v
        ```
        Uwaga: cache przechowuje wartość **znormalizowaną z perspektywy białych** (`v`), a konwersja perspektywy (`v` vs `1 - v`) zachodzi po odczycie z cache — to gwarantuje poprawność niezależnie od tego, kto jest na ruchu (bo Zobrist hash i tak uwzględnia `board.turn`, więc ta sama pozycja z różnym kolorem na ruchu to różne hashe).

     4. **Rozmiar cache i zarządzanie pamięcią** — Przy ~100k iteracji/ruch i typowym współczynniku trafień transpozycji ~5–15% w szachach, cache osiągnie ~85–95k wpisów na ruch. Każdy wpis to `int → float` ≈ ~72 bajty (narzut dict Pythona). Przy jednorazowym użyciu (bez kroku 7): ~100k wpisów ≈ ~7 MB — pomijalny. **Przy reuse drzewa (krok 7):** cache rośnie przez całą grę (~100k/ruch × ~40 ruchów = ~4M wpisów ≈ ~300 MB). Wymaga ograniczenia: limit rozmiaru dict (~500k wpisów) z czyszczeniem najstarszych wpisów, lub czyszczenie całego cache co N ruchów.

     5. **Kolizje hashów** — Zobrist hash 64-bit daje prawdopodobieństwo kolizji ~1/2⁶⁴ na parę pozycji. Przy 100k wpisów, prawdopodobieństwo jakiejkolwiek kolizji ≈ 100k² / 2⁶⁵ ≈ 5·10⁻¹⁰ — pomijalny. Nie wymaga weryfikacji FEN.

   - **⚠ Wpływ na krok 7 (tree reuse):**
     Jeśli cache jest polem instancji (`self.__eval_cache`), naturalnie przetrwa między ruchami. Warto dodać opcjonalne czyszczenie wpisów starszych niż N ruchów (pozycje z wcześniejszych faz gry są mało prawdopodobne do ponownego odwiedzenia), ale nie jest to konieczne ze względu na niski koszt pamięci.

   - **⚠ Wpływ na krok 3:**
     Cache wzmacnia korzyść z eliminacji rolloutów — bez rolloutu ewaluacja jest wywoływana na *tej samej* pozycji co węzeł (nie na losowej pozycji po rollout), więc trafienia w cache są znacznie częstsze (ta sama pozycja może być odwiedzana wielokrotnie w `__select` → `__expand` z różnych ścieżek).

7. ✅ **Reuse drzewa między ruchami**

   **Cel:** W `__run_mcts` (linia 68) drzewo budowane jest od zera (`root = MCTSNode(board)`) przy każdym wywołaniu `make_move`. Zachowanie drzewa między ruchami pozwala ponownie wykorzystać dotychczasowe statystyki `visits`/`value`, oszczędzając budżet iteracji.

   **Analiza techniczna:**

   - **Przepływ gry w `Engine.__run()`** (engine.py, linie 84–91) — Pętla gry wywołuje naprzemiennie `white_player.make_move(board)` i `black_player.make_move(board)` na **wspólnym** obiekcie `board`. Między dwoma kolejnymi wywołaniami `make_move` tego samego gracza na planszy zostały wykonane **dwa ruchy**: (1) własny ruch silnika (wybrany przez MCTS) oraz (2) odpowiedź przeciwnika. Oba ruchy są dostępne w `board.move_stack`.

   - **Nawigacja do poddrzewa** — Po zakończeniu tury silnik zna swój ruch (`best_child.move`). Przy kolejnym wywołaniu `make_move` musi:
     1. Odczytać **dwa ostatnie ruchy** z `board.move_stack`: przedostatni = własny ruch, ostatni = ruch przeciwnika.
     2. W zachowanym drzewie (`self.__root`) znaleźć dziecko odpowiadające własnemu ruchowi → to jest `best_child` z poprzedniej tury (można zachować referencję bezpośrednio).
     3. W tym dziecku znaleźć wnuka odpowiadającego ruchowi przeciwnika: `next(c for c in best_child.children if c.move == opponent_move)`.
     4. Ustawić wnuka jako nowy korzeń: `self.__root = grandchild; self.__root.parent = None`.

   - **Fallback — ruch przeciwnika nieodwiedzony:**
     Jeśli ruch przeciwnika nie był eksplorowany (brak odpowiedniego dziecka w drzewie), np. bo MCTS nie zdążył rozwinąć tego wariantu, drzewo jest bezużyteczne → fallback do tworzenia nowego korzenia: `self.__root = MCTSNode(board)`. W praktyce, jeśli MCTS z 20s budżetem w pełni rozwinie korzeń (~30–40 legalnych ruchów), a `best_child` też został intensywnie odwiedzony, to większość odpowiedzi przeciwnika powinna być w drzewie. Pesymistyczny scenariusz: przeciwnik gra rzadki ruch → stracony zysk z reuse jest pomijalny (bo to i tak mało odwiedzony wariant).

   - **Interakcja z opening book:**
     `make_move` (linia 59–64) może zakończyć się wcześniej przez opening book bez wywołania `__run_mcts`. W takim przypadku drzewo nie zostało zbudowane → `self.__root` powinno pozostać `None`. Przy następnym `make_move`, jeśli `self.__root is None`, tworzy nowy korzeń. Dodatkowo, jeśli opening book jest aktywny, ale nie znalazł ruchu (`make_move` przechodzi do `__run_mcts`), a `self.__root` istnieje z poprzedniej tury — nawigacja poddrzewa działa normalnie.

   - **Nowe pola instancji w `MCTS.__init__`:**
     ```python
     self.__root: Optional[MCTSNode] = None
     self.__last_best_child: Optional[MCTSNode] = None
     ```

   - **Zmiana w `__run_mcts`:**
     ```python
     def __run_mcts(self, board: chess.Board, start_time: float) -> None:
         SIMULATION_TIME = 20.0
         root = self.__get_or_create_root(board)
         # ... pętla MCTS bez zmian ...
         if root.children:
             best_child = root.most_visited_child()
             board.push(best_child.move)
             self.__root = root
             self.__last_best_child = best_child
             # ... logowanie ...
     ```

   - **Nowa metoda `__get_or_create_root`:**
     ```python
     def __get_or_create_root(self, board: chess.Board) -> MCTSNode:
         if self.__root is not None and self.__last_best_child is not None:
             # Szukaj ruchu przeciwnika wśród dzieci last_best_child
             opponent_move = board.move_stack[-1] if board.move_stack else None
             if opponent_move:
                 for child in self.__last_best_child.children:
                     if child.move == opponent_move:
                         child.parent = None  # odetnij od starego drzewa
                         self.__root = None
                         self.__last_best_child = None
                         return child
         # Fallback: nowy korzeń
         self.__root = None
         self.__last_best_child = None
         return MCTSNode(board)
     ```

   - **Zarządzanie pamięcią (GC):**
     `MCTSNode` ma cykliczne referencje (`parent ↔ children`). Po odcięciu poddrzewa (`child.parent = None`) stary korzeń i niepotrzebne gałęzie tracą referencje z `self.__root`/`self.__last_best_child` (ustawiane na `None`). CPython ma cykliczny garbage collector, który obsłuży referencje `parent ↔ children` w odciętych gałęziach. Koszt GC jest pomijalny. Alternatywnie, jawne czyszczenie: `old_root.children = []; old_root.parent = None` — ale nie jest konieczne.

   - **Uwaga: `board.copy(stack=False)` w `MCTSNode.__init__`:**
     Obecna implementacja kopiuje planszę bez stosu ruchów (`stack=False`). To oznacza, że `node.board.move_stack` jest puste — nie można z niego odczytać historii. To nie jest problem, bo do nawigacji poddrzewa używamy `node.move` (ruch prowadzący do węzła) i `board.move_stack` z **głównego** obiektu `board` przekazywanego do `make_move`. Argument `board` w `make_move` to oryginalny obiekt z `Engine` z pełnym stosem ruchów.

   - **Szacowany zysk:**
     Typowe drzewo MCTS po 20s budżecie ma ~100k+ iteracji. Po reuse, poddrzewo zachowuje statystyki ~30–70% oryginalnego drzewa (reszta to obcięte gałęzie alternatywnych ruchów silnika). Przeciwnikowa odpowiedź zachowuje ~5–30% (zależnie od tego, jak równomiernie MCTS eksplorował odpowiedzi). Efektywny zysk: ekwiwalent ~5–30k „darmowych" iteracji na starcie — porównywalne z 1–6s dodatkowego czasu obliczeniowego.

   - **⚠ Wpływ na krok 6 (cache ewaluacji):**
     Cache ewaluacji powinien być polem instancji (inicjalizowanym w `__init__`, nie w `__run_mcts`), aby przetrwał między ruchami razem z drzewem. To jest spójne z rekomendacją z kroku 6. Rozmiar cache będzie rósł przez całą grę (~100k wpisów/ruch × ~40 ruchów = ~4M wpisów ≈ ~300 MB). **To wymaga ograniczenia rozmiaru** — dodać LRU eviction lub czyścić cache przy każdym reuse (zachowując tylko wpisy odwiedzalne z nowego korzenia). Prostsza opcja: czyścić cały cache co N ruchów lub ustawić limit np. 500k wpisów.

### Odrzucone optymalizacje

1. **PUCT zamiast UCT** — Odrzucone. Wymagałoby wytrenowania własnej sieci neuronowej (policy network) do generowania priorów `P(s,a)`. Trenowanie jest kosztowne i niepraktyczne w ramach tego projektu; korzystamy z ewaluacji Stockfisha.

2. **Progressive widening** — Odrzucone. Ograniczenie liczby dzieci węzła do `k · N^α` (Couëtoux et al., 2011; Chaslot et al., 2008) daje marginalny zysk (~0.035% budżetu iteracji), wymaga strojenia dwóch hiperparametrów bez formalnego uzasadnienia dla szachów i ryzykuje pominięcie cichych ruchów taktycznych.

### Dodatkowe poprawki (po przeglądzie kodu)

8. ✅ **Obsługa węzłów terminalnych — unikanie bezproduktywnych iteracji**

   **Cel:** Węzły terminalne (mat/pat/remis) mają `untried_moves = []` i `children = []`. Gdy `__select` do nich dotrze, pętla while kończy się natychmiast (`is_fully_expanded() and children` → `True and []` = falsy). Węzeł wraca do `__run_mcts`, skip expand (brak untried), `__simulate` zwraca wartość z cache (instant), backprop aktualizuje statystyki. Jeśli UCT faworyzuje tę ścieżkę (np. mat = 1.0), tysiące iteracji trafiają w ten sam węzeł bez eksplorowania czegokolwiek nowego.

   **Analiza techniczna:**

   - **Skala problemu** — W końcówkach, gdzie warianty matowe są blisko korzenia, algorytm może tracić >50% budżetu iteracji na powtórne odwiedzanie tych samych terminali. Przy 20s budżecie i ~100k iteracji, to ~50k zmarnowanych iteracji.

   - **Rozwiązanie — pole `is_terminal` w `MCTSNode`:**
     1. Dodać `self.is_terminal: bool = board.is_game_over()` w `MCTSNode.__init__`.
     2. W `__select`: dodać warunek wyjścia — jeśli `node.is_terminal`, zwróć węzeł natychmiast (nie schodź głębiej, nie próbuj expandować).
     3. W `__run_mcts`: jeśli `node.is_terminal` i `node.visits > 0`, pomiń `__simulate` i `__backpropagate` — węzeł był już ewaluowany, dalsze aktualizacje nie wnoszą informacji.

   - **Alternatywne rozwiązanie (prostsze):** W `__run_mcts`, po `__select`, sprawdzić `if not node.untried_moves and not node.children` (terminal leaf) i kontynuować (`continue`) — nie symuluj ani nie propaguj ponownie. To jest mniej inwazyjne, ale nie eliminuje kosztu selekcji (nadal schodzimy do terminala).

   - **Rekomendacja:** Pole `is_terminal` + guard w `__select` i `__run_mcts`. Minimalny koszt pamięci (1 bool/węzeł), maksymalna oszczędność iteracji.

   - **⚠ Wpływ na tree reuse (krok 7):** Żaden — pole `is_terminal` jest immutable (pozycja terminalna nie zmienia się). Przy reuse zachowane węzły zachowują swój status.

9. ✅ **Reset stanu drzewa po opening book**

   **Cel:** Gdy `make_move` (linia 65–70) kończy się ruchem z opening book (`return` na linii 69), pola `__root` i `__last_best_child` **nie są resetowane**. Stare drzewo MCTS wisi w pamięci i przy następnym wywołaniu `__get_or_create_root` algorytm próbuje nawigować w nieaktualnym drzewie. Fallback zadziała (ruch przeciwnika nie pasuje → nowy korzeń), ale:
   - Stare drzewo zajmuje pamięć do czasu fallbacku.
   - Kod jest nieintuicyjny — polega na efekcie ubocznym fallbacku zamiast jawnego resetu.

   **Zmiana:** Dodać reset na początku gałęzi opening book w `make_move`:
   ```python
   if self.opening_book.make_move(board, start_time):
       self.__root = None
       self.__last_best_child = None
       return
   ```

10. ✅ **Poprawić logowanie — dynamiczna nazwa klasy**

    **Cel:** Linie 88 i 93 logują hardkodowane `'MCTS-TRAD'`, niezależnie od tego, czy gra `MCTSTrad` (ewaluator tradycyjny) czy `MCTSNN` (ewaluator Stockfish). Po naprawieniu kroku 1 obie podklasy korzystają z tej samej metody `__run_mcts`, więc log jest mylący.

    **Zmiana:** Zamienić `'MCTS-TRAD'` na `type(self).__name__` (lub `self.__class__.__name__`). Wynik: `MCTSTrad` lub `MCTSNN` w logach — jednoznacznie identyfikuje wariant.

11. ✅ **Dodać `__slots__` do `MCTSNode`**

    **Cel:** Przy setkach tysięcy tworzonych węzłów, `__slots__` eliminuje `__dict__` per instancja — zmniejsza zużycie pamięci ~30–40% i przyspiesza dostęp do atrybutów.

    **Zmiana:** Dodać `__slots__` z listą wszystkich pól (`board`, `parent`, `move`, `children`, `untried_moves`, `visits`, `value`, `player`, `is_terminal`, `_moves_sorted`).

12. ✅ **Wyeliminować podwójne kopiowanie planszy w `__expand`**

    **Cel:** `__expand` kopiuje planszę (`node.board.copy(stack=False)`), a następnie `MCTSNode.__init__` kopiuje ją ponownie (`board.copy(stack=False)`). Każde kopiowanie to ~0.01–0.05ms — przy ~100k iteracji to ~2–10s zmarnowanego czasu.

    **Zmiana:** Dodać parametr `_copy: bool = True` do `MCTSNode.__init__`. W `__expand` przekazywać `_copy=False` (plansza jest już świeżą kopią). Zewnętrzne wywołania (np. `MCTSNode(board)` w `__get_or_create_root`) nadal kopiują domyślnie.

13. ✅ **Sortować untried_moves raz zamiast przy każdej ekspansji**

    **Cel:** `order_moves_mcts.order_moves()` było wywoływane przy każdym `__expand`, re-sortując malejącą listę untried_moves. Plansza w węźle się nie zmienia, więc kolejność ruchów jest deterministyczna — wystarczy sortować raz. Przy n legalnych ruchach: stary koszt O(n² log n) łącznie → nowy koszt O(n log n) jednorazowo.

    **Zmiana:** Pole `_moves_sorted: bool` w `MCTSNode`. W `__expand`: sortuj i odwróć (`reverse()`) przy pierwszym wywołaniu, potem `pop()` O(1) z końca listy (najlepszy ruch). Eliminuje też `list.remove()` O(n).

14. ✅ **Dodać licznik iteracji do logów**

    **Cel:** Log MCTS zawierał czas, wybrany ruch i liczbę odwiedzin najlepszego dziecka, ale nie całkowitą liczbę iteracji MCTS. Ta metryka jest kluczowa do analizy wydajności (throughput iteracji/s).

    **Zmiana:** Zmienna `iterations` inkrementowana w pętli MCTS, logowana w `LOGGER.info`.

15. ✅ **Naprawić obsługę mata w `BoardEvaluatorNN`**

    **Cel:** Stockfish zwraca `{'type': 'mate', 'value': N}` dla pozycji z forsownym matem, gdzie `N` to liczba ruchów do mata (nie centipawny). Kod traktował `N` jak centipawny: mat w 1 → `1/100 = 0.01` → po sigmoidzie ~0.5 (pozycja „równa"). Powinno być `±math.inf` → po sigmoidzie 1.0/0.0.

    **Zmiana:** W `board_evaluator_nn.py` sprawdzać `evaluation['type']`: jeśli `'mate'`, zwracać `math.inf` (wygrywający) lub `-math.inf` (przegrywający).

16. ✅ **Zoptymalizować `best_child` — single-pass z precomputed `log`**

    **Cel:** `best_child()` tworzyło list comprehension (1 przejście), potem szukało max + index (2 przejście), i obliczało `math.log(self.visits)` redundantnie dla każdego dziecka. Przy ~30 dzieciach × ~100k wywołań = ~3M zbędnych `log()` i 2x przejścia po liście.

    **Zmiana:** Precompute `log_parent_visits = math.log(self.visits)`, single-pass `for` loop z `best_score`/`best` tracking. Inicjalizacja od `self.children[0]` eliminuje `Optional` return type.

17. ✅ **Zmniejszyć częstotliwość `time.perf_counter()` w pętli MCTS**

    **Cel:** `time.perf_counter()` wywoływane co iterację = ~200k syscalli na ruch. Sprawdzanie co 128 iteracji (`iterations & 127 == 0`) redukuje do ~800 wywołań. Overshoot max ~0.05s przy 20s budżecie — pomijalny.

    **Zmiana:** `while True` z `break` co 128 iteracji zamiast `while time.perf_counter() < end_time`.

