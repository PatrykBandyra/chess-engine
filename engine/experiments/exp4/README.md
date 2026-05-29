# Eksperyment 4 — Dokładność heurystyki TRAD vs Stockfish + benchmark szybkości

## Cel

Wyizolowanie wkładu **funkcji oceny** od algorytmu przeszukiwania, niezależnie od rozgrywek (oś B). Eksperyment 4 mierzy jakość ewaluatora **bezpośrednio na zestawie pozycji testowych**, porównując z **referencją** (Stockfish d=20).

**Ważna decyzja metodologiczna:** `BoardEvaluatorNN` w tym projekcie jest wrapperem na `Stockfish(depth=10)`, nie siecią neuronową. Porównywanie tego "NN" z `Stockfish(depth=20)` ground truth byłoby auto-skorelowane (ten sam silnik, różne głębokości). Dlatego **NN został wykluczony z 4a i 4b** (gdzie ground truth = SF d=20), ale **pozostaje w 4c** — speed benchmark jest fair (porównanie czasu wykonania, brak ground truth).

Eksperyment odpowiada na trzy pytania:

1. **Dokładność TRAD (4a):** Jak dobrze heurystyka TRAD szacuje wartość pozycji w porównaniu z Stockfishem d=20? Czy lepiej niż 1-ply Stockfish (baseline)?
2. **Zgodność ruchów TRAD (4b):** Jak często MINIMAX_TRAD / MCTS_TRAD wybierają ten sam ruch co Stockfish d=20 top-3?
3. **Szybkość TRAD vs NN (4c):** Ile mikrosekund zajmuje pojedyncze wywołanie `evaluate_board()` dla TRAD (Python heurystyka) vs NN (subprocess Stockfisha)?

## Komponenty (3 sub-eksperymenty)

| # | Co mierzy | Wejście | Ewaluatory / Warianty | Wyjście |
|---|-----------|---------|----------------------|---------|
| 4a | Korelacja ewaluacji z ground truth | 200 pozycji | TRAD, SF-d1 vs SF-d20 | Spearman ρ, MAE, RMSE per ewaluator+faza |
| 4b | Zgodność wybranego ruchu z SF top-3 | 200 pozycji × 2 warianty | MINIMAX_TRAD d=3, MCTS_TRAD 2.61s (oba spojne z Exp 1) | match rate, top-3 match rate per wariant+faza |
| 4c | Czas wywołania `evaluate_board()` | 10000 pozycji | TRAD, NN (pure timing) | Median/mean/p95 latency per ewaluator |

## Pozycje testowe (200, stratyfikowane)

Zestaw 200 pozycji generowany przez `prepare_test_positions.py`:

| Faza | Liczba | Próg `__get_game_phase()` | Charakterystyka |
|------|--------|---------------------------|------------------|
| Otwarcie | 50 | phase > 0.8 | Pełna obsada figur, ~20+ punktów materiału lekkiego |
| Gra środkowa | 100 | 0.3 ≤ phase ≤ 0.8 | Wymieniony pojedynczy materiał, ~10-20 pkt |
| Końcówka | 50 | phase < 0.3 | <10 pkt materiału lekkiego, często bez damy |

**Generacja:** Z każdego z 25 ECO openings rozgrywa się losowe sekwencje ruchów, samplując pozycje aż osiągnięte zostaną docelowe liczby per faza. Deduplikacja po FEN.

**Plik wyjściowy:** `engine/experiments/exp4/test_positions.fen` — jedna pozycja FEN per linia.

## Konfiguracja domyślna

| Parametr | Wartość | Komentarz |
|----------|---------|-----------|
| `--ground-truth-depth` | 20 | Stockfish d=20 jako oracle (4a, 4b) |
| `--nn-depth` | 10 | Głębokość Stockfisha w ewaluatorze NN (tylko 4c) |
| `--minimax-depth` | 3 | Głębokość MINIMAX_TRAD w 4b (spojne z Exp 1) |
| `--mcts-time` | 2.61s | Czas/ruch MCTS_TRAD w 4b (skalibrowany z Exp 1 — spojnosc) |
| `--speed-n` | 10000 | Liczba wywołań `evaluate_board()` w 4c |
| `-ExperimentTag` | `<timestamp yyyyMMdd_HHmmss>` | Tag katalogu wyjsciowego — np. `d3_mcts2.61_run1`. Pozwala miec wiele wynikow obok siebie. |

## Uzasadnienie wyboru `nn-depth = 10`

Wartosc 10 dla Stockfish depth w `BoardEvaluatorNN` jest **pragmatycznym wyborem** opartym na:

1. **Spojnosc z innymi eksperymentami:** Exp 1 uzywa NN evaluator z d=10 (default w `board_evaluator_nn.py` gdy `-dws/-dbs` nie ustawione), Exp 5 uzywa `StockfishDepth=10` jako player benchmark. Ten sam parametr w calym projekcie.

2. **Empiryczny kompromis szybkosc/dokladnosc:**

   | SF depth | Czas/call | MINIMAX_NN d=3 czas/ruch | Komentarz |
   |----------|-----------|---------------------------|-----------|
   | d=5      | ~10ms     | ~0.3s                     | Zbyt plytkie — ewaluator slabszy niz heurystyka TRAD |
   | **d=10** | **~100-300ms** | **~6s** (zmierzone w Exp 1) | Sensowny baseline dla real-time eval |
   | d=15     | ~1-3s     | ~30-60s                   | Graniczne czasowo, niepraktyczne dla turniejow |
   | d=20     | ~5-15s    | ~150-450s                 | Tylko offline analiza (4a/4b ground truth) |

3. **Komentarz w `engine/board_evaluator_nn.py:27-28`:**
   > "Single-threaded Stockfish is only ~10-20% slower for shallow evaluations (depth 10-15) but eliminates context switching."

   Zakres 10-15 jest uznany przez autora za "shallow" (do wielokrotnego wywolania w Minimaxie).

4. **Limitation (do omowienia w pracy mgr):** Brak formalnej kalibracji (np. Spearman ρ jako funkcja depth). Mozna w przyszlosci rozszerzyc Exp 4 o accuracy-vs-depth benchmark (depth ∈ {5, 8, 10, 12, 15}), ale to **out of scope** dla obecnej pracy.

**Sugerowane sformulowanie dla pracy mgr:**
> "Stockfish depth 10 was chosen for the NN evaluator as a pragmatic balance between evaluation accuracy and computational cost (~100-300ms per call, allowing ~6s per move at Minimax depth 3 — feasible for real-time tournament play). Formal accuracy calibration against ground truth (varying depth) is left as future work."

## Procedura uruchomienia

### Krok 0 — Przygotowanie pozycji (jednorazowo, ~30 sek)

```powershell
.\experiments\exp4\run_exp4.ps1
```

Pierwsze uruchomienie automatycznie wywołuje `prepare_test_positions.py`. Jeśli plik `test_positions.fen` już istnieje, ten krok jest pomijany.

Ręczne wygenerowanie:
```bash
python experiments/exp4/prepare_test_positions.py --seed 42
```

### Krok 1 — Pełen eksperyment (~1.75h)

```powershell
.\experiments\exp4\run_exp4.ps1
```

Wykonuje sekwencyjnie 4 fazy:
- **Faza 0:** przygotowanie pozycji (jeśli brak)
- **Faza 1 (4a):** ~14 min — dokładność ewaluacji (TRAD, SF-d1)
- **Faza 2 (4b):** ~30 min — zgodność ruchu (MINIMAX_TRAD d=3, MCTS_TRAD 2.61s — oba spojne z Exp 1)
- **Faza 3 (4c):** ~60 min — benchmark szybkości (TRAD vs NN)

### Tryby alternatywne

```powershell
# Pominąć przygotowanie (gdy test_positions.fen istnieje):
.\experiments\exp4\run_exp4.ps1 -SkipPrep

# Tylko 4a (najszybsza faza, dla quick checku):
.\experiments\exp4\run_exp4.ps1 -SkipMoveAgreement -SkipSpeed

# Pominąć powolne 4b:
.\experiments\exp4\run_exp4.ps1 -SkipMoveAgreement

# Tryb testowy — 20 pozycji zamiast 200:
.\experiments\exp4\run_exp4.ps1 -Limit 20

# Mniejszy benchmark szybkości:
.\experiments\exp4\run_exp4.ps1 -SpeedN 1000

# Z explicit tag (multi-run support):
.\experiments\exp4\run_exp4.ps1 -ExperimentTag 'd3_mcts2.61_run1'
```

**Uwaga:** Exp 4 jest **single-threaded** — nie ma parallelizacji wewnątrz. Każda faza działa w pojedynczym procesie Python. Można jednak uruchomić Exp 4 **równolegle z grającymi eksperymentami** (Exp 2/3/5) — Exp 4 to głównie wywołania Stockfisha (zewnętrzny proces), więc nie konkuruje agresywnie o rdzenie z grającym silnikiem.

## Wyjście — pliki CSV

Wszystkie pliki lądują w `engine/out/exp4_eval_<tag>/` (spojnie z Exp 1/2/3). `<tag>` to wartość `-ExperimentTag` lub timestamp `yyyyMMdd_HHmmss` jeśli nie podany. **Test positions (input) pozostaja w `engine/experiments/exp4/test_positions.fen`** — to plik generowany jednorazowo przez `prepare_test_positions.py`, wspolny dla wszystkich run.

### Metadata (reproducibility)

| Plik | Zawartość |
|------|-----------|
| `_config.json` | Metadane run: experiment_tag, timestamp, args (wszystkie CLI params), Stockfish version, git commit, Python version, platform/arch |

### 4a — Dokładność ewaluacji

| Plik | Zawartość |
|------|-----------|
| `exp4a_evaluations.csv` | Per-pozycja: FEN, faza, eval TRAD, eval SF-d1, eval SF-d20 |
| `exp4a_accuracy_summary.csv` | Per (ewaluator, faza): Spearman ρ, MAE, RMSE, count |
| `exp4a_accuracy_summary.txt` | Human-readable tabela |

### 4b — Zgodność ruchu

| Plik | Zawartość |
|------|-----------|
| `exp4b_moves.csv` | Per (pozycja, wariant): wybrany ruch, czas, czy w SF top-1, czy w SF top-3 |
| `exp4b_move_agreement.csv` | Per wariant (MINIMAX_TRAD, MCTS_TRAD): match rate, top-3 match rate, stratified by phase |

### 4c — Szybkość

| Plik | Zawartość |
|------|-----------|
| `exp4c_speed_results.csv` | Per-call timing dla TRAD i NN |
| `exp4c_speed_summary.txt` | Median/mean/p50/p95/p99 latency per ewaluator |

## Wyjście — wykresy (`plots/`)

### 4a
| Plik | Co pokazuje |
|------|-------------|
| `exp4a_scatter_trad.png` | Scatter plot: TRAD eval vs SF-d20 eval (z linią regresji) |
| `exp4a_scatter_sf_d1.png` | Scatter plot: SF-d1 eval vs SF-d20 eval (baseline) |
| `exp4a_mae_by_phase.png` | Bar chart: MAE per ewaluator × faza |

### 4b
| Plik | Co pokazuje |
|------|-------------|
| `exp4b_match_rate.png` | Bar chart: % match rate (top-1 i top-3) per wariant (2 warianty) |

### 4c
| Plik | Co pokazuje |
|------|-------------|
| `exp4c_speed_box.png` | Box plot: rozkład czasów wywołań TRAD vs NN, log scale |
| `exp4c_speed_hist.png` | Histogram: rozkład czasów per ewaluator |

## Analizy statystyczne

### Dokładność (4a)
- **Spearman rank correlation ρ** — miara zachowania rankingu ewaluacji (bez wrażliwości na skalowanie). Wartość 1.0 = ewaluator idealnie odzwierciedla relatywną wartość pozycji.
- **MAE (Mean Absolute Error)** — w pionkach. Niższe = lepszy.
- **RMSE (Root Mean Squared Error)** — penalizuje duże błędy. Wskazuje czy ewaluator ma katastrofalne błędy w niektórych pozycjach.
- **TRAD vs SF-d1 (kluczowe porównanie):** Czy heurystyka TRAD jest lepsza niż 1-ply Stockfish baseline? Jeśli TAK — heurystyka warta utrzymania. Jeśli NIE — niepotrzebna złożoność.
- **Stratyfikacja po fazach** — czy TRAD jest tak samo dokładny w otwarciu jak w końcówce?

### Zgodność ruchu (4b)
- **Top-1 match rate** — procent pozycji gdzie wariant TRAD wybrał DOKŁADNIE ten sam ruch co SF top-1
- **Top-3 match rate** — procent gdzie wybrany ruch jest wśród SF top-3
- **MINIMAX_TRAD vs MCTS_TRAD (kluczowe porównanie):** Który algorytm lepiej znajduje "najlepsze" ruchy przy tej samej heurystyce TRAD? To **fair** porównanie osi A (algorytm) z izolowaną zmienną.
- **Stratyfikacja po fazach** — w której fazie wariant najlepiej dopasowuje się do "best play"

### Szybkość (4c)
- **Median / p50 / p95 / p99** — dystrybucja czasu wywołania. Mediana ważniejsza niż mean (odporna na outliery).
- **Throughput** — ile wywołań/sek można wykonać. Krytyczne dla MCTS (gdzie eval to bottleneck).
- **Speedup TRAD vs NN** — typowo 50-200× szybszy

## Co z planu badawczego jest istotne dla pracy magisterskiej

Eksperyment 4 dostarcza **niezależnych** od rozgrywek informacji o:

1. **Czy heurystyka TRAD jest sensownym przybliżeniem oceny pozycji?** (4a vs SF-d20 i vs SF-d1 baseline)
2. **Czy algorytm przeszukiwania wpływa na jakość wybieranych ruchów przy stałym ewaluatorze?** (4b: MINIMAX_TRAD vs MCTS_TRAD)
3. **Jaki jest koszt obliczeniowy ewaluatorów?** (4c — TRAD vs NN throughput, niezależne od ground truth)
4. **Czy speedup TRAD vs NN tłumaczy różnice w throughput Minimax/MCTS z Exp 1?** (porównanie 4c z czasami ruchu)

W pracy zazwyczaj prezentuje się:

- **Tabela:** Spearman ρ, MAE, RMSE TRAD i SF-d1 × faza (z `exp4a_accuracy_summary.csv`)
- **Wykres scatter:** TRAD vs SF-d20 (`exp4a_scatter_trad.png`) — wizualizacja korelacji + bias (offset)
- **Bar chart:** match rate per wariant (`exp4b_match_rate.png`) — porównanie MINIMAX_TRAD vs MCTS_TRAD
- **Wykres pudełkowy:** latency TRAD vs NN (`exp4c_speed_box.png`) — kompromis szybkość/dokładność (TRAD jest "wystarczająco dokładny" ale ~100× szybszy)
- **Tabela:** kompromis dokładność/szybkość (ile ms "kosztuje" 0.1 wzrostu Spearmana?)

## Szacunek czasu

| Faza | Czas |
|------|------|
| 0 — przygotowanie pozycji | ~30 sek (jednorazowo) |
| 4a — dokładność | ~14 min |
| 4b — zgodność ruchu | ~30 min (MINIMAX d=3, MCTS 2.61s — oba spojne z Exp 1) |
| 4c — szybkość | ~60 min |
| **TOTAL** | **~1.75h** |

**Uwaga:** Exp 4 jest single-threaded; parallelizacja wewnątrz **nie jest możliwa**. Można jednak uruchomić Exp 4 równolegle z innymi eksperymentami (Exp 2/3/5/6/7), bo Stockfish jako external proces nie konkuruje silnie o rdzenie z silnikiem Pythona.

## Ważne uwagi praktyczne

1. **Stockfish musi być dostępny** w `STOCKFISH_PATH` (z `engine/constants.py`). Sprawdź wcześniej: `echo "uci\nquit" | <STOCKFISH_PATH>`.
2. **Brak gry** — Exp 4 nie rozgrywa partii, więc nie zapisuje plików `metrics_*.jsonl`. Wszystkie wyniki to CSV/PNG w `engine/out/exp4_eval_<tag>/`.
3. **Multi-run support** — dzięki `-ExperimentTag` można uruchamiać wiele wariantów bez nadpisywania (np. `-ExperimentTag d3_only` i `-ExperimentTag d4_compare`). Domyślnie tag = timestamp.
4. **Reproducibility** — każde uruchomienie generuje `_config.json` w output dir z wszystkimi parametrami, git commitem, wersją Stockfisha, wersją Pythona i platformą. Pozwala odtworzyć dokładne warunki run.
3. **Quick smoke test:** `pwsh ./experiments/exp4/run_exp4.ps1 -Limit 20 -SpeedN 1000` (~3 min, sprawdza pipeline)
4. **Ground truth depth = 20** — Stockfish d=20 zazwyczaj uznawany jest za "near-perfect" w pozycjach niespecjalnych. Wzrost do d=25 nie zmieni wyniku istotnie, ale podwaja czas.
5. **NN depth = 10** — używany TYLKO w 4c (speed benchmark). NN nie występuje w 4a ani 4b, ponieważ porównywanie SF-d10 z SF-d20 ground truth jest auto-skorelowane.
6. **Faza 4b jest najdłuższa** — bo każdy z 2 wariantów silnika musi wybrać ruch dla każdej z 200 pozycji = 400 wywołań silnika.
7. **Wyłączenie faz** — jeśli czas krytyczny, można pominąć fazy ` -SkipMoveAgreement` (~27 min mniej).

## Co do dyskusji w pracy

- **Niezależna ocena jakości heurystyki TRAD** — Exp 4a daje twardą liczbę (Spearman ρ, MAE) jak dobrze TRAD szacuje pozycję. Można porównać z innymi heurystykami z literatury.
- **TRAD vs SF-d1 (kluczowe)** — jeśli TRAD ≥ SF-d1, heurystyka jest **uzasadniona**. Jeśli TRAD < SF-d1, można argumentować że "nawet 1-ply Stockfish jest lepszą heurystyką" — ważny wniosek metodologiczny.
- **Stratyfikacja po fazach** — typowo TRAD jest dokładny w otwarciu/grze środkowej, ale słaby w końcówce. Może prowadzić do **postulatu** w pracy: "TRAD wymaga dodania endgame-specific terms (np. tablebase lookups) by konkurować z silnikami opartymi o Stockfish".
- **Algorytm przy stałym ewaluatorze (4b: MINIMAX_TRAD d=3 vs MCTS_TRAD 2.61s)** — to **najczystszy** dostępny test osi A (algorytm) niezależnie od ewaluatora. **Obie strony w 4b uzywaja parametrow z Exp 1** (MINIMAX_TRAD d=3 i MCTS_TRAD 2.61s) — match rate w 4b można bezpośrednio porównywać z Elo z Exp 1 (te same wersje silnikow).
- **Speed/accuracy trade-off (4c)** — NN (subprocess Stockfisha) jest ~50-200× wolniejszy niż TRAD (czysty Python). To najmocniejszy argument za TRAD w budżetowych engine'ach: nawet jeśli dokładność jest niższa, throughput jest dramatycznie wyższy.
- **Powiązanie z Exp 1:** Jeśli MINIMAX_NN wygrał w Exp 1, to **wiemy z konstrukcji** dlaczego — bo używa głębszego Stockfisha do ewaluacji. Exp 4 nie testuje tego porównania (byłoby biased). Jednak **4c throughput** tłumaczy ograniczenia szybkości MINIMAX_NN — z Stockfishem jako ewaluatorem nie można iść tak głęboko jak z TRAD przy tym samym budżecie czasowym.

## Notatka o przyszłości

W pracy magisterskiej warto byłoby pokazać tabelę **konsolidującą wyniki Exp 1 i Exp 4**:

| Ewaluator / Wariant | Spearman ρ (4a) | Match rate (4b) | Elo Exp 1 | Czas/eval (4c) |
|---------------------|------------------|------------------|-----------|----------------|
| TRAD (heurystyka Python) | ? | n/a (per-wariant) | TRAD aggregate | ~5-50 μs |
| SF-d1 (baseline) | ? | n/a | n/a | ~0.5-2 ms |
| NN (Stockfish d=10) | *excluded* | *excluded* | NN aggregate | ~1-5 ms |
| MINIMAX_TRAD_d3 | n/a | ? | from Exp 1 | n/a |
| MCTS_TRAD | n/a | ? | from Exp 1 | n/a |

*NN excluded z 4a/4b: porównanie z SF-d20 ground truth byłoby auto-skorelowane (ten sam silnik na różnych głębokościach). Zachowane w 4c jako uczciwy benchmark szybkości.*

### Co Exp 4 **rzeczywiście** dodaje do pracy

1. **4a — niezależna ocena heurystyki TRAD:** Spearman ρ, MAE, RMSE vs SF-d20. Plus baseline SF-d1.
2. **4b — porównanie algorytmów przy stałym ewaluatorze:** MINIMAX_TRAD vs MCTS_TRAD na 200 pozycjach (clean test osi A).
3. **4c — throughput benchmark:** TRAD ~100× szybszy niż NN, co bezpośrednio tłumaczy ograniczenia głębokości MINIMAX_NN i throughput MCTS_NN.
4. **Stratyfikacja po fazach:** Identyfikuje w której fazie gry TRAD heurystyka zawodzi (zwykle endgame).
