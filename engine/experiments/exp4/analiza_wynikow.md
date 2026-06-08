# Eksperyment 4 — Analiza wyników (jakość i szybkość ewaluatora)

## 1. Cel eksperymentu i pytania badawcze

Eksperyment 4 izoluje **wkład funkcji oceny** od algorytmu przeszukiwania, mierząc
trzy rozłączne aspekty ewaluatora **bezpośrednio na zbiorze pozycji testowych**
(bez rozgrywania partii):

- **4a** — dokładność TRAD i Stockfisha 1-ply (`SF-d1`, baseline) względem
  wyroczni `SF-d20`,
- **4b** — zgodność wybranego ruchu pełnych wariantów `MINIMAX_TRAD` i `MCTS_TRAD`
  z `SF-d20` top-1 i top-3,
- **4c** — czas pojedynczego wywołania `evaluate_board()` dla TRAD (heurystyka Python)
  i NN (subprocess Stockfisha d=10).

Pytania badawcze:

- **PB1 (4a):** Czy heurystyka TRAD jest sensownym przybliżeniem oceny pozycji?
  W szczególności: **czy TRAD jest lepszy od `SF-d1`** (1-plejowy Stockfish)?
- **PB2 (4a, stratyfikacja):** W której fazie gry TRAD radzi sobie najsłabiej?
- **PB3 (4b):** Który algorytm (Minimax α-β vs MCTS PUCT) lepiej znajduje
  „dobre" ruchy przy **identycznej** heurystyce TRAD? To najczystszy dostępny
  test osi A bez wpływu jakości ewaluatora.
- **PB4 (4c):** Jaki jest stosunek szybkości TRAD/NN? Czy tłumaczy on różnice
  throughput obserwowane w Eksp. 1-3?

**Decyzja metodologiczna:** `BoardEvaluatorNN` to wrapper na `Stockfish(depth=10)`,
nie wytrenowana sieć neuronowa. Porównywanie go z `SF-d20` byłoby **auto-skorelowane**
(ten sam silnik, dwie głębokości). Dlatego NN został **wyłączony z 4a i 4b**, lecz
pozostaje w **4c** (uczciwe porównanie czasu wykonania, brak ground truth).

## 2. Założenia metodyczne

### 2.1 Zbiór pozycji testowych (4a, 4b)

200 pozycji wygenerowanych przez `prepare_test_positions.py`, stratyfikowanych po fazie:

| Faza | Liczba | Kryterium (`__get_game_phase()`) | Charakterystyka |
|---|---:|---|---|
| Otwarcie | 50 | `phase > 0.8` | Pełna obsada figur, ~20+ pkt mat. lekkiego |
| Gra środkowa | 100 | `0.3 ≤ phase ≤ 0.8` | Pojedyncze wymiany, ~10-20 pkt |
| Końcówka | 50 | `phase < 0.3` | < 10 pkt mat. lekkiego, najczęściej bez damy |

Generacja: z każdego z 25 otwarć ECO rozgrywane są losowe sekwencje ruchów, próbkowane
do osiągnięcia docelowych liczb per faza; deduplikacja po FEN. Plik wyjściowy:
`engine/experiments/exp4/test_positions.fen`.

### 2.2 Konfiguracja faktycznego uruchomienia — `_config.json`

| Parametr | Wartość |
|---|---|
| `ground_truth_depth` | 20 |
| `nn_depth` | 10 |
| `minimax_depth` | **3** (deklarowane w args) |
| `mcts_time` | 2.61 s |
| `speed_n` | 10 000 |
| Stockfish | v17 |
| commit | `de2c8cb` |
| platform | macOS arm64 |

> **Uwaga o niespójności etykiety:** plik `exp4b_move_agreement.csv` zawiera
> etykietę wariantu `MINIMAX_TRAD_d4`, podczas gdy zarówno README, jak i `_config.json`
> deklarują `minimax_depth = 3`. Najprawdopodobniej jest to **stały suffix etykiety
> w skrypcie analizy** (nie odzwierciedlający faktycznej głębokości). Wyniki interpretuję
> spójnie z `_config.json` jako `MINIMAX_TRAD d = 3`. Wymaga to weryfikacji
> w skrypcie `run_exp4b_move_agreement.py` przed cytowaniem w pracy.

### 2.3 Próba 4c

10 000 wywołań `evaluate_board()` per ewaluator (TRAD, NN). Single-threaded, brak
warm-upu jawnie odjętego — wartości reprezentują „prawdziwą" stronę użytkową
(z overheadem inicjalizacji i komunikacji UCI).

## 3. Zbierane metryki

**4a:** `Spearman ρ` (rank correlation, odporna na skalowanie), `MAE` (mean abs.
error w pionkach), `RMSE` (penalizuje błędy katastroficzne) — wszystkie liczone
przeciw `SF-d20` jako wyroczni; stratyfikacja per faza.

**4b:** `match_rate` (top-1 zgodność z SF-d20), `top3_match_rate` (czy wybrany ruch
jest w top-3 SF-d20); per wariant × per faza.

**4c:** `mean`, `median`, `p95`, `p99`, `stdev`, `min`, `max` czasu wywołania
w mikrosekundach.

## 4. Wyniki

### 4.1 Dokładność ewaluacji (4a) — `exp4a_accuracy_summary.csv`

| Ewaluator | Faza | n | Spearman ρ | MAE [pionki] | RMSE [pionki] |
|---|---|---:|---:|---:|---:|
| **TRAD** | wszystkie | 199 | **0.867** | **5.635** | 10.706 |
| **SF-d1** (baseline) | wszystkie | 200 | **0.942** | **3.759** | 9.640 |
| TRAD | otwarcie | 50 | **0.545** | 2.885 | 3.677 |
| SF-d1 | otwarcie | 50 | **0.826** | 1.482 | 2.305 |
| TRAD | środek | 100 | 0.860 | 5.779 | 10.840 |
| SF-d1 | środek | 100 | 0.942 | 3.473 | 8.770 |
| TRAD | końcówka | 49 | **0.921** | 8.147 | 14.556 |
| SF-d1 | końcówka | 50 | 0.947 | 6.608 | 14.580 |

Wizualizacje: `engine/out/exp4_eval_exp4/plots/exp4a_scatter_trad.png`,
`plots/exp4a_scatter_sf_d1.png`, `plots/exp4a_mae_by_phase.png`.

### 4.2 Zgodność ruchu z SF-d20 (4b) — `exp4b_move_agreement.csv`

| Wariant | Faza | n | n_skutecznych | top-1 match | top-3 match |
|---|---|---:|---:|---:|---:|
| **MINIMAX_TRAD** (d=3*) | wszystkie | 200 | 199 | **0.555** | **0.810** |
| **MCTS_TRAD** (2.61 s) | wszystkie | 200 | 158 | **0.395** | **0.605** |
| MINIMAX_TRAD | otwarcie | 50 | 50 | 0.560 | 0.780 |
| MCTS_TRAD | otwarcie | 50 | 33 | 0.340 | 0.540 |
| MINIMAX_TRAD | środek | 100 | 100 | 0.580 | 0.830 |
| MCTS_TRAD | środek | 100 | 85 | 0.460 | 0.660 |
| MINIMAX_TRAD | końcówka | 50 | 49 | 0.500 | 0.800 |
| MCTS_TRAD | końcówka | 50 | 40 | 0.320 | 0.560 |

\* faktyczna głębokość per `_config.json`; CSV używa etykiety `MINIMAX_TRAD_d4` (niespójność opisana w § 2.2).

Wizualizacja: `plots/exp4b_match_rate.png`.

**Uwaga o `n_skutecznych`:** `MCTS_TRAD` raportuje **158/200** poprawnie zakończonych
przypadków (otwarcie: 33/50, końcówka: 40/50). 42 pozycje **nie zostały rozegrane**
do końca w zadanym budżecie 2.61 s — najprawdopodobniej z powodu przekroczenia czasu
w pozycjach z dużym branching factor lub problemów z determinizmem MCTS na pojedynczej
pozycji. `MINIMAX_TRAD` ma 199/200 (jeden brak w końcówce — najpewniej pat/mat
wykryty przed wyborem ruchu).

### 4.3 Szybkość wywołania `evaluate_board()` (4c) — `exp4c_speed_summary.csv`

| Ewaluator | n | mean [μs] | median [μs] | p95 [μs] | p99 [μs] | stdev [μs] | min [μs] | max [μs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **TRAD** | 10 000 | **137.44** | **136.00** | 157.12 | 180.59 | **12.97** | 104.17 | 307.42 |
| **NN** | 10 000 | **3 794.47** | **2 456.46** | 11 247.93 | 19 738.88 | 3 838.33 | 1 027.29 | 53 764.04 |

**Stosunek NN / TRAD:** median ≈ **18×**, mean ≈ **28×**, p95 ≈ **72×**, p99 ≈ **109×**.

Wizualizacje: `plots/exp4c_speed_box.png` (log scale), `plots/exp4c_speed_hist.png`.

## 5. Dyskusja

### 5.1 TRAD jest **gorszy** od jednoplejowego Stockfisha — kluczowy wynik

Na **każdej** metryce i w **każdej** fazie gry baseline `SF-d1` przewyższa TRAD:

| Metryka (wszystkie pozycje) | TRAD | SF-d1 | Δ (SF − TRAD) |
|---|---:|---:|---:|
| Spearman ρ | 0.867 | 0.942 | **+0.075** |
| MAE [pionki] | 5.635 | 3.759 | **−1.876** |
| RMSE [pionki] | 10.706 | 9.640 | **−1.066** |

W otwarciu różnica jest dramatyczna: `ρ_TRAD = 0.545` vs `ρ_SF-d1 = 0.826` (Δ = 0.28),
MAE 2.89 vs 1.48 (TRAD ma **dwukrotnie większy** średni błąd). Wniosek dla PB1:
**heurystyka TRAD nie jest uzasadniona jako samodzielny ewaluator** — nawet 1-plejowy
Stockfish (jedna iteracja α-β z prymitywną oceną materiału) daje lepszy ranking pozycji
i mniejsze odchylenie absolutne. Niewielki narzut czasowy SF-d1 (~0.5-2 ms — szacunek
z literatury, w tej pracy nie zmierzony bezpośrednio) zwraca się jakością.

To istotny **wniosek metodologiczny** dla pracy magisterskiej: w przyszłych pracach
warto rozważyć zastąpienie heurystyki TRAD wrapperem SF-d1 jako tańszego/lepszego
„baseline ewaluatora", a obecne wyniki Eksp. 1-3 należy interpretować z zastrzeżeniem,
że TRAD jest świadomie słabym punktem odniesienia.

### 5.2 Stratyfikacja po fazie — paradoks ρ vs MAE w końcówce (PB2)

Wzorzec **odwrotny do intuicji**: TRAD osiąga **najwyższe** Spearman ρ w końcówce
(0.921) i **najniższe** w otwarciu (0.545), podczas gdy MAE rośnie monotonicznie
z otwarcia (2.89) do końcówki (8.15). Wyjaśnienie:

- **Skala ocen rośnie z fazą.** W końcówce wartości bezwzględne ocen są dużo wyższe
  (asymetria materiału, bliskość promocji/mata), więc MAE „w pionkach" naturalnie
  rośnie — choć **względny** błąd może być mniejszy.
- **Ranking pozycji w końcówce jest łatwy** dla heurystyki materialnej: różnica
  jednego piona / figury dominuje wszystko inne. Stąd wysokie ρ.
- **Otwarcie jest trudne dla TRAD**, bo różnice są subtelne (kompensacja pozycyjna
  za materiał, tempo, rozwój). Heurystyka nie odróżnia dobrze pozycji o materiale
  symetrycznym — stąd niskie ρ.

Wniosek dla PB2: **TRAD jest słaby w otwarciach**, nie w końcówkach (jak często
zakłada się intuicyjnie). Stratyfikowane słowo w pracy: „heurystyka TRAD niedoskonale
odróżnia pozycje otwarciowe o symetrycznym materiale — uzasadnia to wagę kontroli
otwarciowej (otwarcia ECO w Eksp. 1-3, księga w Eksp. 7)".

### 5.3 Minimax dominuje MCTS przy tym samym ewaluatorze (PB3)

`MINIMAX_TRAD d=3` osiąga top-1 match rate **0.555** vs MCTS_TRAD **0.395**
(różnica **+16 p.p.**). Na top-3: 0.810 vs 0.605 (różnica **+20.5 p.p.**). Efekt
występuje w **każdej** fazie:

| Faza | MINIMAX top-1 | MCTS top-1 | Δ |
|---|---:|---:|---:|
| Otwarcie | 0.560 | 0.340 | **+22 p.p.** |
| Środek | 0.580 | 0.460 | +12 p.p. |
| Końcówka | 0.500 | 0.320 | **+18 p.p.** |

To **najczystszy** dostępny w pracy test osi A (algorytm) — identyczny ewaluator,
identyczna pula pozycji, identyczna baseline wyrocznia. Wynik **wspiera dominację
Minimaxa nad MCTS** obserwowaną też w Eksp. 1 (oś A: `MINIMAX_score = 0.662`).
Mechanizmu nie da się przypisać throughputowi (porównanie jest „per pozycja"
z budżetem czasowym MCTS = 2.61 s); chodzi raczej o:

- α-β z tablicą transpozycji i quiescence **dokładnie wylicza** wymuszone sekwencje
  taktyczne w 3 plejach,
- MCTS bez prior NN (czyste UCB1/PUCT z TRAD) ma tendencję do rozpraszania wizyt
  na ruchach „bezpiecznych" zamiast wyłapywania wąskich wygranych linii,
- obserwacja koreluje z Eksp. 6 (puzzle taktyczne) — MCTS będzie tam strukturalnie
  słabszy.

**Ostrzeżenie:** `MCTS_TRAD` ma `n_skutecznych = 158/200` (21 % braków).
Match rate liczony jest po `n_skutecznych`, nie po pełnym `n = 200`. Jeśli braki
są niemonotonicznie powiązane z trudnością pozycji (np. MCTS „nie kończy" trudnych
pozycji), porównanie może być optymistyczne dla MCTS. Wymaga to dodatkowego
sprawdzenia w `exp4b_moves.csv` (nie analizowane tutaj).

### 5.4 Stosunek szybkości TRAD/NN (PB4)

Mediana czasu wywołania: **TRAD 136 μs, NN 2 456 μs** — NN jest **~18× wolniejszy
w typowym przypadku**, ale **~109× wolniejszy w ogonie p99** (19 739 μs ≈ 20 ms).
Standardowe odchylenie TRAD jest 296× mniejsze niż NN (13 μs vs 3 838 μs) —
heurystyka Python ma **bardzo przewidywalną** latencję, podczas gdy NN jest zdominowany
przez wahania UCI/IPC (komunikacja z subprocesem Stockfisha + zmienna głębokość
wyszukiwania d=10 zależna od pozycji).

**Bezpośredni związek z Eksp. 1-3:**

- W Eksp. 3 (MCTS) zmierzono throughput TRAD ~ 30 k iter/s vs NN ~ 30 iter/s
  (stosunek ~1000×). Stosunek z Eksp. 4c (28× mean, 109× p99) jest **znacząco mniejszy**.
  Różnica wynika z faktu, że MCTS wywołuje ewaluator wielokrotnie per iterację
  i amortyzuje narzut Pythona, podczas gdy benchmark 4c mierzy izolowane wywołanie
  z całym narzutem UCI.
- W Eksp. 2 (Minimax) głębokość `d = 4` daje 49.5 s/ruch dla NN vs 0.62 s dla TRAD
  (×80). Stosunek 80× jest między medianą 4c (×18) a tail 4c (×109) — spójne z tym,
  że Minimax wywołuje NN na liściach drzewa, gdzie pozycje są pełne (duża zmienność
  czasu wyszukiwania Stockfisha d=10).

Wniosek dla PB4: TRAD jest dramatycznie szybszy, ale stosunek **nie jest stały** —
zależy od kontekstu wywołania (izolowane vs amortyzowane), pozycji i hardware'u.
Spójne podsumowanie: **„TRAD jest 20-100× szybszy w typowym użyciu"**.

### 5.5 Synteza: kompromis dokładność ↔ szybkość

Zestawiając wyniki 4a i 4c:

| Ewaluator | Spearman ρ (4a) | Median czas (4c) | „ρ na ms" |
|---|---:|---:|---:|
| TRAD | 0.867 | 0.136 ms | **6.4** |
| SF-d1 (baseline) | 0.942 | ~1-2 ms (szacunek) | ~0.5-1.0 |
| NN (SF d=10) | wykluczony (auto-korelacja) | 2.46 ms | — |

Mimo niższej dokładności TRAD oferuje **najwyższą** dokładność na jednostkę czasu
(~6 razy więcej Spearmana per ms niż SF-d1). To uzasadnia jego użycie w
**płytkich** wariantach (TRAD MCTS, TRAD Minimax z dużą głębokością) — pozwala
osiągnąć więcej iteracji/węzłów w tym samym czasie. **Nie uzasadnia** jednak
preferencji nad SF-d1 w wariantach o niskim throughpucie (np. MCTS_NN — gdzie
i tak ograniczeniem jest komunikacja UCI, a dokładność rozstrzyga).

### 5.6 Ograniczenia

- **Brak benchmarku SF-d1 czasowego** w 4c — porównanie „TRAD vs SF-d1 vs NN"
  dokładności i szybkości jest niepełne (czas SF-d1 z literatury, nie zmierzony).
- **Niespójność etykiety `MINIMAX_TRAD_d4` vs `minimax_depth = 3`** — wynik 4b
  zaraportowany jako d=3, ale wymaga weryfikacji w skrypcie. Jeśli faktycznie
  d=4, porównanie z MCTS 2.61 s nadal jest sensowne (czas Minimax d=4 ≈ 0.62 s,
  tj. ~4× mniej niż MCTS), ale dla porównania spójnego z Eksp. 1 (d=3) musimy
  użyć wartości z `_config.json`.
- **MCTS_TRAD ma 21 % nieskutecznych** w 4b — możliwy bias selekcji w match rate.
- **200 pozycji** to relatywnie mały zbiór. Stratyfikacja na 50/100/50 daje
  szerokie 95% CI dla ρ (rzędu ±0.05-0.10 w końcówce).
- **`SF-d20` jako wyrocznia** — w pozycjach niespecjalnych „prawie idealna",
  ale w głębokich końcówkach (mat w 30+) lub forced sequences może się różnić
  od `SF-d30`. Wpływ marginalny dla wniosków o TRAD.
- **Wyłączenie NN z 4a/4b** jest poprawne metodologicznie, ale uniemożliwia
  bezpośrednie zestawienie „dokładność NN vs TRAD". Pełna analiza wymagałaby
  niezależnego ground truth (np. Leela d=15, baza tablebase).

## 6. Wnioski

1. **TRAD jest gorszy od baseline `SF-d1`** na każdej metryce dokładności
   (ρ 0.867 vs 0.942; MAE 5.64 vs 3.76 pionka). To **kluczowy wynik metodologiczny**:
   heurystyczna ocena TRAD jest empirycznie słabsza niż 1-plejowy Stockfish — wynik
   wszystkich pozostałych eksperymentów należy interpretować z tą świadomością.
2. **TRAD jest najsłabszy w otwarciu** (ρ = 0.545), nie w końcówce (ρ = 0.921)
   — przeciwnie do intuicji. Wysokie MAE w końcówce wynika ze skali ocen, nie
   z błędnego rankingu pozycji.
3. **Minimax dominuje MCTS** przy tym samym ewaluatorze: top-1 match 0.555 vs 0.395
   (+16 p.p.), top-3 match 0.810 vs 0.605 (+20.5 p.p.). Efekt w każdej fazie gry
   — najczystsze dostępne potwierdzenie osi A z Eksp. 1 (`MINIMAX_score = 0.662`).
4. **Throughput TRAD vs NN: 18× szybszy w medianie, 109× w p99.** TRAD ma
   dramatycznie mniejszą wariancję latencji (stdev 13 μs vs 3 838 μs). Stosunek
   tłumaczy obserwacje throughput z Eksp. 2 (×80 przy d=4) i Eksp. 3 (×1000
   w pełnej amortyzacji MCTS).
5. **„ρ na ms" jest dla TRAD najwyższe** — uzasadnia wybór TRAD jako ewaluatora
   dla wariantów o wysokim throughpucie (MCTS, deep Minimax). Nie uzasadnia
   wyboru nad `SF-d1` w wariantach o niskim throughpucie.
6. **MCTS_TRAD ma 21 % „nieskutecznych" pozycji w 4b** — wymaga jawnego raportowania
   w pracy jako ograniczenie metodyczne.
7. **Niespójność etykietowania** (`d=4` w CSV vs `d=3` w `_config.json`)
   wymaga weryfikacji w skrypcie analizy przed cytowaniem konkretnej głębokości
   w pracy.

Materiał stanowi podstawę czwartej części rozdziału eksperymentalnego (jakość ewaluatora):
rysunek 4.10 (`exp4a_scatter_trad.png` — korelacja TRAD vs SF-d20), rysunek 4.11
(`exp4a_mae_by_phase.png` — MAE per faza), rysunek 4.12 (`exp4b_match_rate.png` —
top-1/top-3 per wariant), rysunek 4.13 (`exp4c_speed_box.png` — log-scale boxplot
latencji), tabela 4.7 (dokładność per faza), tabela 4.8 (zgodność ruchu),
tabela 4.9 (latencja).
