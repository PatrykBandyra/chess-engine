# Eksperyment 3 — Analiza wyników (skalowanie budżetu czasowego MCTS)

## 1. Cel eksperymentu i pytanie badawcze

Eksperyment ma odpowiedzieć na pytanie: **jak siła gry MCTS PUCT rośnie wraz z budżetem
czasowym na ruch i jak wygląda krzywa `Elo ≈ α · log₂(t) + β` dla obu ewaluatorów
(TRAD i NN)?** Wynik ma posłużyć do walidacji budżetu `t = 2.61 s` użytego w Eksp. 1
(kalibracja MCTS) oraz do porównania efektywności zasobowej MCTS (Eksp. 3) z efektywnością
głębokościową Minimaxa (Eksp. 2).

Hipotezy:

- **H1:** Elo rośnie monotonicznie z czasem dla obu ewaluatorów; dopasowanie log-liniowe
  ma dawać `R² > 0.9` w mierzonym zakresie.
- **H2:** Parametr `α` („Elo na podwojenie czasu") mieści się w przedziale typowym
  dla MCTS — **50-150 Elo/doubling**.
- **H3:** NN ma drastycznie niższy throughput iteracji (Stockfish d=10 jako wyrocznia
  liścia), ale dzięki lepszej jakości oceny powinien dawać przynajmniej porównywalną
  stopę zwrotu Elo per sekundę zegara.
- **H4:** Entropia rozkładu wizyt korzenia (`root_visit_entropy`) maleje z budżetem —
  większa pewność wyboru ruchu.
- **H5** (sanity check): mecz `t = 20s vs t = 20s` (oba ewaluatory) ma dawać wynik ≈ 0.5.
- **H6** (walidacja kalibracji z Eksp. 1): ekstrapolacja `Elo(t = 2.61s)` powinna
  leżeć między punktami `t = 1s` a `t = 5s`.

## 2. Założenia metodyczne

Eksperyment to **seria pojedynków MCTS o budżecie testowanym vs MCTS o budżecie
referencyjnym `t = 20 s`**, dla każdego z 5 testowanych czasów `t ∈ {1, 5, 10, 20, 40} s`
i każdego z 2 ewaluatorów (TRAD, NN). **10 matchupów × 30 partii = 300 partii.**

| # | Matchup | Ewaluator | `t_white` | `t_black` (anchor) |
|---|---|---|---:|---:|
| 1 | `mcts_trad_1s_vs_20s` | TRAD | 1 s | 20 s |
| 2 | `mcts_trad_5s_vs_20s` | TRAD | 5 s | 20 s |
| 3 | `mcts_trad_10s_vs_20s` | TRAD | 10 s | 20 s |
| 4 | `mcts_trad_20s_vs_20s` | TRAD | 20 s | 20 s |
| 5 | `mcts_trad_40s_vs_20s` | TRAD | 40 s | 20 s |
| 6 | `mcts_nn_1s_vs_20s` | NN | 1 s | 20 s |
| 7 | `mcts_nn_5s_vs_20s` | NN | 5 s | 20 s |
| 8 | `mcts_nn_10s_vs_20s` | NN | 10 s | 20 s |
| 9 | `mcts_nn_20s_vs_20s` | NN | 20 s | 20 s |
| 10 | `mcts_nn_40s_vs_20s` | NN | 40 s | 20 s |

**Wybór `t = 20 s` jako kotwicy** — środek geometrycznej siatki budżetów; ~8× kalibrowany
czas z Eksp. 1 (2.61 s); kompromis między pewnością wyboru a kosztem wall-clock.

**Geometryczna siatka** `{1, 5, 10, 20, 40} s` (log₂ ≈ 0, 2.3, 3.3, 4.3, 5.3) umożliwia
**dopasowanie log-liniowe** `Elo ≈ α · log₂(t) + β`, gdzie `α` interpretujemy jako
„Elo na podwojenie czasu".

**Otwarcia, adjudykacja, książka:** identycznie jak w Eksp. 1 i 2 — 25 pozycji ECO,
adjudykacja ±0.05/20 ruchów (kluczowa dla MCTS-vs-MCTS), brak książki.

**Identyczny wall-clock budget dla TRAD i NN:** różnica wyłącznie w liczbie iteracji
osiąganych w tym samym czasie (NN ~ 30 iter/s vs TRAD ~ 20-40 k iter/s).

## 3. Zbierane metryki

**Per ruch (jsonl):** numer ruchu, strona, UCI, eval, czas (s), faza oraz pełny zestaw
12 metryk MCTS:

- `iterations` — liczba pełnych iteracji PUCT (krytyczne dla throughput),
- `nodes_created`, `max_depth` — rozmiar i głębokość drzewa,
- `eval_calls`, `eval_cache_hits` — wykorzystanie cache ewaluatora,
- `root_children_count`, `best_child_visits` — rozkład wizyt korzenia,
- `root_visit_entropy` — entropia Shannona rozkładu wizyt korzenia (pewność wyboru),
- `convergence_point` — frakcja budżetu, przy której finalny `best_move` się ustabilizował,
- `avg_backprop_depth` — średnia długość ścieżki backpropagacji.

**Analizy zbiorcze:**

- Elo per (ewaluator, czas) z anchor `t = 20s = 0` — `exp3_elo_per_time.csv`,
- log-liniowe dopasowanie `α, β, R²` — `exp3_elo_log_fit.csv`,
- throughput, rozmiar drzewa, głębokość, entropia per (ewaluator, czas) —
  `exp3_time_summary.csv`,
- W/D/L per matchup — `analysis_wdl.csv`.

## 4. Wyniki

### 4.1 Krzywa Elo vs czas — `exp3_elo_per_time.csv`

| Ewaluator | `t = 1 s` | `t = 5 s` | `t = 10 s` | `t = 20 s` (anchor) | `t = 40 s` |
|---|---:|---:|---:|---:|---:|
| **TRAD** | −206.0 | −147.1 | −70.5 | 0.0 | **+82.3** |
| **NN** | −23.2 | 0.0 | **−34.8** (anomalia) | 0.0 | +46.6 |

Wizualizacja: `engine/out/exp3_mcts_time_combined/plots/exp3_elo_curve.png`.

**ΔElo na kolejne podwojenia czasu:**

| Przejście (log₂) | ΔElo TRAD | ΔElo NN |
|---|---:|---:|
| `t = 1 → 2` (interpolacja) | +12 / doubling | +5 / doubling |
| `t = 1 → 5` (×5 ≈ 2.32 doub.) | +59 (≈ +25/doub.) | +23 (≈ +10/doub.) |
| `t = 5 → 10` (×2) | +76.6 | **−34.8** |
| `t = 10 → 20` (×2) | +70.5 | +34.8 |
| `t = 20 → 40` (×2) | +82.3 | +46.6 |

### 4.2 Dopasowanie log-liniowe — `exp3_elo_log_fit.csv`

| Ewaluator | `α` (Elo/podwojenie) | `β` (intercept) | R² |
|---|---:|---:|---:|
| **TRAD** | **+54.28** | −234.24 | **0.94** |
| **NN** | +10.02 | −32.93 | **0.43** |

Wizualizacja: `plots/exp3_elo_curve.png` (z naniesionymi liniami dopasowania).

TRAD pokazuje **bardzo dobre dopasowanie** (R² = 0.94) — log-liniowy model opisuje
skalowanie poprawnie. NN ma **słabe dopasowanie** (R² = 0.43), niemonotoniczna krzywa
łamie założenie modelu log-liniowego.

### 4.3 Macierz W/D/L per matchup — `analysis_wdl.csv`

| Matchup | Gier | W (biały) | D | L | Wynik białego | Śr. ruchów | Śr. czas (B/Cz) |
|---|---:|---:|---:|---:|---:|---:|---|
| TRAD 1s vs 20s | 30 | 0 | 14 | 16 | 0.233 | 76.6 | 1.06 / 19.18 s |
| TRAD 5s vs 20s | 30 | 0 | 18 | 12 | 0.300 | 77.4 | 4.98 / 19.33 s |
| TRAD 10s vs 20s | 30 | 1 | 22 | 7 | 0.400 | 88.1 | 9.95 / 19.70 s |
| **TRAD 20s vs 20s** | 30 | 1 | 26 | 3 | **0.467** | 93.6 | 19.88 / 19.87 s |
| TRAD 40s vs 20s | 30 | 7 | 23 | 0 | **0.617** | 88.6 | 39.41 / 19.91 s |
| NN 1s vs 20s | 30 | 4 | 20 | 6 | 0.467 | 101.4 | 9.26 / 23.95 s |
| NN 5s vs 20s | 30 | 3 | 24 | 3 | 0.500 | 110.1 | 10.04 / 24.02 s |
| NN 10s vs 20s | 30 | 0 | 27 | 3 | **0.450** | 112.3 | 13.90 / 24.01 s |
| **NN 20s vs 20s** | 30 | 3 | 27 | 0 | **0.550** | 101.1 | 24.73 / 24.81 s |
| NN 40s vs 20s | 30 | 5 | 24 | 1 | 0.567 | 100.6 | 43.03 / 23.64 s |

Wizualizacja: `plots/wdl_bars.png`.

**Uwaga:** średni czas białego dla NN przy małym budżecie nie odpowiada zadanemu
(`NN 1s` daje 9.26 s/ruch zamiast 1 s). Jest to konsekwencja działania MCTS NN — pojedyncza
iteracja Stockfish d=10 może trwać kilkaset milisekund, a algorytm wykonuje
co najmniej kilka iteracji per ruch (overhead inicjalizacji + minimum sensowne); zegar
nominalny `t = 1 s` jest **niewystarczalny** dla NN.

### 4.4 Ranking Elo we wspólnej puli — `analysis_elo.csv`

| Wariant | Elo (BT) | Liczba gier |
|---|---:|---:|
| `mcts_trad_40s` | **+131.9** | 30 |
| `mcts_nn_40s` | +64.8 | 30 |
| `mcts_trad_20s` | +51.5 | 150 |
| `mcts_nn_20s` | +18.7 | 150 |
| `mcts_nn_5s` | +18.7 | 30 |
| `mcts_nn_1s` | −4.3 | 30 |
| `mcts_nn_10s` | −15.8 | 30 |
| `mcts_trad_10s` | −19.3 | 30 |
| `mcts_trad_5s` | −94.7 | 30 |
| `mcts_trad_1s` | −151.4 | 30 |

### 4.5 Profil obliczeniowy — `exp3_time_summary.csv`

| Ewal. | `t` | Śr. czas faktyczny | Iteracje (mean) | Throughput (iter/s) | Nodes created | Max depth | Root entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAD | 1 s | 1.07 s | 25 128 | 25 055 | 2 154 | 7.7 | 1.98 |
| TRAD | 5 s | 5.00 s | 168 019 | 33 575 | 8 342 | 10.5 | 1.65 |
| TRAD | 10 s | 9.95 s | 396 391 | 39 657 | 14 723 | 12.4 | 1.36 |
| TRAD | 20 s | 19.92 s | 407 535 | 20 368 | 27 040 | 16.9 | 0.96 |
| TRAD | 40 s | 39.52 s | 1 416 206 | 35 404 | 59 875 | 17.8 | 0.92 |
| NN | 1 s | 8.90 s | 130 | **31** (!) | 131 | 4.3 | 1.97 |
| NN | 5 s | 9.94 s | 225 | 33 | 225 | 4.8 | **2.04** |
| NN | 10 s | 13.70 s | 350 | 30 | 350 | 5.2 | 2.05 |
| NN | 20 s | 24.48 s | 870 | 40 | 870 | 5.9 | 2.09 |
| NN | 40 s | 43.50 s | 1 350 | 33 | 1 350 | 6.5 | **2.13** |

Wizualizacje: `plots/exp3_throughput_curve.png`, `plots/exp3_tree_size_curve.png`,
`plots/exp3_max_depth_curve.png`, `plots/exp3_entropy_curve.png`, `plots/mcts_throughput.png`.

**Stosunek throughput TRAD / NN:** od ~600× (`t = 5s`) do ~1100× (`t = 10s`).
Praktycznie **NN wykonuje ~30 iteracji/s niezależnie od budżetu**, podczas gdy TRAD
osiąga 20-40 k iter/s.

## 5. Dyskusja

### 5.1 TRAD skaluje się przewidywalnie log-liniowo

Krzywa TRAD jest niemal idealnym przykładem klasycznego skalowania MCTS:
**`α = 54.3 Elo / podwojenie czasu`** przy `R² = 0.94`. Wartość mieści się
w dolnej granicy zakresu typowego dla MCTS (50-150). Jest niższa od opisywanych
w literaturze (np. AlphaZero ~100 Elo/doubling) — wynika to prawdopodobnie z dwóch
czynników: (i) ewaluator TRAD nie jest na poziomie sieci neuronowej AlphaZero,
więc dodatkowy czas ma mniejszy „leverage", (ii) deterministyczne otwarcia ECO
spłaszczają wariancję wyników.

ΔElo per podwojenie konsekwentnie: +77 (5→10s), +71 (10→20s), +82 (20→40s) — **praktycznie
płaska stopa zwrotu**, brak wyraźnych malejących zwrotów w mierzonym zakresie. Ekstrapolacja
na `t = 60s` daje ≈ +50 Elo nad `t = 40s`; na `t = 120s` ≈ +110 Elo nad `t = 40s`.

### 5.2 NN — krzywa nie skaluje się log-liniowo

Krzywa NN jest **niemonotoniczna** (`Elo(t = 5s) = 0 > Elo(t = 10s) = −35`)
i ma `R² = 0.43` — model log-liniowy zawodzi. Spojrzenie na metryki tłumaczy mechanizm:

1. **Throughput NN jest praktycznie stały ~30 iter/s** niezależnie od budżetu. Daje to
   przy `t = 1 s` zaledwie 130 iteracji, przy `t = 40 s` — 1 350. Drzewo MCTS jest
   **głodne danych**: brakuje wizyt aby zbudować wiarygodny rozkład wizyt korzenia.
2. **Entropia korzenia NIE maleje z budżetem** — rośnie z 1.97 (`t = 1s`) do 2.13 (`t = 40s`),
   podczas gdy TRAD spada z 1.98 do 0.92. NN nigdy nie osiąga „pewności" wyboru ruchu
   — wizyty korzenia pozostają rozproszone, bo budżet nie pozwala na dostatecznie wiele
   rolloutów aby jeden ruch znacząco przeważył.
3. **Max depth drzewa NN ≤ 6.5** vs TRAD ≤ 17.8 — drzewo NN jest **2-3× płytsze**;
   PUCT z silną wyrocznią liścia daje słabszą eksplorację głębi.
4. **Anomalia `t = 1s vs 20s` (NN)**: białe NN przy nominalnym `1 s` faktycznie zużywa
   9.3 s/ruch (overhead Stockfisha) — to nie jest „naprawdę" mały budżet, lecz raczej
   ~1.5 podwojenia mniej od kotwicy. To tłumaczy dlaczego `Elo(NN 1s) = −23.2`
   jest tylko nieznacznie niższe od kotwicy.

H1, H2 i H4 dla NN **nie potwierdzają się**: nie ma monotonicznego skalowania, `α`
jest zaniedbywalne (10 Elo/doub. ± duży błąd), entropia nie spada. NN MCTS jest
w tej konfiguracji **przebudżetowany** — gdyby wyrocznia Stockfish była tańsza
(np. d=5 zamiast d=10), throughput wzrósłby ~5× i krzywa NN mogłaby uzyskać sensowną
postać log-liniową.

### 5.3 Walidacja kalibracji `t = 2.61 s` z Eksp. 1 (H6)

Ekstrapolacja log-liniowa TRAD do `t = 2.61 s`:
`Elo(2.61) = −234.24 + 54.28 · log₂(2.61) = −234.24 + 54.28 · 1.384 ≈ −159 Elo`.
Wartość leży **między** punktami pomiarowymi `Elo(1) = −206` a `Elo(5) = −147` —
H6 spełniona. Kalibracja MCTS z Eksp. 1 jest spójna z krzywą skalowania TRAD.

Dla NN ekstrapolacja jest bezprzedmiotowa (R² = 0.43), ale i tak surowo:
`Elo(2.61) ≈ −33 + 10 · 1.384 ≈ −19 Elo` — w okolicy `Elo(NN 1s) = −23`.

### 5.4 Sanity check (H5)

- TRAD `20s vs 20s`: 1W/26D/3L → `white_score = 0.467` ≈ 0.5 ✓
- NN `20s vs 20s`: 3W/27D/0L → `white_score = 0.550` ≈ 0.5 ✓

Drobne odchylenia są mieszczone w przedziale ufności dla `n = 30` (~ ±0.18).
**H5 potwierdzona.**

### 5.5 Porównanie z Eksp. 2 — „Elo per zasób" w MCTS vs Minimax

Aby porównać efektywność, należy przeliczyć zasób na wspólną jednostkę (sekundy
czasu zegarowego). Z Eksp. 2:

- Minimax TRAD: `d = 3 → 4 → 5`, koszt 0.21 → 0.61 → 3.51 s/ruch (× 2.96, × 5.7);
  ΔElo: +190, +341.
- MCTS TRAD: podwojenie `t` z 5 → 10 → 20 → 40 s; ΔElo: +77, +71, +82.

Przeliczone na „Elo na podwojenie czasu":

- **Minimax TRAD**: jedno podwojenie czasu odpowiada ułamkowi poziomu głębokości
  (~0.7-0.8 poziomu); przy ΔElo ≈ +190-340 per poziom, daje to ~130-280 Elo per
  podwojenie czasu. Bardzo wysoko — ale przy ekstrapolacji do większych budżetów
  Minimax szybko trafia w eksponencjalny wzrost kosztu.
- **MCTS TRAD**: liniowe ~54 Elo per podwojenie. Wolniej, ale stabilnie.

Dla budżetów rzędu kilku sekund (Eksp. 1) **Minimax dominuje pod względem Elo/s**;
dla budżetów rzędu kilkudziesięciu sekund różnica się zaciera, bo Minimax zaczyna
napotykać limit `d ≤ 5` (`d = 6` poza zasięgiem). Eksperymenty 2 i 3 razem wskazują,
że pole MCTS-jako-alternatywa-Minimaxa rozszerza się tylko przy długich budżetach
i nadal **wymaga lepszego ewaluatora niż TRAD** aby zrównać siłę.

### 5.6 Ograniczenia

- **n = 30 / matchup** + bardzo wysoki odsetek remisów (do 27 / 30 dla `NN 5s/10s`)
  → niska moc statystyczna. Wszystkie estymaty Elo NN obarczone dużym błędem.
- **Throughput NN ograniczony konstrukcyjnie** przez Stockfish d=10. Wnioski o NN MCTS
  dotyczą tej konkretnej wyroczni, nie wytrenowanej sieci neuronowej.
- **`t = 1s` nominalne** dla NN jest niewykonalne (faktycznie 9.3 s) — punkt
  `Elo(NN 1s)` jest fałszywie wysoki na osi `log₂(t)`. Wpływa na fit log-liniowy.
- **Brak punktów `t > 40s`** uniemożliwia wykrycie potencjalnego nasycenia krzywej.
- **Adjudykacja ±0.05/20 ruchów** silnie zmniejsza wariancję wyników (wiele remisów)
  i może ukrywać małe różnice taktyczne.

## 6. Wnioski

1. **MCTS_TRAD skaluje się log-liniowo z `α = 54 Elo / podwojenie czasu`**
   (`R² = 0.94`) w zakresie `t = 1-40 s`. Brak widocznych malejących zwrotów
   w tym zakresie — krzywa nadal wzrasta przy `t → 40 s`.
2. **MCTS_NN nie skaluje się log-liniowo** (`R² = 0.43`, niemonotoniczność
   `Elo(5s) = 0 > Elo(10s) = −35`). Przyczyna: throughput NN ograniczony do
   ~30 iter/s przez koszt wyroczni Stockfish d=10 — drzewo pozostaje płytkie
   (≤ 6.5) i głodne wizyt, entropia korzenia nie spada (1.97 → 2.13).
3. **Kalibracja `t = 2.61 s` z Eksp. 1 jest spójna** z krzywą TRAD: ekstrapolowane
   `Elo(2.61) ≈ −159` mieści się między pomiarami `Elo(1) = −206` i `Elo(5) = −147`.
4. **Sanity check** `t = 20s vs 20s` przeszedł dla obu ewaluatorów (`white_score`
   ≈ 0.47 / 0.55).
5. **Throughput jest cnotą decydującą** w MCTS: TRAD osiąga 20-40 k iter/s,
   NN tylko ~30 iter/s — różnica ~600-1100×. Jakość ewaluatora liścia
   nie kompensuje braku rolloutów.
6. **Wnioski praktyczne dla pozostałych eksperymentów:**
   - W Eksp. 1 budżet `t = 2.61 s` dla MCTS jest sensowny (potwierdzony krzywą TRAD).
   - W Eksp. 8 budżet `t = 60 s` dla MCTS daje ekstrapolowane `Elo_TRAD ≈ 130`
     nad anchor, ale `Elo_NN` może być słabe (~80 Elo nad anchor lub mniej).
   - **MCTS_NN z obecną wyrocznią Stockfish d=10 nie jest konkurencyjnym
     wariantem** w tej pracy — wymagałby tańszej wyroczni lub znacząco większego
     budżetu.

Materiał stanowi podstawę trzeciej części rozdziału eksperymentalnego (skalowanie MCTS):
rysunek 4.7 (`exp3_elo_curve.png` — krzywa Elo vs log₂(t), 2 linie + dopasowania),
rysunek 4.8 (`exp3_throughput_curve.png` — różnica skali TRAD vs NN), rysunek 4.9
(`exp3_entropy_curve.png` — degradacja pewności NN), tabela 4.5 (`α` i R² per ewaluator),
tabela 4.6 (zestawienie throughput / drzewo / głębokość).
