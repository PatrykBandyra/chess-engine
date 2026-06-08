# Eksperyment 2 — Analiza wyników (skalowanie głębokości Minimaxa)

## 1. Cel eksperymentu i pytanie badawcze

Eksperyment ma odpowiedzieć na pytanie: **jak siła gry Minimaxa α-β rośnie wraz z głębokością
przeszukiwania i jaką postać przyjmuje krzywa skalowania dla obu ewaluatorów (TRAD i NN)?**
Wynik ma uzasadnić wybór głębokości referencyjnej `d = 4` używanej w pozostałych eksperymentach
oraz ilościowo opisać kompromis „głębokość ↔ koszt obliczeniowy".

Hipotezy:

- **H1:** Elo rośnie monotonicznie z głębokością dla obu ewaluatorów; każdy dodatkowy poziom
  przynosi wyraźny zysk Elo (rzędu kilkudziesięciu - kilkuset).
- **H2** (malejące zwroty): ΔElo na poziom głębokości spada z `d` — typowe dla silników szachowych.
- **H3:** Krzywe skalowania TRAD i NN różnią się: NN, dzięki dokładniejszej ewaluacji
  liścia, powinien wymagać mniej głębokości dla osiągnięcia tej samej siły
  („głębokość kompensuje słabość ewaluatora").
- **H4:** Czas/ruch rośnie wykładniczo z głębokością; EBF (Effective Branching Factor) ma
  pozostawać w przedziale charakterystycznym dla dobrego α-β + przycinania (~2-6).
- **H5** (sanity check): Mecz `d = 4 vs d = 4` (oba ewaluatory) ma dawać wynik bliski 0.5 ± kilkadziesiąt Elo.

## 2. Założenia metodyczne

Eksperyment to **seria pojedynków „głębokość testowana vs głębokość referencyjna d = 4"**
dla każdego z 2 ewaluatorów (TRAD, NN). Konfiguracja `exp2_minimax_depth.json` definiuje
10 matchupów (`d ∈ {2,3,4,5,6}` × 2 ewaluatory), ale **matchupy z `d = 6` zostały świadomie
pominięte** z powodu wykonalności czasowej (szacunkowo ~28 h/matchup dla TRAD i ~260 h
dla NN). Faktycznie uruchomiono **8 matchupów** × 30 partii = **240 partii**.

| # | Matchup | Ewaluator | `d_white` | `d_black` (anchor) |
|---|---|---|---:|---:|
| 1 | `minimax_trad_d2_vs_d4` | TRAD | 2 | 4 |
| 2 | `minimax_trad_d3_vs_d4` | TRAD | 3 | 4 |
| 3 | `minimax_trad_d4_vs_d4` | TRAD | 4 | 4 |
| 4 | `minimax_trad_d5_vs_d4` | TRAD | 5 | 4 |
| 5 | `minimax_nn_d2_vs_d4` | NN | 2 | 4 |
| 6 | `minimax_nn_d3_vs_d4` | NN | 3 | 4 |
| 7 | `minimax_nn_d4_vs_d4` | NN | 4 | 4 |
| 8 | `minimax_nn_d5_vs_d4` | NN | 5 | 4 |

**Wybór `d = 4` jako kotwicy** jest podyktowany jego „środkową" pozycją w testowanym zakresie;
krzywa Elo jest normalizowana do `Elo(d = 4) = 0`, a pozostałe punkty wyznaczane metodą
Bradleya-Terry'ego.

**Próba:** 30 partii/matchup wystarcza do wykrycia różnic Elo ≥ ~100 z `p < 0.05`. **Otwarcia:**
identycznie jak w Eksp. 1 — 25 ustandaryzowanych pozycji ECO, cyklicznie. **Adjudykacja:**
±0.05/20 ruchów (włączona; bez niej partie `d = 5 vs d = 4` w wyrównanych końcówkach
trwałyby >200 ruchów). **Książka otwarć:** wyłączona.

## 3. Zbierane metryki

**Per ruch (jsonl):** numer ruchu, strona, UCI, eval, czas (s), faza gry oraz pełny zestaw
metryk Minimaxa (21 pól), kluczowe dla skalowania:

- `nodes_searched` — bezwzględna liczba węzłów (skala wykładnicza z głębokością),
- `depth_completed` — czy ID osiągnął docelową głębokość,
- `nodes_per_depth` — wektor węzłów per iteracja ID → liczona `EBF[d] = nodes[d] / nodes[d−1]`,
- `tt_hit_rate`, `tt_cutoff_rate` — efektywność tablicy transpozycji,
- `nmp_success_rate`, `rfp_cutoffs`, `futility_prunes`, `lmp_prunes`, `see_prunes` —
  częstość uruchamiania poszczególnych technik przycinania,
- `check_extensions`, `qs_nodes`, `qs_max_depth` — rozszerzenia szachowe i quiescence.

**Analizy zbiorcze:**

- Elo per (ewaluator, głębokość) z anchor `d = 4 = 0` — `exp2_elo_per_depth.csv`,
- średnia `time_s`, `nodes`, EBF, statystyki TT/przycinania per (ewaluator, głębokość) —
  `exp2_depth_summary.csv`,
- W/D/L per matchup — `analysis_wdl.csv`,
- ranking BT we wspólnej puli wszystkich graczy — `analysis_elo.csv`.

## 4. Wyniki

### 4.1 Krzywa Elo vs głębokość — `exp2_elo_per_depth.csv`

| Ewaluator | `d = 2` | `d = 3` | `d = 4` (anchor) | `d = 5` |
|---|---:|---:|---:|---:|
| **TRAD** | **−256.3** | **−189.6** | 0.0 | **+341.0** |
| **NN** | **+46.5** | **−275.9** | 0.0 | **+256.7** |

Wizualizacja: `engine/out/exp2_minimax_depth_combined/plots/exp2_elo_curve.png`.

**ΔElo per poziom głębokości:**

| Przejście | TRAD | NN |
|---|---:|---:|
| `d = 2 → 3` | +66.7 | **−322.4** (anomalia) |
| `d = 3 → 4` | +189.6 | +275.9 |
| `d = 4 → 5` | +341.0 | +256.7 |
| Średni przyrost (`d = 2 → 5`) | +199 Elo/poziom | +70 Elo/poziom |

### 4.2 Macierz W/D/L per matchup — `analysis_wdl.csv`

| Matchup | Gier | W (biały, mniejsza d) | D | L | Wynik białego | Śr. ruchów | Śr. czas (B/Cz) |
|---|---:|---:|---:|---:|---:|---:|---|
| TRAD d=2 vs d=4 | 30 | 1 | 9 | 20 | 0.183 | 68.1 | 0.11 / 1.30 s |
| TRAD d=3 vs d=4 | 30 | 3 | 9 | 18 | 0.250 | 93.7 | 0.24 / 1.05 s |
| **TRAD d=4 vs d=4** | 30 | 6 | 15 | 9 | **0.450** | 92.7 | 0.66 / 0.72 s |
| TRAD d=5 vs d=4 | 30 | 23 | 7 | 0 | **0.883** | 82.0 | 3.79 / 1.03 s |
| NN d=2 vs d=4 | 30 | 11 | 12 | 7 | **0.567** | 99.1 | 3.74 / 40.81 s |
| NN d=3 vs d=4 | 30 | 0 | 10 | 20 | 0.167 | 92.0 | 9.80 / 42.18 s |
| **NN d=4 vs d=4** | 30 | 10 | 14 | 6 | **0.567** | 106.2 | 45.84 / 46.00 s |
| NN d=5 vs d=4 | 30 | 20 | 9 | 1 | **0.817** | 90.8 | 135.78 / 40.33 s |

Wizualizacja: `plots/wdl_bars.png`.

### 4.3 Ranking Elo we wspólnej puli — `analysis_elo.csv`

| Wariant | Elo (BT) | Liczba gier |
|---|---:|---:|
| `minimax_trad_d5` | **+340.5** | 30 |
| `minimax_nn_d5` | +250.0 | 30 |
| `minimax_nn_d2` | +47.7 | 30 |
| `minimax_trad_d4` | +19.5 | 120 |
| `minimax_nn_d4` | +1.7 | 120 |
| `minimax_trad_d3` | −166.6 | 30 |
| `minimax_trad_d2` | −229.0 | 30 |
| `minimax_nn_d3` | −263.7 | 30 |

Mecz `d = 4 vs d = 4` (oba ewaluatory) daje wynik `~0.45-0.57` (Elo ≈ ±20) — **H5 (sanity
check) potwierdzona** w granicach szumu statystycznego.

### 4.4 Profil obliczeniowy — `exp2_depth_summary.csv`

| Ewal. | `d` | Śr. czas/ruch | Śr. węzłów | TT hit | TT cutoff | NMP succ. | EBF mean | qs_nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAD | 2 | 0.098 s | 153 | 5.9 % | 19.0 % | 0 % | 5.39 | 285 |
| TRAD | 3 | 0.208 s | 427 | 9.1 % | 44.1 % | 0 % | 5.07 | 536 |
| TRAD | 4 | 0.615 s | 1 539 | 9.7 % | 58.7 % | 32.6 % | 9.79 | 1 518 |
| TRAD | 5 | **3.51 s** | 8 926 | 9.7 % | 64.1 % | 45.4 % | 20.24 | 8 737 |
| NN | 2 | 3.73 s | 138 | 2.7 % | 17.1 % | 0 % | 3.86 | 138 |
| NN | 3 | 9.45 s | 378 | 5.4 % | 38.9 % | 0 % | 5.04 | 312 |
| NN | 4 | 49.49 s | 1 686 | 6.2 % | 53.4 % | 34.6 % | 7.48 | 1 248 |
| NN | 5 | **133.69 s** | 8 535 | 7.5 % | 65.0 % | 42.9 % | 20.78 | 6 139 |

Wizualizacje: `plots/exp2_time_curve.png`, `plots/exp2_nodes_curve.png`,
`plots/exp2_ebf_curve.png`, `plots/exp2_pruning_by_depth.png`, `plots/minimax_pruning.png`.

**Stosunek czasu NN / TRAD per głębokość:** d=2: ×38; d=3: ×45; d=4: ×80; d=5: **×38**.
W szerszej średniej **NN jest ok. 40-80× wolniejsze od TRAD** przy identycznej głębokości.

## 5. Dyskusja

### 5.1 Krzywa TRAD — narastające zwroty, nie malejące

Wyniki TRAD są **monotoniczne** i **wyraźnie rosnące**: +67 Elo (`d = 2 → 3`),
+190 Elo (`d = 3 → 4`), +341 Elo (`d = 4 → 5`). Zamiast spodziewanych malejących zwrotów
(H2) obserwujemy **przyspieszenie** przyrostu Elo z głębokością. Najprawdopodobniejsze
wyjaśnienie: heurystyka TRAD jest na tyle „głośna" (szumowa), że na płytkich głębokościach
(`d = 2, 3`) silnik często popełnia błędy taktyczne, które dopiero `d = 4-5` jest w stanie
„rozliczyć" przez quiescence i poszerzone przeszukiwanie szachów. Mat-w-N i wymuszone
sekwencje wykrywalne dopiero przy `d ≥ 5` produkują efekt fazy przejściowej.

Hipoteza alternatywna: zakres `d = 2-5` jest **przed inflekcją** typowej krzywej Elo —
malejące zwroty pojawiają się dopiero przy `d ≥ 6-7`, czego nie zaobserwowano z powodu
wykluczenia `d = 6`. Wiarygodna ekstrapolacja wymagałaby co najmniej jednego punktu
powyżej `d = 5`.

### 5.2 Anomalia NN przy `d = 2`

Najsilniejsza pojedyncza obserwacja to **niemonotoniczność krzywej NN**:
`Elo(d = 2) = +46.5 > Elo(d = 3) = −275.9`. Wariant `MINIMAX_NN d = 2` jest niemal
równy `MINIMAX_NN d = 4` (anchor), a `d = 3` przegrywa drastycznie. W/D/L:
- `NN d = 2 vs d = 4`: 11W/12D/7L (wynik 0.567),
- `NN d = 3 vs d = 4`: 0W/10D/20L (wynik 0.167).

**Możliwe przyczyny:**

1. **Efekt horyzontu z domeszką wyroczni.** Ewaluator NN to subprocess Stockfish d=10
   wywoływany na liściach. Przy `d = 2` Minimax wykonuje płytkie przeszukiwanie,
   ale każdy liść jest oceniany przez 10-plejową analizę Stockfisha — efektywna
   „głębokość taktyczna" wynosi ~12. Przy `d = 3` dodajemy jeden poziom Minimaxowy,
   który **rozsypuje** koherentne plany Stockfisha (Minimax wybiera ruch
   maksymalizujący „swój" odczyt z liścia, ale przy `d = 3` parzysta-nieparzysta
   asymetria horyzontu (move ordering, zero-window) może niszczyć stabilność wyboru).
2. **Mała próba (n = 30) + nietransitywność.** BT MLE zakłada transitywność — wynik
   przeciwko jednej kotwicy może źle estymować Elo w realnej puli przeciwników.
3. **Sumaryczna stochastyczność NN.** Stockfish jest deterministyczny, ale przy
   ograniczonej liczbie wątków i fixed depth wynik per liść może być wrażliwy
   na hash, kolejność ruchów.

Ta anomalia **wymaga jawnego raportowania w pracy** — nie powinno się prezentować
krzywej NN jako gładkiej. W rozdziale eksperymentalnym warto zaproponować follow-up:
powtórzyć matchupy `NN d = 2` i `NN d = 3` z większą próbą (`n = 100+`), ewentualnie
zmierzyć stabilność BT przy zamianie kotwicy na `d = 3`.

### 5.3 NN „kompensuje" głębokość — częściowo

Porównanie tej samej Elo w obu krzywych (z `exp2_elo_per_depth.csv`):

- `MINIMAX_TRAD d = 4` (Elo ≈ 0) ≈ `MINIMAX_NN d = 4` (Elo ≈ 0) — anchory są normalizowane.
- Surowy porównanie w `analysis_elo.csv`: `minimax_nn_d4` = +1.7 Elo, `minimax_trad_d4` = +19.5,
  czyli **TRAD na `d = 4` jest minimalnie silniejszy niż NN na `d = 4`** w tej próbie.
- `minimax_nn_d5` (+250 Elo) jest słabszy niż `minimax_trad_d5` (+340 Elo).

Wniosek częściowo sprzeczny z H3: w obecnej próbie NN **nie kompensuje** głębokości —
sumarycznie ewaluator NN nie daje przewagi nad TRAD przy tej samej głębokości. Wymaga
to ostrożnej interpretacji:

- Może być artefaktem identyfikacji BT przy 30 grach/matchup oraz wspomnianej anomalii NN d=2.
- Wynik Eksp. 1 (gdzie MINIMAX_NN d=3 wyraźnie wygrywa z MINIMAX_TRAD d=3) **nie zgadza się**
  z Eksp. 2: tam NN był wyraźnie silniejszy. Różnica wynika z tego, że w Eksp. 1 obaj gracze
  mieli ten sam algorytm i `d = 3` (czyste porównanie ewaluatora), natomiast tu porównanie
  jest pośrednie (oba grają przeciw kotwicy `d = 4` tego samego ewaluatora). Spójność
  obu wyników można uzyskać dopiero łącząc je z Eksp. 8 (silne parametry).

### 5.4 Koszt obliczeniowy — wykładniczy wzrost

Stosunek czasu kolejnych głębokości (TRAD): `0.098 → 0.208 → 0.615 → 3.51 s`, czyli
× 2.1, × 2.96, × 5.7. Dla NN: `3.73 → 9.45 → 49.5 → 133.7 s`, czyli × 2.53, × 5.24, × 2.7.
Średnie tempo wzrostu mieści się w granicach 2.5-5× per poziom — co jest zgodne z α-β
o EBF ~ 5-7. **Liczba węzłów rośnie podobnie** (TRAD: 153 → 427 → 1539 → 8926 = ×2.8, ×3.6, ×5.8).

**EBF** odczytany ze średnich `nodes_per_depth` rośnie nieoczekiwanie z głębokością
(TRAD: 5.39 → 5.07 → 9.79 → 20.24; NN: 3.86 → 5.04 → 7.48 → 20.78). Główne źródła:

- na większych głębokościach **quiescence search** dorzuca dużą liczbę dodatkowych węzłów
  (`qs_nodes` w TRAD: 285 → 536 → 1518 → 8737 — porównywalne z `nodes_searched`),
- **check extensions** wydłużają linie taktyczne,
- użyta metryka liczy `nodes_per_depth` jako stosunek węzłów ID na kolejnych iteracjach,
  co przy dynamicznym pruning daje wartości wyższe niż teoretyczne `√branching`.

**Techniki przycinania działają zgodnie z konfiguracją:**

- TT hit rate rośnie z głębokością (TRAD: 5.9 % → 9.7 %; NN: 2.7 % → 7.5 %),
- TT cutoff rate rośnie znacząco (TRAD: 19 % → 64 %; NN: 17 % → 65 %),
- NMP aktywuje się dopiero od `d = 4` (próg `depth ≥ 3` w implementacji + reduction `R + 1`
  daje efektywną aktywację przy `d ≥ 4`); skuteczność: 32-45 %.

### 5.5 Praktyczne wnioski dla pozostałych eksperymentów

- **`d = 4` jako default** jest dobrym wyborem dla porównań — leży w środku zakresu siły
  i ma akceptowalny koszt (0.6 s/ruch TRAD, 49 s/ruch NN).
- **`d = 3` dla NN w Eksp. 1** jest uzasadnione kosztem — przejście na `d = 4` wymagałoby
  ~50 s/ruch zamiast ~9 s/ruch (×5).
- **`d = 5` dla TRAD w Eksp. 8** to maksymalne praktyczne (~3.5 s/ruch).
- **`d = 6` jest poza zasięgiem** w tej pracy — wymagałoby ~28 h/matchup dla TRAD
  i ~260 h dla NN.

### 5.6 Ograniczenia

- **n = 30 / matchup** — anomalia NN d=2 może być artefaktem małej próby.
- **Pojedyncza kotwica `d = 4`** — BT pośrednia nie pokrywa par cross-evaluator.
  Bezpośrednie porównanie TRAD vs NN w funkcji głębokości wymagałoby dodatkowych
  matchupów (TRAD `d` vs NN `d`).
- **Brak `d = 6`** uniemożliwia zaobserwowanie potencjalnego punktu nasycenia.
- **EBF mocno zaszumione** (std rzędu wartości średniej) — interpretacja porównawcza
  raczej niż bezwzględna.

## 6. Wnioski

1. **TRAD wykazuje monotoniczne, narastające zwroty z głębokością** w zakresie `d = 2-5`:
   +67, +190, +341 Elo na kolejne poziomy. Sumarycznie ~600 Elo zysku od `d = 2` do `d = 5`.
2. **Krzywa NN jest niemonotoniczna**: `Elo(NN d = 2) = +46.5 > Elo(NN d = 3) = −275.9`.
   Najprawdopodobniejsza przyczyna to interakcja efektu horyzontu z wyrocznią Stockfisha
   na liściach (Stockfish d=10 daje silną ocenę przy `d = 2`, ale jeden poziom Minimaxa
   „rozsypuje" plany przy `d = 3`). Anomalia wymaga jawnego raportowania.
3. **Czas/ruch rośnie wykładniczo** z czynnikiem ×2.5-5 per poziom. NN jest **~40-80× wolniejsze
   od TRAD** przy identycznej głębokości — to praktyczna granica zastosowania NN przy `d ≥ 4`.
4. **Sanity check** `d = 4 vs d = 4` przeszedł: TRAD 0.450, NN 0.567 (Elo ≈ ±20).
5. **`d = 4` jako default** jest uzasadnionym kompromisem siła ↔ koszt. **`d = 5` daje
   ok. +300 Elo** kosztem ×5-6 czasu.
6. Hipoteza „NN kompensuje głębokość" w obecnej próbie **nie potwierdza się jednoznacznie** —
   przy identycznej `d` TRAD jest sumarycznie silniejszy w tej puli (sprzeczność z Eksp. 1
   wymagająca dyskusji w pracy).

Materiał stanowi podstawę drugiej części rozdziału eksperymentalnego (skalowanie zasobów):
rysunek 4.4 (`exp2_elo_curve.png` — krzywa Elo vs głębokość, 2 linie), rysunek 4.5
(`exp2_time_curve.png`, log scale), rysunek 4.6 (`exp2_ebf_curve.png`), tabela 4.3
(ΔElo per poziom + koszt), tabela 4.4 (statystyki przycinania).
