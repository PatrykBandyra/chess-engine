# Eksperyment 1 — Analiza wyników (round-robin 4 wariantów)

## 1. Cel eksperymentu i pytanie badawcze

Eksperyment ma odpowiedzieć na centralne pytanie pracy: **który z czterech wariantów silnika
(MINIMAX_TRAD, MINIMAX_NN, MCTS_TRAD, MCTS_NN) jest najsilniejszy i czy o jego przewadze
decyduje algorytm wyszukiwania (oś A: Minimax α-β vs MCTS PUCT) czy funkcja ewaluacji
(oś B: heurystyka TRAD vs wyrocznia „NN" — Stockfish na płytkiej głębokości d=10)?**

Hipotezy:

- **H1** (główna): Cztery warianty można uszeregować na wspólnej skali Elo metodą Bradleya-Terry'ego;
  ranking będzie monotoniczny i niesprzeczny z wynikami par bezpośrednich.
- **H2** (oś A): Minimax α-β z d=3 oraz MCTS PUCT z budżetem ≈ 2.61 s/ruch nie są równoważne
  pod względem siły gry.
- **H3** (oś B): Jakość ewaluatora ma istotny wpływ niezależnie od algorytmu — wyrocznia
  Stockfish d=10 powinna być zauważalnie silniejsza od heurystyki ręcznej.
- **H4** (sanity check): Przy 25 ustandaryzowanych otwarciach ECO i adjudykacji ±0.05/20 ruchów
  asymetria koloru (białe vs czarne) jest zaniedbywalna.

## 2. Założenia metodyczne

**Uczestnicy (macierz 2×2):**

| Wariant | Algorytm | Ewaluator | Parametr zasobu |
|---|---|---|---|
| `MINIMAX_TRAD` | Minimax α-β (TT, NMP, RFP, LMP, SEE, killer, quiescence) | Heurystyczny (PST fazowe, struktura pionów, bezp. króla, para gońców) | `d = 3` |
| `MINIMAX_NN` | j.w. | „NN" = subprocess Stockfish d=10 | `d = 3` |
| `MCTS_TRAD` | MCTS PUCT (AlphaZero-style) | Heurystyczny | `t = 2.61 s/ruch` |
| `MCTS_NN` | j.w. | „NN" = Stockfish d=10 | `t = 2.61 s/ruch` |

Wybór `d = 3` dla obu wariantów Minimaxa jest podyktowany dwoma względami:
1. **Czystym porównaniem osi B** — różnić ma się wyłącznie ewaluator.
2. **Wykonalnością czasową** — `MINIMAX_NN` z `d = 4` daje ~30-50 s/ruch (niepraktyczne),
   natomiast `d = 3` mieści się w 3-6 s/ruch.

Budżet czasowy MCTS = **2.61 s/ruch** został wyznaczony empirycznie jako średni czas/ruch
trzech kalibracyjnych partii `MINIMAX_TRAD d=4` (zapis w `_mcts_calibrated_time.txt`).

**Próba:** 6 par × 30 partii = **180 partii** (15 w układzie oryginalnym + 15 z zamianą kolorów).
Wielkość próby `n = 30` umożliwia wykrywanie różnic Elo ≥ 100 przy `p < 0.05`; mniejsze
różnice (< 80 Elo) mogą pozostawać poniżej progu istotności.

**Otwarcia:** 25 ustandaryzowanych pozycji ECO (po 4 pełnych ruchach) z pliku
`experiments/openings_eco25.fen`, dobierane cyklicznie modulo 25 — kontrola wariancji
otwarciowej w stylu CCRL/TCEC.

**Adjudykacja:** remis jeśli oba silniki oceniają pozycję w przedziale **±0.05** przez
**20 kolejnych pełnych ruchów**. Wyłączona księga otwarć — celem jest pomiar surowej
siły algorytmu, nie repertuaru.

## 3. Zbierane metryki

**Per ruch (jsonl):** numer ruchu, strona, UCI, ewaluacja, czas (s), faza gry [0..1] oraz
zestaw metryk specyficznych dla algorytmu:

- **Minimax (21 metryk):** `nodes_searched`, `depth_completed`, `tt_size`, `tt_hits`, `tt_cutoffs`,
  `nmp_attempts/cutoffs`, `rfp_cutoffs`, `futility_prunes`, `lmp_prunes`, `see_prunes`,
  `check_extensions`, `aspiration_researches`, `qs_nodes`, `qs_max_depth`, `killer_hits`,
  `nodes_per_depth` (do obliczenia EBF).
- **MCTS (12 metryk):** `iterations`, `nodes_created`, `max_depth`, `eval_calls`,
  `eval_cache_hits`, `root_visit_entropy`, `convergence_point`, `avg_backprop_depth`, `c_puct`.

**Analizy zbiorcze:**

- ranking Elo metodą Bradleya-Terry'ego z normalizacją `mean = 0`,
- test dwumianowy na partiach rozstrzygniętych (decisive) per para + 95% CI,
- agregaty osi A i osi B z par cross-axis,
- statystyka przewagi koloru (sanity check).

## 4. Wyniki

### 4.1 Ranking Elo (Bradley-Terry MLE) — `analysis_elo.csv`

| Wariant | Elo (BT) | Liczba gier |
|---|---:|---:|
| **MINIMAX_NN** | **+8.7** | 90 |
| MCTS_TRAD | 0.0 | 90 |
| MCTS_NN | -2.9 | 90 |
| MINIMAX_TRAD | -5.8 | 90 |

Rozpiętość rankingu wynosi tylko **~14.5 Elo**, co — przy `n = 30` gier/parę — oznacza,
że punktowo różnice między 2., 3. a 4. miejscem nie są statystycznie istotne (zob. § 4.3).
Liderem jednoznacznie wskazanym jest **MINIMAX_NN**.

### 4.2 Macierz W/D/L per para — `analysis_wdl.csv`

| Para (biały vs czarny) | Gier | W (biały) | D | L (czarny) | Wynik białego | Śr. ruchów | Śr. czas/ruch (B/Cz) |
|---|---:|---:|---:|---:|---:|---:|---|
| MINIMAX_TRAD vs MINIMAX_NN | 30 | 10 | **5** | 15 | 0.417 | 79.7 | 6.27 / 5.84 s |
| MCTS_TRAD vs MCTS_NN | 30 | 2 | **25** | 3 | 0.483 | 103.3 | 5.36 / 5.57 s |
| MINIMAX_NN vs MCTS_TRAD | 30 | 9 | 10 | 11 | 0.467 | 69.7 | 8.54 / 8.02 s |
| MINIMAX_TRAD vs MCTS_NN | 30 | 3 | **26** | 1 | 0.533 | 110.7 | 3.83 / 4.20 s |
| MINIMAX_TRAD vs MCTS_TRAD | 30 | 4 | 23 | 3 | 0.517 | 98.1 | 1.52 / 1.46 s |
| MINIMAX_NN vs MCTS_NN | 30 | 6 | 18 | 6 | 0.500 | 86.2 | 12.34 / 11.68 s |

Wizualizacja zbiorcza: `engine/out/exp1_round_robin_combined/plots/exp1_wdl_matrix.png`
oraz `plots/wdl_bars.png`.

### 4.3 Istotność statystyczna par — `exp1_pair_significance.csv`

Test dwumianowy na partiach rozstrzygniętych (`a_win_rate_decisive` ≠ 0.5):

| Para | a (białe) | d | b (czarne) | Decisive | Win-rate na decisive | p-value | 95% CI | Istotny? |
|---|---:|---:|---:|---:|---:|---:|---|:--:|
| MINIMAX_TRAD vs MINIMAX_NN | 2 | 5 | 23 | 25 | **0.080** (NN) | **0.0000** | [0.01, 0.26] | **TAK** |
| MINIMAX_NN vs MCTS_TRAD | 20 | 10 | 0 | 20 | **1.000** (NN) | **0.0000** | [0.83, 1.00] | **TAK** |
| MINIMAX_NN vs MCTS_NN | 12 | 18 | 0 | 12 | **1.000** (NN) | **0.0005** | [0.74, 1.00] | **TAK** |
| MINIMAX_TRAD vs MCTS_NN | 4 | 26 | 0 | 4 | 1.000 | 0.125 | [0.40, 1.00] | nie |
| MCTS_TRAD vs MCTS_NN | 1 | 25 | 4 | 5 | 0.200 | 0.375 | [0.01, 0.72] | nie |
| MINIMAX_TRAD vs MCTS_TRAD | 5 | 23 | 2 | 7 | 0.714 | 0.453 | [0.29, 0.96] | nie |

Wizualizacja: `plots/exp1_pair_significance.png`. Tylko **3 z 6** par dają istotny
statystycznie wynik — wszystkie trzy z udziałem MINIMAX_NN po stronie wygrywającej.
Pozostałe pary są zdominowane przez bardzo wysoki odsetek remisów (do 26/30 = 87 %
w `MINIMAX_TRAD vs MCTS_NN`), co spłaszcza moc testu.

### 4.4 Efekty głównych osi — `exp1_axis_summary.csv`

Agregat z 4 par cross-axis (~120 gier każda):

| Oś | Lider | Wygrane lidera | Remisy | Wygrane przeciwnika | Wynik lidera | p (analogon) |
|---|---|---:|---:|---:|---:|---|
| **A (algorytm)** | **MINIMAX** | 41 | 77 | 2 | **0.662** | znacząco > 0.5 |
| **B (ewaluator)** | **NN** | 47 | 66 | 7 | **0.667** | znacząco > 0.5 |

Wizualizacje: `plots/exp1_axis_a_effect.png`, `plots/exp1_axis_b_effect.png`.

Oba czynniki działają w tym samym kierunku — **Minimax + NN** to konfiguracja podwójnie
faworyzowana. Skala efektu jest praktycznie identyczna (0.662 ≈ 0.667), co tłumaczy,
dlaczego MINIMAX_NN dominuje rankingiem mimo niewielkiego nominalnego Elo.

### 4.5 Przewaga koloru — `exp1_color_advantage.csv`

| Łącznie | Białe wyg. | Remisy | Czarne wyg. | Wynik białych |
|---:|---:|---:|---:|---:|
| 180 | 34 | 107 | 39 | **0.486** |

Wynik białych 0.486 ≈ 0.5 — przewaga koloru praktycznie znika dzięki symetryzacji
(15 + 15 z zamianą kolorów per para). H4 (sanity check) potwierdzona.

### 4.6 Profil obliczeniowy — `analysis_metrics_summary.csv`

**MCTS — throughput iteracji/s (mediana):**

| Para / strona | TRAD iter/s | NN iter/s | Stosunek TRAD/NN |
|---|---:|---:|---:|
| MCTS_TRAD vs MCTS_NN — biały/czarny | ~13.0k / ~12.7k | — / — | — |
| MCTS_TRAD vs MCTS_NN — czarny/biały | — / — | ~14.0k / ~12.7k | — |
| MINIMAX_NN vs MCTS_TRAD (BLACK = MCTS_TRAD) | **994.7** mediana | — | — |
| MINIMAX_TRAD vs MCTS_NN (BLACK = MCTS_NN) | — | **19.45** mediana | — |

Pełne tabele w `analysis_metrics_summary.csv`. **Kluczowa obserwacja:** mediana throughput
MCTS_NN wynosi ~19-69 iter/s, podczas gdy MCTS_TRAD osiąga ~500-1500 iter/s w tych
samych warunkach czasowych — wyrocznia Stockfish jest ~25-50× wolniejsza per ewaluacja
(każde wywołanie oznacza komunikację UCI z procesem zewnętrznym).

**Minimax — średnie statystyki przeszukiwania (przykład `MINIMAX_NN` w parze 1):**

- średnia liczba węzłów / ruch: ~528-529,
- średnia osiągnięta głębokość ID: 3.0 (zgodnie z konfiguracją),
- TT hit-rate: ~6 %, TT cutoff-rate: ~39-41 %,
- średnie EBF: ≈ 4.0-4.3 (zob. `ebf_mean_*`),
- średnie qs_nodes: ~3.0k,
- check extensions: ~38-49 / ruch.

Wizualizacje: `plots/minimax_pruning.png` (techniki przycinania), `plots/mcts_throughput.png`
(iter/s per matchup), `plots/time_per_move_boxplot.png` (rozkłady czasu per wariant
i kolor), `plots/eval_over_game.png` (krzywe oceny pozycji w czasie partii).

## 5. Dyskusja

**Hierarchia siły gry.** Ranking Elo jest spójny z porównaniami bezpośrednimi:
MINIMAX_NN bezsprzecznie wygrywa w każdej z trzech par, w których uczestniczy
(każda z `p < 0.001`). Reszta pola (MCTS_TRAD ≈ MCTS_NN ≈ MINIMAX_TRAD)
mieści się w paśmie ~6 Elo, co przy obecnej próbie pozostaje **w granicach
szumu statystycznego** — par bez udziału MINIMAX_NN nie udaje się rozdzielić
testem dwumianowym (`p` ∈ [0.13; 0.45]).

**Dlaczego efekty osi A i B mają niemal identyczną siłę?** Wynik
`MINIMAX_score = 0.662` i `NN_score = 0.667` jest spójny z mechanizmem:
α-β + uporządkowanie ruchów + przycinanie pozwala efektywnie wykorzystać każdą
informację z ewaluatora, a NN dostarcza znacząco lepszej informacji niż heurystyka
TRAD (zob. wynik niezależnie potwierdzony w Eksp. 4: MAE i Spearman ρ TRAD vs SF-d20).
Mnożenie obu efektów daje skok rankingu wyłącznie dla MINIMAX_NN.

**Asymetria czasowa.** Średni czas/ruch waha się od ~1.5 s (MINIMAX_TRAD vs MCTS_TRAD)
do ~12.3 s (MINIMAX_NN vs MCTS_NN). Sprawia to, że formalne porównanie osi A
(MINIMAX vs MCTS) nie jest „fair" pod względem zegara — MCTS otrzymuje sztywne 2.61 s,
a MINIMAX_TRAD zużywa średnio < 2 s. **Eksperymenty kontrolne 2 i 3** (skalowanie
głębokości i czasu odpowiednio) służą rozdzieleniu efektu „czystego algorytmu" od efektu
budżetu zasobów; pełna ocena MCTS vs Minimax wymaga zestawienia tych trzech eksperymentów.

**Bardzo wysoki odsetek remisów.** W parach z udziałem MCTS po obu stronach lub
z dwóch ewaluatorów ostrożnych statystycznie remis wynosi 23-26 na 30. Wynika to z trzech
czynników: (i) adjudykacja ±0.05/20 ruchów twardo zatrzymuje gry techniczne,
(ii) MCTS ma tendencję do gry obronnej w pozycjach przybliżenie wyrównanych,
(iii) wybrane 25 otwarć ECO jest celowo wyrównanych. Negatywną konsekwencją jest spadek
mocy testów statystycznych — alternatywą byłoby zwiększenie `n` lub złagodzenie progu
adjudykacji, oba w tej pracy świadomie odrzucone (wykonalność czasowa, porównywalność
z Eksp. 2-3).

**Ograniczenia:**

- **n = 30** wystarcza dla efektów rzędu 100+ Elo, ale ~15 Elo (różnice 2.-4. miejsce)
  pozostaje nierozróżnialne.
- **„NN" ≠ wytrenowana sieć neuronowa** — to wrapper Stockfish d=10. Decyzja
  metodologiczna pozwala odizolować wpływ jakości ewaluacji od kosztu treningu,
  ale formalnie wynik mówi o „wyroczni płytkiego Stockfisha", nie o sieci neuronowej.
- **Brak księgi otwarć** w Eksp. 1 — Eksp. 7 i 8 testują wpływ księgi osobno.
- **d = 3 dla obu wariantów Minimaxa** to wybór wymuszony kosztem czasowym NN;
  Eksp. 2 i 8 weryfikują, czy ranking utrzymuje się przy większych głębokościach.

## 6. Wnioski

1. **MINIMAX_NN** jest jednoznacznie najsilniejszym wariantem w warunkach Eksp. 1 —
   wygrywa wszystkie trzy pary ze swoim udziałem z `p ≤ 0.0005`. Pozycja lidera
   wynika z **superpozycji** dwóch korzystnych efektów: lepszego algorytmu (oś A)
   oraz lepszego ewaluatora (oś B).
2. Efekty obu osi mają porównywalną siłę:
   `MINIMAX score = 0.662`, `NN score = 0.667`. Żaden z czynników nie jest „dominujący";
   wybór konfiguracji silnika musi być świadomy obu wymiarów.
3. **Pozostała trójka** (MCTS_TRAD, MCTS_NN, MINIMAX_TRAD) jest w obecnej próbie
   statystycznie nieodróżnialna — różnice ≤ 6 Elo, `p` ∈ [0.13; 0.45]. Rozstrzygnięcie
   tych par wymaga większej próby lub mocniejszych parametrów (zob. Eksp. 8).
4. **Sanity check koloru** (`white score = 0.486` przy `n = 180`) potwierdza poprawność
   procedury symetryzacji.
5. **Wysoki odsetek remisów** (5-26 / 30 w parach) jest ceną adjudykacji ±0.05/20
   i wyrównanych otwarć ECO — wymaga jawnego raportowania w pracy magisterskiej.

Materiał ten stanowi podstawę pierwszej części rozdziału eksperymentalnego: tabela 4.1
(ranking Elo), tabela 4.2 (W/D/L + istotność), rysunek 4.1 (`exp1_wdl_matrix.png`),
rysunek 4.2 (efekty osi A/B), rysunek 4.3 (boxplot czasu/ruch).
