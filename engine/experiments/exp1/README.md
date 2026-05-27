# Eksperyment 1 — Round-Robin między 4 wariantami silnika

## Cel

Ustalenie względnej siły wszystkich 4 wariantów silnika szachowego (oś A: algorytm, oś B: ewaluator), generacja rankingu Elo oraz macierzy W/R/P między każdą parą.

## Uczestnicy (4 warianty)

| Wariant | Algorytm | Ewaluator | Parametr zasobu |
|---|---|---|---|
| `MINIMAX_TRAD` | Minimax α-β | Heurystyczny | `depth = 3` |
| `MINIMAX_NN` | Minimax α-β | "NN" (Stockfish low-depth) | `depth = 3` |
| `MCTS_TRAD` | MCTS (PUCT) | Heurystyczny | `time = 2.61s` (skalibrowane) |
| `MCTS_NN` | MCTS (PUCT) | "NN" (Stockfish low-depth) | `time = 2.61s` |

**Dlaczego oba warianty Minimax mają tę samą głębokość d=3?**
- **Czyste porównanie Osi B (ewaluator)** — różni się tylko ewaluator, nie głębokość przeszukiwania
- **Wykonalne czasowo** — d=4 z ewaluatorem NN dawał ~30-50s/ruch (niepraktyczne); d=3 z NN to ~3-6s/ruch
- **Metodologicznie ciekawsze** — przy mniejszej głębokości, jakość ewaluatora ma **większy** wpływ (mniej kompensacji przez głębsze szukanie), co czyni Axis B bardziej widocznym

**Dlaczego MCTS dostaje 2.61s?** To wynik kalibracji — średnia czasu/ruch zmierzona z 3 partii MINIMAX_TRAD vs MINIMAX_TRAD (447 ruchów, dane w `out/exp1_calibration_*`), zapisana do `_mcts_calibrated_time.txt`. Dzięki temu MCTS dostaje znaczący budżet czasowy. **Uwaga o asymetrii:** MINIMAX_TRAD d=3 zużywa ~0.5s/ruch, MINIMAX_NN d=3 ~3-6s/ruch — MCTS przy 2.61s mieści się w środku, dając mu fair budget vs średni Minimax. Dla **czystego Axis A comparison** decydujący jest porównywalny budżet czasowy między algorytmami, a dla **Axis B** — porównywalna głębokość przeszukiwania między ewaluatorami.

## Struktura meczowa (6 par)

Każda para gra **N=30 partii** (15 w oryginalnym układzie kolorów + 15 z zamianą):

| # | Pair | Komentarz |
|---|---|---|
| 1 | MINIMAX_TRAD vs MINIMAX_NN | Czysta oś B (ewaluator) przy stałym algorytmie |
| 2 | MCTS_TRAD vs MCTS_NN | j.w. dla MCTS |
| 3 | MINIMAX_TRAD vs MCTS_TRAD | Czysta oś A (algorytm) przy stałym ewaluatorze |
| 4 | MINIMAX_TRAD vs MCTS_NN | Cross-axis (jedno z najważniejszych porównań) |
| 5 | MINIMAX_NN vs MCTS_TRAD | Cross-axis |
| 6 | MINIMAX_NN vs MCTS_NN | Cross-axis dla NN |

**Razem: 180 partii.** (Zgodnie z planem badawczym dla Exp 2/3 — jednolita metodologia. N=30 wykrywa różnice Elo ≥100 przy p<0.05; subtelniejsze różnice (<80 Elo) mogą być niestatystycznie istotne.)

## Otwarcia (kontrola wariancji)

Każda z 30 gier w parze startuje od jednej z **25 ustandaryzowanych pozycji ECO** (po 4 pełnych ruchach), z pliku `experiments/openings_eco25.fen`. Pozycje są cyklicznie wybierane modulo 25 (pozycje 1-25 dla pierwszych 25 gier; pozycje 1-5 powtórzone dla gier 26-30 w drugim cyklu — z zamianą kolorów).

Otwarcia: Italian, Ruy Lopez, Scotch, Four Knights, Petrov, Vienna, Sicilian (4 warianty), French, Caro-Kann, Pirc, Scandinavian, QGD (2), Slav, QGA, KID (2), Grünfeld, Nimzo-Indian, English, Catalan, Reti.

**Po co?** Redukcja wariancji otwarciowej — bez tego silniki grałyby te same opening lines każdorazowo (Minimax deterministyczny). Standardowa praktyka w testach typu CCRL/TCEC.

## Adjudykacja (terminacja partii)

Partia kończy się przez:
- **Standardowe reguły szachowe:** mat, pat, reguła 75 ruchów, 5-krotne powtórzenie, niedostateczny materiał
- **Adjudykacja eval-based:** gdy oba silniki oceniają pozycję w przedziale **±0.05** przez **20 kolejnych pełnych ruchów** → remis przez adjudykację

**Po co?** Bez adjudykacji partie MCTS vs MCTS mogłyby trwać setki ruchów w pat-podobnych pozycjach końcowych. Adjudykacja redukuje czas obliczeń ~30-50%.

## Książka otwarć

**Wyłączona** dla wszystkich wariantów. Testujemy surową siłę algorytmu, nie wpływ przygotowanego repertuaru.

## Procedura uruchomienia

### Krok 1 — Kalibracja (raz, ~5-10 min)

```powershell
.\experiments\exp1\run_exp1_calibrate.ps1
```

- Rozgrywa 3 partie MINIMAX_TRAD d=4 vs MINIMAX_TRAD d=4
- Parsuje JSONL, oblicza średni `time_s` per ruch
- Zapisuje do `experiments/exp1/_mcts_calibrated_time.txt`
- **Już wykonano:** wynik = **2.61s** (z 3 partii kalibracyjnych, 447 ruchów łącznie)
- Plik: `experiments/exp1/_mcts_calibrated_time.txt`
- Ponowna kalibracja: `.\experiments\exp1\run_exp1_calibrate.ps1` (default `-Depth 3`)

### Krok 2 — 6 par równolegle (~3-4h obliczeń wall-clock)

```powershell
# Terminal 1:
.\experiments\exp1\run_exp1_pair.ps1 -Pair 1   # MINIMAX_TRAD vs MINIMAX_NN
# Terminal 2:
.\experiments\exp1\run_exp1_pair.ps1 -Pair 2   # MCTS_TRAD vs MCTS_NN
# ... (terminale 3-6 dla par 3-6)
```

Każdy skrypt:
1. Czyta skalibrowany czas MCTS z pliku
2. Generuje single-pair config JSON
3. Woła `run_experiment.ps1` z wspólnym `-OutputSubDir = exp1_round_robin_<data>`
4. Wszystkie 6 par pisze do **tego samego katalogu** (dzięki shared `OutputSubDir`)

### Krok 3 — Analiza zbiorcza (~2 min)

```powershell
.\experiments\exp1\run_exp1_analyze.ps1
```

Wykonuje dwie fazy:
- **Faza 1:** `analyze_experiment.py --elo --plots` (analiza generyczna)
- **Faza 2:** `exp1_round_robin.py` (analiza specyficzna)

## Wyjście — pliki CSV

Katalog wyjściowy: `engine/out/exp1_round_robin_<data>/`

| Plik | Zawartość |
|---|---|
| `_results.csv` | Per-game: pair, game #, opening idx, result, termination, czas |
| `analysis_moves.csv` | Per-move: eval, czas, fazę gry, wszystkie metryki algorytmu (MCTS iterations, Minimax nodes, TT hits, pruning stats, etc.) |
| `analysis_games.csv` | Per-game: result, total moves, termination reason, czas |
| `analysis_wdl.csv` | Per-matchup: 30 gier × {wygrane białe, remisy, wygrane czarne}, white_score, avg_moves, avg_time |
| `analysis_elo.csv` | Bradley-Terry maximum likelihood Elo dla 4 wariantów |
| `analysis_metrics_summary.csv` | Per-matchup × side: mean/std/median wszystkich metryk algorytmu |
| `exp1_pair_significance.csv` | Per-para: binomial test na decisive games, p-value, 95% CI |
| `exp1_axis_summary.csv` | Główne efekty: MINIMAX vs MCTS (oś A) i TRAD vs NN (oś B) — agregat z cross-axis par |
| `exp1_color_advantage.csv` | Overall White vs Black win rate |

## Wyjście — wykresy (`plots/`)

| Plik | Co pokazuje |
|---|---|
| `wdl_bars.png` | Słupki W/D/L per matchup |
| `time_per_move_boxplot.png` | Boxplot czasu/ruch per wariant+kolor |
| `eval_over_game.png` | Krzywe ewaluacji w czasie gry (sample 6 gier) |
| `minimax_pruning.png` | Średnie użycie technik przycinania per matchup (Minimax) |
| `mcts_throughput.png` | Iteracje/s per matchup (MCTS) |
| `exp1_pair_significance.png` | Słupki per para z 95% CI, p-value w etykiecie. Zielone = istotne statystycznie |
| `exp1_axis_a_effect.png` | MINIMAX vs MCTS agregat (oś A — algorytm) |
| `exp1_axis_b_effect.png` | TRAD vs NN agregat (oś B — ewaluator) |
| `exp1_wdl_matrix.png` | 4×4 heatmap score per (variant, opponent) |

## Co mierzymy per ruch (zbierane w JSONL)

**Wspólne (każdy ruch):** numer ruchu, strona, UCI, eval, czas (s), faza gry [0..1]

**MCTS-specific (12 metryk):**
- `iterations` — łączna liczba iteracji MCTS
- `nodes_created`, `max_depth` — rozmiar i głębokość drzewa
- `eval_calls`, `eval_cache_hits` — zliczanie + skuteczność cache
- `skipped_terminals`, `reused_visits` — optymalizacje
- `root_children_count`, `best_child_visits` — rozkład wizyt korzenia
- `root_visit_entropy` — pewność przeszukiwania (Shannon entropy)
- `convergence_point` — w której frakcji budżetu pojawił się finalny best move
- `avg_backprop_depth` — średnia długość ścieżki backpropagacji
- `c_puct` — parametr eksploracji

**Minimax-specific (21+ metryk):**
- `nodes_searched`, `depth_completed`, `tt_size`
- `tt_lookups`, `tt_hits`, `tt_cutoffs` (z których obliczane `tt_hit_rate`, `tt_cutoff_rate`)
- `nmp_attempts`, `nmp_cutoffs`, `rfp_cutoffs`, `futility_prunes`, `lmp_prunes`, `see_prunes`
- `check_extensions`, `aspiration_researches`
- `qs_nodes`, `qs_max_depth`, `see_calls`
- `killer_hits`, `killer_checks`, `pv_from_tt`
- `nodes_per_depth` — lista węzłów per ID iteration (z której obliczany **EBF**)

## Analizy statystyczne

1. **Bradley-Terry maximum likelihood Elo** — wszystkie 4 warianty na jednej skali Elo, zakotwiczenie mean=0
2. **Binomial test per para** — H0: warianty są równo silne; p-value + 95% CI na decisive games
3. **Axis A effect** (algorithm): agreguje 4 cross-axis pary (MINIMAX_x vs MCTS_y) → który algorytm wygrywa średnio
4. **Axis B effect** (evaluator): agreguje 4 cross-axis pary (X_TRAD vs Y_NN) → który ewaluator wygrywa średnio
5. **Color advantage**: wskazuje czy białe mają przewagę w naszych warunkach (sanity check)

## Co z planu badawczego jest istotne dla pracy magisterskiej

Eksperyment 1 odpowiada na **kluczowe pytanie pracy**: który z 4 wariantów jest najsilniejszy, i czy przewaga wynika z **algorytmu** (oś A) czy **ewaluatora** (oś B)?

W pracy zazwyczaj prezentuje się:

- **Tabela:** macierz W/R/P 4×4 (z wykresu `exp1_wdl_matrix.png`)
- **Tabela:** ranking Elo (z `analysis_elo.csv`)
- **Wykres słupkowy:** Axis A vs Axis B main effects (z `exp1_axis_a_effect.png`, `exp1_axis_b_effect.png`)
- **Wykres pudełkowy:** rozkład czasu/ruch (boxplot)
- **Tabela:** istotność statystyczna (p-values per para)

## Szacunek czasu obliczeń

Średnie czasy/ruch przy obecnej konfiguracji (d=3 Minimax, 2.61s MCTS):
- MINIMAX_TRAD d=3: ~0.5s/ruch
- MINIMAX_NN d=3: ~3-6s/ruch (Stockfish jako oracle eval)
- MCTS_TRAD: 2.61s/ruch (fixed)
- MCTS_NN: 2.61s/ruch (fixed)

Szacunkowe czasy par (30 gier, ~80 ruchów/gra):
- **Para 1** (MINIMAX_TRAD vs MINIMAX_NN): zdominowana przez NN side, ~3-4h
- **Para 2** (MCTS_TRAD vs MCTS_NN): ~3-4h
- **Para 3** (MINIMAX_TRAD d=3 vs MCTS_TRAD 2.61s): ~2h
- **Para 4** (MINIMAX_TRAD d=3 vs MCTS_NN 2.61s): ~2h
- **Para 5** (MINIMAX_NN vs MCTS_TRAD): ~3-4h
- **Para 6** (MINIMAX_NN vs MCTS_NN): ~3-4h
- **6 par równolegle:** ~3-4h wall-clock
- **Sekwencyjnie:** ~16-20h
- **+ kalibracja:** już wykonana

**Konsekwencja wyboru d=3 dla Minimax + MCTS 2.61s (z kalibracji d=4):** MCTS dostaje znaczący budżet czasowy mimo niższej głębokości Minimax. Można interpretować jako **podwójne sprawdzenie Axis A**: MCTS z 2.61s vs Minimax d=3 (~0.5s) testuje czy MCTS umie wykorzystać dodatkowy czas; MCTS z 2.61s vs Minimax NN d=3 (~3-6s) testuje czy MCTS poradzi sobie z silniejszym przeciwnikiem mającym podobny budżet. W pracy warto otwarcie omówić tę asymetrię i odwołać się do Eksp. 2/3 jako kontrolnych dla skalowania zasobów.

## Ważne uwagi praktyczne

1. **Kalibracja już wykonana** — `_mcts_calibrated_time.txt` zawiera `2.61` (s/ruch)
2. **Wszystkie 6 par MUSI być uruchomione tego samego dnia** żeby trafiły do tego samego shared dir (tag jest oparty na dacie, format `yyyyMMdd`)
3. **Adjudykacja jest WŁĄCZONA** — bez tego MCTS vs MCTS może produkować bardzo długie partie
4. **Każdy plik metryczny waży ~50-200KB** — 180 gier × ~100KB = ~18MB danych
5. **Quick smoke test:** `.\experiments\exp1\run_exp1_pair.ps1 -Pair 1 -GamesPerPair 2` (4 minuty, sprawdza pipeline)
