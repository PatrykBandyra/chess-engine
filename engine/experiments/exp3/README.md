# Eksperyment 3 — Skalowanie siły MCTS z budżetem czasowym

## Cel

Zmierzenie jak siła gry MCTS rośnie wraz z budżetem czasowym (oś A — wrażliwość zasobowa, analogicznie do Eksp. 2 dla głębokości Minimaxa). Pokazanie krzywej **Elo vs log(time)** z dopasowaniem log-liniowym: o ile Elo zyskujemy przy każdym podwojeniu czasu, dla obu ewaluatorów (TRAD i NN).

## Uczestnicy (10 matchupów)

Analogicznie do Eksp. 2, każdy matchup to **pojedynek MCTS o testowanym budżecie czasowym vs MCTS o stałym czasie referencyjnym = 20s**, dla każdego z 5 testowanych czasów (t ∈ {1, 5, 10, 20, 40}s) i każdego z 2 ewaluatorów:

| # | Matchup | Komentarz |
|---|---|---|
| 1 | MCTS_TRAD 1s vs MCTS_TRAD 20s | Mały budżet → spodziewana porażka |
| 2 | MCTS_TRAD 5s vs MCTS_TRAD 20s | Mierzymy ΔElo |
| 3 | MCTS_TRAD 10s vs MCTS_TRAD 20s | |
| 4 | MCTS_TRAD 20s vs MCTS_TRAD 20s | **Sanity check** — Elo ≈ 0 ± kilkadziesiąt |
| 5 | MCTS_TRAD 40s vs MCTS_TRAD 20s | Większy budżet, mierzymy zysk ⚠ kosztowny |
| 6 | MCTS_NN 1s vs MCTS_NN 20s | Powtórzenie 1-5 dla ewaluatora NN |
| 7 | MCTS_NN 5s vs MCTS_NN 20s | |
| 8 | MCTS_NN 10s vs MCTS_NN 20s | |
| 9 | MCTS_NN 20s vs MCTS_NN 20s | Sanity check NN |
| 10 | MCTS_NN 40s vs MCTS_NN 20s | ⚠ kosztowny |

**Razem: 10 matchupów × 30 partii = 300 partii.**

**Dlaczego 20s jako poziom odniesienia (anchor)?** 20s to "środkowy" budżet w siatce {1, 5, 10, 20, 40} (geometrycznie ≈ 2-3× kalibrowany czas z exp1 = 2.61s × ~8). Daje rozsądną pewność wyboru ruchu (>3000 iteracji TRAD, ~50-300 iteracji NN) przy umiarkowanym koszcie wall-clock. Krzywa Elo jest kotwiczona w punkcie t=20s → Elo = 0; pozostałe punkty wyliczane są względem niego przez Bradley-Terry maximum likelihood.

**Dlaczego dwa ewaluatory osobno?** Aby krzywa **Elo vs log(time)** była rysowana **osobno dla TRAD i NN** — można porównać tempo wzrostu siły: NN ma drastycznie mniejszy throughput (mniej rolloutów/s ze względu na koszt Stockfish-oracle eval), ale każdy rollout jest "głębszy" jakościowo. Czy NN daje więcej Elo na sekundę zegara, czy mniej?

**Geometryczna seria budżetów** (1, 5, 10, 20, 40 — log2: ~0, 2.3, 3.3, 4.3, 5.3): umożliwia **log-liniowe dopasowanie** Elo ≈ α · log₂(t) + β, gdzie α = "Elo per doubling of time".

## Otwarcia (kontrola wariancji)

Identycznie jak w Eksp. 1 i 2: 25 ustandaryzowanych pozycji ECO z `experiments/openings_eco25.fen`, cyklicznie po jednej dla każdej z 30 partii w matchupie (pozycje 1-25 dla pierwszych 25 gier; 1-5 powtórzone dla gier 26-30 z zamianą kolorów).

## Adjudykacja

**WŁĄCZONA** z domyślnymi parametrami (`±0.05`, `20 ruchów`). **Krytyczne dla Eksp. 3**: MCTS vs MCTS w pat-podobnych końcówkach może trwać setki ruchów; bez adjudykacji matchupy 40s vs 20s mogłyby się nigdy nie skończyć.

## Książka otwarć

**Wyłączona** — testujemy czystą siłę algorytmu MCTS przy różnych budżetach.

## Kalibracja MCTS

**Niepotrzebna jako wejście** — Eksp. 3 jawnie ustawia budżety (1, 5, 10, 20, 40s). Ale wyniki Eksp. 3 mogą posłużyć do **walidacji** wartości kalibrowanej z exp1 (`_mcts_calibrated_time.txt` = 2.61s): jeśli Elo(2.61s) leży na krzywej między t=1s a t=5s, kalibracja jest spójna.

## Procedura uruchomienia

### Krok 1 — Uruchom 10 matchupów (równolegle)

```powershell
# Terminal 1:
.\experiments\exp3\run_exp3_matchup.ps1 -Matchup 1   # MCTS_TRAD 1s vs 20s
# Terminal 2:
.\experiments\exp3\run_exp3_matchup.ps1 -Matchup 2   # MCTS_TRAD 5s vs 20s
# ... (terminale 3-10 dla matchupów 3-10)
```

Każdy skrypt:
1. Generuje single-matchup config JSON (`_exp3_matchup{N}.json`)
2. Woła `run_experiment.ps1` z wspólnym `-OutputSubDir = exp3_mcts_time_<data>`
3. Wszystkie 10 matchupów pisze do **tego samego katalogu**

**Wszystkie 10 matchupów MUSI być uruchomione tego samego dnia** (tag oparty na dacie `yyyyMMdd`). Jeśli musisz przerwać i wznowić innego dnia, przekaż jawnie `-ExperimentTag` przy każdym wywołaniu.

**Quick smoke test:** `.\experiments\exp3\run_exp3_matchup.ps1 -Matchup 4 -GamesPerPair 2` (t=20 vs t=20, sanity check, ~30 min).

### Krok 2 — Analiza zbiorcza (~2 min)

```powershell
.\experiments\exp3\run_exp3_analyze.ps1
# lub jawnie:
.\experiments\exp3\run_exp3_analyze.ps1 -ExperimentDir engine\out\exp3_mcts_time_20260530
```

Wykonuje dwie fazy:
- **Faza 1:** `analyze_experiment.py --elo --plots` (analiza generyczna)
- **Faza 2:** `exp3_time_scaling.py` (analiza specyficzna — krzywa Elo vs log(time), log-linear fit, throughput, tree size, depth, entropy)

## Wyjście — pliki CSV

Katalog wyjściowy: `engine/out/exp3_mcts_time_<data>/`

| Plik | Zawartość |
|---|---|
| `_results.csv` | Per-game: matchup, game #, opening idx, result, termination, czas |
| `analysis_moves.csv` | Per-move: eval, czas, fazę gry, wszystkie metryki MCTS |
| `analysis_games.csv` | Per-game: result, total moves, termination reason, czas |
| `analysis_wdl.csv` | Per-matchup: 30 gier × W/D/L, white_score, avg_moves, avg_time |
| `analysis_elo.csv` | Bradley-Terry Elo dla wszystkich 5 czasów × 2 ewaluatory (mixed) |
| `analysis_metrics_summary.csv` | Per-matchup × side: mean/std/median metryk algorytmu |
| `exp3_elo_per_time.csv` | **Kluczowy:** Elo per (ewaluator, czas), z kotwicą w t=20s |
| `exp3_elo_log_fit.csv` | Parametry dopasowania log-liniowego: α (Elo per doubling), β (intercept), R² |
| `exp3_time_summary.csv` | Per-(ewaluator, czas): avg_iterations, throughput (iter/s), tree size, max_depth, entropy |
| `exp3_time_summary.txt` | Human-readable raport |

## Wyjście — wykresy (`plots/`)

| Plik | Co pokazuje |
|---|---|
| `wdl_bars.png` | Słupki W/D/L per matchup |
| `time_per_move_boxplot.png` | Boxplot czasu/ruch per matchup |
| `eval_over_game.png` | Krzywe ewaluacji w czasie gry (próbka) |
| `mcts_throughput.png` | Generyczne iter/s per matchup |
| `exp3_elo_curve.png` | **Kluczowy wykres** — Elo vs log₂(time), 2 linie (TRAD/NN) + log-linear fit |
| `exp3_throughput_curve.png` | Iter/s vs czas, TRAD vs NN (NN powinien mieć dużo niższy throughput) |
| `exp3_tree_size_curve.png` | `nodes_created` vs czas (skala log-log) |
| `exp3_max_depth_curve.png` | Głębokość drzewa vs czas |
| `exp3_entropy_curve.png` | Root visit entropy vs czas (powinna maleć — większa pewność wyboru) |

## Co mierzymy per ruch (zbierane w JSONL)

**Wspólne:** numer ruchu, strona, UCI, eval (centypionki), czas (s), faza gry [0..1]

**MCTS-specific (12 metryk — wszystkie istotne dla Eksp. 3):**
- `iterations` — **kluczowe dla throughput** (iter/s = iterations / time)
- `nodes_created`, `max_depth` — rozmiar i głębokość drzewa
- `eval_calls`, `eval_cache_hits` — zliczanie + skuteczność cache
- `skipped_terminals`, `reused_visits` — optymalizacje
- `root_children_count`, `best_child_visits` — rozkład wizyt korzenia
- `root_visit_entropy` — **pewność przeszukiwania (Shannon entropy)** — powinna maleć z budżetem
- `convergence_point` — w której frakcji budżetu pojawił się finalny best move
- `avg_backprop_depth` — średnia długość ścieżki backpropagacji
- `c_puct` — parametr eksploracji

## Analizy statystyczne

1. **Elo per czas** — Bradley-Terry ML, oddzielnie dla każdego ewaluatora, kotwica = t=20s Elo = 0
2. **Log-liniowe dopasowanie krzywej Elo** — `Elo ≈ α · log₂(t) + β`; α = "Elo per doubling of time" (typowo 50-150 dla MCTS)
3. **Throughput** — `iter/s = iterations / time` per (ewaluator, czas); powinno być względnie stałe per ewaluator (skaluje liniowo z czasem)
4. **Tree size / max depth vs czas** — dopasowanie log-log: typowo `nodes ∝ t^0.9` (subliniowo z powodu cache hits)
5. **Entropy vs czas** — pewność wyboru ruchu; powinna maleć z większym budżetem (bardziej zdecydowane wizyty głównego dziecka)

## Co z planu badawczego jest istotne dla pracy magisterskiej

Eksperyment 3 odpowiada na pytanie: **jak skaluje się siła MCTS z czasem?** Z czego wynika:
- **Czy 2.61s (kalibracja z exp1) to rozsądny budżet** dla porównań w exp1
- **Czy NN skaluje lepiej czy gorzej niż TRAD** — czy "drogi ale dokładny" ewaluator daje większą stopę zwrotu Elo per sekundę?
- **Predykcja siły** dla większych budżetów (60s, 120s) z ekstrapolacji log-liniowej
- **Porównanie z Eksp. 2 (głębokość Minimaxa)** — który algorytm lepiej skaluje z zasobami?

W pracy zazwyczaj prezentuje się:
- **Wykres:** krzywa Elo vs log(time), 2 linie (TRAD/NN) + dopasowanie liniowe — `exp3_elo_curve.png`
- **Tabela:** α (Elo per doubling) per ewaluator — z `exp3_elo_log_fit.csv`
- **Wykres:** throughput TRAD vs NN — pokazuje **cenę jakości** ewaluatora
- **Wykres:** tree size / depth vs czas — pokazuje wzrost przestrzeni przeszukania
- **Porównanie z Eksp. 2:** kto wygrywa wyścig "Elo per sekundę" — Minimax+głębokość czy MCTS+czas?

## Szacunek czasu obliczeń

Per-move budget MCTS jest dosłownie wall-clock — silnik myśli przez zadany czas. Średnia per matchup = (white_time + black_time) / 2. Przy ~60-80 ruchach/gra (z adjudykacją) × 30 gier:

| # | Matchup | Avg czas/ruch | Szac. czas matchupu (z adjudykacją) |
|---|---|---|---|
| 1 | TRAD 1s vs 20s | ~10.5s | ~7-10h |
| 2 | TRAD 5s vs 20s | ~12.5s | ~8-12h |
| 3 | TRAD 10s vs 20s | ~15s | ~10-14h |
| 4 | TRAD 20s vs 20s | ~20s | ~13-18h |
| 5 | TRAD 40s vs 20s | ~30s | ~20-28h ⚠ |
| 6 | NN 1s vs 20s | ~10.5s | ~7-10h |
| 7 | NN 5s vs 20s | ~12.5s | ~8-12h |
| 8 | NN 10s vs 20s | ~15s | ~10-14h |
| 9 | NN 20s vs 20s | ~20s | ~13-18h |
| 10 | NN 40s vs 20s | ~30s | ~20-28h ⚠ |

**Uwaga: MCTS używa wall-clock budgetu, więc TRAD i NN mają **identyczny czas/ruch** przy tym samym budżecie** — różnią się tylko liczbą iteracji wykonanych w tym samym czasie (NN ma drastycznie mniej rolloutów, ale każdy "głębszy" jakościowo).

**Sumarycznie:**
- Wszystkie 10 matchupów serialnie: **~120-160h** (~5-7 dni)
- 10 matchupów równolegle: **bottleneck = matchupy 5 i 10 (40s, ~25h każdy)** ~24h wall-clock
- **Optymalna strategia:** uruchom matchupy 1-4 i 6-9 równolegle (~18h), matchupy 5 i 10 osobno w tle

## Ważne uwagi praktyczne

1. **MCTS używa wall-clock budgetu** — dosłownie tyle, ile zostało podane; brak heurystyk wczesnego zakończenia
2. **NN ma identyczny wall-clock budget jak TRAD** — różnica wyłącznie w throughput (iter/s), nie w czasie/ruch
3. **Wszystkie 10 matchupów MUSI być tego samego dnia** (tag domyślnie oparty na dacie, format `yyyyMMdd`)
4. **Adjudykacja jest KRYTYCZNA** — bez niej MCTS vs MCTS może produkować bardzo długie partie (200+ ruchów w pat-podobnych pozycjach)
5. **Każdy plik metryczny waży ~100-300KB** — 300 gier × ~200KB ≈ ~60MB danych
6. **Matchupy 5 i 10 (40s vs 20s) są heavy** — jeśli ograniczony czasem, rozważ obniżenie `-GamesPerPair 15` dla nich (kosztem szerokości CI)
7. **Kalibracja niepotrzebna jako input** — jawne budżety; ale wynik exp3 może retrospektywnie zwalidować wartość 2.61s z exp1
8. **Sanity check 4 i 9** (20s vs 20s) — Elo powinno być ≈ 0 ± kilkadziesiąt. Jeśli wyraźnie ≠ 0, sygnał problemu z determinizmem MCTS lub asymetrią w pętli gry

## Co do dyskusji w pracy

- **Krzywa malejących zwrotów** — typowo w MCTS ΔElo per podwojenie czasu maleje (~80-150 Elo przy małych budżetach, ~30-60 przy dużych). Dopasowanie log-liniowe powinno dawać R² > 0.95 przy dobrze dobranych punktach
- **Porównanie tempa wzrostu TRAD vs NN** — jeśli NN "rośnie" wolniej (α niższe), oznacza to że bottleneckiem jest **liczba rolloutów**, nie ich jakość. Jeśli NN rośnie szybciej, każdy rollout NN jest "wart więcej" — ale to dziwny wynik, bo NN powinien już mieć lepszy baseline
- **Throughput TRAD vs NN** — typowo TRAD: ~5000-15000 iter/s, NN: ~50-300 iter/s (200× mniej!). Czy NN nadrabia jakością — patrz krzywa Elo
- **Porównanie z Eksp. 2** — jaki budżet czasowy MCTS odpowiada jaką głębokością Minimaxa? np. MCTS_TRAD 20s ≈ MINIMAX_TRAD d=? — daje praktyczną zasadę konwersji
- **Walidacja kalibracji z exp1** — Elo(2.61s) z ekstrapolacji powinno leżeć między t=1s a t=5s; jeśli istotnie różne, kalibracja może być źle zrobiona
- **Czy ekstrapolacja na większe budżety jest sensowna?** — log-linear fit zakłada że krzywa się NIE wypłaszcza w mierzonym zakresie; jeśli przy t=40s widać wypłaszczenie, ekstrapolacja na t=120s daje górne ograniczenie, nie estymatę
