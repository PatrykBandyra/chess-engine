# Eksperyment 5 — Benchmark Stockfishem (kalibracja Elo)

## Cel

Zmierzenie **siły absolutnej** każdego z 4 wariantów silnika przez konfrontację ze Stockfishem na 8 poziomach skill (0-20). W odróżnieniu od Eksp. 1 (gdzie Elo jest *względne* i wszystkie warianty są na jednej skali kotwiczonej do siebie nawzajem), Eksp. 5 daje **Elo na zewnętrznej, kalibrowanej skali** — można powiedzieć "wariant X gra na poziomie ~1800 Elo CCRL".

Dodatkowo: **ACPL** (Average CentiPawn Loss) per wariant — niezależna od wyniku miara jakości decyzji, oparta na re-ewaluacji Stockfishem d=20.

## Uczestnicy (32 matchupy)

| Wariant | Algorytm | Ewaluator | Parametr |
|---|---|---|---|
| `minimax_trad_d4` | Minimax α-β | Heurystyczny | `depth = 4` |
| `minimax_nn_d3` | Minimax α-β | "NN" (Stockfish low-depth) | `depth = 3` |
| `mcts_trad` | MCTS (PUCT) | Heurystyczny | `time = 2.61s` (skalibrowane z exp1) |
| `mcts_nn` | MCTS (PUCT) | "NN" | `time = 2.61s` |

Każdy wariant gra **8 matchupów** vs Stockfish, po jednym na każdy poziom skill:

| Skill | Approx. Elo (CCRL) | Komentarz |
|---|---|---|
| 1 | 900 | Bardzo słaby (skill 0 omijany — interakcja z biblioteką stockfish powoduje crash) |
| 3 | 1100 | Początkujący-średniozaawansowany |
| 5 | 1400 | Średniozaawansowany klubowy |
| 8 | 1700 | Mocny klubowy |
| 10 | 2000 | Mistrz klubowy / FM |
| 13 | 2400 | IM |
| 15 | 2700 | GM |
| 20 | 3500 | Full strength |

(Mapowanie skill → Elo z pliku `_sf_skill_elo.csv`)

**Razem: 4 warianty × 8 matchupów × 20 partii = 640 partii.** ⚠ Najdroższy obliczeniowo eksperyment.

**Dlaczego Stockfish (skill + depth=10)?** Skill 0-20 to oficjalny parametr UCI Stockfisha modulujący siłę. Dodatkowo Stockfish jest puszczany na **stałej głębokości d=10** (parametr `-StockfishDepth`), żeby decyzje były powtarzalne i czas/ruch przewidywalny. Skill mapowane jest na Elo CCRL przez tabelę referencyjną — pozwala interpolować Elo wariantu z krzywej winrate.

**Dlaczego 20 gier per skill level** (a nie 30 jak w exp1-3)? Bo łącznie i tak mamy **160 partii per wariant** (8 levels × 20) — wystarczająco dużo do dopasowania krzywej "score vs SF Elo".

**Dlaczego te same warianty co exp1?** Spójność. Można porównać Elo "internal" (z exp1) z Elo "external" (z exp5) — sanity check. Spodziewane: ranking ten sam, ale wartości Elo mogą się różnić skali (exp1 mean=0; exp5 absolutne).

## Otwarcia (kontrola wariancji)

Identycznie jak w Eksp. 1-3: 25 pozycji ECO z `experiments/openings_eco25.fen`, cyklicznie po jednej dla każdej z 20 partii w matchupie (pozycje 1-20).

## Adjudykacja

**WŁĄCZONA** z domyślnymi parametrami (`±0.05`, `20 ruchów`). Bez tego mecze MCTS vs Stockfish low-skill w pat-podobnych końcówkach mogłyby trwać setki ruchów.

## Książka otwarć

**Wyłączona** dla obu stron — testujemy czystą siłę algorytmów. Stockfish w UCI bez ustawienia `UCI_LimitStrength` używa swojej wewnętrznej, ale tę można potraktować jako "standardowy benchmarkowy przeciwnik" zgodnie z konwencją CCRL.

## Procedura uruchomienia

### Krok 1 — Kalibracja MCTS (raz, jeśli nie wykonana)

```powershell
.\experiments\exp1\run_exp1_calibrate.ps1
```

Eksp. 5 czyta `experiments\exp1\_mcts_calibrated_time.txt`. Jeśli plik nie istnieje, skrypt zakończy się błędem. Domyślna wartość po kalibracji exp1 = `2.61s`.

### Krok 2 — 4 warianty równolegle (~kilkadziesiąt godzin)

```powershell
# Terminal 1:
.\experiments\exp5\run_exp5_variant.ps1 -Variant 1   # MINIMAX_TRAD d=4 vs 8 SF skills
# Terminal 2:
.\experiments\exp5\run_exp5_variant.ps1 -Variant 2   # MINIMAX_NN d=3
# Terminal 3:
.\experiments\exp5\run_exp5_variant.ps1 -Variant 3   # MCTS_TRAD calib
# Terminal 4:
.\experiments\exp5\run_exp5_variant.ps1 -Variant 4   # MCTS_NN calib
```

Każdy skrypt:
1. Czyta skalibrowany MCTS time (dla wariantów MCTS)
2. Generuje config JSON z 8 matchupami (1 per skill level)
3. Woła `run_experiment.ps1` z wspólnym `-OutputSubDir = exp5_stockfish_<data>`
4. Wszystkie 4 warianty piszą do **tego samego katalogu** (= 640 gier w jednym dir-rze)

**Wszystkie 4 warianty MUSI być uruchomione tego samego dnia** (tag domyślnie oparty na dacie, `yyyyMMdd`). Jeśli musisz przerwać, przekaż jawnie `-ExperimentTag`.

**Quick smoke test:** `.\experiments\exp5\run_exp5_variant.ps1 -Variant 1 -GamesPerPair 2` (16 gier × ~1 min ≈ 15-20 min).

**Override:** `-GamesPerPair`, `-StockfishDepth`, `-MctsTime`, `-StockfishPath`.

### Krok 3 — Analiza zbiorcza (3 fazy, długa)

```powershell
# Pełna analiza (Faza 2 jest droga — re-ewaluacja Stockfishem d=20 każdego ruchu):
.\experiments\exp5\run_exp5_analyze.ps1

# Pominięcie re-ewaluacji (szybsze, ale bez ACPL):
.\experiments\exp5\run_exp5_analyze.ps1 -SkipReval

# Re-ewaluacja na mniejszej głębokości (~10× szybciej, mniejsza precyzja):
.\experiments\exp5\run_exp5_analyze.ps1 -RevalDepth 15

# Test pipeline'u: re-ewaluacja tylko pierwszych 20 gier:
.\experiments\exp5\run_exp5_analyze.ps1 -RevalLimit 20
```

Wykonuje trzy fazy:
- **Faza 1:** `analyze_experiment.py --elo --plots` (CSVs, generyczne wykresy)
- **Faza 2:** `stockfish_reval.py --depth 20` — przepuszcza każdy ruch przez Stockfisha d=20, oblicza centypionkową stratę. **Bardzo wolne** (~kilkanaście minut do kilku godzin per 100 gier zależnie od sprzętu)
- **Faza 3:** `exp5_stockfish_bench.py` — interpolacja Elo wariantu z krzywej score vs SF Elo, ACPL summary, blunder rate

## Wyjście — pliki CSV

Katalog wyjściowy: `engine/out/exp5_stockfish_<data>/`

| Plik | Zawartość |
|---|---|
| `_results.csv` | Per-game: variant_label, skill, game #, opening idx, result, termination, czas |
| `analysis_moves.csv` | Per-move: eval, czas, faza gry, metryki algorytmu |
| `analysis_games.csv` | Per-game: result, total moves, termination, czas |
| `analysis_wdl.csv` | Per-matchup (32): 20 gier × W/D/L, white_score, avg_moves, avg_time |
| `analysis_elo.csv` | Bradley-Terry Elo (mieszany — Stockfish skill 20 jako anchor wysoki) |
| `stockfish_reval.csv` | **Faza 2:** Per-move: stockfish_eval_d20, centipawn_loss, is_blunder (>200cp loss) |
| `exp5_variant_summary.csv` | **Kluczowy:** Per-(wariant, skill): score, win_rate, avg_acpl, blunder_rate |
| `exp5_variant_elo.csv` | **Kluczowy:** Interpolowane Elo każdego wariantu (z krzywej score vs SF Elo) |
| `exp5_variant_summary.txt` | Human-readable raport |

## Wyjście — wykresy (`plots/`)

| Plik | Co pokazuje |
|---|---|
| `wdl_bars.png` | Generyczny W/D/L per matchup (32 słupki) |
| `time_per_move_boxplot.png` | Boxplot czasu/ruch per wariant + per skill |
| `eval_over_game.png` | Krzywe ewaluacji (próbka) |
| `exp5_score_curve.png` | **Kluczowy wykres** — score (0-1) vs SF Elo, 4 linie (po jednej na wariant) + interpolacja Elo wariantu |
| `exp5_acpl_by_variant.png` | ACPL per wariant (słupki) — niezależna od wyniku miara jakości |
| `exp5_acpl_by_phase.png` | ACPL stratyfikowany po fazie gry (otwarcie / midgame / endgame) |
| `exp5_blunder_rate.png` | Blunder rate (% ruchów z stratą >200cp) per wariant |

## Co mierzymy per ruch

**Wspólne (z każdego JSONL):** numer ruchu, strona, UCI, eval (centypionki engine-internal), czas, faza gry [0..1]

**Po re-ewaluacji Stockfishem (Faza 2):**
- `stockfish_eval_d20` — ground-truth eval ruchu (po jego wykonaniu) z Stockfisha d=20
- `centipawn_loss` — różnica między najlepszym ruchem (d=20) a faktycznie wykonanym
- `is_blunder` — bool, True jeśli `centipawn_loss > 200`

Plus wszystkie metryki Minimax/MCTS znane z exp1-3.

## Analizy statystyczne

1. **Interpolacja Elo wariantu** — z krzywej `score vs SF_Elo` przez fit logistyczny: punkt gdzie score = 0.5 to estimated Elo wariantu
2. **ACPL per wariant** — miara *jakości decyzji* niezależna od wyniku gry; typowo 30-80 dla wariantów ~2000 Elo, <30 dla GM-poziomu
3. **Blunder rate** — % ruchów z centipawn_loss > 200; pokazuje "konsystencję" wariantu
4. **ACPL by phase** — czy wariant gra gorzej w otwarciu (brak książki) czy końcówce (słaby endgame eval)?
5. **Win rate breakdown per SF skill** — krzywa wzrostu skłonności do remisów + spadku wygranych z rosnącym skill

## Co z planu badawczego jest istotne dla pracy magisterskiej

Eksperyment 5 odpowiada na pytanie: **na jakim absolutnym poziomie gry jest każdy wariant?** Daje:

- **Elo na skali zewnętrznej** (CCRL-podobnej) zamiast tylko relatywnej z exp1 — można cytować w pracy *"variant X plays at ~1800 Elo"*
- **ACPL** — twarda, niezależna od wyniku miara jakości decyzji. Standardowa metryka w literaturze szachowej
- **Krzywa siły** — od jakiego skill levelu wariant przestaje wygrywać; pokazuje "stratygraficzny limit" siły
- **Walidacja exp1** — czy ranking exp1 (relatywny) zgadza się z rankingiem exp5 (absolutnym)?

W pracy zazwyczaj prezentuje się:
- **Tabela:** estimated Elo per wariant (z `exp5_variant_elo.csv`) + porównanie z exp1
- **Wykres:** krzywa score vs SF Elo, 4 linie — `exp5_score_curve.png`
- **Wykres słupkowy:** ACPL per wariant — `exp5_acpl_by_variant.png`
- **Tabela:** ACPL by phase — pokazuje gdzie wariant traci (otwarcie / mid / endgame)
- **Krótka dyskusja:** korelacja Elo (exp1 vs exp5), korelacja Elo vs ACPL

## Szacunek czasu obliczeń

### Faza gier (Krok 2): 4 warianty × 8 skill × 20 gier

Czas/gra zależy od silniejszego z dwóch silników. Stockfish d=10 jest szybki (<0.1s/ruch); bottleneckiem są warianty NN.

| Wariant | Czas/ruch (avg W+B) | 160 gier × ~80 ruchów |
|---|---|---|
| MINIMAX_TRAD d=4 | ~3s | ~10-12h |
| MINIMAX_NN d=3 | ~6s | ~20-25h ⚠ |
| MCTS_TRAD 2.61s | ~2.7s | ~9-11h |
| MCTS_NN 2.61s | ~2.7s | ~9-11h |

**Wszystkie 4 warianty równolegle:** ~25h wall-clock (bottleneck = MINIMAX_NN). Sekwencyjnie ~55h.

### Faza re-ewaluacji (Krok 3, Faza 2)

Stockfish d=20 per ruch: ~3-10s/ruch. Dla 640 gier × ~80 ruchów ≈ **51200 ruchów**.

- d=20: ~50-150h (poniżej praktyczne tylko dla finalnej analizy)
- d=15: ~5-15h (znacznie szybciej, niewielka strata precyzji)
- `-SkipReval`: pomiń (brak ACPL, ale wciąż Elo)

**Rekomendacja:** dla **smoke test analizy** → `-RevalDepth 15 -RevalLimit 50`. Dla **finalnej pracy** → pełne `--depth 20` w tle przez weekend.

### Całość

| Scenariusz | Czas |
|---|---|
| 4 warianty równolegle + reval d=15 | **~25h + ~10h = ~35h** wall-clock |
| 4 warianty równolegle + reval d=20 | ~25h + ~100h = ~125h (5 dni) |
| 4 warianty równolegle + `-SkipReval` | ~25h (najszybciej, bez ACPL) |

## Ważne uwagi praktyczne

1. **Stockfish path musi być poprawny** — domyślny w skrypcie to Windows; na macOS przekaż `-StockfishPath` lub patchuj defaulty (analogicznie jak w exp6/exp7)
2. **MCTS kalibracja jest wymagana** — exp5 woła `Write-Error` jeśli `_mcts_calibrated_time.txt` brak. Uruchom `exp1\run_exp1_calibrate.ps1` jeśli potrzeba
3. **20 gier per matchup (vs 30 w exp1-3)** — łącznie 160/wariant nadal wystarcza. Mniejsza wariancja per-matchup ale lepsza pokrycie skill range
4. **Re-ewaluacja Stockfishem d=20 to OSOBNY proces** — można puścić **po skończeniu Fazy 1** w tle. Warianty same nie czekają na reval
5. **Adjudykacja KLUCZOWA** — bez niej mecze MCTS vs niskim skill (1, 3) mogą produkować bardzo długie remisy
6. **Każdy plik metryczny waży ~50-200KB** — 640 gier × ~100KB = ~64MB danych. Po reval +~30-50MB
7. **Quick smoke test:** `-GamesPerPair 2` per wariant + `-SkipReval` w analizie (~30 min full pipeline)

## Co do dyskusji w pracy

- **Korelacja Elo (exp1) ↔ Elo (exp5)** — jeśli ranking się zgadza, oba pomiary się walidują; jeśli nie, interesujące rozbieżności (np. wariant silny relatywnie ale słaby absolutnie = "siła w niszce", anti-engine play)
- **ACPL vs Elo** — typowo silnik z niższym ACPL ma wyższe Elo, ale **nie** monotonicznie. ACPL 50 może odpowiadać 1900-2200 Elo zależnie od wariancji
- **Blunder rate jako proxy siły** — wariant z mniej niż 1% blunders gra w okolicach FM/IM; >5% to amator
- **ACPL by phase** — typowo silniki z słabym endgame eval mają wyraźnie wyższe ACPL w fazie końcowej. NN może być lepszy taktycznie ale słabszy strategicznie — widoczne w fazach midgame vs endgame
- **Limit skill** — od którego skill levelu wariant przestaje wygrywać? Skill 8 (~1700 Elo)? 13 (~2400)? Wskazuje "praktyczną klasę gry" wariantu
- **Kalibracja Stockfisha** — `_sf_skill_elo.csv` to przybliżona mapa; w pracy warto zaznaczyć źródło tej tabeli (CCRL, własne pomiary, dokumentacja Stockfisha)
