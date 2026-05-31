# Eksperyment 8 — Round-robin najmocniejszych wariantów z książką otwarć

## Cel

Odpowiedź na pytanie: **"w warunkach najbardziej zbliżonych do praktyki (silne parametry + książka), który z 4 wariantów silnika wygrywa?"**.

Wyniki służą jako **punkt odniesienia rankingowy** uzupełniający:
- **exp1** — round-robin 4 wariantów, ale parametry zaniżone (Minimax d=3, MCTS 2.61s), **bez** książki
- **exp7** — wpływ książki, ale **self-play** (każdy wariant sam przeciw sobie), tylko 2 algorytmy (TRAD)
- **exp5** — silne parametry vs Stockfish (skalibrowane Elo), ale 1×1, nie round-robin

Brakuje eksperymentu, który łączy **najsilniejsze praktyczne parametry** z **obecnością książki otwarć** w układzie head-to-head 4 wariantów — exp8 wypełnia tę lukę.

## Uczestnicy (4 warianty)

| Wariant | Algorytm | Ewaluator | Parametr zasobu |
|---|---|---|---|
| `MINIMAX_TRAD` | Minimax α-β | Heurystyczny | **`depth = 5`** |
| `MINIMAX_NN` | Minimax α-β | "NN" (Stockfish low-depth) | **`depth = 4`** |
| `MCTS_TRAD` | MCTS (PUCT) | Heurystyczny | **`time = 60s`** |
| `MCTS_NN` | MCTS (PUCT) | "NN" (Stockfish low-depth) | **`time = 60s`** |

**Uzasadnienie parametrów (na podstawie pomiarów z exp1b/exp2/exp3):**

- **MINIMAX_TRAD d=5** — maksymalna wykonalna głębokość (z exp1b: mean 2.2s, p90 5.4s, max 62s/ruch). d=6 prognozowane ~80s mean — niewykonalne.
- **MINIMAX_NN d=4** — świadomy kompromis. Z exp1b: p90 136s, max 866s. Niektóre partie będą długie, ale d=3 byłoby "zbyt słabe" dla porównania z d=5 TRAD.
- **MCTS 60s** — kompromis czas/jakość. Z krzywej skalowania exp3: ~50-80 Elo niżej niż 120s, ale 2× szybciej.

W odróżnieniu od exp1 (gdzie celem było *czyste Axis B*, więc identyczne d=3 dla obu MINIMAX), exp8 dopuszcza **różne parametry** per wariant — używamy najmocniejszej praktycznej konfiguracji każdego z osobna.

## Struktura meczowa (6 par)

| # | Para | Komentarz |
|---|---|---|
| 1 | MINIMAX_TRAD d=5 vs MINIMAX_NN d=4 | Axis B przy max parametrach |
| 2 | MINIMAX_TRAD d=5 vs MCTS_TRAD 60s | Axis A przy TRAD |
| 3 | MINIMAX_TRAD d=5 vs MCTS_NN 60s | Cross-axis |
| 4 | MINIMAX_NN d=4 vs MCTS_TRAD 60s | Cross-axis |
| 5 | MINIMAX_NN d=4 vs MCTS_NN 60s | Axis A przy NN |
| 6 | MCTS_TRAD 60s vs MCTS_NN 60s | Axis B przy MCTS |

**N=10 partii per para** (5 oryginał + 5 swap colors). **Razem: 60 partii.**

**Uwaga o próbie:** N=10 to **mała próba** — wystarcza do rankingu, ale precyzja Elo będzie umiarkowana (CI ±150–200 Elo dla różnic między wariantami). Różnice <150 Elo należy traktować jako *exploratory*, nie *definitive*.

## Pozycja startowa

**Standardowa pozycja początkowa** (`startpos`, brak `OpeningsFile`). To kluczowe — gdyby partie zaczynały się od pozycji ECO (jak exp1-3), wszystkie ominęłyby książkę (która zawiera tylko popularne otwarcia od ruchu 1).

**Wariancja otwarciowa:** wprowadzana przez **stochastyczną książkę** (`codekiddy.bin`, weighted random) — 10 gier per para da ~5-8 różnych linii otwarciowych mimo identycznej pozycji startowej. To zastępuje rolę 25 pozycji ECO z exp1.

## Książka otwarć

**WŁĄCZONA** dla obu graczy w każdej parze. Tryb domyślny — **stochastyczny** (weighted random, nie `-obs`/best-only). Daje to naturalną wariancję bez konieczności użycia ECO seedów.

## Adjudykacja

**WŁĄCZONA**, standardowe parametry (`±0.05` eval, `20 ruchów`). Krytyczna dla pary 6 (MCTS vs MCTS) — bez niej self-play MCTS może produkować bardzo długie partie.

## Procedura uruchomienia

### Krok 1 — 6 par równolegle

```powershell
# Terminal 1:
.\experiments\exp8\run_exp8_pair.ps1 -Pair 1 -ExperimentTag exp8_1
# Terminal 2:
.\experiments\exp8\run_exp8_pair.ps1 -Pair 2 -ExperimentTag exp8_2
# Terminal 3:
.\experiments\exp8\run_exp8_pair.ps1 -Pair 3 -ExperimentTag exp8_3
# Terminal 4:
.\experiments\exp8\run_exp8_pair.ps1 -Pair 4 -ExperimentTag exp8_4
# Terminal 5:
.\experiments\exp8\run_exp8_pair.ps1 -Pair 5 -ExperimentTag exp8_5
# Terminal 6:
.\experiments\exp8\run_exp8_pair.ps1 -Pair 6 -ExperimentTag exp8_6
```

Każda para pisze do **osobnego** katalogu `out/exp8_strongest_book_exp8_<N>/`. Po zakończeniu wszystkich 6 — scal je w jeden katalog dla analizy zbiorczej (analogicznie do exp2):

```powershell
# Scalanie (przykład — pwsh):
$combined = "out\exp8_strongest_book_combined"
New-Item -ItemType Directory -Path $combined -Force | Out-Null
1..6 | ForEach-Object {
    Copy-Item "out\exp8_strongest_book_exp8_$_\*" $combined -Recurse -Force
}
```

Alternatywnie — jeśli wszystkie 6 par uruchamiasz **tego samego dnia** bez `-ExperimentTag`, wszystkie zapiszą do współdzielonego `out/exp8_strongest_book_<yyyyMMdd>/` (tag oparty na dacie).

### Quick smoke test

```powershell
.\experiments\exp8\run_exp8_pair.ps1 -Pair 1 -GamesPerPair 2 -ExperimentTag smoke
```

Sprawdź w `out/exp8_strongest_book_smoke/`:
- `_results.csv` ma 2 wiersze + header
- 2 pliki `metrics_*.jsonl`
- W metrykach widać `from_book: true` dla pierwszych kilku ruchów (książka aktywna)

### Krok 2 — Analiza zbiorcza

```powershell
.\experiments\exp8\run_exp8_analyze.ps1
# lub jawnie:
.\experiments\exp8\run_exp8_analyze.ps1 -ExperimentDir engine\out\exp8_strongest_book_combined
```

Wykonuje dwie fazy:
- **Faza 1:** `analyze_experiment.py --elo --plots` (analiza generyczna)
- **Faza 2:** `exp1_round_robin.py` (reused — struktura round-robin 4 wariantów jest identyczna)

## Wyjście — pliki CSV

Katalog wyjściowy: `engine/out/exp8_strongest_book_<tag>/`

| Plik | Zawartość |
|---|---|
| `_results.csv` | Per-game: para, game #, wynik, terminacja, czas |
| `analysis_moves.csv` | Per-move: eval, czas, metryki algorytmu, flaga `from_book` |
| `analysis_games.csv` | Per-game: wynik, total moves, terminacja, czas |
| `analysis_wdl.csv` | Per-matchup: 10 gier × W/D/L, white_score, avg_moves, avg_time |
| `analysis_elo.csv` | Bradley-Terry Elo dla 4 wariantów |
| `exp1_pair_significance.csv` | Per-para: binomial test, p-value, 95% CI |
| `exp1_axis_summary.csv` | Główne efekty: MINIMAX vs MCTS (oś A), TRAD vs NN (oś B) |
| `exp1_color_advantage.csv` | White vs Black overall |
| `exp1_round_robin_summary.txt` | Human-readable raport |

## Wyjście — wykresy (`plots/`)

| Plik | Co pokazuje |
|---|---|
| `wdl_bars.png` | Słupki W/D/L per matchup |
| `time_per_move_boxplot.png` | Boxplot czasu/ruch per wariant+kolor |
| `exp1_pair_significance.png` | Per para z 95% CI |
| `exp1_axis_a_effect.png` | MINIMAX vs MCTS agregat |
| `exp1_axis_b_effect.png` | TRAD vs NN agregat |
| `exp1_wdl_matrix.png` | 4×4 heatmap score per (variant, opponent) |

## Szacunek czasu obliczeń

Przy adjudykacji obniżającej długość partii ~50% (~50 pełnych ruchów = 100 półruchów):

| # | Para | Per game | Per pair (10g) |
|---|---|---|---|
| 1 | TRAD d=5 vs NN d=4 | ~30 min (NN d=4 ma p90 136s/ruch) | ~5h |
| 2 | TRAD d=5 vs MCTS_TRAD 60s | ~50 min | ~8-9h |
| 3 | TRAD d=5 vs MCTS_NN 60s | ~50 min | ~8-9h |
| 4 | NN d=4 vs MCTS_TRAD 60s | ~80 min | ~13h |
| 5 | NN d=4 vs MCTS_NN 60s | ~80 min | ~13h |
| 6 | MCTS_TRAD vs MCTS_NN (oba 60s) | ~100 min | **~16-17h** (bottleneck) |

**6 par równolegle: ~17-20h wall-clock** (bottleneck = pair 6, lub 4/5 jeśli NN d=4 ma outliery 800s+).

**Uwaga o kontencji CPU:** przy 14 rdzeniach + równolegle działających exp3/5/7 + 6 procesach exp8 + ich Stockfish-oracle helperach (5 par używa NN → 5 dodatkowych Stockfish-oracle) ≈ 25 procesów. Dodaj ~50% do wall-clock.

## Co dyskutować w pracy

- **Ranking exp8 vs exp1** — czy książka zmienia kolejność wariantów (Axis A/B)? Jeśli tak — który wariant "korzysta" z książki najbardziej?
- **N=10 to mała próba** — wyniki **rankingowe**, nie precyzyjne; różnice Elo <150 są wątpliwie istotne. W pracy zaznaczyć jako *exploratory comparison*.
- **MINIMAX_NN d=4** — jeśli niektóre partie się "wyciągały" (extreme moves >5min/ruch), zaznaczyć w limitacjach.
- **Stochastyczna książka** — sprawdzić w danych ile unikalnych otwarć faktycznie wystąpiło per parze (sanity check wariancji otwarciowej).
- **Porównanie z exp7** — czy efekt książki widoczny w self-play (exp7) powtarza się w head-to-head exp8 dla TRAD vs TRAD?

## Ważne uwagi praktyczne

1. **Pozycja startowa to STANDARD** — bez `-OpeningsFile`. Książka jest aktywna tylko od popularnych otwarć od move 1
2. **Wszystkie 6 par MUSI być uruchomione tego samego dnia** (jeśli używasz dziennego tagu) — inaczej `out/exp8_strongest_book_<data>/` rozjedzie się
3. **Adjudykacja KLUCZOWA** — szczególnie dla pary 6 (MCTS vs MCTS) i par 4/5 (długie partie z NN)
4. **`from_book` flag** musi być zapisywana w JSONL — jeśli sanity check pokazuje 0 book hits, sprawdzić `openings/opening_book.py`
5. **N=10 vs N=30 w exp1** — celowo mniejsze; exp8 to *exploratory ranking*, nie *primary measurement*. Pełna analiza siły jest w exp1/exp5
6. **Quick smoke test:** `.\experiments\exp8\run_exp8_pair.ps1 -Pair 1 -GamesPerPair 2 -ExperimentTag smoke` (~10-15 min, sprawdza pipeline + obecność `from_book`)
