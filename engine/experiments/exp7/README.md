# Eksperyment 7 — Wpływ książki otwarć

## Cel

Zmierzenie czy **książka otwarć** poprawia siłę gry i jak modyfikuje fazę otwarcia. Każdy z 2 algorytmów (MINIMAX_TRAD d=4 i MCTS_TRAD 20s) gra **self-play** w dwóch wariantach: **bez książki** i **z książką**. Porównujemy:

- **Wynik** (W/D/L) — czy książka istotnie zmienia balans
- **Numer ruchu wyjścia z książki** (kiedy silnik wraca do własnego search)
- **Czas fazy otwarcia** (do ~ruchu 12) — książka powinna drastycznie skrócić ten czas

W odróżnieniu od Eksp. 1-3 (gdzie partie startują z 25 pozycji ECO) — tu **partie startują od standardowej pozycji** (move 1), bo inaczej książka byłaby ominięta.

## Uczestnicy (4 konfiguracje)

| # | Konfiguracja | Algorytm | Książka | Komentarz |
|---|---|---|---|---|
| 1 | `minimax_trad_d4_book_off` | MINIMAX_TRAD d=4 | OFF | Baseline Minimax |
| 2 | `minimax_trad_d4_book_on` | MINIMAX_TRAD d=4 | ON | Book ON dla Minimax |
| 3 | `mcts_trad_book_off` | MCTS_TRAD 20s | OFF | Baseline MCTS |
| 4 | `mcts_trad_book_on` | MCTS_TRAD 20s | ON | Book ON dla MCTS |

Każda konfiguracja to **self-play** (tego samego silnika sam przeciw sobie) z `swap colors`. Wynik konfiguracji to dystrybucja W/D/L — interesuje nas **różnica** między OFF a ON dla tego samego algorytmu.

**Razem: 4 konfiguracje × 40 partii = 160 partii.**

**Dlaczego self-play, a nie OFF vs ON head-to-head?** Bo bezpośrednie porównanie "OFF vs ON" tej samej konfiguracji byłoby trywialne — strona z książką dostaje statystycznie lepsze otwarcie (krócej myśli, mniej błędów). Self-play OFF i self-play ON pozwala mierzyć: czy z książką gra w ogóle staje się **bardziej decyzyjna** (mniej remisów), **szybsza** (mniej czasu na otwarcie), **rozkład wyników** itd. — niezależnie od asymetrii.

**Dlaczego tylko TRAD (bez NN)?** Książka otwarć jest **niezależna od ewaluatora** (to deterministyczna polityka), więc dodanie wariantów NN nie wnosi nowej informacji — wynik powtórzyłby się jakościowo. Skupiamy się na **osi A** (algorytm) — czy Minimax i MCTS reagują na książkę **różnie**.

**Dlaczego MCTS 20s (a nie kalibracja 2.61s z exp1)?** 20s daje istotnie więcej iteracji (~spójne z Eksp. 3 i 5), żeby różnica "z książką vs bez" była mierzalna — przy 2.61s MCTS i tak nie zdąży odejść daleko od standardowych otwarć, więc efekt książki byłby zatarty.

## Pozycja startowa

**Standardowa pozycja początkowa** (`startpos`, brak `OpeningsFile`). To kluczowe — gdyby partie zaczynały się z pozycji ECO (jak exp1-3), wszystkie one byłyby **poza** zakresem książki, która składa się z popularnych otwarć typu Italian, Ruy Lopez itd.

**Konsekwencja:** brak kontroli wariancji otwarciowej obecnej w exp1-3. Ale wariancja jest celowa — chcemy zobaczyć **jak silnik bez wskazówek wybiera otwarcia** vs **jak je dyktuje książka**.

## Adjudykacja

**WŁĄCZONA** z domyślnymi parametrami (`±0.05`, `20 ruchów`). Self-play MCTS bez adjudykacji może produkować bardzo długie remisy.

## Procedura uruchomienia

### Krok 1 — Uruchom 4 konfiguracje (równolegle)

```powershell
# Terminal 1:
.\experiments\exp7\run_exp7_config.ps1 -Config 1   # MINIMAX_TRAD d=4 -- book OFF
# Terminal 2:
.\experiments\exp7\run_exp7_config.ps1 -Config 2   # MINIMAX_TRAD d=4 -- book ON
# Terminal 3:
.\experiments\exp7\run_exp7_config.ps1 -Config 3   # MCTS_TRAD 20s    -- book OFF
# Terminal 4:
.\experiments\exp7\run_exp7_config.ps1 -Config 4   # MCTS_TRAD 20s    -- book ON
```

Każdy skrypt:
1. Generuje single-config JSON (`_exp7_config{N}.json`)
2. Woła `run_experiment.ps1` z wspólnym `-OutputSubDir = exp7_opening_book_<data>`
3. Wszystkie 4 konfiguracje piszą do **tego samego katalogu**
4. Konfiguracje z `book=on` przekazują flagę `-OpeningBook $true`

**Wszystkie 4 konfiguracje MUSI być uruchomione tego samego dnia** (tag oparty na dacie `yyyyMMdd`). Jeśli musisz przerwać i wznowić innego dnia, przekaż jawnie `-ExperimentTag`.

**Quick smoke test:** `.\experiments\exp7\run_exp7_config.ps1 -Config 2 -GamesPerPair 2` (~15 min).

**Override MCTS budget** (jeśli 20s za długie): `.\experiments\exp7\run_exp7_config.ps1 -Config 3 -McTsTime 5.0`.

### Krok 2 — Analiza zbiorcza (~2 min)

```powershell
.\experiments\exp7\run_exp7_analyze.ps1
# lub jawnie:
.\experiments\exp7\run_exp7_analyze.ps1 -ExperimentDir engine\out\exp7_opening_book_20260530
```

Wykonuje dwie fazy:
- **Faza 1:** `analyze_experiment.py --elo --plots` (analiza generyczna)
- **Faza 2:** `exp7_book_impact.py` (analiza specyficzna — book exit moves, opening phase time, chi-square + McNemar tests)

## Wyjście — pliki CSV

Katalog wyjściowy: `engine/out/exp7_opening_book_<data>/`

| Plik | Zawartość |
|---|---|
| `_results.csv` | Per-game: condition, game #, result, termination, czas |
| `analysis_moves.csv` | Per-move: eval, czas, faza gry, metryki algorytmu, flaga `from_book` |
| `analysis_games.csv` | Per-game: result, total moves, termination reason, czas |
| `analysis_wdl.csv` | Per-konfiguracja: 40 gier × W/D/L, white_score, avg_moves, avg_time |
| `analysis_elo.csv` | Bradley-Terry Elo (opcjonalnie, niskie znaczenie dla self-play) |
| `exp7_raw_per_game.csv` | **Kluczowy:** per-gra: book_exit_move (numer ruchu wyjścia z książki), opening_phase_time, book_hits |
| `exp7_summary.csv` | Per-(algorytm, book): n, W/D/L, decisive%, avg book_exit, avg opening_time |
| `exp7_statistical_tests.csv` | Chi-square (rozkład W/D/L) + McNemar (paired) — book OFF vs ON per algorytm |
| `exp7_summary.txt` | Human-readable raport |

## Wyjście — wykresy (`plots/`)

| Plik | Co pokazuje |
|---|---|
| `wdl_bars.png` | Generyczny W/D/L per konfiguracja |
| `time_per_move_boxplot.png` | Boxplot czasu/ruch per konfiguracja |
| `eval_over_game.png` | Krzywe ewaluacji (próbka) |
| `exp7_wdl_comparison.png` | **Kluczowy** — book OFF vs ON dla każdego algorytmu (4 słupki, side-by-side) |
| `exp7_book_exit_hist.png` | Histogram numeru ruchu wyjścia z książki (tylko book ON) — pokazuje głębokość książki |
| `exp7_opening_time.png` | Czas fazy otwarcia (do ~ruchu 12) — OFF vs ON; widoczna duża różnica |

## Co mierzymy per ruch / per gra

**Per ruch (analysis_moves.csv):**
- numer ruchu, strona, UCI, eval, czas, faza gry
- **`from_book`** (bool) — czy ten ruch pochodził z książki (kluczowe dla exp7)
- pełne metryki Minimax/MCTS (przydatne do "co się dzieje po wyjściu z książki")

**Per gra (exp7_raw_per_game.csv, derived):**
- `book_exit_move` — numer ruchu, po którym strona przestaje używać książki (None jeśli book OFF)
- `book_hits` — liczba ruchów z `from_book=true` w grze (0 jeśli book OFF)
- `opening_phase_time` — sumaryczny czas obu graczy w ruchach 1-12
- `result`, `termination`

## Analizy statystyczne

1. **Chi-square test (rozkład W/D/L)** — czy proporcje W/D/L różnią się istotnie między OFF a ON dla tego samego algorytmu? H0: rozkłady identyczne
2. **McNemar test (paired)** — dla każdej pary gier OFF/ON o tym samym opening seed (jeśli używany), czy zmiana wyniku jest istotna?
3. **Średni book_exit_move** — kiedy silnik wraca do własnego search? Większe = głębsza książka aktywna
4. **Średni opening_phase_time** — typowo OFF: kilka-kilkanaście sekund, ON: <1s (różnica dramatyczna)
5. **Decisive rate** — % gier zakończonych nie-remisem; czy książka zmienia "decisive%"?

## Co z planu badawczego jest istotne dla pracy magisterskiej

Eksperyment 7 odpowiada na pytanie: **czy książka otwarć ma istotny wpływ na siłę gry, i czy ten wpływ jest taki sam dla Minimaxa i MCTS?**

Hipotezy:
- **H1:** Książka ON skraca fazę otwarcia (czas) — niemal pewne, dolny szacunek 5-10× mniej czasu na pierwszych 12 ruchach
- **H2:** Książka ON zmniejsza wariancję wyniku (więcej remisów) — bo silnik startuje z "bezpiecznej" pozycji
- **H3:** MCTS bardziej zyskuje z książki niż Minimax — bo MCTS w otwarciu marnuje budżet na otwarcia, których ewaluator i tak słabo ocenia

W pracy zazwyczaj prezentuje się:
- **Wykres:** `exp7_wdl_comparison.png` — book OFF vs ON, 2 algorytmy
- **Wykres:** `exp7_opening_time.png` — dramatyczna różnica czasu fazy otwarcia
- **Tabela:** `exp7_summary.csv` — agregat W/D/L + book stats
- **Tabela:** `exp7_statistical_tests.csv` — chi-square + McNemar p-values
- **Krótka dyskusja:** czy efekt książki różni się dla Minimax vs MCTS (test asymetryczności)

## Szacunek czasu obliczeń

Self-play konfiguracje (40 gier × ~80 ruchów × 2 strony = ~6400 ruchów per config):

| # | Config | Czas/ruch | Szac. czas konfiguracji |
|---|---|---|---|
| 1 | MINIMAX_TRAD d=4 OFF | ~3-5s | ~5-8h |
| 2 | MINIMAX_TRAD d=4 ON | ~3-5s (krótsze otwarcie) | ~4-7h |
| 3 | MCTS_TRAD 20s OFF | 20s (fixed) | ~35h ⚠ |
| 4 | MCTS_TRAD 20s ON | 20s (fixed) | ~32h (krótsze otwarcie) |

**Uwaga: MCTS przy 20s jest BARDZO drogie.** Jeśli ograniczony czasem:
- Obniż `-McTsTime 5.0` → ~9h per MCTS config (czyli ~13h dla 3+4)
- Lub `-GamesPerPair 20` → ~17h per MCTS config

**Sumarycznie (z 20s MCTS i 40 gier):**
- 4 konfiguracje równolegle: **~35h wall-clock** (bottleneck = MCTS OFF)
- Sekwencyjnie: ~80h

**Rekomendacja przy ograniczonym czasie:**
- Konfigi 1+2 (Minimax): od razu, równolegle (~6h)
- Konfigi 3+4 (MCTS): obniż `-McTsTime 5.0` → ~10h
- Razem: **~12h wall-clock**

## Ważne uwagi praktyczne

1. **Pozycja startowa to STANDARD** — bez `-OpeningsFile`. To celowe; partie z ECO opening file ominęłyby książkę
2. **Wszystkie 4 konfiguracje MUSI być tego samego dnia** (tag domyślnie oparty na dacie)
3. **Brak NN-wariantów** — książka jest deterministyczna, nie zależy od ewaluatora; wyniki TRAD są reprezentatywne
4. **Self-play, nie head-to-head** — pomiar charakteru gry (decisive%, czas, exit move), nie bezpośrednie porównanie OFF vs ON
5. **MCTS 20s jest bottleneckiem** — rozważ obniżenie budżetu jeśli sprzęt ograniczony
6. **`from_book` flag** musi być prawidłowo zapisywana w JSONL przez warianty z `useBook=true`; jeśli sanity check pokazuje 0 book hits w configu ON, to bug w `openings/opening_book.py`
7. **Adjudykacja KLUCZOWA** — bez niej self-play MCTS może generować bardzo długie partie
8. **40 gier per konfiguracja vs 30 w exp1-3** — większe N kompensuje wyższą wariancję wynikającą z braku ECO startów

## Co do dyskusji w pracy

- **Czas fazy otwarcia** — najsilniejszy efekt; spodziewane 10-50× szybciej z książką. Warto pokazać jako "praktyczna zaleta" niezależnie od siły
- **Wpływ na siłę gry** — czy WDL istotnie różni się? Jeśli nie (p > 0.05), oznacza to że książka jest "neutralna w sile" ale przyspiesza grę
- **MCTS vs Minimax różnice** — MCTS w otwarciu marnuje budżet na pozycje początkowe, gdzie ewaluator słabo dyskryminuje; książka eliminuje ten "marnotrawca" — efekt może być większy niż dla Minimaxa
- **Book exit move** — typowo 4-10 ruchów; pokazuje **jak daleko sięga repertuar książki** dla danego silnika. Jeśli książka jest mała, exit move = niski, efekt minimalny
- **Decisive rate** — czy z książką proporcja remisów rośnie (silniki częściej grają symetryczne otwarcia) czy maleje (książka prowadzi do ostrzejszych otwarć)?
- **Limitacje:** brak NN-wariantów (zaleta: jasne porównanie; wada: nie sprawdzamy interakcji book × NN), self-play (brak head-to-head silników różniących się tylko obecnością książki), zależność od jakości książki (jeśli książka jest minimalna lub off-line, efekt jest sztucznie obniżony)
