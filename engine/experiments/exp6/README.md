# Eksperyment 6 — Dokładność taktyczna na łamigłówkach

## Cel

Zmierzenie **dokładności taktycznej** każdego z 4 wariantów silnika — czyli jak często wariant znajduje *najlepszy ruch* w pozycji, w której taki najlepszy ruch jest jednoznacznie określony (łamigłówka taktyczna z `bm` jako ground truth). W odróżnieniu od Eksp. 1-3 (które mierzą *siłę gry* w pełnych partiach), Eksp. 6 mierzy **jakość pojedynczej decyzji w "ostrej" pozycji**.

Dodatkowo dla wariantów Minimax: rejestracja **minimalnej głębokości ID**, przy której rozwiązanie pojawia się jako PV (z parsowania logu `ID iteration depth=N; move: <uci>`).

## Uczestnicy (4 warianty)

| Wariant | Algorytm | Ewaluator | Parametr zasobu |
|---|---|---|---|
| `MINIMAX_TRAD_d4` | Minimax α-β | Heurystyczny | `depth = 4` |
| `MINIMAX_NN_d3` | Minimax α-β | "NN" (Stockfish low-depth) | `depth = 3` |
| `MCTS_TRAD` | MCTS (PUCT) | Heurystyczny | `time = 2.61s` (skalibrowane, z exp1) |
| `MCTS_NN` | MCTS (PUCT) | "NN" (Stockfish low-depth) | `time = 2.61s` (skalibrowane, z exp1) |

**Uwaga: parametry zasobu są spójne z Eksp. 5 i 7** — nie z Eksp. 1 (gdzie wszystkie Minimax miały d=3 dla czystego Axis B). Tu używamy "praktycznych" konfiguracji: MINIMAX_TRAD d=4 (silniejszy, ale jeszcze wykonalny) i MINIMAX_NN d=3 (kompromis czas/jakość).

## Źródło pozycji

Łamigłówki **nie są generowane** — pochodzą z publicznych zbiorów testowych w formacie EPD (Extended Position Description).

**Wspierane zbiory** (parser w `prepare_puzzles.py`):

| Zbiór | Liczba pozycji | Charakterystyka |
|---|---|---|
| **WAC** (Win At Chess) | 300 | Klasyczne łamigłówki taktyczne, jednoznaczne `bm` |
| **STS** (Strategic Test Suite) | 1500 | 15 tematów strategicznych (etykietowane w polu `c0`) |
| **Bratko-Kopec** | 24 | Standardowy benchmark IM, mix taktyczny+strategiczny |

**Format EPD:** `<FEN> bm <best_move_san>; id "<id>";` — np.:
```
r1bk3r/pp1p3p/8/3pNN2/3P4/2P5/PP4PP/3R1RK1 w - - bm Re8+; id "WAC.001";
```

**Gdzie pobrać pełne zbiory** (publiczne):
- WAC: https://www.chessprogramming.org/Test-Positions (`WAC.epd`)
- STS: https://www.chessprogramming.org/Strategic_Test_Suite

**Próbka w repo:** `puzzles/sample_wac.epd` — 9 wybranych pozycji WAC (smoke test).

## Metodologia (per łamigłówka)

Dla każdej pozycji ze zbioru:

1. **Silnik gra pozycję** — `main.py` z `-i <fen>`. Wariant testowany jest stroną-na-ruch (`side_to_move`).
2. **Przeciwnik:** Stockfish skill 0 (najsłabszy, najszybszy) gra ruch nr 2 — **tylko po to, żeby silnik "wyemitował" pierwszy ruch**. Wynik drugiego ruchu nie jest oceniany.
3. **Szybka adjudykacja:** `-adjt 0.1 -adjm 5` (próg eval ±0.1 przez 5 ruchów) → partia kończy się po kilku ruchach.
4. **Ekstrakcja:** z pliku gry `_temp_..._game.txt` parsujemy pierwszy ruch testowanego wariantu (linia `1: <uci>` jeśli białe, `2: <uci>` jeśli czarne).
5. **Porównanie:** ruch silnika ∈ `best_moves_uci`? → `solved = True/False`.
6. **Bonus dla Minimax:** parsing logu `_temp_..._log.txt` — szukamy najmniejszej głębokości ID, przy której PV move ∈ `best_moves_uci`. Zapisywane jako `min_depth_to_solve`.

**Po co Stockfish skill 0 jako przeciwnik (zamiast np. od razu adjudykacji)?** `main.py` wymaga obu graczy, żeby uruchomić pętlę gry. Skill 0 to minimalny narzut czasowy (~10-50ms/ruch) i nigdy nie blokuje terminacji.

**Po co tak agresywna adjudykacja (`adjt 0.1`, `adjm 5`)?** Interesuje nas **tylko pierwszy ruch silnika** — reszta partii to narzut. Próg ±0.1 przez 5 ruchów kończy partię niemal natychmiast po wykonaniu ruchu nr 1-2.

**Timeout:** 120s per łamigłówka (zabezpieczenie — gdyby silnik się zawiesił).

## Procedura uruchomienia

### Krok 0 — Przygotowanie zbioru łamigłówek (raz)

```powershell
# Smoke test (9 pozycji w repo):
python experiments\exp6\prepare_puzzles.py --input experiments\exp6\puzzles\sample_wac.epd --set WAC

# Pełny WAC (pobrać wac.epd ze strony chessprogramming.org):
python experiments\exp6\prepare_puzzles.py --input experiments\exp6\puzzles\wac.epd --set WAC

# STS:
python experiments\exp6\prepare_puzzles.py --input experiments\exp6\puzzles\STS1-STS15.epd --set STS
```

Produkuje `experiments\exp6\puzzles.json` z polami: `id`, `fen`, `best_moves_san`, `best_moves_uci`, `side_to_move`, `theme`, `difficulty`, `source_id`.

Konwersja SAN→UCI używa `python-chess`. Pozycje, dla których konwersja się nie udała, są pomijane (skipped, raportowane).

### Krok 1 — 4 warianty równolegle

```powershell
# Terminal 1:
.\experiments\exp6\run_exp6_variant.ps1 -Variant 1   # MINIMAX_TRAD d=4
# Terminal 2:
.\experiments\exp6\run_exp6_variant.ps1 -Variant 2   # MINIMAX_NN d=3
# Terminal 3:
.\experiments\exp6\run_exp6_variant.ps1 -Variant 3   # MCTS_TRAD
# Terminal 4:
.\experiments\exp6\run_exp6_variant.ps1 -Variant 4   # MCTS_NN
```

Każdy skrypt:
1. Czyta skalibrowany czas MCTS z `experiments\exp1\_mcts_calibrated_time.txt` (fallback 2.61s jeśli brak — wynik kalibracji udokumentowany w exp1).
2. Resolves absolute path do Stockfisha.
3. Deleguje do helpera `_run_variant_puzzles.py` (per-puzzle subprocess loop).
4. Zapisuje wyniki do `exp6_variant<N>_<NAZWA>_<yyyyMMdd>.csv`.

**Quick test:** `.\experiments\exp6\run_exp6_variant.ps1 -Variant 1 -Limit 10` (10 pozycji, ~kilka minut).

**Override MCTS time:** `.\experiments\exp6\run_exp6_variant.ps1 -Variant 3 -MctsTime 5.0`.

### Krok 2 — Analiza zbiorcza

```powershell
.\experiments\exp6\run_exp6_analyze.ps1
# albo z filtrem tagu (jeśli mieszane dni):
.\experiments\exp6\run_exp6_analyze.ps1 -Tag 20260527
```

Wykonuje `exp6_analyze.py` — czyta wszystkie pliki `exp6_variant*.csv` z katalogu eksperymentu, agreguje i generuje wykresy.

## Wyjście — pliki CSV

Katalog wyjściowy: **`engine/experiments/exp6/`** (te same miejsce co skrypty — nie `out/`).

| Plik | Zawartość |
|---|---|
| `exp6_variant<N>_<NAZWA>_<tag>.csv` | Per-puzzle (4 pliki, jeden na wariant): `puzzle_id`, `theme`, `fen`, `expected_uci`, `expected_san`, `engine_move_uci`, `solved`, `duration_s`, `timed_out`, `min_depth_to_solve` |
| `exp6_solve_rate.csv` | Per-wariant: `n`, `solved`, `avg_time_s`, `timeouts`, `solve_rate` (sortowane malejąco) |
| `exp6_solve_by_theme.csv` | Per-(wariant, temat): `n`, `solved`, `solve_rate` |
| `exp6_minimax_depth_to_solve.csv` | Per-wariant Minimax: `mean`, `median`, `min`, `max`, `count` głębokości ID przy której PV trafia rozwiązanie |
| `exp6_summary.txt` | Human-readable raport |

## Wyjście — wykresy (`plots/`)

| Plik | Co pokazuje |
|---|---|
| `exp6_solve_rate_bars.png` | Słupkowy: solve rate (%) per wariant — **główny wynik** |
| `exp6_solve_by_theme_heatmap.png` | Heatmap (temat × wariant) — które tematy taktyczne sprawiają komu kłopot |
| `exp6_solve_by_theme_radar.png` | Wykres radarowy per wariant (tylko jeśli 3-20 tematów) |
| `exp6_minimax_depth_hist.png` | Histogram głębokości ID, przy której Minimax "odkrywa" rozwiązanie |

## Co mierzymy per łamigłówka

- **`solved`** — czy `engine_move_uci ∈ best_moves_uci`. Główna metryka.
- **`duration_s`** — całkowity czas wywołania `main.py` (z narzutem startu Python).
- **`timed_out`** — czy hit 120s timeout (anomalia, raczej nie powinno się zdarzać).
- **`min_depth_to_solve`** (tylko Minimax) — najmniejsze ID iteration depth, przy którym PV move ∈ rozwiązań. Pozwala porównać **głębokość rozwiązania** TRAD vs NN: jeśli NN znajduje to samo rozwiązanie przy płytszej ID, to NN "widzi" taktykę szybciej.

## Analizy statystyczne

1. **Solve rate per wariant** — główna metryka, n=|puzzles|, sortowane.
2. **Solve rate per temat × wariant** — wskazuje *strukturalne* słabości (np. czy MCTS gorzej radzi sobie z taktyką końcówkową niż z taktyką mat-w-3?).
3. **Min depth to solve (Minimax)** — czy NN potrzebuje mniejszej głębokości, żeby zobaczyć to samo, co TRAD?
4. **Timeout rate** — sanity check (powinien być ~0%).

**Uwaga: brak Bradley-Terry Elo.** Łamigłówki nie są bezpośrednim porównaniem 1-vs-1 → nie ma "wyników gier" do dopasowania. Porównanie wariantów odbywa się przez różnicę w solve rate (z opcjonalnym binomial proportions test, jeśli chcemy istotności).

## Co z planu badawczego jest istotne dla pracy magisterskiej

Eksperyment 6 odpowiada na pytanie: **czy któryś wariant ma strukturalną przewagę w "ostrych" pozycjach taktycznych** — co może być niewidoczne w samych Eksp. 1-3 (gdzie partie zaczynają się od ECO i przebiegają przez tysiące "spokojnych" pozycji)?

W pracy zazwyczaj prezentuje się:
- **Tabela:** solve rate per wariant (z `exp6_solve_rate.csv`)
- **Heatmap:** solve rate per temat × wariant — pokazuje *charakter* mocnych/słabych stron każdego wariantu
- **Histogram:** głębokość ID przy odkryciu rozwiązania (Minimax TRAD vs NN) — pokazuje czy NN szybciej "widzi" taktykę
- **Krótka dyskusja:** czy ranking z Eksp. 6 koreluje z rankingiem Elo z Eksp. 1 (jeśli nie — interesujące rozbieżności)

## Szacunek czasu obliczeń

**Per łamigłówka** (1 ruch silnika + 1 ruch Stockfish skill 0 + ~3-5 ruchów do adjudykacji):

| Wariant | Czas/łamigłówkę | 300 WAC | 1500 STS |
|---|---|---|---|
| MINIMAX_TRAD d=4 | ~5-10s | ~30-50 min | ~3-4h |
| MINIMAX_NN d=3 | ~10-20s | ~50-100 min | ~4-8h |
| MCTS_TRAD (2.61s) | ~3-5s | ~15-25 min | ~1.5-2h |
| MCTS_NN (2.61s) | ~3-5s | ~15-25 min | ~1.5-2h |

**4 warianty równolegle:** bottleneck = MINIMAX_NN.
- Pełny WAC (300): **~1.5h wall-clock**.
- Pełny STS (1500): **~8h wall-clock**.
- WAC + STS razem (1800): **~10h wall-clock**.

**Rekomendacja:** zacząć od pełnego WAC (mniejszy, klasyczny benchmark taktyczny). STS jako rozszerzenie, jeśli zostanie czas.

## Ważne uwagi praktyczne

1. **`puzzles.json` MUSI być wygenerowany przed uruchomieniem wariantów** — patrz Krok 0.
2. **Wyniki lądują w katalogu skryptu, nie w `engine/out/`** — odróżnia exp6 od pozostałych eksperymentów (które piszą do `out/<shared_dir>/`).
3. **Tag jest oparty na dacie** (`yyyyMMdd`). Jeśli musisz dokończyć inny dzień, przekaż `-ExperimentTag` jawnie. Analiza domyślnie bierze wszystkie pliki `exp6_variant*.csv`, możesz filtrować przez `-Tag`.
4. **Brak openings_eco25.fen** — każda łamigłówka ma własny FEN. Książka otwarć jest nieistotna (pozycje są mid/end-game).
5. **Adjudykacja jest agresywna (`adjt 0.1`, `adjm 5`)** — celowo, żeby skończyć partię tuż po pierwszym ruchu.
6. **Stockfish skill 0 jest "filler" przeciwnikiem** — jego ruchy nie wpływają na wynik (`solved` zależy tylko od ruchu nr 1 testowanego silnika).
7. **Timeout 120s** zabezpiecza przed zawieszeniem — w praktyce nie powinno się aktywować.
8. **`min_depth_to_solve` jest dostępny tylko dla Minimax** — wymaga parsing logu, który MCTS nie produkuje w tym formacie.

## Co do dyskusji w pracy

- **Korelacja z Eksp. 1 (Elo)**: czy ranking solve rate ≈ ranking Elo? Rozbieżności są ciekawe — np. wariant silny w grze (Elo) ale słaby w łamigłówkach taktycznych może wskazywać na siłę *pozycyjną* (a nie taktyczną).
- **TRAD vs NN — czy NN "widzi" taktykę głębiej/płycej?** Z `min_depth_to_solve`: jeśli MINIMAX_NN znajduje rozwiązanie przy d=2-3 tam, gdzie MINIMAX_TRAD potrzebuje d=4, to NN dostarcza "tactical hints" w pozycjach niezbalansowanych materiałowo.
- **MCTS vs Minimax na taktyce**: MCTS często gorzej radzi sobie z "ostrymi" pozycjami (wymagającymi konkretnej sekwencji), bo eksploracja stochastyczna nie penalizuje wystarczająco za 1 nieudaną wariację. W pracy warto porównać solve rate MCTS_TRAD vs MINIMAX_TRAD przy podobnym budżecie.
- **Per-theme insights (STS)**: jeśli używasz STS, heatmap pokaże które tematy strategiczne są domeną którego wariantu (np. "King Safety" vs "Open Files" vs "Pawn Structure").
- **Limitacje benchmarku taktycznego**: zbiory typu WAC są stare (lata '90), niektóre pozycje mają więcej niż jedno akceptowalne rozwiązanie — pole `best_moves_uci` to lista, więc jeśli silnik trafi *któreś* z dopuszczonych, liczy się jako solved. Pozycje z STS są etykietowane tematycznie ale nie mają jednoznacznych `bm` taktycznych — ich solve rate należy interpretować bardziej jako "zgodność z preferencją silnym przy danym temacie strategicznym".
