# Eksperyment 2 — Skalowanie siły Minimaxa z głębokością

## Cel

Zmierzenie jak siła gry Minimaxa rośnie wraz z głębokością przeszukiwania (oś A — wrażliwość zasobowa). Pokazanie krzywej malejących zwrotów: o ile Elo przybywa na każdy dodatkowy poziom głębokości, dla obu ewaluatorów (TRAD i NN).

## Uczestnicy (10 matchupów)

W odróżnieniu od Eksp. 1 (round-robin między 4 wariantami), Eksp. 2 to **seria pojedynków pomiędzy MINIMAX a tym samym MINIMAX o stałej głębokości referencyjnej d=4**, dla każdej z 5 testowanych głębokości i każdego z 2 ewaluatorów:

| # | Matchup | Komentarz |
|---|---|---|
| 1 | MINIMAX_TRAD d=2 vs MINIMAX_TRAD d=4 | Płytka głębokość → spodziewana porażka |
| 2 | MINIMAX_TRAD d=3 vs MINIMAX_TRAD d=4 | Słabsza, mierzymy Δ Elo |
| 3 | MINIMAX_TRAD d=4 vs MINIMAX_TRAD d=4 | **Sanity check** — Elo ≈ 0 ± kilkadziesiąt |
| 4 | MINIMAX_TRAD d=5 vs MINIMAX_TRAD d=4 | Głębsza, mierzymy zysk |
| 5 | MINIMAX_TRAD d=6 vs MINIMAX_TRAD d=4 | Najgłębsza, krzywa się spłaszcza? |
| 6 | MINIMAX_NN d=2 vs MINIMAX_NN d=4 | Powtórzenie 1-5 dla ewaluatora NN |
| 7 | MINIMAX_NN d=3 vs MINIMAX_NN d=4 | |
| 8 | MINIMAX_NN d=4 vs MINIMAX_NN d=4 | Sanity check NN |
| 9 | MINIMAX_NN d=5 vs MINIMAX_NN d=4 | |
| 10 | MINIMAX_NN d=6 vs MINIMAX_NN d=4 | |

**Razem: 10 matchupów × 30 partii = 300 partii.**

**Dlaczego d=4 jako poziom odniesienia (anchor)?** d=4 jest "środkową" głębokością, dla której ewaluator daje wynik sensownie skorelowany z faktyczną wartością pozycji (mierzonej Stockfishem d=20). Krzywa Elo jest kotwiczona w punkcie d=4 → Elo = 0; pozostałe punkty wyliczane są względem niego przez Bradley-Terry maximum likelihood.

**Dlaczego dwa ewaluatory osobno?** Aby krzywa Elo vs głębokość była rysowana **osobno dla TRAD i NN** — można porównać tempo wzrostu siły: czy NN dzięki lepszej ewaluacji wymaga mniej głębokości dla tej samej siły (ozn. "głębokość kompensuje słabość ewaluatora").

## Otwarcia (kontrola wariancji)

Identycznie jak w Eksp. 1: 25 ustandaryzowanych pozycji ECO z `experiments/openings_eco25.fen`, cyklicznie po jednej dla każdej z 30 partii w matchupie (pozycje 1-25 dla pierwszych 25 gier; 1-5 powtórzone dla gier 26-30 z zamianą kolorów).

## Adjudykacja

**WŁĄCZONA** z domyślnymi parametrami (`±0.05`, `20 ruchów`). Bez adjudykacji partie d=6 vs d=4 w pozycjach zbalansowanych mogą trwać 200+ ruchów.

## Książka otwarć

**Wyłączona** — testujemy czystą siłę algorytmu na różnych głębokościach.

## Procedura uruchomienia

### Krok 1 — Uruchom 10 matchupów (równolegle)

```powershell
# Terminal 1:
.\experiments\exp2\run_exp2_matchup.ps1 -Matchup 1   # MINIMAX_TRAD d=2 vs d=4
# Terminal 2:
.\experiments\exp2\run_exp2_matchup.ps1 -Matchup 2   # MINIMAX_TRAD d=3 vs d=4
# ... (terminale 3-10 dla matchupów 3-10)
```

Każdy skrypt:
1. Generuje single-matchup config JSON (`_exp2_matchup{N}.json`)
2. Woła `run_experiment.ps1` z wspólnym `-OutputSubDir = exp2_minimax_depth_<data>`
3. Wszystkie 10 matchupów pisze do **tego samego katalogu**

**Wszystkie 10 matchupów MUSI być uruchomione tego samego dnia** (tag oparty na dacie `yyyyMMdd`). Jeśli musisz przerwać i wznowić innego dnia, przekaż jawnie `-ExperimentTag` przy każdym wywołaniu.

### Krok 2 — Analiza zbiorcza (~2 min)

```powershell
.\experiments\exp2\run_exp2_analyze.ps1
```

Wykonuje dwie fazy:
- **Faza 1:** `analyze_experiment.py --elo --plots` (analiza generyczna)
- **Faza 2:** `exp2_depth_scaling.py` (analiza specyficzna — krzywa Elo vs głębokość, EBF, pruning)

## Wyjście — pliki CSV

Katalog wyjściowy: `engine/out/exp2_minimax_depth_<data>/`

| Plik | Zawartość |
|---|---|
| `_results.csv` | Per-game: matchup, game #, opening idx, result, termination, czas |
| `analysis_moves.csv` | Per-move: eval, czas, fazę gry, wszystkie metryki Minimax |
| `analysis_games.csv` | Per-game: result, total moves, termination reason, czas |
| `analysis_wdl.csv` | Per-matchup: 30 gier × W/D/L, white_score, avg_moves, avg_time |
| `analysis_elo.csv` | Bradley-Terry Elo dla wszystkich 5 głębokości × 2 ewaluatory (mixed) |
| `analysis_metrics_summary.csv` | Per-matchup × side: mean/std/median metryk algorytmu |
| `exp2_elo_per_depth.csv` | **Kluczowy:** Elo per (ewaluator, głębokość), z kotwicą w d=4 |
| `exp2_depth_summary.csv` | Per-(ewaluator, głębokość): avg_time, avg_nodes, EBF, pruning rates |
| `exp2_depth_summary.txt` | Human-readable raport |

## Wyjście — wykresy (`plots/`)

| Plik | Co pokazuje |
|---|---|
| `wdl_bars.png` | Słupki W/D/L per matchup |
| `time_per_move_boxplot.png` | Boxplot czasu/ruch per matchup |
| `eval_over_game.png` | Krzywe ewaluacji w czasie gry (próbka) |
| `minimax_pruning.png` | Generyczne użycie pruning techniques |
| `exp2_elo_curve.png` | **Kluczowy wykres** — Elo vs głębokość, 2 linie (TRAD/NN) |
| `exp2_ebf_curve.png` | Efektywny współczynnik rozgałęzienia (EBF) vs głębokość |
| `exp2_time_curve.png` | Średni czas/ruch vs głębokość (log scale) |
| `exp2_nodes_curve.png` | Średnia liczba węzłów vs głębokość (log scale) |
| `exp2_pruning_by_depth.png` | Pruning techniques (NMP, RFP, LMP, futility, SEE) per głębokość |

## Co mierzymy per ruch (zbierane w JSONL)

**Wspólne:** numer ruchu, strona, UCI, eval (centypionki), czas (s), faza gry [0..1]

**Minimax-specific (21+ metryk — wszystkie istotne dla Exp 2):**
- `nodes_searched` — **kluczowe dla EBF**
- `depth_completed` — czy ID osiągnął docelową głębokość (przy szachu może mniej)
- `nodes_per_depth` — **lista węzłów per ID iteration** → z tego wyliczany **EBF[d] = nodes[d] / nodes[d-1]**
- `tt_lookups`, `tt_hits`, `tt_cutoffs` → współczynniki trafień TT (powinny rosnąć z głębokością)
- `nmp_attempts`, `nmp_cutoffs` → skuteczność Null Move Pruning
- `rfp_cutoffs`, `futility_prunes`, `lmp_prunes`, `see_prunes` → pruning techniques (różne maksymalne głębokości aktywacji)
- `check_extensions`, `aspiration_researches`
- `qs_nodes`, `qs_max_depth`, `see_calls`
- `killer_hits`, `killer_checks` → skuteczność killer move
- `pv_from_tt` → ile razy PV move pochodził z TT
- `tt_size` → rozmiar TT po zakończeniu ruchu

## Analizy statystyczne

1. **Elo per głębokość** — Bradley-Terry ML, oddzielnie dla każdego ewaluatora, kotwica = d=4 Elo = 0
2. **Krzywa wzrostu Elo** — ile Elo zyskujemy przechodząc z d=4 do d=5? Z d=5 do d=6? (malejące zwroty)
3. **EBF (Effective Branching Factor)** — `EBF[d] = nodes[d] / nodes[d-1]`; powinno maleć z głębokością przy dobrym pruning
4. **Czas vs głębokość** — dopasowanie wykładnicze; predykcja czasu dla większych głębokości
5. **Pruning vs głębokość** — które techniki włączają się przy wyższych głębokościach (NMP wymaga `depth >= 3`, RFP `depth <= 3`, etc.)

## Co z planu badawczego jest istotne dla pracy magisterskiej

Eksperyment 2 odpowiada na pytanie: **jak skaluje się siła Minimaxa z głębokością?** Z czego wynika:
- **Czy d=4 to dobry default** dla pracy magisterskiej (czy d=3 byłby wystarczający dla porównań? czy d=5 dałby znacznie więcej?)
- **Czy NN kompensuje głębokość** — jeśli MINIMAX_NN d=4 ≈ MINIMAX_TRAD d=5 w Elo, to NN "warty" jest 1 poziomu głębokości
- **Predykcja kosztów** — z czasu/ruch i EBF można ekstrapolować ile zajęłoby d=7, d=8...

W pracy zazwyczaj prezentuje się:
- **Wykres:** krzywa Elo vs głębokość, 2 linie (TRAD/NN) — `exp2_elo_curve.png`
- **Wykres:** czas i węzły vs głębokość (log) — `exp2_time_curve.png`, `exp2_nodes_curve.png`
- **Wykres:** EBF vs głębokość — `exp2_ebf_curve.png`
- **Tabela:** ΔElo per poziom głębokości
- **Tabela:** pruning statistics per głębokość (`exp2_depth_summary.csv`)

## Szacunek czasu obliczeń

Czas/ruch rośnie wykładniczo z głębokością. Z danych Eksp. 1 (d=3):
- MINIMAX_TRAD d=3: ~0.5s/ruch → d=4 ~3-5s, d=5 ~25-50s, d=6 ~250-500s
- MINIMAX_NN d=3: ~3-6s/ruch → d=4 ~30-60s, d=5 ~300-600s, **d=6 niewykonalne (~3000-6000s/ruch)**

Szacunkowe czasy matchupów (30 gier, ~80 ruchów/gra = ~2400 ruchów per matchup):

| # | Matchup | Sredni czas/ruch (białe+czarne) | Szac. czas matchupu |
|---|---|---|---|
| 1 | TRAD d=2 vs d=4 | ~2s | ~1.3h |
| 2 | TRAD d=3 vs d=4 | ~2.5s | ~1.7h |
| 3 | TRAD d=4 vs d=4 | ~4s | ~2.7h |
| 4 | TRAD d=5 vs d=4 | ~15s | ~10h |
| 5 | TRAD d=6 vs d=4 | ~100s | ~67h ⚠ |
| 6 | NN d=2 vs d=4 | ~16s | ~11h |
| 7 | NN d=3 vs d=4 | ~17s | ~11h |
| 8 | NN d=4 vs d=4 | ~30s | ~20h |
| 9 | NN d=5 vs d=4 | ~165s | ~110h ⚠⚠ |
| 10 | NN d=6 vs d=4 | ~1500s | **~1000h niewykonalne** ❌ |

**Sumarycznie:**
- Wszystkie 10 matchupów serialnie: **~1200h** (50 dni)
- 10 matchupów równolegle: **bottleneck = matchup 10 (~1000h)** — niewykonalne
- **Realistycznie wykonalne:** matchupy 1-8 równolegle (~110h wall-clock = ~4.5 dnia)
- **Praktycznie:** Pomiń matchupy 9 i 10 (NN d=5, d=6) — w pracy odnotuj jako "out of scope due to computational cost"

## Ważne uwagi praktyczne

1. **Konfiguracja głębokości NN ma znacznie wyższy koszt** niż TRAD (Stockfish d=10 jako oracle eval)
2. **Wszystkie 10 matchupów MUSI być tego samego dnia** żeby trafiły do shared dir
3. **Rozważ uruchamianie matchupów w grupach:**
   - Grupa A (szybkie, 1-4 + 6-7): ~17h równolegle
   - Grupa B (średnie, 5 + 8): ~67h
   - Grupa C (powolne, 9 + 10): rozważ pominięcie
4. **Quick smoke test:** `.\experiments\exp2\run_exp2_matchup.ps1 -Matchup 3 -GamesPerPair 2` (d=4 vs d=4, najmniej minut)
5. **Adjudykacja jest KLUCZOWA** — bez niej d=6 matchupy mogłyby trwać 2x dłużej w pat-podobnych końcówkach
6. **Kalibracja niepotrzebna** — Exp 2 nie używa MCTS, więc nie czyta `_mcts_calibrated_time.txt`

## Co do dyskusji w pracy

- **Krzywa malejących zwrotów** — typowo w Minimaxie ΔElo per poziom maleje (~70-100 Elo na poziom przy małej głębokości, ~30-50 przy dużej)
- **Porównanie tempa wzrostu TRAD vs NN** — jeśli NN szybciej osiąga "plateau", oznacza to że jakość ewaluatora limituje korzyść z głębszego przeszukiwania
- **EBF jako miara jakości pruning** — w idealnym alpha-beta EBF ≈ sqrt(branching factor); w praktyce dobre engine'y mają EBF ~2-3
- **Praktyczne ograniczenie głębokości** — w pracy pokazać dlaczego d=4 lub d=5 to praktyczny default (NN d=6 niewykonalne w czasie rzeczywistym)
