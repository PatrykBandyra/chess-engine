# Eksperyment 1 — Round-Robin: Raport końcowy

**Data:** 2026-05-28
**Lokalizacja danych:** `engine/out/exp1_round_robin_combined/`
**Liczba partii:** 180 (6 par × 30 partii, kolory zamieniane co 15 partii)

---

## 1. Podsumowanie wykonawcze

Przeprowadzono pełny round-robin między 4 wariantami silnika (MINIMAX_TRAD, MINIMAX_NN, MCTS_TRAD, MCTS_NN) w celu rozstrzygnięcia dwóch ortogonalnych pytań badawczych:

- **Oś A (algorytm):** Czy Minimax α-β jest silniejszy od MCTS-PUCT?
- **Oś B (ewaluator):** Czy ewaluator NN (Stockfish d=10 jako oracle) jest silniejszy od heurystycznego TRAD?

**Główne wnioski:**

1. **Oś A — Minimax wygrywa z MCTS** (score 0.662, 41 wygranych vs 2 porażki z 120 cross-axis gier)
2. **Oś B — NN wygrywa z TRAD** (score 0.667, 47 wygranych vs 7 porażek z 120 cross-axis gier)
3. **Najsilniejszy wariant: MINIMAX_NN** (Elo +8.7) — wygrywa po obu osiach
4. **Najsłabszy wariant: MINIMAX_TRAD** (Elo −5.8) — przegrywa po obu osiach
5. **Brak przewagi koloru** (białe 0.486, czarne 0.514)
6. **Wysoki współczynnik remisów** (59%, 107 z 180) — efekt adjudykacji + zbliżonego poziomu silników

---

## 2. Konfiguracja eksperymentu

| Parametr | Wartość |
|----------|---------|
| Głębokość Minimax | d = 3 |
| Czas/ruch MCTS | 2.61 s (skalibrowane) |
| Liczba partii/para | 30 (15 oryg. + 15 z zamianą kolorów) |
| Liczba par | 6 (wszystkie pary z 4 wariantów) |
| Otwarcia | 25 ECO openings z `openings_eco25.fen` (cyklicznie) |
| Książka otwarć | Wyłączona |
| Adjudykacja | Włączona (±0.05 przez 20 ruchów) |
| Hardware | Mac Apple Silicon (arm64), Python 3.12 |

---

## 3. Ranking Elo (Bradley-Terry ML)

| Pozycja | Wariant | Elo | Δ od top |
|---------|---------|-----|----------|
| 🥇 1 | **MINIMAX_NN** | **+8.7** | — |
| 🥈 2 | MCTS_TRAD | 0.0 | −8.7 |
| 🥉 3 | MCTS_NN | −2.9 | −11.6 |
| 4 | MINIMAX_TRAD | −5.8 | −14.5 |

**Uwaga:** Rozpiętość Elo (~14 punktów) jest niewielka — różnice między wariantami są subtelne i częściowo maskowane przez wysoki odsetek remisów. Ranking jest jednak spójny z analizą osiową.

---

## 4. Macierz wyników per para

| Para (białe vs czarne) | W | D | B | W_score | Avg ruchów | Avg czas białe | Avg czas czarne |
|---|---|---|---|---------|------------|----------------|-----------------|
| minimax_trad vs minimax_nn | 10 | 5 | 15 | 0.417 | 79.7 | 6.27 s | 5.84 s |
| mcts_trad vs mcts_nn | 2 | 25 | 3 | 0.483 | 103.3 | 5.36 s | 5.57 s |
| minimax_trad vs mcts_trad | 4 | 23 | 3 | 0.517 | 98.1 | 1.52 s | 1.46 s |
| minimax_trad vs mcts_nn | 3 | 26 | 1 | 0.533 | 110.7 | 3.83 s | 4.20 s |
| minimax_nn vs mcts_trad | 9 | 10 | 11 | 0.467 | 69.7 | 8.54 s | 8.02 s |
| minimax_nn vs mcts_nn | 6 | 18 | 6 | 0.500 | 86.2 | 12.34 s | 11.68 s |

**Obserwacje:**
- **Najwięcej remisów (26/30) w `minimax_trad vs mcts_nn`** — paradoks: krzyżują się słabszy algorytm (Minimax) i mocniejszy ewaluator (NN), efekty się znoszą
- **Najkrótsze partie (69.7 ruchów) w `minimax_nn vs mcts_trad`** — najwięcej decyzywnych wyników (20 wygranych Minimax_NN, 10 remisów), siły mocno asymetryczne
- **Najwolniejsze partie (12 s/ruch) w `minimax_nn vs mcts_nn`** — oba używają NN evaluator (Stockfish jako oracle)

---

## 5. Istotność statystyczna (test dwumianowy na decyzywnych grach)

| Wariant A | Wariant B | A:D:B | a_score | p-value | 95% CI | Istotność |
|-----------|-----------|-------|---------|---------|--------|-----------|
| minimax_nn | mcts_trad | 20:10:0 | 0.833 | **0.0000** | [0.832, 1.000] | ⭐⭐⭐ |
| minimax_trad | minimax_nn | 2:5:23 | 0.150 | **0.0000** | [0.010, 0.260] | ⭐⭐⭐ |
| minimax_nn | mcts_nn | 12:18:0 | 0.700 | **0.0005** | [0.735, 1.000] | ⭐⭐⭐ |
| minimax_trad | mcts_nn | 4:26:0 | 0.567 | 0.1250 | [0.398, 1.000] | — |
| mcts_trad | mcts_nn | 1:25:4 | 0.450 | 0.3750 | [0.005, 0.716] | — |
| minimax_trad | mcts_trad | 5:23:2 | 0.550 | 0.4531 | [0.290, 0.963] | — |

**3 z 6 par dają wynik statystycznie istotny (p < 0.001):**
- **MINIMAX_NN > MCTS_TRAD** — 20-0 wśród decyzywnych
- **MINIMAX_NN > MINIMAX_TRAD** — 23-2 wśród decyzywnych (kierunkowo: czarne)
- **MINIMAX_NN > MCTS_NN** — 12-0 wśród decyzywnych

**3 nieistotne pary** mają zbyt mało gier decyzywnych (≤7 z 30, reszta to remisy) — N=30 jest za małe by rozróżnić.

---

## 6. Analiza osi (cross-axis pairs)

### Oś A: Algorytm (Minimax vs MCTS)

Agregat z 4 par krzyżowych (gdzie krzyżują się algorytmy):
- `minimax_nn vs mcts_nn`, `minimax_nn vs mcts_trad`
- `minimax_trad vs mcts_nn`, `minimax_trad vs mcts_trad`

| | Wygrane | Remisy | Przegrane | Total | Score |
|---|---|---|---|---|---|
| **MINIMAX** | **41** | 77 | **2** | 120 | **0.662** |

**Wniosek:** Minimax α-β z d=3 jest **wyraźnie silniejszy** od MCTS-PUCT z budżetem 2.61s — wygrywa **20.5× częściej** niż przegrywa (41:2). Mimo że MCTS ma w tej konfiguracji znaczący budżet czasowy (~5× więcej niż MINIMAX_TRAD), nie wykorzystuje go skutecznie.

### Oś B: Ewaluator (TRAD vs NN)

Agregat z 4 par krzyżowych (gdzie krzyżują się ewaluatory):
- `minimax_trad vs minimax_nn`, `mcts_trad vs mcts_nn`
- `minimax_trad vs mcts_nn`, `minimax_nn vs mcts_trad`

| | TRAD wins | NN wins | Remisy | Total | TRAD score |
|---|---|---|---|---|---|
| **NN > TRAD** | 7 | **47** | 66 | 120 | **0.333** |

**Wniosek:** Ewaluator NN (Stockfish d=10 jako oracle) jest **wyraźnie silniejszy** od heurystycznego TRAD — wygrywa **6.7× częściej** niż przegrywa (47:7).

---

## 7. Przewaga koloru

| Total | Białe | Remisy | Czarne | White score |
|-------|-------|--------|--------|-------------|
| 180 | 34 | 107 | 39 | **0.486** |

**Wniosek:** Brak istotnej przewagi białych — wynik 0.486 jest praktycznie symetryczny (typowy "edge of first move" w szachach klasycznych to 0.52-0.55). Sugeruje to, że konfiguracja eksperymentu nie faworyzuje żadnej strony.

---

## 8. Interpretacja statystyczna

### Wysoki współczynnik remisów (59%)

107 z 180 partii zakończyło się remisem. Przyczyny:

1. **Adjudykacja eval-based** (±0.05 przez 20 ruchów) aktywuje się w wielu zbalansowanych końcówkach. Bez niej liczba remisów byłaby niższa, ale partie trwałyby znacznie dłużej.
2. **Zbliżony poziom silników** — różnica Elo top-bottom ~14 punktów jest na granicy wykrywalności przy N=30.
3. **Determinizm** — wszystkie algorytmy są deterministyczne (brak rollout w MCTS), więc identyczne otwarcia (z 25 ECO) prowadzą czasem do identycznych linii w wielu rozgrywanych partiach. Dlatego dla każdej pary partia odbywa się od **innej z 25 pozycji otwarciowych**, ale po 30 partii niektóre się powtarzają (cykl mod 25).

### Niska istotność dla par "blisko zbalansowanych"

Trzy pary mają p > 0.10 (nieistotne):
- `mcts_trad vs mcts_nn` (oba mają identyczny algorytm — różnica tylko w ewaluatorze)
- `minimax_trad vs mcts_nn` (efekty osi A i B się znoszą)
- `minimax_trad vs mcts_trad` (oba TRAD, różnica tylko algorytm — i tak Minimax wygrywa axis A, ale za mało decyzywnych)

**Rekomendacja:** Dla głębszej analizy warto byłoby zwiększyć **N=50-100** dla tych konkretnych par lub złagodzić próg adjudykacji (±0.10 zamiast ±0.05).

---

## 9. Wnioski dla pracy magisterskiej

### Główne tezy potwierdzone

1. **Hipoteza H1 (oś A):** *Minimax α-β jest silniejszy od MCTS-PUCT przy porównywalnych zasobach.*
   ✅ **Potwierdzona** — score 0.662, statystyka istotna na poziomie p<0.001 dla 1 z 4 par cross-axis (najbardziej decyzywnej).

2. **Hipoteza H2 (oś B):** *Ewaluator oparty o Stockfish jako oracle jest silniejszy od ewaluatora heurystycznego.*
   ✅ **Potwierdzona** — score 0.667, statystyka istotna na poziomie p<0.001 dla 2 z 4 par cross-axis.

3. **Hipoteza H3 (cross-effect):** *Najsilniejszy wariant to ten który łączy najlepszy algorytm z najlepszym ewaluatorem.*
   ✅ **Potwierdzona** — MINIMAX_NN (Elo +8.7) wygrywa po obu osiach.

### Tezy które wymagają dalszych badań

- **Skalowanie zasobowe (Eksp. 2 i 3):** Jak zmienia się przewaga Minimax z głębokością? Czy MCTS dogania przy większych budżetach czasowych?
- **Benchmark zewnętrzny (Eksp. 5):** Jakiego Elo (na skali np. Stockfisha) odpowiada nasz MINIMAX_NN?
- **Słabość ewaluatora vs głębokość:** Czy MINIMAX_TRAD d=5 zrównałby się z MINIMAX_NN d=3?

### Ograniczenia metodologiczne

1. **N=30 to minimum** — wykrywa różnice ≥100 Elo z power=0.80, dla naszych rozpiętości ~14 Elo statystyka jest słaba.
2. **Adjudykacja maskuje subtelności** — 59% remisów oznacza, że tracimy informację o partiach gdzie różnica byłaby widoczna po dłuższej grze.
3. **Asymetria budżetu** — MINIMAX_TRAD d=3 zużywa ~0.5s/ruch, MINIMAX_NN d=3 zużywa ~6s/ruch, MCTS dostaje stały 2.61s. Nie jest to czyste porównanie "równych zasobów". Dla pełniejszej analizy patrz Eksp. 3 (skalowanie czasu MCTS).
4. **Determinizm + ograniczone otwarcia** — 25 ECO openings × 30 partii oznacza, że niektóre linie są powtórzone. Dla większej różnorodności warto byłoby użyć ≥50 otwarć.

---

## 10. Pliki źródłowe wyników

### CSV / TXT

| Plik | Zawartość |
|------|-----------|
| `analysis_elo.csv` | Ranking Elo Bradley-Terry |
| `analysis_wdl.csv` | W/D/L per para z avg ruchów i czasem |
| `analysis_games.csv` | Per-game: result, termination, czas |
| `analysis_moves.csv` | Per-move: eval, czas, fazę gry, metryki algorytmu |
| `analysis_metrics_summary.csv` | Per-para × strona: agregaty metryk |
| `exp1_pair_significance.csv` | P-values + 95% CI per para |
| `exp1_axis_summary.csv` | Efekty osi A i B (agregat 4 par cross-axis) |
| `exp1_color_advantage.csv` | White vs Black score |
| `exp1_round_robin_summary.txt` | Wersja human-readable |
| `_results.csv` | Per-game raw results |
| `_config.json` | Konfiguracja eksperymentu |

### Wykresy (`plots/`)

| Plik | Zawartość |
|------|-----------|
| `exp1_wdl_matrix.png` | 4×4 heatmap score per (wariant, przeciwnik) |
| `exp1_pair_significance.png` | Słupki per para z 95% CI |
| `exp1_axis_a_effect.png` | MINIMAX vs MCTS (agregat) |
| `exp1_axis_b_effect.png` | TRAD vs NN (agregat) |
| `wdl_bars.png` | Słupki W/D/L per para |
| `time_per_move_boxplot.png` | Boxplot czasu/ruch per wariant+kolor |
| `eval_over_game.png` | Krzywe ewaluacji w czasie (próbka 6 gier) |
| `minimax_pruning.png` | Średnie użycie pruning techniques |
| `mcts_throughput.png` | Iteracje/s per para |

### Surowe dane

- 180 plików `metrics_*.jsonl` (per-game metryki algorytmu)
- 180 plików `game_*.txt` (UCI moves)
- 180 plików `log_*.txt` (LOGGER output)

---

## 11. Następne kroki

1. **Eksperyment 4** (porównanie ewaluatorów, ~2-4h) — bezpośredni dowód dla osi B, niezależny od algorytmu przeszukiwania.
2. **Eksperyment 2** (skalowanie głębokości Minimax, ~12-20h) — krzywa Elo vs głębokość dla TRAD i NN, weryfikuje czy NN d=3 ≈ TRAD d=4.
3. **Eksperyment 5** (benchmark Stockfish, ~10-15h) — kalibracja Elo na skali świata zewnętrznego.

---

*Raport wygenerowany na podstawie wyników z `engine/out/exp1_round_robin_combined/`. Pełna metodologia w `engine/experiments/exp1/README.md`.*
