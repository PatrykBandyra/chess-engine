# Eksperyment 6 — Analiza wyników (dokładność taktyczna na łamigłówkach)

## 1. Cel eksperymentu i pytanie badawcze

Eksperyment ma odpowiedzieć na pytanie: **czy któryś z 4 wariantów silnika ma
strukturalną przewagę w „ostrych" pozycjach taktycznych, w których istnieje
jednoznaczny najlepszy ruch (`bm` z publicznych zbiorów testowych)?** W odróżnieniu
od Eksp. 1-3 (mierzących sumaryczną siłę gry przez tysiące pozycji w pełnych partiach)
Eksp. 6 mierzy **jakość pojedynczej decyzji w pozycji o znanym rozwiązaniu**.

Dodatkowo dla wariantów Minimax: pomiar **minimalnej głębokości ID**, na której
rozwiązanie pojawia się w PV — pozwala porównać „taktyczną szybkość widzenia"
TRAD vs NN.

Hipotezy:

- **H1:** Minimax dominuje MCTS na łamigłówkach taktycznych — α-β z TT i quiescence
  precyzyjnie wylicza wymuszone sekwencje, MCTS PUCT bez prior NN rozprasza wizyty.
- **H2:** NN (Stockfish d=10) wnosi przewagę w obu algorytmach, ponieważ wyrocznia
  na liściach widzi taktykę bezpośrednio.
- **H3:** `MINIMAX_NN` znajduje rozwiązanie przy **mniejszej głębokości ID** niż
  `MINIMAX_TRAD` — wyrocznia kompensuje płytsze wyszukiwanie.
- **H4** (walidacja z Eksp. 1, 4, 5): ranking solve rate jest zgodny z rankingiem
  Elo absolutnego z Eksp. 5 i z rankingiem move-agreement z Eksp. 4b.
- **H5** (sanity check): `timeout rate ≈ 0%` przy progu 120 s/łamigłówkę.

## 2. Założenia metodyczne

**Uczestnicy (4 warianty):**

| Wariant | Algorytm | Ewaluator | Parametr |
|---|---|---|---|
| `MINIMAX_TRAD_d4` | Minimax α-β | Heurystyczny | `d = 4` |
| `MINIMAX_NN_d3` | Minimax α-β | NN (Stockfish d=10) | `d = 3` |
| `MCTS_TRAD` | MCTS PUCT | Heurystyczny | `t = 2.61 s` |
| `MCTS_NN` | MCTS PUCT | NN | `t = 2.61 s` |

Parametry zasobu są **spójne z Eksp. 5 i 7** (silne praktyczne konfiguracje),
nie z Eksp. 1 (gdzie Minimaxy miały sztywne `d = 3` dla czystego porównania osi B).

**Zbiór łamigłówek:** **300 pozycji WAC** (Win At Chess, klasyczny benchmark
taktyczny z lat 90.) z `puzzles/wac.epd`, sparsowany do `puzzles.json` przez
`prepare_puzzles.py`. Każdy rekord zawiera `fen`, `best_moves_uci` (lista
akceptowanych rozwiązań — zwykle 1, czasem 2-3), `side_to_move`, `theme = "WAC"`.

> Konfiguracja jednolitego pliku tematu: w obecnym uruchomieniu **nie użyto STS
> (1500 zadań strategicznych) ani Bratko-Kopec (24 zadania mieszane)** — heatmapa
> per temat (`exp6_solve_by_theme.csv`) ma tylko jeden wiersz „WAC" per wariant.
> Pełne pokrycie tematyczne (planowane dla pracy magisterskiej) wymaga dodatkowego
> uruchomienia z plikiem STS.

**Procedura per łamigłówka:**

1. `main.py -i <fen>` — wariant testowany jest stroną na ruchu.
2. Przeciwnik = Stockfish skill 0 (najsłabszy/najszybszy filler, ~10-50 ms/ruch).
3. Agresywna adjudykacja `-adjt 0.1 -adjm 5` (eval ±0.1 przez 5 ruchów) — partia
   kończy się kilka ruchów po wykonaniu pierwszego ruchu silnika.
4. Z `game.txt` parsowany jest **pierwszy ruch testowanego wariantu** (linia
   `1: <uci>` lub `2: <uci>`).
5. `solved = (ruch_silnika ∈ best_moves_uci)`.
6. Dla Minimax: z logu parsowana **najmniejsza ID iteration depth**, na której
   PV move trafia rozwiązanie → `min_depth_to_solve`.

**Timeout:** 120 s per łamigłówka (zabezpieczenie przed zawieszeniem).

## 3. Zbierane metryki

**Per łamigłówka (`exp6_variantN_*.csv`):**
- `puzzle_id`, `theme`, `fen`, `expected_uci`, `expected_san`,
- `engine_move_uci` — faktyczny ruch silnika,
- `solved` — boolean,
- `duration_s` — całkowity czas wywołania (wraz z narzutem startu Python),
- `timed_out` — czy hit 120 s,
- `min_depth_to_solve` — tylko Minimax (pusty dla MCTS).

**Agregaty:**

- `exp6_solve_rate.csv` — per wariant: `n`, `solved`, `avg_time_s`, `timeouts`, `solve_rate`,
- `exp6_solve_by_theme.csv` — per (wariant, temat),
- `exp6_minimax_depth_to_solve.csv` — per Minimax wariant: `mean`, `median`, `min`,
  `max`, `count` głębokości znalezienia rozwiązania.

**Brak Bradleya-Terry'ego** — łamigłówki nie są porównaniem 1-vs-1, więc nie ma
„wyników gier" do dopasowania.

## 4. Wyniki

### 4.1 Solve rate per wariant — `exp6_solve_rate.csv`

| Wariant | n | Rozwiązanych | Solve rate | Avg czas/puzzle | Timeouts (z 300) |
|---|---:|---:|---:|---:|---:|
| **MINIMAX_NN d=3** | 300 | **202** | **67.33 %** | 64.19 s | **75 (25.0 %)** |
| **MINIMAX_TRAD d=4** | 300 | 195 | **65.00 %** | 11.34 s | 0 |
| **MCTS_NN** | 300 | 115 | **38.33 %** | 67.65 s | **100 (33.3 %)** |
| **MCTS_TRAD** | 300 | 107 | **35.67 %** | 24.03 s | 0 |

Wizualizacja: `engine/out/exp6_puzzles_combined/plots/exp6_solve_rate_bars.png`.

### 4.2 Solve rate per temat — `exp6_solve_by_theme.csv`

| Wariant | Temat | n | Solved | Solve rate |
|---|---|---:|---:|---:|
| MINIMAX_NN_d3 | WAC | 300 | 202 | 0.6733 |
| MINIMAX_TRAD_d4 | WAC | 300 | 195 | 0.6500 |
| MCTS_NN | WAC | 300 | 115 | 0.3833 |
| MCTS_TRAD | WAC | 300 | 107 | 0.3567 |

Wizualizacja: `plots/exp6_solve_by_theme_heatmap.png` (zredukowana do jednej
kolumny ze względu na brak stratyfikacji tematycznej WAC).

### 4.3 Głębokość ID „odkrycia" rozwiązania (Minimax) — `exp6_minimax_depth_to_solve.csv`

| Wariant | mean | median | min | max | count |
|---|---:|---:|---:|---:|---:|
| **MINIMAX_NN d=3** | **1.04** | **1.0** | 1 | 3 | 202 |
| **MINIMAX_TRAD d=4** | 2.33 | 2.0 | 1 | 4 | 195 |

Wizualizacja: `plots/exp6_minimax_depth_hist.png`.

**Interpretacja:** `MINIMAX_NN` rozwiązuje **niemal wszystko już na głębokości 1**
(mediana = 1, średnia 1.04). `MINIMAX_TRAD` potrzebuje średnio głębokości ~2.3
(mediana 2, max 4).

### 4.4 Czas vs solve rate

Średni czas na łamigłówkę bardzo różni się między wariantami:

| Wariant | Avg czas/puzzle | „Czas na rozwiązanie" (czas/solved) |
|---|---:|---:|
| MINIMAX_TRAD d=4 | 11.34 s | 11.34 × 300 / 195 = **17.4 s** |
| MCTS_TRAD | 24.03 s | 24.03 × 300 / 107 = **67.4 s** |
| MINIMAX_NN d=3 | 64.19 s | 64.19 × 300 / 202 = **95.3 s** |
| MCTS_NN | 67.65 s | 67.65 × 300 / 115 = **176.5 s** |

`MINIMAX_TRAD d=4` jest **najwydajniejszym taktycznie wariantem czasowo** —
ok. 6× szybszy per rozwiązany puzzle niż MINIMAX_NN i ok. 10× szybszy niż MCTS_NN.

## 5. Dyskusja

### 5.1 Minimax dominuje MCTS — efekt rzędu 27-29 p.p. (H1)

`MINIMAX_NN` (67 %) i `MINIMAX_TRAD` (65 %) wyraźnie przewyższają MCTS-y (38 %, 36 %).
Różnica **+29 p.p. NN-NN, +29 p.p. TRAD-TRAD** jest spójna z ustaleniami:

- **Eksp. 4b** (move agreement z SF-d20): `MINIMAX_TRAD` top-1 = 0.555 vs
  `MCTS_TRAD` = 0.395 (+16 p.p.). Eksp. 6 powiększa tę różnicę na łamigłówkach
  taktycznych (+29 p.p.), co potwierdza, że MCTS jest **strukturalnie słaby** w pozycjach
  „ostrych".
- **Mechanizm:** α-β z TT cache'uje wyniki w wymuszonych sekwencjach taktycznych;
  quiescence przedłuża linie wymian/szachów do stabilności. PUCT bez prior NN
  rozprasza wizyty proporcjonalnie do UCB1, co w pozycjach z jedną „wąską"
  wygraną nie wystarcza — większość wizyt idzie na ruchy „bezpieczne",
  a rozwiązanie nie otrzymuje dość rolloutów.

H1 jednoznacznie potwierdzona.

### 5.2 NN dodaje tylko +2.3-2.7 p.p. (H2)

Wbrew oczekiwaniom efekt osi B (TRAD vs NN) jest **bardzo mały**:

| Algorytm | TRAD | NN | Δ (NN − TRAD) |
|---|---:|---:|---:|
| Minimax | 65.00 % | 67.33 % | **+2.33 p.p.** |
| MCTS | 35.67 % | 38.33 % | **+2.67 p.p.** |

Zaskakująco mały skok w porównaniu z:
- Eksp. 1 (gdzie NN dominował: `NN_score = 0.667`),
- Eksp. 5 (gdzie NN dodawało +166 Elo w Minimax, ≥+418 Elo w MCTS).

**Najprawdopodobniejsze wyjaśnienie:** w pojedynczej pozycji taktycznej rozwiązanie
zwykle jest „wymuszone" w 1-2 ruchach. Dla takiej sytuacji **algorytm wyszukiwania
ma większe znaczenie niż jakość ewaluacji** — α-β znajdzie ruch przez głębokość,
nawet ze słabą heurystyką. NN wnosi przewagę głównie tam, gdzie pozycja jest
**niewymuszona** (długie partie z wieloma „cichymi" decyzjami) — tj. w Eksp. 1, 5, 8.

H2 częściowo potwierdzona, ale skala efektu jest minimalna w kontekście taktycznym.

### 5.3 NN „widzi" taktykę przy d=1 (H3) — najmocniejsza obserwacja

`MINIMAX_NN d=3` rozwiązuje 202 z 202 udanych pozycji **na średniej głębokości ID 1.04**
(mediana 1, max 3). `MINIMAX_TRAD d=4` potrzebuje średnio głębokości 2.33 (mediana 2).
Mechanizm:

- Wyrocznia Stockfish d=10 na liściu otrzymuje pozycję **po jednym ruchu** wariantu.
  Jeśli ruch jest „rozwiązaniem" taktycznym, Stockfish d=10 już to widzi w swojej
  ocenie (np. „+15.6 — mat w 2"). Minimax_NN dostaje od razu sygnał: ten ruch
  jest dramatycznie lepszy od pozostałych.
- Heurystyka TRAD ocenia tylko **statyczny obraz** pozycji po pierwszym ruchu —
  nie zauważa mata-w-2 ani nawet wymuszonego wymienienia. Minimax_TRAD musi
  **fizycznie rozegrać** dodatkowy ruch lub dwa, by zobaczyć skutek.

Konsekwencja: `MINIMAX_NN d=3` w praktyce działa jak **„Stockfish d=10 z dodatkowym
ruchem α-β"** — Stockfish-jako-wyrocznia robi całą pracę taktyczną, Minimax-α-β
robi tylko jeden ruch eksploracji. To wyjaśnia również, dlaczego skok solve rate NN
vs TRAD jest mały — `MINIMAX_TRAD d=4` w 65 % przypadków też dochodzi do rozwiązania
przez głębsze przeszukiwanie. **H3 potwierdzona.**

### 5.4 Wysoki timeout rate dla wariantów NN (H5 odrzucona)

| Wariant | Timeouts | % |
|---|---:|---:|
| MINIMAX_TRAD d=4 | 0 | 0 % |
| MCTS_TRAD | 0 | 0 % |
| MINIMAX_NN d=3 | **75** | **25.0 %** |
| MCTS_NN | **100** | **33.3 %** |

**Hipoteza H5 odrzucona.** Warianty NN regularnie przekraczają timeout 120 s na łamigłówkę.
Konsekwencje metodyczne:

1. **Solve rate NN są konserwatywnie zaniżone.** Łamigłówki, na których wariant
   NN „nie zdążył", są liczone jako nierozwiązane. Faktyczna „możliwość rozwiązania
   w nieograniczonym czasie" mogłaby być wyższa.
2. **Avg time/puzzle dla NN jest skrzywione w górę** — średnia 64-68 s odzwierciedla
   głównie czasy łamigłówek dochodzących do timeout (zasięg `~120 s`),
   nie typowy czas decyzji.
3. **Mechanizm timeoutów:** najprawdopodobniej procedura `_run_variant_puzzles.py`
   uruchamia `main.py` jako subprocess i czeka na zakończenie partii. Adjudykacja
   `-adjt 0.1 -adjm 5` wymaga 5 ruchów z eval w przedziale ±0.1 — jeśli MCTS_NN
   lub MINIMAX_NN przy `d = 3` z każdym ruchem trwa 6-30 s, samych 5 ruchów po
   ruchu testowanym daje ~30-150 s, plus ruch pierwszego silnika + ruch Stockfisha.
   120 s timeout jest zbyt niski.

Rekomendacja: w przyszłym uruchomieniu **podnieść timeout do 300 s** lub
**złagodzić adjudykację** (`-adjm 3`). Alternatywnie: skrypt powinien parsować
ruch testowanego wariantu tuż po jego wykonaniu (bez czekania na rozegranie partii
do końca).

### 5.5 Walidacja rankingu z Eksp. 4 i 5 (H4)

| Wariant | Solve rate (Eksp. 6) | Top-1 match rate (Eksp. 4b) | Elo absol. (Eksp. 5) |
|---|---:|---:|---:|
| MINIMAX_NN d=3 | **67.33 %** | brak* | **1683** |
| MINIMAX_TRAD d=4/d=3* | 65.00 % | 55.50 %* | 1517 |
| MCTS_NN | 38.33 % | brak* | 1318 |
| MCTS_TRAD | 35.67 % | 39.50 % | 900 |

\* Eksp. 4b wykluczył NN z porównań (auto-korelacja z SF d=20 ground truth);
podana wartość MINIMAX_TRAD jest dla `d=3`, w Eksp. 6 testowano `d=4`.

Ranking jest **identyczny** między Eksp. 5 i Eksp. 6 (MINIMAX_NN > MINIMAX_TRAD >>
MCTS_NN > MCTS_TRAD). **H4 potwierdzona.** Eksp. 4b porównanie TRAD vs TRAD
(`MINIMAX_TRAD` 0.555 vs `MCTS_TRAD` 0.395, Δ = +16 p.p.) jest spójne z Eksp. 6
(`MINIMAX_TRAD` 0.650 vs `MCTS_TRAD` 0.357, Δ = +29 p.p.) — Eksp. 6 powiększa
gap, bo łamigłówki są wybierane jako pozycje o ostrym rozstrzygnięciu,
gdzie różnica między „znajdę najlepszy ruch" a „znajdę dobry ruch" jest binarna.

### 5.6 Wydajność czasowa — Minimax_TRAD jako najlepszy stosunek jakości do kosztu

Spójrzmy na „solve rate per sekundę" (efektywny throughput taktyczny):

| Wariant | Solve rate / avg time | Interpretacja |
|---|---:|---|
| **MINIMAX_TRAD d=4** | **5.73 %/s** | najlepszy stosunek |
| MCTS_TRAD | 1.48 %/s | |
| MINIMAX_NN d=3 | 1.05 %/s | |
| MCTS_NN | 0.57 %/s | najgorszy stosunek |

`MINIMAX_TRAD d=4` jest **5.5× wydajniejszy taktycznie** niż MINIMAX_NN i **10×**
niż MCTS_NN. Ten wynik **nie zachodzi** w sile sumarycznej (Eksp. 5: MINIMAX_NN
przewyższa MINIMAX_TRAD o 166 Elo), ale ma znaczenie praktyczne: w aplikacjach
o ograniczonym budżecie czasowym (np. szybkie odpowiedzi engine'u) `MINIMAX_TRAD`
oferuje większość siły taktycznej za ułamek kosztu.

### 5.7 Ograniczenia

- **Tylko zbiór WAC (300 pozycji), brak STS i Bratko-Kopec** — niemożliwa stratyfikacja
  per temat strategiczny. Heatmapa `exp6_solve_by_theme_heatmap.png` ma jedną kolumnę.
- **Timeout 120 s jest zbyt niski** dla wariantów NN (25-33 % zadań nie ukończonych).
  Solve rate NN są dolnym oszacowaniem.
- **WAC jest stary i wąsko taktyczny** — głównie maty-w-N i wymuszone wymiany.
  Nie testuje siły strategicznej, pozycyjnej, końcówkowej.
- **Wielokrotne akceptowane ruchy:** `best_moves_uci` to lista — czasem 2-3 ruchy
  są równo dobre. Silnik trafiający „któreś" z dopuszczonych liczy się jako solved.
  Łagodzi to nieco wynik, ale spójnie dla wszystkich wariantów.
- **Brak istotności statystycznej** — różnica 2.33 p.p. (NN vs TRAD w Minimax)
  przy n=300 ma p-value rzędu 0.5 (test dwumianowy proporcji) — nie jest
  statystycznie istotna. Różnica Minimax vs MCTS (~29 p.p.) jest natomiast
  bardzo istotna (p << 0.001).
- **`min_depth_to_solve` policzone tylko dla rozwiązanych pozycji** (202 vs 195
  partii) — wartości średnie pochodzą z różnych podzbiorów, więc porównanie
  nie jest matched-pair.

## 6. Wnioski

1. **Minimax dominuje MCTS na taktyce** — solve rate 65-67 % vs 36-38 % (różnica
   ~29 p.p.). Efekt strukturalny: α-β z TT precyzyjnie wylicza wymuszone sekwencje,
   PUCT je rozprasza. Spójne z Eksp. 4b (różnica +16 p.p.) — w pozycjach ostrych
   gap rośnie.
2. **NN dodaje tylko +2.3-2.7 p.p.** solve rate — niewspółmiernie mało w porównaniu
   z wpływem NN na pełną grę (Eksp. 5: +166 Elo w Minimax, +418 Elo w MCTS).
   Wyjaśnienie: w pozycjach taktycznych algorytm dominuje nad jakością ewaluacji.
3. **MINIMAX_NN znajduje rozwiązania przy ID = 1** (mean 1.04, median 1). Wyrocznia
   Stockfish d=10 wnosi „taktyczny wzrok" bez konieczności dalszego przeszukiwania.
   MINIMAX_TRAD potrzebuje średnio d = 2.33 (z reguły 2 plejów wymuszonej linii).
4. **Wysoki timeout rate dla wariantów NN** (25 % MINIMAX_NN, 33 % MCTS_NN) —
   ograniczenie metodyczne: solve rate NN są konserwatywnie zaniżone. Zalecenie:
   podnieść timeout do 300 s lub zmienić procedurę parsowania ruchu.
5. **Ranking taktyczny jest identyczny z rankingiem absolutnym Eksp. 5**
   (MINIMAX_NN > MINIMAX_TRAD > MCTS_NN > MCTS_TRAD) — walidacja krzyżowa
   obu pomiarów.
6. **MINIMAX_TRAD d=4 ma najlepszy stosunek jakości taktycznej do kosztu czasowego**
   (5.73 %/s vs 1.05 %/s MINIMAX_NN). Wniosek praktyczny dla wdrożeń budżetowych.
7. **Brak pokrycia STS/Bratko-Kopec** ogranicza analizę tematyczną — pełna heatmapa
   per motyw strategiczny (planowana w pracy) wymaga osobnego uruchomienia.

Materiał stanowi podstawę szóstej części rozdziału eksperymentalnego (dokładność
taktyczna): rysunek 4.15 (`exp6_solve_rate_bars.png` — słupkowy solve rate),
rysunek 4.16 (`exp6_minimax_depth_hist.png` — histogram głębokości znalezienia
rozwiązania), tabela 4.12 (solve rate per wariant), tabela 4.13 (głębokość ID),
tabela 4.14 (efektywność czasowa). **Przed finalną wersją pracy** rozważyć
ponowne uruchomienie z `timeout = 300 s` oraz pełnym zbiorem WAC + STS dla
analizy tematycznej.
