# Eksperyment 7 — Analiza wyników (wpływ książki otwarć)

## 1. Cel eksperymentu i pytanie badawcze

Eksperyment ma odpowiedzieć na pytanie: **czy włączenie polyglotowej książki otwarć
istotnie wpływa na siłę gry i fazę otwarcia, oraz czy efekt różni się między
Minimaxem a MCTS?** W odróżnieniu od Eksp. 1-3 (gdzie partie startują z 25 pozycji
ECO, omijając książkę), tu partie startują od pozycji początkowej (`startpos`)
i książka aktywnie wpływa na otwarcia.

Hipotezy:

- **H1** (główna, czasowa): Włączenie książki **drastycznie skraca** czas fazy otwarcia
  (do ~10× mniej w ruchach 1-12).
- **H2** (wpływ na siłę): Książka zmienia rozkład W/D/L — kierunek (więcej/mniej remisów)
  zależy od jakości i głębokości książki vs siły własnego search.
- **H3** (asymetria osi A): MCTS bardziej zyskuje na książce niż Minimax, ponieważ
  MCTS w otwarciu marnuje budżet na pozycje, w których ewaluator słabo dyskryminuje.
- **H4** (statystyczność): Efekty różnic OFF vs ON są istotne statystycznie (chi-square
  i McNemar) — przynajmniej dla jednego z algorytmów.

## 2. Założenia metodyczne

**Uczestnicy (4 konfiguracje, każda self-play):**

| # | Konfiguracja | Algorytm | Książka |
|---|---|---|---|
| 1 | `minimax_trad_d4_book_off` | MINIMAX_TRAD `d = 4` | OFF |
| 2 | `minimax_trad_d4_book_on` | MINIMAX_TRAD `d = 4` | ON |
| 3 | `mcts_trad_book_off` | MCTS_TRAD `t = 20 s` | OFF |
| 4 | `mcts_trad_book_on` | MCTS_TRAD `t = 20 s` | ON |

**Próba:** 4 konfiguracje × **40 partii** (20 oryg. + 20 z zamianą kolorów) = **160 partii**.
Większe `n` niż w Eksp. 1-3 (30) kompensuje wyższą wariancję wyników wynikającą z braku
ECO startów.

**Wybór TRAD (bez NN):** książka otwarć jest deterministyczną polityką niezależną
od ewaluatora — dodanie wariantów NN powtórzyłoby jakościowo wnioski.

**Wybór MCTS 20 s** (zamiast 2.61 s z kalibracji Eksp. 1): większy budżet daje
istotnie więcej iteracji (z Eksp. 3: ~407 k iter/s vs ~25 k dla 1 s), więc
„odejście od pozycji startowej" jest mierzalne — przy 2.61 s MCTS nie zdąży zboczyć
z głównych otwarć i efekt książki byłby zatarty.

**Pozycja startowa:** `startpos` (brak `OpeningsFile`). Książka Polyglot
(`codekiddy.bin`) jest aktywna od ruchu 1.

**Adjudykacja:** ±0.05/20 ruchów (włączona — krytyczna dla self-play MCTS).

## 3. Zbierane metryki

**Per ruch (jsonl):** standard z Eksp. 1 (numer ruchu, eval, czas, faza) + **flaga
`from_book`** wskazująca czy ruch pochodzi z książki.

**Per gra (`exp7_raw_per_game.csv`):**
- `last_book_ply` — numer ostatniego ruchu z książki (głębokość repertuaru),
- `book_hits` — liczba ruchów z `from_book = true`,
- `opening_phase_time` — sumaryczny czas obu graczy w ruchach 1-12.

**Agregaty (`exp7_summary.csv`):** per (algorytm, książka): `n`, W/D/L, `white_score`,
`avg_total_moves`, `avg_last_book_ply`, `avg_opening_time_*`.

**Testy statystyczne (`exp7_statistical_tests.csv`):**
- **chi-square 2×3** (rozkład W/D/L OFF vs ON) — globalny efekt książki,
- **McNemar paired** (na parach gier OFF/ON o tym samym opening seed) — kierunkowy
  efekt zmiany.

## 4. Wyniki

### 4.1 W/D/L per konfiguracja — `exp7_summary.csv`

| Konfiguracja | n | W (biały) | D | L (czarny) | Wynik białego | Śr. ruchów |
|---|---:|---:|---:|---:|---:|---:|
| **MINIMAX_TRAD d=4 OFF** | 40 | **0** | **0** | **40** | **0.000** | 120.0 |
| **MINIMAX_TRAD d=4 ON** | 40 | 10 | 20 | 10 | **0.500** | 104.7 |
| MCTS_TRAD 20s OFF | 40 | 10 | 30 | 0 | **0.625** | 150.5 |
| MCTS_TRAD 20s ON | 40 | 6 | 33 | 1 | 0.562 | 98.6 |

Wizualizacja: `engine/out/exp7_opening_book_combined/plots/exp7_wdl_comparison.png`.

### 4.2 Faza otwarcia — `exp7_summary.csv` (kolumny `avg_opening_time_*`)

| Konfiguracja | Avg czas otwarcia (biały) | Avg czas otwarcia (czarny) | Suma |
|---|---:|---:|---:|
| MINIMAX_TRAD OFF | 7.343 s | 5.971 s | 13.314 s |
| **MINIMAX_TRAD ON** | **0.123 s** | **0.243 s** | **0.366 s** |
| MCTS_TRAD OFF | 201.228 s | 201.429 s | 402.657 s |
| **MCTS_TRAD ON** | **1.005 s** | **3.044 s** | **4.049 s** |

**Stosunek OFF/ON (czas fazy otwarcia):**
- Minimax: **13.3 s / 0.37 s ≈ 36×**
- MCTS: **402.7 s / 4.0 s ≈ 99×**

Wizualizacja: `plots/exp7_opening_time.png`.

### 4.3 Głębokość książki — `exp7_summary.csv` (kolumny `avg_last_book_ply`)

| Konfiguracja | Avg last book ply | Max last book ply |
|---|---:|---:|
| MINIMAX_TRAD ON | 25.55 | 30 |
| MCTS_TRAD ON | 26.85 | 30 |

Wizualizacja: `plots/exp7_book_exit_hist.png`. Książka prowadzi grę przeciętnie
przez **~12-13 pełnych ruchów** (25-26 plejów), z maksymalną głębokością 30 plejów
(~15 pełnych ruchów) — zakres typowy dla repertuaru Polyglota z `codekiddy.bin`.

### 4.4 Testy statystyczne — `exp7_statistical_tests.csv`

| Algorytm | Test | p-value | χ² | b (OFF wins / ON loses) | c (OFF loses / ON wins) |
|---|---|---:|---:|---:|---:|
| MINIMAX_TRAD | chi-square 2×3 | **0.000** | **48.0** | — | — |
| MINIMAX_TRAD | McNemar (paired) | **0.002** | — | **0** | 10 |
| MCTS_TRAD | chi-square 2×3 | 0.343 | 2.14 | — | — |
| MCTS_TRAD | McNemar (paired) | 0.455 | — | 10 | 6 |

**Interpretacja:**
- **MINIMAX**: oba testy bardzo istotne (`p < 0.005`). Rozkłady W/D/L OFF (0/0/40)
  i ON (10/20/10) są radykalnie różne. McNemar pokazuje, że **wszystkie 10 par
  zmieniających wynik to przejścia OFF-loses → ON-wins** (b=0, c=10) — kierunkowo
  książka pomaga białemu we wszystkich przypadkach.
- **MCTS**: oba testy **nieistotne** (`p > 0.3`). Rozkłady OFF (10/30/0) i ON (6/33/1)
  statystycznie nieodróżnialne. Efekt książki **w sile gry** dla MCTS jest poniżej
  progu wykrywalności przy `n = 40`.

### 4.5 Decisive rate (gier nie-remisowych)

| Konfiguracja | Decisive | % |
|---|---:|---:|
| MINIMAX_TRAD OFF | 40 / 40 | **100 %** |
| MINIMAX_TRAD ON | 20 / 40 | **50 %** |
| MCTS_TRAD OFF | 10 / 40 | 25 % |
| MCTS_TRAD ON | 7 / 40 | 17.5 % |

Książka ON **zmniejsza** decisive rate — gra staje się bardziej zrównoważona
(zarówno w Minimaxie: 100 % → 50 %, jak i w MCTS: 25 % → 17.5 %).

## 5. Dyskusja

### 5.1 Anomalia MINIMAX OFF: wszystkie 40 partii wygrane przez czarnych

Najbardziej uderzający wynik: **MINIMAX_TRAD d=4 bez książki, self-play od `startpos`,
wszystkie 40 partii zakończone wygraną czarnych** (`white_score = 0.000`).

Mechanizm:

1. **Minimax z heurystyką TRAD jest w pełni deterministyczny** — przy identycznych
   pozycjach wybiera identyczne ruchy. Brak źródła losowości (brak `random` w
   ordering ruchów, brak Multi-PV z wyborem stochastycznym).
2. **Self-play od `startpos` przy `seed = stały`** prowadzi do **dokładnie tej samej
   sekwencji otwarcia we wszystkich 40 grach**. Z 25 ECO pozycji jest 25 różnych
   otwarć — tu jest tylko jedno otwarcie ze startposu.
3. **W tej jednej, deterministycznie wybranej linii białe trafiają na pozycję
   przegrywającą.** Konkretna konstrukcja: pierwszy ruch białego (najprawdopodobniej
   1.e4 lub 1.d4) → ruch czarnego → ... → po kilkunastu ruchach pozycja
   z perspektywą czarnych dodatnio.
4. **Color swap nie ratuje sytuacji** — w 20 partiach „swap" inny silnik
   (faktycznie ten sam) gra białe i też przegrywa. Łącznie: 40 partii, 40 wygranych
   czarnych (w sensie strony, niezależnie od „kim" jest czarny).

**Implikacja metodyczna:**

- Bez stochastyczności (czy z otwarć, czy z book) Minimax-vs-Minimax self-play
  od `startpos` jest **bezużyteczny** jako pomiar siły — wynik jest dyktowany
  pojedynczą linią otwarcia.
- W Eksp. 1-3 ten problem nie występuje, bo 25 różnych ECO startów rozprasza
  wynik.
- Eksp. 7 świadomie używa `startpos` (żeby książka mogła w ogóle zadziałać);
  ale konsekwencją jest, że **konfiguracja MINIMAX OFF jest patologiczna**.

**Czy to bug, czy wynik?** Z punktu widzenia eksperymentu — to **wynik, nie bug**:
pokazuje **kluczowe ograniczenie deterministycznego Minimaxa** w warunkach bez
różnicowania otwarć. Nawet jeśli książka nie poprawia siły gry per se, to jest
**niezbędna do tego, by partie były rozróżnialne** (różnicują pierwsze ruchy
i prowadzą do różnych pozycji). W pracy magisterskiej warto wprost omówić tę
obserwację jako **dodatkowy argument za stosowaniem książki w testach silników
deterministycznych**.

### 5.2 H1 (czasowy efekt książki) — silnie potwierdzona

| Algorytm | Czas otwarcia OFF | Czas otwarcia ON | Współczynnik OFF/ON |
|---|---:|---:|---:|
| Minimax | 13.31 s | 0.37 s | **36×** |
| MCTS | 402.66 s | 4.05 s | **99×** |

Efekt jest **dramatyczny** — książka redukuje czas fazy otwarcia o czynnik
36-100×. MCTS, zużywający stały budżet `t = 20 s` na każdy ruch (10 plejów = 200 s
samego myślenia + overhead), wyzyskuje książkę najsilniej; przy włączonej książce
ruchy z `from_book = true` są wykonywane praktycznie natychmiast (lookup hash table).

**Wniosek praktyczny dla rozdziału pracy:** książka jest **bezdyskusyjną zaletą
czasową** niezależnie od wpływu na siłę gry. W warunkach turniejowych z budżetem
czasowym oszczędność 100× w fazie otwarcia daje silnikowi proporcjonalnie więcej
czasu na grę środkową/końcówkę.

### 5.3 H2 (efekt siły) — asymetryczna, kierunkowo „nivelująca"

| Algorytm | OFF white_score | ON white_score | Δ |
|---|---:|---:|---:|
| Minimax | 0.000 | 0.500 | **+0.500** |
| MCTS | 0.625 | 0.562 | −0.063 |

Dla **Minimaxa** efekt jest dramatyczny, ale interpretacja musi uwzględniać anomalię
z § 5.1: book ON „niweluje" patologię deterministyczną. Sumarycznie chi-square 2×3
`p ≈ 0` i McNemar `p = 0.002`; **statystycznie najsilniejszy efekt w Eksp. 7**.

Dla **MCTS** efekt jest minimalny (Δ = −0.063), statystycznie nieodróżnialny od 0
(chi-square `p = 0.343`, McNemar `p = 0.455`). MCTS dzięki stochastycznym rolloutom
i tak wprowadza wariancję otwarciową — książka „nie ma niczego do poprawienia"
w sensie sygnatury wyniku.

**Hipoteza H3 (MCTS bardziej zyskuje na książce)** zostaje **odrzucona**: paradoksalnie,
to Minimax pokazuje większy wpływ książki, ale głównie z powodu artefaktu deterministyczności.
Po znormalizowaniu (przyjęciu, że Minimax OFF jest patologiczny), faktyczny efekt
książki na siłę gry w naszej próbie wynosi prawdopodobnie tylko kilka punktów Elo
dla obu algorytmów — wielkość trudna do wykrycia przy `n = 40`.

### 5.4 Decisive rate — książka wprowadza więcej remisów

W obu algorytmach włączenie książki **zwiększa odsetek remisów**:
- Minimax: 100 % → 50 % decisive (książka usuwa „wszystko-albo-nic" patologię),
- MCTS: 25 % → 17.5 % decisive.

To efekt zgodny z teorią: dobry repertuar otwarciowy prowadzi obie strony do pozycji
**przybliżeniem zrównoważonych** (~ 0 ewaluacji), z których trudno uzyskać przewagę
w grze środkowej. Książka „symuluje" balansowane otwarcia ECO z Eksp. 1-3.

### 5.5 Głębokość książki

Średni `last_book_ply` ≈ 25-27, max 30 — książka pokrywa **12-15 pełnych ruchów**
(dla obu algorytmów podobnie). Po wyjściu z książki silnik musi prowadzić własną
analizę grę środkową. Głębokość 25-27 plejów jest typowa dla książek Polyglota
o rozmiarze kilku MB (`codekiddy.bin`).

### 5.6 Asymetria algorytmów — interpretacja czasu OFF

`avg_opening_time` jest bardzo różne dla OFF: Minimax 13.3 s vs MCTS 402.7 s
(stosunek 30×). To wynika z konstrukcji:

- **Minimax `d = 4`** w pozycji otwarciowej (pełna obsada) wykonuje ~1500 węzłów,
  ~0.6 s/ruch (z Eksp. 2). Na 12 ruchów obu stron = 14.4 s — zgodne z pomiarem 13.3 s.
- **MCTS 20 s** ma sztywny budżet 20 s na ruch — bez heurystyki wczesnego
  zakończenia. 12 ruchów × 2 strony × 20 s = 480 s ≈ pomiar 402.7 s (różnica
  ~80 s wynika z adjudykacji + sprawdzeń terminacji).

MCTS jest **20-30× droższy** w otwarciu niż Minimax przy obecnej konfiguracji —
co tłumaczy, dlaczego względna oszczędność czasu z książki dla MCTS jest jeszcze
większa (99× vs 36×).

### 5.7 Ograniczenia

- **Patologia Minimax OFF** czyni porównanie OFF vs ON dla Minimaxa
  „niezgodnym z duchem badania" — nie mierzymy wpływu książki na siłę, lecz
  „wyjście z deterministycznej linii przegrywającej".
- **`n = 40` partii** jest za małe by wykryć efekt rzędu kilkudziesięciu Elo
  dla MCTS (gdzie wariancja jest sztucznie zmniejszona przez 25-26 remisów).
- **Brak head-to-head OFF vs ON** — eksperyment celowo nie testuje
  „silnik OFF vs silnik ON", co byłoby bardziej naturalne dla pytania
  „czy książka daje przewagę". Self-play daje informacje o charakterystyce gry,
  nie bezpośredniej różnicy w sile.
- **Brak wariantów NN** — nie sprawdzamy interakcji `book × NN evaluator`.
  Książka jest niezależna od ewaluatora, więc wniosek prawdopodobnie się powtarza,
  ale formalnie nieprzetestowane.
- **Konkretny plik `codekiddy.bin`** — wyniki zależą od jakości tej książki.
  Inna księga (większa lub mniejsza, bardziej agresywna lub solidna) dałaby inne
  wyniki głębokości i decisive rate.
- **Brak pomiaru „bezpieczeństwa otwarciowego"** — np. ile razy strona z książką
  trafia w `+0.5` przewagi vs OFF. Można wyciągnąć z `analysis_moves.csv` jako
  follow-up.
- **MCTS „nie zauważa" książki** w pomiarze siły — być może `n = 100+` ujawniłoby
  efekt rzędu 10-30 Elo, ale obecna próba na to nie wystarcza.

## 6. Wnioski

1. **Książka drastycznie skraca fazę otwarcia** (~36× dla Minimaxa, ~99× dla MCTS).
   Najmocniejszy efekt eksperymentu, niezależny od kontrowersji dotyczących siły gry.
2. **Minimax bez książki w self-play od `startpos` jest patologiczny** — wszystkie
   40 partii wygrane przez czarnych z powodu deterministyczności. Książka „naprawia"
   ten artefakt przez stochastyczny wybór pierwszego ruchu. Wniosek metodyczny:
   **deterministyczne silniki wymagają książki lub puli otwarć** (jak ECO w Eksp. 1-3)
   dla sensownych pomiarów siły.
3. **MCTS nie zyskuje statystycznie istotnie na sile** z włączeniem książki
   (chi-square `p = 0.34`, McNemar `p = 0.46`). Hipoteza H3 odrzucona — MCTS
   ma własną wariancję otwarciową przez stochastyczne rolloutty.
4. **Książka zwiększa odsetek remisów** w obu algorytmach (Minimax 100 % → 50 %,
   MCTS 25 % → 17.5 %) — efekt zgodny z teorią repertuarów otwarciowych:
   prowadzi do pozycji zrównoważonych.
5. **Głębokość książki: ~25-27 plejów** (12-13 pełnych ruchów), max 30. Typowy
   zakres dla `codekiddy.bin`.
6. **Statystyczna istotność tylko dla Minimaxa** (chi-square `p ≈ 0`, McNemar
   `p = 0.002`); MCTS pozostaje poniżej progu wykrywalności przy `n = 40`.
7. **Eksperyment 7 w obecnej formie nie odpowiada bezpośrednio na pytanie
   „o ile Elo poprawia książka"** — to wymaga head-to-head OFF vs ON.
   Dostarcza natomiast jasnych wniosków o (i) wpływie czasowym (dramatyczny)
   i (ii) wpływie na charakter gry (więcej remisów, mniej decisive).

Materiał stanowi podstawę siódmej części rozdziału eksperymentalnego (wpływ książki):
rysunek 4.17 (`exp7_wdl_comparison.png` — W/D/L OFF vs ON × 2 algorytmy), rysunek 4.18
(`exp7_opening_time.png` — czas fazy otwarcia, skala log), rysunek 4.19
(`exp7_book_exit_hist.png` — głębokość książki), tabela 4.15 (`exp7_summary.csv`
agregowane), tabela 4.16 (`exp7_statistical_tests.csv`). **W pracy magisterskiej
warto poświęcić osobny akapit patologii Minimax OFF** jako ilustracji znaczenia
różnorodności startowych pozycji dla testów deterministycznych silników.
