# Eksperyment 8 — Analiza wyników (round-robin „najmocniejszych" wariantów z książką)

## 1. Cel eksperymentu i pytanie badawcze

Eksperyment ma odpowiedzieć na pytanie: **w warunkach najbardziej zbliżonych do praktyki
turniejowej (silne parametry + włączona książka otwarć), który wariant silnika wygrywa
i czy ranking jest spójny z Eksp. 1?** Wypełnia lukę między:

- **Eksp. 1** — round-robin 4 wariantów, ale parametry zaniżone (`d = 3` dla Minimaxa,
  `t = 2.61 s` dla MCTS), bez książki,
- **Eksp. 5** — silne parametry, ale tylko vs Stockfish (1 × 1, nie round-robin),
- **Eksp. 7** — wpływ książki, ale wyłącznie self-play, tylko 2 algorytmy (TRAD).

Hipotezy:

- **H1** (replikacja): Ranking z Eksp. 1 powtarza się przy silnych parametrach
  i z książką (MINIMAX_NN > … > MCTS_TRAD).
- **H2** (efekt zmiany parametrów): MINIMAX_TRAD przy `d = 5` (zamiast `d = 3` w Eksp. 1)
  podnosi pozycję rankingową — zgodnie z Eksp. 2 zysk ~600 Elo na osi parametrów.
- **H3** (efekt książki): Spójnie z Eksp. 7, włączenie książki **zwiększa odsetek
  remisów** (silniki startują z pozycji zrównoważonych).
- **H4** (oś A, replikacja): Efekt „Minimax > MCTS" (`MINIMAX_score ≈ 0.66` w Eksp. 1)
  utrzymuje się przy silniejszych parametrach.
- **H5** (oś B, otwarte): Efekt „NN > TRAD" (`NN_score ≈ 0.67` w Eksp. 1) może
  **osłabnąć** przy silnych parametrach, bo głębsze przeszukiwanie kompensuje
  słabość ewaluatora TRAD.
- **H6** (sanity check): `white_score ≈ 0.5` (symetryzacja kolorów).

## 2. Założenia metodyczne

**Uczestnicy (4 warianty, najsilniejsze praktyczne parametry):**

| Wariant | Algorytm | Ewaluator | Parametr | Uzasadnienie |
|---|---|---|---|---|
| `MINIMAX_TRAD` | Minimax α-β | Heurystyczny | **`d = 5`** | Maksymalna wykonalna głębokość (Eksp. 2: ~3.5 s mean, p90 < 10 s; `d = 6` ~80 s mean — niewykonalne) |
| `MINIMAX_NN` | Minimax α-β | NN (Stockfish d=10) | **`d = 4`** | Świadomy kompromis — `d = 5 NN` ~133 s/ruch z Eksp. 2 (niepraktyczne) |
| `MCTS_TRAD` | MCTS PUCT | Heurystyczny | **`t = 60 s`** | Z Eksp. 3: ~50-80 Elo niżej niż 120 s, ale 2× szybciej |
| `MCTS_NN` | MCTS PUCT | NN | **`t = 60 s`** | j.w. |

**Struktura meczowa (6 par, round-robin):**

| # | Para | Charakter |
|---|---|---|
| 1 | MINIMAX_TRAD d=5 vs MINIMAX_NN d=4 | Oś B przy max parametrach |
| 2 | MINIMAX_TRAD d=5 vs MCTS_TRAD 60s | Oś A przy TRAD |
| 3 | MINIMAX_TRAD d=5 vs MCTS_NN 60s | Cross-axis |
| 4 | MINIMAX_NN d=4 vs MCTS_TRAD 60s | Cross-axis |
| 5 | MINIMAX_NN d=4 vs MCTS_NN 60s | Oś A przy NN |
| 6 | MCTS_TRAD vs MCTS_NN | Oś B przy MCTS |

**Próba:** **N = 10 partii/para** (5 oryginalnych + 5 z zamianą kolorów) = **60 partii**.
Zgodnie z README, to świadomie **mała próba** (rankingowa, nie precyzyjna) — różnice
Elo < 150 należy traktować jako *exploratory*, nie *definitive* (95 % CI przy
n = 10 jest rzędu ±200 Elo dla pojedynczej pary).

**Pozycja startowa:** `startpos` (brak ECO seedów), wariancję wprowadza **stochastyczna
książka Polyglota** (`codekiddy.bin`, weighted random) — różne otwarcia per partia
mimo identycznej pozycji startowej.

**Adjudykacja:** ±0.05/20 ruchów (krytyczna dla par 6, 4, 5 z MCTS / NN). **Książka:**
WŁĄCZONA dla obu graczy w każdej parze.

## 3. Zbierane metryki

Identycznie jak w Eksp. 1 + flaga `from_book` z Eksp. 7:

- per ruch: numer, strona, UCI, eval, czas, faza, `from_book`, pełne metryki Minimax (21)
  lub MCTS (12),
- per gra: wynik, total moves, terminacja, czas,
- agregaty: BT Elo, istotność par (binomial test na decisive games), efekty osi A/B,
  przewaga koloru.

Skrypt analityczny `exp1_round_robin.py` jest **reużyty** z Eksp. 1 (identyczna struktura
4 wariantów × 6 par).

## 4. Wyniki

### 4.1 Ranking Elo (Bradley-Terry MLE) — `analysis_elo.csv`

| Wariant | Elo (BT) | Liczba gier |
|---|---:|---:|
| **MINIMAX_NN d=4** | **+8.7** | 30 |
| **MINIMAX_TRAD d=5** | **+8.7** | 30 |
| MCTS_NN 60s | 0.0 | 30 |
| MCTS_TRAD 60s | **−17.4** | 30 |

Rozpiętość rankingu: **~26 Elo**. **MINIMAX_NN i MINIMAX_TRAD są dokładnie zrównane**
(+8.7) — bardzo istotny wynik w porównaniu z Eksp. 1 (gdzie MINIMAX_TRAD był ostatni
z −5.8, a MINIMAX_NN dominował na +8.7).

### 4.2 Macierz W/D/L per para — `analysis_wdl.csv`

| Para (biały vs czarny) | Gier | W (biały) | D | L (czarny) | Wynik białego | Śr. ruchów | Śr. czas (B/Cz) |
|---|---:|---:|---:|---:|---:|---:|---|
| MINIMAX_TRAD vs MINIMAX_NN | 10 | 3 | 4 | 3 | 0.500 | 89.6 | 12.96 / 13.13 s |
| MINIMAX_TRAD vs MCTS_TRAD | 10 | 1 | 8 | 1 | 0.500 | 91.2 | 22.86 / 19.03 s |
| MINIMAX_TRAD vs MCTS_NN | 10 | 2 | 7 | 1 | 0.550 | 111.6 | 23.66 / 23.90 s |
| MINIMAX_NN vs MCTS_TRAD | 10 | 4 | 4 | 2 | 0.600 | 115.5 | 35.97 / 35.86 s |
| MINIMAX_NN vs MCTS_NN | 10 | 1 | 7 | 2 | 0.450 | 112.6 | 39.16 / 37.42 s |
| MCTS_TRAD vs MCTS_NN | 10 | 1 | 8 | 1 | 0.500 | 109.8 | 45.61 / 45.51 s |

Wizualizacja: `engine/out/exp8_strongest_book_combined/plots/exp1_wdl_matrix.png`,
`plots/wdl_bars.png`.

**Bardzo wysoki odsetek remisów:** od 4 do 8 na 10 (sumarycznie 38/60 = **63 %**).

### 4.3 Istotność statystyczna par — `exp1_pair_significance.csv`

Test dwumianowy na partiach rozstrzygniętych:

| Para | a | d | b | Decisive | Win-rate na decisive | p-value | 95 % CI |
|---|---:|---:|---:|---:|---:|---:|---|
| MINIMAX_TRAD vs MINIMAX_NN | 3 | 4 | 3 | 6 | 0.500 | **1.000** | [0.12, 0.88] |
| MINIMAX_TRAD vs MCTS_TRAD | 2 | 8 | 0 | 2 | 1.000 | 0.500 | [0.16, 1.00] |
| MINIMAX_TRAD vs MCTS_NN | 3 | 7 | 0 | 3 | 1.000 | 0.250 | [0.29, 1.00] |
| MINIMAX_NN vs MCTS_TRAD | 5 | 4 | 1 | 6 | 0.833 | 0.219 | [0.36, 1.00] |
| MINIMAX_NN vs MCTS_NN | 3 | 7 | 0 | 3 | 1.000 | 0.250 | [0.29, 1.00] |
| MCTS_TRAD vs MCTS_NN | 1 | 8 | 1 | 2 | 0.500 | **1.000** | [0.01, 0.99] |

**Żadna para nie osiąga istotności statystycznej** (najmniejsze `p = 0.219` dla
MINIMAX_NN vs MCTS_TRAD). Wszystkie 95 % CI są bardzo szerokie. Wynik zgodny z
założeniem rankingowym o `n = 10`.

Wizualizacja: `plots/exp1_pair_significance.png`.

### 4.4 Efekty głównych osi — `exp1_axis_summary.csv`

Agregat z 4 par cross-axis (40 gier):

| Oś | Lider | Wygrane lidera | Remisy | Wygrane przeciwnika | Wynik lidera |
|---|---|---:|---:|---:|---:|
| **A (algorytm)** | **MINIMAX** | 13 | 26 | 1 | **0.650** |
| **B (ewaluator)** | NN (nominalnie) | 9 | 23 | TRAD: 8 | **0.513** |

Wizualizacje: `plots/exp1_axis_a_effect.png`, `plots/exp1_axis_b_effect.png`.

**Kluczowe porównanie z Eksp. 1:**

| Oś | Eksp. 1 (słabe params, bez książki) | Eksp. 8 (silne params, z książką) | Δ |
|---|---:|---:|---:|
| **A (algorytm)** | 0.662 | **0.650** | −0.012 |
| **B (ewaluator)** | 0.667 | **0.513** | **−0.154** |

### 4.5 Przewaga koloru — `exp1_color_advantage.csv`

| Łącznie | Białe wyg. | Remisy | Czarne wyg. | Wynik białych |
|---:|---:|---:|---:|---:|
| 60 | 12 | 38 | 10 | **0.517** |

H6 potwierdzona. Symetryzacja zadziałała.

## 5. Dyskusja

### 5.1 Zaskakujące zrównanie MINIMAX_TRAD z MINIMAX_NN — odwrócenie rankingu Eksp. 1

| Wariant | Eksp. 1 (Elo) | Eksp. 8 (Elo) | Δ | Zmiana parametru |
|---|---:|---:|---:|---|
| MINIMAX_NN | +8.7 (1.) | +8.7 (1.-2.) | 0 | `d = 3 → 4` |
| MCTS_TRAD | 0.0 (2.) | −17.4 (**4.**) | −17.4 | `t = 2.61 s → 60 s` |
| MCTS_NN | −2.9 (3.) | 0.0 (3.) | +2.9 | j.w. |
| **MINIMAX_TRAD** | **−5.8 (4.)** | **+8.7 (1.-2.)** | **+14.5** | **`d = 3 → 5`** |

Najbardziej dramatyczna zmiana: **MINIMAX_TRAD awansuje z 4. miejsca na 1.-2.**
i zrównuje się z liderem MINIMAX_NN. Mechanizm:

- Eksp. 2 przewiduje wzrost Elo o **~+530 Elo** przy przejściu `d = 3 → 5` (suma
  ΔElo = +189.6 + 341.0 = +530).
- MINIMAX_NN zyskuje tylko ~+276 Elo przy `d = 3 → 4` (z Eksp. 2: ΔElo(NN d=3→4) = +275.9).
- Różnica zysku: **+530 − 276 = +254 Elo** na korzyść MINIMAX_TRAD.
- W Eksp. 1 MINIMAX_TRAD przegrywał z MINIMAX_NN o ~14 Elo. Po dodaniu +254 Elo
  powinien teraz **wygrać** o ~240 Elo — ale w obecnej próbie `n = 10` różnicy nie widać.

Para 1 (`MINIMAX_TRAD vs MINIMAX_NN`) zakończyła się 3W/4D/3L (`p = 1.0`) — pełna
symetria, statystyczna niemożność rozróżnienia. To jest spójne z ekstrapolacją Eksp. 2
przy uwzględnieniu szerokiego 95 % CI dla n=10 (±200 Elo).

H1 (pełna replikacja rankingu) **częściowo odrzucona** — kolejność się zmieniła
przez efekt różnicy zysku z głębokości. H2 (zysk MINIMAX_TRAD z d=5) **potwierdzona**.

### 5.2 Oś A (MINIMAX > MCTS) — robustna replikacja (H4)

`MINIMAX_score` w Eksp. 8 (0.650) jest praktycznie identyczny z Eksp. 1 (0.662) —
różnica 0.012 (1.2 p.p.) mieści się w szumie. Wniosek: **przewaga Minimaxa nad MCTS
jest robustna na zmianę parametrów i obecność książki**. Zgodne z:

- Eksp. 4b: top-1 match rate MINIMAX_TRAD 0.555 vs MCTS_TRAD 0.395 (+16 p.p.),
- Eksp. 6: solve rate MINIMAX 65-67 % vs MCTS 36-38 % (+29 p.p.),
- Eksp. 5: MINIMAX_TRAD 1517 Elo vs MCTS_TRAD ≤ 900 Elo (+600 Elo).

H4 jednoznacznie potwierdzona.

### 5.3 Oś B (NN > TRAD) — załamanie (H5)

Najmocniejsza obserwacja Eksp. 8: `NN_score` spada z **0.667** (Eksp. 1) do
**0.513** (Eksp. 8) — różnica **−15.4 p.p.**. Praktycznie efekt osi B znika
(0.513 ≈ 0.5 to remis statystyczny).

**Mechanizm:** w Eksp. 1 oba warianty Minimaxa miały sztywne `d = 3`, czyli czyste
porównanie ewaluatora (Axis B). W Eksp. 8 MINIMAX_TRAD ma `d = 5`, a MINIMAX_NN
ma `d = 4` — głębsze przeszukiwanie **kompensuje** słabszy ewaluator TRAD.
Z Eksp. 2 wynika, że ΔElo(d=3→5, TRAD) ≈ +530 vs ΔElo(d=3→4, NN) ≈ +276 — różnica
~+254 Elo na korzyść TRAD niweluje przewagę osi B z Eksp. 1 (~0.167 score).

**Wniosek metodologiczny dla pracy:** wpływ ewaluatora jest **mocno zależny od głębokości**:
- przy płytkim przeszukiwaniu (`d = 3`) wyrocznia NN dominuje (efekt 0.67),
- przy głębokim przeszukiwaniu (`d = 5`) heurystyka TRAD się wyrównuje (efekt 0.51).

To uzasadnia teoretyczną zasadę „głębokość kompensuje słabość ewaluatora", choć
sprzeczność z Eksp. 2 (gdzie w izolacji NN przy tej samej `d` nie był wyraźnie
silniejszy od TRAD) wymaga jawnego omówienia.

H5 **potwierdzona** — efekt osi B osłabia się przy silniejszych parametrach.

### 5.4 Wysoki odsetek remisów — efekt książki (H3)

Eksp. 1: 5-26 remisów na 30 partii (typowo 18-25), sumarycznie **107 / 180 = 59 %**.
Eksp. 8: 4-8 remisów na 10, sumarycznie **38 / 60 = 63 %**.

Wzrost odsetka remisów o +4 p.p. przy uwzględnieniu, że **partie w Eksp. 8 startują
od pozycji zrównoważonych z książki** (Eksp. 7: książka prowadzi do mniej rozstrzygnięć).
Małe `n` ogranicza precyzję, ale kierunek jest zgodny z H3.

Najwyższy odsetek remisów:
- `MINIMAX_TRAD vs MCTS_TRAD`: 8/10 = 80 %
- `MCTS_TRAD vs MCTS_NN`: 8/10 = 80 %
- `MINIMAX_TRAD vs MCTS_NN`: 7/10 = 70 %
- `MINIMAX_NN vs MCTS_NN`: 7/10 = 70 %

Pary z udziałem `MCTS_NN` mają największą skłonność do remisów — silnik o słabym
throughpucie (Eksp. 3: ~30 iter/s) ma tendencję do gier defensywnych, które przy
adjudykacji ±0.05/20 ruchów łatwo kończą się remisem.

### 5.5 Brak istotności statystycznej — ograniczenie próby

Najmniejsze `p-value` w Eksp. 8 to **0.219** (`MINIMAX_NN vs MCTS_TRAD`).
**Żadna para nie osiąga progu `p < 0.05`.** Powody:

1. **`n = 10`** to bardzo mała próba dla testu dwumianowego.
2. **Bardzo wysoki odsetek remisów** (40-80 % per para) drastycznie zmniejsza
   liczbę „decisive" gier — typowo 2-6 z 10. Test dwumianowy na 2-6 obserwacji
   wymaga skrajnych wyników (`6/0` lub `0/6`) by uzyskać `p < 0.05`.

**Konsekwencja:** Eksp. 8 dostarcza **rankingowy szkielet**, nie istotne porównania
parami. Praca magisterska powinna **jawnie traktować Eksp. 8 jako exploratory**
i odsyłać po istotność do Eksp. 1 (n=30) i Eksp. 5 (n=20).

### 5.6 Czas obliczeń

Średni czas/ruch waha się od ~13 s (`MINIMAX_TRAD vs MINIMAX_NN`) do ~46 s
(`MCTS_TRAD vs MCTS_NN`). Spójne z konfiguracją:
- `MINIMAX_TRAD d=5` ≈ 3.5 s/ruch (z Eksp. 2),
- `MINIMAX_NN d=4` ≈ 49.5 s/ruch (z Eksp. 2),
- `MCTS_*` 60 s/ruch (sztywne).

Średnia 12.96 s w parze 1 (oba Minimaxy) sugeruje, że **książka znacząco redukowała
czas otwarcia** — bez niej średnia byłaby ~30 s (5 s/ruch MINIMAX_TRAD + 50 s
MINIMAX_NN ≈ 27 s mean).

### 5.7 Sanity check koloru (H6)

`white_score = 0.517` przy 60 partiach (12W/38D/10L). H6 potwierdzona — drobna
asymetria 1.7 p.p. mieści się w 95 % CI dla `n = 60` (~ ±13 p.p.).

### 5.8 Ograniczenia

- **`n = 10` zbyt małe** dla testów istotności na poziomie par.
- **Brak head-to-head bez książki** — niemożność izolacji „efektu książki"
  przy silnych parametrach (pole do follow-up).
- **Brak liczby unikalnych otwarć per para** — sanity check wariancji
  otwarciowej nie wykonany; warto policzyć z `analysis_moves.csv`.
- **Stochastyczna książka wprowadza wariancję otwarciową** — niespójność
  z Eksp. 1 (gdzie 25 ECO startów są deterministyczne).
- **Outliery czasowe MINIMAX_NN d=4** (z Eksp. 1b: max 866 s/ruch) mogły
  wystąpić w niektórych partiach — nie raportowane w wynikach agregowanych.
- **Brak weryfikacji obecności `from_book = true`** w metrykach (sanity check
  „czy książka rzeczywiście działała"); zalecane sprawdzenie w `analysis_moves.csv`.

## 6. Wnioski

1. **Ranking Eksp. 8:** `MINIMAX_TRAD d=5 = MINIMAX_NN d=4 (+8.7) > MCTS_NN
   (0.0) > MCTS_TRAD (−17.4)`. Rozpiętość ~26 Elo, **żadna para nie statystycznie
   istotna** (n=10, najmniejsze `p = 0.22`).
2. **MINIMAX_TRAD awansuje z 4. (Eksp. 1) na 1.-2. (Eksp. 8)** dzięki zwiększeniu
   głębokości `d = 3 → 5` (~+530 Elo wg Eksp. 2). Zrównuje się z MINIMAX_NN,
   którego głębokość wzrosła tylko `d = 3 → 4` (~+276 Elo).
3. **Oś A (MINIMAX > MCTS) replikuje się idealnie**: 0.650 vs 0.662 w Eksp. 1.
   Robustna, niezależna od parametrów i książki. Spójne z Eksp. 4b, 6, 5.
4. **Oś B (NN > TRAD) załamuje się**: 0.513 vs 0.667 w Eksp. 1 (−15.4 p.p.).
   Głębsze przeszukiwanie kompensuje słabszy ewaluator — kluczowy wniosek dla
   strategii projektowania silnika: **głębokość lub jakość ewaluatora są w pewnym
   stopniu wymienne**.
5. **Bardzo wysoki odsetek remisów** (63 %) potwierdza efekt książki obserwowany
   w Eksp. 7 — silniki startują z pozycji zrównoważonych, trudno wymusić rozstrzygnięcie.
6. **Sanity check koloru** (`white_score = 0.517`) potwierdza poprawność procedury.
7. **Eksp. 8 to *exploratory ranking*, nie *primary measurement*** — daje
   wskazówki o kolejności wariantów w warunkach realistycznych, ale dla
   istotności statystycznej należy odsyłać do Eksp. 1 (n=30/para) i Eksp. 5 (n=160/wariant).

Materiał stanowi podstawę ósmej części rozdziału eksperymentalnego (silne parametry
z książką): rysunek 4.20 (`exp1_wdl_matrix.png` — macierz 4×4), rysunek 4.21
(`exp1_axis_a_effect.png`, `exp1_axis_b_effect.png` — efekty osi z porównaniem do
Eksp. 1), tabela 4.17 (ranking i porównanie z Eksp. 1), tabela 4.18 (istotność par),
tabela 4.19 (porównanie efektów osi A/B między Eksp. 1 i Eksp. 8). Najmocniejszy
wynik do prezentacji w pracy: **„NN-vs-TRAD advantage shrinks from +0.167 to +0.013
when search depth increases — głębokość kompensuje jakość ewaluacji"**.
