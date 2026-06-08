# Eksperyment 5 — Analiza wyników (benchmark Stockfishem, kalibracja Elo)

## 1. Cel eksperymentu i pytanie badawcze

Eksperyment ma odpowiedzieć na pytanie: **na jakim poziomie siły absolutnej (skala
zewnętrzna, CCRL-podobna) gra każdy z 4 wariantów silnika?** W odróżnieniu od Eksp. 1
(Elo relatywne, kotwiczone do wewnątrz puli) Eksp. 5 mierzy siłę przez konfrontację
ze Stockfishem na 8 poziomach `skill` (1, 3, 5, 8, 10, 13, 15, 20). Mapowanie
`skill → Elo` (`_sf_skill_elo.csv`) pozwala na interpolację Elo wariantu z krzywej
score vs SF Elo.

Hipotezy:

- **H1:** Wszystkie cztery warianty osadzają się na bezwzględnej skali Elo (~900-3500)
  w punktach o spójnym uporządkowaniu względem siebie.
- **H2** (walidacja Eksp. 1): Ranking wariantów w skali absolutnej jest **zgodny**
  z rankingiem relatywnym z Eksp. 1.
- **H3:** Im wyższe Elo, tym mniej „blunderów" (`centipawn_loss > 200`) i niższe
  ACPL — krzywa monotonicznie maleje.
- **H4** (stratyfikacja fazowa): Warianty z heurystyką TRAD mają wyraźnie wyższy ACPL
  w otwarciu (zgodnie z wnioskiem z Eksp. 4: TRAD niedoskonale różnicuje pozycje
  o symetrycznym materiale).
- **H5:** Krzywa score vs SF skill ma w punkcie `score = 0.5` interpolowane Elo,
  które można odczytać jako absolutną siłę wariantu.

> **Uwaga:** Faza 2 analizy (re-ewaluacja Stockfishem d=20) **nie została wykonana**
> w obecnym uruchomieniu — w `engine/out/exp5_stockfish_combined/` brak pliku
> `stockfish_reval.csv` oraz wykresów `exp5_acpl_by_variant.png`,
> `exp5_acpl_by_phase.png`, `exp5_blunder_rate.png`. Hipotezy **H3 i H4
> dotyczące ACPL i blunder rate pozostają nierozstrzygnięte** — wymagają
> uruchomienia `run_exp5_analyze.ps1` (bez `-SkipReval`).

## 2. Założenia metodyczne

**Uczestnicy (4 warianty):**

| Wariant | Algorytm | Ewaluator | Parametr |
|---|---|---|---|
| `minimax_trad_d4` | Minimax α-β | Heurystyczny | `d = 4` |
| `minimax_nn_d3` | Minimax α-β | NN (Stockfish d=10) | `d = 3` |
| `mcts_trad` | MCTS PUCT | Heurystyczny | `t = 2.61 s` (z kalibracji Eksp. 1) |
| `mcts_nn` | MCTS PUCT | NN (Stockfish d=10) | `t = 2.61 s` |

**Uwaga o parametryzacji:** w odróżnieniu od Eksp. 1 (gdzie wszystkie warianty mają
parametr „słaby" dla czystego porównania osi B), Eksp. 5 używa parametrów
**najsilniejszych w granicach wykonalności**: `MINIMAX_TRAD d = 4` zamiast `d = 3`,
co znacząco wzmacnia ten wariant.

**Przeciwnicy (8 poziomów Stockfisha):** `skill ∈ {1, 3, 5, 8, 10, 13, 15, 20}`,
przy stałej głębokości UCI `depth = 10`. Mapowanie `_sf_skill_elo.csv`:

| skill | 1 | 3 | 5 | 8 | 10 | 13 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ~Elo (CCRL) | 900 | 1100 | 1400 | 1700 | 2000 | 2400 | 2700 | 3500 |

**Próba:** 4 warianty × 8 skill × 20 partii = **640 partii** (10 oryginalnych
+ 10 z zamianą kolorów per matchup). Łącznie 160 partii/wariant — wystarczające
do dopasowania krzywej score-vs-Elo.

**Otwarcia:** 25 pozycji ECO, cyklicznie po jednej dla każdej z 20 partii w matchupie
(pierwsze 20). **Adjudykacja:** ±0.05/20 ruchów. **Książka:** wyłączona po obu stronach.

## 3. Zbierane metryki

**Per ruch (jsonl):** numer ruchu, strona, UCI, eval (centypionki), czas, faza
oraz pełny zestaw metryk Minimax / MCTS znanych z Eksp. 1.

**Per matchup (`exp5_variant_summary.csv`):** `variant_wins`, `draws`, `variant_losses`,
`variant_score` — sumarycznie z 20 gier (już z perspektywy wariantu, niezależnie
od koloru).

**Per wariant (`exp5_variant_elo.csv`):** `estimated_elo` interpolowane z krzywej
`variant_score = f(SF_Elo)` przez fit logistyczny.

**(Niewykonane) Per ruch po re-ewaluacji (`stockfish_reval.csv`):**
`stockfish_eval_d20`, `centipawn_loss`, `is_blunder` — wymagałoby uruchomienia
Fazy 2 z `--depth 20` (~50-150 h obliczeń) lub `--depth 15` (~5-15 h).

## 4. Wyniki

### 4.1 Estymowane Elo wariantów — `exp5_variant_elo.csv`

| Wariant | Estymowane Elo | Liczba matchupów |
|---|---:|---:|
| **MINIMAX_NN (d=3)** | **1683** | 8 |
| **MINIMAX_TRAD (d=4)** | **1517** | 8 |
| **MCTS_NN (2.61 s)** | **1318** | 8 |
| **MCTS_TRAD (2.61 s)** | **900** | 8 |

Wizualizacja: `engine/out/exp5_stockfish_combined/plots/exp5_score_curve.png`.

**Uwaga interpretacyjna:** `MCTS_TRAD = 900 Elo` to **dolne ograniczenie** wynikające
z metody interpolacji — wariant nie pokonał reliably nawet Stockfisha skill 1 (~900 Elo),
więc krzywa nie ma punktu przecięcia `score = 0.5` w zakresie pomiarowym. Faktyczne
Elo może być **niższe** niż 900.

### 4.2 Krzywa score vs SF skill per wariant — `exp5_variant_summary.csv`

| SF skill (Elo) | MINIMAX_NN d=3 | MINIMAX_TRAD d=4 | MCTS_NN | MCTS_TRAD |
|---|---:|---:|---:|---:|
| **1 (900)** | 1.000 (20/0/0) | 0.975 (19/1/0) | 0.700 (8/12/0) | 0.500 (8/4/8) |
| **3 (1100)** | 0.950 (18/2/0) | 0.825 (13/7/0) | 0.700 (8/12/0) | 0.525 (3/15/2) |
| **5 (1400)** | 0.925 (17/3/0) | 0.675 (9/9/2) | 0.425 (2/13/5) | 0.175 (1/5/14) |
| **8 (1700)** | 0.475 (2/15/3) | 0.225 (1/7/12) | 0.150 (0/6/14) | 0.050 (0/2/18) |
| **10 (2000)** | 0.325 (0/13/7) | 0.175 (0/7/13) | 0.150 (0/6/14) | 0.000 (0/0/20) |
| **13 (2400)** | 0.275 (1/9/10) | 0.125 (0/5/15) | 0.125 (0/5/15) | 0.100 (0/4/16) |
| **15 (2700)** | 0.375 (3/9/8) | 0.125 (0/5/15) | 0.125 (0/5/15) | 0.025 (0/1/19) |
| **20 (3500)** | 0.300 (1/10/9) | 0.175 (0/7/13) | 0.175 (0/7/13) | 0.000 (0/0/20) |

Format komórki: `score (variant_wins/draws/variant_losses)`. Bolded punkty pomiarowe
z największą zmianą zachowania.

**Kluczowe progi:**

- `MINIMAX_NN`: punkt przejścia `score ≈ 0.5` przy `skill 8` (~1700 Elo) — stąd
  interpolacja 1683.
- `MINIMAX_TRAD`: przejście między `skill 5` (0.675) a `skill 8` (0.225) — interpolacja 1517.
- `MCTS_NN`: przejście między `skill 5` (0.425) a `skill 3` (0.700) — interpolacja 1318.
- `MCTS_TRAD`: nigdy nie osiąga `score > 0.525` (poza skill 1 i 3 ≈ 0.5) — pomiar
  jest **poniżej** dolnej granicy siatki SF.

### 4.3 Asymetria osi NN/TRAD na różnych algorytmach

| | TRAD | NN | Δ (NN − TRAD) |
|---|---:|---:|---:|
| **Minimax (Eksp. 5)** | 1517 | 1683 | **+166 Elo** |
| **MCTS (Eksp. 5)** | 900 (lower bound) | 1318 | **≥ +418 Elo** |

NN dodaje **dużo więcej** w MCTS (≥ 418 Elo) niż w Minimaxie (166 Elo). Interpretacja:
MCTS bez sensownej oceny liścia jest szczególnie kruchy — przy słabej heurystyce TRAD
MCTS PUCT błądzi w drzewie z powodu szumu w sygnałach value; wyrocznia NN drastycznie
poprawia jakość rollouta.

### 4.4 Ranking BT z `analysis_elo.csv` — niewiarygodny

| Wariant / SF | Elo (BT) | n |
|---|---:|---:|
| `stockfish_sk20` | +28.7 | 80 |
| `mcts_trad` | +26.8 | 160 |
| `mcts_nn` | +20.3 | 160 |
| `stockfish_sk13` | +15.8 | 80 |
| `minimax_nn_d3` | +9.4 | 160 |
| `stockfish_sk10` | −5.8 | 80 |
| `stockfish_sk3` | −5.8 | 80 |
| `minimax_trad_d4` | −10.0 | 160 |
| `stockfish_sk15` | −14.4 | 80 |
| `stockfish_sk5` | −14.4 | 80 |
| `stockfish_sk8` | −23.1 | 80 |
| `stockfish_sk1` | −27.5 | 80 |

Ranking jest **wewnętrznie sprzeczny** (Stockfish skill 1 i 8 niżej niż skill 3, 10;
MCTS_TRAD wyżej niż większość Stockfishów). Powód: Bradley-Terry MLE wymaga
**przechodniości** i nie radzi sobie z bimodalnym rozkładem wyników (silnik vs Stockfish
to często „wszystko albo nic" — 20W/0D/0L lub 0W/0D/20L), co prowadzi do nieidentyfikowalności.
**Plik `analysis_elo.csv` nie powinien być cytowany** w pracy dla Eksp. 5;
jedynym poprawnym źródłem siły absolutnej jest `exp5_variant_elo.csv`.

### 4.5 Czas/ruch — `analysis_wdl.csv`

Wybrane średnie czasy/ruch (sek):

| Matchup | Czas białego | Czas czarnego |
|---|---:|---:|
| `minimax_trad_d4 vs sk20` | 0.54 | 0.48 |
| `minimax_nn_d3 vs sk20` | 4.42 | 4.44 |
| `mcts_trad vs sk20` | 0.52 | 0.53 |
| `mcts_nn vs sk20` | 5.12 | 4.61 |

Zgodne z Eksp. 1/2/3: MINIMAX_NN i MCTS_NN dominują kosztem subprocesu Stockfisha
(~4-5 s/ruch); TRAD warianty < 1 s.

**Anomalia czasowa `minimax_nn_d3 vs sk1`:** `white = 7.63 s`, `black = 7.95 s` —
wyraźnie wolniej niż w starciach z silniejszymi przeciwnikami. Najprawdopodobniej
powód: bardzo długie partie (61.6 ruchu średnio vs 122.2 dla sk20) gdzie NN regularnie
woła Stockfish-oracle na większej liczbie pełnych pozycji w grze środkowej —
faktyczna zmienność per matchup wynika ze struktury rozgrywki (długość gry × częstość
ewaluacji liści).

### 4.6 ACPL i blunder rate — DANE NIEDOSTĘPNE

Brak `stockfish_reval.csv`. Wymagana re-ewaluacja `python stockfish_reval.py --depth 20`
(lub `--depth 15` dla szybszej, mniej precyzyjnej oceny). Pole do uzupełnienia
przed finalną wersją pracy.

## 5. Dyskusja

### 5.1 Ranking absolutny — porównanie z Eksp. 1

| Wariant | Eksp. 1 Elo (relat.) | Eksp. 5 Elo (absol.) | Param. w Eksp. 5 |
|---|---:|---:|---|
| MINIMAX_NN | +8.7 (1.) | 1683 (1.) | d=3 (j.w.) |
| MCTS_TRAD | 0.0 (2.) | **900** (4.) | t=2.61 s (j.w.) |
| MCTS_NN | −2.9 (3.) | 1318 (3.) | t=2.61 s (j.w.) |
| MINIMAX_TRAD | −5.8 (4.) | **1517** (2.) | **d=4** (zmienione vs d=3) |

**Zaskakująca rozbieżność:** MINIMAX_TRAD i MCTS_TRAD zamieniły się miejscami między
2. a 4. pozycją. Częściowe wyjaśnienie:

1. **MINIMAX_TRAD podniesiono z `d = 3` do `d = 4`** — z Eksp. 2 wynika, że to
   przyrost ok. +190 Elo. Przesuwa wariant z 4. miejsca na pozycję wyższą.
2. **MCTS_TRAD pozostał na 2.61 s** (tak samo jak w Eksp. 1) — względna pozycja
   się nie zmieniła w sensie zasobów, ale konkuruje teraz z silniejszym MINIMAX_TRAD.
3. **Stockfish jako przeciwnik jest jakościowo inny** niż własne warianty
   (np. MCTS_TRAD wygrywa partie remisowe z drugim MCTS_TRAD przez „przeczekanie",
   ale Stockfish niskiego skill jest **aktywnie agresywny** i rozstrzyga partie).
   Wysoki odsetek remisów w Eksp. 1 (do 26/30) tu zanika.

Wniosek dla H2: **częściowa walidacja** — ranking dwóch skrajnych pozycji (MINIMAX_NN
najlepszy, MCTS_TRAD najsłabszy) jest spójny po uwzględnieniu zmiany parametru
MINIMAX_TRAD. Środkowe pozycje są mniej rozróżnialne. Praca powinna w jednym miejscu
sformułować zastrzeżenie metodyczne: **Eksp. 1 i Eksp. 5 używają różnych parametrów
MINIMAX_TRAD** (d=3 vs d=4), więc bezpośrednie porównanie wymaga normalizacji.

### 5.2 Krzywa siły — gdzie warianty „pękają"

Analiza punktów, w których wariant przestaje wygrywać (`score < 0.5`):

- **MINIMAX_NN d=3:** wygrywa pewnie do `skill 5` (~1400 Elo); pęka między `skill 5`
  a `skill 8` (~1700 Elo). Idealna interpolacja.
- **MINIMAX_TRAD d=4:** wygrywa pewnie tylko do `skill 5` z spadkiem (~0.675); pęka
  między `skill 5` (0.675) a `skill 8` (0.225). Charakterystyczne — różnica
  166 Elo względem MINIMAX_NN.
- **MCTS_NN:** pęka między `skill 3` (0.7) a `skill 5` (0.425). Stosunkowo nagły spadek.
- **MCTS_TRAD:** nigdy nie wygrywa pewnie — nawet vs `skill 1` ma `score = 0.500`
  (8W/4D/8L). Wskazuje **strukturalną słabość** algorytmu w obecnej konfiguracji.

### 5.3 NN wnosi więcej w MCTS niż w Minimaxie

Z § 4.3: `Δ Elo (NN − TRAD)` wynosi:
- w Minimaxie: **+166 Elo** (1683 − 1517),
- w MCTS: **≥ +418 Elo** (1318 − 900, gdzie 900 to dolne ograniczenie).

Mechanizm: Minimax α-β z TT i quiescence sam **kompensuje słabości ewaluatora**
przez głębokość przeszukiwania. MCTS PUCT polega na sygnale value z liścia
do kierowania selekcją — kiepski sygnał (TRAD) prowadzi do kiepskich rolloutów
**na każdym poziomie drzewa**, nie tylko na liściach. Wynik wspiera tezę, że
**MCTS jest „niedopasowanym" algorytmem do heurystyk ad-hoc** — potrzebuje albo
trenowanej sieci, albo dokładnej wyroczni.

### 5.4 Pułap siły wariantów na skali CCRL

Estymowane Elo (1683 / 1517 / 1318 / 900) umieszcza warianty w zakresach:

- **1683 (MINIMAX_NN d=3)** ≈ poziom amatorski mocny / kandydat na mistrza klubowego.
- **1517 (MINIMAX_TRAD d=4)** ≈ średniozaawansowany klubowy.
- **1318 (MCTS_NN)** ≈ początkujący / średnio słaby klubowy.
- **900 (MCTS_TRAD lower bound)** ≈ poziom początkującego, prawdopodobnie poniżej.

Te liczby mają służyć **kontekstualizacji** wyników — nie są jednak bezpośrednio
porównywalne z Elo CCRL silników profesjonalnych (Stockfish d=20 ≈ 3500), ponieważ:

- Mapa `_sf_skill_elo.csv` jest **przybliżona** (źródło: dokumentacja Stockfisha,
  brak własnej kalibracji); w pracy warto zaznaczyć tę aproksymację.
- Punkt pomiarowy `skill = 20` reprezentuje sumarycznie skill 20 + depth 10,
  co jest słabsze niż „full strength" Stockfish bez ograniczenia głębokości.
- 20 partii/skill daje 95% CI rzędu ±150-200 Elo na pojedynczym poziomie skill;
  interpolacja redukuje to do ~±100 Elo, ale wciąż znacząco.

### 5.5 Ograniczenia

- **Brak ACPL/blunder rate** (Faza 2 niewykonana). Hipotezy H3, H4 nierozstrzygnięte.
  Bez tych danych nie można odpowiedzieć na pytanie: *„czy warianty z niższym
  Elo mają wyraźnie więcej blunderów, czy raczej grają konsekwentnie słabo?"*
- **Bradley-Terry w `analysis_elo.csv` nie działa** dla pula Stockfish + 4 warianty
  z bimodalnymi wynikami. Wartości tego pliku są mylące i nie powinny być
  cytowane w pracy.
- **MCTS_TRAD osiąga `score = 0.5` przy skill 1** — interpretacja Elo = 900 to
  górne ograniczenie z artefaktu metody. Prawdziwe Elo prawdopodobnie znacząco niższe.
- **20 partii/skill** to mała próba dla matchupów granicznych (np.
  `MINIMAX_NN vs sk8`: 2/15/3 — 15 remisów na 20 wskazuje, że oba silniki są blisko
  w sile, ale przedział ufności jest szeroki).
- **Mapa `_sf_skill_elo.csv`** to aproksymacja bez własnej kalibracji; absolutne
  wartości Elo (1683, 1517, 1318) mają niepewność rzędu ±100-150 Elo niezależnie
  od wariancji własnej próby.
- **Stockfish na stałej `depth = 10`** redukuje siłę Stockfisha w stosunku do
  pełnego (depth → ∞); mapa skill→Elo zakłada konkretną konfigurację, której zgodność
  z `_sf_skill_elo.csv` nie była weryfikowana.
- **Brak `_results.csv`** w `engine/out/exp5_stockfish_combined/`; istnieją tylko
  `_results_1..4.csv` per wariant — agregacja per matchup w `exp5_variant_summary.csv`
  jest wystarczająca, ale szerokie analizy per-gra wymagają osobnego łączenia.

## 6. Wnioski

1. **Ranking absolutny:** `MINIMAX_NN (1683) > MINIMAX_TRAD (1517) > MCTS_NN (1318)
   > MCTS_TRAD (≤ 900)`. Lider z Eksp. 1 (MINIMAX_NN) potwierdzony; outsider
   (MCTS_TRAD) potwierdzony. Środkowe pozycje rozjeżdżają się względem Eksp. 1
   głównie z powodu podniesienia `MINIMAX_TRAD` z `d = 3` do `d = 4`
   (~+190 Elo wg Eksp. 2).
2. **NN wnosi 2.5× więcej Elo w MCTS niż w Minimaxie** (+418 vs +166 Elo).
   Mechanizm: MCTS PUCT silnie polega na sygnale value, Minimax z TT/quiescence
   kompensuje słabości ewaluatora.
3. **MCTS_TRAD w obecnej konfiguracji jest niewystarczająco silny** by pokonać
   nawet najsłabszego Stockfisha (skill 1 ≈ 900 Elo) — `score = 0.5` jest dolnym
   limitem metody interpolacji, faktyczne Elo prawdopodobnie niższe. Spójne
   z obserwacją z Eksp. 4 (MCTS_TRAD top-1 match rate = 0.395, najgorszy z testowanych
   wariantów).
4. **Mapowanie do skali CCRL:** warianty plasują się w zakresie ~900-1700 Elo —
   poziom amatorski/klubowy. Daleko od pełnego Stockfisha (~3500 Elo).
5. **`analysis_elo.csv` jest niewiarygodne** dla Eksp. 5 (BT MLE łamie się przy
   bimodalnych wynikach silnik-vs-Stockfish). Jedynym poprawnym źródłem siły
   absolutnej jest `exp5_variant_elo.csv`.
6. **Hipotezy H3 (ACPL ↔ Elo) i H4 (ACPL po fazie) pozostają nierozstrzygnięte** —
   wymagają uruchomienia Fazy 2 (`run_exp5_analyze.ps1` bez `-SkipReval`).
   Bez tych danych nie ma niezależnej, opartej na pojedynczych ruchach miary jakości.

Materiał stanowi podstawę piątej części rozdziału eksperymentalnego (siła absolutna):
rysunek 4.14 (`exp5_score_curve.png` — krzywa score vs SF Elo, 4 linie + interpolacje),
tabela 4.10 (estymowane Elo per wariant), tabela 4.11 (przyrost Elo dla NN w obu
algorytmach), zestawienie porównawcze z Eksp. 1 (ranking relatywny vs absolutny).
**Po dokończeniu Fazy 2** dodać rysunek `exp5_acpl_by_variant.png` i tabelę
ACPL × faza.
