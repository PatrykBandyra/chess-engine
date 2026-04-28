# Analiza i plan napraw `BoardEvaluatorTrad`

## Cel dokumentu

Ten dokument rozwija wcześniejszą analizę `BoardEvaluatorTrad` i przekształca ją w konkretny plan naprawczy. Obejmuje:

- weryfikację problemów względem aktualnej implementacji `engine/board_evaluator_trad.py`,
- korekty błędnych lub zbyt uproszczonych wniosków z poprzedniej analizy,
- priorytety napraw,
- konkretne kroki implementacyjne,
- plan testów regresyjnych.

Kod silnika **nie powinien być modyfikowany na etapie samej analizy**. Ten plan jest przeznaczony do dalszego, kontrolowanego wdrożenia.

---

# 1. Executive summary

`BoardEvaluatorTrad` zawiera sensowny zestaw klasycznych heurystyk:

- materiał,
- PST / piece-square tables,
- struktura pionowa,
- bezpieczeństwo króla,
- para gońców,
- aktywność wież,
- mobilność i aktywność figur,
- proste zagrożenia / hanging pieces,
- aktywność króla w końcówce.

Najważniejsze problemy nie wynikają z braku heurystyk, tylko z ich **kontraktu, kalibracji i testowalności**.

Najważniejszy błąd do naprawy to **niespójna orientacja PST**. Obecne `__pst_value()` zakłada jeden format tablic, ale same tablice nie wyglądają na konsekwentnie zapisane w tej samej konwencji. W praktyce może to prowadzić do karania naturalnych ruchów rozwojowych, zwłaszcza pionów, i potencjalnie błędnego wartościowania innych figur.

Drugi istotny problem to to, że `PIECE_SQUARE_TABLE_END` jest kopią `PIECE_SQUARE_TABLE_MID` dla wszystkich figur poza królem. Interpolacja fazy gry jest więc realna głównie dla króla.

Pozostałe problemy — king safety, threats/hanging pieces, passed pawn helpers i open-file scan — są ważne, ale powinny być traktowane jako stabilizacja/tuning/refaktor, a nie jako pierwsze krytyczne poprawki.

Po późniejszym audycie aktualnej implementacji większość etapów P0/P1/P3 została wdrożona, a rekomendacje z Etapu 6 zostały zabezpieczone testami i lokalnymi poprawkami:

- `__get_game_phase()` nadal może zwracać pełną końcówkę (`phase == 0.0`) w pozycjach z hetmanami, np. `KQ vs KQ`, ale `king safety` ma teraz lokalny efektywny floor przy obecności hetmanów.
- `__evaluate_endgame_king_activity()` osłabia składnik centralizacji króla przy obecności hetmanów.
- `PIECE_SQUARE_TABLE_END` realnie różni się od `MID` dla pionów i króla, a kopie `MID` dla skoczków, gońców, wież i hetmana są świadomą decyzją zabezpieczoną testem dokumentacyjnym.
- Dodano pełny test symetrii całej oceny `evaluate_board(board) == -evaluate_board(board.mirror())` dla zestawu pozycji reprezentatywnych.
- Dodano testy dokumentujące ograniczenie statycznych/pinned atakujących w `king safety` i `threats`.

---

# 2. Status wdrożenia wcześniejszego planu heurystyk

Strukturalnie większość zaplanowanych usprawnień została wdrożona:

- `__get_game_phase()` istnieje i jest używany.
- `__pst_value()` interpoluje PST między middlegame/endgame.
- `__evaluate_pawn_structure()` obsługuje doubled/isolated/passed pawns.
- `__evaluate_king_safety()` uwzględnia pawn shield, open files i ataki w strefie króla.
- `__evaluate_minor_piece_features()` uwzględnia bishop pair.
- `__evaluate_rook_activity()` uwzględnia open/semi-open files, 7. linię i doubled rooks.
- `__evaluate_threats_and_hanging_pieces()` dodaje lekką statyczną ocenę zagrożeń.
- `__evaluate_endgame_king_activity()` nagradza aktywność króla w końcówce.
- `__evaluate_mobility_and_activity()` liczy mobilność bez mutowania `board.turn`.
- `evaluate_board()` agreguje wszystkie składniki.
- Istnieje plik testów regresyjnych `engine/test_board_evaluator_trad_regression.py`.

Jednak wdrożenie wymaga dopracowania merytorycznego, szczególnie w zakresie PST i stabilności wag.

---

# 3. Najważniejsze korekty względem poprzedniej analizy

Poprzednia analiza była w większości użyteczna, ale zawierała kilka błędów i uproszczeń.

## 3.1. Problem PST jest szerszy niż tylko piony

Poprzednia analiza słusznie wskazała problem z pionami, ale zbyt wąsko opisała zakres problemu.

Nie należy zakładać, że `KNIGHT`, `BISHOP` i `QUEEN` są pionowo symetryczne. W aktualnych tabelach nie są. To oznacza, że orientacja indeksowania ma znaczenie również dla nich.

Wniosek implementacyjny:

- nie naprawiać PST przez prosty wyjątek typu „dla pionów indeksuj odwrotnie”,
- wykonać pełny audyt wszystkich tablic PST,
- przyjąć jeden jawny kontrakt orientacji dla wszystkich tabel.

## 3.2. `board.attackers()` wykrywa obronę piona od tyłu

Poprzednia analiza twierdziła, że obrona wolnego piona od tyłu przez wieżę/hetmana nie jest wykrywana przez:

```python
board.attackers(color, square)
```

To jest niepoprawne. `python-chess` traktuje figurę liniową za pionem jako atakującą/chroniącą pole piona, jeśli linia jest otwarta i nie ma blokujących figur.

Wniosek implementacyjny:

- nie usuwać obecnego użycia `board.attackers(color, square)` w `evaluate_passed_pawn()` z powodu tej tezy,
- można natomiast rozbudować heurystykę wsparcia wolnego piona, bo obecnie daje taki sam mały bonus dla każdego wsparcia figurowego.

## 3.3. Open-file scan w king safety jest nieoptymalny, ale funkcjonalnie poprawny

Kod skanuje `chess.SQUARES` i wielokrotnie wywołuje `board.piece_at()`. To jest nieefektywne, ale nie widać tu błędu logicznego.

Wniosek implementacyjny:

- potraktować to jako refaktor/wydajność,
- nie priorytetyzować ponad PST i stabilizację king safety.

## 3.4. King safety i threats to częściowo tuning, nie same bugi

Wielokrotne liczenie atakującej figury w strefie króla jest realnym ryzykiem przeskalowania. Nie jest jednak automatycznie błędem, jeśli intencją było karanie liczby atakowanych pól. Problemem jest brak limitu, brak normalizacji i brak bardziej stabilnego modelu attack units.

Podobnie `threats/hanging pieces` są bardzo uproszczone. Można je poprawić, ale agresywna zmiana wag bez testów pozycyjnych może pogorszyć ewaluację.

---

# 4. Problem P0: niespójna orientacja PST

## 4.1. Aktualny kod

W `__pst_value()`:

```python
pst_index = chess.square_mirror(square) if color == chess.WHITE else square
pst_mid = self.PIECE_SQUARE_TABLE_MID[piece_type][pst_index]
pst_end = self.PIECE_SQUARE_TABLE_END[piece_type][pst_index]
return self.PST_SCALE * (phase * pst_mid + (1 - phase) * pst_end)
```

Docstring mówi, że tablice PST są zapisane w kolejności rank 8 -> rank 1, czyli top-down. Przy takim kontrakcie:

- biały powinien używać `chess.square_mirror(square)`,
- czarny powinien używać `square`.

Problem polega na tym, że same tablice nie wyglądają na konsekwentnie zapisane w tej konwencji.

## 4.2. Symptomy dla pionów

Dla aktualnej implementacji biały pion może otrzymywać wyższą wartość na `e2` niż na `e4` i `e6`.

Przykład logiczny:

- biały pion `e2` -> indeks po mirrorze odpowiada `e7`, gdzie tabela pionów daje duży bonus,
- biały pion `e4` -> niższy bonus,
- biały pion `e6` -> jeszcze niższy lub zerowy bonus.

To oznacza, że PST może karać naturalny rozwój pionów.

## 4.3. Problem dotyczy potencjalnie wszystkich figur

Nie należy przyjmować, że pozostałe tablice są bezpieczne tylko dlatego, że część z nich wygląda „prawie symetrycznie”.

Do audytu wymagają osobno:

- `PAWN`,
- `KNIGHT`,
- `BISHOP`,
- `ROOK`,
- `QUEEN`,
- `KING`,
- zarówno `MID`, jak i `END`.

Szczególnie istotne:

- `KING_MID` powinien premiować bezpieczeństwo/roszadę, nie centrum,
- `KING_END` powinien premiować centralizację,
- `PAWN_END` powinien mocniej premiować awans, zwłaszcza w końcówce,
- `KNIGHT` powinien preferować centrum i naturalne pola rozwojowe,
- `BISHOP` powinien preferować aktywne przekątne i unikać narożników,
- `ROOK` powinien preferować 7. linię i aktywne pliki,
- `QUEEN` zwykle nie powinna być zbyt agresywnie premiowana za wczesne wyjścia.

## 4.4. Rekomendowany kontrakt PST

Najbezpieczniejszy i najmniej mylący kontrakt:

> Wszystkie PST przechowujemy w naturalnym porządku `python-chess`, czyli bottom-up: `a1 = index 0`, `h8 = index 63`, z perspektywy białych.

Wtedy `__pst_value()` powinno logicznie działać tak:

```python
pst_index = square if color == chess.WHITE else chess.square_mirror(square)
```

Konsekwencje:

- biały używa tabeli bezpośrednio,
- czarny jest mapowany przez mirror do perspektywy białych,
- docstring musi zostać zmieniony,
- wszystkie tablice muszą być przekonwertowane/sprawdzone pod tę konwencję.

Alternatywnie można zachować kontrakt top-down, ale wtedy należy przekonwertować wszystkie tablice do top-down i pozostawić obecne `__pst_value()`. Ważne jest, aby kontrakt był jeden i jawny.

## 4.5. Kroki implementacyjne P0

1. Wybrać docelową konwencję PST.
   - Rekomendacja: bottom-up / `python-chess` natural order.
2. Zmienić docstring `__pst_value()` tak, aby opisywał realny kontrakt.
3. Dostosować `__pst_value()` do wybranego kontraktu.
4. Przejrzeć i przekonwertować każdą tabelę w `PIECE_SQUARE_TABLE_MID`.
5. Przejrzeć i przekonwertować każdą tabelę w `PIECE_SQUARE_TABLE_END`.
6. Dodać testy regresyjne dla `__pst_value()`.
7. Dodać test symetrii kolorów.
8. Uruchomić istniejące testy i nowe testy.

## 4.6. Testy regresyjne P0

Dodać do `engine/test_board_evaluator_trad_regression.py` testy bezpośrednio wywołujące prywatne `__pst_value()` przez name mangling, analogicznie jak obecne testy wywołują `__get_game_phase()`.

Minimalne testy:

1. Pion biały:
   - `PST(white pawn e4) >= PST(white pawn e2)` albo jasno zdefiniowana oczekiwana relacja.
   - `PST(white pawn e6) > PST(white pawn e2)` jeśli tabela końcowa/midgame ma premiować awans.
2. Pion czarny:
   - `PST(black pawn e5) == PST(white pawn e4)`.
   - `PST(black pawn e3) == PST(white pawn e6)`.
3. Skoczek:
   - `PST(white knight f3/c3) > PST(white knight g1/b1)`.
   - analogicznie dla czarnego `f6/c6` względem `g8/b8`.
4. Król middlegame:
   - bezpieczne pola przy roszadzie powinny być lepsze niż centrum w middlegame.
5. Król endgame:
   - centrum powinno być lepsze niż narożnik w endgame.
6. Symetria:
   - lustrzane pozycje białych/czarnych powinny dawać przeciwne wyniki albo identyczne wartości PST dla odpowiadających figur.

---

# 5. Problem P1: `PIECE_SQUARE_TABLE_END` jest częściowo kopią `MID`

## 5.1. Diagnoza

Historycznie `PIECE_SQUARE_TABLE_END` było identyczne jak `PIECE_SQUARE_TABLE_MID` dla:

- pionów,
- skoczków,
- gońców,
- wież,
- hetmana.

Różni się tylko król.

Status po aktualnym wdrożeniu:

- `PAWN_END` zostało rozdzielone od `PAWN_MID` i mocniej premiuje awans w końcówce.
- `KING_END` ma osobną tabelę końcówkową i premiuje centralizację.
- `KNIGHT_END`, `BISHOP_END`, `ROOK_END` i `QUEEN_END` nadal są kopiami tabel middlegame.

To oznacza, że interpolacja fazy:

```python
phase * pst_mid + (1 - phase) * pst_end
```

nie ma żadnego efektu dla większości figur.

## 5.2. Konsekwencje

- Piony są już dodatkowo premiowane za awans w końcówce przez PST.
- Skoczki/gońce/wieże/hetman nadal nie zmieniają preferencji pozycyjnych w zależności od fazy.
- Wartość `phase` wpływa na PST pionów i króla, king safety, bishop pair, rook activity, passed pawns i king activity, ale nie na PST większości figur.

## 5.3. Kroki implementacyjne P1

1. Po naprawie kontraktu PST przygotować realne tabele endgame.
2. Najpierw zmienić `PAWN_END` i `KING_END`, bo mają największe znaczenie końcówkowe.
3. Następnie rozważyć osobne `KNIGHT_END`, `BISHOP_END`, `ROOK_END`, `QUEEN_END`.
4. Nie przesadzać ze skalą, bo `PST_SCALE = 0.01` oznacza, że 100 punktów w tabeli = 1 pion.
5. Dodać test, że `PIECE_SQUARE_TABLE_END[chess.PAWN] != PIECE_SQUARE_TABLE_MID[chess.PAWN]`.
6. Dodać test pozycyjny: zaawansowany pion w końcówce powinien być oceniany lepiej niż analogiczny pion w middlegame przy zachowaniu pozostałych warunków.

## 5.4. Sugestie heurystyczne dla `PAWN_END`

W końcówce piony blisko promocji powinny mieć większy ciężar. Przykładowo:

- pion na 2. linii: mały bonus,
- pion na 4. linii: umiarkowany bonus,
- pion na 5. linii: istotny bonus,
- pion na 6. linii: duży bonus,
- pion na 7. linii: bardzo duży bonus.

Należy uważać, aby nie dublować nadmiernie bonusów z `__evaluate_pawn_structure()`, gdzie passed pawns już dostają bonus zależny od awansu.

---

# 6. Problem P1: stabilność `__evaluate_king_safety()`

## 6.1. Aktualny kod

Ataki w strefie króla są liczone tak:

```python
attack_penalty = 0.0
for sq in zone:
    for attacker in board.attackers(not color, sq):
        piece = board.piece_at(attacker)
        if piece:
            attack_penalty += get_piece_value(piece) * self.KING_ATTACK_WEIGHT
```

Ta sama figura może zostać policzona wiele razy, jeżeli atakuje wiele pól w strefie króla.

## 6.2. Diagnoza

To nie musi być czysty bug, bo liczba atakowanych pól wokół króla jest ważna. Problemem jest to, że kara rośnie liniowo bez limitu i jest proporcjonalna do wartości figury za każde atakowane pole.

Przykładowo hetman atakujący wiele pól strefy króla może wygenerować karę kilku pionów. To może zdominować materiał i inne heurystyki.

## 6.3. Rekomendowany kierunek

Wprowadzić stabilniejszy model, bez pełnej przebudowy silnika.

Opcja A — deduplikacja atakujących:

- zebrać `unique_attackers`,
- każdą figurę policzyć raz,
- ewentualnie dodać mały bonus/kara za liczbę atakowanych pól.

Opcja B — attack units:

- liczyć unikalnych atakujących,
- przypisać jednostki ataku zależne od typu figury,
- użyć ograniczonej/nieliniowej funkcji kary.

Opcja C — hybryda:

- osobno liczyć liczbę unikalnych atakujących,
- osobno liczyć liczbę atakowanych pól strefy,
- oba składniki ograniczyć capem.

## 6.4. Kroki implementacyjne P1

1. Utworzyć lokalne zbiory:
   - `unique_attackers`,
   - `attacked_zone_squares`.
2. Dla każdego pola strefy dodać atakujących do zbioru.
3. Policzyć karę za unikalnych atakujących na podstawie typu figury.
4. Dodać ograniczoną karę za liczbę atakowanych pól.
5. Dodać stałe klasowe, np.:
   - `KING_ATTACKER_WEIGHT_BY_PIECE`,
   - `KING_ATTACKED_ZONE_SQUARE_WEIGHT`,
   - `KING_ATTACK_PENALTY_LIMIT`.
6. Zachować mnożenie przez `phase` w `evaluate_board()`.
7. Dodać test regresyjny, że jedna figura atakująca wiele pól nie generuje ekstremalnej kary.

## 6.5. Testy regresyjne P1

1. Pozycja z hetmanem atakującym kilka pól wokół króla.
2. Porównać karę z pozycją, gdzie hetman atakuje jedno pole.
3. Sprawdzić, że kara rośnie, ale nie eksploduje liniowo bez limitu.
4. Sprawdzić, że `evaluate_board()` nadal nie mutuje `board.fen()`.

---

# 7. Problem P2: `threats/hanging pieces`

## 7.1. Aktualny model

Kod ocenia:

- niebronione figury,
- figury atakowane przez piony,
- figury atakowane przez niższą figurę.

Przykład:

```python
if not defenders:
    penalty += min(self.HANGING_PIECE_PENALTY_LIMIT,
                   self.HANGING_PIECE_VALUE_WEIGHT * piece_value)

if pawn_attackers:
    penalty += min(self.PAWN_ATTACKED_PIECE_PENALTY_LIMIT,
                   self.PAWN_ATTACKED_PIECE_VALUE_WEIGHT * piece_value)

if weakest_attacker_value < piece_value:
    penalty += min(self.LOWER_VALUE_ATTACK_PENALTY_LIMIT,
                   self.LOWER_VALUE_ATTACK_WEIGHT * (piece_value - weakest_attacker_value))
```

## 7.2. Diagnoza

Uwagi z poprzedniego planu są częściowo trafne:

- kara za wiszącą figurę może być zbyt niska,
- lower-value attack ignoruje szczegółową analizę wymian,
- model może dublować ocenę z search/quiescence,
- zbyt agresywne podniesienie wag może dawać fałszywe kary.

Nie należy jednak traktować każdej bronionej figury atakowanej przez piona jako „zdrowej”. Pion atakujący gońca/skoczka często jest realną groźbą, nawet jeśli figura jest broniona.

## 7.3. Kroki implementacyjne P2

1. Nie zmieniać wag przed naprawą PST.
2. Dodać testy dokumentujące obecne zachowanie dla kilku prostych pozycji.
3. Rozważyć prosty filtr wymian:
   - jeśli figura jest niebroniona, kara zostaje,
   - jeśli figura jest broniona, ale najtańszy atakujący jest dużo tańszy niż figura, kara zostaje, lecz niższa,
   - jeśli najtańszy obrońca i atakujący sugerują wymianę nieopłacalną dla atakującego, kara może być zmniejszona.
4. Rozważyć zwiększenie `HANGING_PIECE_PENALTY_LIMIT`, ale dopiero po testach.
5. Nie przekraczać skali, w której statyczna heurystyka zastępuje search.

## 7.4. Testy regresyjne P2

Przykładowe pozycje:

1. Niebroniony skoczek atakowany przez gońca/piona powinien obniżyć ocenę strony posiadającej skoczka.
2. Niebroniona dama atakowana przez lekką figurę powinna być karana wyraźniej niż niebroniony skoczek.
3. Broniona figura atakowana przez pion powinna mieć karę mniejszą niż figura niebroniona.
4. Pozycja bez ataków powinna mieć threats score bliski 0.

---

# 8. Problem P2/P3: duplikacja `is_passed_pawn()`

## 8.1. Diagnoza

`is_passed_pawn()` występuje jako funkcja lokalna w:

- `__evaluate_pawn_structure()`,
- `__evaluate_endgame_king_activity()`.

Logika jest taka sama. To jest problem utrzymaniowy i potencjalnie wydajnościowy.

## 8.2. Kroki implementacyjne

1. Wyciągnąć metodę prywatną:

```python
def __is_passed_pawn(self, board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    ...
```

2. Użyć jej w obu miejscach.
3. Opcjonalnie dodać helper:

```python
def __passed_pawns(self, board: chess.Board, color: chess.Color) -> list[chess.Square]:
    ...
```

4. Jeśli wydajność stanie się problemem, policzyć passed pawns raz w `evaluate_board()` i przekazać/cache'ować lokalnie. Na razie prosty refaktor wystarczy.

## 8.3. Testy regresyjne

Istniejący test `test_passed_pawn_bonus_increases_with_advancement` powinien nadal przechodzić.

Dodać można:

- pion z przeciwstawnym pionem na tej samej linii przed nim nie jest passed pawn,
- pion z przeciwnym pionem na sąsiedniej linii przed nim nie jest passed pawn,
- pion bez przeciwnych pionów przed nim na tej samej/sąsiedniej linii jest passed pawn.

---

# 9. Problem P3: open-file scan w `__evaluate_king_safety()`

## 9.1. Diagnoza

Obecny kod działa funkcjonalnie, ale skanuje wszystkie pola:

```python
for sq in chess.SQUARES
```

i wielokrotnie wywołuje `board.piece_at(sq)`.

## 9.2. Kroki implementacyjne

1. Przed pętlą po plikach wyliczyć:

```python
own_pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
opp_pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, not color)}
```

2. Dla każdego pliku wokół króla sprawdzić członkostwo w zbiorze.
3. Zachować tę samą semantykę open/semi-open:
   - brak własnych i brak przeciwnych pionów -> open file penalty,
   - brak własnych, są przeciwne piony -> semi-open file penalty.
4. Dodać test porównujący wynik starej i nowej logiki albo prosty test pozycyjny dla open/semi-open/closed file.

---

# 10. Problem P2: endgame king activity przy obecności hetmanów

## 10.1. Diagnoza

`__get_game_phase()` opiera się tylko na materiale niepionowym. To jest standardowe podejście i samo w sobie nie jest błędem.

Ryzyko pojawia się w `__evaluate_endgame_king_activity()`: król może otrzymywać bonus za centralizację, gdy na planszy nadal są hetmany, jeśli `phase < 1.0`.

Aktualne bonusy są małe, więc nie jest to krytyczne. Może jednak prowadzić do zbyt wczesnego zachęcania króla do centrum.

Audyt aktualnej implementacji wskazuje dodatkowy, konkretny przypadek: `KQ vs KQ` może otrzymać `phase == 0.0`, bo dwa hetmany mają łączną wartość `19.0`, a `ENDGAME_MATERIAL_THRESHOLD` wynosi `20.0`. W takiej sytuacji:

- `king_safety_score = phase * self.__evaluate_king_safety(board)` całkowicie wyłącza ocenę bezpieczeństwa króla,
- `__evaluate_endgame_king_activity()` działa z pełną wagą końcówkową,
- pozycje z hetmanami mogą być traktowane jak czyste końcówki królewsko-pionowe.

To jest ważniejszy problem niż sama centralizacja króla, bo dotyczy równocześnie wyłączenia king safety i aktywacji endgame king activity.

## 10.2. Kroki implementacyjne

1. Nie zmieniać `__get_game_phase()` jako pierwszej poprawki.
2. Dodać osobny mnożnik dla centralizacji króla:
   - pełny bonus, gdy hetmanów nie ma,
   - zmniejszony bonus, gdy jedna lub obie strony mają hetmana.
3. Nie wyłączać koniecznie całej aktywności króla, bo presja na piony lub zatrzymywanie wolnych pionów może być nadal sensowne.
4. Rozważyć osobne stałe:
   - `ENDGAME_KING_CENTER_QUEENS_ON_BOARD_MULTIPLIER`,
   - `ENDGAME_KING_ACTIVITY_QUEENS_ON_BOARD_MULTIPLIER`.
5. Rozważyć niezależny mnożnik dla king safety przy obecnych hetmanach, np. dolny limit efektywnej fazy tylko dla king safety:

```python
effective_king_safety_phase = max(phase, KING_SAFETY_QUEENS_ON_BOARD_PHASE_FLOOR) if queens_on_board else phase
```

6. Alternatywnie skorygować `__get_game_phase()` tak, aby sama obecność hetmanów nie dawała pełnej końcówki, ale trzeba uważać, by nie popsuć prostego i standardowego kontraktu fazy gry.

## 10.3. Testy regresyjne

1. W czystej końcówce król w centrum powinien być lepszy niż król w rogu.
2. Przy obecności hetmanów różnica centrum/róg powinna być mniejsza.
3. `test_endgame_king_activity_is_inactive_in_middlegame_and_active_in_endgame` powinien nadal przechodzić.
4. Dla pozycji `KQ vs KQ` test powinien dokumentować, że king safety nie jest całkowicie ignorowane albo że aktywność króla jest osłabiona.
5. Dodać test porównujący centrum/róg przy samych królach oraz przy `KQ vs KQ`; różnica przy hetmanach powinna być wyraźnie mniejsza.

---

# 11. Passed pawn support — korekta wcześniejszej diagnozy

## 11.1. Aktualny kod

```python
friendly_pawn_attackers = board.attackers(color, square) & pawns
if friendly_pawn_attackers:
    bonus += self.PASSED_PAWN_SUPPORTED_BY_PAWN_BONUS + \
             self.PASSED_PAWN_SUPPORTED_BY_PAWN_ENDGAME_BONUS * endgame_weight
elif board.attackers(color, square):
    bonus += self.PASSED_PAWN_SUPPORTED_BY_PIECE_BONUS
```

## 11.2. Co jest poprawne

`board.attackers(color, square)` wykrywa figury wspierające pion, także liniowo od tyłu, o ile geometria szachowa na to pozwala.

Nie należy więc poprawiać tej logiki na podstawie błędnego założenia, że wieża/hetman „z tyłu” nie są wykrywane.

## 11.3. Co można ulepszyć

Obecna heurystyka jest mało szczegółowa:

- taki sam bonus za skoczka, gońca, wieżę, hetmana i króla,
- brak oceny, czy pole przed pionem jest kontrolowane,
- brak oceny ścieżki promocji,
- brak rozróżnienia wsparcia z tyłu przez wieżę/hetmana od przypadkowej obrony.

To jednak nie jest priorytet przed PST.

## 11.4. Test dokumentacyjny

Dodać test, który potwierdza obecne zachowanie `board.attackers()`:

- biały pion na `e4`, biała wieża na `e1`, brak blokujących figur,
- `chess.E1 in board.attackers(chess.WHITE, chess.E4)` powinno być `True`.

To zabezpieczy przed ponownym wprowadzeniem błędnej diagnozy.

---

# 12. Co działa dobrze i należy zachować

Następujące elementy są poprawne lub przynajmniej dobrym kierunkiem:

- `evaluate_board()` nie mutuje `board.turn` ani FEN.
- Mobility liczy atakowane pola przez `board.attacks(sq)` i filtruje własne figury przez `~board.occupied_co[color]`.
- Terminalne stany są obsłużone jasno:
  - mat -> `±math.inf`,
  - remisy -> `0.0`.
- Materiał króla nie jest dodawany do material score.
- King safety jest mnożone przez `phase`, więc naturalnie słabnie w końcówce.
- Endgame king activity jest zerowe przy `endgame_weight <= 0.0`.
- Rook activity ma sensowne składniki: open/semi-open files, seventh rank, doubled rooks.
- Bishop pair bonus uwzględnia otwartość pozycji i fazę.
- Testy regresyjne sprawdzają brak mutacji pozycji.

Te elementy powinny zostać utrzymane podczas refaktoru.

---

# 13. Priorytety wdrożenia

## P0 — poprawność PST

Najważniejszy etap.

Zakres:

1. Ustalić kontrakt orientacji PST.
2. Poprawić `__pst_value()` i docstring.
3. Przekonwertować wszystkie PST.
4. Dodać testy orientacji i symetrii.

Bez tego dalszy tuning wag jest mało wiarygodny.

## P1 — realne PST_END

Zakres:

1. Rozdzielić `MID` i `END` przynajmniej dla pionów.
2. Upewnić się, że król ma poprawny kontrakt w obu fazach.
3. Dodać testy, że faza gry realnie wpływa na PST.

## P1 — stabilizacja king safety

Zakres:

1. Ograniczyć wielokrotne naliczanie tej samej figury.
2. Dodać cap lub model attack units.
3. Dodać testy zakresowe.

## P2 — threats/hanging pieces

Zakres:

1. Najpierw testy dokumentujące obecne zachowanie.
2. Potem ostrożny tuning wag lub prosty filtr wymian.

## P2 — endgame king activity z hetmanami

Zakres:

1. Nie zmieniać całej fazy gry.
2. Dodać mnożnik osłabiający centralizację króla przy hetmanach.

## P1/P2 — phase, hetmany i king safety

Zakres:

1. Dodać test dokumentujący, że `KQ vs KQ` obecnie może mieć `phase == 0.0`.
2. Zdecydować, czy poprawka ma być w `__get_game_phase()`, czy lokalnie w king safety / endgame king activity.
3. Preferowana bezpieczna poprawka: nie zmieniać globalnego kontraktu `phase`, tylko:
   - osłabić endgame king activity przy hetmanach,
   - dodać minimalny efektywny mnożnik king safety przy hetmanach.
4. Dodać test, że obecność hetmanów zmniejsza zachętę do centralizacji króla i nie zeruje całkowicie bezpieczeństwa króla.

## P3 — refaktor passed pawns i open-file scan

Zakres:

1. Wyciągnąć `__is_passed_pawn()`.
2. Zoptymalizować open-file scan.
3. Zachować istniejące wyniki testów.

## P4 — dalsze heurystyki

Dopiero po stabilizacji powyższych:

- outposts,
- bad bishop,
- space evaluation,
- pawn chains/levers/backward pawns,
- tempo bonus,
- mop-up evaluation.

---

# 14. Proponowana kolejność commitów / etapów pracy

## Etap 1 — testy pokazujące obecne problemy PST ✅ WYKONANE

Bez zmiany produkcyjnego kodu:

1. Dodać testy `__pst_value()` dla pionów, skoczków, króla.
2. Potwierdzić, że część testów obecnie failuje.
3. Dodać test symetrii white/black.

Cel: mieć czerwone testy opisujące błąd.

Status: wykonane. Dodano testy `__pst_value()` dla pionów, symetrii kolorów, skoczków oraz króla mid/end. Aktualna implementacja reprodukuje oczekiwane czerwone przypadki dla pionów.

## Etap 2 — kontrakt PST i konwersja tabel ✅ WYKONANE

1. Wybrać bottom-up jako docelowy format.
2. Zmienić `__pst_value()`.
3. Przekonwertować tabele.
4. Uruchomić testy PST.
5. Uruchomić pełne testy regresyjne.

Cel: stabilny i jawny kontrakt PST.

Status: wykonane. Przyjęto kontrakt bottom-up zgodny z naturalnym porządkiem `python-chess`: białe figury używają `square`, czarne `chess.square_mirror(square)`. Piony pozostały w dotychczasowej orientacji, a pozostałe PST zostały odwrócone pionowo. Testy regresyjne `test_board_evaluator_trad_regression.py` przechodzą.

## Etap 3 — realne PST_END ✅ WYKONANE

1. Rozdzielić `PAWN_END` od `PAWN_MID`.
2. Zweryfikować `KING_END` po konwersji.
3. Opcjonalnie rozdzielić pozostałe figury.
4. Dodać testy fazowania.

Cel: interpolacja fazy działa realnie.

Status: wykonane. `PIECE_SQUARE_TABLE_END[chess.PAWN]` różni się od `MID` i mocniej premiuje zaawansowane piony w końcówce. Dodano testy interpolacji fazy, symetrii kolorów w endgame PST oraz większego spreadu awansu pionów w końcówce. Testy regresyjne przechodzą.

## Etap 4 — king safety ✅ WYKONANE

1. Dodać test pozycji z jedną figurą atakującą wiele pól.
2. Zmienić model ataków na deduplikowany lub attack-units.
3. Dodać limity.
4. Porównać wyniki wybranych pozycji reprezentatywnych.

Cel: brak ekstremalnych kar dominujących materiał.

Status: wykonane. Model ataków w strefie króla deduplikuje atakujące figury, dodaje mały limitowany składnik za liczbę atakowanych pól i ogranicza całkowitą karę. Dodano test regresyjny porównujący jednego hetmana atakującego jedno pole strefy z tym samym hetmanem atakującym wiele pól. Testy regresyjne przechodzą.

## Etap 5 — refaktory i tuning ✅ WYKONANE

1. Wyciągnąć `__is_passed_pawn()`.
2. Zoptymalizować open-file scan.
3. Dopiero potem tuning `threats`.

Cel: poprawa utrzymania i wydajności bez zmiany sensu ewaluacji.

Status: wykonane. Wyciągnięto wspólne helpery `__is_passed_pawn()` i `__passed_pawns()`, użyto ich w strukturze pionowej oraz aktywności króla w końcówce, a open/semi-open file scan w king safety zoptymalizowano przez zbiory plików pionów. Dodano testy regresyjne dla helperów passed pawns oraz kolejności closed > semi-open > open w king safety. Dodano też testy dla `__evaluate_threats_and_hanging_pieces()` i minimalny filtr unikający podwójnego karania figury atakowanej pionem jako jednocześnie `pawn attacked` i `lower-value attacked`. Testy regresyjne przechodzą.

## Etap 6 — rekomendacje po audycie aktualnej implementacji ✅ WYKONANE

Cel: zamknąć pozostałe problemy kalibracji po wdrożeniu bazowych napraw PST i stabilizacji king safety. Ten etap powinien być wdrażany małymi, testowalnymi krokami, najlepiej w kolejności 6.1 → 6.2 → 6.3 → 6.4 → 6.5 → 6.6.

### Etap 6.1 — pełna symetria całej ewaluacji ✅ WYKONANE

Zakres:

1. Dodać test regresyjny dla `evaluate_board(board) == -evaluate_board(board.mirror())` w tolerancji numerycznej.
2. Użyć kilku pozycji reprezentatywnych, obejmujących co najmniej:
   - pozycję startową,
   - typową pozycję middlegame,
   - końcówkę pionową,
   - pozycję z hetmanami,
   - pozycję z aktywnością wież,
   - pozycję z `threats/hanging pieces`.
3. W tym samym teście albo helperze potwierdzić, że `evaluate_board()` nie mutuje `board.fen()`.

Kryteria akceptacji:

- dla każdej pozycji `abs(evaluate_board(board) + evaluate_board(board.mirror()))` jest bliskie `0.0`,
- test przechodzi razem z istniejącym `test_start_position_is_balanced`,
- brak mutacji FEN dla pozycji oryginalnej i lustrzanej.

Status: wykonane. Dodano test `test_representative_positions_keep_full_mirror_symmetry`, który obejmuje pozycję startową, middlegame, końcówkę pionową, pozycję z hetmanami, aktywność wież oraz `threats/hanging pieces`. Test sprawdza symetrię `evaluate_board(board) == -evaluate_board(board.mirror())` i brak mutacji FEN dla pozycji oryginalnej oraz lustrzanej. Testy regresyjne przechodzą.

### Etap 6.2 — reprodukcja problemu `KQ vs KQ` / `phase` ✅ WYKONANE

Zakres:

1. Dodać test dokumentujący aktualne zachowanie `__get_game_phase()` dla pozycji z samymi królami i hetmanami.
2. Jawnie sprawdzić przypadek, w którym dwa hetmany mają łączny materiał `19.0`, czyli mniej niż `ENDGAME_MATERIAL_THRESHOLD == 20.0`.
3. Udokumentować skutek w `evaluate_board()`: `king_safety_score = phase * __evaluate_king_safety(board)` może wyzerować king safety, mimo obecności hetmanów.
4. Nie naprawiać jeszcze kodu w tym samym kroku, jeśli celem jest najpierw czerwony test/regresja problemu.

Kryteria akceptacji:

- test jasno pokazuje, czy `KQ vs KQ` otrzymuje `phase == 0.0`,
- test opisuje oczekiwane zachowanie docelowe przed wdrożeniem poprawki,
- po późniejszej poprawce test zostaje zaktualizowany tak, aby chronił przed ponownym wyłączeniem king safety przy hetmanach.

Status: wykonane. Dodano testy `test_game_phase_treats_kq_vs_kq_as_full_endgame` oraz `test_phase_zero_nullifies_king_safety_in_kq_vs_kq`. Testy dokumentują obecne zachowanie: pozycja `KQ vs KQ` może mieć `phase == 0.0`, a niezerowy surowy wynik `__evaluate_king_safety()` jest zerowany przez mnożenie `phase * king_safety`. Kod produkcyjny nie został jeszcze zmieniony; poprawka należy do Etapu 6.3.

### Etap 6.3 — king safety przy obecnych hetmanach ✅ WYKONANE

Zakres:

1. Preferowana poprawka: nie zmieniać globalnego kontraktu `__get_game_phase()`, tylko dodać lokalny efektywny mnożnik king safety w `evaluate_board()`.
2. Rozważyć stałą klasową, np. `KING_SAFETY_QUEENS_ON_BOARD_PHASE_FLOOR`.
3. Przykładowy kierunek:

```text
queens_on_board = bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))
effective_king_safety_phase = max(phase, self.KING_SAFETY_QUEENS_ON_BOARD_PHASE_FLOOR) if queens_on_board else phase
king_safety_score = effective_king_safety_phase * self.__evaluate_king_safety(board)
```

4. Zachować naturalne wygaszanie king safety w czystych końcówkach bez hetmanów.
5. Dodać test porównujący pozycję z odsłoniętym królem przy hetmanach przed/po zmianie efektywnego mnożnika.

Kryteria akceptacji:

- przy obecnych hetmanach king safety nie spada do zera wyłącznie dlatego, że `phase == 0.0`,
- czyste końcówki bez hetmanów nadal nie są nadmiernie karane przez king safety,
- symetria `evaluate_board(board) == -evaluate_board(board.mirror())` pozostaje zachowana.

Status: wykonane. Dodano stałą `KING_SAFETY_QUEENS_ON_BOARD_PHASE_FLOOR = 0.35` oraz helper `__effective_king_safety_phase()`. `evaluate_board()` używa teraz efektywnej fazy tylko dla `king_safety_score`, bez zmiany globalnego kontraktu `__get_game_phase()`. Dodano testy `test_raw_phase_would_nullify_king_safety_in_kq_vs_kq`, `test_king_safety_phase_uses_queen_floor_only_when_queens_are_present` oraz `test_evaluate_board_applies_queen_king_safety_floor_without_mutation`. Testy regresyjne przechodzą.

### Etap 6.4 — osłabienie endgame king activity przy hetmanach ✅ WYKONANE

Zakres:

1. Dodać mnożnik osłabiający centralizację króla, gdy na planszy jest co najmniej jeden hetman.
2. Rozważyć osobne mnożniki:
   - `ENDGAME_KING_CENTER_QUEENS_ON_BOARD_MULTIPLIER` dla samej centralizacji,
   - `ENDGAME_KING_ACTIVITY_QUEENS_ON_BOARD_MULTIPLIER` dla pozostałych składników aktywności, jeśli testy pokażą taką potrzebę.
3. Nie wyłączać całej aktywności króla automatycznie, bo wsparcie własnych wolnych pionów i zatrzymywanie cudzych może nadal mieć sens.
4. Dodać testy porównujące:
   - czystą końcówkę bez hetmanów: król w centrum vs król w rogu,
   - analogiczną pozycję z hetmanami: różnica centrum/róg powinna być mniejsza.

Kryteria akceptacji:

- aktywny król nadal jest premiowany w czystej końcówce,
- przy `KQ vs KQ` lub pozycjach z hetmanami premia za centralizację jest wyraźnie mniejsza,
- test `test_endgame_king_activity_is_inactive_in_middlegame_and_active_in_endgame` nadal przechodzi.

Status: wykonane. Dodano stałą `ENDGAME_KING_CENTER_QUEENS_ON_BOARD_MULTIPLIER = 0.35`. W `__evaluate_endgame_king_activity()` osłabiany jest wyłącznie składnik centralizacji króla, gdy na planszy jest co najmniej jeden hetman; presja na piony oraz wsparcie/zatrzymywanie wolnych pionów pozostają bez dodatkowego osłabienia. Dodano test `test_endgame_king_center_bonus_is_reduced_when_queens_are_on_board`, który porównuje spread centrum/róg w czystej końcówce i w analogicznej pozycji z hetmanami. Testy regresyjne przechodzą.

### Etap 6.5 — decyzja o `PST_END` dla skoczka, gońca, wieży i hetmana ✅ WYKONANE

Zakres:

1. Podjąć jawną decyzję, czy `KNIGHT_END`, `BISHOP_END`, `ROOK_END` i `QUEEN_END` mają pozostać kopiami `MID`, czy mają dostać osobne tabele końcówkowe.
2. Opcja A — świadomie zostawić kopie:
   - dodać test dokumentacyjny potwierdzający równość tych tabel,
   - opisać, że na tym etapie fazowanie PST dotyczy głównie pionów i króla.
3. Opcja B — rozdzielić wybrane tabele:
   - przygotować osobne wartości końcówkowe,
   - dodać testy interpolacji,
   - dodać testy symetrii kolorów,
   - sprawdzić, że skala `PST_SCALE = 0.01` nie powoduje przeważenia materiału.

Kryteria akceptacji:

- decyzja jest zapisana w planie i zabezpieczona testem,
- brak przypadkowej, nieudokumentowanej zmiany tabel PST,
- pełne testy regresyjne nadal przechodzą.

Status: wykonane. Podjęto świadomą decyzję, że na tym etapie `KNIGHT_END`, `BISHOP_END`, `ROOK_END` i `QUEEN_END` pozostają kopiami tabel middlegame. Oznacza to, że fazowanie PST jest obecnie realne głównie dla pionów i króla. Dodano test `test_minor_and_major_piece_endgame_psts_intentionally_match_middlegame`, który zabezpiecza równość `END == MID` dla skoczka, gońca, wieży i hetmana, oraz test `test_pawn_and_king_endgame_psts_remain_distinct_from_middlegame`, który potwierdza, że pion i król pozostają rozdzielone. Testy regresyjne przechodzą.

### Etap 6.6 — pinned/static attackers w `king safety` i `threats` ✅ WYKONANE

Zakres:

1. Dodać testy dokumentujące, jak `board.attackers()` traktuje atakujących związanych lub taktycznie nierealnych.
2. Przygotować warianty dla:
   - `__evaluate_king_safety()` — figura formalnie atakuje strefę króla, ale jest związana,
   - `__evaluate_threats_and_hanging_pieces()` — figura formalnie atakuje bierkę, ale wymiana jest nielegalna albo taktycznie nieopłacalna.
3. Dodać warianty symetryczne dla białych i czarnych.
4. Na tym etapie nie trzeba wdrażać pełnego SEE w ewaluatorze; celem może być dokumentacja ograniczeń statycznej heurystyki.

Kryteria akceptacji:

- testy jasno pokazują ograniczenia statycznego `board.attackers()`,
- zachowanie jest symetryczne kolorystycznie,
- jeśli wyniki są akceptowane jako ograniczenie heurystyki, zostają opisane w komentarzu/test name; jeśli nie, powstaje osobny plan poprawki z SEE lub filtrem legalności/pinów.

Status: wykonane. Dodano test `test_king_safety_documents_pinned_attackers_are_counted_statically`, który pokazuje, że związany skoczek nadal jest liczony przez `board.attackers()` jako atakujący pola strefy króla i wpływa na `__evaluate_king_safety()`. Dodano też test `test_threats_document_pinned_attackers_are_counted_statically`, który pokazuje analogiczne ograniczenie w `__evaluate_threats_and_hanging_pieces()`: związany skoczek formalnie atakujący hetmana nadal generuje statyczną karę. Oba testy mają warianty lustrzane dla białych i czarnych oraz potwierdzają symetrię znaku. Nie wdrożono pełnego SEE ani filtrowania pinów; zachowanie zostało świadomie udokumentowane jako ograniczenie statycznej heurystyki. Testy regresyjne przechodzą.

---

# 15. Proponowane komendy testowe

Z katalogu `engine`:

```powershell
python -m unittest test_board_evaluator_trad_regression.py
```

Z katalogu głównego repozytorium, jeśli importy zostaną dostosowane lub ścieżka będzie ustawiona:

```powershell
python -m unittest discover engine
```

Jeśli pojawią się problemy z importami modułów bez pakietu, najbezpieczniej uruchamiać testy z katalogu `engine`, tak jak obecna struktura importów sugeruje.

---

# 16. Kryteria akceptacji po wdrożeniu P0/P1

Po wdrożeniu pierwszych etapów powinny być spełnione warunki:

1. `evaluate_board(chess.Board())` nadal zwraca wartość bliską `0.0`.
2. Ewaluacja nie mutuje `board.fen()`.
3. Lustrzane pozycje białych/czarnych mają symetryczne wyniki.
4. Naturalne rozwinięcie figur nie jest karane przez PST.
5. Biały i czarny odpowiednik tej samej pozycji figury mają równą wartość PST.
6. Król w middlegame preferuje bezpieczeństwo, a w endgame centrum.
7. `PIECE_SQUARE_TABLE_END` realnie różni się od `MID` przynajmniej dla pionów i króla.
8. Wszystkie istniejące testy regresyjne przechodzą.
9. Dla pozycji z hetmanami king safety nie powinno być przypadkowo całkowicie wyłączane wyłącznie z powodu `phase == 0.0`.
10. Aktywność króla w końcówce powinna być osłabiona przy obecności hetmanów względem czystych końcówek bez hetmanów.
11. Pełna ocena powinna przechodzić test symetrii `evaluate_board(board) == -evaluate_board(board.mirror())` dla reprezentatywnego zestawu pozycji.

---

# 17. Odpowiedź na pytanie: czy heurystyki są wystarczające?

Po naprawie PST i stabilizacji wag implementacja może być wystarczająca jako klasyczna ewaluacja dla prostego/średniego silnika hobbystycznego.

Nie będzie jednak porównywalna z nowoczesnymi silnikami bez:

- lepszego searchu,
- quiescence search / SEE,
- transposition tables,
- bardziej zaawansowanego king safety,
- strojenia wag na partiach/test suite,
- bardziej szczegółowej struktury pionowej,
- outpostów,
- bad bishop,
- space evaluation,
- tempo bonus,
- mop-up evaluation.

Najpierw jednak należy naprawić podstawy, bo tuning zaawansowanych heurystyk na błędnie zorientowanych PST byłby niewiarygodny.

---

# 18. TL;DR — skorygowana lista zadań po audycie

1. **P0: Ujednolicić orientację PST dla wszystkich figur**, nie tylko pionów — status: wykonane.
2. **P0: Dodać testy orientacji i symetrii PST** — status: wykonane.
3. **P1: Rozdzielić `PIECE_SQUARE_TABLE_END` od `MID` dla pionów i króla** — status: wykonane.
4. **P1/P2: Zdecydować, czy rozdzielać `KNIGHT_END`, `BISHOP_END`, `ROOK_END`, `QUEEN_END`** — status: wykonane; na tym etapie świadomie pozostają kopiami `MID`.
5. **P1: Ustabilizować king safety**, aby jedna figura nie generowała ekstremalnej kary przez wiele pól strefy — status: wykonane.
6. **P1/P2: Naprawić interakcję `phase`, hetmanów i king safety**, szczególnie przypadek `KQ vs KQ`, gdzie `phase` może spaść do `0.0` — status: wykonane lokalnym floor dla king safety.
7. **P2: Osłabić centralizację/aktywność króla przy hetmanach**, bez koniecznego przebudowywania całej fazy gry — status: wykonane dla składnika centralizacji.
8. **P2: Dodać pełny test symetrii `evaluate_board(board) == -evaluate_board(board.mirror())`** dla pozycji reprezentatywnych — status: wykonane.
9. **P2: Ostrożnie poprawić threats/hanging pieces**, najlepiej po dodaniu testów i prostego filtra wymian / dokumentacji ograniczeń bez SEE — status: częściowo wykonane; ograniczenia bez SEE zostały udokumentowane testami.
10. **P2: Dodać testy pinned/static attackers** dla `king safety` i `threats`, bo `board.attackers()` jest statyczne — status: wykonane.
11. **P3: Wyciągnąć wspólne `__is_passed_pawn()`** — status: wykonane.
12. **P3: Zoptymalizować open-file scan w king safety** — status: wykonane.
13. **Nie naprawiać passed-pawn support na podstawie tezy, że `board.attackers()` nie widzi obrony od tyłu** — ta teza jest błędna.
14. **Dalsze heurystyki dodawać dopiero po stabilizacji regresji**.
