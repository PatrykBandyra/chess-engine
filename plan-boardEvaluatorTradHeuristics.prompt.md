# Plan implementacji usprawnień `BoardEvaluatorTrad`

Poniżej masz plan implementacji **maksymalnie 10 kroków**, ułożony od największego wpływu do mniejszego. Nic nie zmieniam teraz w kodzie — to tylko plan dla `BoardEvaluatorTrad`.

## Plan implementacji usprawnień `BoardEvaluatorTrad`

### 1. [x] Naprawić orientację `PIECE_SQUARE_TABLE_*`

**Cel:** usunąć prawdopodobnie najważniejszy błąd obecnej ewaluacji.

Obecne indeksowanie wygląda podejrzanie:

```python
sq if color == chess.WHITE else chess.square_mirror(sq)
```

Dla klasycznych PST zapisanych od 8. rzędu do 1. rzędu powinno być raczej odwrotnie:

```python
chess.square_mirror(sq) if color == chess.WHITE else sq
```

**Zakres implementacji:**

- dodać pomocniczą metodę typu `__pst_value(piece_type, square, color, phase)`,
- w jednym miejscu obsłużyć mirrorowanie,
- usunąć duplikację lookupu mid/end,
- zostawić wynik nadal z perspektywy białych.

**Oczekiwany efekt:**

- lepsza gra debiutowa,
- poprawne premiowanie rozwoju figur,
- poprawne premiowanie roszady,
- mniej absurdalnych kar dla białego króla na `g1`/`c1`.

**Ryzyko/koszt:** średnie — po zmianie oceny mogą się mocno przesunąć, więc trzeba później dostroić wagi.

**Testy:**

- startowa pozycja powinna dawać około `0.0`,
- symetryczne pozycje powinny dawać przeciwne wartości po zamianie kolorów,
- biały skoczek na `f3` i czarny skoczek na `f6` powinny być oceniane symetrycznie,
- roszada nie powinna być karana PST króla.

---

### 2. [x] Nie liczyć wartości materialnej króla jak zwykłej figury

**Cel:** uprościć i ustabilizować materiał.

Obecnie `PIECE_VALUES` zawiera:

```python
chess.KING: 100_000
```

W praktyce wartości królów się znoszą, ale to niepotrzebnie zwiększa liczby pośrednie i może utrudniać tuning. Mat i tak obsługuje osobna logika.

**Zakres implementacji:**

- w pętli materiału pominąć materialną wartość króla,
- nadal można liczyć PST króla,
- terminalne maty zostawić bez zmian.

**Oczekiwany efekt:**

- czystsza skala ewaluacji,
- łatwiejsze strojenie,
- mniejsze ryzyko efektów ubocznych.

**Ryzyko/koszt:** niskie.

**Testy:**

- startowa pozycja nadal `0.0`,
- pozycje z równym materiałem nie powinny zmienić znaku oceny,
- mat nadal zwraca `±inf`.

---

### 3. [x] Sfazować `__evaluate_king_safety`

**Cel:** król ma być chroniony w middlegame, ale aktywny w endgame.

Obecnie `king_safety_score` działa tak samo w każdej fazie. To jest problem, bo w końcówce brak tarczy pionowej i otwarte linie przy królu nie powinny być tak mocno karane.

**Zakres implementacji:**

- w `evaluate_board()` użyć:

```python
king_safety_score = phase * self.__evaluate_king_safety(board)
```

- ewentualnie dodatkowo osłabić ataki, gdy nie ma hetmanów,
- później można dodać osobną heurystykę aktywności króla w końcówce.

**Oczekiwany efekt:**

- lepsze końcówki,
- mniej pasywnego trzymania króla w rogu,
- bardziej naturalne przejście middlegame → endgame.

**Ryzyko/koszt:** niskie.

**Testy:**

- w middlegame odsłonięty król powinien być karany,
- w końcówce aktywny król w centrum nie powinien dostawać dużej kary,
- pozycje bez hetmanów powinny mieć mniejszy wpływ king safety.

---

### 4. [x] Ulepszyć passed pawns w `__evaluate_pawn_structure`

**Cel:** lepiej oceniać realną siłę wolnych pionów.

Obecnie każdy passed pawn ma stały bonus:

```python
bonus = 0.3 * passed_bonus
```

To za mało precyzyjne. Wolny pion na 7. rzędzie jest dużo ważniejszy niż wolny pion na 2. rzędzie.

**Zakres implementacji:**

Dodać bonus zależny od:

- rzędu piona,
- czy jest zablokowany,
- czy jest broniony przez własnego piona/figurę,
- czy ma sąsiedniego wolnego piona,
- fazy gry — passed pawns ważniejsze w końcówce.

Przykładowo:

```text
passed pawn na 2/7 początkowo: mały bonus
passed pawn na 5/4: średni bonus
passed pawn na 6/3: duży bonus
passed pawn na 7/2: bardzo duży bonus
```

**Oczekiwany efekt:**

- dużo lepsze końcówki pionowe,
- lepsza gra wieżowa,
- poprawniejsze decyzje wymian prowadzących do wolnego piona.

**Ryzyko/koszt:** średnie — łatwo przeszacować dalekie wolne piony.

**Testy:**

- wolny pion na `e4`, `e5`, `e6`, `e7` powinien mieć rosnącą wartość,
- zablokowany wolny pion powinien być mniej wartościowy,
- chroniony wolny pion powinien być bardziej wartościowy,
- connected passers powinny być premiowane.

---

### 5. [x] Dodać bishop pair

**Cel:** premiować trwałą przewagę dwóch gońców.

To jedna z najprostszych heurystyk o bardzo dobrym stosunku efekt/koszt.

**Zakres implementacji:**

- dodać metodę np. `__evaluate_minor_piece_features(board)`,
- jeśli strona ma co najmniej dwa gońce, bonus np. `0.25–0.40`,
- opcjonalnie większy bonus w otwartych pozycjach,
- opcjonalnie lekko większy w middlegame/endgame niż w zamkniętym debiucie.

**Oczekiwany efekt:**

- lepsze decyzje przy wymianach lekkich figur,
- mniejsze oddawanie pary gońców bez rekompensaty,
- bardziej naturalna ocena otwartych pozycji.

**Ryzyko/koszt:** niskie.

**Testy:**

- pozycja z parą gońców vs goniec + skoczek przy równym materiale powinna być lekko lepsza dla strony z parą gońców,
- bonus nie powinien być większy niż typowa wartość piona,
- w zamkniętej pozycji bonus może być mniejszy.

---

### 6. [x] Dodać aktywność wież: otwarte linie, półotwarte linie, 7. rząd

**Cel:** poprawić ocenę jednej z najważniejszych figur w grze środkowej i końcówce.

Obecne PST wieży częściowo to próbuje robić, ale to za mało.

**Zakres implementacji:**

Dodać heurystykę:

```text
wieża na otwartej linii: bonus
wieża na półotwartej linii: mniejszy bonus
wieża na 7. rzędzie dla białych / 2. rzędzie dla czarnych: bonus
dwie wieże na tej samej otwartej linii: opcjonalny bonus
```

**Oczekiwany efekt:**

- lepsze ustawianie wież,
- większe wykorzystanie otwartych linii,
- lepsza gra przeciw osłabionym pionom.

**Ryzyko/koszt:** niskie–średnie — trzeba uważać, żeby nie dublować nadmiernie z mobilnością.

**Testy:**

- wieża na otwartej linii powinna być oceniona lepiej niż za własnym pionem,
- wieża na półotwartej linii powinna mieć mniejszy bonus niż na otwartej,
- biała wieża na 7. rzędzie i czarna na 2. rzędzie powinny być premiowane symetrycznie.

---

### 7. [x] Poprawić `__evaluate_mobility_and_activity`

**Cel:** ograniczyć fałszywe bonusy mobilności.

Obecnie:

```python
mobility = len(board.attacks(sq))
```

To liczy pseudo-ataki, również pola zajęte przez własne figury. Figura zamknięta za własnymi pionami może więc dostać zbyt dobry wynik.

**Zakres implementacji:**

- wykluczyć własne figury:

```python
mobility = len(board.attacks(sq) & ~board.occupied_co[color])
```

- osobno liczyć kontrolę centrum:

```python
central_control = len(board.attacks(sq) & central_squares)
```

- rozważyć nieco mniejsze wagi mobilności po dodaniu nowych heurystyk,
- opcjonalnie karać pola kontrolowane przez piony przeciwnika.

**Oczekiwany efekt:**

- aktywne figury będą oceniane lepiej,
- zablokowane figury gorzej,
- mniejszy szum w ocenie pozycji zamkniętych.

**Ryzyko/koszt:** średnie — może zmienić dużo ocen i wymaga ponownego dostrojenia wag.

**Testy:**

- goniec zamknięty za własnymi pionami powinien mieć niską mobilność,
- skoczek w centrum powinien mieć lepszą ocenę niż na brzegu,
- hetman z dużą liczbą realnych pól powinien dostać bonus, ale nie dominować nad materiałem.

---

### 8. [x] Dodać threats / hanging pieces

**Cel:** poprawić ocenę pozycji taktycznych, szczególnie na granicy horyzontu wyszukiwania.

Masz już quiescence search i SEE w minimaxie, więc nie trzeba robić bardzo ciężkiej heurystyki. Wystarczy lekka statyczna ocena wiszących figur.

**Zakres implementacji:**

Dodać wykrywanie:

- figura atakowana i niebroniona,
- figura atakowana przez pion,
- ciężka figura atakowana przez lekką,
- hetman/wieża pod atakiem i bez wystarczającej obrony.

Przykład koncepcyjny:

```text
jeśli figura jest atakowana przez przeciwnika
i nie jest broniona przez swoją stronę:
    kara zależna od wartości figury
```

**Oczekiwany efekt:**

- mniej podstawiania figur,
- lepsza statyczna ocena pozycji z groźbami,
- lepsze decyzje na małych głębokościach.

**Ryzyko/koszt:** średnie — możliwe podwójne liczenie taktyki razem z search/QS.

**Testy:**

- wiszący skoczek powinien pogarszać ocenę,
- broniony skoczek pod atakiem powinien mieć dużo mniejszą karę,
- hetman atakowany przez gońca/skoczka powinien generować wyraźną karę,
- pozycja z taktyczną pułapką nie powinna być oceniana ekstremalnie bez searcha.

---

### 9. [x] Dodać aktywność króla w końcówce

**Cel:** uzupełnić osłabione `king_safety` w endgame pozytywną heurystyką.

Po sfazowaniu king safety król w końcówce nie będzie już mocno karany, ale warto go aktywnie premiować.

**Zakres implementacji:**

Dodać osobną metodę np. `__evaluate_endgame_king_activity(board, phase)`.

Premiować:

- króla bliżej centrum,
- króla bliżej własnych wolnych pionów,
- króla bliżej pionów przeciwnika,
- króla bliżej strefy promocji przeciwnika, jeśli trzeba zatrzymać passed pawna.

Skalowanie:

```python
endgame_weight = 1.0 - phase
```

**Oczekiwany efekt:**

- lepsze końcówki króle+piony,
- mniej pasywnej gry królem,
- lepsze wykorzystywanie przewagi materialnej w końcówkach.

**Ryzyko/koszt:** średnie — trzeba uważać, żeby król nie szedł do centrum w pozycjach z hetmanami.

**Testy:**

- w końcówce król w centrum powinien być lepszy niż w rogu,
- w middlegame ta heurystyka powinna mieć prawie zerowy wpływ,
- król blisko wolnego piona powinien poprawiać ocenę.

---

### 10. [x] Ujednolicić wagi i przygotować mały zestaw testów regresyjnych FEN

**Cel:** uniknąć chaosu po dodaniu wielu heurystyk.

Po dodaniu kilku bonusów istnieje ryzyko, że ocena pozycyjna zacznie dominować nad materiałem. Dlatego trzeba uporządkować skale.

**Zakres implementacji:**

- przenieść wagi do stałych klasowych, np.:

```python
BISHOP_PAIR_BONUS = 0.30
ROOK_OPEN_FILE_BONUS = 0.25
ROOK_SEMI_OPEN_FILE_BONUS = 0.15
ROOK_SEVENTH_RANK_BONUS = 0.25
```

- sprawdzić, czy suma bonusów pozycyjnych nie daje absurdalnych wyników,
- przygotować mały zestaw pozycji testowych:
  - start,
  - symetria,
  - pozycja po roszadzie,
  - przewaga piona,
  - przewaga figury,
  - wolny pion na 6./7. rzędzie,
  - odsłonięty król,
  - końcówka z aktywnym królem,
  - wisząca figura,
  - wieża na otwartej linii.

**Oczekiwany efekt:**

- stabilna ewaluacja,
- łatwiejsze dalsze strojenie,
- mniejsze ryzyko regresji.

**Ryzyko/koszt:** średnie — samo strojenie wymaga iteracji.

**Testy:**

- startowa pozycja około `0.0`,
- przewaga piona około `+1.0`,
- przewaga lekkiej figury około `+3.0`,
- bonusy pozycyjne nie powinny regularnie przebijać dużej straty materiału,
- symetryczne pozycje powinny mieć przeciwne znaki.

---

## Rekomendowana kolejność wdrażania

Najpierw wdrożyłbym tylko fundamenty:

1. [x] PST orientation.
2. [x] Pominięcie materialnej wartości króla.
3. [x] Fazowanie king safety.
4. [x] Lepsze passed pawns.
5. [x] Bishop pair.
6. [x] Rook files / 7th rank.
7. [x] Mobility cleanup.
8. [x] Hanging pieces.
9. [x] Endgame king activity.
10. [x] Tuning wag i testy regresyjne.

Jeśli chcesz robić to etapami, najlepszy pierwszy pakiet to:

```text
1 + 2 + 3 + 7
```

czyli najpierw poprawić błędy i szum w obecnych heurystykach, a dopiero potem dodawać nowe.

