# Sprawozdanie: rozpoznawanie odręcznej cyfry z obrazu 8×8

Repozytorium: https://github.com/digitalStyx/mio-agents-ml

Narzędzie: Claude Code 2.1.330, model Claude Opus 5.5 (`claude-opus-5-5`), poziom rozumowania medium (zmienna środowiskowa `CLAUDE_EFFORT=medium`).

Zadanie: lokalny prototyp, który z pliku PNG 8×8 z jedną odręczną cyfrą (format UCI optdigits) zwraca cyfrę 0–9. Wywołanie z terminala, bez etykiety. Działa na CPU, w `.venv` z Pythonem 3.13 i scikit-learn. Podział danych bez zmian: train (600) do uczenia, validation (200) do wyboru ustawień, test (200) do jednej oceny. Plan: `project/plan.md`, instrukcja uruchomienia: `README.md`, podsumowanie pracy: `project/results/podsumowanie.md`.

## 1. Podejście i alternatywa

**Wejście:** 64 piksele wiersz po wierszu, podzielone przez 255. Bez standaryzacji, wyśrodkowywania i augmentacji, bo obrazy są już znormalizowane w źródle. Ta sama funkcja (`project/digits/data.py`) obsługuje uczenie, ocenę i CLI. Odrzuca każdy plik inny niż PNG w trybie `L` 8×8.

**Podejście główne:** SVM z jądrem RBF. Wybrano go, bo mierzy podobieństwo całych obrazów, ma dwa parametry i jest standardowo mocny na tych danych.

**Alternatywa:** kNN. To najprostszy sensowny model, a jego decyzje da się wytłumaczyć sąsiadami z train.

**Punkty odniesienia:** klasa najczęstsza i najbliższy centroid.

Reguła wyboru została zapisana w planie przed uruchomieniem:
- najwięcej trafień na validation;
- przy remisie wariant gładszy (mniejsze `C`, potem mniejsze `gamma`; większe `k`);
- kNN zastępuje SVM tylko wtedy, gdy wygrywa o więcej niż 2 obrazy.

Wyniki na validation (`project/results/validation_search.csv`):

| Model | Warianty | Najlepszy | Trafienia |
|---|---|---|---|
| klasa najczęstsza | 1 | – | 20/200 (10,0%) |
| najbliższy centroid | 1 | – | 178/200 (89,0%) |
| kNN | 10 | k=1 i k=3 (uniform/distance), wybrano k=3 uniform | 193/200 (96,5%) |
| **SVM RBF** | 24 | **C=10 i C=100 przy gamma=0,3, wybrano C=10** | **196/200 (98,0%)** |

Plan przewidywał siatkę 20 wariantów SVM. Najlepsze `gamma=0,3` leżało na jej brzegu, więc zgodnie z planem dodano `gamma=1,0` (4 warianty). Te warianty dały 167–189 trafień, więc wybór się nie zmienił. kNN miał 3 trafienia mniej, więc według reguły wybrano SVM `C=10, gamma=0,3`. Model uczono tylko na train (`project/models/model.joblib`, metadane w `model.json`).

## 2. Organizacja współpracy

Przebieg sesji:
1. Rozpoznanie danych i środowiska bez pisania i instalowania. Sprawdzono sumy z manifestu, format PNG, liczebności i duplikaty oraz brak pakietów w systemowym Pythonie 3.15.
2. Ustalenia użytkownika: wejście tylko w formacie optdigits, brak progu minimalnego wyniku, brak odmowy odpowiedzi, stały podział danych.
3. `plan.md` bez kodu.
4. Główne zlecenie.
5. Zatrzymanie przed testem i akceptacja.
6. Poprawka marginesu.
7. Sprawozdanie.

Główne zlecenie (dosłownie):

```text
teraz glowne zadanie. zrob wg planu dzialajacy przebieg: uczenie, wybor ustawien na validation, zapisany model w models/, predict ktory bierze sciezke do png i wypisuje cyfre bez uczenia i bez etykiety, ocena na tescie z raportem, przewidywaniami i plansza pomylek w results/. do tego README w lab5/: srodowisko, gdzie sa dane, jak odpalic calosc od zera i gdzie sa wyniki. sprawozdania nie pisz.

pracujesz tylko w lab5/, data/ nie ruszasz, nic nie commitujesz.

sprawdzenia zapisz jako testy zeby szly jedna komenda. oczekiwane wyniki bierz z planu, opisu danych i manifestu a nie z tego co zwraca kod. minimum:
- zly plik (inny rozmiar, kolor, nie png) konczy sie bledem a nie cyfra
- uczenie i wybor ustawien nie czytaja test.csv
- predict z terminala na zapisanym modelu daje te same cyfry co model przed zapisem
- plik z przewidywaniami ma 200 wierszy z image_id z test.csv, a przeliczenie zapisanym modelem daje dla kazdego id te sama cyfre
- liczby w raporcie zgadzaja sie z plikiem przewidywan

bledy w kodzie i ustawienia poprawiaj sam, tylko zapisuj co zmieniles. wroc do mnie jak cos wymaga odejscia od planu albo pakietu spoza planu.

jedno zatrzymanie obowiazkowe: przed ocena na tescie pokaz wyniki wszystkich wariantow na validation, wybrany model i pomylki na validation, i czekaj na ok. test ma byc raz.

koniec jest wtedy jak testy przechodza, calosc da sie odpalic wg README na swiezym srodowisku, a w results/ jest krotkie podsumowanie: co zrobione, co odbiega od planu, czego nie sprawdzono.
```

**Dlaczego taki zakres.** Podejście, siatki ustawień i regułę wyboru ustalono w planie wcześniej, więc decyzje w trakcie były mechaniczne i nie wymagały konsultacji. Cały przebieg zlecono naraz. Zamiast nadzoru krok po kroku zastosowano trzy zabezpieczenia:
- granice: tylko `lab5/`, `data/` bez zmian, bez commitów, powrót przy odejściu od planu lub pakiecie spoza planu;
- testy z oczekiwaniami spoza kodu (README danych, manifest, plan);
- jasny warunek końca.

**Dlaczego jedno zatrzymanie.** Ocena na teście to jedyny krok, którego nie da się uczciwie powtórzyć. Wszystko przed nią jest odtwarzalne i sprawdzane testami. Zatrzymanie pozwoliło zatwierdzić wybór na podstawie validation, zanim ktokolwiek zobaczył wynik testu. Test uruchomiono raz. Skrypt `digits.evaluate` odmawia ponownego uruchomienia, gdy raport istnieje.

**Moment, w którym wynik sprawdzenia zmienił działanie: margines.** Po ocenie użytkownik zauważył, że kolumna `margin` w `test_predictions.csv` prawie się nie zmienia i dla pomyłek wynosi tyle samo co dla poprawnych. W pliku: min 0,973, mediana 1,066, pomyłki 0,997, 1,037 i 1,044.

Diagnoza, przeprowadzona tylko na validation: `decision_function` w kształcie `ovr` to liczba głosów 45 klasyfikatorów par plus pewność ściśnięta do (−1/3, 1/3). Zwycięzca ma prawie zawsze 9 głosów, a druga cyfra 8 (195 z 200 obrazów validation), więc różnica wynosi ok. 1. Przy starym marginesie 53% poprawnych przewidywań miało niższą wartość niż przeciętna pomyłka. Przy wyniku klasyfikatora pary (przewidziana, druga) ten odsetek spadł do 7%.

Wdrożono propozycję zaakceptowaną przez użytkownika:
- nowa kolumna `margin_ovo` odczytana z zapisanego modelu, bez ponownego uczenia;
- pliki validation przeliczone;
- do `test_predictions.csv` dopisana kolumna skryptem `digits/add_margin_ovo.py`, który przerywa pracę, gdy przeliczona cyfra różni się od zapisanej;
- nowy test niezależnie przeliczający margines.

Sprawdzono sumami SHA-256, że `model.joblib`, `model.json`, `test_report.json` i `test_report.md` się nie zmieniły. Margines zmienił opis jednej pomyłki: 3 → 9 przestała być „nietypowym stylem” i okazała się przypadkiem granicznym (0,09).

**Błąd w trakcie: nadpisany README.** Plik `lab5/README.md` (opis danych dostarczony z zadaniem) został przy pisaniu instrukcji nadpisany. `lab5/` nie jest śledzony w git. Plik odtworzono z treści odczytanej na początku sesji. Rozmiar zgadza się z pierwotnym (2106 B). Po sesji porównano odtworzoną część z treścią pliku wypisaną na początku sesji (zapis w `sesje/sesja-1.jsonl`): jest identyczna. Instrukcję dopisano pod oryginalnym tekstem.

**Pozostałe zdarzenia:**
- Test „uczenie nie czyta `test.csv`” (hak audytu w osobnym procesie) sprawdzono na kopii kodu celowo czytającej `test.csv`. Ta kopia test oblała.
- Kompletność README sprawdzono w świeżej kopii z nowym `.venv`. Uczenie dało identyczne `model.json` (poza datą) i pliki validation. Po skopiowaniu wyników testu wszystkie testy przechodzą. Oceny na teście nie uruchomiono tam drugi raz.

## 3. Wyniki i pomyłki

Źródło: `project/results/test_report.md` i `test_report.json`, przewidywania w `test_predictions.csv`.

| Model | Test | 95% CI (Wilson) |
|---|---|---|
| **SVM C=10, gamma=0,3** | **197/200 (98,5%)** | 95,7–99,5% |
| najbliższy centroid | 186/200 (93,0%) | 88,6–95,8% |
| klasa najczęstsza | 20/200 (10,0%) | 6,6–14,9% |

- Na validation było 196/200 (98,0%). Validation pochodzi od tych samych autorów co train, a test od innych, a mimo to wynik się nie obniżył.
- Cyfry 1, 3 i 7 mają 19/20 trafień (CI 76,4–99,1%), pozostałe 20/20.

**Najczęściej mylone cyfry.** Każda para wystąpiła raz: 1 → 6, 3 → 9, 7 → 9. Powtarza się kierunek pomyłek: **9 jest przewidziana w 2 z 3 pomyłek na teście** i w 3 z 4 na validation (1, 2 i 5 → 9, `validation_errors.csv`).

**Błędne obrazy z testu** (`test_errors.csv`, plansza `test_errors.png` z 3 najbliższymi obrazami z train):

| image_id | Pomyłka | Druga kandydatka | `margin_ovo` | 3 najbliższe z train | Omówienie |
|---|---|---|---|---|---|
| digit-826cb4bf68f5 | 1 → 6 | 1 | 0,029 | 1, 6, 6 | Jedynka z szeroką podstawą. Obraz leży praktycznie na granicy 1/6, a prawdziwa cyfra jest drugą kandydatką. |
| digit-4c5c0534e4b8 | 3 → 9 | 2 | 0,085 | 8, 3, 3 | Trójka z domkniętą górną częścią. Przypadek graniczny: dwa z trzech sąsiadów to trójki. |
| digit-09e2a05d6bd2 | 7 → 9 | 4 | 0,578 | 9, 9, 9 | Siódemka z zamkniętą górą. Model jest pewny, 7 nie jest nawet drugą kandydatką, a wszyscy trzej najbliżsi sąsiedzi z train to dziewiątki. |

Żaden obraz nie wygląda na błąd etykiety. Dwie z trzech pomyłek mają `margin_ovo` poniżej 0,1. Poniżej tej wartości są tylko 2 ze 197 poprawnych przewidywań na teście (mediana dla poprawnych: 0,87). Pomyłek z testu nie użyto do zmiany modelu.

## 4. Ocena, ograniczenia, czego nie sprawdzono

**Ocena.** W uzgodnionym zakresie (obrazy w formacie optdigits, bez progu i bez odmowy) prototyp można przyjąć.
- **Uczciwość pomiaru:** test policzono raz, na autorach spoza train, a wybór dokonano wcześniej na validation według reguły z planu.
- **Wynik:** 98,5% z dolną granicą przedziału 95,7%. Na tych samych obrazach model ma o 11 trafień więcej niż najbliższy centroid.
- **Testy:** 47 testów przechodzi jedną komendą (`python -m pytest`). Obejmują wszystkie sprawdzenia ze zlecenia oraz zgodność wyboru z regułą i marginesu z modelem.

**Ograniczenia:**
- Wynik dotyczy tylko znaków wyciętych i znormalizowanych jak w optdigits. Nie mówi nic o innym sposobie wycinania ani o całych formularzach.
- 200 obrazów testowych to przedział o szerokości ok. 4 p.p. dla całości i kilkunastu p.p. dla pojedynczej cyfry. Różnica 3 obrazów mieści się w szumie.
- Model zawsze zwraca cyfrę, także dla pustego pola czy litery. Tak ustalono.
- `margin_ovo` nie jest skalibrowanym prawdopodobieństwem. Przy 4 pomyłkach na validation nie wyznaczono progu do ręcznej weryfikacji.
- Model zapisany przez joblib wymaga tej samej wersji scikit-learn (1.9.1) i zaufanego źródła.
- `test_predictions.csv` zawiera starą, mylącą kolumnę `margin`, zostawioną, żeby nie zmieniać pliku z jednorazowej oceny. Nowy przebieg od zera zapisze tylko `margin_ovo`.

**Czego nie sprawdzono:**
- Obrazów od zespołu, czyli spoza dostarczonego zbioru.
- Liczby autorów w teście: manifest nie zawiera ich identyfikatorów.
- Innych systemów niż macOS arm64 ani innych wersji Pythona i pakietów.
- Ponownego `digits.evaluate` w świeżym środowisku. Celowo pominięto, bo test miał być liczony raz.
- Zgodności odtworzonej części `README.md` z plikiem z paczki zadania. Porównano ją tylko z treścią wypisaną na początku sesji.
