# Podsumowanie

## Co zrobiono

- **Środowisko:** `.venv` na Pythonie 3.13.17 z pakietami z planu (numpy, scikit-learn, Pillow, pytest), wersje przypięte w `requirements.txt`. Działa na CPU, bez usług zewnętrznych.
- **Przygotowanie obrazu:** jedna funkcja (`digits/data.py`) dla uczenia, oceny i CLI. Przyjmuje tylko PNG, tryb `L`, 8×8. Zamienia obraz na 64 wartości wiersz po wierszu, podzielone przez 255, bez standaryzacji. Każdy inny plik kończy się błędem.
- **Wybór ustawień:** wszystkie modele uczone na train (600), porównane na validation (200). Wszystkie warianty są w `validation_search.csv`.

  | Model | Najlepsze ustawienia | Validation |
  |---|---|---|
  | klasa najczęstsza | – | 20/200 (10,0%) |
  | najbliższy centroid | – | 178/200 (89,0%) |
  | kNN | k=3, uniform | 193/200 (96,5%) |
  | **SVM RBF (wybrany)** | **C=10, gamma=0,3** | **196/200 (98,0%)** |

  Według reguły z planu wygrał SVM. kNN nie wyprzedził go o więcej niż 2 obrazy, bo miał 3 trafienia mniej. Między C=10 i C=100 był remis (196), a reguła remisu wybrała mniejsze C.
- **Model końcowy:** uczony tylko na train i zapisany w `models/model.joblib`. Metadane są w `models/model.json`: ustawienia, opis wejścia, wersje bibliotek, sumy SHA-256 train/validation, wynik na validation.
- **Ocena na teście:** uruchomiona raz, po akceptacji wyboru.

  | Model | Test | 95% CI Wilsona |
  |---|---|---|
  | **SVM C=10, gamma=0,3** | **197/200 (98,5%)** | 95,7–99,5% |
  | najbliższy centroid | 186/200 (93,0%) | 88,6–95,8% |
  | klasa najczęstsza | 20/200 (10,0%) | 6,6–14,9% |

  Trzy cyfry mają po jednej pomyłce (1, 3 i 7, każda 19/20), pozostałe 20/20. Szczegóły są w `test_report.md`.
- **Pomyłki** (pełna ocena na planszach `validation_errors.png` i `test_errors.png`):

  `margin_ovo` to wynik klasyfikatora pary (przewidziana, druga kandydatka). Na validation mediana dla poprawnych wynosi 0,96, a 5% poprawnych ma poniżej 0,35. Na teście mediana wynosi 0,87, a 5% poprawnych ma poniżej 0,41.

  | Zbiór | Pomyłka | Druga | `margin_ovo` | Kategoria | Opis |
  |---|---|---|---|---|---|
  | validation | 4 → 6 | 1 | 0,59 | znak niejednoznaczny także dla człowieka | czwórka z domkniętą dolną pętlą, 3 najbliższe z train to szóstki; model jest pewny |
  | validation | 1 → 9 | 3 | 0,38 | nietypowy styl | jedynka z długim daszkiem, prawdziwa cyfra nie jest nawet drugą kandydatką |
  | validation | 2 → 9 | 2 | 0,31 | nietypowy styl, przypadek graniczny | dwójka z pętlą u góry |
  | validation | 5 → 9 | 5 | 0,29 | przypadek graniczny | piątka z prawie zamkniętą górą |
  | test | 7 → 9 | 4 | 0,58 | nietypowy styl | siódemka z zamkniętą górą, 3 najbliższe z train to dziewiątki; model jest pewny, a 7 nie jest drugą kandydatką |
  | test | 3 → 9 | 2 | 0,09 | przypadek graniczny | trójka z domkniętą górną częścią; po poprawce marginesu widać, że to przypadek tuż przy granicy (wcześniej opisany jako „nietypowy styl”) |
  | test | 1 → 6 | 1 | 0,03 | przypadek graniczny | jedynka z szeroką podstawą, obraz praktycznie na granicy 1/6 |

  Żadna pomyłka nie wygląda na błąd etykiety. Pomyłek z testu nie użyto do zmiany modelu.
- **CLI:** `python -m digits.predict plik.png` wypisuje samą cyfrę. Przy złym pliku kończy się kodem 2, przy braku modelu kodem 3.
- **Testy:** `python -m pytest` uruchamia 47 testów, wszystkie przechodzą. Oczekiwane wartości wzięto z README danych, manifestu i planu. Sprawdzają:
  - odrzucanie złych plików przez funkcję i przez CLI: 11 przypadków, m.in. inny rozmiar, RGB, alfa, paleta, 16 bitów, JPEG/BMP, tekst, pusty plik, brak pliku;
  - że uczenie nie otwiera `test.csv` ani żadnego obrazu testowego. Pełne uczenie działa w osobnym procesie z hakiem audytu, który rejestruje każdy otwarty plik. Sprawdzono też, że kopia kodu celowo czytająca `test.csv` ten test oblewa;
  - że CLI na zapisanym modelu daje te same cyfry co model w pamięci przed zapisem;
  - że plik przewidywań ma 200 wierszy z `image_id` z `test.csv`, a ponowne przeliczenie zapisanym modelem daje dla każdego id tę samą cyfrę;
  - że accuracy, CI, macierz pomyłek, wyniki dla cyfr, pary pomyłek, raport `.md` i plik pomyłek zgadzają się z plikiem przewidywań (Wilson liczony w teście niezależnie);
  - że wybór ustawień zgadza się z regułą z planu, odtworzoną w teście z `validation_search.csv`, i że najlepszy SVM nie leży na brzegu siatki;
  - że `margin_ovo` w plikach validation i testu zgadza się z wynikiem klasyfikatora pary z zapisanego modelu, przeliczonym w teście niezależnie, a druga kandydatka to najwyższy wynik `ovr` poza cyfrą przewidzianą.
- **README:** instrukcja jest w `lab5/README.md`, pod oryginalnym opisem danych.

## Odstępstwa od planu

- **Siatka SVM poszerzona o `gamma=1,0`.** Najlepsze `gamma=0,3` wypadło na brzegu siatki. To przewidziana w planie reguła, a nie zmiana decyzji. Wariant `gamma=1,0` wypadł gorzej (189 trafień), więc wybór się nie zmienił. Poszerzenie jest zapisane w kolumnie `round` w `validation_search.csv`.
- **Układ katalogów.** Kod jest w `project/digits/` zamiast `src/` (o `src/` mówiłem tylko w rozmowie, nie w `plan.md`). Dzięki temu `python -m digits.predict` działa z `project/` bez instalacji pakietu.
- **Wyciszone ostrzeżenie `NearestCentroid`.** Dotyczy pikseli stałych w klasie na brzegach obrazu i nie wpływa na wynik.
- **Kody wyjścia CLI** (2 dla złego pliku, 3 dla braku modelu) i **blokada ponownej oceny** (także gdy istnieje samo `test_predictions.csv`) to doprecyzowania, których nie było w planie.
- **README.** Na początku przez pomyłkę nadpisałem istniejący `lab5/README.md` (opis danych). lab5 nie jest śledzony w git, więc przywróciłem go z treści odczytanej na początku pracy. Rozmiar zgadza się co do bajtu (2106 B). Instrukcję dopisałem pod nim, a oryginalny tekst został bez zmian.

- **Poprawka marginesu po ocenie na teście.** Kolumna `margin` z pierwszej wersji była różnicą wyników `ovr` z SVC. Te wyniki to liczba głosów 45 klasyfikatorów par plus pewność ściśnięta do (−1/3, 1/3). Zwycięzca ma prawie zawsze 9 głosów, a druga cyfra 8 (195 z 200 obrazów validation), więc margines wynosił ok. 1 dla prawie wszystkich obrazów, także dla pomyłek. Na validation 53% poprawnych miało mniejszy margines niż przeciętna pomyłka. Nowa kolumna `margin_ovo` to wynik klasyfikatora pary (przewidziana, druga), odczytany z zapisanego modelu w kształcie `ovo`, bez ponownego uczenia. Przy niej ten odsetek spada do 7%. Zmiany:
  - `digits/analysis.py` (`top2`, `pair_margin`), `train.py` i `evaluate.py` zapisują `margin_ovo` zamiast `margin`;
  - pliki validation przeliczono z zapisanego modelu. Są identyczne z tymi, które daje świeże `digits.train` nowym kodem (sprawdzone w osobnej kopii);
  - do `test_predictions.csv` dopisano kolumnę `margin_ovo` skryptem `digits/add_margin_ovo.py`, który wczytuje zapisany model i przerywa, gdy cyfra lub druga kandydatka wyszłaby inna niż w pliku. Stara kolumna `margin` i pozostałe kolumny są bez zmian. `test_errors.csv` i `test_errors.png` utworzono od nowa, z marginesem na planszy;
  - `model.joblib`, `model.json`, `test_report.json` i `test_report.md` mają te same sumy SHA-256 co przed zmianą. To nie była ponowna ocena: model przeliczył wyniki na obrazach testowych, ale żadna decyzja ani liczba w raporcie od tego nie zależy;
  - opis pomyłki 3 → 9 na teście zmieniono z „nietypowy styl” na „przypadek graniczny” (margines 0,09).

## Czego nie sprawdzono

- **Margines nie jest skalibrowanym prawdopodobieństwem.** Na validation są tylko 4 pomyłki, więc nie wyznaczono progu, poniżej którego warto by kierować obraz do ręcznej weryfikacji. Odmowy odpowiedzi i tak nie robimy, ale jeśli się pojawi, próg trzeba ustalić na osobnych danych.

- **Druga ocena testu w świeżym środowisku.** Zrobiłem świeżą kopię kodu i danych, nowy `.venv` z `requirements.txt` i przeszedłem kroki README. Uczenie dało identyczne `model.json` (poza datą), `validation_search.csv` i `validation_predictions.csv`. Predict zadziałał, pytest dał 36 zaliczonych i 7 pominiętych. Po skopiowaniu wyników testu z głównego przebiegu wszystkie testy przechodzą (43, a po dodaniu testów marginesu 47), czyli świeży model odtwarza wszystkie 200 przewidywań. Samego `digits.evaluate` nie uruchomiłem tam drugi raz, żeby nie liczyć testu ponownie. Sprawdziłem tylko, że odmawia uruchomienia przy istniejącym raporcie.
- **Inne systemy i wersje.** Nie sprawdzono innych systemów niż macOS arm64 ani innych wersji Pythona i pakietów.
- **Obrazy spoza dostarczonego zbioru.** Nie sprawdzono obrazów od zespołu ani znaków nietypowo wyciętych lub znormalizowanych. Wynik dotyczy tylko formatu optdigits.
- **Autorzy w teście.** Manifest nie zawiera identyfikatorów autorów, więc nie wiadomo, ilu autorów jest w teście ani czy któryś jest nadreprezentowany.
- **Niepewność wyniku.** Przy 200 obrazach testowych różnica między 98,5% a np. 97% (3 obrazy) mieści się w szumie. Przedziały dla pojedynczych cyfr (20 obrazów) mają szerokość kilkunastu punktów procentowych.
- **Wejście, które nie jest cyfrą.** Model zawsze zwraca cyfrę, także dla pustego pola czy litery. Zgodnie z ustaleniami tego nie obsługujemy.
