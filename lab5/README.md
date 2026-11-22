# Dane do Labu 5: odręcznie zapisane cyfry

Katalog `data/` zawiera 1000 obrazów PNG i odpowiadające im etykiety od `0` do `9`. Nie zawiera implementacji ani wytrenowanego modelu.

## Format

Każdy obraz ma 8 × 8 pikseli i jeden 8-bitowy kanał szarości. Ciemne tło ma wartość bliską 0, a jasny znak wartości bliższe 255. Pliki PNG zachowują układ 64 wartości źródłowego obrazu. Wartości od 0 do 16 przeliczono wzorem `round(wartość * 255 / 16)`.

Pliki `data/train.csv`, `data/validation.csv` i `data/test.csv` mają kolumny:

| Kolumna | Znaczenie |
|---|---|
| `image_id` | Stały identyfikator obrazu. |
| `image_path` | Ścieżka do PNG względem katalogu `data/`, na przykład `images/digit-….png`. |
| `label` | Prawidłowa cyfra od 0 do 9. |

Zbiór do uczenia zawiera 600 obrazów, walidacyjny 200 i testowy 200. Każda cyfra ma odpowiednio 60, 20 i 20 przykładów. W dostarczonych podziałach identyfikatory i obrazy nie powtarzają się między zbiorami.

## Pochodzenie i zakres danych

Źródło: E. Alpaydin i C. Kaynak, *Optical Recognition of Handwritten Digits* (1998), [UCI Machine Learning Repository](https://doi.org/10.24432/C50P49). Dane są udostępnione na licencji [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

To przygotowany podzbiór dydaktyczny. Obrazy do uczenia i walidacji wybrano z oryginalnego `optdigits.tra`, a do końcowej oceny z `optdigits.tes`. Oryginalny zbiór rozdziela te dwa pliki według autorów pisma. Podział na uczenie i walidację w tym zadaniu nie rozdziela dodatkowo autorów.

Dokonane zmiany: wybór 100 obrazów każdej cyfry, przygotowanie podziałów i identyfikatorów oraz zapis macierzy 8 × 8 jako PNG. Losowanie wykonano z ziarnem `20261114`. `data/manifest.json` zawiera źródło, sumy kontrolne, pozycje wybranych rekordów w plikach źródłowych i przypisanie do zbiorów.

Obrazy są już wyodrębnionymi, przygotowanymi znakami. Wynik na tych danych nie potwierdza rozpoznawania całych formularzy, zdjęć z telefonu ani skuteczności na każdym stylu pisma.

---

# Prototyp: rozpoznawanie cyfry z obrazu 8×8

Ta część opisuje rozwiązanie. Część powyżej to oryginalny opis danych, pozostawiony bez zmian.

Podajesz ścieżkę do pliku PNG 8×8 z jedną odręczną cyfrą i dostajesz cyfrę 0–9. Model to SVM z jądrem RBF (scikit-learn), uczony na 600 obrazach z train. Działa lokalnie na CPU, bez usług zewnętrznych.

## Co gdzie jest

| Ścieżka | Zawartość |
|---|---|
| `data/` | dane wejściowe (opisane wyżej, nie są modyfikowane) |
| `project/plan.md` | plan: podejście, przygotowanie obrazu, wybór ustawień, ocena |
| `project/digits/` | kod: `data.py` (wczytywanie i kontrola formatu), `train.py`, `predict.py`, `evaluate.py`, `analysis.py`, `add_margin_ovo.py` (jednorazowe uzupełnienie wyników, zob. niżej) |
| `project/tests/` | testy (pytest) |
| `project/models/` | zapisany model: `model.joblib` i metadane `model.json` |
| `project/results/` | wyniki: przegląd ustawień, przewidywania, raport z testu, plansze pomyłek, `podsumowanie.md` |
| `project/requirements.txt` | przypięte wersje pakietów |

Kod czyta dane z `data/` (ścieżka liczona względem `project/`, można ją zmienić przez `--data-dir`). Przy każdym wczytaniu sprawdza sumy SHA-256 plików CSV z `data/manifest.json`. Podział jest stały: train do uczenia, validation do wyboru ustawień, test do jednej oceny na końcu.

## Środowisko

- Python 3.13 (sprawdzone na 3.13.17, macOS arm64), tylko CPU.
- Pakiety z `project/requirements.txt`: numpy 2.5.5, scikit-learn 1.9.1, Pillow 13.0.0, pytest 9.2.0 i ich zależności.

Wszystkie polecenia poniżej uruchamia się z katalogu `lab5/project/`.

```bash
cd lab5/project
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Całość od zera

```bash
# 1. Uczenie na train i wybór ustawień na validation (kilkadziesiąt sekund).
#    Zapisuje models/model.joblib, models/model.json i results/validation_*.
.venv/bin/python -m digits.train

# 2. Jednorazowa ocena na teście. Zapisuje results/test_*.
#    Odmawia uruchomienia, jeśli results/test_report.json lub test_predictions.csv już istnieje.
.venv/bin/python -m digits.evaluate

# 3. Testy, jedna komenda (ok. 30 s).
.venv/bin/python -m pytest
```

Przed krokiem 2 testy dotyczące wyników z testu są pomijane z komunikatem `najpierw uruchom python -m digits.evaluate`. Po kroku 2 wszystkie powinny przechodzić.

Powtórzenie od zera wymaga świadomego usunięcia `models/` i `results/`. Bez tego ocena na teście się nie uruchomi, bo test ocenia się raz.

## Rozpoznawanie cyfry

```bash
.venv/bin/python -m digits.predict ../data/images/digit-2e78088d9c42.png
# 1
```

- Wypisuje samą cyfrę. Nie uczy modelu i nie potrzebuje etykiet ani plików CSV.
- Akceptuje tylko PNG 8×8 w trybie `L` (8-bitowa skala szarości). Każdy inny plik kończy się komunikatem `błąd: ...` na stderr i kodem wyjścia 2, bez cyfry na stdout. Dotyczy to innego rozmiaru, koloru, kanału alfa, 16 bitów i plików, które nie są PNG. Brak modelu daje kod 3.
- Inny model wskazuje się przez `--model ścieżka/model.joblib`.
- Model zapisano przez joblib (pickle). Wczytuj go tylko z zaufanego źródła i tą samą wersją scikit-learn. Przy innej wersji `predict` wypisze ostrzeżenie.

## Wyniki

| Plik w `project/results/` | Zawartość |
|---|---|
| `podsumowanie.md` | co zrobiono, odstępstwa od planu, czego nie sprawdzono |
| `validation_search.csv` | wszystkie warianty (punkty odniesienia, SVM, kNN) i ich trafienia na validation |
| `validation_predictions.csv`, `validation_errors.csv`, `validation_errors.png` | przewidywania i pomyłki wybranego modelu na validation |
| `test_report.md`, `test_report.json` | ocena na teście: accuracy z 95% CI Wilsona, trafność dla każdej cyfry, macierz pomyłek, punkty odniesienia |
| `test_predictions.csv` | 200 przewidywań: `image_id`, prawdziwa cyfra, przewidziana, druga kandydatka, `margin_ovo`; ten plik zawiera też starą kolumnę `margin` z pierwszej wersji, której nie należy używać (zob. niżej) |
| `test_errors.csv`, `test_errors.png` | pomyłki na teście z 3 najbliższymi obrazami z train |

### Kolumna `margin_ovo`

`margin_ovo` to wynik klasyfikatora SVM dla pary (cyfra przewidziana, druga kandydatka), dodatni na korzyść przewidzianej. Mała wartość (np. poniżej 0,3) oznacza obraz blisko granicy między dwiema cyframi. Na validation mediana dla poprawnych wynosi 0,96, a pomyłki mają 0,29–0,59. Na teście dwie z trzech pomyłek mają 0,03 i 0,09.

W `test_predictions.csv` jest też stara kolumna `margin`, zapisana przy pierwszej ocenie. To różnica wyników `ovr`, czyli głównie różnica liczby głosów, więc prawie zawsze wynosi ok. 1 i nie pokazuje przypadków granicznych. Została tylko po to, żeby nie zmieniać pliku z jednorazowej oceny. Kolumnę `margin_ovo` dopisał do niego z zapisanego modelu `python -m digits.add_margin_ovo`; przewidziane cyfry i raport się nie zmieniły. Nowy przebieg od zera zapisuje już tylko `margin_ovo`.

Wynik na teście: **197/200 (98,5%, 95% CI 95,7–99,5%)**. Dla porównania najbliższy centroid osiąga 93,0%, a klasa najczęstsza 10%.
