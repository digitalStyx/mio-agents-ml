# Plan: rozpoznawanie odręcznej cyfry z obrazu 8×8

## 1. Cel i ustalenia

Lokalny prototyp, który z jednego pliku PNG 8×8 (jedna odręczna cyfra) zwraca cyfrę 0–9.

Ustalenia z zespołem:
- wejście to wyłącznie gotowe obrazy w formacie z `data/`: 8×8, 8-bitowa skala szarości, jasny znak na ciemnym tle, wartości z siatki `round(v*255/16)`, znak już wycięty i wyśrodkowany. Całe formularze i zdjęcia są poza zakresem;
- interfejs: terminal, podaje się ścieżkę do PNG, wynikiem jest sama cyfra. Etykieta nie jest potrzebna;
- model zawsze zwraca cyfrę. Nie ma odmowy odpowiedzi ani klasy „to nie cyfra”;
- nie ma progu minimalnej skuteczności. Wynik ma być policzony uczciwie;
- wszystko działa lokalnie na CPU, w `.venv` na Pythonie 3.13 ze scikit-learn, bez płatnych usług;
- podział danych zostaje bez zmian: train służy do uczenia, validation do wyboru ustawień, a test jest liczony raz, na końcu.

Stan danych (sprawdzony przed planowaniem):
- 1000 obrazów 8×8 L, bez duplikatów pikseli;
- sumy SHA-256 CSV i obrazów zgadzają się z `manifest.json`;
- każda cyfra ma 60/20/20 przykładów w train/validation/test.

## 2. Podejście i alternatywa

**Podejście główne: SVM z jądrem RBF** (`sklearn.svm.SVC`) na 64 wartościach pikseli.
- Jądro RBF mierzy podobieństwo całych obrazów, więc dobrze znosi drobne różnice w kształcie i przesunięcie o piksel.
- Ma dwa parametry (`C`, `gamma`), które da się przejrzeć siatką na 600 przykładach w kilka sekund.
- Na tych danych (UCI optdigits) to jedno ze standardowo najlepszych prostych podejść.
- Do przeglądu pomyłek daje wartości `decision_function`, z których widać margines między dwiema najlepszymi klasami. *Poprawka po ocenie:* margines liczy się z wyniku klasyfikatora pary tych dwóch cyfr (kształt `ovo`), a nie z różnicy wyników `ovr`, bo te ostatnie to głównie liczba głosów i prawie zawsze dają ok. 1 (zob. `results/podsumowanie.md`).

**Alternatywa do porównania: k najbliższych sąsiadów** (`KNeighborsClassifier`, odległość euklidesowa).
- Nie ma właściwego uczenia, bo przewiduje według najbardziej podobnych obrazów z train. Jest najprostszym sensownym modelem, z którym trzeba się zmierzyć.
- Każdą pomyłkę da się wprost wytłumaczyć, pokazując sąsiadów z train.
- Jest wolniejszy przy predykcji, bo porównuje z całym train, ale przy 600 obrazach nie ma to znaczenia.

**Punkty odniesienia** (to nie są kandydaci, tylko skala porównania):
- klasa najczęstsza: 10%, bo klasy są równoliczne;
- najbliższy centroid, czyli średni obraz każdej cyfry.

Jeśli SVM nie wyprzedza wyraźnie najbliższego centroidu, model niewiele wnosi.

Odrzucone na tym etapie:
- **sieci neuronowe (CNN, MLP):** przy 600 obrazach 8×8 i bez GPU nie oczekuje się przewagi, a trzeba by stroić znacznie więcej ustawień;
- **augmentacja (przesunięcia, obroty):** obrazy są już znormalizowane.

## 3. Przygotowanie obrazu

Wczytywanie i przygotowanie robi jedna funkcja, używana przy uczeniu, ocenie i w CLI. Dzięki temu nowe obrazy przechodzą dokładnie tę samą drogę co dane uczące.

1. Wczytanie PNG przez Pillow.
2. Kontrola formatu: tryb `L` (8-bit, jeden kanał) i rozmiar dokładnie 8×8. Każdy inny plik kończy się czytelnym komunikatem błędu i kodem wyjścia ≠ 0. Nie ma cichego skalowania, konwersji kolorów ani odwracania jasności. To kontrola formatu pliku, a nie klasa „nie cyfra”.
3. Zamiana na wektor 64 liczb, wiersz po wierszu (tak jak w źródle), podzielony przez 255. Wartości trafiają do [0, 1].
4. Bez standaryzacji cech. Wszystkie piksele mają tę samą skalę. Skrajne kolumny są prawie zawsze 0, więc standaryzacja sztucznie wzmocniłaby w nich pojedyncze jasne piksele.
5. Bez wyśrodkowywania, przycinania i wygładzania, bo obrazy są już przygotowane w źródle.

## 4. Wybór ustawień

Wszystkie modele uczy się tylko na train (600 obrazów) i porównuje tylko na validation (200 obrazów). `test.csv` nie jest czytany na tym etapie i pilnuje tego test w pytest.

Siatki ustawień:

| Model | Ustawienia | Liczba wariantów |
|---|---|---|
| SVM RBF | `C` ∈ {0,1; 1; 10; 100}, `gamma` ∈ {0,003; 0,01; 0,03; 0,1; 0,3} | 20 |
| kNN | `k` ∈ {1, 3, 5, 7, 9}, `weights` ∈ {uniform, distance} | 10 |

Zakres `gamma` obejmuje wartość domyślną `scale` (dla tych danych ok. 0,1). Jeśli najlepszy wynik wypadnie na brzegu siatki, siatkę poszerza się w tę stronę i zapisuje to w wynikach.

Reguła wyboru, ustalona przed uruchomieniem:
1. W każdym modelu wybiera się wariant o największej liczbie trafień na validation. Przy remisie wygrywa wariant gładszy: dla SVM mniejsze `C`, potem mniejsze `gamma`, dla kNN większe `k`, potem `uniform`.
2. Między SVM a kNN wygrywa model z większą liczbą trafień na validation. Jeśli różnica wynosi najwyżej 2 obrazy (1 p.p., czyli poziom szumu przy 200 przykładach), zostaje SVM jako podejście główne.
3. Zwycięski wariant uczy się ponownie **tylko na train** (bez łączenia z validation, zgodnie z ustalonym podziałem) i zamraża przed testem.

Zapisywane są wyniki wszystkich wariantów, a nie tylko najlepszego (`results/validation_search.csv`). Pomyłki najlepszego wariantu na validation przegląda się przed testem tak jak w części 6. Ewentualne zmiany wprowadza się tylko na podstawie validation.

Validation pochodzi od tych samych autorów co train (oba z `optdigits.tra`), więc jej wynik jest optymistyczny. Służy do wyboru, a nie do raportowania skuteczności.

## 5. Ocena na teście (raz)

Test (200 obrazów z `optdigits.tes`, inni autorzy pisma niż w train) ocenia się jednym skryptem, po zamrożeniu modelu. Skrypt odmawia ponownego uruchomienia, jeśli raport testu już istnieje. Ewentualną zmianę modelu po teście trzeba wtedy opisać wprost, a nowy wynik przestaje być niezależny.

Liczone są:
- **accuracy** z 95% przedziałem ufności Wilsona, obok punktu odniesienia 10% i wyniku najbliższego centroidu (uczonego na train). Jedna pomyłka to 0,5 p.p.;
- **trafność dla każdej cyfry**, w liczbach (np. 19/20) i z przedziałem Wilsona. Przy 20 przykładach na cyfrę różnice między cyframi są mało pewne, dlatego podaje się liczby, a nie tylko procenty;
- **macierz pomyłek 10×10** w liczbach bezwzględnych;
- **najczęstsze pary pomyłek** (prawdziwa → przewidziana);
- wynik na validation obok, dla porównania, z adnotacją, że jest optymistyczny.

Na teście ocenia się tylko zamrożony model i punkty odniesienia. Alternatywa była porównana na validation i na teście nie jest wybierana ponownie.

## 6. Oglądanie pomyłek

Dla każdego źle sklasyfikowanego obrazu (najpierw na validation, a na końcu na teście) zapisywane są:
- `image_id`, prawdziwa i przewidziana cyfra, druga najbardziej prawdopodobna cyfra oraz margines `margin_ovo`: wynik klasyfikatora SVM dla pary (przewidziana, druga), dodatni na korzyść przewidzianej. Mały margines oznacza przypadek graniczny, duży oznacza pewną pomyłkę. *(Poprawione po ocenie; pierwsza wersja liczyła różnicę wyników `ovr`, która tego nie pokazywała.)*;
- obraz w powiększeniu: zbiorcza plansza PNG (każdy obraz 8×8 powiększony bez wygładzania, z podpisem „prawdziwa → przewidziana”), generowana Pillow, bez matplotlib;
- obok każdej pomyłki 3 najbliższe obrazy z train z ich etykietami. Pokazuje to, czy znak jest nietypowy, podobny do innej cyfry, czy może ma błędną etykietę.

Każdą pomyłkę opisuje się krótko jedną kategorią: nietypowy styl pisma, znak niejednoznaczny także dla człowieka, możliwy błąd etykiety albo pomyłka bez oczywistej przyczyny. Pomyłek z testu nie używa się do poprawiania modelu.

## 7. Co zapisujemy

Model ma się dać uruchomić później i odtworzyć.

- `models/model.joblib`: wytrenowany klasyfikator. Pickle/joblib wczytuje się tylko z zaufanego źródła i tą samą wersją scikit-learn.
- `models/model.json`, metadane:
  - typ modelu i ustawienia;
  - opis przygotowania obrazu (tryb `L`, 8×8, wiersz po wierszu, /255);
  - kolejność klas;
  - wersje Pythona, numpy, scikit-learn i Pillow;
  - SHA-256 `train.csv` i `validation.csv`;
  - liczba obrazów uczących;
  - wynik na validation;
  - data uczenia.
- `requirements.txt` z przypiętymi wersjami.
- `results/validation_search.csv`: wszystkie warianty i ich wyniki na validation.
- `results/validation_errors.csv` i `results/validation_errors.png`.
- `results/test_report.json` oraz czytelny `results/test_report.md`: metryki z części 5.
- `results/test_errors.csv` i `results/test_errors.png`.

CLI przy starcie sprawdza zgodność wersji scikit-learn z `model.json` i ostrzega przy różnicy.

## 8. Kolejność prac

1. `.venv` na Pythonie 3.13 i `requirements.txt` (numpy, scikit-learn, Pillow, pytest).
2. Funkcja przygotowania obrazu, z testami formatu (zły rozmiar, zły tryb, plik, który nie jest PNG).
3. Punkty odniesienia i przegląd siatek na validation, zapis wyników.
4. Przegląd pomyłek na validation, decyzja, zamrożenie i zapis modelu.
5. CLI: `python -m digits.predict ścieżka.png` wypisuje jedną cyfrę. Test, że wynik to dokładnie jedna cyfra 0–9 i że model po wczytaniu daje te same predykcje.
6. Jednorazowa ocena na teście, raport i przegląd pomyłek.
7. README z instrukcją uruchomienia i sprawozdanie.

## 9. Ograniczenia, które trafią do raportu

- Wynik dotyczy wyłącznie obrazów w formacie optdigits. Nie mówi nic o innych sposobach wycinania i normalizacji znaków.
- Test ma 200 obrazów, więc przedział ufności dla accuracy ok. 97% ma szerokość kilku punktów procentowych, a dla pojedynczej cyfry kilkunastu.
- Manifest nie zawiera identyfikatorów autorów, więc nie wiadomo, ilu autorów obejmuje test i czy niektórzy są w nim nadreprezentowani.
- Model zawsze zwraca cyfrę, także dla znaku, który cyfrą nie jest.
