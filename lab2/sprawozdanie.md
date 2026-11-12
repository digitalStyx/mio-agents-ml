# Sprawozdanie: rozpoznanie danych i plan klasyfikatora spamu

Repozytorium: https://github.com/digitalStyx/mio-agents-ml

Narzędzie: Claude Code 2.1.322, model Claude Opus 5.5 (`claude-opus-5-5`), poziom rozumowania medium (zmienna środowiskowa `CLAUDE_EFFORT=medium`).

## 1. Cel i zakres

Celem jest lokalny prototyp, który na podstawie tekstu wiadomości (temat i treść) ocenia, czy jest to spam, czy ham. Na tym etapie poznano dane i ustalono plan, bez trenowania modelu. Pliki w `project/` pozostały bez zmian. Odczyt i analiza odbyły się skryptami poza projektem, a sumy SHA-256 obu CSV zgadzają się z `manifest.json`.

Dane to podzbiór korpusu Apache SpamAssassin (2002–2003): `train.csv` (180 wiadomości: 120 ham, 60 spam) i `validation.csv` (60 wiadomości: 40 ham, 20 spam). Kolumny: `message_id` (tylko identyfikator), `text` (temat i treść), `label`. Wszystkie identyfikatory, podziały i etykiety zgadzają się z manifestem. Nie ma pustych tekstów ani dokładnych duplikatów między zbiorami.

Prototyp przyjmuje tylko tekst w formacie kolumny `text`. Zamiana surowego maila na ten format jest poza zakresem (zob. cz. 5).

## 2. Wybrane podejście

**Reprezentacja:** TF-IDF na pojedynczych słowach, min_df = 2, słowa z tematu oznaczone osobno (`subj_`). Słownik i wagi IDF dopasowuje się tylko na train. Kolejność słów się gubi, ale spam rozpoznaje się głównie po słownictwie (`remove`, `free`, `money`, `receive`). Próg min_df = 2 odcina słowa występujące w jednym mailu: na pokazie z jednym spamem właśnie takie słowa (`smokes`, `'ripped`) dostawały zawyżone wagi. Pary słów i n-gramy znakowe zostają wariantami do sprawdzenia później. Przy 180 mailach reprezentacji nie komplikuje się od razu.

**Model:** regresja logistyczna. Model liniowy z regularyzacją dobrze radzi sobie z rzadkimi wektorami o tysiącach wymiarów przy małej liczbie przykładów. Ma jeden główny parametr (`C`), wagi słów da się odczytać wprost (to podstawa kontroli śladów korpusu) i zwraca prawdopodobieństwo, co pozwala świadomie ustawić próg.

**Porównanie z naiwnym Bayesem (alternatywa):** Bayes tylko zlicza częstości słów w klasach, więc uczy się szybko i przy bardzo małych zbiorach bywa lepszy. Jest też czytelny. Jego prawdopodobieństwa są jednak skrajne, zwykle bliskie 0 albo 1, co utrudnia ustawienie progu. Ponadto zakłada niezależność słów i naturalnie pracuje na zliczeniach, a nie na TF-IDF. Oba modele porównuje się w walidacji krzyżowej na train. Przy podobnym wyniku wybierana jest regresja ze względu na próg.

**Odrzucone:** XGBoost. Przy 180 przykładach drzewa mają po kilka maili w liściach, a wiele parametrów do strojenia daje wybór na poziomie szumu. Drzewa łatwo opierają się też na pojedynczym słowie, np. stopce listy. Wraca do rozważenia przy większej ilości danych lub po dodaniu metadanych.

## 3. Kroki planu i co z czego wynika

1. **Wczytanie i kontrola.** Parser CSV (pola zawierają przecinki i nowe linie), weryfikacja sum z manifestu. Podział train/validation pozostaje bez zmian.
2. **Czyszczenie tekstu** stałą, nieuczoną regułą, taką samą dla train, validation i nowych maili: wycięcie stopek list dyskusyjnych, znaczników `[…]` na początku tematu, zamiana zanonimizowanych adresów `zzzz@spamassassin.taint.org` (także w linkach) na token. *Wynika z przeglądu danych:* stopka listy ILUG jest w 18 z 60 spamów treningowych, a znaczniki list występują w obu klasach. Mówią, którą drogą wiadomość trafiła do korpusu, a nie czy jest spamem. Wzorce układa się na train, a na validation tylko sprawdza się, czy niczego nie przeoczyły. Cytaty `>`, `wrote` i `Re:` zostają, bo częściowo niosą prawdziwy sygnał rozmowy.
3. **Lista kopii kampanii spamowych**, ustalona przed uczeniem i potem niezmieniana. 6 z 20 spamów walidacyjnych to warianty spamów z train: `27bcc632c0a2fc78`, `eec728149e9886d7`, `5587807b85eebb8d`, `182f0590841142c4`, `03d402d1641ab499`, `8f6272eaef144851`. Wyznaczono je podobieństwem pięciowyrazowych fragmentów, a potem ręcznym przeglądem par do 0,2. Przegląd był konieczny, bo próg 0,5 nie złapał dwóch oczywistych kopii (0,39 i 0,56). Przykład: spam o tanich papierosach różni się od kopii jednym słowem w temacie i jedną cyfrą kodu, a podobieństwo kosinusowe ich wektorów TF-IDF wynosi 0,974. W train jest 9 podobnych par spamów, które w walidacji krzyżowej muszą trafić do tej samej części.
4. **Punkty odniesienia:** „zawsze ham” oraz reguła słów kluczowych (spam przy co najmniej *k* słowach z listy). Listę buduje się tylko ze statystyk train, a *k* dobiera się tą samą regułą progu co dla modelu. Jeśli regresja nie jest wyraźnie lepsza od reguły, model niewiele wnosi.
5. **Wybór modelu i parametrów:** walidacja krzyżowa na train, 5 części, `StratifiedGroupKFold` (kopie w jednej grupie). Pipeline z wektoryzacją dopasowywaną osobno w każdej części. Regresję i Bayesa porównuje się czułością przy regule progu.
6. **Próg:** najniższy próg, przy którym w przewidywaniach z walidacji krzyżowej jest co najwyżej 1 fałszywy alarm na 120 hamów, a wśród takich progów ten o najwyższej czułości. Tolerancja 1 maila chroni przed tym, żeby jeden nietypowy ham wyznaczał próg. Próg zostaje zamrożony przed oceną.
7. **Kontrola śladów korpusu w wagach** (na train, przed oceną). Przegląda się ~30 najsilniejszych wag w każdą stronę.
   - Słowa, które czyszczenie miało usunąć (`listinfo`, `mailman`, `ilug`, `listmaster`, `zzzz`, `taint`, `sourceforge`, `newsisfree`): poprawia się regułę czyszczenia i powtarza walidację krzyżową.
   - Słowa związane z formą rozmowy (`wrote`, `subj_re`, `date`): porównuje się wersję z nimi i bez nich. Jeśli bez nich wynik jest gorszy najwyżej o 1 maila, wybierana jest wersja bez nich. W przeciwnym razie zostają i trafiają do ograniczeń.
8. **Model końcowy** uczony na całym train z wybranymi ustawieniami. Zapis: `joblib` (cały pipeline) oraz JSON ze słownikiem, IDF, wagami, wyrazem wolnym, progiem i metadanymi (suma `train.csv`, `C`, wersje bibliotek, wersja reguł czyszczenia). Pickle wczytuje się tylko z zaufanego źródła i tą samą wersją scikit-learn.
9. **Jednorazowa ocena na validation.**

Etykiety są potrzebne przy uczeniu, w walidacji krzyżowej i przy ocenie. Przy klasyfikacji nowego maila wchodzi tylko tekst. Słownik i IDF powstają z tekstów train bez etykiet.

## 4. Planowana ocena

Spam jest klasą pozytywną. Raportowane są:
- macierz pomyłek w liczbach bezwzględnych;
- czułość dla spamu;
- precyzję dla spamu;
- odsetek fałszywych alarmów (hamy oznaczone jako spam);
- trafność, zawsze obok punktu odniesienia „zawsze ham”.

Przy każdej proporcji podaje się 95% przedział ufności Wilsona.

Raportowane są dwa wyniki:
1. **pełna walidacja**, 60 wiadomości, „zawsze ham” = 66,7%;
2. **bez 6 kopii kampanii spamowych**, 54 wiadomości (40 ham, 14 spam), „zawsze ham” = 74,1%. Ten wynik pokazuje uogólnianie, a nie pamięć modelu.

Wspólnych wątków hamu się nie wyłącza, bo w prawdziwej skrzynce też przychodzą odpowiedzi w znanych wątkach.

Dokładność ocen jest ograniczona liczebnościami. Czułość 18/20 ma przedział ok. 70–97%. Brak fałszywych alarmów na 40 hamach oznacza odsetek fałszywych alarmów do ok. 9%. Walidacja nie potwierdzi więc celu typu „poniżej 1% fałszywych alarmów”.

## 5. Niewiadome

| Co niepewne | Jak sprawdzić | Co zmieniłoby plan |
|---|---|---|
| Konwersja surowego maila (MIME, kodowania, HTML) do formatu kolumny `text` nie jest opisana | Napisać konwersję i porównać jej wynik z `text` dla plików wskazanych w manifeście (`member`) | Rozbieżności oznaczają, że prototyp nie przyjmie prawdziwych maili bez dopracowania konwersji. To osobne zadanie przed użyciem poza CSV |
| Czy lista 6 kopii i 9 par w train jest pełna | Przegląd par o niższym podobieństwie, porównanie linków i podpisów (np. ten sam link wypisania się) | Nowe kopie poszerzają listę i grupy w walidacji krzyżowej. Trzeba to zrobić przed uczeniem |
| Wspólne wątki hamu (6 z 40 hamów walidacyjnych ma wątek w train) mogą zawyżać wynik na hamie | Wynik na tych 6 hamach osobno, wyłącznie informacyjnie | Fałszywe alarmy skupione poza tymi wątkami wskazują, że odsetek fałszywych alarmów na nowych rozmowach jest wyższy |
| Na ile model opiera się na formie rozmowy (`wrote`, `Re:`) i śladach korpusu | Krok 7: przegląd wag i porównanie wersji z tymi słowami i bez nich | Silna zależność oznacza ostrzeżenie w ograniczeniach albo usunięcie tych cech |
| Stabilność progu dobranego na 120 hamach | Rozrzut progu i liczby fałszywych alarmów między częściami walidacji krzyżowej | Duży rozrzut: ostrożniejszy próg albo więcej danych hamu |
| Czy wynik przenosi się na współczesną pocztę (dane z 2002–2003, ham z `easy_ham`, proporcja klas ustalona na zajęcia) | Nie da się na tych danych; potrzebna jest świeża, oznaczona próbka z docelowego kanału | Słaby wynik na świeżych danych: nowy zbiór uczący, ewentualnie warianty reprezentacji (n-gramy znakowe) lub metadane i XGBoost |
| Środowisko: brak pandas i scikit-learn w systemie | Utworzyć `venv` i zainstalować scikit-learn przed implementacją | Bez wpływu na plan, tylko na przygotowanie |
