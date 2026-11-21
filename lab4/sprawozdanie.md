# Sprawozdanie: przegląd prototypu klasyfikatora spamu

Repozytorium: https://github.com/digitalStyx/mio-agents-ml

Narzędzie: Claude Code 2.1.330, model Claude Opus 5.5 (`claude-opus-5-5`), poziom rozumowania medium (zmienna środowiskowa `CLAUDE_EFFORT=medium`).

## 1. Cel i oceniany rezultat

Celem przeglądu było ustalenie, czy prototyp z Laboratorium 3 spełnia uzgodnienia. Uwagę skupiono na trzech pytaniach:

- czy uczenie nie widzi validation;
- czy `predict` i `evaluate` korzystają z zapisanego modelu i niczego nie uczą;
- czy przewidywania są porównywane z etykietami tych samych wiadomości.

Sprawdzono też, czy testy obejmują wymagane punkty.

Podstawa oceny:
- plan: [lab3/project/plan.md](../lab3/project/plan.md);
- zlecenie i przebieg pracy: [lab3/sprawozdanie.md](../lab3/sprawozdanie.md), cz. 1;
- oceniany projekt: [lab3/project/](../lab3/project/) (kod, model, wyniki, testy, README).

Wszystkie uruchomienia odbyły się na kopiach projektu. Katalog `lab3/` pozostał bez zmian.

## 2. Ustalenia przeglądu

### Potwierdzone błędy

1. **Zapisany model zależy od bieżącego kodu czyszczenia.**
   - *Skutek:* zmiana `src/clean.py` po treningu po cichu zmienia przewidywania zamrożonego modelu. Nie zauważą tego ani `evaluate`, ani `predict`, ani testy. Obecnych wyników to nie dotyczy, bo `clean.py` zmieniono ostatnio przed zapisem modelu.
   - *Podstawa:*
     - pipeline zapisuje tylko odwołanie do funkcji `clean` ([features.py:28](../lab3/project/src/features.py));
     - wersja reguł czyszczenia nie jest sprawdzana przy wczytaniu modelu ([evaluate.py:27-31](../lab3/project/src/evaluate.py), [predict.py:18-20](../lab3/project/src/predict.py));
     - na kopii wyłączono jedną regułę: ten sam plik modelu dał inne prawdopodobieństwa dla 6 hamów, a ocena i 26 testów przeszły ([f_zmiana_clean_po_zapisie.txt](przeglad/wyniki/f_zmiana_clean_po_zapisie.txt)).
2. **Test miar może zawodzić przy poprawnych wynikach.**
   - *Skutek:* przy wyniku 100% (np. trafność 60/60) `test_measures_for_60_and_54` pada. Obecne wyniki przechodzą przypadkiem.
   - *Podstawa:*
     - przez zaokrąglenie górna granica przedziału Wilsona dla k = n wychodzi poniżej 1,0 (0,9999999999999998), co dotyczy 59 wartości n z zakresu 1–200, w tym n = 60; analogicznie dolna granica dla k = 0 wychodzi powyżej 0 dla 36 wartości n;
     - `min(1.0, …)` tego nie koryguje ([metrics.py:14](../lab3/project/src/metrics.py)), a test wymaga, by granica nie była mniejsza od wartości ([test_outputs.py:66-67](../lab3/project/tests/test_outputs.py));
     - wynik: [h_mutacje_testy.txt](przeglad/wyniki/h_mutacje_testy.txt).
3. **README podaje nieaktualny próg.**
   - *Skutek:* przykład wywołania `predict` nie zgadza się z tym, co zwraca program.
   - *Podstawa:* [README.md:48](../lab3/project/README.md) podaje 0,4695, a `predict` zwraca 0,5096 ([e_evaluate_predict.txt](przeglad/wyniki/e_evaluate_predict.txt)).

### Słabości testów (potwierdzone)

Sprawdzono je, wprowadzając w kopii celowe błędy i uruchamiając testy ([h_mutacje_testy.txt](przeglad/wyniki/h_mutacje_testy.txt)):

| błąd wprowadzony w kopii | wynik testów |
|---|---|
| M1: `evaluate` przypisuje prawdopodobieństwa przesunięte o jedną wiadomość (czułość spada do 7/20) | przechodzą |
| M2: walidacja krzyżowa w `train` bez grup | przechodzą |
| M3: model końcowy uczony na train razem z validation | pada tylko test Wilsona z powodu błędu 2, a nie z powodu wycieku |

Oznacza to, że:

- **grupy w walidacji krzyżowej:** test sprawdza sam podział ([test_cv_groups.py:37](../lab3/project/tests/test_cv_groups.py)), ale nie to, czy `train.py` przekazuje grupy;
- **zapis i odczyt modelu:** test uczy nowy model i porównuje go w jednym procesie ([test_model_io.py:23-33](../lab3/project/tests/test_model_io.py)), a zapisanego modelu z niczym nie porównuje;
- **plik przewidywań:** sprawdzane są identyfikatory i etykiety, ale nie to, czy `p_spam` należy do danej wiadomości;
- **brak testów** blokady ponownej oceny i tego, że uczenie nie czyta validation;
- **uruchomienie od zera:** przed treningiem 8 testów jest pomijanych, a całość kończy się wynikiem „OK”.

### Podejrzenia (bez wpływu na obecne wyniki)

- **Zatwierdzone pary mogą zniknąć.** `near_duplicates` przenosi decyzje tylko do par, które znowu zostaną znalezione ([near_duplicates.py:108-116](../lab3/project/src/near_duplicates.py)). Para, która po zmianie czyszczenia wypadnie z kandydatów, straci decyzję bez ostrzeżenia, a razem z nią zniknie jej grupa w walidacji krzyżowej. Obecnie ponowne uruchomienie daje identyczny plik ([d_near_duplicates_rerun.txt](przeglad/wyniki/d_near_duplicates_rerun.txt)).
- **Pozostawiony dopisek w konfiguracji.** W [config.toml](../lab3/project/config.toml) przy `validation_exclude_decisions` został dopisek „DO POTWIERDZENIA”. Wynik zgadza się z 6 kopiami z planu.
- **Porównanie bez warunku.** Wersje ze słowami rozmowy i bez nich są porównywane zawsze ([train.py:113-114](../lab3/project/src/train.py)), a plan wymaga tego tylko wtedy, gdy te słowa są w czołówce wag. W tym przebiegu były, więc wynik się nie zmienia.

## 3. Dowody

**Kluczowy dowód: trening bez pliku validation.** Z kopii projektu usunięto `data/validation.csv` i uruchomiono `src.train`. Trening przebiegł bez błędu, a słownik, wagi, wyraz wolny i próg wyszły identyczne z modelem z lab3 ([c_train_bez_validation.txt](przeglad/wyniki/c_train_bez_validation.txt)). Skoro model powstaje identyczny bez dostępu do validation, to validation nie brała udziału w uczeniu ani w wyborze progu. Dowód nie mówi natomiast nic o regułach czyszczenia: część wzorców poprawiono po obejrzeniu validation ([SUMMARY.md](../lab3/project/results/SUMMARY.md), odstępstwo 3). Ten wpływ jest zapisany w kodzie, więc test bez pliku go nie wykrywa.

| zachowanie | wynik | co potwierdza | czego nie obejmuje |
|---|---|---|---|
| uczenie nie widzi validation | [c_train_bez_validation.txt](przeglad/wyniki/c_train_bez_validation.txt): model identyczny bez pliku validation | `train` nie czyta validation; zapisany model powstał wyłącznie z train | reguł czyszczenia poprawianych po obejrzeniu validation; żaden test tego nie pilnuje (M3) |
| przebieg od zera według README | [g_porownanie.txt](przeglad/wyniki/g_porownanie.txt), [g_przebieg_od_zera.log](przeglad/wyniki/g_przebieg_od_zera.log): świeże `.venv` z `requirements.txt` daje model (parametry bit w bit, joblib bajtowo), walidację krzyżową, 60 przewidywań i miary identyczne z lab3; 26 testów przechodzi | odtwarzalność całego przebiegu i instrukcji z README | innych systemów i wersji Pythona niż 3.13 na macOS |
| `predict` wczytuje model z pliku i nie uczy | [test_model_io.py](../lab3/project/tests/test_model_io.py) (`test_predict_cli_loads_file_without_training`); [e_evaluate_predict.txt](przeglad/wyniki/e_evaluate_predict.txt) | model jest wczytywany z pliku, plik się nie zmienia, wynik zgadza się z pipeline'em | zależności modelu od bieżącego `clean.py` (błąd 1) |
| `evaluate` używa zapisanego modelu, ocena raz | [e_evaluate_predict.txt](przeglad/wyniki/e_evaluate_predict.txt): blokada odmawia drugiej oceny, ponowna ocena po jej usunięciu daje identyczny wynik; [log sesji lab3](../lab3/sesje/sesja-1.jsonl), wpisy 583–585: zatrzymanie i zgoda przed oceną | ocena jest odtwarzalna z pliku modelu; na validation oceniano raz, po zatrzymaniu | blokada to zwykły plik usuwany poleceniem z README; brak testu blokady |
| przewidywania dla tych samych wiadomości | [b_przewidywania.txt](przeglad/wyniki/b_przewidywania.txt): przeliczenie 60 wiadomości po `message_id`, różnica p ≤ 4,8e-7, 0 niezgodności | [validation_predictions.csv](../lab3/project/results/validation_predictions.csv) jest poprawnie przypisany do wiadomości | ochrony na przyszłość: testy nie wykrywają przesunięcia (M1) |
| miary dla 60 i 54 wiadomości | [validation_report.md](../lab3/project/results/validation_report.md): 19/1/0/40 i 13/1/0/40, „zawsze ham” 66,7% i 74,1%; [test_outputs.py](../lab3/project/tests/test_outputs.py) | liczebności, lista 6 kopii i wyliczenia zgodne z planem i manifestem | pełności listy kopii; test przedziałów Wilsona jest zawodny (błąd 2) |
| grupy kopii nie są rozbijane w walidacji krzyżowej | [test_cv_groups.py](../lab3/project/tests/test_cv_groups.py) (5 ziaren); [cv_report.md](../lab3/project/results/cv_report.md): 163 grupy; [d_near_duplicates_rerun.txt](przeglad/wyniki/d_near_duplicates_rerun.txt) | podział respektuje grupy; pary i decyzje się odtwarzają | tego, czy `train.py` przekazuje grupy (M2; potwierdza to tylko lektura [train.py:29-30](../lab3/project/src/train.py)); par poniżej progów wyszukiwania |
| reguła progu i zamrożony próg | [test_threshold.py](../lab3/project/tests/test_threshold.py); [cv_report.md](../lab3/project/results/cv_report.md), tabela stabilności; próg 0,5096 odtworzony w [g_porownanie.txt](przeglad/wyniki/g_porownanie.txt) | reguła działa na zmyślonych wynikach; próg to uzgodnione maksimum z 5 ziaren | zachowania progu na nowych danych: na validation najwyższy ham ma p = 0,447, a przeoczony spam 0,456, przy 40 hamach |

## 4. Decyzja

**Prototyp przyjęto w uzgodnionym zakresie:** lokalny przebieg, zapisany model, przewidywania i miary.

**Uzasadnienie:**

- Trzy wymagania kluczowe są spełnione i potwierdzone uruchomieniem:
  - uczenie bez validation;
  - model wczytywany z pliku;
  - przewidywania przypisane do właściwych wiadomości.
- Cały przebieg odtwarza się od zera w świeżym środowisku z identycznym wynikiem.
- Trzy błędy i słabości testów nie zmieniają obecnych wyników. Dotyczą odporności na przyszłe zmiany i trafiają na listę poprawek przed dalszą pracą.

**Lista do poprawy przed dalszą pracą:**

1. Sprawdzanie wersji reguł czyszczenia przy wczytaniu modelu albo zapis kodu czyszczenia razem z modelem (błąd 1).
2. Granice przedziału Wilsona dla k = 0 i k = n albo tolerancja w teście (błąd 2).
3. Próg w przykładzie w README (błąd 3).
4. Testy:
   - przypisania `p_spam` do wiadomości (przeliczenie zapisanym modelem);
   - tego, że `train` przekazuje grupy;
   - tego, że uczenie nie czyta validation;
   - blokady ponownej oceny;
   - porównania zapisanego modelu z zapisanymi przewidywaniami.
5. Podejrzenia z cz. 2: zachowanie decyzji o parach przy ponownym wyszukiwaniu, dopisek w `config.toml`.

**Ograniczenia, które zostają (niesprawdzane):**

- **Czystość wyniku na validation.** Część reguł czyszczenia poprawiono po obejrzeniu validation, a skala tego wpływu jest nieznana.
- **Pełność listy kopii.** Nieznalezione kopie zawyżają wynik na 54 wiadomościach.
- **Odsetek fałszywych alarmów.** 0 na 40 hamów oznacza wartość do 8,8% (przedział Wilsona), więc walidacja nie potwierdza niskiego odsetka.
- **Zakres danych.** Spam w innych językach jest praktycznie niewidoczny dla modelu. Zachowanie na poczcie nietechnicznej i współczesnej jest nieznane.
- **Format wejścia.** Brak konwersji surowego maila do formatu `text`.

**Nowe wyniki z przeglądu:**

- [przeglad/README.md](przeglad/README.md): opis sprawdzeń i wprowadzonych błędów;
- [przeglad/wyniki/](przeglad/wyniki/): wyniki sprawdzeń B–H, w tym przebieg od zera (`g_*`);
- [przeglad/skrypty/](przeglad/skrypty/): skrypty porównań.
