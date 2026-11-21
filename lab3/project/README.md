# Prototyp klasyfikatora spamu

Klasyfikator spam/ham dla tekstu wiadomości (temat i treść, format kolumny `text`). Wykonuje [plan.md](plan.md):
czyszczenie stałą regułą, TF-IDF na słowach, regresję logistyczną porównaną z naiwnym Bayesem i regułą słów
kluczowych, walidację krzyżową z grupami kopii kampanii, próg z co najwyżej 1 fałszywym alarmem i jednorazową
ocenę na `validation.csv`. Ustawienia są w [config.toml](config.toml), podsumowanie w [results/SUMMARY.md](results/SUMMARY.md).

## Instalacja

Potrzebny Python 3.13. Domyślny `python3` w systemie to 3.15, a dla niego scikit-learn może nie mieć gotowych paczek.

```sh
cd project
/opt/homebrew/bin/python3.13 -m venv .venv     # lub inna ścieżka do Pythona 3.13
.venv/bin/pip install -r requirements.txt       # scikit-learn 1.9.1, numpy, scipy, joblib (dokładne wersje)
```

## Uruchomienie od zera

Wszystkie polecenia uruchamia się z katalogu `project/`.

```sh
.venv/bin/python -m src.near_duplicates   # 1. kandydaci na kopie -> results/near_duplicates.json (decyzje zostają)
.venv/bin/python -m src.train             # 2. walidacja krzyżowa, próg, kontrola wag, model -> models/
.venv/bin/python -m src.evaluate          # 3. JEDNORAZOWA ocena na validation -> results/validation_*
.venv/bin/python -m unittest discover -s tests -t .   # 4. testy
```

1. `near_duplicates` przelicza kandydatów i zachowuje zatwierdzone decyzje (`copy`, `same_sender`, `thread`,
   `different`). Jeśli po zmianie reguł czyszczenia pojawi się para bez decyzji, `train` odmówi działania,
   dopóki ktoś jej nie przejrzy. Ten krok można pominąć, bo zatwierdzony plik jest w repozytorium.
2. `train` używa tylko `train.csv`. Trwa około 30 s.
3. `evaluate` sprawdza, czy model uczono na tym samym `train.csv` i tą samą wersją scikit-learn. Po ocenie
   zapisuje `results/validation_done.lock` i drugi raz nie uruchomi się (plan: ocena tylko raz). Żeby odtworzyć
   cały przebieg od zera, usuń wyniki (bez pliku z parami):
   ```sh
   rm -rf models results/cv_report.* results/validation_* results/validation_done.lock
   ```
4. Testy wymagające modelu i wyników oceny są pomijane (`skipped`), dopóki ich nie ma.

## Klasyfikacja nowej wiadomości

Skrypt wczytuje zapisany model z `models/` i niczego nie uczy od nowa:

```sh
printf 'Subject: Cheap meds\n\nClick here to receive FREE money\n' | .venv/bin/python -m src.predict
.venv/bin/python -m src.predict wiadomosc.txt
# {"p_spam": 0.6772, "threshold": 0.4694748425776396, "label": "spam"}
```

Wejście musi być w formacie kolumny `text` (`Subject: ...`, pusty wiersz, treść). Zamiana surowego maila
(MIME, HTML) na ten format jest poza zakresem. Plik `.joblib` to pickle: wczytuj go tylko z zaufanego źródła
i tą samą wersją scikit-learn.

## Gdzie są wyniki

| plik | zawartość |
|---|---|
| `models/spam_model.joblib` | cały pipeline: czyszczenie, TF-IDF, regresja |
| `models/spam_model.json` | słownik, IDF, wagi, wyraz wolny, próg, metadane (suma `train.csv`, C, wersje bibliotek, wersja reguł czyszczenia) |
| `results/near_duplicates.json` | pary kopii i nadawców z decyzjami (krok 3) |
| `results/cv_report.md` / `.json` | walidacja krzyżowa: punkty odniesienia, siatka, próg, wagi, stabilność progu |
| `results/validation_predictions.csv` | 60 przewidywań: `message_id`, etykieta, `p_spam`, przewidywanie, czy kopia kampanii |
| `results/validation_report.md` / `.json` | macierz pomyłek i miary z przedziałami Wilsona dla 60 i 54 wiadomości |
| `results/SUMMARY.md` | co zrobiono, co odbiega od planu, czego nie sprawdzono |

## Struktura kodu

- `src/data.py`: wczytanie CSV z kontrolą sum z manifestu
- `src/clean.py`: reguła czyszczenia (wersja w `config.toml`)
- `src/features.py`: analizator i wektoryzatory
- `src/near_duplicates.py`, `src/groups.py`: kandydaci na kopie, grupy do CV, lista kopii w walidacji
- `src/models.py`: pipeline'y, reguła słów kluczowych, reguła progu
- `src/train.py`, `src/evaluate.py`, `src/predict.py`: przebieg i klasyfikacja
- `src/metrics.py`: miary i przedziały Wilsona
- `tests/`: testy (`unittest`, bez dodatkowych zależności)

---

# Dane (oryginalny opis paczki z Laboratorium 2)

Paczka służy do rozpoznania problemu klasyfikacji spamu i przygotowania planu rozwiązania. Zawiera dane oraz opis ich pochodzenia.

## Zawartość

- `data/train.csv` — wiadomości przeznaczone do uczenia.
- `data/validation.csv` — osobne wiadomości przeznaczone do oceny.
- `data/README.md` — opis kolumn, przygotowania danych i ograniczeń zbioru.
- `data/manifest.json` — informacje o pochodzeniu wiadomości i sumy kontrolne plików.

## Początek pracy

Otwórz folder `project/` w aplikacji agenta. Zapoznaj się z [opisem danych](data/README.md) i zleć agentowi odczyt plików CSV. Ustal, co zawierają i jakie obserwacje mają znaczenie dla planu.

Do odczytu danych nie trzeba instalować bibliotek ML. Wybór sposobu reprezentacji tekstu, modelu i miar oceny należy do zadania. Nie trzeba teraz implementować całego rozwiązania ani trenować końcowego modelu.

Zachowaj dostarczone pliki i podział danych. Jeśli wykonujesz małą próbę pomocną w podjęciu decyzji, zapisuj jej pliki osobno.
