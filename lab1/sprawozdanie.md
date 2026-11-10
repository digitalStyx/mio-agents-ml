# Sprawozdanie: pierwsze uruchomienie, wiadomości SpamAssassin

Repozytorium: https://github.com/digitalStyx/mio-agents-ml

Narzędzie: Claude Code 2.1.322, model Claude Opus 5.5 (`claude-opus-5-5`), poziom rozumowania medium (odczytany ze zmiennej środowiskowej sesji `CLAUDE_EFFORT=medium`).

Końcowy raport: [lab1/project/results/exercise-summary.json](https://github.com/digitalStyx/mio-agents-ml/blob/main/lab1/project/results/exercise-summary.json)

## 1. Zlecony wynik, dozwolone zmiany i informacje do odbioru

Zlecono raport dla `data/exercise.csv` zapisany w `results/exercise-summary.json`. Ustalono, że kod, dane i testy pozostają bez zmian, z jednym wyjątkiem: później miała zostać zmieniona jedna etykieta w danych, po czym raport miał zostać odświeżony. Doprecyzowano, że etykietę zmienia agent na wskazanie, a nie użytkownik ręcznie, oraz że `data/manifest.json` zostaje bez aktualizacji i ma być zgłoszony jako niezrobiony. Do odbioru zażądano: położenia raportu, jego pełnej treści, użytych danych, wykonanego polecenia i sposobu jego zakończenia, wyniku testu oraz wprost wskazania, co nie zostało zrobione.

## 2. Jak powstał raport

Przed uruchomieniem sprawdzono katalog roboczy (`lab1/project`) i dostępność Pythona. Polecenie `python3` było dostępne w wersji 3.15.0, polecenie `python` nie istniało. Polecenia wykonywał Claude Code przez narzędzie Bash na lokalnym komputerze użytkownika (macOS). Katalog `results/` przed pierwszym uruchomieniem nie istniał, więc raport nie mógł pochodzić z wcześniejszego uruchomienia.

```bash
python3 describe_data.py --data data/exercise.csv --out results/exercise-summary.json
```

Proces zakończył się kodem 0 i wypisał `Messages: 8; labels: {'ham': 5, 'spam': 3}`. Pole `input_file` wskazywało `data/exercise.csv`, a `input_sha256` (`95e9049f...ed6ed07`) zgadzało się z sumą policzoną osobno poleceniem `shasum -a 256`. Liczebność oznacza 8 rekordów CSV, czyli 8 wiadomości, z czego 5 ma etykietę `ham`, a 3 etykietę `spam`. Rekordy są liczone przez parser CSV, a nie po liniach, bo pole `text` jednej wiadomości zajmuje wiele linii. Etykiety pochodzą z korpusu Apache SpamAssassin; skrypt ich nie wylicza, tylko je zlicza.

## 3. Wyjaśniany fragment kodu i przypadek ze spacją

Omawiano `describe_data.py`, funkcję `main()`, linie 56 do 62: wywołanie `describe()` (linia 56), utworzenie katalogu i zapis raportu (linie 57 do 60) oraz obsługę wyjątków `except ... parser.exit(1, ...)` (linie 61 i 62). Wyjaśnienie własnymi słowami z rozmowy:

> czyli skrypt najpierw sprawdza cały plik w pamięci, a na dysk pisze dopiero na samym końcu. jak cokolwiek jest nie tak to wychodzi z kodem 1 zanim dotknie results/, więc albo nie ma raportu wcale, albo leży stary. dlatego sam plik raportu o niczym nie świadczy, trzeba patrzeć na kod wyjścia i to co wypisało uruchomienie.

Agent uzupełnił, że błędne argumenty kończą się kodem 2, a błąd w trakcie samego zapisu może zostawić pusty lub ucięty plik.

Przypadek do przewidzenia: po udanym uruchomieniu w ostatnim rekordzie etykieta dostaje spację na końcu. Przewidziano kod wyjścia 1, ponieważ etykieta jest porównywana dokładnie, bez obcinania spacji; komunikat `Error: Record N: label must be ham or spam.` na stderr z numerem ostatniego rekordu, bez `Report saved` i bez liczb; w katalogu wyjściowym stary raport bez zmian, z tym samym `total_messages` i starą sumą kontrolną, która nie pasuje już do zmienionego pliku.

Sprawdzono to bez ingerencji w projekt: kopię `exercise.csv` umieszczono w katalogu tymczasowym sesji i tam zapisywano raport. Pierwsze uruchomienie na nietkniętej kopii zakończyło się kodem 0 (8 wiadomości, 5 ham, 3 spam). Agent w swoim przypadku źle założył, że ostatni rekord ma etykietę `spam`; w rzeczywistości ma `ham`, dlatego kopię zepsuto na `ham ` ze spacją. Mechanizm pozostał ten sam. Drugie uruchomienie zakończyło się kodem 1, stdout był pusty, a stderr zawierał `Error: Record 8: label must be ham or spam.`. Polecenie `cmp` wykazało, że raport jest identyczny bajt w bajt z raportem z pierwszego uruchomienia, z sumą `95e9049f...`, podczas gdy zepsuta kopia miała sumę `41c5d955...`. Wniosek: przewidywanie się potwierdziło; nieudane uruchomienie zostawia stary raport, który wygląda poprawnie, ale opisuje dane, których już nie ma. Po sprawdzeniu suma `data/exercise.csv` w projekcie była niezmieniona, a `results/` nadal nie istniał.

## 4. Test

```bash
python3 -m unittest discover -s tests -v
```

Wynik: `Ran 1 test`, `OK`, kod wyjścia 0. Test uruchamia skrypt na trzech syntetycznych wiadomościach (2 ham, 1 spam) w katalogu tymczasowym i sprawdza zakończenie, czytelność raportu JSON, wskazanie właściwego wejścia oraz liczebności 3, 2 i 1 zapisane na sztywno w teście. Nie czyta `exercise.csv` i nie dotyka `results/exercise-summary.json`, co potwierdzono niezmienioną datą modyfikacji i rozmiarem raportu. Test nie mówi więc nic o raporcie dla danych ćwiczeniowych ani o jakości przyszłego modelu. Uruchomienie testu utworzyło `tests/__pycache__`, który na polecenie usunięto.

## 5. Zmiana etykiety i odświeżenie raportu

Zmieniono etykietę wiadomości `exercise-001` z `ham` na `spam`, jako symulację nowej wersji danych. Projekt nie jest śledzony przez gita, więc zamiast `git diff` wybrano kopię oryginału w katalogu tymczasowym i porównanie `diff`. Zmianę wykonano przez podmianę fragmentu kończącego rekord `exercise-001` tuż przed początkiem `exercise-002`, z kontrolą, że fragment występuje w pliku dokładnie raz. Porównanie wykazało jedną zmienioną linię:

```bash
24c24
< ",ham
---
> ",spam
```

Przed odświeżeniem ustalono, że raport w `results/` jest nieaktualny i sam się nie odświeży. Przewidziano spadek `ham` z 5 na 4, wzrost `spam` z 3 na 4, sumę 8, inną sumę kontrolną i to samo `input_file`. Do potwierdzenia wybrano ponowne uruchomienie tego samego polecenia i odczyt raportu. Testu nie uruchamiano ponownie, ponieważ kod się nie zmienił, a test nie czyta `exercise.csv`.

```bash
python3 describe_data.py --data data/exercise.csv --out results/exercise-summary.json
```

Proces zakończył się kodem 0 i wypisał `Messages: 8; labels: {'ham': 4, 'spam': 4}`. Pełna treść końcowego raportu:

```json
{
  "input_file": "/Users/radoslawgniado/mio-agents-ml/lab1/project/data/exercise.csv",
  "input_sha256": "7dc8080df480e67e5361218ae0e75f26c963ede9d72788985b25cedb818b8152",
  "total_messages": 8,
  "label_counts": {
    "ham": 4,
    "spam": 4
  },
  "runtime": {
    "python_executable": "/opt/homebrew/opt/python@3.15/bin/python3.15",
    "python_version": "3.15.0",
    "working_directory": "/Users/radoslawgniado/mio-agents-ml/lab1/project"
  }
}
```

| | Przed zmianą | Po zmianie |
|---|---|---|
| `total_messages` | 8 | 8 |
| `ham` | 5 | 4 |
| `spam` | 3 | 4 |
| `input_sha256` | `95e9049f...ed6ed07` | `7dc8080d...818b8152` |
| `input_file` | `data/exercise.csv` | `data/exercise.csv` |

Przewidywanie potwierdziło się we wszystkich punktach. Nowa suma kontrolna w raporcie zgadza się z sumą obecnego pliku policzoną osobno, więc raport opisuje dane po zmianie.

Nie zrobiono: aktualizacji `data/manifest.json`, którego suma kontrolna dla `exercise.csv` nie zgadza się już z plikiem; commita zmian, a katalog projektu nie był śledzony przez gita; ponownego uruchomienia testu po zmianie danych. Kopia danych sprzed zmiany istnieje wyłącznie w katalogu tymczasowym sesji, a w projekcie nie ma innej kopii poprzedniej wersji.
