# Sprawozdanie: implementacja klasyfikatora spamu według planu

Repozytorium: https://github.com/digitalStyx/mio-agents-ml

Narzędzie: Claude Code 2.1.330, model Claude Opus 5.5 (`claude-opus-5-5`), poziom rozumowania medium (zmienna środowiskowa `CLAUDE_EFFORT=medium`).

## 1. Delegacja

Główne zlecenie (dosłownie):

```text
dobra, teraz glowne zadanie. chce miec dzialajacy przebieg wg plan.md: uczenie, zapisany model w models/, przewidywania dla validation i miary w results/, i predict ktory wczytuje model z pliku i nie uczy od nowa. do tego README: co zainstalowac, jak odpalic calosc od zera i gdzie sa wyniki.

kontekst masz w plan.md, config.toml i w tym co juz napisales w src/, pary sa zatwierdzone. pracujesz tylko w tym katalogu, data/ i plan.md nie ruszasz, nic nie commitujesz.

sprawdzenia zapisz jako testy w tests/ zeby dalo sie je odpalic jedna komenda. oczekiwane wyniki bierz z planu i manifestu a nie z tego co akurat zwraca kod. minimum:
- czyszczenie na krotkich przykladach: stopka listy i znacznik z tematu znikaja, zzzz@ zamienia sie na token, cytaty > i wrote zostaja
- prog na zmyslonych wynikach: wybiera najnizszy prog z max 1 falszywym alarmem
- zadna grupa kopii nie jest rozbita miedzy czesci walidacji krzyzowej
- model wczytany z pliku daje te same przewidywania co przed zapisem
- plik z przewidywaniami ma 60 wierszy z id z validation, a miary sa policzone dla 60 i dla 54

bledy w kodzie, reguly czyszczenia i ustawienia z configa poprawiaj sam, bez pytania, tylko zapisuj co zmieniles. wroc do mnie jak cos wymaga odejscia od planu (inny model, inna regula progu, inne miary, zmiana par) albo doinstalowania czegos spoza requirements.

jedno zatrzymanie obowiazkowe: po train a przed ocena na validation pokaz mi wyniki walidacji krzyzowej, wybrany prog i przeglad wag, i czekaj na ok. ocena na validation ma byc raz.

koniec jest wtedy jak testy przechodza, calosc da sie odpalic wg README, a w results/ jest krotkie podsumowanie: co zrobione, co odbiega od planu, czego nie sprawdzono.
```

**Przed zleceniem** zrobiono cztery rzeczy:

- **Środowisko:** `.venv` na Pythonie 3.13 (systemowy 3.15 pominięto) i `requirements.txt` z dokładnymi wersjami (scikit-learn 1.9.1).
- **Czyszczenie, wersja 1:** wzorce stopek i znaczników ułożono na train. Przegląd validation wykazał przeoczenia (zacytowane stopki SourceForge, adresy `zzzz…` w domenach), więc wzorce uogólniono.
- **Pary kopii** policzono od nowa i zatwierdzono po ręcznym przeglądzie 48 kandydatów. 6 kopii w walidacji zgadza się z planem. Doszła kategoria „ten sam nadawca” (wspólny link wypisania lub adres zwrotny, inna oferta), grupowana w walidacji krzyżowej.
- **Kod przebiegu:** `train.py`, `evaluate.py` i `predict.py` napisano razem z wykrywaniem par, ale ich nie uruchamiano. Uczenie, testy, poprawki i ocena odbyły się po zleceniu.

**Po zatrzymaniu** zmieniono regułę progu. Próg z jednego ziarna (0,4695) przy innych podziałach dawał 2 fałszywe alarmy. Zamrożono najwyższy z progów dla 5 ziaren (0,5096), co plan przewidywał jako reakcję na duży rozrzut.

## 2. Organizacja pracy

- **Jedno zlecenie.** Uczenie, zapis modelu, ocena, `predict`, testy i README tworzą jeden ciąg, w którym każdy krok zależy od poprzedniego. Sprawdzenia trwają sekundy (trening ok. 30 s, testy ok. 1 s), więc dzielenie pracy na osobne zlecenia nic by nie dało.
- **Jedno zatrzymanie, przed oceną na validation.** Tylko tego kroku nie da się uczciwie powtórzyć. Błędy w kodzie, reguły czyszczenia i ustawienia można poprawiać i przeliczać dowolnie, bo dotyczą wyłącznie train. Ocenę zabezpieczono w kodzie: `evaluate` odmawia ponownego uruchomienia, uruchomienia z modelem uczonym na innym `train.csv` i z inną wersją scikit-learn.
- **Oczekiwane wyniki testów** pochodzą z planu i manifestu, a nie z wyjścia kodu:
  - 6 identyfikatorów kopii z planu;
  - liczebności 180/120/60 i 60/40/20;
  - „zawsze ham” 66,7% i 74,1%, 14 spamów bez kopii;
  - przedziały Wilsona 70–97% dla 18/20 i do ok. 9% dla 0/40;
  - identyfikatory walidacji z manifestu.

  Regułę progu sprawdzono na zmyślonych wynikach z ręcznie policzonym przedziałem. Testy używają `unittest`, więc nie trzeba instalować niczego spoza `requirements.txt`.

## 3. Moment feedbacku: reguła czyszczenia v2

Przegląd wag modelu końcowego (krok 7 planu) pokazał w czołówce wag hamu słowa, które nie dotyczą treści:

- `supplied` (−0,547, pierwsze miejsce),
- `url` (−0,315, 18. miejsce),
- `archives`.

Pochodziły z nagłówka wpisów z czytników RSS na początku treści, np.:

```text
URL: http://boingboing.net/#85516609
Date: Not supplied
```

Taki nagłówek mówi, którą drogą wiadomość trafiła do korpusu, a nie czy jest spamem. To ten sam rodzaj śladu co newsisfree, który plan każe usuwać. Regułę czyszczenia rozszerzono o usuwanie tego nagłówka (`FEED_HEADER` w `src/clean.py`, wersja reguł 2).

| | v1 | v2 |
|---|---|---|
| hamy train zawierające po czyszczeniu `supplied` | 14 ze 120 | 0 |
| hamy validation zawierające po czyszczeniu `supplied` | 2 z 40 | 0 |
| hamy train z nagłówkiem `URL:` lub `Date:` na początku treści | 22 | 1 (zacytowany nagłówek maila, celowo zostawiony) |
| `supplied` w czołówce wag hamu | 1. miejsce, −0,547 | brak w top 30 |
| `url` w czołówce wag hamu | 18. miejsce, −0,315 | brak w top 30 |
| najsilniejsza waga hamu | `supplied` −0,547 | `it` −0,494 |
| walidacja krzyżowa (próg z ziarna głównego) | 57/60 spamów, 1/120 FP, próg 0,4717 | 57/60 spamów, 1/120 FP, próg 0,4695 |

Wynik walidacji krzyżowej się nie zmienił. Model przestał jednak rozpoznawać część hamu po formacie bramki RSS i opiera się na treści.

## 4. Rezultat i ograniczenia

**Rezultat:**

- kod i dane: [project/](project/);
- instrukcja instalacji i uruchomienia od zera: [project/README.md](project/README.md);
- model: [project/models/](project/models/), `spam_model.joblib` (cały pipeline) i `spam_model.json` (słownik, IDF, wagi, próg, metadane);
- wyniki:
  - [podsumowanie](project/results/SUMMARY.md),
  - [walidacja krzyżowa](project/results/cv_report.md),
  - [ocena na validation](project/results/validation_report.md),
  - [przewidywania](project/results/validation_predictions.csv),
  - [pary kopii](project/results/near_duplicates.json).

Wybrany model to regresja logistyczna na TF-IDF (C = 1,0, bez słów `date`, `subj_re`, `wrote`). Wyniki jednorazowej oceny na validation:

| | 60 wiadomości | bez 6 kopii (54) |
|---|---|---|
| wykryte spamy | 19/20 = 95,0% (76,4–99,1%) | 13/14 = 92,9% (68,5–98,7%) |
| precyzja | 19/19 = 100% (83,2–100%) | 13/13 = 100% (77,2–100%) |
| fałszywe alarmy | 0/40 = 0% (0–8,8%) | 0/40 = 0% (0–8,8%) |
| trafność („zawsze ham”) | 98,3% (66,7%) | 98,1% (74,1%) |

W nawiasach przy proporcjach podano 95% przedziały Wilsona. Jedyna pomyłka to francuski newsletter (p = 0,456). Wszystkie 26 testów przechodzi. Świeże środowisko z `requirements.txt` daje identyczne wagi i próg.

**Odstępstwa od planu:**

1. **Próg:** najwyższy z progów dla 5 ziaren zamiast progu z jednego podziału. W walidacji krzyżowej wykrywa 55 zamiast 57 spamów. Na validation oba progi dają ten sam wynik.
2. **Grupy w walidacji krzyżowej są szersze niż w planie:** 25 spamów w 8 grupach (kopie i ten sam nadawca) zamiast 9 par.
3. **Czyszczenie wykracza poza listę z planu:** nagłówki RSS (v2). Część wzorców uogólniono po przeglądzie validation, więc reguły nie powstały wyłącznie na train.
4. **Ustawienia nieokreślone w planie** są w `config.toml`: tokeny a–z, `sublinear_tf`, `class_weight="balanced"`, siatki parametrów, „wyraźnie lepszy” = co najmniej 2 spamy.
5. **Wspólne wątki hamu:** znaleziono 3 zamiast 6 podanych w planie (inny próg wyszukiwania). Wynik na nich jest tylko informacyjny.

**Czego nie sprawdzono:**

- **Spam w innych językach:** tokenizer bierze tylko litery a–z. Przeoczone zostały spam japoński i serbski (w walidacji krzyżowej) oraz francuski (na validation).
- **Poczta nietechniczna:** wagi hamu odzwierciedlają styl list technicznych (`rpm`, `server`, `red hat`, `sep`). Odsetek fałszywych alarmów na prywatnej lub nietechnicznej poczcie jest nieznany.
- **Kalibracja:** prawdopodobieństwa są ściśnięte wokół 0,5, więc `p_spam` nie jest skalibrowanym prawdopodobieństwem.
- **Liczebność:** 0 fałszywych alarmów na 40 hamach nie wyklucza odsetka do 8,8%.
- **Nie wykonano:**
  - konwersji surowego maila do formatu `text`,
  - wariantów reprezentacji (pary słów, n-gramy znakowe),
  - oceny na współczesnej poczcie,
  - przeglądu par poniżej progów wyszukiwania.
