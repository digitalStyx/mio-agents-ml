# Podsumowanie: prototyp klasyfikatora spamu

Stan na 2026-11-21. Szczegóły: `cv_report.md` (walidacja krzyżowa), `validation_report.md` (ocena),
`validation_predictions.csv` (60 przewidywań), `near_duplicates.json` (pary kopii). Uruchomienie: `../README.md`.

## Co zrobiono

Przebieg według `plan.md`, kroki 1-9:

1. **Wczytanie** z kontrolą sum SHA-256, liczebności i identyfikatorów z `manifest.json`.
2. **Czyszczenie** stałą regułą (wersja 2): stopki list (Mailman, ILUG, SourceForge, Yahoo Groups, również w cytatach), znaczniki list w temacie, nagłówki wpisów RSS, zanonimizowane adresy zamienione na token `anonaddr`. Cytaty `>`, `wrote` i `Re:` zostają.
3. **Kopie kampanii** wyznaczone na nowo: Jaccard pięciowyrazowych fragmentów, kosinus TF-IDF, wspólne linki i adresy, a potem ręczny przegląd 48 par, zatwierdzony przez użytkownika. 6 kopii w walidacji jest zgodnych z planem.
4. **Punkty odniesienia** w CV: „zawsze ham” 0/60 spamów, reguła słów kluczowych 30/60 przy 0 fałszywych alarmach.
5. **Wybór modelu:** `StratifiedGroupKFold` (5 części, 163 grupy), wektoryzacja dopasowywana w każdej części. Regresja logistyczna 57-58/60, naiwny Bayes 57/60. Bayes nie jest lepszy o co najmniej 2 maile, więc wybrano regresję (C = 1,0).
6. **Próg:** reguła z planu (najniższy próg z co najwyżej 1 fałszywym alarmem na 120 hamów), następnie wariant ostrożniejszy, zob. odstępstwa. Zamrożony próg 0,5096, w CV 55/60 spamów i 1/120 fałszywy alarm.
7. **Kontrola wag:** śladów korpusu z listy planu nie ma w czołówce. `date`, `subj_re` i `wrote` były w czołówce wag hamu. Bez nich model traci 1 spam, więc zgodnie z planem wybrano wersję bez nich.
8. **Model końcowy** na całym train: `models/spam_model.joblib` (cały pipeline) i `models/spam_model.json` (słownik, IDF, wagi, wyraz wolny, próg, metadane). Świeże środowisko z `requirements.txt` daje identyczne wagi i próg.
9. **Jednorazowa ocena** na validation, z blokadą przed powtórzeniem:

| | pełna walidacja (60) | bez 6 kopii (54) |
|---|---|---|
| macierz pomyłek (TP / FN / FP / TN) | 19 / 1 / 0 / 40 | 13 / 1 / 0 / 40 |
| czułość dla spamu | 95,0% (76,4–99,1%) | 92,9% (68,5–98,7%) |
| precyzja dla spamu | 100% (83,2–100%) | 100% (77,2–100%) |
| odsetek fałszywych alarmów | 0% (0–8,8%) | 0% (0–8,8%) |
| trafność, w nawiasie „zawsze ham” | 98,3% (66,7%) | 98,1% (74,1%) |

W nawiasach przy proporcjach podano 95% przedziały Wilsona.

- Jedyna pomyłka: francuski newsletter `017f0817d4c92951` (p = 0,456).
- 2 spamy od nadawców znanych z train, ale z inną ofertą, zostały wykryte.
- 3 hamy ze wspólnym wątkiem w train: 0 fałszywych alarmów.

**Testy:** `.venv/bin/python -m unittest discover -s tests -t .`, 26 testów, wszystkie przechodzą. Obejmują czyszczenie, regułę progu, grupy w CV, zapis i odczyt modelu, `predict` bez ponownego uczenia, plik przewidywań (60 identyfikatorów) oraz miary dla 60 i 54 wiadomości. Wartości oczekiwane pochodzą z planu i manifestu.

## Co odbiega od planu

1. **Próg (decyzja użytkownika po CV).** Próg z ziarna głównego (0,4695) przy innych podziałach CV dawał 2 fałszywe alarmy. Zamrożono najwyższy z progów wyznaczonych regułą planu dla 5 ziaren (2026–2030): 0,5096. Kosztem są 2 spamy w CV (55 zamiast 57). Na validation oba progi dają ten sam wynik: najwyższy ham ma p = 0,447, przeoczony spam p = 0,456.
2. **Grupy w CV są szersze niż w planie.** Plan mówił o 9 parach spamów w train. Wyszło 16 par kopii, głównie z pięcioosobowej kampanii „adult”, oraz nowa kategoria „ten sam nadawca” (wspólny link wypisania, adres zwrotny lub szablon, inna oferta). W CV grupowano obie kategorie: 25 spamów w 8 grupach. Z wyniku na 54 mailach wyłączono tylko 6 kopii z planu.
3. **Reguły czyszczenia poza listą z planu.** Usuwane są też nagłówki wpisów RSS (`URL:` / `Date:`). Słowa `supplied` i `url` były w czołówce wag hamu. Część wzorców (zacytowane stopki SourceForge, adresy `zzzz…` w domenach) uogólniono po przeglądzie validation. Plan przewidywał taki przegląd, ale oznacza to, że reguły nie powstały wyłącznie na train.
4. **Ustawienia, których plan nie określał** (`config.toml`): tokeny tylko z liter a–z, `sublinear_tf`, `class_weight="balanced"`, siatka C i alpha, lista słów kluczowych dobierana w CV z rozmiarów 10/20/30/50, „wyraźnie lepszy” = co najmniej 2 spamy więcej.
5. **Wspólne wątki hamu:** znaleziono 3, a plan podawał 6, bo próg wyszukiwania był inny. Wynik na nich jest tylko informacyjny.

## Czego nie sprawdzono / ograniczenia

- **Spam w innych językach.** Tokenizer bierze tylko litery a–z. W CV przeoczone zostały spam japoński (żadnych tokenów), serbski i angielski spam adult, a na validation spam francuski. Spam spoza angielskiego jest praktycznie niewidoczny dla modelu.
- **Model uczy się „stylu list technicznych”.** Wagi hamu to głównie słowa rozmowy i tematy techniczne (`it`, `i'm`, `rpm`, `server`, `red hat`, `sep` jako ślad dat z 2002 roku). Fałszywe alarmy na prywatnej lub nietechnicznej poczcie mogą być znacznie częstsze.
- **Pamięć kampanii.** Część wag spamu (`membership`, `vip`, `access`, `q`) pochodzi z jednej kampanii z 5 kopiami w train.
- **Prawdopodobieństwa są ściśnięte wokół 0,5** (hamy ok. 0,35–0,45, spamy od 0,45). `p_spam` nie jest skalibrowanym prawdopodobieństwem, a margines wokół progu jest mały.
- **Mała próba.** Brak fałszywych alarmów na 40 hamach oznacza górną granicę 8,8%. Walidacja nie potwierdza celu typu „poniżej 1% fałszywych alarmów”.
- **Nie zrobiono:** konwersji surowego maila (MIME, HTML) do formatu `text` i porównania z plikami `member` z manifestu; wariantów reprezentacji (pary słów, n-gramy znakowe); oceny na współczesnej poczcie; przeglądu par poniżej progów wyszukiwania. Eksport JSON sprawdzono testem tylko dla regresji logistycznej.
- Środowisko testowano tylko na Pythonie 3.13 (macOS).
