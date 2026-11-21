# Przegląd lab3/project: uruchomione sprawdzenia

Wszystko uruchamiano na kopii `lab3/project` (rsync bez `.venv` i `__pycache__`) interpreterem
`lab3/project/.venv/bin/python` (scikit-learn 1.9.1), z `PYTHONDONTWRITEBYTECODE=1`. Kopie usunięto po
sprawdzeniu; `lab3/` bez zmian (`git status` czysty dla lab3).

| plik | co sprawdza | wynik |
|---|---|---|
| `wyniki/b_przewidywania.txt` (`skrypty/b_przewidywania.py`) | zapisany model przelicza validation, porównanie z CSV po `message_id` | 60 wierszy, różnica p ≤ 4,8e-7, 0 złych etykiet, 0 złych przewidywań |
| `wyniki/c_train_bez_validation.txt` | `src.train` na kopii z usuniętym `data/validation.csv` | trening działa; słownik, wagi, wyraz wolny i próg identyczne z lab3 |
| `wyniki/d_near_duplicates_rerun.txt` | krok 1 README (`src.near_duplicates`) | plik par identyczny, decyzje zachowane |
| `wyniki/e_evaluate_predict.txt` | blokada `evaluate`, ponowna ocena, przykład z README | blokada działa; wynik identyczny; README podaje stary próg 0,4695 |
| `wyniki/f_zmiana_clean_po_zapisie.txt` | zmiana `src/clean.py` po zapisie modelu (bez podbicia wersji) | wczytany model daje inne p dla 6 wiadomości; `evaluate` i testy nie protestują |
| `wyniki/h_mutacje_testy.txt` | celowe błędy w kopii: M1 przesunięcie p_spam w evaluate, M2 CV bez grup, M3 uczenie na train+validation | M1 i M2: testy OK; M3: pada tylko test Wilsona (błąd zaokrąglenia przy 60/60), nie test wycieku |

Mutacje (sed na kopii):
- M1 `src/evaluate.py`: `proba = numpy.roll(pipe.predict_proba(X)[:, 1], 1)`, potem `src.evaluate`.
- M2 `src/train.py`: `groups=None`, `StratifiedKFold` zamiast `StratifiedGroupKFold` w `cross_val_predict`.
- M3 `src/train.py`: model końcowy `.fit()` na train + validation, potem `src.train` i `src.evaluate`.

## G: przebieg od zera w świeżym środowisku

Kopia `lab3/project` bez `.venv`, nowe `.venv` z `/opt/homebrew/bin/python3.13` (3.13.17) i
`pip install -r requirements.txt` (`wyniki/g_pip_freeze.txt`). Kroki wg README: usunięcie wyników,
`src.near_duplicates`, `src.train`, `src.evaluate`, testy (`wyniki/g_przebieg_od_zera.log`).
Porównanie z lab3: `skrypty/g_porownanie.py` -> `wyniki/g_porownanie.txt`. Model (parametry bit w bit
i plik joblib bajtowo), raport CV, 60 przewidywań i miary dla 60/54 identyczne. Testy: przed
treningiem `OK (skipped=8)`, po ocenie 26 OK. Kopię ze środowiskiem usunięto.
