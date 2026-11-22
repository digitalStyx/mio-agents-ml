# Ocena na zbiorze testowym

Model: SVC {'C': 10, 'gamma': 0.3}, uczony na train (600 obrazów). Test: 200 obrazów z `optdigits.tes` (inni autorzy pisma niż w train). Ocena wykonana raz.

| Model | Trafienia | Accuracy | 95% CI (Wilson) |
|---|---|---|---|
| **wybrany model** | 197/200 | 98.5% | 95.7% – 99.5% |
| klasa najczęstsza | 20/200 | 10.0% | 6.6% – 14.9% |
| najbliższy centroid | 186/200 | 93.0% | 88.6% – 95.8% |

Dla porównania validation: 196/200 (98.0%). Validation pochodzi od tych samych autorów co train, więc jest optymistyczna.

Jedna pomyłka na teście to 0,5 p.p.

## Trafność dla każdej cyfry

| Cyfra | Trafienia | 95% CI |
|---|---|---|
| 0 | 20/20 | 83.9% – 100.0% |
| 1 | 19/20 | 76.4% – 99.1% |
| 2 | 20/20 | 83.9% – 100.0% |
| 3 | 19/20 | 76.4% – 99.1% |
| 4 | 20/20 | 83.9% – 100.0% |
| 5 | 20/20 | 83.9% – 100.0% |
| 6 | 20/20 | 83.9% – 100.0% |
| 7 | 19/20 | 76.4% – 99.1% |
| 8 | 20/20 | 83.9% – 100.0% |
| 9 | 20/20 | 83.9% – 100.0% |

## Macierz pomyłek (wiersze: prawdziwa, kolumny: przewidziana)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 20 | · | · | · | · | · | · | · | · | · |
| **1** | · | 19 | · | · | · | · | 1 | · | · | · |
| **2** | · | · | 20 | · | · | · | · | · | · | · |
| **3** | · | · | · | 19 | · | · | · | · | · | 1 |
| **4** | · | · | · | · | 20 | · | · | · | · | · |
| **5** | · | · | · | · | · | 20 | · | · | · | · |
| **6** | · | · | · | · | · | · | 20 | · | · | · |
| **7** | · | · | · | · | · | · | · | 19 | · | 1 |
| **8** | · | · | · | · | · | · | · | · | 20 | · |
| **9** | · | · | · | · | · | · | · | · | · | 20 |

## Pary pomyłek (prawdziwa → przewidziana)

- 1 → 6: 1
- 3 → 9: 1
- 7 → 9: 1

Szczegóły pomyłek: `test_errors.csv`, plansza: `test_errors.png`, wszystkie przewidywania: `test_predictions.csv`.
