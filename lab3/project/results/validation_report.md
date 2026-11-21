# Ocena na validation (jednorazowa)

Model logreg (parametr 1.0), próg 0.5096, reguły czyszczenia v2. Ocena: 2026-11-21T18:23:38+00:00.

## Pełna walidacja (60 wiadomości)

| | przewidziany spam | przewidziany ham |
|---|---|---|
| spam | 19 | 1 |
| ham | 0 | 40 |

- czułość (spam): 19/20 = 95.0% (95% CI 76.4-99.1%)
- precyzja (spam): 19/19 = 100.0% (95% CI 83.2-100.0%)
- odsetek fałszywych alarmów: 0/40 = 0.0% (95% CI 0.0-8.8%)
- trafność: 59/60 = 98.3% (95% CI 91.1-99.7%); zawsze ham: 40/60 = 66.7% (95% CI 54.1-77.3%)

## Bez kopii kampanii (54 wiadomości)

| | przewidziany spam | przewidziany ham |
|---|---|---|
| spam | 13 | 1 |
| ham | 0 | 40 |

- czułość (spam): 13/14 = 92.9% (95% CI 68.5-98.7%)
- precyzja (spam): 13/13 = 100.0% (95% CI 77.2-100.0%)
- odsetek fałszywych alarmów: 0/40 = 0.0% (95% CI 0.0-8.8%)
- trafność: 53/54 = 98.1% (95% CI 90.2-99.7%); zawsze ham: 40/54 = 74.1% (95% CI 61.1-83.9%)

Hamy ze wspólnym wątkiem w train (3, informacyjnie): fałszywe alarmy 0.

Spamy od nadawcy znanego z train, inna oferta (2, w obu wynikach, informacyjnie): wykryte 2.

## Pomyłki

- `017f0817d4c92951` spam -> ham, p=0.456
