# Walidacja krzyżowa na train

Utworzono: 2026-11-21T18:23:06+00:00. Grup w CV: 163. Reguła progu: <= 1 FP w predykcjach out-of-fold.

## Punkty odniesienia

- zawsze ham: TP 0/60
- słowa kluczowe, lista 10: TP 30, FP 0, k >= 6.5
- słowa kluczowe, lista 20: TP 18, FP 1, k >= 12.5
- słowa kluczowe, lista 30: TP 13, FP 1, k >= 18.5
- słowa kluczowe, lista 50: TP 7, FP 1, k >= 29.5

## Siatka parametrów

| model | parametr | próg | TP | FP |
|---|---|---|---|---|
| logreg | 0.01 | 0.5014 | 58/60 | 1/120 |
| logreg | 0.03 | 0.5019 | 57/60 | 1/120 |
| logreg | 0.1 | 0.4989 | 57/60 | 1/120 |
| logreg | 0.3 | 0.4887 | 57/60 | 1/120 |
| logreg | 1.0 | 0.4677 | 57/60 | 1/120 |
| logreg | 3.0 | 0.4422 | 57/60 | 1/120 |
| logreg | 10.0 | 0.4099 | 57/60 | 1/120 |
| logreg | 30.0 | 0.3789 | 57/60 | 1/120 |
| logreg | 100.0 | 0.3451 | 57/60 | 1/120 |
| naive_bayes | 0.01 | 0.9858 | 54/60 | 1/120 |
| naive_bayes | 0.03 | 0.9854 | 55/60 | 1/120 |
| naive_bayes | 0.1 | 0.9724 | 55/60 | 1/120 |
| naive_bayes | 0.3 | 0.9132 | 57/60 | 1/120 |
| naive_bayes | 1.0 | 0.8357 | 57/60 | 1/120 |

## Najsilniejsze wagi modelu końcowego

Spam: `your` +1.12, `you` +1.04, `our` +1.02, `please` +0.903, `$` +0.851, `free` +0.835, `click` +0.793, `remove` +0.722, `we` +0.709, `money` +0.685, `will` +0.594, `email` +0.57, `from` +0.516, `here` +0.473, `membership` +0.458, `http` +0.455, `days` +0.451, `below` +0.447, `no` +0.443, `o` +0.437, `name` +0.435, `access` +0.429, `receive` +0.427, `rates` +0.426, `vip` +0.425, `go` +0.417, `q` +0.417, `send` +0.413, `million` +0.407, `visit` +0.406

Ham: `it` -0.494, `org` -0.471, `i'm` -0.471, `but` -0.434, `they` -0.41, `the` -0.404, `subj_the` -0.396, `that` -0.374, `spam` -0.374, `a` -0.354, `rpm` -0.348, `say` -0.348, `subj_a` -0.345, `server` -0.334, `i` -0.329, `html` -0.312, `some` -0.309, `is` -0.302, `would` -0.286, `folder` -0.282, `there` -0.278, `sep` -0.275, `had` -0.271, `still` -0.27, `researchers` -0.263, `doesn't` -0.256, `trying` -0.248, `red` -0.246, `could` -0.244, `i'd` -0.242

Ślady korpusu w czołówce: brak

Model przed usunięciem słów rozmowy, ham: `subj_re` -0.011, `wrote` -0.0104, `org` -0.00839, `i'm` -0.00838, `it` -0.00794, `but` -0.00776, `subj_the` -0.00757, `they` -0.00669, `rpm` -0.0066, `subj_a` -0.00653, `spam` -0.00632, `the` -0.00628, `say` -0.00627, `that` -0.00596, `i` -0.00583

## Słowa formy rozmowy

W czołówce wag: ['subj_re', 'wrote']. Z nimi: TP 58, FP 1 (logreg 0.01). Bez nich: TP 57, FP 1 (logreg 1.0). Decyzja: bez słów rozmowy.

## Wybrany model

logreg, parametr 1.0, próg 0.5096, TP 55/60, FP 1/120. Przewaga nad regułą słów: 25 spamów (wyraźna).

## Stabilność progu

Próg z ziarna głównego: 0.4695. Zamrożony próg (max_over_seeds, odstępstwo od planu): 0.5096.

| ziarno | próg | TP | FP | TP/FP przy zamrożonym progu |
|---|---|---|---|---|
| 2026 | 0.4695 | 57 | 1 | 55/1 |
| 2027 | 0.5096 | 55 | 1 | 55/1 |
| 2028 | 0.4939 | 57 | 1 | 55/1 |
| 2029 | 0.4929 | 55 | 1 | 54/1 |
| 2030 | 0.4931 | 57 | 1 | 55/0 |

Części (ziarno główne), przy zamrożonym progu: #0: TP 12/12, FP 1/24; #1: TP 12/12, FP 0/24; #2: TP 11/12, FP 0/24; #3: TP 11/12, FP 0/24; #4: TP 9/12, FP 0/24
