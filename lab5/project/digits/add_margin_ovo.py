"""Jednorazowe uzupełnienie wyników o kolumnę `margin_ovo` z zapisanego modelu.

    python -m digits.add_margin_ovo

Pierwsza wersja zapisywała kolumnę `margin` jako różnicę wyników `ovr` z SVC,
czyli głównie różnicę liczby głosów (prawie zawsze ~1), bez informacji o tym,
czy przypadek jest graniczny. Skrypt:

- validation: przepisuje `validation_predictions.csv`, `validation_errors.csv`
  i planszę w nowym formacie, takim jak tworzy teraz `digits.train`;
- test: do `test_predictions.csv` dopisuje kolumnę `margin_ovo` (pozostałe kolumny,
  w tym stara `margin`, zostają bez zmian), a `test_errors.csv` i planszę tworzy od nowa.

Nie uczy modelu, nie zmienia `model.joblib` ani raportu z testu. Przerywa, jeśli
przeliczona cyfra lub druga kandydatka różni się od zapisanej w pliku.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import numpy as np

from . import DATA_DIR, MODELS_DIR, RESULTS_DIR
from .analysis import error_board, error_rows, top2, write_error_csv, write_predictions
from .data import load_split


def _check(rows: list[dict], split, pred, second) -> None:
    if [r["image_id"] for r in rows] != split.ids:
        sys.exit(f"{split.name}: kolejność image_id w pliku różni się od CSV")
    for r, p, s in zip(rows, pred, second):
        if int(r["predicted"]) != int(p) or int(r["second"]) != int(s):
            sys.exit(f"{split.name}: {r['image_id']} przeliczony inaczej niż w pliku, przerywam")


def main() -> int:
    model = joblib.load(MODELS_DIR / "model.joblib")
    train = load_split(DATA_DIR, "train")

    # validation: nowy format, ten sam co z digits.train
    val = load_split(DATA_DIR, "validation")
    path = RESULTS_DIR / "validation_predictions.csv"
    old = list(csv.DictReader(open(path, newline="")))
    pred, second, margin = top2(model, val.X)
    _check(old, val, pred, second)
    write_predictions(path, val.ids, val.y, pred, second, margin)
    rows = error_rows(val.ids, val.y, pred, second, margin, val.X, train.X, train.y, train.ids)
    write_error_csv(rows, RESULTS_DIR / "validation_errors.csv")
    error_board(rows, val.X, train.X, train.y, RESULTS_DIR / "validation_errors.png",
                f"validation: {len(rows)} pomylek / {len(val.y)}")

    # test: tylko dopisanie kolumny; cyfry i raport bez zmian
    test = load_split(DATA_DIR, "test")
    path = RESULTS_DIR / "test_predictions.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields, old = list(reader.fieldnames), list(reader)
    if "margin_ovo" in fields:
        print("test_predictions.csv ma już kolumnę margin_ovo, pomijam test")
        return 0
    pred, second, margin = top2(model, test.X)
    _check(old, test, pred, second)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["margin_ovo"])
        w.writeheader()
        for r, m in zip(old, margin):
            w.writerow({**r, "margin_ovo": round(float(m), 4)})
    y = np.asarray(test.y)
    rows = error_rows(test.ids, y, pred, second, margin, test.X, train.X, train.y, train.ids)
    write_error_csv(rows, RESULTS_DIR / "test_errors.csv")
    error_board(rows, test.X, train.X, train.y, RESULTS_DIR / "test_errors.png",
                f"test: {len(rows)} pomylek / {len(y)}")
    print("dopisano margin_ovo: validation (nowy format), test (nowa kolumna)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
