"""Uczenie na train, wybór ustawień na validation, zapis modelu.

Nie czyta test.csv ani obrazów testowych.

    python -m digits.train [--data-dir ../data] [--models-dir models] [--results-dir results]
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import PIL
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC

from . import CLASSES, DATA_DIR, MODELS_DIR, RESULTS_DIR
from .analysis import error_board, error_rows, top2, write_error_csv, write_predictions
from .data import load_split

# Siatki z planu (część 4) i drabinki do poszerzania, gdy najlepszy wariant wypada na brzegu.
SVM_C = [0.1, 1, 10, 100]
SVM_GAMMA = [0.003, 0.01, 0.03, 0.1, 0.3]
KNN_K = [1, 3, 5, 7, 9]
KNN_WEIGHTS = ["uniform", "distance"]
C_LADDER = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
GAMMA_LADDER = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
K_LADDER = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
TIE_MARGIN = 2  # różnica na validation (w obrazach), poniżej której zostaje SVM
MAX_EXTENSIONS = 5


def make_svm(C: float, gamma: float) -> SVC:
    return SVC(kernel="rbf", C=C, gamma=gamma)


def make_knn(k: int, weights: str) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=k, weights=weights)


def _extend(values: list, ladder: list, best, low: bool = True, high: bool = True) -> list:
    """Dokłada sąsiedni stopień drabinki, jeśli `best` leży na brzegu `values`."""
    values = sorted(values)
    pos = ladder.index
    if high and best == values[-1] and pos(best) + 1 < len(ladder):
        values.append(ladder[pos(best) + 1])
    if low and best == values[0] and pos(best) > 0:
        values.insert(0, ladder[pos(best) - 1])
    return values


def search(train, val, family: str, log: list[dict]) -> dict:
    """Przegląd siatki jednej rodziny modeli z poszerzaniem na brzegach."""
    if family == "svm":
        axes = {"C": list(SVM_C), "gamma": list(SVM_GAMMA)}
        # Remis: mniejsze C, potem mniejsze gamma (model gładszy).
        key = lambda r: (-r["correct"], r["C"], r["gamma"])  # noqa: E731
    else:
        axes = {"k": list(KNN_K), "weights": list(KNN_WEIGHTS)}
        # Remis: większe k, potem uniform.
        key = lambda r: (-r["correct"], -r["k"], KNN_WEIGHTS.index(r["weights"]))  # noqa: E731

    done: dict[tuple, dict] = {}
    for round_no in range(MAX_EXTENSIONS + 1):
        if family == "svm":
            combos = [{"C": c, "gamma": g} for c in axes["C"] for g in axes["gamma"]]
        else:
            combos = [{"k": k, "weights": w} for k in axes["k"] for w in axes["weights"]]
        for params in combos:
            tag = tuple(sorted(params.items()))
            if tag in done:
                continue
            model = make_svm(**params) if family == "svm" else make_knn(**params)
            model.fit(train.X, train.y)
            correct = int((model.predict(val.X) == val.y).sum())
            row = {"model": family, "C": "", "gamma": "", "k": "", "weights": "", **params,
                   "correct": correct, "n": len(val.y), "accuracy": round(correct / len(val.y), 4),
                   "round": round_no}
            done[tag] = row
            log.append(row)
        best = min(done.values(), key=key)
        if family == "svm":
            new = {"C": _extend(axes["C"], C_LADDER, best["C"]),
                   "gamma": _extend(axes["gamma"], GAMMA_LADDER, best["gamma"])}
        else:
            new = {"k": _extend(axes["k"], K_LADDER, best["k"], low=False), "weights": axes["weights"]}
        if new == axes:
            return best
        print(f"  {family}: najlepszy wariant {_params(best)} na brzegu siatki, poszerzam: {new}")
        axes = new
    return best


def _params(row: dict) -> dict:
    keys = ("C", "gamma") if row["model"] == "svm" else ("k", "weights")
    return {k: row[k] for k in keys}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.data_dir, "train")
    val = load_split(args.data_dir, "validation")
    print(f"train: {len(train.y)} obrazów, validation: {len(val.y)} obrazów")

    log: list[dict] = []
    for name, model in [("baseline_most_frequent", DummyClassifier(strategy="most_frequent")),
                        ("baseline_nearest_centroid", NearestCentroid())]:
        with warnings.catch_warnings():
            # NearestCentroid ostrzega o pikselach stałych w klasie (brzegi obrazu) - bez znaczenia tutaj.
            warnings.simplefilter("ignore", UserWarning)
            model.fit(train.X, train.y)
        correct = int((model.predict(val.X) == val.y).sum())
        log.append({"model": name, "C": "", "gamma": "", "k": "", "weights": "", "correct": correct,
                    "n": len(val.y), "accuracy": round(correct / len(val.y), 4), "round": 0})

    best_svm = search(train, val, "svm", log)
    best_knn = search(train, val, "knn", log)

    if best_knn["correct"] - best_svm["correct"] > TIE_MARGIN:
        chosen, reason = best_knn, f"kNN lepszy od SVM o więcej niż {TIE_MARGIN} obrazy na validation"
    else:
        diff = best_knn["correct"] - best_svm["correct"]
        chosen, reason = best_svm, (f"SVM: kNN ma {diff:+d} trafień względem SVM na validation "
                                    f"(przewaga kNN ≤ {TIE_MARGIN}), zostaje podejście główne")

    with open(args.results_dir / "validation_search.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "C", "gamma", "k", "weights", "correct", "n",
                                          "accuracy", "round"])
        w.writeheader()
        w.writerows(log)

    # Zwycięski wariant uczony od nowa, tylko na train, i zamrażany.
    params = _params(chosen)
    model = make_svm(**params) if chosen["model"] == "svm" else make_knn(**params)
    model.fit(train.X, train.y)
    assert list(model.classes_) == CLASSES

    pred, second, margin = top2(model, val.X)
    val_correct = int((pred == val.y).sum())
    assert val_correct == chosen["correct"]

    write_predictions(args.results_dir / "validation_predictions.csv", val.ids, val.y, pred, second, margin)
    rows = error_rows(val.ids, val.y, pred, second, margin, val.X, train.X, train.y, train.ids)
    write_error_csv(rows, args.results_dir / "validation_errors.csv")
    error_board(rows, val.X, train.X, train.y, args.results_dir / "validation_errors.png",
                f"validation: {len(rows)} pomylek / {len(val.y)}")

    model_path = args.models_dir / "model.joblib"
    joblib.dump(model, model_path)
    meta = {
        "model": chosen["model"],
        "estimator": type(model).__name__,
        "params": params,
        "selection_reason": reason,
        "best_svm": {**_params(best_svm), "correct": best_svm["correct"]},
        "best_knn": {**_params(best_knn), "correct": best_knn["correct"]},
        "classes": CLASSES,
        "input": {"format": "PNG", "mode": "L", "size": [8, 8],
                  "features": "64 piksele wiersz po wierszu, wartość/255, bez standaryzacji"},
        "trained_on": {"split": "train", "n_images": len(train.y), "train_csv_sha256": train.csv_sha256},
        "validation": {"n_images": len(val.y), "correct": val_correct, "accuracy": val_correct / len(val.y),
                       "validation_csv_sha256": val.csv_sha256},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "pillow": PIL.__version__},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.models_dir / "model.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print("\nWyniki na validation (trafienia / 200):")
    for r in log:
        p = {k: r[k] for k in ("C", "gamma", "k", "weights") if r[k] != ""}
        print(f"  {r['model']:<26} {str(p):<36} {r['correct']:>3}  ({r['accuracy']:.1%})"
              + (f"  [poszerzenie {r['round']}]" if r["round"] else ""))
    print(f"\nNajlepszy SVM: {_params(best_svm)} -> {best_svm['correct']}")
    print(f"Najlepszy kNN: {_params(best_knn)} -> {best_knn['correct']}")
    print(f"Wybrany: {chosen['model']} {params} ({reason})")
    print(f"Pomyłki na validation: {len(rows)}; zapisano {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
