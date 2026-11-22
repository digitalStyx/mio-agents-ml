"""Jednorazowa ocena zamrożonego modelu na zbiorze testowym.

    python -m digits.evaluate [--data-dir ../data] [--models-dir models] [--results-dir results]

Odmawia uruchomienia, jeśli w katalogu wyników jest już test_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import NearestCentroid

from . import CLASSES, DATA_DIR, MODELS_DIR, RESULTS_DIR
from .analysis import (confusion, error_board, error_rows, per_class, top2, top_confusions,
                       wilson, write_error_csv, write_predictions)
from .data import load_split
from .predict import load_model


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_markdown(report: dict, path: Path) -> None:
    a = report["accuracy"]
    lines = [
        "# Ocena na zbiorze testowym",
        "",
        f"Model: {report['model']['estimator']} {report['model']['params']}, uczony na train "
        f"({report['model']['trained_on_n']} obrazów). Test: {a['n']} obrazów z `optdigits.tes` "
        "(inni autorzy pisma niż w train). Ocena wykonana raz.",
        "",
        "| Model | Trafienia | Accuracy | 95% CI (Wilson) |",
        "|---|---|---|---|",
        f"| **wybrany model** | {a['correct']}/{a['n']} | {_pct(a['value'])} | "
        f"{_pct(a['ci95'][0])} – {_pct(a['ci95'][1])} |",
    ]
    for b in report["baselines"]:
        lines.append(f"| {b['name']} | {b['correct']}/{b['n']} | {_pct(b['accuracy'])} | "
                     f"{_pct(b['ci95'][0])} – {_pct(b['ci95'][1])} |")
    v = report["validation"]
    lines += [
        "",
        f"Dla porównania validation: {v['correct']}/{v['n']} ({_pct(v['accuracy'])}). "
        "Validation pochodzi od tych samych autorów co train, więc jest optymistyczna.",
        "",
        "Jedna pomyłka na teście to 0,5 p.p.",
        "",
        "## Trafność dla każdej cyfry",
        "",
        "| Cyfra | Trafienia | 95% CI |",
        "|---|---|---|",
    ]
    for c in report["per_class"]:
        lines.append(f"| {c['digit']} | {c['correct']}/{c['n']} | {_pct(c['ci95'][0])} – {_pct(c['ci95'][1])} |")
    lines += ["", "## Macierz pomyłek (wiersze: prawdziwa, kolumny: przewidziana)", "",
              "| | " + " | ".join(str(c) for c in CLASSES) + " |",
              "|---" * (len(CLASSES) + 1) + "|"]
    for t, row in enumerate(report["confusion_matrix"]):
        lines.append(f"| **{t}** | " + " | ".join(str(v) if v else "·" for v in row) + " |")
    lines += ["", "## Pary pomyłek (prawdziwa → przewidziana)", ""]
    if report["top_confusions"]:
        lines += [f"- {p['true']} → {p['predicted']}: {p['count']}" for p in report["top_confusions"]]
    else:
        lines.append("- brak")
    lines += ["", "Szczegóły pomyłek: `test_errors.csv`, plansza: `test_errors.png`, "
              "wszystkie przewidywania: `test_predictions.csv`.", ""]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)

    report_path = args.results_dir / "test_report.json"
    existing = [p for p in (report_path, args.results_dir / "test_predictions.csv") if p.exists()]
    if existing:
        print(f"błąd: {existing[0]} już istnieje. Test ocenia się raz; "
              "jeśli zmieniasz model po teście, opisz to i usuń raport świadomie.", file=sys.stderr)
        return 1

    meta = json.loads((args.models_dir / "model.json").read_text())
    model = load_model(args.models_dir / "model.joblib")
    train = load_split(args.data_dir, "train")
    test = load_split(args.data_dir, "test")

    pred, second, margin = top2(model, test.X)
    correct = int((pred == test.y).sum())
    n = len(test.y)

    baselines = []
    for name, base in [("klasa najczęstsza", DummyClassifier(strategy="most_frequent")),
                       ("najbliższy centroid", NearestCentroid())]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # NearestCentroid: piksele stałe w klasie
            base.fit(train.X, train.y)
        k = int((base.predict(test.X) == test.y).sum())
        baselines.append({"name": name, "correct": k, "n": n, "accuracy": k / n, "ci95": list(wilson(k, n))})

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.results_dir / "test_predictions.csv", test.ids, test.y, pred, second, margin)
    rows = error_rows(test.ids, test.y, pred, second, margin, test.X, train.X, train.y, train.ids)
    write_error_csv(rows, args.results_dir / "test_errors.csv")
    error_board(rows, test.X, train.X, train.y, args.results_dir / "test_errors.png",
                f"test: {len(rows)} pomylek / {n}")

    report = {
        "model": {"estimator": meta["estimator"], "params": meta["params"],
                  "trained_on_n": meta["trained_on"]["n_images"],
                  "train_csv_sha256": meta["trained_on"]["train_csv_sha256"]},
        "test_csv_sha256": test.csv_sha256,
        "accuracy": {"correct": correct, "n": n, "value": correct / n, "ci95": list(wilson(correct, n))},
        "errors": n - correct,
        "per_class": per_class(test.y, pred),
        "confusion_matrix": confusion(test.y, pred),
        "top_confusions": top_confusions(test.y, pred),
        "baselines": baselines,
        "validation": {k: meta["validation"][k] for k in ("correct", "n_images", "accuracy")}
                      | {"n": meta["validation"]["n_images"]},
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_markdown(report, args.results_dir / "test_report.md")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lo, hi = report["accuracy"]["ci95"]
    print(f"test: {correct}/{n} = {correct / n:.1%} (95% CI {lo:.1%} – {hi:.1%}), pomyłek: {n - correct}")
    print(f"raport: {args.results_dir / 'test_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
