"""Wyniki oceny na teście: plik przewidywań, zgodność z modelem i z raportem.

Wymagają wcześniejszego jednorazowego `python -m digits.evaluate`.
"""

from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pytest

from conftest import DATA, MODELS, RESULTS, manifest, read_csv, require
from digits.data import load_image


def wilson(k, n, z=1.959963984540054):
    # Wzór Wilsona zapisany niezależnie od digits.analysis.
    p = k / n
    a = p + z * z / (2 * n)
    b = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return (a - b) / d, (a + b) / d


@pytest.fixture(scope="module")
def preds():
    return read_csv(require(RESULTS / "test_predictions.csv", "python -m digits.evaluate"))


@pytest.fixture(scope="module")
def report():
    return json.loads(require(RESULTS / "test_report.json", "python -m digits.evaluate").read_text())


@pytest.fixture(scope="module")
def test_rows():
    return read_csv(DATA / "test.csv")


def test_predictions_cover_test_csv(preds, test_rows):
    assert len(preds) == 200
    assert [p["image_id"] for p in preds] == [r["image_id"] for r in test_rows]
    assert len({p["image_id"] for p in preds}) == 200
    assert [p["label"] for p in preds] == [r["label"] for r in test_rows]
    assert {p["predicted"] for p in preds} <= {str(d) for d in range(10)}
    test_ids = {e["image_id"] for e in manifest()["images"] if e["split"] == "test"}
    assert {p["image_id"] for p in preds} == test_ids


def test_saved_model_reproduces_each_prediction(preds, test_rows):
    model = joblib.load(MODELS / "model.joblib")
    path = {r["image_id"]: DATA / r["image_path"] for r in test_rows}
    X = np.vstack([load_image(path[p["image_id"]]) for p in preds])
    again = model.predict(X)
    for p, d in zip(preds, again):
        assert int(p["predicted"]) == int(d), p["image_id"]


def test_report_accuracy_matches_predictions(preds, report):
    correct = sum(p["label"] == p["predicted"] for p in preds)
    acc = report["accuracy"]
    assert acc["n"] == 200
    assert acc["correct"] == correct
    assert report["errors"] == 200 - correct
    assert acc["value"] == pytest.approx(correct / 200)
    assert acc["ci95"] == pytest.approx(list(wilson(correct, 200)))


def test_report_per_class_and_confusion_match_predictions(preds, report):
    cm = [[0] * 10 for _ in range(10)]
    for p in preds:
        cm[int(p["label"])][int(p["predicted"])] += 1
    assert report["confusion_matrix"] == cm
    assert sum(map(sum, cm)) == 200
    for c in report["per_class"]:
        d = c["digit"]
        assert c["n"] == 20 == sum(cm[d])
        assert c["correct"] == cm[d][d]
        assert c["ci95"] == pytest.approx(list(wilson(cm[d][d], 20)))
    pairs = {(t, q): cm[t][q] for t in range(10) for q in range(10) if t != q and cm[t][q]}
    assert {(x["true"], x["predicted"]): x["count"] for x in report["top_confusions"]} == pairs


def test_report_markdown_shows_same_numbers(report):
    md = require(RESULTS / "test_report.md", "python -m digits.evaluate").read_text()
    acc = report["accuracy"]
    assert f"{acc['correct']}/200" in md
    assert f"{100 * acc['value']:.1f}%" in md
    for c in report["per_class"]:
        assert f"| {c['digit']} | {c['correct']}/20 |" in md


def test_error_files_match_predictions(preds):
    errors = read_csv(require(RESULTS / "test_errors.csv", "python -m digits.evaluate"))
    wrong = {p["image_id"]: p for p in preds if p["label"] != p["predicted"]}
    assert {e["image_id"] for e in errors} == set(wrong)
    for e in errors:
        assert e["label"] == wrong[e["image_id"]]["label"]
        assert e["predicted"] == wrong[e["image_id"]]["predicted"]
    assert (RESULTS / "test_errors.png").is_file()


def test_baseline_majority_is_ten_percent(report):
    # Klasy w teście są równoliczne (README), więc klasa najczęstsza trafia 20/200.
    b = next(b for b in report["baselines"] if b["name"] == "klasa najczęstsza")
    assert b["correct"] == 20 and b["n"] == 200
