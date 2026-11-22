"""Kolumna `margin_ovo`: wynik klasyfikatora pary (przewidziana, druga kandydatka).

Przeliczenie niezależne od digits.analysis: indeks pary wyznaczany z jawnej listy par
(i, j), i < j, w kolejności, w jakiej SVC zwraca je przy decision_function_shape='ovo'
(dokumentacja scikit-learn); wartość dodatnia oznacza klasę i.
"""

from __future__ import annotations

import copy

import joblib
import numpy as np
import pytest

from conftest import DATA, MODELS, RESULTS, read_csv, require
from digits.data import load_image

PAIRS = [(i, j) for i in range(10) for j in range(i + 1, 10)]


@pytest.fixture(scope="module")
def model():
    return joblib.load(require(MODELS / "model.joblib", "python -m digits.train"))


@pytest.mark.parametrize("split", ["validation", "test"])
def test_margin_ovo_matches_pairwise_svm(model, split):
    preds_path = RESULTS / f"{split}_predictions.csv"
    rows = read_csv(require(preds_path, "python -m digits.train" if split == "validation"
                            else "python -m digits.evaluate"))
    assert "margin_ovo" in rows[0]
    paths = {r["image_id"]: DATA / r["image_path"] for r in read_csv(DATA / f"{split}.csv")}
    X = np.vstack([load_image(paths[r["image_id"]]) for r in rows])

    ovo_model = copy.deepcopy(model)
    ovo_model.decision_function_shape = "ovo"
    ovo = ovo_model.decision_function(X)
    ovr_model = copy.deepcopy(model)
    ovr_model.decision_function_shape = "ovr"
    ovr = ovr_model.decision_function(X)

    for k, r in enumerate(rows):
        p, s = int(r["predicted"]), int(r["second"])
        # Druga kandydatka: najwyższy wynik ovr poza cyfrą przewidzianą.
        others = [c for c in range(10) if c != p]
        assert s == max(others, key=lambda c: ovr[k, c]), r["image_id"]
        i, j = sorted((p, s))
        value = ovo[k, PAIRS.index((i, j))]
        expected = value if p == i else -value
        assert float(r["margin_ovo"]) == pytest.approx(expected, abs=1e-4), r["image_id"]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_error_files_carry_same_margin(split):
    preds = {r["image_id"]: r for r in read_csv(require(RESULTS / f"{split}_predictions.csv", "pipeline"))}
    errors = read_csv(require(RESULTS / f"{split}_errors.csv", "pipeline"))
    for e in errors:
        assert e["margin_ovo"] == preds[e["image_id"]]["margin_ovo"]
        assert e["second"] == preds[e["image_id"]]["second"]
