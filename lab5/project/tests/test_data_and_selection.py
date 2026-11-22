"""Zgodność danych z opisem oraz wyboru ustawień z regułą z plan.md (część 4)."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from conftest import DATA, MODELS, RESULTS, manifest, read_csv, require
from digits.data import load_split

# README danych: 600/200/200 obrazów, 60/20/20 każdej cyfry.
EXPECTED = {"train": (600, 60), "validation": (200, 20), "test": (200, 20)}
PLAN_SVM = [(c, g) for c in (0.1, 1, 10, 100) for g in (0.003, 0.01, 0.03, 0.1, 0.3)]
PLAN_KNN = [(k, w) for k in (1, 3, 5, 7, 9) for w in ("uniform", "distance")]
TIE_MARGIN = 2


@pytest.mark.parametrize("split", ["train", "validation"])
def test_split_matches_readme_and_manifest(split):
    s = load_split(DATA, split)  # sprawdza też SHA-256 CSV z manifestu
    n, per = EXPECTED[split]
    assert len(s.y) == n and s.X.shape == (n, 64)
    assert Counter(s.y.tolist()) == {d: per for d in range(10)}
    m = {e["image_id"]: e for e in manifest()["images"]}
    assert all(m[i]["split"] == split and m[i]["label"] == int(y) for i, y in zip(s.ids, s.y))


def test_test_csv_matches_readme():
    rows = read_csv(DATA / "test.csv")
    assert len(rows) == EXPECTED["test"][0]
    assert Counter(r["label"] for r in rows) == {str(d): 20 for d in range(10)}


@pytest.fixture(scope="module")
def search():
    rows = read_csv(require(RESULTS / "validation_search.csv", "python -m digits.train"))
    for r in rows:
        r["correct"] = int(r["correct"])
        r["n"] = int(r["n"])
    return rows


@pytest.fixture(scope="module")
def meta():
    return json.loads(require(MODELS / "model.json", "python -m digits.train").read_text())


def test_search_covers_plan_grid(search):
    svm = {(float(r["C"]), float(r["gamma"])) for r in search if r["model"] == "svm"}
    knn = {(int(r["k"]), r["weights"]) for r in search if r["model"] == "knn"}
    assert set(PLAN_SVM) <= svm
    assert set(PLAN_KNN) <= knn
    assert {r["model"] for r in search} >= {"baseline_most_frequent", "baseline_nearest_centroid"}
    assert all(r["n"] == 200 for r in search)
    # Klasy równoliczne, więc klasa najczęstsza trafia dokładnie 20/200.
    assert next(r for r in search if r["model"] == "baseline_most_frequent")["correct"] == 20


def _best(search, family):
    rows = [r for r in search if r["model"] == family]
    if family == "svm":
        return min(rows, key=lambda r: (-r["correct"], float(r["C"]), float(r["gamma"])))
    return min(rows, key=lambda r: (-r["correct"], -int(r["k"]), ("uniform", "distance").index(r["weights"])))


def test_selection_follows_plan_rule(search, meta):
    svm, knn = _best(search, "svm"), _best(search, "knn")
    if knn["correct"] - svm["correct"] > TIE_MARGIN:
        assert meta["model"] == "knn"
        assert meta["params"] == {"k": int(knn["k"]), "weights": knn["weights"]}
        chosen = knn
    else:
        assert meta["model"] == "svm"
        assert meta["params"] == {"C": float(svm["C"]), "gamma": float(svm["gamma"])}
        chosen = svm
    assert meta["validation"]["correct"] == chosen["correct"]


def test_best_svm_not_on_grid_edge(search):
    """Plan: wynik na brzegu siatki wymaga poszerzenia siatki w tę stronę."""
    svm = [r for r in search if r["model"] == "svm"]
    best = _best(search, "svm")
    cs = sorted({float(r["C"]) for r in svm})
    gs = sorted({float(r["gamma"]) for r in svm})
    assert cs[0] < float(best["C"]) < cs[-1] or float(best["C"]) == cs[0] == 0.001
    assert gs[0] < float(best["gamma"]) < gs[-1]


def test_model_metadata_matches_manifest(meta):
    m = manifest()
    assert meta["trained_on"]["split"] == "train"
    assert meta["trained_on"]["n_images"] == 600
    assert meta["trained_on"]["train_csv_sha256"] == m["splits"]["train"]["sha256"]
    assert meta["validation"]["validation_csv_sha256"] == m["splits"]["validation"]["sha256"]
    assert meta["classes"] == list(range(10))
    assert meta["input"]["size"] == [8, 8] and meta["input"]["mode"] == "L"


def test_validation_predictions_consistent(meta):
    rows = read_csv(require(RESULTS / "validation_predictions.csv", "python -m digits.train"))
    val = read_csv(DATA / "validation.csv")
    assert [r["image_id"] for r in rows] == [r["image_id"] for r in val]
    assert [r["label"] for r in rows] == [r["label"] for r in val]
    correct = sum(r["label"] == r["predicted"] for r in rows)
    assert correct == meta["validation"]["correct"]
    errors = read_csv(RESULTS / "validation_errors.csv")
    assert {e["image_id"] for e in errors} == {r["image_id"] for r in rows if r["label"] != r["predicted"]}
