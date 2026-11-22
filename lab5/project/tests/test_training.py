"""Uczenie i wybór ustawień nie dotykają zbioru testowego; zapisany model działa z terminala."""

from __future__ import annotations

import json
import os
import subprocess

import joblib
import numpy as np
import pytest

from conftest import DATA, MODELS, PROJECT, PY, read_csv, require, run_module
from digits.data import load_image, load_split

AUDIT_RUNNER = r'''
import json, os, sys, runpy
log = []
def hook(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        log.append(os.path.realpath(os.fsdecode(args[0])))
sys.addaudithook(hook)
sys.argv = ["digits.train"] + sys.argv[2:]
try:
    runpy.run_module("digits.train", run_name="__main__")
except SystemExit as e:
    if e.code not in (0, None):
        raise
finally:
    with open(os.environ["OPEN_LOG"], "w") as f:
        json.dump(log, f)
'''


@pytest.fixture(scope="module")
def isolated_training(tmp_path_factory):
    """Pełne uczenie do katalogu tymczasowego z zapisem każdego otwartego pliku."""
    tmp = tmp_path_factory.mktemp("train")
    runner = tmp / "runner.py"
    runner.write_text(AUDIT_RUNNER)
    log = tmp / "opened.json"
    env = {"PYTHONPATH": str(PROJECT), "PATH": "/usr/bin:/bin", "OPEN_LOG": str(log)}
    r = subprocess.run([PY, str(runner), "--", "--models-dir", str(tmp / "models"),
                        "--results-dir", str(tmp / "results")],
                       cwd=PROJECT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    return tmp, set(json.loads(log.read_text()))


def test_training_does_not_read_test_data(isolated_training):
    _, opened = isolated_training
    test_csv = os.path.realpath(DATA / "test.csv")
    test_images = {os.path.realpath(DATA / r["image_path"]) for r in read_csv(DATA / "test.csv")}
    # Kontrola, że hak w ogóle rejestruje odczyty danych.
    assert os.path.realpath(DATA / "train.csv") in opened
    assert os.path.realpath(DATA / "validation.csv") in opened
    assert test_csv not in opened
    assert not (test_images & opened), "uczenie otworzyło obrazy testowe"


def test_training_writes_planned_artifacts(isolated_training):
    tmp, _ = isolated_training
    for p in ["models/model.joblib", "models/model.json", "results/validation_search.csv",
              "results/validation_predictions.csv", "results/validation_errors.csv",
              "results/validation_errors.png"]:
        assert (tmp / p).is_file(), p
    assert not list((tmp / "results").glob("test_*")), "uczenie nie może tworzyć wyników testu"


def _sample_images(per_class: int = 1) -> list:
    """Po `per_class` obrazów każdej cyfry z validation."""
    seen: dict[str, int] = {}
    out = []
    for r in read_csv(DATA / "validation.csv"):
        if seen.get(r["label"], 0) < per_class:
            seen[r["label"]] = seen.get(r["label"], 0) + 1
            out.append(DATA / r["image_path"])
    return out


def _cli_digits(model_path, images, tmp_path) -> list[int]:
    out = []
    for img in images:
        # Kopia poza katalogiem danych: predict nie potrzebuje CSV ani etykiet.
        local = tmp_path / img.name
        local.write_bytes(img.read_bytes())
        r = run_module("digits.predict", str(local), "--model", str(model_path), cwd=tmp_path)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() in {str(d) for d in range(10)}, r.stdout
        assert r.stdout.count("\n") == 1
        out.append(int(r.stdout))
    return out


@pytest.mark.parametrize("which", ["izolowany", "projektowy"])
def test_cli_matches_model_before_saving(which, isolated_training, tmp_path):
    if which == "izolowany":
        models_dir = isolated_training[0] / "models"
    else:
        models_dir = require(MODELS, "python -m digits.train")
        require(models_dir / "model.joblib", "python -m digits.train")
    meta = json.loads((models_dir / "model.json").read_text())

    # Model „przed zapisem”: te same ustawienia, uczony w pamięci na train.
    from digits.train import make_knn, make_svm
    make = make_svm if meta["model"] == "svm" else make_knn
    train = load_split(DATA, "train")
    in_memory = make(**meta["params"]).fit(train.X, train.y)

    images = _sample_images(per_class=2)
    X = np.vstack([load_image(p) for p in images])
    expected = [int(d) for d in in_memory.predict(X)]
    assert _cli_digits(models_dir / "model.joblib", images, tmp_path) == expected
    # Zapisany obiekt wczytany ponownie daje to samo na całym validation.
    val = load_split(DATA, "validation")
    assert np.array_equal(joblib.load(models_dir / "model.joblib").predict(val.X), in_memory.predict(val.X))


def test_evaluate_refuses_second_run(tmp_path):
    (tmp_path / "test_report.json").write_text("{}")
    r = run_module("digits.evaluate", "--results-dir", str(tmp_path))
    assert r.returncode != 0
    assert "już istnieje" in r.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["test_report.json"]
