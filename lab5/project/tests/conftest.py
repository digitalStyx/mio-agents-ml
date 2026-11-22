"""Wspólne ścieżki i pomocnicze funkcje testów.

Oczekiwane wartości pochodzą z README danych, manifest.json i plan.md,
a nie z tego, co zwraca kod projektu.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT.parent / "data"
MODELS = PROJECT / "models"
RESULTS = PROJECT / "results"
PY = sys.executable


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def manifest() -> dict:
    return json.loads((DATA / "manifest.json").read_text())


def run_module(module: str, *args: str, cwd: Path = PROJECT, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-m", module, *args], cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env={"PYTHONPATH": str(PROJECT), "PATH": "/usr/bin:/bin"})


def require(path: Path, step: str) -> Path:
    if not path.exists():
        pytest.skip(f"brak {path.relative_to(PROJECT)}: najpierw uruchom {step}")
    return path
