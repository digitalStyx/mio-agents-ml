"""Wczytywanie obrazów i podziałów danych.

`load_image` jest jedyną drogą od pliku PNG do wektora cech: używa jej uczenie,
ocena i CLI, więc nowe obrazy przechodzą to samo przygotowanie co dane uczące.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = (8, 8)
N_FEATURES = IMAGE_SIZE[0] * IMAGE_SIZE[1]
ALLOWED_SPLITS = ("train", "validation", "test")


class ImageFormatError(ValueError):
    """Plik nie jest obrazem w formacie wejścia (PNG, tryb L, 8×8)."""


def load_image(path: str | Path) -> np.ndarray:
    """Zwraca wektor 64 liczb z [0, 1] (wiersz po wierszu, wartość/255).

    Nie skaluje, nie konwertuje kolorów i nie odwraca jasności: każdy plik
    spoza formatu kończy się ImageFormatError.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageFormatError(f"nie ma takiego pliku: {path}")
    try:
        with Image.open(path) as img:
            fmt, mode, size = img.format, img.mode, img.size
            if fmt != "PNG":
                raise ImageFormatError(f"{path}: oczekiwano PNG, jest {fmt}")
            if mode != "L":
                raise ImageFormatError(
                    f"{path}: oczekiwano 8-bitowej skali szarości (tryb L), jest tryb {mode}"
                )
            if size != IMAGE_SIZE:
                raise ImageFormatError(f"{path}: oczekiwano 8×8 pikseli, jest {size[0]}×{size[1]}")
            pixels = np.asarray(img, dtype=np.float64)
    except UnidentifiedImageError as exc:
        raise ImageFormatError(f"{path}: to nie jest obraz PNG") from exc
    except OSError as exc:  # uszkodzony lub ucięty plik
        raise ImageFormatError(f"{path}: nie da się odczytać obrazu ({exc})") from exc
    return pixels.reshape(N_FEATURES) / 255.0


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Split:
    name: str
    ids: list[str]
    paths: list[Path]
    X: np.ndarray
    y: np.ndarray
    csv_sha256: str


def load_split(data_dir: str | Path, name: str) -> Split:
    """Wczytuje jeden podział i sprawdza sumę CSV z manifest.json."""
    if name not in ALLOWED_SPLITS:
        raise ValueError(f"nieznany podział: {name}")
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{name}.csv"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    expected = manifest["splits"][name]["sha256"]
    raw = csv_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{csv_path}: suma SHA-256 {actual} różni się od manifestu {expected}")

    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    ids = [r["image_id"] for r in rows]
    paths = [data_dir / r["image_path"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    X = np.vstack([load_image(p) for p in paths])
    return Split(name=name, ids=ids, paths=paths, X=X, y=y, csv_sha256=actual)
