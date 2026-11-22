"""Metryki, przegląd pomyłek i plansza z pomyłkami."""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.neighbors import NearestNeighbors

from . import CLASSES


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% przedział ufności Wilsona dla proporcji k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _with_shape(model, shape: str):
    """Płytka kopia SVC z innym kształtem decision_function (bez ponownego uczenia)."""
    m = copy.copy(model)
    m.decision_function_shape = shape
    return m


def class_scores(model, X: np.ndarray) -> np.ndarray:
    """Wyniki klas (n, 10) do ustalenia drugiej kandydatki.

    SVM: decision_function w kształcie `ovr`, czyli liczba głosów 45 klasyfikatorów par
    plus pewność ściśnięta do (-1/3, 1/3). Nadaje się do rankingu, ale nie jako margines.
    kNN: predict_proba.
    """
    if hasattr(model, "decision_function"):
        scores = _with_shape(model, "ovr").decision_function(X)
    else:
        scores = model.predict_proba(X)
    assert list(model.classes_) == CLASSES
    return np.asarray(scores)


def pair_margin(model, X: np.ndarray, pred: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Margines `margin_ovo`: wynik klasyfikatora pary (przewidziana, druga).

    SVC w kształcie `ovo` zwraca kolumny dla par (i, j), i < j, w kolejności
    leksykograficznej; wartość dodatnia oznacza klasę i. Znak odwracamy tak, żeby
    wartość dodatnia oznaczała cyfrę przewidzianą. Mała wartość = obraz blisko
    granicy między dwiema najlepszymi cyframi. Dla kNN: różnica prawdopodobieństw.
    """
    rows = np.arange(len(X))
    if not hasattr(model, "decision_function"):
        proba = model.predict_proba(X)
        return proba[rows, pred] - proba[rows, second]
    ovo = _with_shape(model, "ovo").decision_function(X)
    k = len(CLASSES)
    lo, hi = np.minimum(pred, second), np.maximum(pred, second)
    col = lo * k - lo * (lo + 1) // 2 + (hi - lo - 1)  # indeks pary (lo, hi)
    value = ovo[rows, col]
    return np.where(pred == lo, value, -value)


def top2(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Przewidziana cyfra, druga najlepsza cyfra i margines `margin_ovo` między nimi."""
    scores = class_scores(model, X)
    pred = np.asarray(model.predict(X))
    order = np.argsort(-scores, axis=1, kind="stable")
    second = np.where(order[:, 0] == pred, order[:, 1], order[:, 0])
    return pred, second, pair_margin(model, X, pred, second)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    m = [[0] * len(CLASSES) for _ in CLASSES]
    for t, p in zip(y_true, y_pred):
        m[int(t)][int(p)] += 1
    return m


def per_class(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    out = []
    for c in CLASSES:
        mask = y_true == c
        n = int(mask.sum())
        k = int((y_pred[mask] == c).sum())
        lo, hi = wilson(k, n)
        out.append({"digit": c, "correct": k, "n": n, "accuracy": k / n if n else None,
                    "ci95": [lo, hi]})
    return out


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    m = confusion(y_true, y_pred)
    pairs = [{"true": t, "predicted": p, "count": m[t][p]}
             for t in CLASSES for p in CLASSES if t != p and m[t][p] > 0]
    return sorted(pairs, key=lambda d: (-d["count"], d["true"], d["predicted"]))


def error_rows(ids, y_true, pred, second, margin, X, X_train, y_train, ids_train,
               n_neighbors: int = 3) -> list[dict]:
    """Pomyłki z 3 najbliższymi obrazami z train (odległość euklidesowa na pikselach)."""
    wrong = np.flatnonzero(pred != y_true)
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_train)
    rows = []
    if len(wrong) == 0:
        return rows
    dist, idx = nn.kneighbors(X[wrong])
    for j, i in enumerate(wrong):
        rows.append({
            "image_id": ids[i],
            "label": int(y_true[i]),
            "predicted": int(pred[i]),
            "second": int(second[i]),
            "margin_ovo": round(float(margin[i]), 4),
            "neighbors": ";".join(f"{ids_train[n]}:{int(y_train[n])}" for n in idx[j]),
            "_index": int(i),
            "_neighbor_index": [int(n) for n in idx[j]],
        })
    rows.sort(key=lambda r: -r["margin_ovo"])  # najpierw najbardziej pewne pomyłki
    return rows


PREDICTION_FIELDS = ["image_id", "label", "predicted", "second", "margin_ovo"]


def write_predictions(path: Path, ids, y_true, pred, second, margin) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PREDICTION_FIELDS)
        for i in range(len(ids)):
            w.writerow([ids[i], int(y_true[i]), int(pred[i]), int(second[i]), round(float(margin[i]), 4)])


def write_error_csv(rows: list[dict], path: Path) -> None:
    fields = ["image_id", "label", "predicted", "second", "margin_ovo", "neighbors"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _tile(vec: np.ndarray, scale: int) -> Image.Image:
    img = Image.fromarray(np.round(vec.reshape(8, 8) * 255).astype(np.uint8), mode="L")
    return img.resize((8 * scale, 8 * scale), Image.Resampling.NEAREST).convert("RGB")


def error_board(rows: list[dict], X: np.ndarray, X_train: np.ndarray, y_train: np.ndarray,
                path: Path, title: str, scale: int = 8) -> None:
    """Plansza: wiersz = pomyłka (prawdziwa -> przewidziana) + 3 najbliższe obrazy z train."""
    tile = 8 * scale
    gap, caption_h, header_h = 10, 16, 40
    col_w = tile + gap
    row_h = tile + caption_h + gap
    width = max(gap + col_w * 4 + 2 * gap, 470)
    height = header_h + max(1, len(rows)) * row_h + gap
    board = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    draw.text((gap, 5), title, fill=(0, 0, 0), font=font)
    draw.text((gap, 20), "czerwona ramka: prawdziwa -> przewidziana, m = margin_ovo; obok 3 najblizsze z train",
              fill=(90, 90, 90), font=font)
    if not rows:
        draw.text((gap, header_h), "brak pomyłek", fill=(0, 0, 0), font=font)
    for r_i, r in enumerate(rows):
        y0 = header_h + r_i * row_h
        x0 = gap
        board.paste(_tile(X[r["_index"]], scale), (x0, y0))
        draw.rectangle([x0 - 2, y0 - 2, x0 + tile + 1, y0 + tile + 1], outline=(200, 0, 0), width=2)
        draw.text((x0, y0 + tile + 2), f"{r['label']} -> {r['predicted']}  m={r['margin_ovo']:.2f}", fill=(200, 0, 0), font=font)
        x0 += col_w + 2 * gap
        for n in r["_neighbor_index"]:
            board.paste(_tile(X_train[n], scale), (x0, y0))
            draw.text((x0, y0 + tile + 2), f"train: {int(y_train[n])}", fill=(60, 60, 60), font=font)
            x0 += col_w
    board.save(path)
