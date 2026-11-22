"""Rozpoznaje cyfrę z jednego pliku PNG 8×8 i wypisuje ją na stdout.

    python -m digits.predict ścieżka.png [--model models/model.joblib]

Nie uczy modelu i nie potrzebuje etykiet. Plik spoza formatu (inny rozmiar,
tryb koloru, nie PNG) kończy się komunikatem na stderr i kodem wyjścia 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import sklearn

from . import MODELS_DIR
from .data import ImageFormatError, load_image

EXIT_BAD_INPUT = 2
EXIT_NO_MODEL = 3


def load_model(model_path: Path):
    meta_path = model_path.with_suffix(".json")
    if meta_path.is_file():
        trained_with = json.loads(meta_path.read_text())["versions"]["scikit-learn"]
        if trained_with != sklearn.__version__:
            print(f"ostrzeżenie: model zapisano scikit-learn {trained_with}, "
                  f"a uruchomiono {sklearn.__version__}", file=sys.stderr)
    return joblib.load(model_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path, help="plik PNG 8×8, skala szarości (tryb L)")
    ap.add_argument("--model", type=Path, default=MODELS_DIR / "model.joblib")
    args = ap.parse_args(argv)

    try:
        x = load_image(args.image)
    except ImageFormatError as exc:
        print(f"błąd: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    if not args.model.is_file():
        print(f"błąd: brak modelu {args.model}; najpierw uruchom python -m digits.train", file=sys.stderr)
        return EXIT_NO_MODEL

    model = load_model(args.model)
    print(int(model.predict(x.reshape(1, -1))[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
