"""Klasyfikacja nowej wiadomości w formacie kolumny `text` ("Subject: ...\\n\\ntreść").

Użycie: .venv/bin/python -m src.predict plik.txt   (lub tekst na stdin)
Model wczytywany przez joblib/pickle - tylko z zaufanego źródła i tą samą wersją scikit-learn.
"""
import json
import sys

import joblib
import sklearn

from .data import ROOT, load_config


def main():
    cfg = load_config()
    models = ROOT / cfg["paths"]["models"]
    meta = json.loads((models / "spam_model.json").read_text(encoding="utf-8"))["meta"]
    if meta["versions"]["scikit-learn"] != sklearn.__version__:
        sys.exit("Inna wersja scikit-learn niż przy zapisie modelu")
    pipe = joblib.load(models / "spam_model.joblib")
    text = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()
    p = float(pipe.predict_proba([text])[0, 1])
    print(json.dumps({"p_spam": round(p, 4), "threshold": meta["threshold"],
                      "label": "spam" if p >= meta["threshold"] else "ham"}))


if __name__ == "__main__":
    main()
