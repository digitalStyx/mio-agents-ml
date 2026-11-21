"""Wczytanie konfiguracji i danych z kontrolą sum z manifestu."""
import csv
import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(cfg, name):
    """Zwraca listę słowników (message_id, text, label) po weryfikacji z manifestem."""
    path = ROOT / cfg["paths"][name]
    manifest = json.loads((ROOT / cfg["paths"]["manifest"]).read_text(encoding="utf-8"))
    expected = manifest["files"][path.name]
    actual = sha256(path)
    if actual != expected["sha256"]:
        raise ValueError(f"{path.name}: suma SHA-256 {actual} różna od manifestu")
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    n_spam = sum(r["label"] == "spam" for r in rows)
    if (len(rows), n_spam) != (expected["rows"], expected["spam"]):
        raise ValueError(f"{path.name}: liczebności niezgodne z manifestem")
    split = "train" if name == "train" else "validation"
    in_manifest = {m["message_id"] for m in manifest["messages"] if m["split"] == split}
    if {r["message_id"] for r in rows} != in_manifest:
        raise ValueError(f"{path.name}: identyfikatory niezgodne z manifestem")
    return rows


def train_sha256(cfg):
    return sha256(ROOT / cfg["paths"]["train"])
