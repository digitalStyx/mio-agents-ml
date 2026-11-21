"""Grupy do StratifiedGroupKFold i lista kopii w walidacji, z zatwierdzonych par."""
import json

from .data import ROOT


def _load_pairs(cfg):
    d = json.loads((ROOT / cfg["paths"]["near_duplicates"]).read_text(encoding="utf-8"))
    undecided = [p for p in d["pairs"] if p.get("decision") is None]
    if undecided:
        raise ValueError(f"{len(undecided)} par bez decyzji w {cfg['paths']['near_duplicates']}")
    return d["pairs"]


def train_groups(cfg, train_rows):
    """Numer grupy dla każdego wiersza train (union-find po zatwierdzonych parach)."""
    nd = cfg["near_duplicates"]
    ids = [r["message_id"] for r in train_rows]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in _load_pairs(cfg):
        if (p["kind"] == "train-train" and p["decision"] in nd["cv_group_decisions"]
                and p["a_label"] in nd["group_labels"] and p["b_label"] in nd["group_labels"]):
            parent[find(p["a"])] = find(p["b"])
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in ids]


def validation_copies(cfg):
    """Identyfikatory wiadomości walidacyjnych wyłączanych z wyniku "bez kopii"."""
    excl = cfg["near_duplicates"]["validation_exclude_decisions"]
    return sorted({p["a"] for p in _load_pairs(cfg)
                   if p["kind"] == "validation-train" and p["decision"] in excl and p["a_label"] == "spam"})
