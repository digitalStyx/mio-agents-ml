"""Kroki 3 i 5 planu: grupy kopii nie są rozbijane między części walidacji krzyżowej."""
import json
import unittest
from collections import defaultdict

from src.data import ROOT, load_config, load_split
from src.groups import train_groups, validation_copies
from src.train import splitter

# Plan, krok 3: 6 spamów walidacyjnych to warianty spamów z train
PLAN_VALIDATION_COPIES = {"27bcc632c0a2fc78", "eec728149e9886d7", "5587807b85eebb8d",
                          "182f0590841142c4", "03d402d1641ab499", "8f6272eaef144851"}


class TestGroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.train = load_split(cls.cfg, "train")
        cls.y = [1 if r["label"] == "spam" else 0 for r in cls.train]
        cls.groups = train_groups(cls.cfg, cls.train)
        cls.pairs = json.loads((ROOT / cls.cfg["paths"]["near_duplicates"]).read_text("utf-8"))["pairs"]

    def test_approved_pairs_share_group(self):
        gid = {r["message_id"]: g for r, g in zip(self.train, self.groups)}
        decisions = self.cfg["near_duplicates"]["cv_group_decisions"]
        approved = [p for p in self.pairs if p["kind"] == "train-train" and p["decision"] in decisions]
        self.assertTrue(approved)
        for p in approved:
            self.assertEqual(gid[p["a"]], gid[p["b"]], (p["a"], p["b"]))

    def test_no_group_split_across_folds(self):
        X = [r["text"] for r in self.train]
        seeds = {self.cfg["seed"], *self.cfg["cv"]["stability_seeds"]}
        for seed in seeds:
            fold_of = defaultdict(set)
            for k, (_, test_idx) in enumerate(splitter(self.cfg, seed).split(X, self.y, self.groups)):
                for i in test_idx:
                    fold_of[self.groups[i]].add(k)
            split = {g: f for g, f in fold_of.items() if len(f) > 1}
            self.assertEqual(split, {}, f"ziarno {seed}")
            self.assertEqual(len(fold_of), len(set(self.groups)))   # każdy mail w jakiejś części

    def test_folds_are_five_and_stratified(self):
        X = [r["text"] for r in self.train]
        folds = list(splitter(self.cfg, self.cfg["seed"]).split(X, self.y, self.groups))
        self.assertEqual(len(folds), 5)
        for _, te in folds:
            self.assertGreater(sum(self.y[i] for i in te), 0)

    def test_validation_copies_match_plan(self):
        self.assertEqual(set(validation_copies(self.cfg)), PLAN_VALIDATION_COPIES)


if __name__ == "__main__":
    unittest.main()
