"""Krok 8 planu: zapis modelu i wczytanie bez ponownego uczenia."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from src.data import ROOT, load_config, load_split, train_sha256
from src.features import make_analyzer
from src.models import logreg_pipeline

MODEL = ROOT / "models" / "spam_model.joblib"
META = ROOT / "models" / "spam_model.json"


class TestRoundTrip(unittest.TestCase):
    def test_loaded_pipeline_predicts_the_same(self):
        cfg = load_config()
        train = load_split(cfg, "train")
        X = [r["text"] for r in train]
        y = [1 if r["label"] == "spam" else 0 for r in train]
        pipe = logreg_pipeline(cfg, 1.0).fit(X, y)
        before = pipe.predict_proba(X)[:, 1]
        with tempfile.TemporaryDirectory() as d:
            joblib.dump(pipe, Path(d) / "m.joblib")
            loaded = joblib.load(Path(d) / "m.joblib")
        np.testing.assert_array_equal(loaded.predict_proba(X)[:, 1], before)


@unittest.skipUnless(MODEL.exists(), "brak models/spam_model.joblib - uruchom python -m src.train")
class TestSavedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.pipe = joblib.load(MODEL)
        cls.saved = json.loads(META.read_text(encoding="utf-8"))
        cls.texts = [r["text"] for r in load_split(cls.cfg, "train")[:20]]

    def test_metadata(self):
        m = self.saved["meta"]
        self.assertEqual(m["train_sha256"], train_sha256(self.cfg))
        self.assertEqual(m["cleaning_version"], self.cfg["cleaning"]["version"])
        for key in ("threshold", "param", "versions", "drop_tokens"):
            self.assertIn(key, m)
        self.assertTrue(0 < m["threshold"] < 1)

    def test_frozen_threshold_is_max_over_seeds(self):
        """Odstępstwo uzgodnione z użytkownikiem: próg = najwyższy z progów dla 5 ziaren CV."""
        cv = json.loads((ROOT / "results" / "cv_report.json").read_text(encoding="utf-8"))
        seeds = cv["stability"]["seeds"]
        self.assertEqual(len(seeds), 5)
        self.assertEqual(self.saved["meta"]["threshold"], max(s["threshold"] for s in seeds))
        for s in seeds:   # przy zamrożonym progu żadne ziarno nie przekracza 1 FP
            self.assertLessEqual(s["at_frozen_threshold"]["fp"], 1)

    def test_json_parameters_reproduce_joblib(self):
        """Słownik, IDF, wagi i wyraz wolny z JSON-a dają te same prawdopodobieństwa co pipeline."""
        m, p = self.saved["meta"], self.saved["params"]
        if m["model"] != "logreg":
            self.skipTest("eksport JSON sprawdzany dla regresji logistycznej")
        an = make_analyzer(self.cfg, m["drop_tokens"])
        vocab, idf, coef = p["vocabulary"], np.array(p["idf"]), np.array(p["coef"])
        expected = self.pipe.predict_proba(self.texts)[:, 1]
        for text, exp in zip(self.texts, expected):
            tf = Counter(t for t in an(text) if t in vocab)
            v = np.zeros(len(vocab))
            for t, c in tf.items():
                v[vocab[t]] = (1 + math.log(c)) * idf[vocab[t]]   # sublinear_tf
            n = np.linalg.norm(v)
            z = (v / n if n else v) @ coef + p["intercept"]
            self.assertAlmostEqual(1 / (1 + math.exp(-z)), exp, places=10)

    def test_predict_cli_loads_file_without_training(self):
        mtime = MODEL.stat().st_mtime_ns
        text = self.texts[0]
        out = subprocess.run([sys.executable, "-m", "src.predict"], input=text, capture_output=True,
                             text=True, cwd=ROOT, check=True)
        res = json.loads(out.stdout)
        p = float(self.pipe.predict_proba([text])[0, 1])
        self.assertAlmostEqual(res["p_spam"], p, places=4)
        self.assertEqual(res["label"], "spam" if p >= self.saved["meta"]["threshold"] else "ham")
        self.assertEqual(MODEL.stat().st_mtime_ns, mtime)   # plik modelu nienaruszony
        src = (ROOT / "src" / "predict.py").read_text(encoding="utf-8")
        self.assertNotIn(".fit(", src)


if __name__ == "__main__":
    unittest.main()
