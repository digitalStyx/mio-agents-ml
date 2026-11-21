"""Dane wejściowe zgodne z manifestem; przewidywania i miary dla validation (60 i 54 wiadomości)."""
import csv
import json
import unittest

from src.data import ROOT, load_config, load_split
from src.metrics import report

RESULTS = ROOT / "results"
PRED = RESULTS / "validation_predictions.csv"
REPORT = RESULTS / "validation_report.json"
MANIFEST = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
PLAN_VALIDATION_COPIES = {"27bcc632c0a2fc78", "eec728149e9886d7", "5587807b85eebb8d",
                          "182f0590841142c4", "03d402d1641ab499", "8f6272eaef144851"}
VALIDATION_IDS = {m["message_id"] for m in MANIFEST["messages"] if m["split"] == "validation"}


class TestData(unittest.TestCase):
    def test_splits_match_manifest_and_plan(self):
        cfg = load_config()   # load_split sprawdza sumy SHA-256 i identyfikatory z manifestem
        for name, (n, ham, spam) in {"train": (180, 120, 60), "validation": (60, 40, 20)}.items():
            rows = load_split(cfg, name)
            labels = [r["label"] for r in rows]
            self.assertEqual((len(rows), labels.count("ham"), labels.count("spam")), (n, ham, spam))
            f = MANIFEST["files"][f"{name}.csv"]
            self.assertEqual((f["rows"], f["ham"], f["spam"]), (n, ham, spam))


@unittest.skipUnless(PRED.exists() and REPORT.exists(),
                     "brak wyników oceny - uruchom python -m src.evaluate")
class TestValidationOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(PRED, encoding="utf-8", newline="") as f:
            cls.rows = list(csv.DictReader(f))
        cls.rep = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.labels = {r["message_id"]: r["label"] for r in load_split(load_config(), "validation")}

    def test_predictions_file(self):
        self.assertEqual(len(self.rows), 60)
        ids = [r["message_id"] for r in self.rows]
        self.assertEqual(len(set(ids)), 60)
        self.assertEqual(set(ids), VALIDATION_IDS)
        for r in self.rows:
            self.assertEqual(r["label"], self.labels[r["message_id"]])
            self.assertIn(r["pred"], ("spam", "ham"))
            self.assertTrue(0.0 <= float(r["p_spam"]) <= 1.0)
        self.assertEqual({r["message_id"] for r in self.rows if r["campaign_copy"] == "1"},
                         PLAN_VALIDATION_COPIES)

    def test_measures_for_60_and_54(self):
        full, wo = self.rep["full"], self.rep["without_copies"]
        self.assertEqual(full["accuracy"]["n"], 60)
        self.assertEqual(wo["accuracy"]["n"], 54)
        # plan cz. 4: "zawsze ham" = 40/60 = 66,7% oraz 40/54 = 74,1%; 14 spamów bez kopii
        self.assertEqual(full["always_ham_accuracy"]["k"], 40)
        self.assertAlmostEqual(full["always_ham_accuracy"]["value"], 0.667, places=3)
        self.assertAlmostEqual(wo["always_ham_accuracy"]["value"], 0.741, places=3)
        self.assertEqual(full["recall_spam"]["n"], 20)
        self.assertEqual(wo["recall_spam"]["n"], 14)
        self.assertEqual(full["false_positive_rate"]["n"], 40)
        self.assertEqual(wo["false_positive_rate"]["n"], 40)
        self.assertEqual(set(self.rep["copies_excluded"]), PLAN_VALIDATION_COPIES)
        for r in (full, wo):
            for key in ("recall_spam", "precision_spam", "false_positive_rate", "accuracy"):
                self.assertLessEqual(r[key]["ci_low"], r[key]["value"])
                self.assertGreaterEqual(r[key]["ci_high"], r[key]["value"])

    def test_measures_agree_with_predictions_file(self):
        y = [1 if r["label"] == "spam" else 0 for r in self.rows]
        p = [1 if r["pred"] == "spam" else 0 for r in self.rows]
        keep = [i for i, r in enumerate(self.rows) if r["campaign_copy"] == "0"]
        self.assertEqual(report(y, p)["confusion"], self.rep["full"]["confusion"])
        self.assertEqual(report([y[i] for i in keep], [p[i] for i in keep])["confusion"],
                         self.rep["without_copies"]["confusion"])

    def test_threshold_applied(self):
        thr = self.rep["model"]["threshold"]
        for r in self.rows:
            self.assertEqual(r["pred"], "spam" if float(r["p_spam"]) >= thr else "ham")


if __name__ == "__main__":
    unittest.main()
