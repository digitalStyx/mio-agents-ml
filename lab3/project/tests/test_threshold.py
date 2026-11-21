"""Krok 6 planu: najniższy próg z co najwyżej 1 fałszywym alarmem, na zmyślonych wynikach."""
import unittest

from src.models import at_threshold, choose_threshold


def counts(ham, spam, thr):
    return sum(s >= thr for s in spam), sum(h >= thr for h in ham)


class TestThreshold(unittest.TestCase):
    def check(self, ham, spam, lo, hi, tp, fp, max_fp=1):
        y = [0] * len(ham) + [1] * len(spam)
        thr = choose_threshold(ham + spam, y, max_fp)
        self.assertGreater(thr, lo)
        self.assertLessEqual(thr, hi)
        self.assertEqual(counts(ham, spam, thr), (tp, fp))
        r = at_threshold(ham + spam, y, thr)
        self.assertEqual((r["tp"], r["fp"]), (tp, fp))
        return thr

    def test_one_false_alarm_allowed(self):
        # hamy 0.9 i 0.6: 0.9 może być fałszywym alarmem, 0.6 już nie -> próg w (0.6, 0.7]
        self.check(ham=[0.9, 0.6, 0.3, 0.2], spam=[0.95, 0.8, 0.7, 0.5], lo=0.6, hi=0.7, tp=3, fp=1)

    def test_lowest_threshold_uses_the_tolerance(self):
        # spamy wyżej niż hamy: najniższy próg dopuszcza 1 FP (ham 0.2) -> próg w (0.1, 0.2]
        self.check(ham=[0.1, 0.2], spam=[0.8, 0.9], lo=0.1, hi=0.2, tp=2, fp=1)

    def test_tied_hams_cannot_both_pass(self):
        # dwa hamy 0.7 dałyby 2 FP, więc próg musi być powyżej 0.7
        self.check(ham=[0.7, 0.7, 0.1], spam=[0.7, 0.9], lo=0.7, hi=0.9, tp=1, fp=0)

    def test_no_lower_threshold_satisfies_rule(self):
        ham, spam = [0.9, 0.6, 0.3, 0.2], [0.95, 0.8, 0.7, 0.5]
        thr = self.check(ham, spam, lo=0.6, hi=0.7, tp=3, fp=1)
        for s in sorted(set(ham + spam)):
            if s < thr:   # każdy niższy zaobserwowany próg łamie regułę
                self.assertGreater(counts(ham, spam, s)[1], 1)

    def test_zero_tolerance(self):
        self.check(ham=[0.9, 0.6, 0.3], spam=[0.95, 0.8, 0.7], lo=0.9, hi=0.95, tp=1, fp=0, max_fp=0)


if __name__ == "__main__":
    unittest.main()
