"""Plan cz. 4: przedziały Wilsona - wartości podane w planie."""
import unittest

from src.metrics import wilson


class TestWilson(unittest.TestCase):
    def test_plan_examples(self):
        lo, hi = wilson(18, 20)             # "czułość 18/20 ma przedział ok. 70-97%"
        self.assertAlmostEqual(lo, 0.70, delta=0.01)
        self.assertAlmostEqual(hi, 0.97, delta=0.01)
        lo, hi = wilson(0, 40)              # "brak FP na 40 hamach: do ok. 9%"
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.09, delta=0.005)


if __name__ == "__main__":
    unittest.main()
