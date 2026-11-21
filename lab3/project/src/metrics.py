"""Miary oceny ze spamem jako klasą pozytywną i przedziałami Wilsona."""
import math
from statistics import NormalDist


def wilson(k, n, level=0.95):
    if n == 0:
        return (float("nan"), float("nan"))
    z = NormalDist().inv_cdf(0.5 + level / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def confusion(y_true, y_pred):
    """y: 1 = spam, 0 = ham."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def report(y_true, y_pred, level=0.95):
    c = confusion(y_true, y_pred)
    n = len(y_true)

    def prop(k, m):
        lo, hi = wilson(k, m, level)
        return {"k": k, "n": m, "value": k / m if m else float("nan"), "ci_low": lo, "ci_high": hi}

    n_ham = c["fp"] + c["tn"]
    return {
        "confusion": c,
        "recall_spam": prop(c["tp"], c["tp"] + c["fn"]),
        "precision_spam": prop(c["tp"], c["tp"] + c["fp"]),
        "false_positive_rate": prop(c["fp"], n_ham),
        "accuracy": prop(c["tp"] + c["tn"], n),
        "always_ham_accuracy": prop(n_ham, n),
    }
