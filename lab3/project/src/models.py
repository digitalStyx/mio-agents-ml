"""Modele, punkty odniesienia i reguła progu."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .features import make_analyzer, make_counts, make_tfidf


def logreg_pipeline(cfg, C, drop_tokens=()):
    lr = cfg["logreg"]
    return Pipeline([
        ("vec", make_tfidf(cfg, drop_tokens)),
        ("clf", LogisticRegression(C=C, class_weight=lr["class_weight"], solver=lr["solver"],
                                   max_iter=lr["max_iter"], random_state=cfg["seed"])),
    ])


def nb_pipeline(cfg, alpha, drop_tokens=()):
    return Pipeline([("vec", make_counts(cfg, drop_tokens)), ("clf", MultinomialNB(alpha=alpha))])


class KeywordRule(ClassifierMixin, BaseEstimator):
    """Punkt odniesienia: wynik = liczba różnych słów z listy obecnych w wiadomości.

    Listę buduje się w fit() tylko z danych uczących: słowa występujące w >= min_spam_docs spamach,
    posortowane wg różnicy wygładzonych częstości dokumentowych spam - ham.
    """

    def __init__(self, cfg=None, list_size=20, min_spam_docs=3):
        self.cfg = cfg
        self.list_size = list_size
        self.min_spam_docs = min_spam_docs

    def fit(self, X, y):
        y = np.asarray(y)
        an = make_analyzer(self.cfg)
        docs = [set(an(t)) for t in X]
        n_spam, n_ham = int((y == 1).sum()), int((y == 0).sum())
        df_s, df_h = {}, {}
        for d, lab in zip(docs, y):
            target = df_s if lab == 1 else df_h
            for w in d:
                target[w] = target.get(w, 0) + 1
        scored = [((df_s[w] + 1) / (n_spam + 2) - (df_h.get(w, 0) + 1) / (n_ham + 2), w)
                  for w in df_s if df_s[w] >= self.min_spam_docs]
        scored.sort(key=lambda t: (-t[0], t[1]))
        self.keywords_ = [w for _, w in scored[: self.list_size]]
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        an = make_analyzer(self.cfg)
        kw = set(self.keywords_)
        return np.array([len(kw & set(an(t))) for t in X], dtype=float)

    def predict(self, X):
        return (self.decision_function(X) >= 1).astype(int)


class AlwaysHam(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return np.zeros(len(X))

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def choose_threshold(scores, y, max_fp):
    """Reguła progu z planu (krok 6).

    Najniższy próg, przy którym co najwyżej max_fp hamów ma wynik >= progu (najniższy taki próg
    daje zarazem najwyższą czułość). Próg ustawiany w połowie między (max_fp+1)-szym najwyższym
    wynikiem hamu a najbliższym wyższym zaobserwowanym wynikiem: te same pomyłki co "tuż nad",
    ale z zapasem na nowe dane.
    """
    scores, y = np.asarray(scores, float), np.asarray(y)
    ham = np.sort(scores[y == 0])[::-1]
    if len(ham) <= max_fp:
        return float(scores.min())
    bound = ham[max_fp]                     # ten ham i niższe muszą być poniżej progu
    above = np.unique(scores[scores > bound])
    thr = float((bound + above[0]) / 2) if len(above) else float(np.nextafter(bound, np.inf))
    return thr


def at_threshold(scores, y, thr):
    scores, y = np.asarray(scores, float), np.asarray(y)
    pred = scores >= thr
    return {"threshold": thr, "tp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum()),
            "n_spam": int((y == 1).sum()), "n_ham": int((y == 0).sum())}
