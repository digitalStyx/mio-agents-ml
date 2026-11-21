"""Analizator tekstu: czyszczenie + tokenizacja z osobnym oznaczeniem słów tematu.

Klasa (a nie lambda), żeby pipeline dało się zapisać przez joblib.
Wektoryzatory dopasowuje się wyłącznie na train (w CV: na części uczącej).
"""
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .clean import clean


class Analyzer:
    def __init__(self, token_pattern, subject_prefix="subj_", lowercase=True,
                 anon_token="anonaddr", drop_tokens=()):
        self.token_pattern = token_pattern
        self.subject_prefix = subject_prefix
        self.lowercase = lowercase
        self.anon_token = anon_token
        self.drop_tokens = tuple(drop_tokens)

    def _tokens(self, s):
        if self.lowercase:
            s = s.lower()
        return re.findall(self.token_pattern, s)

    def __call__(self, text):
        subject, body = clean(text, self.anon_token)
        toks = [self.subject_prefix + t for t in self._tokens(subject)] + self._tokens(body)
        if self.drop_tokens:
            drop = set(self.drop_tokens)
            toks = [t for t in toks if t not in drop]
        return toks


def make_analyzer(cfg, drop_tokens=()):
    f = cfg["features"]
    return Analyzer(f["token_pattern"], f["subject_prefix"], f["lowercase"],
                    cfg["cleaning"]["anon_token"], drop_tokens)


def make_tfidf(cfg, drop_tokens=()):
    f = cfg["features"]
    return TfidfVectorizer(analyzer=make_analyzer(cfg, drop_tokens), min_df=f["min_df"],
                           sublinear_tf=f["sublinear_tf"], norm=f["norm"])


def make_counts(cfg, drop_tokens=()):
    return CountVectorizer(analyzer=make_analyzer(cfg, drop_tokens), min_df=cfg["features"]["min_df"])
