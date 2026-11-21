"""Kroki 4-8 planu: walidacja krzyżowa, wybór modelu i progu, kontrola wag, model końcowy.

Nie dotyka validation.csv. Użycie: .venv/bin/python -m src.train
"""
import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from .data import ROOT, load_config, load_split, train_sha256
from .groups import train_groups
from .models import (AlwaysHam, KeywordRule, at_threshold, choose_threshold, logreg_pipeline,
                     nb_pipeline)


def splitter(cfg, seed):
    return StratifiedGroupKFold(n_splits=cfg["cv"]["n_splits"], shuffle=cfg["cv"]["shuffle"],
                                random_state=seed)


def oof_scores(est, X, y, groups, cv):
    if hasattr(est, "predict_proba") and not isinstance(est, (KeywordRule, AlwaysHam)):
        return cross_val_predict(est, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return cross_val_predict(est, X, y, groups=groups, cv=cv, method="decision_function")


def evaluate_grid(name, make, grid, tie_simpler, cfg, X, y, groups, cv):
    """Dla każdego parametru: wyniki OOF, próg wg reguły, TP/FP. Zwraca (najlepszy, wszystkie)."""
    rows = []
    for param in grid:
        s = oof_scores(make(param), X, y, groups, cv)
        thr = choose_threshold(s, y, cfg["threshold"]["max_false_positives"])
        rows.append({"model": name, "param": param, **at_threshold(s, y, thr), "_scores": s})
    # remis: więcej TP, potem mniej FP, potem prostszy model
    best = min(rows, key=lambda r: (-r["tp"], r["fp"], tie_simpler(r["param"])))
    return best, rows


def select_model(cfg, X, y, groups, cv, drop_tokens=()):
    lr_best, lr_rows = evaluate_grid("logreg", lambda C: logreg_pipeline(cfg, C, drop_tokens),
                                     cfg["logreg"]["C_grid"], lambda C: C, cfg, X, y, groups, cv)
    nb_best, nb_rows = evaluate_grid("naive_bayes", lambda a: nb_pipeline(cfg, a, drop_tokens),
                                     cfg["naive_bayes"]["alpha_grid"], lambda a: -a, cfg, X, y, groups, cv)
    chosen = nb_best if nb_best["tp"] >= lr_best["tp"] + cfg["selection"]["min_gain"] else lr_best
    return chosen, lr_rows + nb_rows


def make_chosen(cfg, chosen, drop_tokens=()):
    if chosen["model"] == "logreg":
        return logreg_pipeline(cfg, chosen["param"], drop_tokens)
    return nb_pipeline(cfg, chosen["param"], drop_tokens)


def feature_weights(pipe):
    vocab = pipe.named_steps["vec"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        w = clf.coef_[0]
    else:  # NB: log-iloraz prawdopodobieństw słowa w klasach
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    return vocab, np.asarray(w)


def top_weights(pipe, n):
    vocab, w = feature_weights(pipe)
    order = np.argsort(w)
    return ([(vocab[i], float(w[i])) for i in order[::-1][:n]],
            [(vocab[i], float(w[i])) for i in order[:n]])


def strip(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def main():
    cfg = load_config()
    train = load_split(cfg, "train")
    X = [r["text"] for r in train]
    y = np.array([1 if r["label"] == "spam" else 0 for r in train])
    groups = train_groups(cfg, train)
    cv = splitter(cfg, cfg["seed"])
    max_fp = cfg["threshold"]["max_false_positives"]
    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "train_sha256": train_sha256(cfg), "n_groups": len(set(groups))}

    # Krok 4: punkty odniesienia
    kb = cfg["keyword_baseline"]
    kw_best, kw_rows = evaluate_grid(
        "keywords", lambda n: KeywordRule(cfg, n, kb["min_spam_docs"]), kb["list_sizes"],
        lambda n: n, cfg, X, y, groups, cv)
    ham_s = oof_scores(AlwaysHam(), X, y, groups, cv)
    report["baselines"] = {"always_ham": at_threshold(ham_s, y, 0.5), "keywords": strip(kw_rows),
                           "keywords_best": strip([kw_best])[0]}

    # Krok 5-6: regresja vs Bayes, próg
    chosen, grid_rows = select_model(cfg, X, y, groups, cv)
    report["grid"] = strip(grid_rows)

    # Krok 7: kontrola wag (na całym train). Najpierw model ze wszystkimi słowami:
    # czy słowa formy rozmowy są w czołówce; ślady korpusu sprawdza się w modelu końcowym niżej.
    tc = cfg["trace_check"]
    top_spam, top_ham = top_weights(make_chosen(cfg, chosen).fit(X, y), tc["top_n"])
    top_tokens = [t for t, _ in top_spam + top_ham]
    report["weights_with_conversation"] = {"top_spam": top_spam, "top_ham": top_ham}

    conv = tc["conversation_words"]
    chosen_wo, grid_wo = select_model(cfg, X, y, groups, cv, drop_tokens=conv)
    use_without = chosen_wo["tp"] >= chosen["tp"] - tc["max_loss_without"]
    report["conversation_check"] = {
        "words": conv, "in_top": sorted(set(conv) & set(top_tokens)),
        "with": strip([chosen])[0], "without": strip([chosen_wo])[0], "grid_without": strip(grid_wo),
        "decision": "bez słów rozmowy" if use_without else "ze słowami rozmowy (do ograniczeń)",
    }
    drop = conv if use_without else []
    final = chosen_wo if use_without else chosen
    threshold = final["threshold"]
    report["chosen"] = {**strip([final])[0], "drop_tokens": drop}

    # Ślady korpusu w wagach modelu końcowego
    pipe = make_chosen(cfg, final, drop).fit(X, y)
    f_spam, f_ham = top_weights(pipe, tc["top_n"])
    traces = sorted({t for t, _ in f_spam + f_ham for tr in tc["corpus_traces"] if tr in t})
    report["trace_check"] = {"top_spam": f_spam, "top_ham": f_ham, "corpus_traces_in_top": traces}
    if traces:
        report["stopped"] = "ślady korpusu w najsilniejszych wagach - popraw src/clean.py i podbij wersję"
        write_reports(cfg, report)
        sys.exit(f"STOP: ślady korpusu w wagach: {traces}. Zob. results/cv_report.md")
    # Stabilność progu: ta sama reguła przy innych ziarnach podziału
    stab, seed_scores = [], []
    for seed in cfg["cv"]["stability_seeds"]:
        s = oof_scores(make_chosen(cfg, final, drop), X, y, groups, splitter(cfg, seed))
        seed_scores.append(s)
        stab.append({"seed": seed, **at_threshold(s, y, choose_threshold(s, y, max_fp))})
    # ODSTĘPSTWO od planu (decyzja użytkownika): przy rozrzucie progu między ziarnami
    # zamraża się ostrożniejszy próg = najwyższy z progów wyznaczonych dla wszystkich ziaren.
    main_threshold = threshold
    if cfg["threshold"]["aggregate"] == "max_over_seeds":
        threshold = max(r["threshold"] for r in stab)
    for r, s in zip(stab, seed_scores):
        r["at_frozen_threshold"] = at_threshold(s, y, threshold)
    folds = []
    for k, (_, te) in enumerate(cv.split(X, y, groups)):
        folds.append({"fold": k, **at_threshold(final["_scores"][te], y[te], threshold)})
    report["stability"] = {"seeds": stab, "folds_at_threshold": folds,
                           "main_seed_threshold": main_threshold,
                           "aggregate": cfg["threshold"]["aggregate"], "frozen_threshold": threshold}
    final = {**final, **at_threshold(final["_scores"], y, threshold)}   # TP/FP przy zamrożonym progu
    report["chosen"] = {**strip([final])[0], "drop_tokens": drop}
    report["keywords_vs_model_gain"] = final["tp"] - kw_best["tp"]
    report["model_clearly_better_than_keywords"] = (final["tp"] - kw_best["tp"]) >= cfg["selection"]["min_gain"]

    # Krok 8: model końcowy na całym train (pipe dopasowany wyżej)
    save_model(cfg, pipe, final, threshold, drop, report)
    write_reports(cfg, report)
    print(f"Model: {final['model']} param={final['param']} próg={threshold:.4f} "
          f"CV: TP {final['tp']}/{final['n_spam']}, FP {final['fp']}/{final['n_ham']}")


def save_model(cfg, pipe, final, threshold, drop, report):
    out = ROOT / cfg["paths"]["models"]
    out.mkdir(exist_ok=True)
    joblib.dump(pipe, out / "spam_model.joblib")
    vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]
    vocab, w = feature_weights(pipe)
    params = {"vocabulary": {t: int(i) for t, i in vec.vocabulary_.items()}}
    if final["model"] == "logreg":
        params |= {"idf": vec.idf_.tolist(), "coef": clf.coef_[0].tolist(),
                   "intercept": float(clf.intercept_[0])}
    else:
        params |= {"feature_log_prob": clf.feature_log_prob_.tolist(),
                   "class_log_prior": clf.class_log_prior_.tolist()}
    meta = {
        "model": final["model"], "param": final["param"], "threshold": threshold,
        "positive_class": "spam", "drop_tokens": drop,
        "train_sha256": report["train_sha256"], "cleaning_version": cfg["cleaning"]["version"],
        "features": cfg["features"], "created": report["created"],
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__},
        "cv": {"tp": final["tp"], "fp": final["fp"], "n_spam": final["n_spam"], "n_ham": final["n_ham"]},
    }
    (out / "spam_model.json").write_text(json.dumps({"meta": meta, "params": params}, ensure_ascii=False),
                                         encoding="utf-8")


def write_reports(cfg, report):
    out = ROOT / cfg["paths"]["results"]
    out.mkdir(exist_ok=True)
    (out / "cv_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# Walidacja krzyżowa na train", "",
         f"Utworzono: {report['created']}. Grup w CV: {report['n_groups']}. Reguła progu: "
         f"<= {cfg['threshold']['max_false_positives']} FP w predykcjach out-of-fold.", ""]
    if "baselines" in report:
        b = report["baselines"]
        L += ["## Punkty odniesienia", "",
              f"- zawsze ham: TP 0/{b['always_ham']['n_spam']}",
              *[f"- słowa kluczowe, lista {r['param']}: TP {r['tp']}, FP {r['fp']}, k >= {r['threshold']:.1f}"
                for r in b["keywords"]], ""]
    if "grid" in report:
        L += ["## Siatka parametrów", "", "| model | parametr | próg | TP | FP |", "|---|---|---|---|---|",
              *[f"| {r['model']} | {r['param']} | {r['threshold']:.4f} | {r['tp']}/{r['n_spam']} | "
                f"{r['fp']}/{r['n_ham']} |" for r in report["grid"]], ""]
    if "trace_check" in report:
        t = report["trace_check"]
        L += ["## Najsilniejsze wagi modelu końcowego", "",
              "Spam: " + ", ".join(f"`{w}` {v:+.3g}" for w, v in t["top_spam"]), "",
              "Ham: " + ", ".join(f"`{w}` {v:+.3g}" for w, v in t["top_ham"]), "",
              f"Ślady korpusu w czołówce: {t['corpus_traces_in_top'] or 'brak'}", ""]
    if "weights_with_conversation" in report:
        t = report["weights_with_conversation"]
        L += ["Model przed usunięciem słów rozmowy, ham: "
              + ", ".join(f"`{w}` {v:+.3g}" for w, v in t["top_ham"][:15]), ""]
    if "stopped" in report:
        L += [f"**Zatrzymano:** {report['stopped']}", ""]
    if "conversation_check" in report:
        c = report["conversation_check"]
        L += ["## Słowa formy rozmowy", "",
              f"W czołówce wag: {c['in_top'] or 'brak'}. Z nimi: TP {c['with']['tp']}, FP {c['with']['fp']} "
              f"({c['with']['model']} {c['with']['param']}). Bez nich: TP {c['without']['tp']}, "
              f"FP {c['without']['fp']} ({c['without']['model']} {c['without']['param']}). "
              f"Decyzja: {c['decision']}.", ""]
    if "chosen" in report:
        ch = report["chosen"]
        L += ["## Wybrany model", "",
              f"{ch['model']}, parametr {ch['param']}, próg {ch['threshold']:.4f}, "
              f"TP {ch['tp']}/{ch['n_spam']}, FP {ch['fp']}/{ch['n_ham']}. "
              f"Przewaga nad regułą słów: {report['keywords_vs_model_gain']} spamów "
              f"({'wyraźna' if report['model_clearly_better_than_keywords'] else 'NIEWYRAŹNA'}).", "",
              "## Stabilność progu", "",
              f"Próg z ziarna głównego: {report['stability']['main_seed_threshold']:.4f}. Zamrożony próg "
              f"({report['stability']['aggregate']}, odstępstwo od planu): "
              f"{report['stability']['frozen_threshold']:.4f}.", "",
              "| ziarno | próg | TP | FP | TP/FP przy zamrożonym progu |",
              "|---|---|---|---|---|",
              *[f"| {s['seed']} | {s['threshold']:.4f} | {s['tp']} | {s['fp']} | "
                f"{s['at_frozen_threshold']['tp']}/{s['at_frozen_threshold']['fp']} |"
                for s in report["stability"]["seeds"]], "",
              "Części (ziarno główne), przy zamrożonym progu: " + "; ".join(
                  f"#{f['fold']}: TP {f['tp']}/{f['n_spam']}, FP {f['fp']}/{f['n_ham']}"
                  for f in report["stability"]["folds_at_threshold"]), ""]
    (out / "cv_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
