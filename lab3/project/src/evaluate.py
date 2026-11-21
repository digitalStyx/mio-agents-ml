"""Krok 9 planu: jednorazowa ocena zamrożonego modelu na validation.csv.

Odmawia działania, jeśli brak modelu, model uczono na innym train.csv lub ocena już się odbyła
(results/validation_done.lock). Użycie: .venv/bin/python -m src.evaluate
"""
import csv
import json
import sys
from datetime import datetime, timezone

import joblib
import sklearn

from .data import ROOT, load_config, load_split, train_sha256
from .groups import _load_pairs, validation_copies
from .metrics import report


def main():
    cfg = load_config()
    models, results = ROOT / cfg["paths"]["models"], ROOT / cfg["paths"]["results"]
    lock = results / "validation_done.lock"
    if lock.exists():
        sys.exit(f"Ocena na validation już się odbyła ({lock.read_text().strip()}). Plan: tylko raz.")
    if not (models / "spam_model.joblib").exists():
        sys.exit("Brak modelu - najpierw .venv/bin/python -m src.train")
    meta = json.loads((models / "spam_model.json").read_text(encoding="utf-8"))["meta"]
    if meta["train_sha256"] != train_sha256(cfg):
        sys.exit("Model uczono na innym train.csv")
    if meta["versions"]["scikit-learn"] != sklearn.__version__:
        sys.exit("Inna wersja scikit-learn niż przy zapisie modelu")

    pipe = joblib.load(models / "spam_model.joblib")
    val = load_split(cfg, "validation")
    X = [r["text"] for r in val]
    y = [1 if r["label"] == "spam" else 0 for r in val]
    proba = pipe.predict_proba(X)[:, 1]
    pred = [int(p >= meta["threshold"]) for p in proba]
    level = cfg["report"]["ci_level"]

    copies = set(validation_copies(cfg))
    pairs = _load_pairs(cfg)
    thread_hams = sorted({p["a"] for p in pairs
                          if p["kind"] == "validation-train" and p["decision"] == "thread"})
    same_sender = sorted({p["a"] for p in pairs if p["kind"] == "validation-train"
                          and p["decision"] == "same_sender" and p["a"] not in copies})
    keep = [i for i, r in enumerate(val) if r["message_id"] not in copies]
    th = [i for i, r in enumerate(val) if r["message_id"] in thread_hams]

    out = {
        "evaluated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": meta, "copies_excluded": sorted(copies), "thread_hams": thread_hams,
        "full": report(y, pred, level),
        "without_copies": report([y[i] for i in keep], [pred[i] for i in keep], level),
        "thread_hams_fp": sum(pred[i] for i in th),
        "same_sender_spams": same_sender,
        "same_sender_detected": sum(pred[i] for i, r in enumerate(val) if r["message_id"] in same_sender),
        "predictions": [{"message_id": r["message_id"], "label": r["label"], "p_spam": float(p),
                         "pred": "spam" if q else "ham"} for r, p, q in zip(val, proba, pred)],
    }
    results.mkdir(exist_ok=True)
    with open(results / "validation_predictions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["message_id", "label", "p_spam", "pred", "campaign_copy"])
        for p in out["predictions"]:
            w.writerow([p["message_id"], p["label"], f"{p['p_spam']:.6f}", p["pred"],
                        int(p["message_id"] in copies)])
    (results / "validation_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    (results / "validation_report.md").write_text(markdown(out), encoding="utf-8")
    lock.write_text(out["evaluated"] + "\n")
    print(markdown(out))


def fmt(p):
    return f"{p['k']}/{p['n']} = {100 * p['value']:.1f}% (95% CI {100 * p['ci_low']:.1f}-{100 * p['ci_high']:.1f}%)"


def section(title, r):
    c = r["confusion"]
    return [f"## {title}", "",
            "| | przewidziany spam | przewidziany ham |", "|---|---|---|",
            f"| spam | {c['tp']} | {c['fn']} |", f"| ham | {c['fp']} | {c['tn']} |", "",
            f"- czułość (spam): {fmt(r['recall_spam'])}",
            f"- precyzja (spam): {fmt(r['precision_spam'])}",
            f"- odsetek fałszywych alarmów: {fmt(r['false_positive_rate'])}",
            f"- trafność: {fmt(r['accuracy'])}; zawsze ham: {fmt(r['always_ham_accuracy'])}", ""]


def markdown(out):
    m = out["model"]
    L = ["# Ocena na validation (jednorazowa)", "",
         f"Model {m['model']} (parametr {m['param']}), próg {m['threshold']:.4f}, "
         f"reguły czyszczenia v{m['cleaning_version']}. Ocena: {out['evaluated']}.", ""]
    L += section(f"Pełna walidacja ({out['full']['accuracy']['n']} wiadomości)", out["full"])
    L += section(f"Bez kopii kampanii ({out['without_copies']['accuracy']['n']} wiadomości)",
                 out["without_copies"])
    L += [f"Hamy ze wspólnym wątkiem w train ({len(out['thread_hams'])}, informacyjnie): "
          f"fałszywe alarmy {out['thread_hams_fp']}.", "",
          f"Spamy od nadawcy znanego z train, inna oferta ({len(out['same_sender_spams'])}, "
          f"w obu wynikach, informacyjnie): wykryte {out['same_sender_detected']}.", ""]
    errs = [p for p in out["predictions"] if p["label"] != p["pred"]]
    L += ["## Pomyłki", ""] + [f"- `{p['message_id']}` {p['label']} -> {p['pred']}, p={p['p_spam']:.3f}"
                               for p in errs] + [""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
