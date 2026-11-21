"""B: zapisany model -> validation; porownanie z results/validation_predictions.csv po message_id."""
import csv, json, sys
from pathlib import Path
K = Path(sys.argv[1]); sys.path.insert(0, str(K))
import joblib
from src.data import load_config, load_split
cfg = load_config()
pipe = joblib.load(K / "models/spam_model.joblib")
meta = json.loads((K / "models/spam_model.json").read_text())["meta"]
val = load_split(cfg, "validation")
p = pipe.predict_proba([r["text"] for r in val])[:, 1]
mine = {r["message_id"]: (r["label"], float(q)) for r, q in zip(val, p)}
rows = list(csv.DictReader(open(K / "results/validation_predictions.csv")))
maxdiff = max(abs(float(r["p_spam"]) - mine[r["message_id"]][1]) for r in rows)
lab_bad = [r["message_id"] for r in rows if r["label"] != mine[r["message_id"]][0]]
pred_bad = [r["message_id"] for r in rows if r["pred"] != ("spam" if mine[r["message_id"]][1] >= meta["threshold"] else "ham")]
ham = max(q for l, q in mine.values() if l == "ham"); spam_min = min(q for l, q in mine.values() if l == "spam")
print(f"wierszy {len(rows)}, max |p_csv - p_model| = {maxdiff:.2e}, zle etykiety {lab_bad}, zle pred {pred_bad}")
print(f"prog {meta['threshold']:.4f}, najwyzszy ham p={ham:.4f}, najnizszy spam p={spam_min:.4f}")
