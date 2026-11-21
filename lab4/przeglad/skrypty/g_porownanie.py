"""G: porownanie przebiegu od zera (swieze srodowisko) z lab3. Uzycie: python g_porownanie.py <nowy> <lab3>"""
import csv, json, sys
from pathlib import Path
N, O = Path(sys.argv[1]), Path(sys.argv[2])
def js(p, d): return json.loads((d / p).read_text(encoding="utf-8"))
mn, mo = js("models/spam_model.json", N), js("models/spam_model.json", O)
print("## model (models/spam_model.json)")
for k in ("model", "param", "threshold", "drop_tokens", "cleaning_version", "train_sha256", "versions", "cv"):
    print(f"{k}: {'TAK' if mn['meta'][k] == mo['meta'][k] else 'NIE'}  {mn['meta'][k]}")
for k in ("vocabulary", "idf", "coef", "intercept"):
    print(f"{k} bit w bit identyczne: {'TAK' if mn['params'][k] == mo['params'][k] else 'NIE'}")
print(f"joblib bajtowo identyczny: {'TAK' if (N/'models/spam_model.joblib').read_bytes() == (O/'models/spam_model.joblib').read_bytes() else 'NIE (pickle moze roznic sie np. adresami; liczy sie zgodnosc parametrow i przewidywan)'}")
cn, co = js("results/cv_report.json", N), js("results/cv_report.json", O)
print("\n## walidacja krzyzowa (results/cv_report.json, bez daty utworzenia)")
for k in ("baselines", "grid", "conversation_check", "chosen", "trace_check", "stability"):
    print(f"{k}: {'TAK' if cn[k] == co[k] else 'NIE'}")
print("\n## przewidywania (results/validation_predictions.csv)")
rn = {r["message_id"]: r for r in csv.DictReader(open(N / "results/validation_predictions.csv"))}
ro = {r["message_id"]: r for r in csv.DictReader(open(O / "results/validation_predictions.csv"))}
print(f"te same id: {'TAK' if rn.keys() == ro.keys() else 'NIE'} ({len(rn)})")
for c in ("label", "p_spam", "pred", "campaign_copy"):
    diff = [i for i in rn if rn[i][c] != ro[i][c]]
    print(f"{c}: rozne w {len(diff)} wierszach")
vn, vo = js("results/validation_report.json", N), js("results/validation_report.json", O)
print("\n## miary (results/validation_report.json)")
for k in ("full", "without_copies", "copies_excluded", "thread_hams", "thread_hams_fp", "same_sender_spams", "same_sender_detected"):
    print(f"{k}: {'TAK' if vn[k] == vo[k] else 'NIE'}")
