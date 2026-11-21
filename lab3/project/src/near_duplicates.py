"""Krok 3 planu: kandydaci na kopie kampanii (train-train i validation-train).

Podobieństwo: Jaccard pięciowyrazowych fragmentów oczyszczonego tekstu (bez prefiksu tematu)
oraz kosinus TF-IDF (wektoryzator dopasowany na train). Dodatkowo wspólne linki.
Wynik trafia do results/near_duplicates.json z polem "decision" do ręcznego zatwierdzenia;
trening używa tylko par z decision == "copy". Już zatwierdzone decyzje są zachowywane.

Użycie: .venv/bin/python -m src.near_duplicates
"""
import json
import re
from itertools import combinations

from sklearn.metrics.pairwise import cosine_similarity

from .clean import clean
from .data import ROOT, load_config, load_split, train_sha256
from .features import make_tfidf

URL = re.compile(r"https?://([^/\s\"'>]+)(/[^\s\"'>]*)?", re.I)
MAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
GENERIC = {"cgi-bin", "images", "index", "remove", "unsubscribe", "optout", "default",
           "public_html", "www.yahoo.com", "rd.yahoo.com"}


def link_keys(text, cfg):
    """Charakterystyczne hosty, segmenty ścieżek i adresy e-mail z oczyszczonego tekstu."""
    s, b = clean(text, cfg["cleaning"]["anon_token"])
    t = s + "\n" + b
    keys = set()
    for host, path in URL.findall(t):
        host = host.lower().split(":")[0]
        if host in GENERIC:
            continue
        keys.add("host:" + host)
        for seg in (path or "").split("?")[0].strip("/").split("/"):
            seg = seg.lower()
            if len(seg) >= 5 and seg not in GENERIC and not re.search(r"\.(html?|asp|cgi|php|jpe?g|gif)$", seg):
                keys.add("seg:" + seg)
    keys |= {"mail:" + m.lower() for m in MAIL.findall(t)}
    return keys


def words(text, cfg):
    s, b = clean(text, cfg["cleaning"]["anon_token"])
    return re.findall(cfg["features"]["token_pattern"], (s + "\n" + b).lower())


def shingles(ws, n):
    return {tuple(ws[i:i + n]) for i in range(max(1, len(ws) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def urls(text):
    return {(h.lower() + (p or "")).rstrip(".,)") for h, p in URL.findall(text)}


def main():
    cfg = load_config()
    nd = cfg["near_duplicates"]
    train = load_split(cfg, "train")
    val = load_split(cfg, "validation")

    tfidf = make_tfidf(cfg).fit([r["text"] for r in train])
    Xt = tfidf.transform([r["text"] for r in train])
    Xv = tfidf.transform([r["text"] for r in val])
    cos_tt = cosine_similarity(Xt)
    cos_vt = cosine_similarity(Xv, Xt)

    sh = {r["message_id"]: shingles(words(r["text"], cfg), nd["shingle_size"]) for r in train + val}
    ul = {r["message_id"]: urls(r["text"]) for r in train + val}
    lk = {r["message_id"]: link_keys(r["text"], cfg) for r in train + val}

    def candidate(a, b, cos, kind):
        j = jaccard(sh[a["message_id"]], sh[b["message_id"]])
        shared_links = sorted(lk[a["message_id"]] & lk[b["message_id"]])
        by_link = bool(shared_links) and a["label"] == b["label"] == "spam"
        if j < nd["review_threshold"] and cos < nd["cosine_review_threshold"] and not by_link:
            return None
        return {
            "kind": kind,
            "a": a["message_id"], "a_label": a["label"],
            "b": b["message_id"], "b_label": b["label"],
            "jaccard_5": round(j, 3), "cosine_tfidf": round(float(cos), 3),
            "shared_urls": sorted(ul[a["message_id"]] & ul[b["message_id"]])[:5],
            "shared_link_keys": shared_links[:6],
            "a_subject": a["text"].split("\n", 1)[0][:100],
            "b_subject": b["text"].split("\n", 1)[0][:100],
            "decision": None,
        }

    pairs = []
    for i, j in combinations(range(len(train)), 2):
        c = candidate(train[i], train[j], cos_tt[i, j], "train-train")
        if c:
            pairs.append(c)
    for i, a in enumerate(val):
        for j, b in enumerate(train):
            c = candidate(a, b, cos_vt[i, j], "validation-train")
            if c:
                pairs.append(c)
    pairs.sort(key=lambda p: (p["kind"], -max(p["jaccard_5"], p["cosine_tfidf"])))

    out = ROOT / cfg["paths"]["near_duplicates"]
    old = {}
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
        old = {(p["a"], p["b"]): p for p in prev["pairs"]}
    for p in pairs:
        if (p["a"], p["b"]) in old:
            p["decision"] = old[(p["a"], p["b"])].get("decision")
            if old[(p["a"], p["b"])].get("note"):
                p["note"] = old[(p["a"], p["b"])]["note"]

    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "train_sha256": train_sha256(cfg),
        "cleaning_version": cfg["cleaning"]["version"],
        "method": f"Jaccard {nd['shingle_size']}-wyrazowych fragmentów >= {nd['review_threshold']} "
                  f"lub kosinus TF-IDF >= {nd['cosine_review_threshold']} "
                  f"lub (para spamów) wspólny charakterystyczny host/ścieżka/adres; decyzje ręczne",
        "decision_values": {"copy": "kopia/wariant tej samej wiadomości lub kampanii",
                            "same_sender": "ten sam nadawca/operacja (szablon, link wypisania), inna oferta",
                            "thread": "ten sam wątek lub źródło, różna treść (ham)",
                            "different": "podobieństwo przypadkowe/szablonowe"},
        "pairs": pairs,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(pairs)} kandydatów -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
