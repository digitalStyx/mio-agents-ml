"""Stała, nieuczona reguła czyszczenia tekstu (krok 2 planu).

Usuwa ślady drogi, którą wiadomość trafiła do korpusu: stopki list dyskusyjnych,
znaczniki list w temacie i zanonimizowane adresy. Wzorce ułożono na train.
Przy każdej zmianie podbić [cleaning].version w config.toml.
"""
import re

# Stopki doklejane przez listy, zakotwiczone na końcu tekstu. Stosowane wielokrotnie,
# bo stopka sponsora SourceForge stoi nad stopką Mailmana.
FOOTERS = [
    # ILUG / ILUG-Social (Mailman, linux.ie)
    re.compile(r"\n-- ?\n\s*Irish Linux Users' Group[^\n]*\n[^\n]*listinfo[^\n]*\n"
               r"[^\n]*listmaster@linux\.ie[^\n]*\s*\Z"),
    # Ogólna stopka Mailmana: linia podkreśleń, do 4 linii, link listinfo/mailman
    re.compile(r"\n_{20,}\n(?:[^\n]*\n){0,4}?[^\n]*(?:listinfo|mailman)[^\n]*\s*\Z"),
    # Reklama sponsora SourceForge
    re.compile(r"\n-{20,}\n\s*This sf\.net email is sponsored by[\s\S]{0,500}?\Z"),
    # Yahoo! Groups (z reklamą sponsora lub bez)
    re.compile(r"\n-{5,} ?Yahoo! Groups Sponsor[\s\S]{0,1000}?"
               r"Your use of Yahoo! Groups is subject to[^\n]*\s*\Z"),
    re.compile(r"\nTo unsubscribe from this group, send an email to:[\s\S]{0,400}?"
               r"Your use of Yahoo! Groups is subject to[^\n]*\s*\Z"),
]

# Pojedyncze linie stopek, także zacytowane ("> ...") w odpowiedziach lub bez linii podkreśleń
FOOTER_LINES = re.compile(
    r"^[ \t>]*(?:\S*https?://\S*listinfo/\S*.*(?:\n[ \t>]*information\.)?"
    r"|Irish Linux Users' Group[^\n]*"
    r"|List maintainer: listmaster@\S*"
    r"|[\w.-]+ mailing list\b.*"
    r"|In remembrance\s*\n[ \t>]*www\.osdn\.com/911/\S*.*)[ \t]*$\n?", re.M | re.I)

# Blok sponsora SourceForge w środku tekstu (np. w cytacie): od linii myślników do linii podkreśleń
SF_SPONSOR_BLOCK = re.compile(
    r"^[ \t>]*-{20,}[ \t]*\n[ \t>]*This sf\.net email is sponsored by(?:.*\n){0,6}?[ \t>]*_{20,}[ \t]*$",
    re.M | re.I)

# Adresy list ILUG jako odbiorca (Cc, link wypisania) - traktowane jak zanonimizowany adres
LIST_ADDR = re.compile(r"\b(?:ilug|social)(?:-admin|-request)?@linux\.ie\b", re.I)

# Znaczniki list na początku tematu, np. "[ILUG]", "Re: [zzzzteana]". Tag musi zaczynać się literą,
# więc "[2]" zostaje. "Re:" / "Fw:" zostają (plan: niosą częściowo sygnał rozmowy).
SUBJECT_TAG = re.compile(r"^((?:\s*(?:re|fw|fwd)\s*:)*)\s*\[[A-Za-z][^\]\n]{0,30}\]\s*", re.I)

# Linia z linkiem kanału newsisfree (nagłówek wiadomości z czytnika RSS, nie treść)
NEWSISFREE = re.compile(r"^URL: https?://www\.newsisfree\.com/\S*[ \t]*\n?", re.M)

# v2: cały nagłówek wpisu z czytnika RSS na początku treści ("URL: ..." / "Date: ...").
# 22 ze 120 hamów train; "supplied" (z "Date: Not supplied") i "url" były w czołówce wag hamu.
FEED_HEADER = re.compile(
    r"\A\s*(?:URL: \S+[ \t]*\n)?Date: [^\n]*\n(?=[ \t]*\n)"   # linia Date: zakończona pustym wierszem
    r"|\A\s*URL: \S+[ \t]*\n")

# Zanonimizowane adresy: wszystko w domenie spamassassin.taint.org (także w linkach)
# oraz samodzielne "zzzz" / "zzzz@domena" (spam wstawiał adres odbiorcy).
ANON_DOMAIN = re.compile(r"[\w.+%-]*@?[\w.-]*spamassassin\.taint\.org\b", re.I)
ANON_ZZZZ = re.compile(r"\bzzzz[\w.@-]*", re.I)   # zzzz, zzzz@x.com, zzzzason.org, zzzzcards


def split_subject(text):
    """Zwraca (temat, treść). Każdy tekst zaczyna się od 'Subject:'."""
    first, _, body = text.partition("\n")
    subject = first[len("Subject:"):] if first.startswith("Subject:") else ""
    if not first.startswith("Subject:"):
        body = text
    return subject.strip(), body


def clean(text, anon_token="anonaddr"):
    subject, body = split_subject(text)
    for _ in range(4):
        before = body
        for pat in FOOTERS:
            body = pat.sub("\n", body)
        if body == before:
            break
    subject = SUBJECT_TAG.sub(lambda m: (m.group(1).strip() + " ") if m.group(1) else "", subject).strip()
    body = NEWSISFREE.sub("", body)
    body = FEED_HEADER.sub("", body)
    body = SF_SPONSOR_BLOCK.sub("", body)
    body = FOOTER_LINES.sub("", body)
    out = []
    for part in (subject, body):
        part = LIST_ADDR.sub(f" {anon_token} ", part)
        part = ANON_DOMAIN.sub(f" {anon_token} ", part)
        part = ANON_ZZZZ.sub(f" {anon_token} ", part)
        out.append(part)
    return out[0], out[1]
