# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 11c: the librarian — BM25 retrieval over out/fts.db with a
definition-first rerank. Used by the SFT probe and by the demo server.
retrieve(db, question, k) -> list of {pmid, title, ctx} trimmed for a
512-token prompt budget."""

import re
import sqlite3

STOP = set("what is are the a an of in on for to and or does do how why "
           "which that this it its with by as at from be been was were "
           "define describe explain tell me about".split())
SYN = {"voltage": ["potential"], "potential": ["voltage"],
       "vmem": ["potential"], "depolarisation": ["depolarization"],
       "hyperpolarisation": ["hyperpolarization"], "signalling": ["signaling"],
       "tumour": ["tumor"], "fibre": ["fiber"], "behaviour": ["behavior"]}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def open_db(path):
    # read-only; check_same_thread off so a threaded HTTP server can use it
    # (callers serialize access with a lock)
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                         check_same_thread=False)
    db.execute("PRAGMA query_only=ON")
    return db


def terms_of(question):
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", question.lower())
    return [w for w in words if w not in STOP and len(w) > 2]


def retrieve(db, question, k=2, max_ctx_words=90):
    ts = terms_of(question)
    if not ts:
        return []
    wide = ts + [s for t in ts for s in SYN.get(t, [])]
    rows = []
    for q in (" ".join(f'"{t}"' for t in ts),            # all terms (AND)
              " OR ".join(f'"{t}"' for t in wide)):      # any term + synonyms
        try:
            rows = db.execute(
                "SELECT pmid, title, abstract, bm25(docs) FROM docs "
                "WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT 20",
                (q,)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if rows:
            break
    if not rows:
        return []

    main = ts[0] if len(ts) == 1 else " ".join(ts)

    DEF_PAT = (r"\b{}\w*\s+(?:is|are)\s+(?:defined|(?:a|an|the)\s+"
               r"(?!important|critical|key|major|essential|common|significant|"
               r"interesting|promising|powerful|useful|valuable)\w+)")
    WEAK_PAT = r"\b{}\w*\s+(?:is|are)\s+(?:also|likely|well|not|often|still)\b"

    def score(row):
        pmid, title, ab, bm = row
        s = -bm                                  # higher = better
        first = " ".join(SENT_SPLIT.split(ab)[:2]).lower()
        for t in ts:                              # true-definition bonus
            if re.search(DEF_PAT.format(re.escape(t)), first):
                s += 12
            elif re.search(WEAK_PAT.format(re.escape(t)), first):
                s -= 4
        if all(t in title.lower() for t in ts):
            s += 3
        return s

    rows.sort(key=score, reverse=True)
    out = []
    for pmid, title, ab, _ in rows[:k]:
        sents = SENT_SPLIT.split(ab)
        # prefer the window holding a definitional sentence for the main term
        start = 0
        for i, s in enumerate(sents):
            if re.search(DEF_PAT.format(re.escape(ts[0])), s.lower()):
                start = max(0, i - 1)
                break
        ctx_words = []
        for s in sents[start:]:
            ctx_words += s.split()
            if len(ctx_words) >= max_ctx_words:
                break
        out.append({"pmid": pmid, "title": title,
                    "ctx": " ".join(ctx_words[:max_ctx_words])})
    return out


def retrieve_def(db, question, k=6, max_ctx_words=90):
    """Definition-targeted retrieval: phrase-match '<term> is/are' directly,
    so the hits CONTAIN a defining sentence by construction."""
    ts = terms_of(question)
    if not ts:
        return []
    phrase = " ".join(ts)
    q = f'"{phrase} is" OR "{phrase} are"'
    try:
        rows = db.execute(
            "SELECT pmid, title, abstract, bm25(docs) FROM docs "
            "WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
            (q, k * 3)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    out = []
    pat = re.compile(rf"\b{re.escape(ts[0])}[\w\-]*(?:\s+\w+)*?\s+(?:is|are)\s+"
                     r"(?:a|an|the|defined)\b", re.I)
    for pmid, title, ab, _ in rows:
        sents = SENT_SPLIT.split(ab)
        start = 0
        for i, s in enumerate(sents):
            if phrase in s.lower() and re.search(r"\b(is|are)\b", s.lower()):
                start = max(0, i - 1)
                break
        ctx_words = []
        for s in sents[start:]:
            ctx_words += s.split()
            if len(ctx_words) >= max_ctx_words:
                break
        ctx = " ".join(ctx_words[:max_ctx_words])
        if phrase in ctx.lower():
            out.append({"pmid": pmid, "title": title, "ctx": ctx})
        if len(out) >= k:
            break
    return out


if __name__ == "__main__":
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    db = open_db(os.path.join(here, "out",
                              sys.argv[2] if len(sys.argv) > 2 else "fts.db"))
    for r in retrieve(db, sys.argv[1], k=3):
        print(r["pmid"], "|", r["title"][:70])
        print("   ", r["ctx"][:200], "\n")
