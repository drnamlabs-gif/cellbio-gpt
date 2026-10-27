# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 11b: self-supervised definition-QA mined from our own corpus.

A 40M model can't answer "what is X?" from parametric memory (stage 6e), and
it has never been taught to READ a passed-in context. This builds that skill
without any teacher model: find sentences of the form "<term> is/are a ...",
keep the surrounding abstract as context, ask "What is <term>?", and the gold
answer is the definitional sentence itself — extractive reading, the honest
primitive at this scale.

Outputs out/: defqa_train.jsonl, defqa_val.jsonl (combo-free; split by pmid
md5), defqa_meta.json. Seed 42. --smoke = 3 slices.
"""

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))
from bpe import BPE  # noqa: E402

CORPUS = os.path.join(HERE, "..", "data", "corpus")
OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
PROMPT_FMT = "### Instruction:\n{}\n\n### Response:\n"
Q_FMT = ["Context: {ctx}\n\nAnswer from the context: What {be} {term}?",
         "Context: {ctx}\n\nUsing only the context above, define {term}."]
MAX_TOK, N_TARGET, PER_SLICE, VAL_FRAC = 496, 30000, 80, 0.03

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
DEF_RE = re.compile(
    r"^(?:The\s|An?\s)?([A-Z][A-Za-z0-9\-/]*(?:\s[a-z0-9\-/()+']+){0,4})\s"
    r"(is|are)\s(?:an?\s|the\s|defined\s|one\s|essential|a\stype|a\sclass|"
    r"a\skey|a\smajor|a\scommon|important|widely|known|characterized)")
BAD_STARTS = {"there", "it", "this", "these", "those", "that", "here", "we",
              "they", "he", "she", "however", "thus", "although", "results",
              "data", "methods", "conclusion", "background", "objective",
              "aim", "purpose", "study", "studies", "patients", "treatment"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tok = BPE.load(TOKENIZER)
    files = sorted(glob.glob(os.path.join(CORPUS, "*.jsonl.gz")))
    if args.smoke:
        files = files[:3]
    train, val, seen_terms = [], [], {}
    for fi, fp in enumerate(files):
        kept = 0
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                if kept >= PER_SLICE:
                    break
                r = json.loads(line)
                ab = r.get("abstract", "")
                if len(ab) < 200:
                    continue
                sents = SENT_SPLIT.split(ab)
                for si, s in enumerate(sents):
                    m = DEF_RE.match(s.strip())
                    if not m:
                        continue
                    term, be = m.group(1).strip(), m.group(2)
                    tl = term.lower()
                    if (tl.split()[0] in BAD_STARTS or len(term) < 3
                            or len(s.split()) < 8 or len(s.split()) > 60
                            or seen_terms.get(tl, 0) >= 4):
                        continue
                    ctx_sents = sents[max(0, si - 2):si + 3]
                    ctx = " ".join(x.strip() for x in ctx_sents)
                    v = int(hashlib.md5(
                        f"defqa|{r['pmid']}|{si}".encode()).hexdigest()[:8], 16)
                    q = Q_FMT[v % 2].format(ctx=ctx, be=be, term=term)
                    full = PROMPT_FMT.format(q) + s.strip()
                    nt = len(tok.encode(full))
                    while nt > MAX_TOK and len(ctx_sents) > 1:
                        ctx_sents = ctx_sents[:-1] if si - max(0, si - 2) < len(
                            ctx_sents) - 1 else ctx_sents[1:]
                        ctx = " ".join(x.strip() for x in ctx_sents)
                        q = Q_FMT[v % 2].format(ctx=ctx, be=be, term=term)
                        nt = len(tok.encode(PROMPT_FMT.format(q) + s.strip()))
                    if nt > MAX_TOK or s.strip() not in ctx:
                        continue
                    seen_terms[tl] = seen_terms.get(tl, 0) + 1
                    ex = {"pmid": r["pmid"], "term": term, "prompt": q,
                          "answer": s.strip()}
                    (val if (v / 0xFFFFFFFF) < VAL_FRAC else train).append(ex)
                    kept += 1
                    if kept >= PER_SLICE:
                        break
        if fi % 40 == 0 or fi == len(files) - 1:
            print(f"{fi + 1}/{len(files)} slices, {len(train):,} train "
                  f"{len(val):,} val", flush=True)
        if len(train) + len(val) >= N_TARGET:
            break
    with open(os.path.join(OUT, "defqa_train.jsonl"), "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(p) + "\n")
    with open(os.path.join(OUT, "defqa_val.jsonl"), "w", encoding="utf-8") as f:
        for p in val:
            f.write(json.dumps(p) + "\n")
    meta = {"seed": 42, "n_train": len(train), "n_val": len(val),
            "distinct_terms": len(seen_terms), "max_tok": MAX_TOK}
    with open(os.path.join(OUT, "defqa_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))
    for ex in train[:3]:
        print("\n--- sample:", ex["term"], "\n", ex["prompt"][:220], "\n=>",
              ex["answer"][:160])


if __name__ == "__main__":
    main()
