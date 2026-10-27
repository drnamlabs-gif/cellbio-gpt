# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Track B recipe v2, post-training data: a MIXTURE instead of 100% short tasks.

Why: the mirrored SFT (97.5k PubMed tasks, median answer 15 words) taught the
model to stop after one sentence. Real domain chat models post-train on long,
multi-turn general dialogue PLUS domain tasks. Three parts, all as chat
messages (system / user / assistant):

  general     smol-smoltalk shard 0 (HuggingFaceTB, Apache 2.0): 115k
              multi-turn conversations, median 187-word answers — the data
              SmolLM2-135M-Instruct itself was tuned on
  dom_long    mined from OUR PubMed train slices (never the 5 val slices):
                explain   "Describe what the study titled ... investigated and
                          found." -> the abstract (120-350 words); 30% get a
                          2nd turn "What is the main conclusion?" -> last sentence
                findings  abstract minus its last 2 sentences + "What do these
                          results suggest?" -> those 2 sentences
  dom_short   a sample of Track A's 04_sft tasks (definitions, abbreviations,
              titles), so short questions still get short answers

Outputs data/sft_v2/{train,val}.jsonl and stats.json. Seed 42.

--decouple_system (v2.1 -> data/sft_v2b/): v2 gave EVERY domain example our
system prompt and no general example had it, so the system prompt became a
mode switch (with it: short PubMed answers; without: long generic chat). Here
every example, general or domain, draws its system prompt from the same
distribution (none / ours / generic), so only the question can decide the
answer's form.
"""

import glob
import gzip
import json
import os
import random
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "data", "sft_v2")
SEED = 42
N_GENERAL, N_EXPLAIN, N_FINDINGS, N_SHORT = 80_000, 20_000, 10_000, 20_000
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
SYSTEM = ("You are a careful assistant for cell biology and bioelectricity. "
          "Answer clearly; say so when you are unsure.")


def general(rng):
    p = os.path.join(ROOT, "data", "smoltalk", "data", "train-00000-of-00004.parquet")
    d = pd.read_parquet(p)
    idx = rng.sample(range(len(d)), N_GENERAL)
    out = []
    for i in idx:
        msgs = [{"role": m["role"], "content": m["content"]} for m in d.messages[i]]
        out.append({"src": "general:" + d.source[i], "messages": msgs})
    return out


def domain_long(rng):
    slices = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", "slice_*.jsonl.gz")))[:-5]
    rng.shuffle(slices)
    explain, findings = [], []
    for fp in slices:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                t, a = d["title"].strip(), d["abstract"].strip()
                nw = len(a.split())
                if not t or not 120 <= nw <= 350 or rng.random() > 0.05:
                    continue
                sents = SENT.split(a)
                if len(sents) < 5:
                    continue
                if len(explain) < N_EXPLAIN and rng.random() < 0.67:
                    msgs = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": f"Describe what the study titled "
                                                        f"\"{t}\" investigated and found."},
                            {"role": "assistant", "content": a}]
                    if rng.random() < 0.3:
                        msgs += [{"role": "user", "content": "What is the main conclusion?"},
                                 {"role": "assistant", "content": sents[-1]}]
                    explain.append({"src": "dom_long:explain", "pmid": d["pmid"], "messages": msgs})
                elif len(findings) < N_FINDINGS:
                    ctx = " ".join(sents[:-2])
                    findings.append({"src": "dom_long:findings", "pmid": d["pmid"], "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"{ctx}\n\nWhat do these results suggest?"},
                        {"role": "assistant", "content": " ".join(sents[-2:])}]})
                if len(explain) >= N_EXPLAIN and len(findings) >= N_FINDINGS:
                    return explain + findings
    return explain + findings


def domain_short(rng):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "04_sft", "out", "sft_train.jsonl"),
                                        encoding="utf-8")]
    out = []
    for r in rng.sample(rows, N_SHORT):
        out.append({"src": "dom_short:" + r["task"], "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["answer"]}]})
    return out


SYSTEMS = [None, SYSTEM, "You are a helpful assistant."]


def decouple(data, rng):
    for r in data:
        msgs = [m for m in r["messages"] if m["role"] != "system"]
        s = rng.choice(SYSTEMS)
        r["messages"] = ([{"role": "system", "content": s}] if s else []) + msgs
    return data


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--decouple_system", action="store_true")
    args = ap.parse_args()
    global OUT
    if args.decouple_system:
        OUT = OUT.rstrip("/\\") + "b"
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    data = general(rng) + domain_long(rng) + domain_short(rng)
    if args.decouple_system:
        data = decouple(data, random.Random(SEED + 7))
    rng.shuffle(data)
    n_val = len(data) // 100
    stats = {}
    for name, part in (("val", data[:n_val]), ("train", data[n_val:])):
        with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        stats[name] = len(part)
    by = {}
    for r in data:
        k = r["src"].split(":")[0] if r["src"].startswith("general") else r["src"]
        w = [len(m["content"].split()) for m in r["messages"] if m["role"] == "assistant"]
        b = by.setdefault(k, {"n": 0, "words": []})
        b["n"] += 1
        b["words"].append(sum(w) / len(w))
    for k, b in by.items():
        ws = sorted(b.pop("words"))
        b["median_answer_words"] = ws[len(ws) // 2]
    stats["parts"] = by
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
