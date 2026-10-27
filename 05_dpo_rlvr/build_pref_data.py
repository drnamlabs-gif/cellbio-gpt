# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 5a-data: mine DPO preference pairs — no external LLM, no human labels.

For each SFT prompt (tasks: qtitle, defn, abbrev — titlegen already works):
  chosen   = the gold answer from the corpus
  rejected = the SFT model's OWN sample at temperature 0.8 (on-policy — it
             punishes the habits the model actually has), except for half the
             abbrev prompts where rejected = the observed circular failure
             "ABBR stands for ABBR." (hard negative, targets a real bug).
Pairs where the sample equals the gold are dropped.

Generation is batched by exact prompt length (prompts of equal token count are
stacked), KV-cache on. Output: out/pref_train.jsonl, out/pref_val.jsonl,
out/pref_meta.json. Seed 42. --smoke = 300 prompts.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))
sys.path.insert(0, os.path.join(HERE, "..", "02_transformer"))
sys.path.insert(0, os.path.join(HERE, "..", "04_sft"))
from bpe import BPE                      # noqa: E402
from model import GPT, GPTConfig        # noqa: E402
from sft import PROMPT_FMT              # noqa: E402

SEED = 42
OUT = os.path.join(HERE, "out")
SFT_OUT = os.path.join(HERE, "..", "04_sft", "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
SFT_MODEL = os.path.join(SFT_OUT, "sft_final.pt")

PER_TASK = 4000          # prompts sampled per task
GEN_TOKENS = 80
TEMP = 0.8
ABBR_RE = re.compile(r"what does ([A-Za-z]{2,6}s?) stand for", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda"

    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    ck = torch.load(SFT_MODEL, weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"SFT model loaded (step {ck['step']}, val {ck['val_loss']:.4f})")

    rng = np.random.default_rng(SEED)
    by_task = defaultdict(list)
    with open(os.path.join(SFT_OUT, "sft_train.jsonl"), encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if ex["task"] in ("qtitle", "defn", "abbrev"):
                by_task[ex["task"]].append(ex)

    per = 100 if args.smoke else PER_TASK
    picked = []
    for k, v in by_task.items():
        idx = rng.permutation(len(v))[:per]
        picked.extend(v[i] for i in idx)
    print(f"{len(picked)} prompts picked ({per}/task)")

    # half of abbrev -> circular hard negative, no generation needed
    pairs, to_gen = [], []
    for ex in picked:
        if ex["task"] == "abbrev" and rng.random() < 0.5:
            m = ABBR_RE.search(ex["prompt"])
            if m:
                pairs.append({**ex, "rejected": f"{m.group(1)} stands for {m.group(1)}.",
                              "neg_type": "circular"})
                continue
        to_gen.append(ex)

    # batch generation, grouped by exact prompt token length
    buckets = defaultdict(list)
    for ex in to_gen:
        ids = tok.encode(PROMPT_FMT.format(ex["prompt"]))
        if len(ids) <= 400:
            buckets[len(ids)].append((ex, ids))
    t0, n_done = time.time(), 0
    with torch.no_grad():
        for L, items in sorted(buckets.items()):
            for i in range(0, len(items), 64):
                chunk = items[i:i + 64]
                idx = torch.tensor([ids for _, ids in chunk], device=device)
                out = model.generate(idx, GEN_TOKENS, temperature=TEMP, top_k=50)
                for (ex, _), row in zip(chunk, out.tolist()):
                    ans = row[L:]
                    ans = ans[:ans.index(eot)] if eot in ans else ans
                    rej = tok.decode(ans).strip()
                    if rej and rej != ex["answer"].strip():
                        pairs.append({**ex, "rejected": rej, "neg_type": "on_policy"})
                n_done += len(chunk)
            if n_done and n_done % 2000 < 64:
                print(f"  generated {n_done}/{len(to_gen)} "
                      f"({n_done/(time.time()-t0):.0f}/s)", flush=True)

    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order]
    n_val = 50 if args.smoke else 500
    val, train = pairs[:n_val], pairs[n_val:]
    suff = "_smoke" if args.smoke else ""
    for name, part in (("pref_train", train), ("pref_val", val)):
        with open(os.path.join(OUT, f"{name}{suff}.jsonl"), "w", encoding="utf-8") as f:
            for ex in part:
                f.write(json.dumps({k: ex[k] for k in
                                    ("task", "prompt", "answer", "rejected", "neg_type")})
                        + "\n")
    stats = {"n_train": len(train), "n_val": len(val),
             "by_task": {k: sum(1 for e in train if e["task"] == k) for k in by_task},
             "by_neg": {k: sum(1 for e in train if e["neg_type"] == k)
                        for k in ("on_policy", "circular")},
             "seed": SEED, "temp": TEMP, "date": "2026-10-23"}
    json.dump(stats, open(os.path.join(OUT, f"pref_meta{suff}.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))
    for ex in train[:3]:
        print(f"\n[{ex['task']}/{ex['neg_type']}] Q: {ex['prompt'][:90]}\n"
              f"  chosen:   {ex['answer'][:110]}\n  rejected: {ex['rejected'][:110]}")


if __name__ == "__main__":
    main()
