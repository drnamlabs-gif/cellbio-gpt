# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Diagnose the off-grid collapse: is the 40M model an implicit lookup table
(its prediction tracks the nearest NODE's truth) or did the unseen decimal
number format break its reading? Prints per-problem table + summary stats.
Saves out/offgrid_diagnosis.json."""

import json
import math
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from betse_grpo import OUT, TOKENIZER, generate_batched, load_problems  # noqa: E402
from bpe import BPE               # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

NUM_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*mV")


def log2d(a, b):
    return math.sqrt(sum((math.log2(x) - math.log2(y)) ** 2 for x, y in zip(a, b)))


def main():
    torch.manual_seed(42)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    trainf = sys.argv[sys.argv.index("--train") + 1] if "--train" in sys.argv \
        else "betse_train.jsonl"
    ckf = sys.argv[sys.argv.index("--ckpt") + 1] if "--ckpt" in sys.argv \
        else "bg8_final.pt"
    train = [p for p in load_problems(trainf) if p["kind"] == "nernst"]
    probs = [p for p in load_problems("betse_offgrid.jsonl")
             if p["kind"] == "nernst"][::2]          # one phrasing per combo
    ck = torch.load(os.path.join(OUT, ckf), weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    outs = generate_batched(model, tok, probs, device, eot, temperature=0.0)

    rows = []
    for p, (_, ans) in zip(probs, outs):
        text = tok.decode(ans[:-1] if ans and ans[-1] == eot else ans)
        m = NUM_RE.search(text)
        pred = float(m.group(1)) if m else None
        nn = min(train, key=lambda q: log2d(q["combo"], p["combo"]))
        rows.append({"combo": p["combo"], "truth": p["truth"]["value"],
                     "nn_combo": nn["combo"], "nn_truth": nn["truth"]["value"],
                     "pred": pred, "text": text[:60]})

    ok = [r for r in rows if r["pred"] is not None]
    print(f"{len(ok)}/{len(rows)} parseable predictions")
    pr = np.array([r["pred"] for r in ok])
    tr = np.array([r["truth"] for r in ok], dtype=float)
    nn = np.array([r["nn_truth"] for r in ok], dtype=float)
    print(f"MAE model vs TRUTH    : {np.abs(pr - tr).mean():6.2f} mV")
    print(f"MAE model vs NN-NODE  : {np.abs(pr - nn).mean():6.2f} mV")
    print(f"MAE nn-node vs truth  : {np.abs(nn - tr).mean():6.2f} mV   (= 1-NN's own error)")
    print(f"corr(pred, truth) {np.corrcoef(pr, tr)[0,1]:.3f} | corr(pred, nn) "
          f"{np.corrcoef(pr, nn)[0,1]:.3f}")
    closer_nn = int(np.sum(np.abs(pr - nn) < np.abs(pr - tr)))
    print(f"pred closer to nn-node than to truth: {closer_nn}/{len(ok)}")
    pred_vals = sorted({r['pred'] for r in ok})
    train_vals = sorted({q['truth']['value'] for q in train})
    novel = [v for v in pred_vals if v not in train_vals]
    print(f"distinct predicted values: {len(pred_vals)}; "
          f"NOT in train answer set: {len(novel)} {novel[:12]}")
    print("\nsample rows (combo -> truth | nn_truth | pred):")
    for r in rows[:12]:
        print(f"  {str(r['combo']):20s} -> {r['truth']:+4d} | {r['nn_truth']:+4d} | "
              f"{r['pred'] if r['pred'] is not None else 'PARSE-FAIL'}   '{r['text'][:40]}'")
    with open(os.path.join(OUT, "offgrid_diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("saved", os.path.join(OUT, "offgrid_diagnosis.json"))


if __name__ == "__main__":
    main()
