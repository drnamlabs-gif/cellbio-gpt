# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8e verdict: case-2 (random-point-trained) models on the sealed
off-grid exam, with floors computed from the case-2 phone book
(betse_train2_tr.jsonl). One-time unsealing. Saves out/case2_offgrid.json."""

import json
import math
import os
import sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from betse_grpo import OUT, TOKENIZER, eval_val, load_problems  # noqa: E402
from bpe import BPE                       # noqa: E402
from model import GPT, GPTConfig          # noqa: E402
from bioelectric import reward as score   # noqa: E402


def dist(a, b):
    return sum((math.log2(x) - math.log2(y)) ** 2 for x, y in zip(a, b))


def metrics(val, texts):
    rs, by = [], defaultdict(list)
    for p, t in zip(val, texts):
        r = score(p["kind"], p["truth"], t)
        rs.append(r)
        by[p["kind"]].append(r)
    return {"val_reward": float(np.mean(rs)),
            "exact": float(np.mean([r == 1.0 for r in rs])),
            "by_kind": {k: float(np.mean(v)) for k, v in by.items()}}


def main():
    torch.manual_seed(42)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    train = load_problems("betse_train2_tr.jsonl")
    exam = load_problems("betse_offgrid.jsonl")
    results = {}

    const = {k: Counter(p["answer"] for p in train if p["kind"] == k)
             .most_common(1)[0][0] for k in {p["kind"] for p in train}}
    results["constant (train2 mode)"] = metrics(exam, [const[p["kind"]] for p in exam])
    nn_texts = []
    for p in exam:
        pool = [q for q in train if q["kind"] == p["kind"]]
        nn_texts.append(min(pool, key=lambda q: dist(q["combo"], p["combo"]))["answer"])
    results["1-NN on train2 (no LLM)"] = metrics(exam, nn_texts)

    for name, fn in [("case2 warm-start SFT", "bg8_wsr.pt"),
                     ("case2 +GRPO vs simulator", "bg8_finalr.pt")]:
        ck = torch.load(os.path.join(OUT, fn), weights_only=False)
        model = GPT(GPTConfig(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        vr, ve, vk = eval_val(model, tok, exam, device, eot)
        results[name] = {"val_reward": vr, "exact": ve, "by_kind": vk}
        del model
        torch.cuda.empty_cache()

    for n, r in results.items():
        print(f"{n:28s} val {r['val_reward']:.3f} exact {r['exact']:.3f} {r['by_kind']}")
    with open(os.path.join(OUT, "case2_offgrid.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": 42, "exam": "betse_offgrid.jsonl",
                   "results": results}, f, indent=2)
    print("saved", os.path.join(OUT, "case2_offgrid.json"))


if __name__ == "__main__":
    main()
