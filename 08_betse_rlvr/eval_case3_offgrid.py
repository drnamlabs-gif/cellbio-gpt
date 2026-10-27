# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8e case 3: sealed off-grid exam at k=5 anchors (the 40M's 5-shot).

Same language starting line as the pretrained 5-shot condition: five worked
nearest-neighbor examples appear AS TEXT in the prompt (anchors drawn from
the random TRAIN pool only — no test truth ever enters a prompt). Evaluates
the mixed-k warm start and the k=5-GRPO checkpoint, at k=5 and k=0.
Saves out/case3_offgrid.json."""

import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baldwin_betse import (OUT, TOKENIZER, build_prompt_ids, eval_set,
                           load_problems)  # noqa: E402
from bpe import BPE               # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def main():
    torch.manual_seed(42)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    train = load_problems("betse_train2_tr.jsonl")
    exam = load_problems("betse_offgrid.jsonl")
    pool = defaultdict(list)
    for p in train:
        pool[p["kind"]].append(p)
    enc5 = [build_prompt_ids(p, 5, pool, tok) for p in exam]
    enc0 = [build_prompt_ids(p, 0, pool, tok) for p in exam]

    results = {}
    for name, fn in [("case3 ws (mixed-k)", "bw8r_ws.pt"),
                     ("case3 +GRPO@k5", "bw8_final_k5_r.pt")]:
        ck = torch.load(os.path.join(OUT, fn), weights_only=False)
        model = GPT(GPTConfig(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        r5, by5 = eval_set(model, tok, exam, enc5, device, eot)
        r0, by0 = eval_set(model, tok, exam, enc0, device, eot)
        results[name] = {"k5": {"val_reward": r5, "by_kind": by5},
                         "k0": {"val_reward": r0, "by_kind": by0}}
        print(f"{name:22s} k=5 {r5:.3f} {by5} | k=0 {r0:.3f} {by0}", flush=True)
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(OUT, "case3_offgrid.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": 42, "exam": "betse_offgrid.jsonl",
                   "anchors": "5 nearest from betse_train2_tr (text in prompt)",
                   "results": results}, f, indent=2)
    print("saved", os.path.join(OUT, "case3_offgrid.json"))


if __name__ == "__main__":
    main()
