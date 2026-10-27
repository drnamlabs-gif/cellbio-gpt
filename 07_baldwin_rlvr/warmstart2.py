# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 7 warm-start v2: teach the PLASTICITY CHANNEL itself.

The stage-5b warm-start only ever saw single-instruction prompts, so worked
examples in context are out-of-distribution noise (measured: sens gap -0.46 —
the scaffold HURTS). Hinton & Nowlan's genotypes could learn; ours must be
taught how to learn before Baldwinian selection has anything to act on.

Same recipe as 5b's warmstart (init dpo_final_v2, 400 SFT steps on the 3,000
ws problems, LR 3e-5 cosine) with ONE change: each example is formatted with
k ~ U{0..5} worked neighbor examples (train-pool scaffolds, loss only on the
final answer). Saves out/ws2_final.pt + a before/after sens check. Seed 42.
"""

import json
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baldwin_grpo import (BPE, GPT, GPTConfig, MAX_SEQ, TOKENIZER,  # noqa: E402
                          build_scaffold, eval_sens, grpo)
from collections import defaultdict  # noqa: E402

SEED = 42
OUT = os.path.join(HERE, "out")
INIT = os.path.join(HERE, "..", "05_dpo_rlvr", "out", "dpo_final_v2.pt")
STEPS, BATCH, LR = 400, 32, 3e-5


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: ≤30GB of 32GB
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    ws = grpo.load_problems("rlvr_ws.jsonl")
    train = grpo.load_problems("rlvr_train.jsonl")
    val = grpo.load_problems("rlvr_val.jsonl")
    pool_by_kind = defaultdict(list)
    for p in train:
        pool_by_kind[p["kind"]].append(p)
    rng = np.random.default_rng(SEED)

    xs, ys = [], []
    for p in ws:
        k = int(rng.integers(0, 6))
        pr = build_scaffold(p, k, pool_by_kind, rng, tok)
        ans = tok.encode(p["answer"]) + [eot]
        ids = (pr + ans)[:MAX_SEQ]
        if len(ids) <= len(pr):
            continue
        xs.append(ids)
        # y[t] = ids[t+1], masked (-100) on the prompt part
        ys.append([-100] * (len(pr) - 1) + ids[len(pr):])
    print(f"{len(xs)} ws examples (mixed k 0..5), max len {max(map(len, xs))}")

    ck = torch.load(INIT, weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=True)
    order = rng.permutation(len(xs))
    model.train()
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = LR * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        idx = order[(step * BATCH) % len(xs):][:BATCH]
        if len(idx) < BATCH:
            order = rng.permutation(len(xs))
            idx = order[:BATCH]
        L = max(len(xs[i]) for i in idx)
        xb = np.full((len(idx), L - 1), eot, dtype=np.int64)
        yb = np.full((len(idx), L - 1), -100, dtype=np.int64)
        for r, i in enumerate(idx):
            xb[r, :len(xs[i]) - 1] = xs[i][:-1]
            yb[r, :len(ys[i])] = ys[i][: L - 1] if len(ys[i]) > L - 1 else ys[i]
        xt = torch.from_numpy(xb).to(device)
        yt = torch.from_numpy(yb).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = model(xt)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               yt.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            print(f"step {step:3d}/{STEPS} loss {loss.item():.4f}", flush=True)

    model.eval()
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "seed": SEED,
                "date": "2026-10-23", "steps": STEPS, "init": "dpo_final_v2",
                "format": "mixed-k 0..5 scaffolds"},
               os.path.join(OUT, "ws2_final.pt"))

    srng = np.random.default_rng(123)
    sub = [val[i] for i in srng.choice(len(val), 160, replace=False)]
    enc_k5 = [build_scaffold(p, 5, pool_by_kind, srng, tok) for p in sub]
    enc_k0 = [build_scaffold(p, 0, pool_by_kind, srng, tok) for p in sub]
    sens = eval_sens(model, tok, sub, enc_k5, enc_k0, device, eot)
    r0, e0, kk = grpo.eval_val(model, tok, val, device, eot)
    print(f"ws2 baseline: val(k=0) {r0:.3f} {kk} | sens {sens}")
    with open(os.path.join(OUT, "ws2_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "val_k0": r0, "by_kind": kk, "sens": sens,
                   "n_examples": len(xs)}, f, indent=2)


if __name__ == "__main__":
    main()
