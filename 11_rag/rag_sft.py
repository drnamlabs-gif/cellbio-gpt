# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 11d: teach the 40M to READ — SFT on the self-mined definition QA.

From dpo_final_v2 (chat model, pre-RL-tax), 800 steps batch 32, prompt
masked. Eval on 946 held-out items: token-F1 + exact match + faithfulness
(answer is a substring of the given context). Ends with the live probe:
the three questions that exposed the encyclopedia hole, answered with real
BM25 retrieval from out/fts.db. Saves out/rag_qa.pt, rag_log.json. Seed 42.
--smoke = 60 steps, 60 val items.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))
sys.path.insert(0, os.path.join(HERE, "..", "02_transformer"))
sys.path.insert(0, HERE)
from bpe import BPE               # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from rag_retrieve import open_db, retrieve  # noqa: E402

SEED = 42
OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
INIT = os.path.join(HERE, "..", "05_dpo_rlvr", "out", "dpo_final_v2.pt")
PROMPT_FMT = "### Instruction:\n{}\n\n### Response:\n"
STEPS, BATCH, LR, MAX_NEW = 800, 32, 3e-5, 96


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def f1(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


@torch.no_grad()
def gen(model, tok, prompts, device, eot):
    outs = []
    for i in range(0, len(prompts), 64):
        batch = prompts[i:i + 64]
        enc = [tok.encode(PROMPT_FMT.format(p)) for p in batch]
        L = max(len(e) for e in enc)
        x = torch.full((len(enc), L), eot, dtype=torch.long, device=device)
        for r, e in enumerate(enc):
            x[r, L - len(e):] = torch.tensor(e, device=device)  # left-pad
        out = model.generate(x, MAX_NEW, temperature=0.0)
        for r, e in enumerate(enc):
            ans = out[r].tolist()[L:]
            ans = ans[:ans.index(eot)] if eot in ans else ans
            outs.append(tok.decode(ans).strip())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    steps = 60 if args.smoke else STEPS
    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: <=30GB of 32GB
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    train = [json.loads(l) for l in open(os.path.join(OUT, "defqa_train.jsonl"),
                                         encoding="utf-8")]
    val = [json.loads(l) for l in open(os.path.join(OUT, "defqa_val.jsonl"),
                                       encoding="utf-8")]
    if args.smoke:
        val = val[:60]
    rng = np.random.default_rng(SEED)
    print(f"{len(train)} train, {len(val)} val")

    ck = torch.load(INIT, weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    xs, ys = [], []
    for p in train:
        pr = tok.encode(PROMPT_FMT.format(p["prompt"]))
        ids = pr + tok.encode(p["answer"]) + [eot]
        xs.append(ids)
        ys.append([-100] * (len(pr) - 1) + ids[len(pr):])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=True)
    order = rng.permutation(len(xs))
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = LR * 0.5 * (1 + math.cos(math.pi * step / steps))
        idx = order[(step * BATCH) % len(xs):][:BATCH]
        if len(idx) < BATCH:
            order = rng.permutation(len(xs))
            idx = order[:BATCH]
        L = max(len(xs[i]) for i in idx)
        xb = np.full((len(idx), L - 1), eot, dtype=np.int64)
        yb = np.full((len(idx), L - 1), -100, dtype=np.int64)
        for r, i in enumerate(idx):
            xb[r, :len(xs[i]) - 1] = xs[i][:-1]
            yb[r, :len(ys[i])] = ys[i]
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
        if step % 100 == 0 or step == steps - 1:
            print(f"sft {step:3d}/{steps} loss {loss.item():.4f}", flush=True)
    model.eval()
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "seed": SEED,
                "task": "defqa extractive reading"},
               os.path.join(OUT, "rag_qa.pt"))

    preds = gen(model, tok, [p["prompt"] for p in val], device, eot)
    f1s = [f1(pr, p["answer"]) for pr, p in zip(preds, val)]
    em = float(np.mean([norm(pr) == norm(p["answer"])
                        for pr, p in zip(preds, val)]))
    faith = float(np.mean([" ".join(norm(pr)) in " ".join(norm(p["prompt"]))
                           for pr, p in zip(preds, val)]))
    log = {"seed": SEED, "steps": steps, "n_train": len(train),
           "n_val": len(val), "token_f1": float(np.mean(f1s)),
           "exact": em, "faithful_substring": faith}
    print(json.dumps(log))

    db_path = os.path.join(OUT, "fts.db")
    probes = ["What is bioelectricity?", "What is membrane voltage?",
              "What does membrane depolarization trigger?"]
    if os.path.exists(db_path):
        db = open_db(db_path)
        log["probes"] = []
        for q in probes:
            hits = retrieve(db, q, k=1)
            if not hits:
                print(f"\nQ: {q}\n  (no retrieval hit)")
                continue
            h = hits[0]
            full = f"Context: {h['ctx']}\n\nAnswer from the context: {q}"
            ans = gen(model, tok, [full], device, eot)[0]
            print(f"\nQ: {q}\n  src PMID {h['pmid']}: {h['title'][:70]}\n"
                  f"  A: {ans[:260]}")
            log["probes"].append({"q": q, "pmid": h["pmid"],
                                  "title": h["title"], "answer": ans})
    with open(os.path.join(OUT, "rag_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
