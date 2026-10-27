# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8b: RLVR where the verifier is the real BETSE simulator's output.

Two phases in one script (seed 42):
  1. warm-start: SFT on the TRAIN-combo gold answers (teaches the answer
     format and memorizes seen combos — the "evolution" baseline)
  2. GRPO: 16 prompts x K=8 samples/step, graded reward() against cached sim
     truths (numeric mV bands 1.0/0.6/0.3/0.05; categorical exact), group
     advantage, KL 0.05 leash to the warm-start, numeric oversampled 60%.

Eval: greedy on the held-out-combo val problems every 50 steps — generalization
means predicting the simulator's outcome for ion/permeability combos never
seen in training. Saves out/bg8_log.jsonl, bg8_ws.pt, bg8_best.pt, bg8_final.pt.
--smoke = 60 SFT + 20 GRPO steps.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))
sys.path.insert(0, os.path.join(HERE, "..", "02_transformer"))
sys.path.insert(0, os.path.join(HERE, "..", "04_sft"))
sys.path.insert(0, os.path.join(HERE, "..", "05_dpo_rlvr"))
from bpe import BPE                      # noqa: E402
from model import GPT, GPTConfig         # noqa: E402
from sft import PROMPT_FMT               # noqa: E402
from bioelectric import reward as score  # noqa: E402

SEED = 42
OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
INIT = os.path.join(HERE, "..", "05_dpo_rlvr", "out", "dpo_final_v2.pt")

WS_STEPS, WS_BATCH, WS_LR = 300, 32, 3e-5
P_PROMPTS, K, MAX_NEW = 16, 8, 16
RL_STEPS, RL_LR, WARMUP, KL_COEF = 200, 2e-6, 20, 0.05
EVAL_EVERY = 50


def load_problems(name):
    return [json.loads(l) for l in open(os.path.join(OUT, name), encoding="utf-8")]


@torch.no_grad()
def generate_batched(model, tok, problems, device, eot, temperature):
    buckets = defaultdict(list)
    enc = []
    for i, p in enumerate(problems):
        ids = tok.encode(PROMPT_FMT.format(p["prompt"]))
        enc.append(ids)
        buckets[len(ids)].append(i)
    outs = [None] * len(problems)
    for L, idxs in sorted(buckets.items()):
        for s in range(0, len(idxs), 128):
            chunk = idxs[s:s + 128]
            x = torch.tensor([enc[i] for i in chunk], device=device)
            out = model.generate(x, MAX_NEW, temperature=temperature)
            for i, row in zip(chunk, out.tolist()):
                ans = row[L:]
                ans = ans[:ans.index(eot) + 1] if eot in ans else ans
                outs[i] = (enc[i], ans)
    return outs


@torch.no_grad()
def eval_val(model, tok, val, device, eot):
    outs = generate_batched(model, tok, val, device, eot, temperature=0.0)
    rs, by_kind = [], defaultdict(list)
    for p, (_, ans) in zip(val, outs):
        text = tok.decode(ans[:-1] if ans and ans[-1] == eot else ans)
        r = score(p["kind"], p["truth"], text)
        rs.append(r)
        by_kind[p["kind"]].append(r)
    return (float(np.mean(rs)), float(np.mean([r == 1.0 for r in rs])),
            {k: float(np.mean(v)) for k, v in by_kind.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", default="betse_train.jsonl")
    ap.add_argument("--val", default="betse_val.jsonl")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    ws_steps = 60 if args.smoke else WS_STEPS
    rl_steps = 20 if args.smoke else RL_STEPS
    suff = "_smoke" if args.smoke else args.suffix

    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: <=30GB of 32GB
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    train = load_problems(args.train)
    val = load_problems(args.val)
    tr_num = [p for p in train if p["kind"] == "nernst"]
    tr_cat = [p for p in train if p["kind"] != "nernst"]
    rng = np.random.default_rng(SEED)
    print(f"{len(train)} train ({len(tr_num)} numeric), {len(val)} val (held-out combos)")

    ck = torch.load(INIT, weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    policy = GPT(cfg).to(device)
    policy.load_state_dict(ck["model"])

    # ---- phase 1: warm-start SFT on train gold answers ----
    xs, ys = [], []
    for p in train:
        pr = tok.encode(PROMPT_FMT.format(p["prompt"]))
        ids = pr + tok.encode(p["answer"]) + [eot]
        xs.append(ids)
        ys.append([-100] * (len(pr) - 1) + ids[len(pr):])
    opt = torch.optim.AdamW(policy.parameters(), lr=WS_LR, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=True)
    order = rng.permutation(len(xs))
    policy.train()
    for step in range(ws_steps):
        for g in opt.param_groups:
            g["lr"] = WS_LR * 0.5 * (1 + math.cos(math.pi * step / ws_steps))
        idx = order[(step * WS_BATCH) % len(xs):][:WS_BATCH]
        if len(idx) < WS_BATCH:
            order = rng.permutation(len(xs))
            idx = order[:WS_BATCH]
        L = max(len(xs[i]) for i in idx)
        xb = np.full((len(idx), L - 1), eot, dtype=np.int64)
        yb = np.full((len(idx), L - 1), -100, dtype=np.int64)
        for r, i in enumerate(idx):
            xb[r, :len(xs[i]) - 1] = xs[i][:-1]
            yb[r, :len(ys[i])] = ys[i]
        xt = torch.from_numpy(xb).to(device)
        yt = torch.from_numpy(yb).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = policy(xt)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               yt.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == ws_steps - 1:
            print(f"ws {step:3d}/{ws_steps} loss {loss.item():.4f}", flush=True)
    policy.eval()
    torch.save({"model": policy.state_dict(), "cfg": vars(cfg), "seed": SEED},
               os.path.join(OUT, f"bg8_ws{suff}.pt"))

    ref = GPT(cfg).to(device)
    ref.load_state_dict(policy.state_dict())
    ref.eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=RL_LR, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=True)

    r0, e0, k0 = eval_val(policy, tok, val, device, eot)
    print(f"warm-start baseline: val {r0:.3f} exact {e0:.3f} {k0}")
    log_f = open(os.path.join(OUT, f"bg8_log{suff}.jsonl"), "a", encoding="utf-8")
    log_f.write(json.dumps({"step": -1, "val_reward": r0, "val_exact": e0,
                            "by_kind": k0, "baseline": True, "seed": SEED}) + "\n")

    # ---- phase 2: GRPO against the sim-truth verifier ----
    best = r0
    t0 = time.time()
    for step in range(rl_steps):
        for g in opt.param_groups:
            g["lr"] = RL_LR * (step + 1) / WARMUP if step < WARMUP else RL_LR
        n_num = int(P_PROMPTS * 0.6)
        picks = [tr_num[i] for i in rng.integers(0, len(tr_num), n_num)] + \
                [tr_cat[i] for i in rng.integers(0, len(tr_cat), P_PROMPTS - n_num)]
        probs = [p for p in picks for _ in range(K)]
        policy.eval()
        outs = generate_batched(policy, tok, probs, device, eot, temperature=1.0)
        policy.train()
        rs = np.zeros(len(probs))
        for i, (p, (_, ans)) in enumerate(zip(probs, outs)):
            text = tok.decode(ans[:-1] if ans and ans[-1] == eot else ans)
            rs[i] = score(p["kind"], p["truth"], text)
        adv = np.zeros_like(rs)
        for gi in range(P_PROMPTS):
            grp = rs[gi * K:(gi + 1) * K]
            if grp.std() > 1e-6:
                adv[gi * K:(gi + 1) * K] = (grp - grp.mean()) / (grp.std() + 1e-6)
        keep = [i for i in range(len(probs)) if adv[i] != 0 and len(outs[i][1]) > 0]
        if not keep:
            print(f"step {step}: all groups uniform, skipped", flush=True)
            continue
        L = max(len(outs[i][0]) + len(outs[i][1]) for i in keep)
        B = len(keep)
        x = np.full((B, L - 1), eot, dtype=np.int64)
        y = np.full((B, L - 1), -100, dtype=np.int64)
        a_tok = np.zeros((B, L - 1), dtype=np.float32)
        for r_i, i in enumerate(keep):
            pr, ans = outs[i]
            ids = pr + ans
            x[r_i, :len(ids) - 1] = ids[:-1]
            y[r_i, len(pr) - 1:len(ids) - 1] = ans
            a_tok[r_i, len(pr) - 1:len(ids) - 1] = adv[i]
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        at = torch.from_numpy(a_tok).to(device)
        mask = yt != -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = policy(xt)
        logp = F.log_softmax(logits.float(), dim=-1)
        tok_lp = torch.gather(logp, 2, yt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rlogits, _, _ = ref(xt)
        rlogp = F.log_softmax(rlogits.float(), dim=-1)
        rtok_lp = torch.gather(rlogp, 2, yt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        n_tok = mask.sum()
        pg = -(at * tok_lp)[mask].sum() / n_tok
        kl = (tok_lp - rtok_lp)[mask].sum() / n_tok
        loss = pg + KL_COEF * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 or step == rl_steps - 1:
            print(f"rl {step:3d}/{rl_steps}  R {rs.mean():.3f}  exact "
                  f"{np.mean(rs == 1.0):.3f}  kl {kl.item():+.4f}  kept {B}/128  "
                  f"{(time.time() - t0) / (step + 1):.2f} s/step", flush=True)
            log_f.write(json.dumps({"step": step, "reward": rs.mean(),
                                    "exact": float(np.mean(rs == 1.0)),
                                    "kl": kl.item(), "kept": B}) + "\n")
            log_f.flush()
        if (step + 1) % EVAL_EVERY == 0 or step == rl_steps - 1:
            vr, ve, vk = eval_val(policy, tok, val, device, eot)
            print(f"  VAL {vr:.3f} exact {ve:.3f} {vk}", flush=True)
            log_f.write(json.dumps({"step": step, "val_reward": vr,
                                    "val_exact": ve, "by_kind": vk}) + "\n")
            log_f.flush()
            if vr > best:
                best = vr
                torch.save({"model": policy.state_dict(), "cfg": vars(cfg),
                            "step": step, "val_reward": vr, "seed": SEED},
                           os.path.join(OUT, f"bg8_best{suff}.pt"))
    vr, ve, vk = eval_val(policy, tok, val, device, eot)
    torch.save({"model": policy.state_dict(), "cfg": vars(cfg), "seed": SEED,
                "val_reward": vr}, os.path.join(OUT, f"bg8_final{suff}.pt"))
    log_f.write(json.dumps({"step": rl_steps, "val_reward": vr, "val_exact": ve,
                            "by_kind": vk, "final": True}) + "\n")
    log_f.close()
    print(f"DONE. warm-start {r0:.3f} -> GRPO {vr:.3f} (best {best:.3f}); {vk}")


if __name__ == "__main__":
    main()
