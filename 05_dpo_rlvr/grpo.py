# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 5b: GRPO with the bioelectric-physics verifier (RLVR).

Per step:
  1. draw P prompts, sample K answers each (temp 1.0, the policy's own dist)
  2. verifier scores every answer -> rewards r in [0,1]  (bioelectric.reward:
     Nernst-equation ground truth; the model only ever sees the scalar)
  3. group advantage A_i = (r_i - mean_group) / (std_group + eps); groups with
     zero std contribute no gradient (all-same answers teach nothing)
  4. policy-gradient loss on answer tokens: -mean(A * logpi_theta), plus a
     KL penalty KL_COEF * mean(logpi_theta - logpi_ref) keeping the policy
     near the warm-start reference (no reward model, no critic — that is GRPO)

Init + reference: ws_final.pt. Eval: greedy on the 727 combo-held-out val
problems every 50 steps (accuracy = mean reward; exact = reward==1 rate).
Saves grpo_log.jsonl, grpo_best.pt (by val reward), grpo_final.pt. Seed 42.
"""

import argparse
import json
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
from bpe import BPE                      # noqa: E402
from model import GPT, GPTConfig        # noqa: E402
from sft import PROMPT_FMT              # noqa: E402
from bioelectric import reward as score  # noqa: E402

SEED = 42
OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
WS = os.path.join(OUT, "ws_final.pt")

P_PROMPTS, K = 16, 8          # 128 sequences per step
MAX_NEW = 24
STEPS = 300
LR, WARMUP = 2e-6, 20
KL_COEF = 0.05
EVAL_EVERY, CKPT_EVERY = 50, 100


def load_problems(name):
    return [json.loads(l) for l in open(os.path.join(OUT, name), encoding="utf-8")]


@torch.no_grad()
def generate_batched(model, tok, problems, device, eot, temperature):
    """Returns list of sampled token-lists (answer part only), bucketed by length."""
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
    args = ap.parse_args()
    steps_total = 30 if args.smoke else STEPS
    suff = "_smoke" if args.smoke else ""

    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    train = load_problems("rlvr_train.jsonl")
    val = load_problems("rlvr_val.jsonl")
    rng = np.random.default_rng(SEED)
    # GRPO draws concentrate where headroom is: warm-start already saturates
    # the categorical tasks, so 60% of prompts are the numeric nernst task
    tr_nernst = [p for p in train if p["kind"] == "nernst"]
    tr_other = [p for p in train if p["kind"] != "nernst"]
    print(f"{len(train)} train problems ({len(tr_nernst)} nernst, oversampled 60%), "
          f"{len(val)} val (combo-held-out)")

    ck = torch.load(WS, weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    policy = GPT(cfg).to(device)
    policy.load_state_dict(ck["model"])
    ref = GPT(cfg).to(device)
    ref.load_state_dict(ck["model"])
    ref.eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=True)

    r0, e0, k0 = eval_val(policy, tok, val, device, eot)
    print(f"WARM-START baseline: val reward {r0:.3f}, exact {e0:.3f}, {k0}")
    log_f = open(os.path.join(OUT, f"grpo_log{suff}.jsonl"), "a", encoding="utf-8")
    log_f.write(json.dumps({"step": -1, "val_reward": r0, "val_exact": e0,
                            "by_kind": k0, "baseline": True}) + "\n")

    best = r0
    t0 = time.time()
    for step in range(steps_total):
        for g in opt.param_groups:
            g["lr"] = LR * (step + 1) / WARMUP if step < WARMUP else LR

        n_num = int(P_PROMPTS * 0.6)
        picks = [tr_nernst[i] for i in rng.integers(0, len(tr_nernst), n_num)] + \
                [tr_other[i] for i in rng.integers(0, len(tr_other), P_PROMPTS - n_num)]
        probs = [p for p in picks for _ in range(K)]
        policy.eval()
        outs = generate_batched(policy, tok, probs, device, eot, temperature=1.0)
        policy.train()

        rs = np.zeros(len(probs), dtype=np.float64)
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

        if step % 5 == 0 or step == steps_total - 1:
            dt = (time.time() - t0) / (step + 1)
            print(f"step {step:4d}/{steps_total}  R {rs.mean():.3f}  "
                  f"exact {np.mean(rs == 1.0):.3f}  kl {kl.item():+.4f}  "
                  f"kept {B}/{len(probs)}  {dt:.2f} s/step", flush=True)
            log_f.write(json.dumps({"step": step, "reward": rs.mean(),
                                    "exact": float(np.mean(rs == 1.0)),
                                    "kl": kl.item(), "kept": B}) + "\n")
            log_f.flush()

        if (step + 1) % EVAL_EVERY == 0 or step == steps_total - 1:
            vr, ve, vk = eval_val(policy, tok, val, device, eot)
            print(f"  VAL reward {vr:.3f}  exact {ve:.3f}  {vk}", flush=True)
            log_f.write(json.dumps({"step": step, "val_reward": vr,
                                    "val_exact": ve, "by_kind": vk}) + "\n")
            log_f.flush()
            if vr > best:
                best = vr
                torch.save({"model": policy.state_dict(), "cfg": vars(cfg),
                            "step": step, "val_reward": vr, "seed": SEED},
                           os.path.join(OUT, f"grpo_best{suff}.pt"))
        if (step + 1) % CKPT_EVERY == 0:
            torch.save({"model": policy.state_dict(), "opt": opt.state_dict(),
                        "cfg": vars(cfg), "step": step, "seed": SEED},
                       os.path.join(OUT, f"grpo_ckpt{suff}.pt"))

    vr, ve, vk = eval_val(policy, tok, val, device, eot)
    torch.save({"model": policy.state_dict(), "cfg": vars(cfg),
                "step": steps_total, "val_reward": vr, "seed": SEED},
               os.path.join(OUT, f"grpo_final{suff}.pt"))
    log_f.write(json.dumps({"step": steps_total, "val_reward": vr,
                            "val_exact": ve, "by_kind": vk, "final": True}) + "\n")
    log_f.close()
    print(f"DONE. warm-start {r0:.3f} -> GRPO {vr:.3f} val reward "
          f"(best {best:.3f}). Saved grpo_final{suff}.pt")


if __name__ == "__main__":
    main()
