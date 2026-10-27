# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 7: phenotype-first RLVR — Baldwinian GRPO.

Identical to stage 5b's GRPO except for one thing: the PLASTICITY CHANNEL.
Each training prompt carries k worked examples of NEIGHBOR combos (same
problem kind, different parameters, drawn from the TRAIN pool only), and k is
annealed 5 -> 0 over training (genetic assimilation pressure). The verifier,
group advantage, KL leash, oversampling and eval protocol are unchanged, so
the stage-5b run is literally the k==0 control arm.

Arms (--arm):
  anneal  k = max(0, 5 - step//50) over 300 steps  (the Baldwin arm)
  k5      k fixed at 5                              (crutch-never-removed control)

Headline metric: held-out val reward at k=0 (same 727 problems, same greedy
eval as stage 5b). Side metric: context sensitivity = reward(k=5) - reward(k=0)
on a fixed 160-problem val subset (scaffolds drawn from train, so no val
answer ever appears in any prompt). Seed 42. --smoke = 30 steps, fast anneal.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))
sys.path.insert(0, os.path.join(HERE, "..", "02_transformer"))
sys.path.insert(0, os.path.join(HERE, "..", "04_sft"))
sys.path.insert(0, os.path.join(HERE, "..", "05_dpo_rlvr"))
from bpe import BPE                       # noqa: E402
from model import GPT, GPTConfig          # noqa: E402
from sft import PROMPT_FMT                # noqa: E402
import grpo                               # noqa: E402  (eval_val, load_problems)
from bioelectric import reward as score   # noqa: E402

SEED = 42
OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
WS = os.path.join(HERE, "..", "05_dpo_rlvr", "out", "ws_final.pt")

P_PROMPTS, K = 16, 8
MAX_NEW = 24
MAX_SEQ = 512
STEPS = 300
LR, WARMUP = 2e-6, 20
KL_COEF = 0.05
EVAL_EVERY = 50
SENS_N = 160


def combo_key(p):
    return (p["kind"], p["ion"], p["cin"], p["cout"], p["cout2"])


def build_scaffold(p, k, pool_by_kind, rng, tok):
    """Encode prompt for problem p with k worked neighbor examples (train pool).
    Drops examples if the sequence would not fit MAX_SEQ - MAX_NEW."""
    tail = PROMPT_FMT.format(p["prompt"])
    if k == 0:
        return tok.encode(tail)
    pool = pool_by_kind[p["kind"]]
    me = combo_key(p)
    exs, seen = [], {me}
    for i in rng.permutation(len(pool)):
        e = pool[i]
        ck = combo_key(e)
        if ck not in seen:
            seen.add(ck)
            exs.append(e)
            if len(exs) == k:
                break
    while True:
        parts = [PROMPT_FMT.format(e["prompt"]) + e["answer"] for e in exs]
        ids = tok.encode("\n\n".join(parts + [tail]))
        if len(ids) <= MAX_SEQ - MAX_NEW or not exs:
            return ids
        exs = exs[:-1]


@torch.no_grad()
def generate_from_ids(model, enc, device, eot, temperature):
    """enc: list of token lists. Returns (prompt_ids, answer_ids) per item."""
    buckets = defaultdict(list)
    for i, ids in enumerate(enc):
        buckets[len(ids)].append(i)
    outs = [None] * len(enc)
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
def eval_sens(model, tok, sub, enc_k5, enc_k0, device, eot):
    """Context sensitivity on the fixed val subset: reward at k=5 vs k=0."""
    res = {}
    for name, enc in [("k5", enc_k5), ("k0", enc_k0)]:
        outs = generate_from_ids(model, enc, device, eot, temperature=0.0)
        rs = []
        for p, (_, ans) in zip(sub, outs):
            text = tok.decode(ans[:-1] if ans and ans[-1] == eot else ans)
            rs.append(score(p["kind"], p["truth"], text))
        res[name] = float(np.mean(rs))
    res["gap"] = res["k5"] - res["k0"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["anneal", "k5", "k0"], default="anneal")
    ap.add_argument("--ws", default=os.path.join(OUT, "ws2_final.pt"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    steps_total = 30 if args.smoke else STEPS
    suff = f"_{args.arm}" + ("_smoke" if args.smoke else "")

    def k_of(step):
        if args.arm == "k5":
            return 5
        if args.arm == "k0":
            return 0
        window = 5 if args.smoke else 50
        return max(0, 5 - step // window)

    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)   # leave ~3 GB VRAM for the desktop
    os.makedirs(OUT, exist_ok=True)
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    train = grpo.load_problems("rlvr_train.jsonl")
    val = grpo.load_problems("rlvr_val.jsonl")
    rng = np.random.default_rng(SEED)
    tr_nernst = [p for p in train if p["kind"] == "nernst"]
    tr_other = [p for p in train if p["kind"] != "nernst"]
    pool_by_kind = defaultdict(list)
    for p in train:
        pool_by_kind[p["kind"]].append(p)

    # fixed context-sensitivity subset + its scaffolds (deterministic)
    srng = np.random.default_rng(123)
    sub = [val[i] for i in srng.choice(len(val), SENS_N, replace=False)]
    enc_k5 = [build_scaffold(p, 5, pool_by_kind, srng, tok) for p in sub]
    enc_k0 = [build_scaffold(p, 0, pool_by_kind, srng, tok) for p in sub]
    print(f"{len(train)} train, {len(val)} val, sens subset {SENS_N} "
          f"(arm={args.arm}, scaffold lens k5: max "
          f"{max(len(e) for e in enc_k5)} tokens)")

    ck = torch.load(args.ws, weights_only=False)
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

    r0, e0, kk0 = grpo.eval_val(policy, tok, val, device, eot)
    s0 = eval_sens(policy, tok, sub, enc_k5, enc_k0, device, eot)
    print(f"WARM-START baseline: val(k=0) {r0:.3f} {kk0} | sens {s0}")
    log_f = open(os.path.join(OUT, f"bg_log{suff}.jsonl"), "a", encoding="utf-8")
    log_f.write(json.dumps({"step": -1, "val_reward": r0, "val_exact": e0,
                            "by_kind": kk0, "sens": s0, "baseline": True,
                            "arm": args.arm, "ws": os.path.basename(args.ws),
                            "seed": SEED}) + "\n")

    best = r0
    t0 = time.time()
    for step in range(steps_total):
        for g in opt.param_groups:
            g["lr"] = LR * (step + 1) / WARMUP if step < WARMUP else LR
        kk = k_of(step)

        n_num = int(P_PROMPTS * 0.6)
        picks = [tr_nernst[i] for i in rng.integers(0, len(tr_nernst), n_num)] + \
                [tr_other[i] for i in rng.integers(0, len(tr_other), P_PROMPTS - n_num)]
        # one scaffold per pick, shared by its K samples (groups stay comparable)
        enc_p = [build_scaffold(p, kk, pool_by_kind, rng, tok) for p in picks]
        probs = [p for p in picks for _ in range(K)]
        enc = [e for e in enc_p for _ in range(K)]
        policy.eval()
        outs = generate_from_ids(policy, enc, device, eot, temperature=1.0)
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
        torch.cuda.empty_cache()                       # hand freed VRAM back to Windows

        if step % 5 == 0 or step == steps_total - 1:
            dt = (time.time() - t0) / (step + 1)
            print(f"step {step:4d}/{steps_total}  k={kk}  R {rs.mean():.3f}  "
                  f"exact {np.mean(rs == 1.0):.3f}  kl {kl.item():+.4f}  "
                  f"kept {B}/{len(probs)}  {dt:.2f} s/step", flush=True)
            log_f.write(json.dumps({"step": step, "k": kk, "reward": rs.mean(),
                                    "exact": float(np.mean(rs == 1.0)),
                                    "kl": kl.item(), "kept": B}) + "\n")
            log_f.flush()

        if (step + 1) % EVAL_EVERY == 0 or step == steps_total - 1:
            vr, ve, vk = grpo.eval_val(policy, tok, val, device, eot)
            sens = eval_sens(policy, tok, sub, enc_k5, enc_k0, device, eot)
            print(f"  VAL(k=0) {vr:.3f}  {vk}  sens {sens}", flush=True)
            log_f.write(json.dumps({"step": step, "val_reward": vr,
                                    "val_exact": ve, "by_kind": vk,
                                    "sens": sens}) + "\n")
            log_f.flush()
            if vr > best:
                best = vr
                torch.save({"model": policy.state_dict(), "cfg": vars(cfg),
                            "step": step, "val_reward": vr, "seed": SEED,
                            "arm": args.arm},
                           os.path.join(OUT, f"bg_best{suff}.pt"))

    vr, ve, vk = grpo.eval_val(policy, tok, val, device, eot)
    sens = eval_sens(policy, tok, sub, enc_k5, enc_k0, device, eot)
    torch.save({"model": policy.state_dict(), "cfg": vars(cfg),
                "step": steps_total, "val_reward": vr, "seed": SEED,
                "arm": args.arm},
               os.path.join(OUT, f"bg_final{suff}.pt"))
    log_f.write(json.dumps({"step": steps_total, "val_reward": vr,
                            "val_exact": ve, "by_kind": vk, "sens": sens,
                            "final": True}) + "\n")
    log_f.close()
    print(f"DONE arm={args.arm}. ws {r0:.3f} -> {vr:.3f} val(k=0) "
          f"(best {best:.3f}); final sens {sens}; by_kind {vk}")


if __name__ == "__main__":
    main()
