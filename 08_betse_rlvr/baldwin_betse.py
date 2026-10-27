# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8c: Baldwinian GRPO on the REAL-simulator problems.

The Baldwin analogy, kept in spirit rather than letter:
  plasticity   = k worked NEAREST-NEIGHBOR anchors in the prompt — real local
                 experience (nearby combos with their sim-measured Vmem), not
                 random examples: on a smooth response surface these are
                 genuinely informative, so the within-life channel can help
  selection    = GRPO reward on the post-plasticity phenotype (unchanged)
  assimilation = anneal k 5 -> 0 over training; the organism is pushed from
                 "interpolate from experience in front of you" to "innate"

Lesson from stage 7 carried over: plasticity must be TAUGHT first — the shared
warm-start (bw8_ws.pt) is SFT with mixed k ~ U{0..5} anchor formatting.

Arms (--arm): k0 (direct control) | anneal (Baldwin) | k5 (crutch control).
Headline: held-out-combo val at k=0 (same eval as betse_grpo.py, so the
stage-8b run 0.945/0.912 is the single-format reference). Side metric:
sens gap = reward(k=5) - reward(k=0) on the 98 val problems (anchors always
drawn from TRAIN combos — no val truth ever enters a prompt). Seed 42.
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
WS_PT = os.path.join(OUT, "bw8_ws.pt")

WS_STEPS, WS_BATCH, WS_LR = 400, 32, 3e-5
P_PROMPTS, K, MAX_NEW, MAX_SEQ = 16, 8, 16, 512
RL_STEPS, RL_LR, WARMUP, KL_COEF = 300, 2e-6, 20, 0.05  # 300 = 50 steps at k=0 after the anneal (v1 ran 200 and never trained at k=0)
EVAL_EVERY = 50


def load_problems(name):
    return [json.loads(l) for l in open(os.path.join(OUT, name), encoding="utf-8")]


def dist(c1, c2):
    return sum((math.log2(a) - math.log2(b)) ** 2 for a, b in zip(c1, c2))


def neighbors(p, pool):
    """Train-pool problems of the same kind, nearest first in log-knob space."""
    me = tuple(p["combo"])
    cand = [q for q in pool[p["kind"]] if tuple(q["combo"]) != me]
    return sorted(cand, key=lambda q: dist(q["combo"], me))


def build_prompt_ids(p, k, pool, tok, rng=None):
    """k nearest-neighbor worked examples + target. rng: sample k of nearest 8."""
    tail = PROMPT_FMT.format(p["prompt"])
    if k == 0:
        return tok.encode(tail)
    near = neighbors(p, pool)
    if rng is not None:
        pick = rng.permutation(min(8, len(near)))[:k]
        exs = [near[i] for i in sorted(pick)]
    else:
        exs = near[:k]
    while True:
        parts = [PROMPT_FMT.format(e["prompt"]) + e["answer"] for e in exs]
        ids = tok.encode("\n\n".join(parts + [tail]))
        if len(ids) <= MAX_SEQ - MAX_NEW or not exs:
            return ids
        exs = exs[:-1]


@torch.no_grad()
def generate_from_ids(model, enc, device, eot, temperature):
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
def eval_set(model, tok, probs, enc, device, eot):
    outs = generate_from_ids(model, enc, device, eot, temperature=0.0)
    rs, by_kind = [], defaultdict(list)
    for p, (_, ans) in zip(probs, outs):
        text = tok.decode(ans[:-1] if ans and ans[-1] == eot else ans)
        r = score(p["kind"], p["truth"], text)
        rs.append(r)
        by_kind[p["kind"]].append(r)
    return float(np.mean(rs)), {k: float(np.mean(v)) for k, v in by_kind.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["k0", "anneal", "k5"], default="anneal")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", default="betse_train.jsonl")
    ap.add_argument("--val", default="betse_val.jsonl")
    ap.add_argument("--wspt", default="bw8_ws.pt")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rl_steps = 20 if args.smoke else RL_STEPS
    suff = f"_{args.arm}" + (f"_{args.tag}" if args.tag else "") + \
        ("_smoke" if args.smoke else "")
    ws_pt = os.path.join(OUT, args.wspt)

    def k_of(step):
        if args.arm == "k5":
            return 5
        if args.arm == "k0":
            return 0
        window = 4 if args.smoke else 50
        return max(0, 5 - step // window)

    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: <=30GB of 32GB
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    train = load_problems(args.train)
    val = load_problems(args.val)
    pool = defaultdict(list)
    for p in train:
        pool[p["kind"]].append(p)
    tr_num = [p for p in train if p["kind"] == "nernst"]
    tr_cat = [p for p in train if p["kind"] != "nernst"]
    rng = np.random.default_rng(SEED)

    # ---- shared warm-start: mixed-k anchor format (train once, reuse) ----
    if not os.path.exists(ws_pt):
        print(f"training shared warm-start {args.wspt} (mixed k 0..5) ...")
        init_ck = torch.load(INIT, weights_only=False)
        policy = GPT(GPTConfig(**init_ck["cfg"])).to(device)
        policy.load_state_dict(init_ck["model"])
        xs, ys = [], []
        for p in train:
            kk = int(rng.integers(0, 6))
            pr = build_prompt_ids(p, kk, pool, tok, rng=rng)
            ids = (pr + tok.encode(p["answer"]) + [eot])[:MAX_SEQ]
            if len(ids) > len(pr):
                xs.append(ids)
                ys.append([-100] * (len(pr) - 1) + ids[len(pr):])
        opt = torch.optim.AdamW(policy.parameters(), lr=WS_LR, betas=(0.9, 0.95),
                                weight_decay=0.0, fused=True)
        order = rng.permutation(len(xs))
        policy.train()
        for step in range(WS_STEPS):
            for g in opt.param_groups:
                g["lr"] = WS_LR * 0.5 * (1 + math.cos(math.pi * step / WS_STEPS))
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
            if step % 100 == 0 or step == WS_STEPS - 1:
                print(f"ws {step:3d}/{WS_STEPS} loss {loss.item():.4f}", flush=True)
        policy.eval()
        torch.save({"model": policy.state_dict(), "cfg": init_ck["cfg"],
                    "seed": SEED, "format": "mixed-k nearest-neighbor anchors"}, ws_pt)

    ck = torch.load(ws_pt, weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    policy = GPT(cfg).to(device)
    policy.load_state_dict(ck["model"])
    policy.eval()
    ref = GPT(cfg).to(device)
    ref.load_state_dict(ck["model"])
    ref.eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=RL_LR, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=True)

    enc_v0 = [build_prompt_ids(p, 0, pool, tok) for p in val]
    enc_v5 = [build_prompt_ids(p, 5, pool, tok) for p in val]

    def sens(model):
        r0, k0_ = eval_set(model, tok, val, enc_v0, device, eot)
        r5, _ = eval_set(model, tok, val, enc_v5, device, eot)
        return r0, k0_, {"k5": r5, "k0": r0, "gap": r5 - r0}

    r0, by0, s0 = sens(policy)
    print(f"ws baseline: val(k=0) {r0:.3f} {by0} | sens {s0}")
    log_f = open(os.path.join(OUT, f"bw8_log{suff}.jsonl"), "a", encoding="utf-8")
    log_f.write(json.dumps({"step": -1, "val_reward": r0, "by_kind": by0,
                            "sens": s0, "baseline": True, "arm": args.arm,
                            "seed": SEED}) + "\n")

    best = r0
    t0 = time.time()
    for step in range(rl_steps):
        for g in opt.param_groups:
            g["lr"] = RL_LR * (step + 1) / WARMUP if step < WARMUP else RL_LR
        kk = k_of(step)
        n_num = int(P_PROMPTS * 0.6)
        picks = [tr_num[i] for i in rng.integers(0, len(tr_num), n_num)] + \
                [tr_cat[i] for i in rng.integers(0, len(tr_cat), P_PROMPTS - n_num)]
        enc_p = [build_prompt_ids(p, kk, pool, tok, rng=rng) for p in picks]
        probs = [p for p in picks for _ in range(K)]
        enc = [e for e in enc_p for _ in range(K)]
        policy.eval()
        outs = generate_from_ids(policy, enc, device, eot, temperature=1.0)
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
        (pg + KL_COEF * kl).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 5 == 0 or step == rl_steps - 1:
            print(f"rl {step:3d}/{rl_steps}  k={kk}  R {rs.mean():.3f}  "
                  f"kl {kl.item():+.4f}  kept {B}/128  "
                  f"{(time.time() - t0) / (step + 1):.2f} s/step", flush=True)
            log_f.write(json.dumps({"step": step, "k": kk, "reward": rs.mean(),
                                    "kl": kl.item(), "kept": B}) + "\n")
            log_f.flush()
        if (step + 1) % EVAL_EVERY == 0 or step == rl_steps - 1:
            vr, byk, ss = sens(policy)
            print(f"  VAL(k=0) {vr:.3f} {byk} sens {ss}", flush=True)
            log_f.write(json.dumps({"step": step, "val_reward": vr,
                                    "by_kind": byk, "sens": ss}) + "\n")
            log_f.flush()
            if vr > best:
                best = vr
                torch.save({"model": policy.state_dict(), "cfg": vars(cfg),
                            "step": step, "val_reward": vr, "seed": SEED,
                            "arm": args.arm}, os.path.join(OUT, f"bw8_best{suff}.pt"))
    vr, byk, ss = sens(policy)
    torch.save({"model": policy.state_dict(), "cfg": vars(cfg), "seed": SEED,
                "arm": args.arm, "val_reward": vr},
               os.path.join(OUT, f"bw8_final{suff}.pt"))
    log_f.write(json.dumps({"step": rl_steps, "val_reward": vr, "by_kind": byk,
                            "sens": ss, "final": True}) + "\n")
    log_f.close()
    print(f"DONE arm={args.arm}: ws {r0:.3f} -> {vr:.3f} (best {best:.3f}) {byk} sens {ss}")


if __name__ == "__main__":
    main()
