# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 5a: DPO from scratch (see out/explain_dpo.png).

Loss, derived in the figure and implemented here literally:
  margin = beta * [ (logpi_theta(y+|x) - logpi_ref(y+|x))
                  - (logpi_theta(y-|x) - logpi_ref(y-|x)) ]
  L = -log sigmoid(margin)
Sequence log-probs are summed over ANSWER tokens only (prompt masked -100,
same trick as SFT). pi_ref = frozen copy of the SFT model.

Policy init = SFT model; chosen/rejected forwarded in ONE batch (policy and
ref each run once per step). Metrics: loss, preference accuracy (margin>0),
margin mean. Val every 100 steps -> dpo_best.pt; final -> dpo_final.pt.
Seed 42. --smoke = 60 steps on the smoke pairs.
"""

import argparse
import json
import math
import os
import sys
import time

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

SEED = 42
OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
SFT_MODEL = os.path.join(HERE, "..", "04_sft", "out", "sft_final.pt")

BETA = 0.1
PAIRS_PER_STEP = 16            # 16 pairs = 32 sequences per forward
EPOCHS = 1
LR, MIN_LR, WARMUP = 5e-6, 5e-7, 50
MAX_LEN = 512
EVAL_EVERY = 100


def encode_pairs(path, tok, tasks=None):
    eot = tok.vocab_size - 1
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if tasks and ex["task"] not in tasks:
                continue
            p = tok.encode(PROMPT_FMT.format(ex["prompt"]))
            c = tok.encode(ex["answer"]) + [eot]
            r = tok.encode(ex["rejected"]) + [eot]
            if len(p) + max(len(c), len(r)) > MAX_LEN:
                continue
            out.append((p, c, r))
    return out


def seq_logprob(model, x, y, autocast=True):
    """Sum of log P(target) over positions where y != -100. x,y: (B,T)."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        logits, _, _ = model(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = y != -100
    safe_y = y.clamp(min=0)
    tok_lp = torch.gather(logp, 2, safe_y.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask).sum(dim=1)


class PairBatches:
    def __init__(self, pairs, rng, eot):
        self.pairs, self.rng, self.eot = pairs, rng, eot

    def epoch(self, device):
        order = self.rng.permutation(len(self.pairs))
        B = PAIRS_PER_STEP
        for i in range(0, len(order) - B + 1, B):
            chunk = [self.pairs[j] for j in order[i:i + B]]
            seqs = []
            for p, c, r in chunk:
                seqs.append((p + c, [-100] * (len(p) - 1) + c))
            for p, c, r in chunk:
                seqs.append((p + r, [-100] * (len(p) - 1) + r))
            L = max(len(s) for s, _ in seqs)
            x = np.full((2 * B, L - 0), self.eot, dtype=np.int64)
            y = np.full((2 * B, L - 0), -100, dtype=np.int64)
            for row, (ids, tgt) in enumerate(seqs):
                x[row, :len(ids) - 1] = ids[:-1]
                y[row, :len(tgt)] = tgt
            yield (torch.from_numpy(x[:, :L - 1]).to(device),
                   torch.from_numpy(y[:, :L - 1]).to(device))

    def steps_per_epoch(self):
        return len(self.pairs) // PAIRS_PER_STEP


def lr_at(step, total):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, total - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(t, 1.0)))


def dpo_stats(policy, ref, x, y):
    B2 = x.shape[0]
    B = B2 // 2
    lp_pol = seq_logprob(policy, x, y)
    with torch.no_grad():
        lp_ref = seq_logprob(ref, x, y)
    margin = BETA * ((lp_pol[:B] - lp_ref[:B]) - (lp_pol[B:] - lp_ref[B:]))
    loss = -F.logsigmoid(margin).mean()
    acc = (margin > 0).float().mean().item()
    return loss, acc, margin.mean().item()


@torch.no_grad()
def eval_val(policy, ref, vb, device):
    policy.eval()
    L, A, M, n = 0.0, 0.0, 0.0, 0
    for x, y in vb.epoch(device):
        loss, acc, m = dpo_stats(policy, ref, x, y)
        L += loss.item(); A += acc; M += m; n += 1
    policy.train()
    return L / max(1, n), A / max(1, n), M / max(1, n)


def main():
    global BETA, LR
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--tasks", type=str, default="",
                    help="comma list, e.g. defn,abbrev (default: all)")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for output files, e.g. _v2 (never overwrite v1)")
    args = ap.parse_args()
    BETA, LR = args.beta, args.lr
    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    suff = ("_smoke" if args.smoke else "") + args.tag
    dsuff = "_smoke" if args.smoke else ""
    task_filter = set(args.tasks.split(",")) if args.tasks else None

    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    pairs = encode_pairs(os.path.join(OUT, f"pref_train{dsuff}.jsonl"), tok, task_filter)
    vpairs = encode_pairs(os.path.join(OUT, f"pref_val{dsuff}.jsonl"), tok, task_filter)
    print(f"train {len(pairs)} pairs, val {len(vpairs)} "
          f"(tasks={args.tasks or 'all'}, beta={BETA}, lr={LR})")

    ck = torch.load(SFT_MODEL, weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    policy = GPT(cfg).to(device)
    policy.load_state_dict(ck["model"])
    ref = GPT(cfg).to(device)
    ref.load_state_dict(ck["model"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    decay = [p for n, p in policy.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in policy.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.0},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.95), fused=True)

    tb = PairBatches(pairs, np.random.default_rng(SEED), eot)
    vb = PairBatches(vpairs, np.random.default_rng(SEED + 1), eot)
    total = 60 if args.smoke else tb.steps_per_epoch() * EPOCHS
    print(f"steps {total}, beta {BETA}, lr {LR}")

    log_f = open(os.path.join(OUT, f"dpo_log{suff}.jsonl"), "a", encoding="utf-8")
    best_acc, step, t0 = 0.0, 0, time.time()
    policy.train()
    done = False
    while not done:
        for x, y in tb.epoch(device):
            if step >= total:
                done = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total)
            loss, acc, m = dpo_stats(policy, ref, x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            if step % 10 == 0 or step == total - 1:
                dt = (time.time() - t0) / (step + 1)
                print(f"step {step:4d}/{total}  loss {loss.item():.4f}  "
                      f"acc {acc:.3f}  margin {m:+.3f}  {dt*1000:.0f} ms/step",
                      flush=True)
                log_f.write(json.dumps({"step": step, "loss": loss.item(),
                                        "acc": acc, "margin": m}) + "\n")
                log_f.flush()
            if step > 0 and step % EVAL_EVERY == 0:
                vl, va, vm = eval_val(policy, ref, vb, device)
                print(f"  val loss {vl:.4f}  val acc {va:.3f}  margin {vm:+.3f}",
                      flush=True)
                log_f.write(json.dumps({"step": step, "val_loss": vl,
                                        "val_acc": va, "val_margin": vm}) + "\n")
                log_f.flush()
                if va > best_acc:
                    best_acc = va
                    torch.save({"model": policy.state_dict(), "cfg": vars(cfg),
                                "step": step, "val_acc": va, "seed": SEED},
                               os.path.join(OUT, f"dpo_best{suff}.pt"))
            step += 1

    vl, va, vm = eval_val(policy, ref, vb, device)
    torch.save({"model": policy.state_dict(), "cfg": vars(cfg), "step": step,
                "val_acc": va, "val_loss": vl, "seed": SEED},
               os.path.join(OUT, f"dpo_final{suff}.pt"))
    log_f.write(json.dumps({"step": step, "val_loss": vl, "val_acc": va,
                            "val_margin": vm, "final": True}) + "\n")
    log_f.close()
    print(f"DONE. val acc {va:.3f} (best {best_acc:.3f}), margin {vm:+.3f}. "
          f"Saved dpo_final{suff}.pt / dpo_best{suff}.pt")


if __name__ == "__main__":
    main()
