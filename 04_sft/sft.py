# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 4b: supervised fine-tuning of the pretrained 40M model.

The ONLY changes vs stage-3 pretraining (see out/explain_sft.png):
  1. data = (instruction, answer) pairs in a fixed plain-text chat template
  2. loss mask: prompt+padding targets set to -100 (F.cross_entropy default
     ignore_index), so gradient flows only through answer tokens + <|eot|>
Everything else identical: same GPT, AdamW (decay on matrices only), bf16,
warmup+cosine, grad clip 1.0, seed 42. Init from 03_pretrain/out/final.pt,
LR 10x lower than pretraining.

Template (eot = token 16383, reused as end-of-answer):
  ### Instruction:\n{prompt}\n\n### Response:\n{answer}<eot>

Persistence: out/sft_log.jsonl, sft_best.pt (val), sft_final.pt,
sft_samples.txt. --smoke = 100 steps on the smoke dataset.
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))
sys.path.insert(0, os.path.join(HERE, "..", "02_transformer"))
from bpe import BPE                      # noqa: E402
from model import GPT, GPTConfig        # noqa: E402

SEED = 42
OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
PRETRAINED = os.path.join(HERE, "..", "03_pretrain", "out", "final.pt")

MAX_LEN = 512
BATCH = 64
EPOCHS = 2
LR, MIN_LR, WARMUP, WEIGHT_DECAY = 6e-5, 6e-6, 100, 0.1
EVAL_EVERY, SAMPLE_EVERY = 200, 1000

PROMPT_FMT = "### Instruction:\n{}\n\n### Response:\n"

EVAL_PROMPTS = [
    "What does membrane depolarization trigger?",
    "What is a gap junction?",
    "In cell biology, what does ATP stand for?",
    "Write a concise title for this abstract:\nPlanarians regenerate "
    "complete heads after amputation. We show that blocking gap junction "
    "communication with octanol alters the bioelectric pattern of the "
    "blastema and produces two-headed animals, indicating that membrane "
    "voltage gradients instruct anterior-posterior polarity during "
    "regeneration.",
]


def encode_examples(path, tok):
    """Tokenize to (x, y) pairs; y = -100 on prompt tokens, answer ids + eot after."""
    eot = tok.vocab_size - 1
    xs, ys = [], []
    n_skip = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            p = tok.encode(PROMPT_FMT.format(ex["prompt"]))
            a = tok.encode(ex["answer"]) + [eot]
            ids = p + a
            if len(ids) > MAX_LEN:
                n_skip += 1
                continue
            x = ids[:-1]
            y = [-100] * (len(p) - 1) + a
            assert len(x) == len(y)
            xs.append(np.array(x, dtype=np.int64))
            ys.append(np.array(y, dtype=np.int64))
    return xs, ys, n_skip


class Batches:
    """Shuffled padded batches; pad x with eot, pad y with -100."""

    def __init__(self, xs, ys, rng, eot):
        self.xs, self.ys, self.rng, self.eot = xs, ys, rng, eot

    def epoch(self, device):
        order = self.rng.permutation(len(self.xs))
        for i in range(0, len(order) - BATCH + 1, BATCH):
            idx = order[i:i + BATCH]
            L = max(len(self.xs[j]) for j in idx)
            x = np.full((BATCH, L), self.eot, dtype=np.int64)
            y = np.full((BATCH, L), -100, dtype=np.int64)
            for r, j in enumerate(idx):
                x[r, :len(self.xs[j])] = self.xs[j]
                y[r, :len(self.ys[j])] = self.ys[j]
            yield (torch.from_numpy(x).pin_memory().to(device, non_blocking=True),
                   torch.from_numpy(y).pin_memory().to(device, non_blocking=True))

    def steps_per_epoch(self):
        return len(self.xs) // BATCH


def lr_at(step, total):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, total - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def eval_val(model, vb, device):
    model.eval()
    losses = []
    for x, y in vb.epoch(device):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def gen_samples(model, tok, device, eot):
    outs = []
    model.eval()
    for p in EVAL_PROMPTS:
        ids = torch.tensor([tok.encode(PROMPT_FMT.format(p))], device=device)
        out = model.generate(ids, 90, temperature=0.0)[0].tolist()
        ans = out[ids.shape[1]:]
        ans = ans[:ans.index(eot)] if eot in ans else ans
        outs.append((p, tok.decode(ans).strip()))
    model.train()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    suff = "_smoke" if args.smoke else ""

    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    xs, ys, nsk = encode_examples(os.path.join(OUT, f"sft_train{suff}.jsonl"), tok)
    vxs, vys, _ = encode_examples(os.path.join(OUT, f"sft_val{suff}.jsonl"), tok)
    print(f"train {len(xs)} examples ({nsk} skipped >512 tok), val {len(vxs)}")

    tb = Batches(xs, ys, np.random.default_rng(SEED), eot)
    vb = Batches(vxs, vys, np.random.default_rng(SEED + 1), eot)
    total_steps = 100 if args.smoke else tb.steps_per_epoch() * EPOCHS
    print(f"steps {total_steps} ({EPOCHS} epochs)")

    ck = torch.load(PRETRAINED, weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    print(f"loaded pretrained (step {ck['step']}, val {ck['val_loss']:.4f}), "
          f"params {model.num_params()/1e6:.1f}M")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.95), fused=True)

    log_f = open(os.path.join(OUT, f"sft_log{suff}.jsonl"), "a", encoding="utf-8")
    best_val, step, t0 = float("inf"), 0, time.time()
    model.train()
    done = False
    while not done:
        for x, y in tb.epoch(device):
            if step >= total_steps:
                done = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 25 == 0 or step == total_steps - 1:
                dt = (time.time() - t0) / (step + 1)
                print(f"step {step:5d}/{total_steps}  loss {loss.item():.3f}  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  {dt*1000:.0f} ms/step",
                      flush=True)
                log_f.write(json.dumps({"step": step, "loss": loss.item(),
                                        "lr": opt.param_groups[0]["lr"]}) + "\n")
                log_f.flush()
            if step > 0 and step % EVAL_EVERY == 0:
                vl = eval_val(model, vb, device)
                print(f"  val loss {vl:.4f}", flush=True)
                log_f.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
                log_f.flush()
                if vl < best_val:
                    best_val = vl
                    torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                                "step": step, "val_loss": vl, "seed": SEED},
                               os.path.join(OUT, f"sft_best{suff}.pt"))
            if step > 0 and step % SAMPLE_EVERY == 0:
                with open(os.path.join(OUT, f"sft_samples{suff}.txt"), "a",
                          encoding="utf-8") as f:
                    f.write(f"\n=== step {step} ===\n")
                    for p, a in gen_samples(model, tok, device, eot):
                        f.write(f"Q: {p[:90]}\nA: {a}\n\n")
            step += 1

    vl = eval_val(model, vb, device)
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step,
                "val_loss": vl, "seed": SEED},
               os.path.join(OUT, f"sft_final{suff}.pt"))
    with open(os.path.join(OUT, f"sft_samples{suff}.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n=== FINAL (step {step}, val {vl:.4f}) ===\n")
        for p, a in gen_samples(model, tok, device, eot):
            f.write(f"Q: {p[:90]}\nA: {a}\n\n")
    log_f.write(json.dumps({"step": step, "val_loss": vl, "final": True}) + "\n")
    log_f.close()
    print(f"DONE. final val loss {vl:.4f} (best {best_val:.4f}). "
          f"Saved sft_final{suff}.pt / sft_best{suff}.pt")


if __name__ == "__main__":
    main()
