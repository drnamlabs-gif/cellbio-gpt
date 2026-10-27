# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 3: pretrain the 40M-param GPT on the full cell-biology corpus.

Recipe (the standard rituals, each one load-bearing — see README/explainer):
  bf16 autocast · fused SDPA attention · AdamW with weight decay on matrices
  only · LR warmup 500 steps then cosine decay · grad clip 1.0 · seed 42

Persistence (crash-safe): checkpoint every 1000 steps (model+optimizer+rng),
`--resume` continues from the last one; metrics appended to out/log.jsonl;
val loss every 1000 steps; sample generation every 5000 steps to out/samples.txt;
final: out/final.pt, out/loss_curve.png.

Usage:
  python pretrain.py --smoke     # 300 steps, measures step time -> ETA
  python pretrain.py             # full run, ~1 epoch over ~0.8B tokens
  python pretrain.py --resume    # continue after a crash/stop
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
TOKENS = os.path.join(HERE, "..", "data", "tokens")
OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")

# ---- config (40M params, Chinchilla-matched to the ~0.8B-token corpus) ----
N_LAYER, N_HEAD, D_MODEL, SEQ = 10, 8, 512, 512
BATCH = 64
LR, MIN_LR, WARMUP, WEIGHT_DECAY = 6e-4, 6e-5, 500, 0.1
CKPT_EVERY = EVAL_EVERY = 1000
SAMPLE_EVERY = 5000
EVAL_BATCHES = 50


class Data:
    def __init__(self, split, rng):
        self.arr = np.memmap(os.path.join(TOKENS, f"{split}.bin"),
                             dtype=np.uint16, mode="r")
        self.rng = rng

    def batch(self, device):
        ix = self.rng.integers(0, len(self.arr) - SEQ - 1, BATCH)
        x = torch.from_numpy(np.stack([self.arr[i:i + SEQ] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([self.arr[i + 1:i + SEQ + 1] for i in ix]).astype(np.int64))
        return x.pin_memory().to(device, non_blocking=True), \
               y.pin_memory().to(device, non_blocking=True)


def lr_at(step, total):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, total - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def eval_val(model, val, device):
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = val.batch(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model, tok, device, prompt="Membrane potential"):
    ids = torch.tensor([tok.encode(prompt)], device=device)
    out = model.generate(ids, 80, temperature=0.8, top_k=50)
    model.train()
    return tok.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    tok = BPE.load(TOKENIZER)
    train = Data("train", np.random.default_rng(SEED))
    val = Data("val", np.random.default_rng(SEED + 1))
    n_tok = len(train.arr)
    total_steps = 300 if args.smoke else n_tok // (BATCH * SEQ)   # ~1 epoch
    print(f"train tokens {n_tok/1e6:.0f}M, steps {total_steps}, "
          f"tokens/step {BATCH*SEQ}")

    cfg = GPTConfig(vocab_size=tok.vocab_size, n_layer=N_LAYER, n_head=N_HEAD,
                    d_model=D_MODEL, max_seq=SEQ, fused=True)
    model = GPT(cfg).to(device)
    print(f"params {model.num_params()/1e6:.1f}M")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.95), fused=True)

    start_step, best_val = 0, float("inf")
    ckpt_path = os.path.join(OUT, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step, best_val = ck["step"] + 1, ck["best_val"]
        torch.set_rng_state(ck["rng"])
        train.rng = np.random.default_rng(ck["np_rng"])
        print(f"resumed at step {start_step}")

    log_path = os.path.join(OUT, "log.jsonl")
    log_f = open(log_path, "a", encoding="utf-8")
    t0 = time.time()
    model.train()
    for step in range(start_step, total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps)
        x, y = train.batch(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0 or step == total_steps - 1:
            dt = (time.time() - t0) / max(1, step - start_step + 1)
            eta_h = dt * (total_steps - step) / 3600
            print(f"step {step:6d}/{total_steps}  loss {loss.item():.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{dt*1000:.0f} ms/step  ETA {eta_h:.1f}h", flush=True)
            log_f.write(json.dumps({"step": step, "loss": loss.item(),
                                    "lr": opt.param_groups[0]["lr"],
                                    "ms_step": dt * 1000}) + "\n")
            log_f.flush()

        if step > 0 and step % EVAL_EVERY == 0:
            vl = eval_val(model, val, device)
            print(f"  val loss {vl:.4f}", flush=True)
            log_f.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
            log_f.flush()
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                            "step": step, "val_loss": vl, "seed": SEED},
                           os.path.join(OUT, "best.pt"))

        if step > 0 and step % CKPT_EVERY == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "cfg": vars(cfg), "step": step, "best_val": best_val,
                        "seed": SEED, "rng": torch.get_rng_state(),
                        "np_rng": train.rng.bit_generator.state},
                       ckpt_path)

        if step > 0 and step % SAMPLE_EVERY == 0:
            s = sample(model, tok, device)
            with open(os.path.join(OUT, "samples.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n--- step {step} ---\n{s}\n")

    vl = eval_val(model, val, device)
    torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                "step": total_steps, "val_loss": vl, "seed": SEED},
               os.path.join(OUT, "final.pt"))
    s = sample(model, tok, device)
    with open(os.path.join(OUT, "samples.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n--- FINAL (step {total_steps}, val {vl:.4f}) ---\n{s}\n")
    log_f.write(json.dumps({"step": total_steps, "val_loss": vl, "final": True}) + "\n")
    log_f.close()
    print(f"DONE. final val loss {vl:.4f}. Saved final.pt / best.pt / log.jsonl")


if __name__ == "__main__":
    main()
