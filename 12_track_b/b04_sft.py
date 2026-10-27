# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Track B, stage 4: supervised fine-tuning — the mirror of 04_sft/sft.py.

Mirrored exactly from Track A: the same files (04_sft/out/sft_train.jsonl,
sft_val.jsonl, 97.5k self-mined PubMed tasks), the same plain-text template
  ### Instruction:\n{prompt}\n\n### Response:\n{answer}<eot>
the same loss mask (prompt and padding = -100, gradient only on answer + eot),
batch 64, 2 epochs, AdamW (decay on matrices), bf16 autocast, warmup 100 +
cosine, clip 1.0, seed 42, LR = 1/10 of the pretraining LR, same 4 sample
prompts. Init = Track B stage 3 (out/cpt_<tag>/final).

Different by necessity: the model's own tokenizer (eot = <|endoftext|>), and
MAX_LEN 1024 instead of 512 because the general tokenizer spends more tokens
on the same text (skips are counted and logged). Batch 64 is run as micro 16
x accum 4 (same effective batch).

Outputs out/sft_<tag>/: log.jsonl, samples.txt, best/ (val), final/.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_DATA = os.path.join(HERE, "..", "04_sft", "out")
sys.path.insert(0, os.path.join(HERE, "..", "04_sft"))
SEED = 42
MAX_LEN = 1024
BATCH, MICRO = 64, 16
EPOCHS = 2
WARMUP, WEIGHT_DECAY = 100, 0.1
EVAL_EVERY, SAMPLE_EVERY = 200, 500
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
    "What is bioelectricity?",
    "What is cancer?",
]


def encode_examples(path, tok, eot):
    xs, ys, n_skip = [], [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            p = tok.encode(PROMPT_FMT.format(ex["prompt"]))
            a = tok.encode(ex["answer"]) + [eot]
            ids = p + a
            if len(ids) > MAX_LEN:
                n_skip += 1
                continue
            xs.append(np.array(ids[:-1], dtype=np.int64))
            ys.append(np.array([-100] * (len(p) - 1) + a, dtype=np.int64))
    return xs, ys, n_skip


def batches(xs, ys, rng, eot, size):
    order = rng.permutation(len(xs))
    for i in range(0, len(order) - size + 1, size):
        idx = order[i:i + size]
        L = max(len(xs[j]) for j in idx)
        x = np.full((size, L), eot, dtype=np.int64)
        y = np.full((size, L), -100, dtype=np.int64)
        for r, j in enumerate(idx):
            x[r, :len(xs[j])] = xs[j]
            y[r, :len(ys[j])] = ys[j]
        yield torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()


def loss_sum(model, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x).logits
    return torch.nn.functional.cross_entropy(
        logits.float().view(-1, logits.size(-1)), y.view(-1),
        ignore_index=-100, reduction="sum")


@torch.no_grad()
def eval_val(model, vx, vy, eot):
    model.eval()
    tot, cnt = 0.0, 0
    for x, y in batches(vx, vy, np.random.default_rng(SEED + 1), eot, MICRO):
        tot += loss_sum(model, x, y).item()
        cnt += (y != -100).sum().item()
    model.train()
    return tot / max(1, cnt)


@torch.no_grad()
def gen_samples(model, tok, eot):
    model.eval()
    outs = []
    for p in EVAL_PROMPTS:
        enc = tok(PROMPT_FMT.format(p), return_tensors="pt").to("cuda")
        out = model.generate(**enc, max_new_tokens=120, do_sample=False,
                             eos_token_id=eot, pad_token_id=eot)
        outs.append((p, tok.decode(out[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()))
    model.train()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="SmolLM2-135M")
    ap.add_argument("--lr", type=float, default=1e-5)   # CPT used 1e-4
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: <=30GB of 32GB
    init = os.path.join(HERE, "out", f"cpt_{args.tag}", "final")
    out = os.path.join(HERE, "out", f"sft_{args.tag}" + ("_smoke" if args.smoke else ""))
    os.makedirs(out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(init)
    eot = tok.eos_token_id
    xs, ys, nsk = encode_examples(os.path.join(SFT_DATA, "sft_train.jsonl"), tok, eot)
    vx, vy, vsk = encode_examples(os.path.join(SFT_DATA, "sft_val.jsonl"), tok, eot)
    if args.smoke:
        vx, vy = vx[:256], vy[:256]
    steps_ep = len(xs) // BATCH
    total = 60 if args.smoke else steps_ep * EPOCHS
    accum = BATCH // MICRO
    min_lr = args.lr / 10
    print(f"train {len(xs)} ({nsk} skipped >{MAX_LEN} tok), val {len(vx)} ({vsk} skipped) | "
          f"{total} steps, batch {BATCH} = {MICRO} x {accum}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(init, dtype=torch.float32).cuda()
    model.config.use_cache = False
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)

    log_f = open(os.path.join(out, "log.jsonl"), "a", encoding="utf-8")
    smp_path = os.path.join(out, "samples.txt")
    meta = {"init": init, "n_train": len(xs), "n_skip": nsk, "n_val": len(vx),
            "steps": total, "lr": args.lr, "max_len": MAX_LEN, "seed": SEED}
    log_f.write(json.dumps({"meta": meta}) + "\n")

    def write_samples(label):
        with open(smp_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {label} ===\n")
            for p, a in gen_samples(model, tok, eot):
                f.write(f"Q: {p[:90]}\nA: {a}\n\n")

    vl = eval_val(model, vx, vy, eot)
    print(f"step 0 (stage-3 model, no SFT): val {vl:.4f}", flush=True)
    log_f.write(json.dumps({"step": 0, "val_loss": vl}) + "\n")
    write_samples("step 0 (before SFT)")

    rng = np.random.default_rng(SEED)
    best, step, t0 = float("inf"), 0, time.time()
    model.train()
    while step < total:
        for x, y in batches(xs, ys, rng, eot, BATCH):
            if step >= total:
                break
            t = (step - WARMUP) / max(1, total - WARMUP)
            lr = args.lr * (step + 1) / WARMUP if step < WARMUP else \
                min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))
            for g in opt.param_groups:
                g["lr"] = lr
            n_tok = (y != -100).sum().item()
            tot = 0.0
            for k in range(accum):
                s = slice(k * MICRO, (k + 1) * MICRO)
                l = loss_sum(model, x[s], y[s]) / n_tok
                l.backward()
                tot += l.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 25 == 0 or step == total - 1:
                dt = (time.time() - t0) / (step + 1)
                print(f"step {step:5d}/{total}  loss {tot:.3f}  lr {lr:.2e}  "
                      f"{dt * 1000:.0f} ms/step  ETA {(total - step) * dt / 60:.0f} min",
                      flush=True)
                log_f.write(json.dumps({"step": step, "loss": tot, "lr": lr}) + "\n")
                log_f.flush()
            if step > 0 and step % EVAL_EVERY == 0:
                vl = eval_val(model, vx, vy, eot)
                print(f"  val loss {vl:.4f}", flush=True)
                log_f.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
                log_f.flush()
                if vl < best:
                    best = vl
                    model.save_pretrained(os.path.join(out, "best"))
                    tok.save_pretrained(os.path.join(out, "best"))
            if step > 0 and step % SAMPLE_EVERY == 0:
                write_samples(f"step {step}")
            step += 1

    vl = eval_val(model, vx, vy, eot)
    model.save_pretrained(os.path.join(out, "final"))
    tok.save_pretrained(os.path.join(out, "final"))
    write_samples(f"FINAL (step {step}, val {vl:.4f})")
    log_f.write(json.dumps({"step": step, "val_loss": vl, "final": True}) + "\n")
    log_f.close()
    print(f"DONE. final val {vl:.4f} (best {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
