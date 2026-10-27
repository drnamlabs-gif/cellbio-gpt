# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 5b warm-start: short SFT on 3k worked bioelectric problems.

Without this, GRPO has zero gradient: the DPO model has never seen the task,
every sample scores 0, every group advantage is 0. A few hundred SFT steps
teach the answer FORMAT; GRPO then has a reward landscape to climb.
Init: dpo_final_v2.pt. Reuses the stage-4 SFT machinery (same template,
same loss mask). Saves ws_final.pt + ws_log.jsonl. Seed 42.
"""

import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))
sys.path.insert(0, os.path.join(HERE, "..", "02_transformer"))
sys.path.insert(0, os.path.join(HERE, "..", "04_sft"))
from bpe import BPE                      # noqa: E402
from model import GPT, GPTConfig        # noqa: E402
from sft import Batches, encode_examples  # noqa: E402

SEED = 42
OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
INIT = os.path.join(OUT, "dpo_final_v2.pt")
STEPS, BATCH_LR, WARMUP = 400, 3e-5, 40


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1

    xs, ys, _ = encode_examples(os.path.join(OUT, "rlvr_ws.jsonl"), tok)
    tb = Batches(xs, ys, np.random.default_rng(SEED), eot)
    print(f"{len(xs)} warm-start examples, {STEPS} steps")

    ck = torch.load(INIT, weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])

    opt = torch.optim.AdamW(model.parameters(), lr=BATCH_LR,
                            betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    log_f = open(os.path.join(OUT, "ws_log.jsonl"), "a", encoding="utf-8")
    model.train()
    step, t0 = 0, time.time()
    while step < STEPS:
        for x, y in tb.epoch(device):
            if step >= STEPS:
                break
            lr = BATCH_LR * (step + 1) / WARMUP if step < WARMUP else \
                BATCH_LR * 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (STEPS - WARMUP)))
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 50 == 0 or step == STEPS - 1:
                dt = (time.time() - t0) / (step + 1)
                print(f"step {step:4d}/{STEPS}  loss {loss.item():.4f}  "
                      f"{dt*1000:.0f} ms/step", flush=True)
                log_f.write(json.dumps({"step": step, "loss": loss.item()}) + "\n")
            step += 1
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": STEPS,
                "seed": SEED, "init": "dpo_final_v2.pt"},
               os.path.join(OUT, "ws_final.pt"))
    log_f.close()
    print("DONE. saved ws_final.pt")


if __name__ == "__main__":
    main()
