# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Track B, stage 3: continue pretraining an open BASE model on our PubMed
corpus — the mirror of Track A's 03_pretrain/pretrain.py, except that the
starting weights are a pretrained general LLM instead of random init.

Mirrored from Track A: same documents, same temporal val split, 1 epoch,
full-parameter training, bf16 autocast, warmup + cosine, clip 1.0, seed 42,
resumable checkpoints, periodic val + generation samples, everything logged.
Different by necessity: the model's own tokenizer (data/tokens_qwen), and val
is reported as bits-per-byte so it compares across tokenizers. Step 0 is
evaluated before any update: that is the "untouched pretrained model" point.

Usage:
  python b03_cpt.py --model Qwen/Qwen2.5-0.5B --smoke     # throughput check
  python b03_cpt.py --model Qwen/Qwen2.5-0.5B             # full epoch
  python b03_cpt.py --model ... --resume
Outputs out/cpt_<tag>/: log.jsonl, samples.txt, ckpt/ (resume), final/,
loss.png. Optimizer: AdamW (fp32 master weights) or Adafactor (--optim).
"""

import argparse
import json
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
SAMPLE_PROMPT = "Membrane potential"


def batches(data, seq, micro, rng):
    n = len(data) - seq - 1
    while True:
        idx = rng.integers(0, n, micro)
        x = np.stack([data[i:i + seq] for i in idx]).astype(np.int64)
        y = np.stack([data[i + 1:i + seq + 1] for i in idx]).astype(np.int64)
        yield torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def eval_bpb(model, data, seq, n_tokens, tok_per_byte, micro=2):
    model.eval()
    n_win = max(1, n_tokens // seq)
    starts = np.linspace(0, len(data) - seq - 2, n_win).astype(np.int64)
    tot, cnt = 0.0, 0
    for i in range(0, len(starts), micro):
        st = starts[i:i + micro]
        x = torch.from_numpy(np.stack([data[s:s + seq] for s in st]).astype(np.int64)).cuda()
        y = torch.from_numpy(np.stack([data[s + 1:s + seq + 1] for s in st]).astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                                 y.view(-1), reduction="sum")
        tot += loss.item()
        cnt += y.numel()
    model.train()
    nats = tot / cnt
    return nats, nats * tok_per_byte / math.log(2)


@torch.no_grad()
def sample(model, tok):
    model.eval()
    enc = tok(SAMPLE_PROMPT, return_tensors="pt").to("cuda")
    out = model.generate(**enc, max_new_tokens=90, do_sample=False,
                         repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    model.train()
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=2)   # 152k-vocab logits are large
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--optim", choices=["adamw", "adafactor"], default="adamw")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_tokens", type=int, default=1_000_000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tokdir", default="tokens_qwen")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="small models fit without activation checkpointing (faster)")
    ap.add_argument("--lora_r", type=int, default=0,
                    help="0 = full-parameter; >0 = freeze the base, train LoRA on all linears")
    ap.add_argument("--max_steps", type=int, default=0, help="pilot run: stop (and decay) here")
    ap.add_argument("--bf16_base", action="store_true",
                    help="LoRA only: keep the frozen base in bf16, adapters in fp32 (faster)")
    args = ap.parse_args()

    TOK_DIR = os.path.join(HERE, "..", "data", args.tokdir)
    tag = args.model.split("/")[-1] + (f"_lora{args.lora_r}" if args.lora_r else "") + \
        (f"_{args.max_steps}" if args.max_steps else "") + ("_smoke" if args.smoke else "")
    out = os.path.join(HERE, "out", f"cpt_{tag}")
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(TOK_DIR, "meta.json")))
    val = np.memmap(os.path.join(TOK_DIR, "val.bin"), dtype=np.uint32, mode="r")
    train_path = os.path.join(TOK_DIR, "train.bin")
    train = np.memmap(train_path if not args.smoke or os.path.exists(train_path)
                      else os.path.join(TOK_DIR, "val.bin"), dtype=np.uint32, mode="r")
    tok_per_byte = meta["n_val_tokens"] / meta["n_val_bytes"]
    tokens_per_step = args.seq * args.micro * args.accum
    steps = 30 if args.smoke else int(args.epochs * meta["n_train_tokens"] / tokens_per_step)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    warm = 10 if args.smoke else min(500, steps // 20)

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(10)
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: <=30GB of 32GB
    tok = AutoTokenizer.from_pretrained(args.model)
    src = os.path.join(out, "ckpt") if args.resume else args.model
    base_dtype = torch.bfloat16 if (args.lora_r and args.bf16_base) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model if args.lora_r else src, dtype=base_dtype).cuda()
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if args.lora_r:
        from peft import LoraConfig, PeftModel, get_peft_model
        if args.resume:
            model = PeftModel.from_pretrained(model, src, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        model.print_trainable_parameters()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.optim == "adamw":
        decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                 {"params": nodecay, "weight_decay": 0.0}],
                                lr=args.lr, betas=(0.9, 0.95), fused=True)
    else:
        opt = Adafactor(model.parameters(), lr=args.lr, relative_step=False,
                        scale_parameter=False, warmup_init=False)
    start = 0
    log_path = os.path.join(out, "log.jsonl")
    if args.resume:
        st = torch.load(os.path.join(out, "ckpt", "trainer.pt"), weights_only=False)
        opt.load_state_dict(st["opt"])
        rng = st["rng"]
        start = st["step"] + 1
    print(f"{args.model}: {n_par / 1e6:.1f}M trainable params | {steps} steps x "
          f"{tokens_per_step:,} tok = {steps * tokens_per_step / 1e9:.2f}B tokens | "
          f"start {start}", flush=True)

    log_f = open(log_path, "a", encoding="utf-8")
    smp = open(os.path.join(out, "samples.txt"), "a", encoding="utf-8")
    if start == 0:
        nats, bpb = eval_bpb(model, val, args.seq,
                             200_000 if args.smoke else args.eval_tokens, tok_per_byte)
        print(f"step 0 (untouched): val {nats:.4f} nats/token, {bpb:.4f} bits/byte", flush=True)
        log_f.write(json.dumps({"step": 0, "val_nats": nats, "val_bpb": bpb,
                                "untouched": True}) + "\n")
        smp.write(f"=== step 0 (untouched) ===\n{sample(model, tok)}\n\n")
    gen = batches(train, args.seq, args.micro, rng)
    model.train()
    t0 = time.time()
    for step in range(start, steps):
        lr = args.lr * (step + 1) / warm if step < warm else \
            args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
        for g in opt.param_groups:
            g["lr"] = lr
        tot = 0.0
        for _ in range(args.accum):
            x, y = next(gen)
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x).logits
            loss = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.size(-1)), y.view(-1))
            (loss / args.accum).backward()
            tot += loss.item() / args.accum
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        el = time.time() - t0
        rate = (step - start + 1) * tokens_per_step / el
        rec = {"step": step + 1, "loss": tot, "lr": lr, "grad_norm": gn,
               "tok_per_s": round(rate), "elapsed_s": round(el)}
        if (step + 1) % 10 == 0 or step + 1 == steps:
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
        if (step + 1) % 50 == 0 or step + 1 == steps:
            eta = (steps - step - 1) * tokens_per_step / rate / 3600
            print(f"step {step + 1}/{steps} loss {tot:.4f} lr {lr:.2e} "
                  f"{rate / 1e3:.1f}k tok/s ETA {eta:.1f}h "
                  f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        if not args.smoke and ((step + 1) % args.eval_every == 0 or step + 1 == steps):
            nats, bpb = eval_bpb(model, val, args.seq, args.eval_tokens, tok_per_byte)
            print(f"  VAL step {step + 1}: {nats:.4f} nats/token, {bpb:.4f} bits/byte", flush=True)
            log_f.write(json.dumps({"step": step + 1, "val_nats": nats, "val_bpb": bpb}) + "\n")
            log_f.flush()
            smp.write(f"=== step {step + 1} ===\n{sample(model, tok)}\n\n")
            smp.flush()
            model.save_pretrained(os.path.join(out, "ckpt"))
            torch.save({"opt": opt.state_dict(), "rng": rng, "step": step},
                       os.path.join(out, "ckpt", "trainer.pt"))
    if args.smoke:
        print(f"SMOKE: {rate / 1e3:.1f}k tok/s, peak mem "
              f"{torch.cuda.max_memory_allocated() / 1e9:.1f}GB -> full epoch "
              f"{meta['n_train_tokens'] / rate / 3600:.1f}h", flush=True)
        return
    nats, bpb = eval_bpb(model, val, args.seq, meta["n_val_tokens"], tok_per_byte)
    log_f.write(json.dumps({"step": steps, "val_nats_full": nats,
                            "val_bpb_full": bpb, "final": True}) + "\n")
    log_f.close()
    smp.write(f"=== final ===\n{sample(model, tok)}\n\n")
    smp.close()
    model.save_pretrained(os.path.join(out, "final"))
    tok.save_pretrained(os.path.join(out, "final"))
    print(f"DONE: full val {nats:.4f} nats/token, {bpb:.4f} bits/byte")


if __name__ == "__main__":
    main()
