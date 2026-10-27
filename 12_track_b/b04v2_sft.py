# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Track B recipe v2, post-training: chat-format SFT on the MIXTURE built by
b04v2_build_data.py, starting from the stage-3 PubMed model (0.640 bits/byte).

Differences from the mirrored b04_sft.py (which made answers short):
  - data: 80k long multi-turn general dialogues + 30k long domain tasks +
    20k short domain tasks (not 97.5k one-liners)
  - format: the real chat template <|im_start|>role\\n...<|im_end|> (SmolLM2-
    Instruct's), so turns have an explicit end token and multi-turn works
  - loss on EVERY assistant turn (+ its <|im_end|>), nothing else
Same as before: full-parameter, AdamW, bf16 autocast, warmup + cosine to
10%, clip 1.0, seed 42, VRAM cap.

Measured, not eyeballed (log.jsonl + probe.json):
  - val answer loss every 500 steps
  - bits/byte on the PubMed temporal val (1M tokens) before and after: how
    much domain knowledge the post-training costs
  - probes: answer length in words, whether the model stops on its own
    (<|im_end|> before the 420-token cap), and a 3-turn follow-up dialogue

Outputs out/sft2_<tag>/: log.jsonl, probe.json, samples.txt, final/.
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b03_cpt import eval_bpb  # noqa: E402

SEED = 42
DATA = os.path.join(HERE, "..", "data", "sft_v2")
TOK_DIR = os.path.join(HERE, "..", "data", "tokens_smol")
CHAT_TOK = "HuggingFaceTB/SmolLM2-135M-Instruct"   # same ids as the base, has the template
MAX_LEN = 2048
SYSTEM = ("You are a careful assistant for cell biology and bioelectricity. "
          "Answer clearly; say so when you are unsure.")
PROBES = ["What is bioelectricity?",
          "What is cancer?",
          "How do gap junctions coordinate membrane voltage across a tissue?",
          "Why does depolarizing a cell's resting potential change its behavior?",
          "What is a gap junction?",
          "In cell biology, what does ATP stand for?"]
DIALOGUE = ["What is the resting membrane potential?",
            "Explain more.",
            "What happens to it if extracellular potassium doubles?"]
CAP = 420


def encode(msgs, tok, im_end):
    """ChatML token ids + labels (-100 except assistant content and its <|im_end|>)."""
    ids, lab = [], []
    for m in msgs:
        head = tok.encode(f"<|im_start|>{m['role']}\n", add_special_tokens=False)
        body = tok.encode(m["content"], add_special_tokens=False) + [im_end]
        tail = tok.encode("\n", add_special_tokens=False)
        ids += head + body + tail
        lab += [-100] * len(head) + (body if m["role"] == "assistant" else [-100] * len(body)) \
            + [-100] * len(tail)
    return ids, lab


def load(path, tok, im_end):
    xs, ys, skip = [], [], 0
    for line in open(path, encoding="utf-8"):
        ids, lab = encode(json.loads(line)["messages"], tok, im_end)
        if len(ids) > MAX_LEN:
            skip += 1
            continue
        xs.append(np.array(ids[:-1], dtype=np.int64))
        ys.append(np.array(lab[1:], dtype=np.int64))
    return xs, ys, skip


def make_batches(xs, rng, size):
    """Length-bucketed batches (less padding), shuffled order."""
    order = rng.permutation(len(xs))
    out = []
    for i in range(0, len(order), size * 64):
        mega = sorted(order[i:i + size * 64], key=lambda j: len(xs[j]))
        out += [mega[k:k + size] for k in range(0, len(mega), size) if len(mega[k:k + size]) == size]
    rng.shuffle(out)
    return out


def collate(idx, xs, ys, pad):
    L = max(len(xs[j]) for j in idx)
    x = np.full((len(idx), L), pad, dtype=np.int64)
    y = np.full((len(idx), L), -100, dtype=np.int64)
    for r, j in enumerate(idx):
        x[r, :len(xs[j])] = xs[j]
        y[r, :len(ys[j])] = ys[j]
    return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()


def loss_sum(model, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x).logits
    return torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                             y.view(-1), ignore_index=-100, reduction="sum")


@torch.no_grad()
def val_loss(model, vx, vy, pad, micro):
    model.eval()
    tot, cnt = 0.0, 0
    for b in make_batches(vx, np.random.default_rng(SEED + 1), micro):
        x, y = collate(b, vx, vy, pad)
        tot += loss_sum(model, x, y).item()
        cnt += (y != -100).sum().item()
    model.train()
    return tot / max(1, cnt)


@torch.no_grad()
def reply(model, tok, msgs, im_end):
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + \
        "<|im_start|>assistant\n"
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(**enc, max_new_tokens=CAP, do_sample=False, repetition_penalty=1.05,
                         eos_token_id=im_end, pad_token_id=im_end)
    new = out[0][enc["input_ids"].shape[1]:].tolist()
    stopped = im_end in new
    return tok.decode(new, skip_special_tokens=True).strip(), stopped


def probe(model, tok, im_end):
    model.eval()
    rows = []
    for q in PROBES:
        a, s = reply(model, tok, [{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": q}], im_end)
        rows.append({"q": q, "a": a, "words": len(a.split()), "stopped": s})
    nosys = []
    for q in PROBES:
        a, s = reply(model, tok, [{"role": "user", "content": q}], im_end)
        nosys.append({"q": "[no system] " + q, "a": a, "words": len(a.split()), "stopped": s})
    msgs, dia = [{"role": "system", "content": SYSTEM}], []
    for q in DIALOGUE:
        msgs.append({"role": "user", "content": q})
        a, s = reply(model, tok, msgs, im_end)
        msgs.append({"role": "assistant", "content": a})
        dia.append({"q": q, "a": a, "words": len(a.split()), "stopped": s})
    model.train()
    return {"single": rows, "single_nosys": nosys, "dialogue": dia,
            "median_words": float(np.median([r["words"] for r in rows])),
            "median_words_nosys": float(np.median([r["words"] for r in nosys])),
            "stop_rate": float(np.mean([r["stopped"] for r in rows + dia]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(HERE, "out", "cpt_SmolLM2-135M", "final"))
    ap.add_argument("--tag", default="SmolLM2-135M")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default="sft_v2")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    global DATA
    DATA = os.path.join(HERE, "..", "data", args.data)

    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: <=30GB of 32GB
    out = os.path.join(HERE, "out", f"sft2{args.suffix}_{args.tag}" + ("_smoke" if args.smoke else ""))
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(CHAT_TOK)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    xs, ys, sk = load(os.path.join(DATA, "train.jsonl"), tok, im_end)
    vx, vy, _ = load(os.path.join(DATA, "val.jsonl"), tok, im_end)
    if args.smoke:
        xs, ys, vx, vy = xs[:3000], ys[:3000], vx[:200], vy[:200]
    accum = args.batch // args.micro
    n_tok = sum(len(x) for x in xs)
    rng = np.random.default_rng(SEED)
    steps_ep = len(xs) // args.batch
    total = 40 if args.smoke else int(steps_ep * args.epochs)
    warm = max(10, total // 33)
    print(f"train {len(xs)} convs ({sk} skipped >{MAX_LEN}), {n_tok / 1e6:.1f}M tokens | "
          f"val {len(vx)} | {total} steps, batch {args.batch} = {args.micro} x {accum}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.init, dtype=torch.float32).cuda()
    model.config.use_cache = False
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    val_bin = np.memmap(os.path.join(TOK_DIR, "val.bin"), dtype=np.uint32, mode="r")
    meta = json.load(open(os.path.join(TOK_DIR, "meta.json")))
    tpb = meta["n_val_tokens"] / meta["n_val_bytes"]
    bpb_tokens = 200_000 if args.smoke else 1_000_000

    log = open(os.path.join(out, "log.jsonl"), "a", encoding="utf-8")
    log.write(json.dumps({"meta": {**vars(args), "n_train": len(xs), "n_skip": sk,
                                   "train_tokens": n_tok, "steps": total}}) + "\n")
    _, bpb0 = eval_bpb(model, val_bin, 1024, bpb_tokens, tpb)
    vl0 = val_loss(model, vx, vy, im_end, args.micro)
    p0 = probe(model, tok, im_end)
    print(f"step 0: PubMed {bpb0:.4f} bits/byte | val answer loss {vl0:.4f} | probe median "
          f"{p0['median_words']} words, stop rate {p0['stop_rate']:.2f}", flush=True)
    log.write(json.dumps({"step": 0, "bpb": bpb0, "val_loss": vl0,
                          "median_words": p0["median_words"], "stop_rate": p0["stop_rate"]}) + "\n")
    probes = {"before": p0}

    batches = make_batches(xs, rng, args.batch)
    step, t0 = 0, time.time()
    model.train()
    while step < total:
        for b in batches:
            if step >= total:
                break
            t = (step - warm) / max(1, total - warm)
            lr = args.lr * (step + 1) / warm if step < warm else \
                args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = collate(b, xs, ys, im_end)
            ntok = (y != -100).sum().item()
            tot = 0.0
            for k in range(accum):
                s = slice(k * args.micro, (k + 1) * args.micro)
                l = loss_sum(model, x[s], y[s]) / ntok
                l.backward()
                tot += l.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == total:
                dt = (time.time() - t0) / step
                print(f"step {step}/{total} loss {tot:.3f} lr {lr:.2e} "
                      f"{dt:.2f} s/step ETA {(total - step) * dt / 60:.0f} min", flush=True)
                log.write(json.dumps({"step": step, "loss": tot, "lr": lr}) + "\n")
                log.flush()
            if step % 500 == 0 and step < total:
                vl = val_loss(model, vx, vy, im_end, args.micro)
                print(f"  val answer loss {vl:.4f}", flush=True)
                log.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
                log.flush()
        batches = make_batches(xs, rng, args.batch)

    _, bpb1 = eval_bpb(model, val_bin, 1024, bpb_tokens, tpb)
    vl1 = val_loss(model, vx, vy, im_end, args.micro)
    p1 = probe(model, tok, im_end)
    probes["after"] = p1
    log.write(json.dumps({"step": step, "bpb": bpb1, "val_loss": vl1, "final": True,
                          "median_words": p1["median_words"], "stop_rate": p1["stop_rate"]}) + "\n")
    log.close()
    json.dump(probes, open(os.path.join(out, "probe.json"), "w", encoding="utf-8"), indent=2)
    with open(os.path.join(out, "samples.txt"), "w", encoding="utf-8") as f:
        for when, p in probes.items():
            f.write(f"===== {when} =====\n")
            for r in p["single"] + p["single_nosys"] + p["dialogue"]:
                f.write(f"Q: {r['q']}\nA ({r['words']} words, stopped={r['stopped']}): {r['a']}\n\n")
    model.save_pretrained(os.path.join(out, "final"))
    tok.save_pretrained(os.path.join(out, "final"))
    print(f"DONE: PubMed {bpb0:.4f} -> {bpb1:.4f} bits/byte | answer loss {vl0:.4f} -> {vl1:.4f} | "
          f"median words {p0['median_words']} -> {p1['median_words']} | stop rate "
          f"{p0['stop_rate']:.2f} -> {p1['stop_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
