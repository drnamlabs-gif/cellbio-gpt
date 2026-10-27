# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Track B, stage B1: pour our domain database into a borrowed language model.

LoRA SFT on an open instruct model with a mixture built ONLY from this
project's own data (no teacher model, no external instruction data):
  sim      forward BETSE problems, lattice + random train sets (oversampled x3)
  textbook Nernst / deltaE / polarity problems from stage 5 (1500 sampled)
  reading  self-mined definition QA with context (3000 sampled)
Loss on assistant tokens only. Same answer-format system prompts as B0, so
scores are directly comparable.

Saves to out/ (everything kept for later comparison):
  b1_<tag>_log.jsonl   step, loss, lr, tokens, seconds
  b1_<tag>_eval.json   exam A, exam B, textbook val, reading F1, long answers,
                       dialogue — same sets as B0
  b1_<tag>_lora/       the LoRA adapter
  b1_<tag>.png         loss curve + B0 vs B1 vs Track A on every metric
Usage: python b1_domain_sft.py [--model ...] [--steps 500] [--smoke]
Seed 42.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "05_dpo_rlvr"))
from b0_baseline import DIALOGUE, LONG_QS, SYSTEM as SIM_SYSTEM, chat_batch  # noqa: E402
from bioelectric import reward as score  # noqa: E402

SEED = 42
TEXT_SYSTEM = (
    "You are a bioelectricity expert. Answer with EXACTLY ONE line.\n"
    "Equilibrium potential questions: 'E_K = <integer> mV.' (use the ion's symbol).\n"
    "Direction questions: 'E_K increases.' or 'E_K decreases.'\n"
    "Polarity questions: 'The cell depolarizes.' or 'The cell hyperpolarizes.'")
READ_SYSTEM = ("Answer using only the provided context. Reply with the single "
               "sentence from the context that defines the term, quoted exactly.")


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def f1(pred, gold):
    p, g = norm(pred), norm(gold)
    c = sum((Counter(p) & Counter(g)).values())
    if not p or not g or c == 0:
        return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)


def build_mixture(rng):
    sim_dir = os.path.join(ROOT, "08_betse_rlvr", "out")
    sim = jl(os.path.join(sim_dir, "betse_train.jsonl")) + \
        jl(os.path.join(sim_dir, "betse_train2_tr.jsonl"))
    text = jl(os.path.join(ROOT, "05_dpo_rlvr", "out", "rlvr_train.jsonl"))
    read = jl(os.path.join(ROOT, "11_rag", "out", "defqa_train.jsonl"))
    rng.shuffle(text)
    rng.shuffle(read)
    ex = []
    for p in sim * 3:
        ex.append((SIM_SYSTEM, p["prompt"], p["answer"], "sim"))
    for p in text[:1500]:
        ex.append((TEXT_SYSTEM, p["prompt"], p["answer"], "textbook"))
    for p in read[:3000]:
        ex.append((READ_SYSTEM, p["prompt"], p["answer"], "reading"))
    rng.shuffle(ex)
    return ex


def encode(tok, system, user, answer, max_len=768):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prefix + answer + tok.eos_token
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return f_ids, labels[:len(f_ids)]


def collate(batch, pad_id):
    L = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for r, (x, y) in enumerate(batch):
        ids[r, :len(x)] = torch.tensor(x)
        lab[r, :len(y)] = torch.tensor(y)
        att[r, :len(x)] = 1
    return ids, lab, att


def score_set(model, tok, probs, system, max_new=24):
    convs = [[{"role": "system", "content": system},
              {"role": "user", "content": p["prompt"]}] for p in probs]
    texts = chat_batch(model, tok, convs, max_new)
    rs, by = [], defaultdict(list)
    for p, t in zip(probs, texts):
        r = score(p["kind"], p["truth"], t)
        rs.append(r)
        by[p["kind"]].append(r)
    return {"val_reward": sum(rs) / len(rs),
            "exact": sum(r == 1.0 for r in rs) / len(rs),
            "by_kind": {k: sum(v) / len(v) for k, v in by.items()},
            "n": len(rs), "sample": texts[:3]}


def evaluate(model, tok, smoke):
    sim_dir = os.path.join(ROOT, "08_betse_rlvr", "out")
    rng = random.Random(SEED)
    res = {}
    res["exam_A"] = score_set(model, tok, jl(os.path.join(sim_dir, "betse_val.jsonl")),
                              SIM_SYSTEM)
    exam_b = jl(os.path.join(sim_dir, "betse_offgrid.jsonl"))
    res["exam_B"] = score_set(model, tok, exam_b[:40] if smoke else exam_b, SIM_SYSTEM)
    tv = jl(os.path.join(ROOT, "05_dpo_rlvr", "out", "rlvr_val.jsonl"))
    rng.shuffle(tv)
    res["textbook_val"] = score_set(model, tok, tv[:60 if smoke else 300], TEXT_SYSTEM)
    rv = jl(os.path.join(ROOT, "11_rag", "out", "defqa_val.jsonl"))
    rng.shuffle(rv)
    rv = rv[:30 if smoke else 200]
    convs = [[{"role": "system", "content": READ_SYSTEM},
              {"role": "user", "content": p["prompt"]}] for p in rv]
    preds = chat_batch(model, tok, convs, 96)
    res["reading_val"] = {
        "token_f1": sum(f1(a, p["answer"]) for a, p in zip(preds, rv)) / len(rv),
        "faithful": sum(" ".join(norm(a)) in " ".join(norm(p["prompt"]))
                        for a, p in zip(preds, rv)) / len(rv),
        "n": len(rv), "sample": preds[:3]}
    longs = chat_batch(model, tok, [[{"role": "user", "content": q}]
                                    for q in LONG_QS[:1 if smoke else 3]], 220)
    res["long"] = [{"q": q, "a": a, "words": len(a.split())}
                   for q, a in zip(LONG_QS, longs)]
    conv, turns = [], []
    for u in DIALOGUE[:1 if smoke else 3]:
        conv.append({"role": "user", "content": u})
        a = chat_batch(model, tok, [conv], 180)[0]
        conv.append({"role": "assistant", "content": a})
        turns.append({"user": u, "assistant": a})
    res["dialogue"] = turns
    return res


def plot(tag, log_path, ev, b0_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUT, BG = "#1a2b32", "#5c6f76", "#FBFAF7"
    TEAL, ORANGE, GREY = "#0E7C86", "#C97B1E", "#9AA7AC"
    rows = [json.loads(l) for l in open(log_path, encoding="utf-8")]
    b0 = json.load(open(b0_path, encoding="utf-8")) if os.path.exists(b0_path) else None
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor=BG)
    ax[0].plot([r["step"] for r in rows], [r["loss"] for r in rows], color=TEAL, lw=1.6)
    ax[0].set(xlabel="step", ylabel="train loss (assistant tokens)",
              title="B1 LoRA domain SFT")
    names = ["exam A", "exam B"]
    track_a = [0.945, 0.835]
    b0v = [b0["exam_A"]["val_reward"], b0["exam_B"]["val_reward"]] if b0 else [0, 0]
    b1v = [ev["exam_A"]["val_reward"], ev["exam_B"]["val_reward"]]
    x = range(len(names))
    w = 0.26
    ax[1].bar([i - w for i in x], b0v, w, color=GREY, label="B0 open model, untouched")
    ax[1].bar([i for i in x], b1v, w, color=ORANGE, label="B1 + our domain data (LoRA)")
    ax[1].bar([i + w for i in x], track_a, w, color=TEAL, label="Track A, 40M from scratch")
    for xs, vs in [([i - w for i in x], b0v), (list(x), b1v), ([i + w for i in x], track_a)]:
        for xi, v in zip(xs, vs):
            ax[1].text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    ax[1].axhline(0.538, color=MUT, ls=":", lw=1)
    ax[1].text(1.45, 0.548, "constant floor, exam A", fontsize=7.5, color=MUT, ha="right")
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels(names)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("held-out reward (same verifier)")
    ax[1].set_title("Same exams as Track A")
    ax[1].legend(frameon=False, fontsize=8, loc="lower left")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"Track B · B1 · {tag} · seed 42", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"b1_{tag}.png"), dpi=130, facecolor=BG,
                bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    steps = 30 if args.smoke else args.steps
    tag = args.model.split("/")[-1] + ("_smoke" if args.smoke else "")
    os.makedirs(OUT, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: <=30GB of 32GB
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable LoRA params: {n_tr / 1e6:.1f}M")

    rng = random.Random(SEED)
    mix = build_mixture(rng)
    print("mixture:", dict(Counter(m[3] for m in mix)), "total", len(mix))
    enc = [encode(tok, s, u, a) for s, u, a, _ in mix]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    log_path = os.path.join(OUT, f"b1_{tag}_log.jsonl")
    log_f = open(log_path, "w", encoding="utf-8")
    model.train()
    t0, cursor, warm = time.time(), 0, max(1, steps // 15)
    for step in range(steps):
        lr = args.lr * (step + 1) / warm if step < warm else \
            args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr
        tot, ntok = 0.0, 0
        for _ in range(args.accum):
            batch = [enc[(cursor + i) % len(enc)] for i in range(args.micro)]
            cursor += args.micro
            ids, lab, att = collate(batch, tok.pad_token_id)
            ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / args.accum).backward()
            tot += loss.item() / args.accum
            ntok += int((lab != -100).sum())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        log_f.write(json.dumps({"step": step, "loss": tot, "lr": lr, "tokens": ntok,
                                "sec": round(time.time() - t0, 1)}) + "\n")
        if step % 25 == 0 or step == steps - 1:
            log_f.flush()
            print(f"step {step:4d}/{steps} loss {tot:.4f} lr {lr:.2e} "
                  f"{(time.time() - t0) / (step + 1):.2f}s/step", flush=True)
    log_f.close()
    model.save_pretrained(os.path.join(OUT, f"b1_{tag}_lora"))

    model.eval()
    tok.padding_side = "left"
    ev = evaluate(model, tok, args.smoke)
    ev.update({"model": args.model, "steps": steps, "seed": SEED,
               "mixture": dict(Counter(m[3] for m in mix)),
               "lora": {"r": 16, "alpha": 32, "trainable_M": round(n_tr / 1e6, 2)}})
    with open(os.path.join(OUT, f"b1_{tag}_eval.json"), "w", encoding="utf-8") as f:
        json.dump(ev, f, indent=2)
    for k in ("exam_A", "exam_B", "textbook_val"):
        print(k, {kk: ev[k][kk] for kk in ("val_reward", "exact", "by_kind")})
    print("reading_val", {k: ev["reading_val"][k] for k in ("token_f1", "faithful")})
    for x in ev["long"]:
        print(f"\nQ: {x['q']}\nA ({x['words']} words): {x['a']}")
    for t in ev["dialogue"]:
        print(f"\n[user] {t['user']}\n[model] {t['assistant']}")
    plot(tag, log_path, ev, os.path.join(OUT, f"b0_{args.model.split('/')[-1]}.json"))
    print("saved", os.path.join(OUT, f"b1_{tag}.png"))


if __name__ == "__main__":
    main()
