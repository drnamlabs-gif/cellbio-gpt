# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 6b: the eval harness — same three exams for every rung of the ladder.

Models (frozen checkpoints): base (03), SFT (04), DPO v2 (5a), GRPO (5b).
Exams:
  1. LANGUAGE   perplexity on the temporal-holdout val tokens (val.bin),
                full sequential pass, 1024-token windows
  2. KNOWLEDGE  PubMedQA pqa_labeled, likelihood scoring: pick the option
                (yes/no/maybe) with the highest length-normalized logprob as
                a continuation of the SFT-template prompt; report accuracy +
                macro-F1 on the FULL set and on the contamination-CLEAN subset
  3. PHYSICS    the 727 combo-held-out BETSE/Nernst problems from stage 5b,
                greedy generation, verifier reward (reuses grpo.eval_val)

Saves out/eval_results.json (+ per-model PubMedQA predictions for the figure).
Seed 42. --smoke runs 40 PubMedQA items / 10 val batches / 60 physics problems.
"""

import argparse
import json
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
from bpe import BPE                # noqa: E402
from model import GPT, GPTConfig   # noqa: E402
from sft import PROMPT_FMT         # noqa: E402
import grpo                        # noqa: E402  (eval_val, generate_batched)

SEED = 42
OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
VAL_BIN = os.path.join(HERE, "..", "data", "tokens", "val.bin")
MODELS = {
    "base": os.path.join(HERE, "..", "03_pretrain", "out", "final.pt"),
    "sft": os.path.join(HERE, "..", "04_sft", "out", "sft_final.pt"),
    "dpo": os.path.join(HERE, "..", "05_dpo_rlvr", "out", "dpo_final_v2.pt"),
    "grpo": os.path.join(HERE, "..", "05_dpo_rlvr", "out", "grpo_final.pt"),
}
CTX, PPL_BATCH = 512, 64  # model block_size is 512
OPTIONS = ["yes", "no", "maybe"]
Q_FMT = "{ctx}\nQuestion: {q}\nAnswer yes, no, or maybe."


def load_model(path, device):
    ck = torch.load(path, weights_only=False)
    m = GPT(GPTConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def perplexity(model, val, device, max_batches=None):
    n_win = len(val) // CTX
    losses, n = [], 0
    for s in range(0, n_win, PPL_BATCH):
        w = val[s * CTX:(s + min(PPL_BATCH, n_win - s)) * CTX]
        x = torch.from_numpy(w.astype(np.int64)).view(-1, CTX).to(device)
        logits, _, _ = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               x[:, 1:].reshape(-1), reduction="sum")
        losses.append(loss.item())
        n += x[:, 1:].numel()
        if max_batches and len(losses) >= max_batches:
            break
    ce = sum(losses) / n
    return ce, float(np.exp(ce))


@torch.no_grad()
def pubmedqa(model, tok, items, device, eot):
    """Likelihood scoring; returns per-item predictions."""
    # pre-encode prompts (truncate context so prompt+option fits CTX)
    opt_ids = [tok.encode(o) for o in OPTIONS]
    max_opt = max(len(o) for o in opt_ids)
    seqs, meta = [], []  # one seq per (item, option)
    for i, it in enumerate(items):
        ctx = " ".join(it["contexts"])
        p = tok.encode(PROMPT_FMT.format(Q_FMT.format(ctx=ctx, q=it["question"])))
        if len(p) + max_opt > CTX:
            # drop context tokens from the middle of the prompt head
            tail = tok.encode(PROMPT_FMT.format(Q_FMT.format(ctx="", q=it["question"])))
            keep = CTX - max_opt - len(tail)
            p = tok.encode(PROMPT_FMT.format(
                Q_FMT.format(ctx=tok.decode(tok.encode(ctx)[:keep]), q=it["question"])))
            p = p[:CTX - max_opt]
        for o in opt_ids:
            seqs.append((p, o))
            meta.append(i)
    # bucket by total length for batched forward passes
    buckets = defaultdict(list)
    for j, (p, o) in enumerate(seqs):
        buckets[len(p) + len(o)].append(j)
    scores = np.zeros(len(seqs))
    for L, idxs in sorted(buckets.items()):
        for s in range(0, len(idxs), 48):
            chunk = idxs[s:s + 48]
            x = torch.tensor([seqs[j][0] + seqs[j][1] for j in chunk], device=device)
            logits, _, _ = model(x[:, :-1])
            logp = F.log_softmax(logits, dim=-1)
            tgt = x[:, 1:]
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for row, j in enumerate(chunk):
                n_opt = len(seqs[j][1])
                scores[j] = tok_lp[row, -n_opt:].mean().item()
    preds = []
    for i, it in enumerate(items):
        sc = [scores[3 * i + k] for k in range(3)]
        preds.append({"pubid": it["pubid"], "label": it["label"],
                      "pred": OPTIONS[int(np.argmax(sc))],
                      "contaminated": it["contaminated"]})
    return preds


def qa_metrics(preds):
    acc = float(np.mean([p["pred"] == p["label"] for p in preds]))
    f1s = []
    for c in OPTIONS:
        tp = sum(p["pred"] == c and p["label"] == c for p in preds)
        fp = sum(p["pred"] == c and p["label"] != c for p in preds)
        fn = sum(p["pred"] != c and p["label"] == c for p in preds)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"acc": acc, "macro_f1": float(np.mean(f1s)), "n": len(preds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    suff = "_smoke" if args.smoke else ""

    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: ≤30GB of 32GB
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1

    val_tokens = np.fromfile(VAL_BIN, dtype=np.uint16)
    items = [json.loads(l) for l in open(os.path.join(OUT, "pubmedqa.jsonl"),
                                         encoding="utf-8")]
    physics = grpo.load_problems("rlvr_val.jsonl")
    if args.smoke:
        items, physics = items[:40], physics[:60]
    maj = max(set(p["label"] for p in items),
              key=[p["label"] for p in items].count)

    results = {"seed": SEED, "date": "2026-10-23", "smoke": args.smoke,
               "baselines": {"chance_acc": 1 / 3,
                             "majority_acc": float(np.mean([p["label"] == maj for p in items])),
                             "majority_acc_clean": float(np.mean(
                                 [p["label"] == maj for p in items if not p["contaminated"]]))},
               "models": {}}
    all_preds = {}
    for name, path in MODELS.items():
        t0 = time.time()
        model = load_model(path, device)
        ce, ppl = perplexity(model, val_tokens, device,
                             max_batches=10 if args.smoke else None)
        preds = pubmedqa(model, tok, items, device, eot)
        clean = [p for p in preds if not p["contaminated"]]
        vr, vexact, by_kind = grpo.eval_val(model, tok, physics, device, eot)
        results["models"][name] = {
            "ckpt": os.path.relpath(path, os.path.join(HERE, "..")),
            "val_ce": ce, "val_ppl": ppl,
            "pubmedqa_full": qa_metrics(preds),
            "pubmedqa_clean": qa_metrics(clean),
            "betse_reward": vr, "betse_exact": vexact,
            "betse_by_kind": by_kind,
        }
        all_preds[name] = preds
        del model
        torch.cuda.empty_cache()
        r = results["models"][name]
        print(f"{name:5s} ppl {ppl:6.2f} | PubMedQA full {r['pubmedqa_full']['acc']:.3f} "
              f"clean {r['pubmedqa_clean']['acc']:.3f} | BETSE {vr:.3f}  "
              f"({time.time() - t0:.0f}s)")

    with open(os.path.join(OUT, f"eval_results{suff}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, f"eval_preds{suff}.json"), "w", encoding="utf-8") as f:
        json.dump(all_preds, f)
    print("saved", os.path.join(OUT, f"eval_results{suff}.json"))


if __name__ == "__main__":
    main()
