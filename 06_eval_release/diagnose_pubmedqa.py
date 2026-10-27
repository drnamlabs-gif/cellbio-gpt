# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 6e: root-cause the PubMedQA failure ("never says no").

Observed: all four checkpoints predict "no" exactly 0/1000 times. Three
hypotheses, tested in order:
  H1 tokenization artifact — the options encode into unusual pieces
  H2 prior bias — the model prefers "yes" BEFORE seeing any question;
     raw likelihood scoring then measures the prior, not the answer.
     Fix to test: contextual calibration / PMI scoring (Zhao et al. 2021):
     score = mean_logprob(option | prompt) - mean_logprob(option | empty prompt)
  H3 the model does not read the abstract at all — re-score every item with
     the context REMOVED; if accuracy barely moves, the evidence was unused.

Saves out/pubmedqa_diagnosis.json. Seed 42.
"""

import json
import os
import sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_harness import (BPE, CTX, MODELS, OPTIONS, PROMPT_FMT, Q_FMT,  # noqa: E402
                          TOKENIZER, load_model)

OUT = os.path.join(HERE, "out")
SEED = 42


@torch.no_grad()
def option_scores(model, tok, prompts, device):
    """mean token logprob of each option as continuation; prompts: token lists."""
    opt_ids = [tok.encode(o) for o in OPTIONS]
    seqs = [(p, o) for p in prompts for o in opt_ids]
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
            tok_lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
            for row, j in enumerate(chunk):
                n_opt = len(seqs[j][1])
                scores[j] = tok_lp[row, -n_opt:].mean().item()
    return scores.reshape(len(prompts), len(OPTIONS))


def encode_prompts(tok, items, with_context):
    opt_max = max(len(tok.encode(o)) for o in OPTIONS)
    out = []
    for it in items:
        ctx = " ".join(it["contexts"]) if with_context else ""
        p = tok.encode(PROMPT_FMT.format(Q_FMT.format(ctx=ctx, q=it["question"])))
        if len(p) + opt_max > CTX:
            tail = tok.encode(PROMPT_FMT.format(Q_FMT.format(ctx="", q=it["question"])))
            keep = CTX - opt_max - len(tail)
            p = tok.encode(PROMPT_FMT.format(
                Q_FMT.format(ctx=tok.decode(tok.encode(ctx)[:keep]), q=it["question"])))
            p = p[:CTX - opt_max]
        out.append(p)
    return out


def metrics(sc, items, name):
    preds = [OPTIONS[i] for i in sc.argmax(1)]
    acc = float(np.mean([p == it["label"] for p, it in zip(preds, items)]))
    clean = [(p, it) for p, it in zip(preds, items) if not it["contaminated"]]
    acc_c = float(np.mean([p == it["label"] for p, it in clean]))
    by_label = {lb: float(np.mean([p == lb for p, it in zip(preds, items)
                                   if it["label"] == lb] or [0.0]))
                for lb in OPTIONS}
    return {"scoring": name, "acc_full": acc, "acc_clean": acc_c,
            "pred_dist": dict(Counter(preds)), "recall_by_label": by_label}


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: ≤30GB of 32GB
    tok = BPE.load(TOKENIZER)

    # H1: tokenization of the options
    tok_report = {o: [tok.decode([t]) for t in tok.encode(o)] for o in OPTIONS}
    print("H1 option tokens:", tok_report)

    items = [json.loads(l) for l in open(os.path.join(OUT, "pubmedqa.jsonl"),
                                         encoding="utf-8")]
    prompts_full = encode_prompts(tok, items, with_context=True)
    prompts_noctx = encode_prompts(tok, items, with_context=False)
    # content-free prompt for the prior (template only, no question)
    prior_prompt = [tok.encode(PROMPT_FMT.format(Q_FMT.format(ctx="", q="")))]

    report = {"seed": SEED, "date": "2026-10-23", "option_tokens": tok_report,
              "models": {}}
    for name, path in MODELS.items():
        model = load_model(path, device)
        prior = option_scores(model, tok, prior_prompt, device)[0]
        sc_full = option_scores(model, tok, prompts_full, device)
        sc_noctx = option_scores(model, tok, prompts_noctx, device)
        rows = [
            metrics(sc_full, items, "raw (as in harness)"),
            metrics(sc_full - prior, items, "calibrated (minus prior)"),
            metrics(sc_noctx, items, "no-context raw"),
            metrics(sc_noctx - prior, items, "no-context calibrated"),
        ]
        report["models"][name] = {
            "prior_mean_logprob": {o: float(p) for o, p in zip(OPTIONS, prior)},
            "results": rows,
        }
        print(f"\n{name}: prior logprob "
              + "  ".join(f"{o} {p:.2f}" for o, p in zip(OPTIONS, prior)))
        for r in rows:
            print(f"  {r['scoring']:26s} acc {r['acc_full']:.3f} "
                  f"(clean {r['acc_clean']:.3f})  preds {r['pred_dist']}")
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT, "pubmedqa_diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nsaved", os.path.join(OUT, "pubmedqa_diagnosis.json"))


if __name__ == "__main__":
    main()
