# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8d: baseline table for the real-sim exam (98 held-out-combo problems).

Who takes the exam:
  non-LLM floors ... constant predictor (train mode answer per kind),
                     1-nearest-neighbor in log2 knob space (pure interpolation,
                     no language model at all)
  generic ladder ... base / SFT / DPO / GRPO-textbook checkpoints, zero-shot
                     (never saw a sim problem)
  fine-tuned ....... bg8_ws (warm-start SFT on train combos), bg8_final (+GRPO)

Same verifier as training: score(kind, truth, text). Greedy decoding, seed 42.
Saves out/betse_baselines.json + out/betse_baselines.png.
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from betse_grpo import OUT, TOKENIZER, eval_val, load_problems  # noqa: E402
from bpe import BPE                       # noqa: E402
from model import GPT, GPTConfig          # noqa: E402
from bioelectric import reward as score   # noqa: E402

SEED = 42
ROOT = os.path.join(HERE, "..")
CKPTS = [
    ("base (pretrain only)", os.path.join(ROOT, "03_pretrain", "out", "final.pt")),
    ("+ SFT (chat)", os.path.join(ROOT, "04_sft", "out", "sft_final.pt")),
    ("+ DPO", os.path.join(ROOT, "05_dpo_rlvr", "out", "dpo_final_v2.pt")),
    ("+ GRPO textbook (5b)", os.path.join(ROOT, "05_dpo_rlvr", "out", "grpo_final.pt")),
    ("sim warm-start SFT (8b)", os.path.join(OUT, "bg8_ws.pt")),
    ("+ GRPO vs simulator (8b)", os.path.join(OUT, "bg8_final.pt")),
]


def metrics(val, texts):
    rs, by = [], defaultdict(list)
    for p, t in zip(val, texts):
        r = score(p["kind"], p["truth"], t)
        rs.append(r)
        by[p["kind"]].append(r)
    return {"val_reward": float(np.mean(rs)),
            "exact": float(np.mean([r == 1.0 for r in rs])),
            "by_kind": {k: float(np.mean(v)) for k, v in by.items()}}


def main():
    exam = sys.argv[sys.argv.index("--exam") + 1] if "--exam" in sys.argv \
        else "betse_val.jsonl"
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    suff = f"_{tag}" if tag else ""
    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: <=30GB of 32GB
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    train = load_problems("betse_train.jsonl")
    val = load_problems(exam)
    print(f"exam {exam}: {len(val)} problems")
    results = {}

    # ---- floor 1: constant predictor (train mode answer per kind) ----
    const = {}
    for kind in {p["kind"] for p in train}:
        const[kind] = Counter(p["answer"] for p in train
                              if p["kind"] == kind).most_common(1)[0][0]
    results["constant (train mode)"] = metrics(val, [const[p["kind"]] for p in val])
    print("constant answers:", const)

    # ---- floor 2: 1-NN interpolation in log2 knob space (no LLM) ----
    def dist(a, b):
        return sum((math.log2(x) - math.log2(y)) ** 2 for x, y in zip(a, b))

    nn_texts = []
    for p in val:
        pool = [q for q in train if q["kind"] == p["kind"]]
        nn_texts.append(min(pool, key=lambda q: dist(q["combo"], p["combo"]))["answer"])
    results["1-NN knob lookup (no LLM)"] = metrics(val, nn_texts)

    # ---- the ladder + fine-tuned models, same exam, greedy ----
    for name, path in CKPTS:
        ck = torch.load(path, weights_only=False)
        cfg = GPTConfig(**ck["cfg"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        vr, ve, vk = eval_val(model, tok, val, device, eot)
        results[name] = {"val_reward": vr, "exact": ve, "by_kind": vk}
        print(f"{name:28s} val {vr:.3f} exact {ve:.3f} {vk}", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT, f"betse_baselines{suff}.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "n_val": len(val), "results": results}, f, indent=2)
    print("saved", os.path.join(OUT, f"betse_baselines{suff}.json"))

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUT, BG = "#1a2b32", "#5c6f76", "#FBFAF7"
    TEAL, ORANGE, RED = "#0E7C86", "#C97B1E", "#B3403A"
    names = list(results)
    colors = [MUT, MUT, RED, RED, RED, RED, TEAL, TEAL][:len(names)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=BG)
    for ax, key, title in [(axes[0], "val_reward", "held-out val reward (all 98)"),
                           (axes[1], None, "held-out NUMERIC reward (nernst)")]:
        vals = [results[n][key] if key else results[n]["by_kind"].get("nernst", 0.0)
                for n in names]
        bars = ax.barh(range(len(names)), vals, color=colors, height=0.62)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=10, color=INK)
        for b, v in zip(bars, vals):
            ax.text(min(v + 0.015, 0.97), b.get_y() + b.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
    axes[0].text(0.02, len(names) - 0.4, "gray = no-LLM floors   red = generic ladder "
                 "(zero-shot)   teal = fine-tuned on sim problems",
                 fontsize=7.5, color=MUT)
    fig.suptitle("Who passes the BETSE physics exam? 98 held-out combos, greedy, seed 42",
                 fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"betse_baselines{suff}.png"), dpi=130, facecolor=BG,
                bbox_inches="tight")
    print("saved", os.path.join(OUT, f"betse_baselines{suff}.png"))


if __name__ == "__main__":
    main()
