# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 6c: the release scoreboard — out/eval_results.png from eval_results.json.

Left: PubMedQA accuracy (full + contamination-clean) vs chance and majority.
Middle: BETSE verifier reward on combo-held-out physics problems.
Right: perplexity on the temporal holdout (the alignment tax, visible).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
INK, MUT = "#1a2b32", "#5c6f76"
TEAL, ORANGE, RED, GREEN = "#0E7C86", "#C97B1E", "#B3403A", "#3A7D44"
BG = "#FBFAF7"

res = json.load(open(os.path.join(OUT, "eval_results.json"), encoding="utf-8"))
names = ["base", "sft", "dpo", "grpo"]
labels = ["base\n(pretrain)", "SFT", "DPO v2", "GRPO"]
M = res["models"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=BG)

ax = axes[0]
xpos = np.arange(len(names))
full = [M[n]["pubmedqa_full"]["acc"] for n in names]
clean = [M[n]["pubmedqa_clean"]["acc"] for n in names]
ax.bar(xpos - 0.18, full, 0.36, color=TEAL, label=f"full (n={M['base']['pubmedqa_full']['n']})")
ax.bar(xpos + 0.18, clean, 0.36, color="#7FB3B8", label=f"clean (n={M['base']['pubmedqa_clean']['n']})")
for i, (f_, c_) in enumerate(zip(full, clean)):
    ax.text(i - 0.18, f_, f"{f_:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + 0.18, c_, f"{c_:.3f}", ha="center", va="bottom", fontsize=8)
ax.axhline(res["baselines"]["majority_acc"], color=ORANGE, ls="--", lw=1.2,
           label=f"majority \"yes\" {res['baselines']['majority_acc']:.3f}")
ax.axhline(1 / 3, color=MUT, ls=":", lw=1.2, label="chance 0.333")
ax.set_xticks(xpos, labels)
ax.set(ylabel="accuracy", title="PubMedQA (likelihood scoring)")
ax.set_ylim(0, 0.85)
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axes[1]
betse = [M[n]["betse_reward"] for n in names]
ax.bar(xpos, betse, 0.5, color=GREEN)
for i, b in enumerate(betse):
    ax.text(i, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(xpos, labels)
ax.set(ylabel="mean verifier reward", title="BETSE physics (727 held-out combos)")
ax.set_ylim(0, 1.12)

ax = axes[2]
ppl = [M[n]["val_ppl"] for n in names]
ax.bar(xpos, ppl, 0.5, color=RED, alpha=0.85)
for i, p in enumerate(ppl):
    ax.text(i, p, f"{p:.2f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(xpos, labels)
ax.set(ylabel="perplexity (temporal holdout)", title="Language modeling — the alignment tax")
ax.set_ylim(0, max(ppl) * 1.18)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(alpha=0.25, axis="y", lw=0.5)
fig.suptitle("The post-training ladder, scored on exams it never set (seed 42)",
             fontsize=12, weight="bold", color=INK, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "eval_results.png"), dpi=130, facecolor=BG,
            bbox_inches="tight")
print("saved", os.path.join(OUT, "eval_results.png"))
for n in names:
    r = M[n]
    print(f"{n:5s} ppl {r['val_ppl']:6.2f} | QA full {r['pubmedqa_full']['acc']:.3f} "
          f"(F1 {r['pubmedqa_full']['macro_f1']:.3f}) clean {r['pubmedqa_clean']['acc']:.3f} "
          f"| BETSE {r['betse_reward']:.3f}")
