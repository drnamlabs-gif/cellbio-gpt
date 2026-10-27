# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 6e figure: why the ladder never answers "no" on PubMedQA.

Left: the smoking gun — the model's PRIOR logprob of each option before any
question is shown ("no" is ~6.6 nats less likely than "yes").
Right: accuracy under raw scoring vs prior-calibrated vs context-removed —
calibration frees the "no" answer but accuracy FALLS toward chance, and
deleting the abstract barely moves the raw score: the prior was doing the
scoring, not the evidence. Saves out/pubmedqa_diagnosis.png.
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
TEAL, ORANGE, RED, GREEN, PURPLE = "#0E7C86", "#C97B1E", "#B3403A", "#3A7D44", "#6B5B95"
BG = "#FBFAF7"

rep = json.load(open(os.path.join(OUT, "pubmedqa_diagnosis.json"), encoding="utf-8"))
names = ["base", "sft", "dpo", "grpo"]
labels = ["base", "SFT", "DPO v2", "GRPO"]
OPTS = ["yes", "no", "maybe"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=BG)

ax = axes[0]
xpos = np.arange(len(names))
w = 0.26
cols = {"yes": TEAL, "no": RED, "maybe": ORANGE}
for k, o in enumerate(OPTS):
    v = [rep["models"][n]["prior_mean_logprob"][o] for n in names]
    ax.bar(xpos + (k - 1) * w, v, w, color=cols[o], label=f'"{o}"')
    for i, vv in enumerate(v):
        ax.text(i + (k - 1) * w, vv - 0.25, f"{vv:.1f}", ha="center", va="top",
                fontsize=7.5, color=INK)
ax.set_xticks(xpos, labels)
ax.set(ylabel="mean token logprob of option\n(empty prompt — PRIOR only)",
       title='The smoking gun: "no" is ~7 nats less likely\nbefore the model sees any question')
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.invert_yaxis()  # more likely = higher bar visually

ax = axes[1]
modes = [("raw (as in harness)", TEAL, "raw"),
         ("calibrated (minus prior)", PURPLE, "calibrated (PMI)"),
         ("no-context raw", "#7FB3B8", "abstract removed")]
for k, (mode, c, lab) in enumerate(modes):
    v = []
    for n in names:
        r = [x for x in rep["models"][n]["results"] if x["scoring"] == mode][0]
        v.append(r["acc_full"])
    ax.bar(xpos + (k - 1) * 0.26, v, 0.26, color=c, label=lab)
    for i, vv in enumerate(v):
        ax.text(i + (k - 1) * 0.26, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=7.5)
ax.axhline(0.552, color=ORANGE, ls="--", lw=1.2, label='majority "yes" 0.552')
ax.axhline(1 / 3, color=MUT, ls=":", lw=1.2, label="chance 0.333")
ax.set_xticks(xpos, labels)
ax.set(ylabel="PubMedQA accuracy", title="Unbias it and the score FALLS to chance;\ndelete the abstract and raw barely moves")
ax.set_ylim(0, 0.75)
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(alpha=0.25, axis="y", lw=0.5)
fig.suptitle("Root cause: the prior was taking the exam, not the knowledge (seed 42)",
             fontsize=12, weight="bold", color=INK, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pubmedqa_diagnosis.png"), dpi=130, facecolor=BG,
            bbox_inches="tight")
print("saved", os.path.join(OUT, "pubmedqa_diagnosis.png"))
