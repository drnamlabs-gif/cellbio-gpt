# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The two-exam figure (paper Fig 3): exam A = lattice holdout (easy, every
test point has a trained neighbor), exam B = off-grid (sealed, decimals,
no node nearby). Lattice training aces A and collapses on B; diverse random
training recovers most of B. Floors included. Saves out/two_exams.png."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
INK, MUT, BG = "#1a2b32", "#5c6f76", "#FBFAF7"
TEAL, ORANGE, RED, GREEN, PURPLE = "#0E7C86", "#C97B1E", "#B3403A", "#3A7D44", "#6B5B95"


def j(name):
    return json.load(open(os.path.join(OUT, name), encoding="utf-8"))


bA = j("betse_baselines.json")["results"]
oA = j("ollama_baselines.json")
bB = j("betse_baselines_offgrid.json")["results"]
c2 = j("case2_offgrid.json")["results"]

best5 = max((k for k in oA if k.endswith("5-shot")), key=lambda k: oA[k]["val_reward"])
best0 = max((k for k in oA if k.endswith("0-shot")), key=lambda k: oA[k]["val_reward"])

c3 = j("case3_offgrid.json")["results"]

examA = [
    ("constant floor", bA["constant (train mode)"]["val_reward"], MUT),
    ("own ladder zero-shot (best)", bA["+ GRPO textbook (5b)"]["val_reward"], RED),
    (f"best pretrained 0-shot\n({best0.split()[0]})", oA[best0]["val_reward"], ORANGE),
    (f"best pretrained 5-shot\n({best5.split()[0]})", oA[best5]["val_reward"], ORANGE),
    ("40M lattice-trained +GRPO", bA["+ GRPO vs simulator (8b)"]["val_reward"], TEAL),
]
examB = [
    ("constant floor", bB["constant (train mode)"]["val_reward"], MUT),
    ("40M lattice-trained (case 1)", bB["+ GRPO vs simulator (8b)"]["val_reward"], RED),
    ("40M random-trained (case 2)", c2["case2 +GRPO vs simulator"]["val_reward"], GREEN),
    ("40M random-trained, 5-shot\n(case 3, mixed-k SFT)", c3["case3 ws (mixed-k)"]["k5"]["val_reward"], TEAL),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=BG)
for ax, rows, title, note in [
        (axes[0], examA, "Exam A — lattice holdout (98 problems)",
         "every test point has an adjacent trained node"),
        (axes[1], examB, "Exam B — OFF-GRID, sealed (234 problems)",
         "decimal knobs between nodes; nothing within 0.25 log2")]:
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    bars = ax.barh(range(len(rows)), vals, color=cols, height=0.62)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    for b, v in zip(bars, vals):
        ax.text(min(v + 0.015, 0.96), b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8.5, color=INK)
    ax.set_title(f"{title}\n{note}", fontsize=10, color=INK)
    ax.set_xlabel("held-out reward (same verifier)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
fig.suptitle("Exam design decides what a score means — lattice training collapses off-grid; "
             "diverse data + 5 worked examples in context recover it",
             fontsize=11.5, weight="bold", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "two_exams.png"), dpi=130, facecolor=BG,
            bbox_inches="tight")
print("saved", os.path.join(OUT, "two_exams.png"))
