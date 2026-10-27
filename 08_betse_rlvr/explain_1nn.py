# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Explainer: what the 1-NN knob-lookup baseline is (no LLM, no learning).

For every exam problem, find the TRAINING combo closest in knob space
(log2 distance on K+, Dm_K, Dm_Na) and copy its sim-measured answer verbatim.
Scores 0.927 because the response surface is smooth: a near neighbor's Vmem
is almost always within the reward bands. Saves out/explain_1nn.png.
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
INK, MUT, BG = "#1a2b32", "#5c6f76", "#FBFAF7"
TEAL, ORANGE, RED, GREEN = "#0E7C86", "#C97B1E", "#B3403A", "#3A7D44"


def load(name):
    seen = set()
    out = []
    for l in open(os.path.join(OUT, name), encoding="utf-8"):
        p = json.loads(l)
        if p["kind"] == "nernst" and tuple(p["combo"]) not in seen:
            seen.add(tuple(p["combo"]))
            out.append(p)
    return out


def dist(a, b):
    return sum((math.log2(x) - math.log2(y)) ** 2 for x, y in zip(a, b))


train, val = load("betse_train.jsonl"), load("betse_val.jsonl")
# pick a val point with a clean story: nearest neighbor differs only in one knob
q = min(val, key=lambda p: dist(p["combo"],
                                min(train, key=lambda t: dist(t["combo"], p["combo"]))["combo"]))
nn = min(train, key=lambda t: dist(t["combo"], q["combo"]))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor=BG,
                         gridspec_kw={"width_ratios": [1.25, 1]})

# ---- left: the knob map (2 of the 3 knobs), truth as color ----
ax = axes[0]
tr_same_dn = [p for p in train if p["combo"][2] == q["combo"][2]]
xs = [p["combo"][0] for p in tr_same_dn]
ys = [p["combo"][1] for p in tr_same_dn]
cs = [p["truth"]["value"] for p in tr_same_dn]
sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=120, vmin=-75, vmax=-10,
                edgecolors="white", lw=1.2, zorder=3, label="train combos (sim already run)")
ax.scatter([q["combo"][0]], [q["combo"][1]], marker="*", s=420, c=RED,
           edgecolors="white", lw=1.2, zorder=5, label="exam combo (never simulated for the taker)")
ax.annotate("", xy=(nn["combo"][0], nn["combo"][1]),
            xytext=(q["combo"][0], q["combo"][1]),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2), zorder=4)
ax.annotate(f"nearest neighbor\ntruth = {nn['truth']['value']:+d} mV\n→ COPY it",
            xy=(nn["combo"][0], nn["combo"][1]), xytext=(14, -22),
            textcoords="offset points", fontsize=8.5, color=RED, weight="bold")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("extracellular K+ (mM)  [knob 1]", fontsize=9)
ax.set_ylabel("Dm_K multiplier  [knob 2]", fontsize=9)
ax.set_title(f"The whole algorithm: find the closest dot, copy its answer\n"
             f"(slice shown at Dm_Na ×{q['combo'][2]}; distance = log2 deltas on all 3 knobs)",
             fontsize=9.5, color=INK)
cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("sim truth: mean Vmem (mV)", fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(alpha=0.2, which="both", lw=0.4)

# ---- right: why it works here / when it breaks ----
ax = axes[1]; ax.axis("off")
exam = (q["combo"][0], q["combo"][1], q["combo"][2])
near = (nn["combo"][0], nn["combo"][1], nn["combo"][2])
ax.text(0.02, 0.97, "This example, concretely:", fontsize=10.5, weight="bold",
        color=INK, va="top")
ax.text(0.02, 0.88,
        f"exam asks:   K+={exam[0]} mM, Dm_K ×{exam[1]}, Dm_Na ×{exam[2]}  → Vmem?\n"
        f"closest seen: K+={near[0]} mM, Dm_K ×{near[1]}, Dm_Na ×{near[2]}  → {nn['truth']['value']:+d} mV\n"
        f"1-NN answers: \"{nn['answer']}\"\n"
        f"true answer:  \"{q['answer']}\"   (truth {q['truth']['value']:+d} mV)",
        fontsize=9, family="monospace", color=INK, va="top")
ax.text(0.02, 0.56, "No model. No learning. No physics.\nJust a phone book of 150 finished sims.",
        fontsize=10.5, weight="bold", color=RED, va="top")
ax.text(0.02, 0.42,
        "Why it scores 0.927 here: the Vmem response surface is SMOOTH,\n"
        "so a neighbor's answer usually lands inside the ±2/±10 mV\n"
        "reward bands. Smooth landscape → lookup ≈ intelligence.",
        fontsize=9, color=MUT, va="top")
ax.text(0.02, 0.22,
        "What it CANNOT do (and the model can):\n"
        "• interpolate BETWEEN dots — model numeric 0.912 vs 1-NN 0.856\n"
        "• answer when the grid is sparse (neighbors far away)\n"
        "• read a reworded question — it never sees text at all",
        fontsize=9, color=TEAL, va="top")

fig.suptitle("The 1-NN lookup baseline: the dumbest possible rival — and why it must be reported",
             fontsize=12, weight="bold", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "explain_1nn.png"), dpi=130, facecolor=BG,
            bbox_inches="tight")
print("saved", os.path.join(OUT, "explain_1nn.png"))
print("exam", q["combo"], q["truth"], "| nn", nn["combo"], nn["truth"])
