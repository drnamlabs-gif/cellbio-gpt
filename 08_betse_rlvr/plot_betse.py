# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8 results figure: the simulator's truth landscape + the RLVR run.

Left:   what 150 real BETSE sims say — mean tissue Vmem vs bath K+, one line
        per Dm_K scale (Dm_Na x1 slice): the function the model must learn.
Middle: GRPO train reward + held-out-combo val reward vs step.
Right:  val reward by kind, warm-start vs after GRPO.
Saves out/betse_results.png.
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

truths = [json.loads(l) for l in open(os.path.join(OUT, "sim_truths.jsonl"),
                                      encoding="utf-8")]
steps, rews, vsteps, vrews = [], [], [], []
baseline, final = None, None
with open(os.path.join(OUT, "bg8_log.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("baseline"):
            baseline = r
        elif "val_reward" in r:
            vsteps.append(max(r["step"], 0)); vrews.append(r["val_reward"])
            if r.get("final"):
                final = r
        elif "reward" in r:
            steps.append(r["step"]); rews.append(r["reward"])

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), facecolor=BG)

ax = axes[0]
dmks = sorted({t["combo"][1] for t in truths})
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(dmks)))
for c, dk in zip(cmap, dmks):
    pts = sorted((t["combo"][0], t["mean_vmem_mV"]) for t in truths
                 if t["combo"][1] == dk and t["combo"][2] == 1)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
            lw=1.6, color=c, label=f"Dm_K x{dk}")
ax.set(xscale="log", xlabel="bath K+ (mM, log)", ylabel="mean tissue Vmem (mV)",
       title="The verifier: 150 real BETSE sims\n(Dm_Na x1 slice shown)")
ax.legend(frameon=False, fontsize=7.5)

ax = axes[1]
ax.plot(steps, rews, color=TEAL, lw=1, alpha=0.5, label="train batch reward")
if len(rews) > 7:
    k = np.ones(7) / 7
    sm = np.convolve(rews, k, mode="valid")
    ax.plot(steps[3:3 + len(sm)], sm, color=TEAL, lw=2)
ax.plot(vsteps, vrews, color=ORANGE, lw=2, marker="o", ms=4,
        label="val (held-out combos, greedy)")
ax.axhline(baseline["val_reward"], color=RED, ls="--", lw=1.2,
           label=f"warm-start {baseline['val_reward']:.3f}")
ax.set(xlabel="GRPO step", ylabel="mean verifier reward",
       title="GRPO against the simulator's truths")
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
kinds = ["nernst", "polarity"]
lbl = ["mean Vmem\n(numeric mV)", "de/hyperpolarized\nvs baseline"]
b = [baseline["by_kind"][k] for k in kinds]
g = [final["by_kind"][k] for k in kinds]
xp = np.arange(2)
ax.bar(xp - 0.18, b, 0.36, color=RED, label="warm-start")
ax.bar(xp + 0.18, g, 0.36, color=GREEN, label="after GRPO")
for i, (bb, gg) in enumerate(zip(b, g)):
    ax.text(i - 0.18, bb, f"{bb:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + 0.18, gg, f"{gg:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(xp, lbl)
ax.set(ylabel="val reward by kind", title="Held-out combos: before vs after RL")
ax.set_ylim(0, 1.12)
ax.legend(frameon=False, fontsize=9)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(alpha=0.25, lw=0.5)
fig.suptitle("RLVR with the REAL BETSE simulator as verifier (seed 42)",
             fontsize=12, weight="bold", color=INK, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "betse_results.png"), dpi=130, facecolor=BG,
            bbox_inches="tight")
print("saved", os.path.join(OUT, "betse_results.png"))
print("baseline:", baseline["by_kind"], "\nfinal:   ", final["by_kind"])
