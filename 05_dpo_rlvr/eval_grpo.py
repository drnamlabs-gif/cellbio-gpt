# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 5b-eval: plot the GRPO run and the warm-start -> GRPO comparison.

Left: train-batch reward + combo-held-out val reward vs step (from grpo_log.jsonl).
Right: val reward by problem kind, warm-start baseline vs final GRPO policy.
Saves out/grpo_results.png.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEAL, ORANGE, RED, GREEN = "#0E7C86", "#C97B1E", "#B3403A", "#3A7D44"

steps, rews, vsteps, vrews = [], [], [], []
baseline, final = None, None
with open(os.path.join(OUT, "grpo_log.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("baseline"):
            baseline = r
        elif r.get("final"):
            final = r
            vsteps.append(r["step"]); vrews.append(r["val_reward"])
        elif "val_reward" in r:
            vsteps.append(r["step"]); vrews.append(r["val_reward"])
        elif "reward" in r:
            steps.append(r["step"]); rews.append(r["reward"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor="#FBFAF7")
ax = axes[0]
ax.plot(steps, rews, color=TEAL, lw=1, alpha=0.45, label="train batch reward (temp 1.0)")
if len(rews) > 9:
    k = np.ones(9) / 9
    sm = np.convolve(rews, k, mode="valid")
    ax.plot(steps[4:4 + len(sm)], sm, color=TEAL, lw=2)
ax.plot(vsteps, vrews, color=ORANGE, lw=2, marker="o", ms=4,
        label="val reward (held-out combos, greedy)")
ax.axhline(baseline["val_reward"], color=RED, ls="--", lw=1.2,
           label=f"warm-start baseline {baseline['val_reward']:.3f}")
ax.set(xlabel="GRPO step", ylabel="mean verifier reward",
       title="GRPO on the bioelectric verifier (seed 42)")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.grid(alpha=0.25, lw=0.5)

ax = axes[1]
kinds = ["nernst", "deltaE", "polarity"]
b = [baseline["by_kind"][k] for k in kinds]
g = [final["by_kind"][k] for k in kinds]
xpos = np.arange(len(kinds))
ax.bar(xpos - 0.18, b, 0.36, color=RED, label="warm-start")
ax.bar(xpos + 0.18, g, 0.36, color=GREEN, label="after GRPO")
for i, (bb, gg) in enumerate(zip(b, g)):
    ax.text(i - 0.18, bb, f"{bb:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + 0.18, gg, f"{gg:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(xpos, ["nernst\n(numeric mV)", "deltaE\n(incr/decr)", "polarity\n(de/hyper)"])
ax.set(ylabel="val reward by kind", title="Held-out combos: before vs after RL")
ax.set_ylim(0, 1.12)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25, axis="y", lw=0.5)
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "grpo_results.png"), dpi=130, facecolor="#FBFAF7")
print("saved", os.path.join(OUT, "grpo_results.png"))
print("baseline:", baseline["by_kind"])
print("final:   ", final["by_kind"])
