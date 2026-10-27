# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Plot train/val loss curves from out/log.jsonl -> out/loss_curve.png."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ACCENT, VOLT = "#0E7C86", "#C97B1E"

steps, losses, vsteps, vlosses = [], [], [], []
with open(os.path.join(OUT, "log.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if "val_loss" in r:
            vsteps.append(r["step"]); vlosses.append(r["val_loss"])
        elif "loss" in r:
            steps.append(r["step"]); losses.append(r["loss"])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(steps, losses, color=ACCENT, lw=1.2, alpha=0.5, label="train (per-batch)")
ax.plot(vsteps, vlosses, color=VOLT, lw=2, marker="o", ms=3, label="val (9.2M tok holdout)")
if vlosses:
    ax.annotate(f"final val {vlosses[-1]:.3f}", (vsteps[-1], vlosses[-1]),
                xytext=(-90, -16), textcoords="offset points", color=VOLT)
ax.set(xlabel="step", ylabel="cross-entropy loss",
       title="Pretraining 40M params on 0.99B cell-biology tokens (seed 42)")
ax.legend(frameon=False)
ax.grid(alpha=0.25, lw=0.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "loss_curve.png"), dpi=120)
print("saved", os.path.join(OUT, "loss_curve.png"))
