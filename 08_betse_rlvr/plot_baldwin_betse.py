# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8c figure: Baldwin on the real-sim problems — 3 arms vs the
single-format stage-8b reference. Saves out/baldwin_betse_results.png."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
INK, MUT = "#1a2b32", "#5c6f76"
TEAL, ORANGE, RED, GREEN, PURPLE = "#0E7C86", "#C97B1E", "#B3403A", "#3A7D44", "#6B5B95"
BG = "#FBFAF7"
ARMS = [("k0", MUT, "k=0 fixed (direct control)"),
        ("anneal", GREEN, "anneal k 5->0 (Baldwin)"),
        ("k5", PURPLE, "k=5 fixed (crutch control)")]


def load_arm(arm):
    steps, vr, num, gap = [], [], [], []
    with open(os.path.join(OUT, f"bw8_log_{arm}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "val_reward" in r:
                steps.append(max(r["step"], 0) if not r.get("baseline") else 0)
                vr.append(r["val_reward"])
                num.append(r["by_kind"]["nernst"])
                gap.append(r["sens"]["gap"])
    return steps, vr, num, gap


ref = {}
with open(os.path.join(OUT, "bg8_log.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("final"):
            ref = r

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), facecolor=BG)
for arm, c, lab in ARMS:
    steps, vr, num, gap = load_arm(arm)
    axes[0].plot(steps, vr, color=c, lw=2, marker="o", ms=3.5, label=lab)
    axes[1].plot(steps, num, color=c, lw=2, marker="o", ms=3.5, label=lab)
    axes[2].plot(steps, gap, color=c, lw=2, marker="o", ms=3.5, label=lab)

axes[0].axhline(ref["val_reward"], color=ORANGE, ls="--", lw=1.3,
                label=f"8b single-format ref {ref['val_reward']:.3f}")
axes[0].set(xlabel="GRPO step", ylabel="held-out val reward (k=0)",
            title="Assimilation test (scaffold removed)")
axes[1].axhline(ref["by_kind"]["nernst"], color=ORANGE, ls="--", lw=1.3,
                label=f"8b ref {ref['by_kind']['nernst']:.3f}")
axes[1].set(xlabel="GRPO step", ylabel="held-out NUMERIC reward (k=0)",
            title="Numeric mean-Vmem prediction")
axes[2].axhline(0, color=MUT, ls=":", lw=1)
axes[2].set(xlabel="GRPO step", ylabel="reward(k=5) − reward(k=0)",
            title="Context sensitivity (98 val problems)")
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(alpha=0.25, lw=0.5)
    a.legend(frameon=False, fontsize=7.5)
fig.suptitle("Baldwinian GRPO on REAL-simulator problems — shared mixed-k warm-start, seed 42",
             fontsize=11.5, weight="bold", color=INK, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "baldwin_betse_results.png"), dpi=130, facecolor=BG,
            bbox_inches="tight")
print("saved", os.path.join(OUT, "baldwin_betse_results.png"))
for arm, _, _ in ARMS:
    steps, vr, num, gap = load_arm(arm)
    print(f"{arm:7s} val {vr[0]:.3f}->{vr[-1]:.3f} | numeric {num[0]:.3f}->{num[-1]:.3f} "
          f"(max {max(num):.3f}) | gap {gap[0]:+.3f}->{gap[-1]:+.3f}")
