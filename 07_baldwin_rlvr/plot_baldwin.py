# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 7 results figure: Baldwin arm vs the two controls.

Left:   held-out val reward at k=0 (the assimilation test) vs step, 3 arms,
        with the stage-5b single-format reference (ws1 -> direct GRPO).
Middle: held-out NERNST (numeric) reward at k=0 vs step — the ceiling P1 targets.
Right:  context sensitivity (reward k=5 minus k=0 on the fixed 160 subset) —
        P2: does the model use, then internalize, the scaffold?
Saves out/baldwin_results.png.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
OUT5B = os.path.join(HERE, "..", "05_dpo_rlvr", "out")
INK, MUT = "#1a2b32", "#5c6f76"
TEAL, ORANGE, RED, GREEN, PURPLE = "#0E7C86", "#C97B1E", "#B3403A", "#3A7D44", "#6B5B95"
BG = "#FBFAF7"
ARMS = [("k0", MUT, "k=0 fixed (direct GRPO control)"),
        ("anneal", GREEN, "anneal k 5->0 (Baldwin arm)"),
        ("k5", PURPLE, "k=5 fixed (crutch never removed)")]


def load_arm(arm):
    steps, vr, nernst, gap = [], [], [], []
    with open(os.path.join(OUT, f"bg_log_{arm}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "val_reward" in r:
                steps.append(max(r["step"], 0) if not r.get("baseline") else 0)
                vr.append(r["val_reward"])
                nernst.append(r["by_kind"]["nernst"])
                gap.append(r["sens"]["gap"])
    return steps, vr, nernst, gap


# stage-5b reference (ws1, single-format, k==0)
ref_steps, ref_nernst, ref_vr = [], [], []
with open(os.path.join(OUT5B, "grpo_log.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if "val_reward" in r:
            ref_steps.append(max(r["step"], 0) if not r.get("baseline") else 0)
            ref_vr.append(r["val_reward"])
            ref_nernst.append(r["by_kind"]["nernst"])

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), facecolor=BG)

for arm, c, lab in ARMS:
    steps, vr, nernst, gap = load_arm(arm)
    axes[0].plot(steps, vr, color=c, lw=2, marker="o", ms=3.5, label=lab)
    axes[1].plot(steps, nernst, color=c, lw=2, marker="o", ms=3.5, label=lab)
    axes[2].plot(steps, gap, color=c, lw=2, marker="o", ms=3.5, label=lab)

axes[0].plot(ref_steps, ref_vr, color=ORANGE, lw=1.4, ls="--",
             label="stage 5b ref (ws1, single-format)")
axes[0].set(xlabel="GRPO step", ylabel="held-out val reward (k=0)",
            title="Assimilation test: innate skill, scaffold removed")

axes[1].plot(ref_steps, ref_nernst, color=ORANGE, lw=1.4, ls="--",
             label="stage 5b ref (ceiling 0.60)")
axes[1].axhline(0.60, color=ORANGE, ls=":", lw=1)
axes[1].set(xlabel="GRPO step", ylabel="held-out NERNST reward (k=0)",
            title="P1: the numeric ceiling")

axes[2].axhline(0.0, color=MUT, ls=":", lw=1)
axes[2].set(xlabel="GRPO step", ylabel="reward(k=5) − reward(k=0)",
            title="P2: context sensitivity (fixed 160-problem subset)")

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(alpha=0.25, lw=0.5)
    a.legend(frameon=False, fontsize=7.5)
fig.suptitle("Baldwinian GRPO vs controls — all arms start from the same mixed-format warm-start ws2 (seed 42)",
             fontsize=11.5, weight="bold", color=INK, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "baldwin_results.png"), dpi=130, facecolor=BG,
            bbox_inches="tight")
print("saved", os.path.join(OUT, "baldwin_results.png"))
for arm, _, _ in ARMS:
    steps, vr, nernst, gap = load_arm(arm)
    print(f"{arm:7s} val(k=0) {vr[0]:.3f}->{vr[-1]:.3f} | nernst {nernst[0]:.3f}->{nernst[-1]:.3f}"
          f" (max {max(nernst):.3f}) | gap {gap[0]:+.3f}->{gap[-1]:+.3f}")
