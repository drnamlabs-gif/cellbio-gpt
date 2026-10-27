# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Explainer figure: why SFT is not enough, and what DPO adds.

Three panels:
  A. SFT's blind spot: imitation has no concept of a WRONG answer
  B. the DPO mechanism: preference pair -> 4 log-probs -> sigmoid margin loss,
     with the frozen reference model as an anchor (beta = how far we may drift)
  C. where the pairs come from here (self-mined, on-policy) and what comes next
     (GRPO + BETSE verifier = rewards from a simulator instead of pairs)
Saves out/explain_dpo.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

INK, MUT = "#1a2b32", "#5c6f76"
TEAL, ORANGE, RED, GREEN, PURPLE = "#0E7C86", "#C97B1E", "#B3403A", "#3A7D44", "#6B5B95"
BG = "#FBFAF7"

fig = plt.figure(figsize=(13, 10.5), facecolor=BG)


def box(ax, x, y, w, h, fc, ec, text, fs=9, tc=None, weight="normal", pad=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc or INK, weight=weight)


# ---------------- Panel A ----------------
axA = fig.add_axes([0.04, 0.70, 0.92, 0.26]); axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
axA.text(0, 9.5, "A — SFT's blind spot: imitation never defines WRONG",
         fontsize=12, weight="bold", color=INK)
axA.text(0, 8.5, "SFT maximizes P(gold answer). It never pushes DOWN on anything — so the model's own bad habits survive:",
         fontsize=9.5, color=MUT)
box(axA, 0.2, 4.6, 4.5, 2.6, "#F7EDED", RED,
    'Observed after our SFT (stage 4):\n"ATP stands for ATP."  (circular)\n'
    '"...is a common feature of many cell\ntypes, and it is thought to be a\ncommon feature of many cell types."', fs=9)
box(axA, 5.3, 4.6, 4.5, 2.6, "#EEF3F4", TEAL,
    "Why SFT cannot fix this:\ncross-entropy on gold tokens gives\nzero gradient about outputs the\n"
    "model prefers but we never showed it.\nWe need a signal that RANKS outputs.", fs=9)
axA.text(0.2, 3.4, "DPO (Direct Preference Optimization): learn from pairs (better, worse) directly — no reward model,\n"
                   "no RL loop — just a new loss on the same forward pass.",
         fontsize=9.5, color=INK, weight="bold")

# ---------------- Panel B ----------------
axB = fig.add_axes([0.04, 0.30, 0.92, 0.36]); axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
axB.text(0, 9.6, "B — The DPO loss: four log-probs and a sigmoid",
         fontsize=12, weight="bold", color=INK)

box(axB, 0.2, 7.0, 2.2, 1.6, "#EEF3F4", TEAL, "prompt x\n\"What does\nGJ stand for?\"", fs=8.5)
box(axB, 2.9, 7.8, 3.0, 1.2, "#EDF3EE", GREEN, "chosen y⁺: \"GJ stands\nfor gap junction.\"", fs=8.5)
box(axB, 2.9, 6.4, 3.0, 1.2, "#F7EDED", RED, "rejected y⁻: \"GJ stands\nfor GJ.\"  (model's own)", fs=8.5)

box(axB, 6.4, 7.8, 1.6, 1.2, "#E8E6E0", PURPLE, "policy πθ\n(trained)", fs=8.5)
box(axB, 6.4, 6.4, 1.6, 1.2, "#E8E6E0", MUT, "ref πref\n(frozen SFT)", fs=8.5)
axB.text(8.3, 7.6, "→ logπθ(y⁺|x), logπθ(y⁻|x)\n→ logπref(y⁺|x), logπref(y⁻|x)",
         fontsize=8.5, color=INK, va="center")

box(axB, 0.2, 3.6, 9.6, 1.9, "#F4EFE6", ORANGE,
    r"$L_{DPO} = -\log\,\sigma\left(\beta\left[(\log\pi_\theta(y^+|x)-\log\pi_{ref}(y^+|x))-"
    r"(\log\pi_\theta(y^-|x)-\log\pi_{ref}(y^-|x))\right]\right)$", fs=12)
axB.text(0.3, 2.9, "Read it as: grow the margin \"how much MORE than the reference I prefer y⁺ over y⁻\".",
         fontsize=9.5, color=INK)
axB.text(0.3, 2.1, "β (we use 0.1) = leash length: small β lets the policy drift far from the SFT anchor; the ref term\n"
                   "stops reward hacking (without it the cheapest win is: assign y⁻ probability zero — and fluency dies with it).",
         fontsize=9, color=MUT)
axB.text(0.3, 0.7, "Log-probs are summed ONLY over answer tokens (same mask trick as SFT; prompt tokens contribute nothing).",
         fontsize=9, color=MUT)

# ---------------- Panel C ----------------
axC = fig.add_axes([0.04, 0.02, 0.92, 0.24]); axC.set_xlim(0, 10); axC.set_ylim(0, 10); axC.axis("off")
axC.text(0, 9.4, "C — Our pairs (5a, today) and the simulator step (5b, next)",
         fontsize=12, weight="bold", color=INK)
box(axC, 0.2, 3.6, 4.5, 4.6, "#EDF3EE", GREEN,
    "5a  DPO pairs, self-mined (no external LLM):\n"
    "chosen = gold answer from the corpus\n"
    "rejected = the SFT model's OWN sample\n"
    "(on-policy: punishes real habits), plus\n"
    "hard negatives: circular \"X stands for X\"", fs=9)
box(axC, 5.3, 3.6, 4.5, 4.6, "#EFEAF4", PURPLE,
    "5b  GRPO + BETSE (the moat):\n"
    "sample K answers per prompt, score each\n"
    "with the BETSE bioelectric simulator\n"
    "(verifiable reward), advantage = r − mean(r)\n"
    "within the group → policy gradient", fs=9)
axC.text(0.2, 2.4, "Same ladder the frontier labs climb: imitate (SFT) → rank (DPO) → verify (RLVR). Ours just fits on one 5090.",
         fontsize=9.5, color=MUT)

fig.savefig(os.path.join(OUT, "explain_dpo.png"), dpi=130, facecolor=BG)
print("saved", os.path.join(OUT, "explain_dpo.png"))
