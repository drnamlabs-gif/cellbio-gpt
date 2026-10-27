# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Explainer figure: RLVR with GRPO — reward from a verifier, not from taste.

Panels:
  A. the RLVR idea: ground truth comes from PHYSICS (Nernst — the equation
     BETSE integrates); the model only ever receives a scalar
  B. one GRPO step, drawn concretely: K samples -> rewards -> group-relative
     advantages -> push up / push down; no critic, no reward model
  C. two practical truths: warm-start or no gradient; uniform groups teach
     nothing (gradient lives where outcomes DIFFER)
Saves out/explain_grpo.png.
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


# Panel A
axA = fig.add_axes([0.04, 0.70, 0.92, 0.26]); axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
axA.text(0, 9.5, "A — RLVR: the reward is VERIFIED, not preferred",
         fontsize=12, weight="bold", color=INK)
box(axA, 0.2, 5.4, 2.9, 2.6, "#EEF3F4", TEAL,
    "Problem generator\n(ion, [in], [out] combos)\n\"Compute E_K at 37 C,\n"
    "[K+]in=140, [K+]out=5\"", fs=8.5)
box(axA, 3.6, 5.4, 2.9, 2.6, "#E8E6E0", PURPLE,
    "model's answer\n\"E_K = -89 mV.\"", fs=9)
box(axA, 7.0, 5.4, 2.8, 2.6, "#EDF3EE", GREEN,
    "VERIFIER (physics):\nE = (RT/zF)·ln(out/in)\n= −89.1 mV → reward 1.0\n"
    "(±2 full, ±10/±25 partial)", fs=8.5)
axA.annotate("", xy=(3.5, 6.7), xytext=(3.15, 6.7), arrowprops=dict(arrowstyle="->", color=MUT))
axA.annotate("", xy=(6.9, 6.7), xytext=(6.55, 6.7), arrowprops=dict(arrowstyle="->", color=MUT))
axA.text(0.2, 4.2, "The Nernst equation is the core membrane physics BETSE integrates — the verifier IS the simulator's law,\n"
                   "evaluated directly. DPO ranked frozen pairs; RL explores the model's OWN samples and gets told which were right.",
         fontsize=9, color=MUT)
axA.text(0.2, 2.2, "Same recipe as frontier math/code RLVR (DeepSeek-R1, o-series): generator + checker, no human labels.",
         fontsize=9.5, color=INK, weight="bold")

# Panel B
axB = fig.add_axes([0.04, 0.30, 0.92, 0.36]); axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
axB.text(0, 9.6, "B — One GRPO step: the group IS the baseline",
         fontsize=12, weight="bold", color=INK)
box(axB, 0.2, 6.8, 2.2, 1.8, "#EEF3F4", TEAL, "prompt x\n(1 of 16\nper step)", fs=8.5)
rewards = [1.0, 0.6, 1.0, 0.05, 0.6, 1.0, 0.3, 0.6]
mean_r = sum(rewards) / len(rewards)
for i, r in enumerate(rewards):
    xx = 3.0 + i * 0.85
    c = GREEN if r > mean_r else RED
    box(axB, xx, 7.3, 0.72, 1.0, "#FFFFFF", c, f"{r:.2f}", fs=8)
    arrow = "↑" if r > mean_r else "↓"
    axB.text(xx + 0.36, 6.8, arrow, ha="center", fontsize=12, color=c, weight="bold")
axB.text(3.0, 9.0, f"K = 8 samples (temp 1.0), verifier rewards; group mean = {mean_r:.2f}",
         fontsize=9, color=MUT)
box(axB, 0.2, 3.6, 9.6, 2.2, "#F4EFE6", ORANGE,
    r"$A_i = \dfrac{r_i - \mathrm{mean}(r_{group})}{\mathrm{std}(r_{group})}\qquad"
    r"L = -\dfrac{1}{|T|}\sum_{i,t} A_i\,\log\pi_\theta(y_{i,t}|x)\;+\;"
    r"\lambda_{KL}\,(\log\pi_\theta - \log\pi_{ref})$", fs=12)
axB.text(0.3, 2.6, "Above-average answers get pushed UP, below-average DOWN — the baseline is free (the group itself),",
         fontsize=9.5, color=INK)
axB.text(0.3, 1.8, "so there is no critic network and no learned reward model (PPO's two extra models deleted — that is GRPO).",
         fontsize=9.5, color=INK)
axB.text(0.3, 0.7, "λ_KL = 0.05 leashes the policy to the warm-start reference — reward hacking by gibberish gets paid in KL.",
         fontsize=9, color=MUT)

# Panel C
axC = fig.add_axes([0.04, 0.02, 0.92, 0.24]); axC.set_xlim(0, 10); axC.set_ylim(0, 10); axC.axis("off")
axC.text(0, 9.4, "C — Two practical truths (both bit us, both are in the log)",
         fontsize=12, weight="bold", color=INK)
box(axC, 0.2, 3.4, 4.6, 4.8, "#F7EDED", RED,
    "No warm-start → no gradient:\na model that scores 0 on every sample\n"
    "has zero group variance everywhere.\n400 SFT steps on worked examples\n"
    "first, then RL has a slope to climb.", fs=9)
box(axC, 5.2, 3.4, 4.6, 4.8, "#EEF3F4", TEAL,
    "Uniform groups are skipped:\nwhere warm-start already solves the task\n"
    "(sign rules: 100% on held-out combos),\nall 8 rewards tie → advantage 0.\n"
    "So GRPO oversamples the NUMERIC task\n(60%) — gradient goes where headroom is.", fs=9)

fig.savefig(os.path.join(OUT, "explain_grpo.png"), dpi=130, facecolor=BG)
print("saved", os.path.join(OUT, "explain_grpo.png"))
