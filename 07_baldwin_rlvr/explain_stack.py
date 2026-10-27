# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Explainer figure: the Adaptation Stack view of LLM training, and the
Baldwinian training algorithm it suggests.

Panels:
  A. the three timescales of biological adaptation and their LLM analogs
     (evolution=pretraining, lifetime learning=ICL, culture=SFT/distillation)
     — with our pipeline's stages placed, and the missing layer flagged
  B. the stage-6 diagnosis re-read in stack terms: an instinct-only creature
  C. the Baldwin effect: lifetime learning smooths the fitness landscape
     (Hinton & Nowlan 1987), then assimilation bakes the behavior in
  D. the proposed algorithm: phenotype-first RLVR (Baldwinian GRPO)
Saves out/explain_stack.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
INK, MUT = "#1a2b32", "#5c6f76"
TEAL, ORANGE, RED, GREEN, PURPLE = "#0E7C86", "#C97B1E", "#B3403A", "#3A7D44", "#6B5B95"
BG = "#FBFAF7"

fig = plt.figure(figsize=(13, 14), facecolor=BG)


def box(ax, x, y, w, h, fc, ec, text, fs=9, tc=None, weight="normal", pad=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc or INK, weight=weight)


# ---------- Panel A: the stack ----------
axA = fig.add_axes([0.03, 0.72, 0.94, 0.25]); axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
axA.text(0, 9.6, "A — The Adaptation Stack: three timescales, one LLM mapping",
         fontsize=12.5, weight="bold", color=INK)
rows = [
    (6.4, "#EEF3F4", TEAL, "EVOLUTION", "population, slowest\ninherited: instinct in the GENOME",
     "= pretraining (weights)\nour stage 3: 988M tokens -> 40M params"),
    (3.9, "#F7EDED", RED, "LIFETIME LEARNING", "one individual, fastest\nnothing inherited: lives in the BRAIN'S STATE",
     "= in-context learning / test-time adaptation\nour repo: ABSENT at 40M  <-- the hole"),
    (1.4, "#F4EFE6", ORANGE, "CULTURE", "between individuals, medium\ninherited: transmitted BEHAVIOR/DATA",
     "= SFT on demonstrations, distillation, preferences\nour stages 4 + 5a (mined demos, DPO pairs)"),
]
for y, fc, ec, name, bio, llm in rows:
    box(axA, 0.15, y, 2.1, 2.1, fc, ec, name, fs=10, weight="bold")
    box(axA, 2.45, y, 3.55, 2.1, "#FFFFFF", ec, bio, fs=8.5)
    box(axA, 6.2, y, 3.6, 2.1, fc, ec, llm, fs=8.5)
axA.text(0.15, 0.55, "GRPO (our 5b) is the odd one out: trial-and-error DURING the episode, but credit is written straight into the weights\n"
                     "— a Lamarckian shortcut biology does not have. Nothing in our pipeline trains the middle layer.",
         fontsize=9.5, color=INK, weight="bold", va="top")

# ---------- Panel B: diagnosis re-read ----------
axB = fig.add_axes([0.03, 0.555, 0.94, 0.13]); axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
axB.text(0, 9.2, "B — Stage-6 diagnosis, re-read: we built an instinct-only creature",
         fontsize=12.5, weight="bold", color=INK)
box(axB, 0.15, 1.2, 4.6, 6.2, "#F7EDED", RED,
    "PubMedQA asks: read THIS abstract, weigh the\nevidence, answer. That is a within-lifetime task.\n"
    "Measured: deleting the abstract moved accuracy\nby ~2 points — the middle layer is empty.", fs=9)
box(axB, 5.05, 1.2, 4.75, 6.2, "#EEF3F4", TEAL,
    "\"The prior took the exam\" = instinct answered.\nSame story for ln(): direct-to-genome learning\n"
    "can only memorize combos (train 0.87, held-out\n0.60). No plasticity -> no landscape smoothing.", fs=9)

# ---------- Panel C: Baldwin effect ----------
axC = fig.add_axes([0.03, 0.31, 0.94, 0.215]); axC.set_xlim(0, 10); axC.set_ylim(0, 10); axC.axis("off")
axC.text(0, 9.5, "C — The Baldwin effect: learning smooths the landscape, then selection assimilates",
         fontsize=12.5, weight="bold", color=INK)
# two mini landscape plots
for k, (x0, title, smooth) in enumerate([
        (0.085, "without lifetime learning", False),
        (0.55, "with lifetime learning", True)]):
    ax = fig.add_axes([x0, 0.365, 0.36, 0.10], facecolor="#FFFFFF")
    g = np.linspace(-3, 3, 400)
    if smooth:
        f = np.exp(-g ** 2 / 1.6)
        ax.plot(g, f, color=GREEN, lw=2)
        ax.annotate("selection can climb", xy=(-1.25, 0.42), xytext=(-2.95, 0.78),
                    fontsize=8, color=GREEN,
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    else:
        f = np.where(np.abs(g) < 0.06, 1.0, 0.02)
        ax.plot(g, f, color=RED, lw=2)
        ax.text(0.5, 0.78, "needle in a haystack:\nreward only at the exact\nsolution, no gradient",
                fontsize=8, color=RED)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("genotype (weights)", fontsize=8)
    ax.set_ylabel("reward", fontsize=8)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
axC.text(0.1, 1.5, "Hinton & Nowlan 1987: genotypes that can LEARN the last steps within a lifetime turn a spike into a slope.\n"
                   "Then genetic assimilation: once the slope is climbable, selection moves the knowledge from plastic to innate — no Lamarck needed.",
         fontsize=9.5, color=INK, va="top")

# ---------- Panel D: the algorithm ----------
axD = fig.add_axes([0.03, 0.02, 0.94, 0.26]); axD.set_xlim(0, 10); axD.set_ylim(0, 10); axD.axis("off")
axD.text(0, 9.6, "D — Proposed: phenotype-first RLVR (Baldwinian GRPO) — stage 7, runnable on this repo",
         fontsize=12.5, weight="bold", color=INK)
box(axD, 0.15, 6.2, 2.9, 2.6, "#EEF3F4", TEAL,
    "PLASTICITY CHANNEL\nprompt = problem + k worked\nexamples of NEIGHBOR combos\n(the lifetime-learning scaffold)", fs=8.5)
box(axD, 3.35, 6.2, 2.9, 2.6, "#E8E6E0", PURPLE,
    "model answers\n(K samples, as in GRPO)", fs=9)
box(axD, 6.55, 6.2, 3.25, 2.6, "#EDF3EE", GREEN,
    "verifier reward + GRPO update\n= selection on the\nPOST-LEARNING phenotype", fs=8.5)
axD.annotate("", xy=(3.3, 7.5), xytext=(3.1, 7.5), arrowprops=dict(arrowstyle="->", color=MUT))
axD.annotate("", xy=(6.5, 7.5), xytext=(6.3, 7.5), arrowprops=dict(arrowstyle="->", color=MUT))
box(axD, 0.15, 3.1, 9.65, 2.3, "#F4EFE6", ORANGE,
    "ASSIMILATION SCHEDULE: anneal k from 5 to 0 over training — the scaffold is slowly removed,\n"
    "pressuring the circuit to move from \"interpolate from the examples in front of you\" to \"know it innately\".", fs=9.5)
axD.text(0.15, 2.3, "Predictions (falsifiable, seed 42):  P1 — final k=0 held-out nernst beats direct GRPO's 0.60 ceiling.  "
                    "P2 — context-sensitivity rises above 0\n(the model starts READING).  "
                    "Controls: k fixed at 0 (= stage 5b, already run) and k fixed at 5 (crutch never removed).  Cost: minutes on the 5090.",
         fontsize=9, color=INK, va="top")
axD.text(0.15, 0.55, "Cultural rung, same machinery: verified-correct traces become SFT data for the next generation (distillation = culture).",
         fontsize=9, color=MUT, va="top")

fig.savefig(os.path.join(OUT, "explain_stack.png"), dpi=130, facecolor=BG)
print("saved", os.path.join(OUT, "explain_stack.png"))
