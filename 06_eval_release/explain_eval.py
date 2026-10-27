# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Explainer figure: what an eval harness is and why release needs one.

Panels:
  A. training metrics are self-graded — a release is scored on exams the
     model never set: three axes (language, knowledge, physics)
  B. how a small LM takes a multiple-choice exam: likelihood scoring
     (compare mean token logprob of each option; no fragile answer parsing)
  C. contamination: PubMedQA is built FROM PubMed abstracts = our corpus;
     flag PMID overlap, always report the clean subset
Saves out/explain_eval.png.
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


# Panel A — three exam axes
axA = fig.add_axes([0.04, 0.68, 0.92, 0.28]); axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
axA.text(0, 9.5, "A — Release = exams the model never set for itself",
         fontsize=12, weight="bold", color=INK)
axA.text(0, 8.4, "Every training-time number so far (val loss, DPO accuracy, GRPO reward) was computed by the same code that\n"
                 "trained the model. An eval harness scores the frozen checkpoints on three independent axes:",
         fontsize=9, color=MUT, va="top")
box(axA, 0.2, 2.2, 2.9, 3.6, "#EEF3F4", TEAL,
    "LANGUAGE\nperplexity on the\ntemporal holdout\n(5 newest corpus slices,\nnever trained on)", fs=9)
box(axA, 3.55, 2.2, 2.9, 3.6, "#F4EFE6", ORANGE,
    "KNOWLEDGE\nPubMedQA (1k expert-\nlabeled yes/no/maybe\nbiomedical questions,\nexternal benchmark)", fs=9)
box(axA, 6.9, 2.2, 2.9, 3.6, "#EDF3EE", GREEN,
    "PHYSICS\nBETSE verifier on\n727 combo-held-out\nNernst problems\n(stage 5b's val set)", fs=9)
axA.text(0.2, 1.2, "Same three numbers for every rung of the ladder: base → SFT → DPO → GRPO. The ladder shows what each stage bought — and what it cost.",
         fontsize=9.5, color=INK, weight="bold")

# Panel B — likelihood scoring
axB = fig.add_axes([0.04, 0.32, 0.92, 0.32]); axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
axB.text(0, 9.6, "B — How a 40M model takes a multiple-choice exam: likelihood scoring",
         fontsize=12, weight="bold", color=INK)
box(axB, 0.2, 5.2, 3.4, 3.2, "#EEF3F4", TEAL,
    "prompt (SFT template)\n\"<abstract context>\nQuestion: does X affect Y?\nAnswer yes, no, or maybe.\"", fs=8.5)
opts = [("\"yes\"", "-0.9", GREEN, True), ("\"no\"", "-2.1", RED, False), ("\"maybe\"", "-1.6", RED, False)]
for i, (o, lp, c, win) in enumerate(opts):
    yy = 7.2 - i * 1.55
    box(axB, 4.4, yy, 1.7, 1.1, "#FFFFFF", c, o, fs=9, weight="bold" if win else "normal")
    axB.text(6.35, yy + 0.55, f"mean token logprob {lp}", fontsize=9, va="center",
             color=GREEN if win else MUT, weight="bold" if win else "normal")
    axB.annotate("", xy=(4.35, yy + 0.55), xytext=(3.75, 6.8),
                 arrowprops=dict(arrowstyle="->", color=MUT, lw=0.9))
axB.text(9.0, 6.85, "← argmax\n    = answer", fontsize=9.5, color=GREEN, weight="bold", va="center")
axB.text(0.2, 2.6, "The model is never asked to WRITE the answer — we measure which option it finds most probable as a continuation\n"
                   "(length-normalized). Robust for small models: no parsing of free-form text, chance level is an honest 1/3.",
         fontsize=9, color=MUT, va="top")
axB.text(0.2, 0.9, "Baselines to beat: chance 33.3% and majority-class (always \"yes\") 55.2% — a model below majority has learned no signal.",
         fontsize=9.5, color=INK, weight="bold")

# Panel C — contamination
axC = fig.add_axes([0.04, 0.02, 0.92, 0.26]); axC.set_xlim(0, 10); axC.set_ylim(0, 10); axC.axis("off")
axC.text(0, 9.4, "C — Contamination audit: the benchmark is made of our training data",
         fontsize=12, weight="bold", color=INK)
box(axC, 0.2, 3.2, 4.6, 5.0, "#F7EDED", RED,
    "PubMedQA questions are generated FROM\nPubMed abstracts — and the pretraining\n"
    "corpus IS 3.2M PubMed abstracts.\nAn item whose source PMID is in the\n"
    "corpus was literally seen in training.", fs=9)
box(axC, 5.2, 3.2, 4.6, 5.0, "#EDF3EE", GREEN,
    "We cannot un-train, so: scan all 483\ncorpus slices for the 1k benchmark PMIDs,\n"
    "flag every overlap, and report the CLEAN\nsubset score next to the full score.\n"
    "If clean ≈ full, the score is real.", fs=9)
axC.text(0.2, 2.0, "Frontier labs run exactly this audit (n-gram / document overlap). At our scale it is exact: PMIDs are unique keys.",
         fontsize=9, color=MUT)

fig.savefig(os.path.join(OUT, "explain_eval.png"), dpi=130, facecolor=BG)
print("saved", os.path.join(OUT, "explain_eval.png"))
