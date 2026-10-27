# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Explainer figure: what "bits per byte" means and why lower is better.

Measures real anchors on the SAME temporal-val text the models are scored on:
raw text (8 bits/byte), gzip and xz compression — then places the models on
the same scale. Saves out/explain_bpb.png and out/bpb_anchors.json.
"""

import glob
import gzip
import json
import lzma
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CORPUS = os.path.join(HERE, "..", "data", "corpus")
INK, MUT, BG = "#1a2b32", "#5c6f76", "#FBFAF7"
TEAL, ORANGE, RED, GREY = "#0E7C86", "#C97B1E", "#B3403A", "#9AA7AC"

files = sorted(glob.glob(os.path.join(CORPUS, "slice_*.jsonl.gz")))[-5:]
texts = []
for fp in files:
    with gzip.open(fp, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["title"] + "\n" + d["abstract"])
raw = "\n".join(texts).encode("utf-8")
n = len(raw)
gz = len(gzip.compress(raw, 9)) * 8 / n
xz = len(lzma.compress(raw, preset=6)) * 8 / n
anchors = {"bytes": n, "raw": 8.0, "gzip": gz, "xz": xz}
with open(os.path.join(OUT, "bpb_anchors.json"), "w") as f:
    json.dump(anchors, f, indent=2)
print(anchors)

rows = [("raw text, no knowledge", 8.0, GREY),
        ("gzip (zip-file compression)", gz, GREY),
        ("xz (best ordinary compressor)", xz, GREY),
        ("SmolLM2-135M, untouched", 0.825, ORANGE),
        ("Qwen2.5-0.5B, untouched", 0.712, ORANGE),
        ("Track A 40M, after 1 epoch PubMed", 0.698, TEAL)]

fig = plt.figure(figsize=(14, 6.4), facecolor=BG)
ax0 = fig.add_axes([0.03, 0.08, 0.40, 0.82]); ax0.axis("off")
ax0.set_xlim(0, 10); ax0.set_ylim(0, 10)
ax0.text(0, 9.6, "The game: guess the next piece of the abstract", fontsize=13,
         weight="bold", color=INK)
ax0.text(0, 8.7, "“The resting membrane potential is set mainly by ___”", fontsize=11.5,
         color=INK, style="italic")
guesses = [("potassium", 0.62), ("the", 0.10), ("sodium", 0.06), ("K+", 0.05), ("other", 0.17)]
for i, (w, p) in enumerate(guesses):
    y = 7.6 - i * 0.62
    ax0.barh(y, p * 8, height=0.42, left=1.9, color=TEAL if i == 0 else GREY)
    ax0.text(1.8, y, w, ha="right", va="center", fontsize=10.5, color=INK)
    ax0.text(1.95 + p * 8, y, f" {p:.0%}", va="center", fontsize=10, color=INK)
ax0.text(0, 4.2, "The true next word is “potassium”.", fontsize=11, color=INK)
ax0.text(0, 3.45, "Surprise = −log₂(0.62) = 0.69 bits", fontsize=11.5, color=TEAL,
         weight="bold")
ax0.text(0, 2.55, "A model that put only 6% on it would pay −log₂(0.06) = 4.1 bits.\n"
         "Add the surprise up over every word of the held-out text,\n"
         "divide by the number of letters (bytes) in that text:",
         fontsize=10.5, color=MUT, va="top")
ax0.text(0, 0.55, "bits per byte = how surprised the model is, per letter.\n"
         "Lower = it predicts the science better = it knows more.",
         fontsize=11.5, color=INK, weight="bold", va="center")

ax1 = fig.add_axes([0.53, 0.21, 0.44, 0.65])
names = [r[0] for r in rows]
vals = [r[1] for r in rows]
cols = [r[2] for r in rows]
bars = ax1.barh(range(len(rows)), vals, color=cols, height=0.62)
ax1.set_xscale("log")
ax1.set_xlim(0.5, 10)
ax1.set_xticks([0.5, 0.7, 1, 2, 4, 8])
ax1.set_xticklabels(["0.5", "0.7", "1", "2", "4", "8"])
ax1.set_yticks(range(len(rows)))
ax1.set_yticklabels(names, fontsize=9.5)
ax1.invert_yaxis()
for b, v in zip(bars, vals):
    ax1.text(v * 1.04, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center",
             fontsize=9.5, color=INK)
ax1.set_xlabel("bits per byte on the held-out PubMed text (log scale) — lower is better",
               fontsize=9.5)
ax1.set_title("Same text, same ruler: compression = understanding", fontsize=11.5,
              color=INK)
ax1.annotate("where does SmolLM2 land after\nthe same PubMed epoch? (running now)",
             xy=(0.6, 4.6), xytext=(1.25, 3.7), fontsize=9, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED))
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(alpha=0.25, lw=0.5, axis="x", which="both")
fig.text(0.53, 0.02, "Why per BYTE, not per token: each model cuts text into different "
         "token pieces,\nbut the letters are identical, so bytes are the one ruler they "
         "all share.\nVal = the 5 newest PubMed slices (Jul–Dec 2026), never trained on.",
         fontsize=8.5, color=MUT)
fig.savefig(os.path.join(OUT, "explain_bpb.png"), dpi=130, facecolor=BG)
print("saved", os.path.join(OUT, "explain_bpb.png"))
