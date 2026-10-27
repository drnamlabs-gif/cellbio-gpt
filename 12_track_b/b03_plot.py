# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Plot Track B stage-3 val bits/byte vs tokens read, with Track A as reference.

Reads out/cpt_<tag>/log.jsonl, writes out/cpt_<tag>/val_bpb.png.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M"
D = os.path.join(HERE, "out", f"cpt_{TAG}")
TOK_PER_STEP = 32768
TRACK_A = 0.698
INK, MUT, BG, TEAL, ORANGE = "#1a2b32", "#5c6f76", "#FBFAF7", "#0E7C86", "#C97B1E"

pts, final = [], None
for line in open(os.path.join(D, "log.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if "val_bpb" in r:
        pts.append((r["step"] * TOK_PER_STEP / 1e9, r["val_bpb"]))
    if "val_bpb_full" in r:
        final = (r["step"] * TOK_PER_STEP / 1e9, r["val_bpb_full"])

fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=BG)
ax.set_facecolor(BG)
x, y = zip(*pts)
ax.plot(x, y, color=ORANGE, lw=2, marker="o", ms=4, label=f"Track B · {TAG} (1M-token val)")
ax.axhline(TRACK_A, color=TEAL, lw=2, ls="--", label=f"Track A · 40M from scratch, end of epoch ({TRACK_A})")
if final:
    ax.plot(*final, marker="*", ms=16, color=ORANGE, mec=INK, ls="none",
            label=f"Track B final, full val ({final[1]:.3f})")
ax.annotate(f"untouched: {pts[0][1]:.3f}", (pts[0][0], pts[0][1]), xytext=(0.12, 0.80),
            color=MUT, fontsize=9, arrowprops=dict(arrowstyle="->", color=MUT))
cross = next((p for p in pts if p[1] <= TRACK_A), None)
if cross:
    ax.annotate(f"matches Track A after\n{cross[0] * 1000:.0f}M tokens", cross,
                xytext=(0.25, 0.73), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK))
ax.set_xlabel("PubMed tokens read in continued pretraining (billions)")
ax.set_ylabel("bits per byte on future PubMed (lower = better)")
ax.set_title("Same corpus, same temporal val: pretrained start vs random start", color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "val_bpb.png"), dpi=130, facecolor=BG)
print("saved", os.path.join(D, "val_bpb.png"), "final", final)
