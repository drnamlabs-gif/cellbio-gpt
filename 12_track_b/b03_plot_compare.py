# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage-3 routes on one figure: full-parameter CPT of the base vs LoRA CPT of
the instruct model, both on the same PubMed tokens and temporal val, with
Track A (40M from scratch) as reference. Writes out/cpt_routes.png."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_PER_STEP = 32768
TRACK_A = 0.698
INK, MUT, BG = "#1a2b32", "#5c6f76", "#FBFAF7"
ROUTES = [("cpt_SmolLM2-135M", "SmolLM2-135M base · full-parameter (all 135M weights)", "#C97B1E"),
          ("cpt_SmolLM2-135M-Instruct_lora64", "SmolLM2-135M-Instruct · LoRA (19.5M, base frozen)", "#5B4FA3")]


def read(tag):
    pts, final = [], None
    for line in open(os.path.join(HERE, "out", tag, "log.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if "val_bpb" in r:
            pts.append((r["step"] * TOK_PER_STEP / 1e9, r["val_bpb"]))
        if "val_bpb_full" in r:
            final = (r["step"] * TOK_PER_STEP / 1e9, r["val_bpb_full"])
    return pts, final


fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)
for tag, label, col in ROUTES:
    pts, final = read(tag)
    x, y = zip(*pts)
    ax.plot(x, y, color=col, lw=2, marker="o", ms=3, label=label)
    ax.annotate(f"start {pts[0][1]:.3f}", pts[0], xytext=(8, 0), textcoords="offset points",
                color=col, fontsize=9, va="center")
    if final:
        ax.plot(*final, marker="*", ms=15, color=col, mec=INK, ls="none")
        ax.annotate(f"final {final[1]:.3f}", final, xytext=(-10, 8), textcoords="offset points",
                    color=col, fontsize=9.5, ha="right", weight="bold")
ax.axhline(TRACK_A, color="#0E7C86", lw=2, ls="--", label=f"Track A · 40M from scratch ({TRACK_A})")
ax.set_xlabel("PubMed tokens read in continued pretraining (billions)")
ax.set_ylabel("bits per byte on future PubMed (lower = better)")
ax.set_title("Two ways to add PubMed to a pretrained model (★ = full val)", color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "out", "cpt_routes.png"), dpi=130, facecolor=BG)
print("saved out/cpt_routes.png")
