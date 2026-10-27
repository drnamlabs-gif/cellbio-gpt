# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Explainer figure: what actually crosses the LLM <-> BETSE boundary.

BETSE's inputs are a YAML config (numbers) + bitmap geometry masks (geo/*.png).
The LLM never sends text to BETSE. The bridge is make_sim_problems.py:
  3 knob numbers -> edited YAML -> BETSE sim -> mean Vmem (one number)
  -> wrapped into an English problem whose gold answer IS that number.
The verifier just parses the mV value out of the LLM's text and compares.
Saves out/explain_interface.png.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
INK, MUT = "#1a2b32", "#5c6f76"
TEAL, ORANGE, RED, GREEN = "#0E7C86", "#C97B1E", "#B3403A", "#3A7D44"
BG = "#FBFAF7"

# real example: first train problem
with open(os.path.join(OUT, "betse_train.jsonl"), encoding="utf-8") as f:
    ex = json.loads(f.readline())

fig = plt.figure(figsize=(14.5, 7.6), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, color, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8",
                                fc="white", ec=color, lw=1.8))
    ax.text(x + w / 2, y + h - 3, title, ha="center", va="top",
            fontsize=10.5, weight="bold", color=color)

def arrow(x1, y1, x2, y2, color, label, dy=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=2, color=color))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center",
            fontsize=8.5, color=color, weight="bold")

ax.text(50, 97, "The LLM never talks to BETSE. One Python script is the translator.",
        ha="center", fontsize=13, weight="bold", color=INK)

# ---- left column: BETSE's world (bitmaps + yaml numbers) -------------------
box(2, 42, 27, 48, TEAL, "BETSE's inputs (no language)")
img = mpimg.imread(os.path.join(HERE, "validate_one", "geo", "square", "base.png"))
axi = fig.add_axes([0.045, 0.52, 0.10, 0.18]); axi.imshow(img, cmap="gray"); axi.axis("off")
axi.set_title("geo/square/base.png\n(tissue shape bitmap — FIXED)", fontsize=7.5, color=MUT)
ax.text(16.5, 72, "sim2.yaml — the 3 knobs we edit:", fontsize=8.5, color=INK, ha="left",
        transform=ax.transData)
ax.text(16.5, 62, "env K+:  5.0 → 2.0 mM\nenv Na+: 145 → 148 mM\nDm_K:  ×0.25\nDm_Na: ×0.5",
        fontsize=8.5, color=TEAL, family="monospace", va="top")
ax.text(15.5, 46, "bitmap = WHERE the cells are (unchanged)\nYAML numbers = CHEMISTRY around them (varied)",
        fontsize=8, color=MUT, ha="center")

# ---- middle: the simulator + readout ---------------------------------------
box(35, 56, 24, 24, ORANGE, "BETSE (physics, ~13 s)")
ax.text(47, 68, "solves ion flux +\nelectrodiffusion PDEs\non that bitmap tissue", ha="center",
        fontsize=8.5, color=INK)
ax.text(47, 60.5, "output: Vmem of every cell", ha="center", fontsize=8.5, color=ORANGE)

box(35, 28, 24, 18, RED, "Readout (1 number)")
ax.text(47, 36.5, f"mean Vmem = {ex['truth']['value']:+d} mV", ha="center",
        fontsize=11, weight="bold", color=RED, family="monospace")
ax.text(47, 31.5, "this number is the TRUTH", ha="center", fontsize=8, color=MUT)

# ---- right column: the LLM's world (text only) -----------------------------
box(66, 52, 32, 38, GREEN, "LLM's world (text only)")
prompt_wrapped = ("\"A BETSE tissue simulation runs with\n"
                  "extracellular K+ = 2 mM, extracellular\n"
                  "Na+ = 148 mM, K+ membrane permeability\n"
                  "scaled x0.25 and Na+ permeability\n"
                  "scaled x0.5. What steady-state mean\n"
                  "Vmem (mV) does the tissue reach?\"")
ax.text(82, 84, prompt_wrapped, ha="center", va="top", fontsize=8,
        color=INK, family="monospace")
ax.text(82, 61, f"model generates:  \"Mean Vmem = {ex['truth']['value']:+d} mV.\"",
        ha="center", fontsize=8.5, color=GREEN, family="monospace", weight="bold")
ax.text(82, 56, "same 3 knob numbers, spelled out in English", ha="center",
        fontsize=8, color=MUT)

box(66, 24, 32, 20, INK, "Verifier (the RLVR reward)")
ax.text(82, 36.5, "parse mV from the model's text,\ncompare to sim truth:\n"
        "exact ±2 mV → 1.0   |   ±10 mV → partial   |   else 0",
        ha="center", va="top", fontsize=8, color=INK)

# ---- arrows ----------------------------------------------------------------
arrow(29.5, 68, 34.5, 68, TEAL, "numbers in")
arrow(47, 55.5, 47, 46.5, ORANGE, "", dy=0)
arrow(59.5, 37, 65.5, 37, RED, "truth → reward check")
arrow(30.5, 92, 65.5, 92, MUT, "same 3 knobs → English prompt (make_sim_problems.py)")
arrow(82, 51.5, 82, 45.5, GREEN, "")

ax.text(50, 18, "150 sims were run ONCE (offline) → 564 text problems with sim-certified answers.\n"
        "During GRPO the model only reads/writes text; the reward is 'did your number match the physics?'.\n"
        "The bitmap channel (tissue SHAPE as input) is untouched — that is a future vision-language extension.",
        ha="center", fontsize=9.5, color=MUT)
ax.text(50, 7, f"real example shown: combo K+={ex['cout2']} mM, Dm_K ×{ex['cin']}, Dm_Na ×{ex['cout']} "
        f"→ truth {ex['truth']['value']:+d} mV (out/sim_truths.jsonl)",
        ha="center", fontsize=8, color=MUT, style="italic")

fig.savefig(os.path.join(OUT, "explain_interface.png"), dpi=130, facecolor=BG,
            bbox_inches="tight")
print("saved", os.path.join(OUT, "explain_interface.png"))
