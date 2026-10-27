# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Explainer figure: what SFT is and why the pretrained base model needs it.

Three panels:
  A. the base model is an autocomplete, not an assistant (same weights, wrong behavior)
  B. the ONE mechanism SFT adds: (instruction, answer) pairs + loss mask on the prompt
  C. scale: SFT is tiny next to pretraining -> it shapes behavior, it cannot add knowledge
Saves out/explain_sft.png.
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
TEAL, ORANGE, RED, GREEN = "#0E7C86", "#C97B1E", "#B3403A", "#3A7D44"
BG = "#FBFAF7"

fig = plt.figure(figsize=(13, 10), facecolor=BG)


def box(ax, x, y, w, h, fc, ec, text, fs=9, tc=None, weight="normal", pad=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc or INK, weight=weight, wrap=True)


# ---------------- Panel A: base model = autocomplete ----------------
axA = fig.add_axes([0.04, 0.66, 0.92, 0.30]); axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
axA.text(0, 9.6, "A — The pretrained model is an AUTOCOMPLETE, not an assistant",
         fontsize=12, weight="bold", color=INK)
axA.text(0, 8.7, "Same 40M weights, same question — the only thing it knows is \"continue the text\":",
         fontsize=9.5, color=MUT)

box(axA, 0.2, 5.2, 2.6, 2.2, "#EEF3F4", TEAL,
    "Prompt:\n\"What does membrane\ndepolarization trigger?\"", fs=9)
axA.annotate("", xy=(3.6, 6.3), xytext=(2.9, 6.3),
             arrowprops=dict(arrowstyle="->", color=MUT, lw=1.5))
box(axA, 3.7, 5.2, 6.1, 2.2, "#F7EDED", RED,
    "BASE model continues like a PubMed abstract:\n\"...in cardiac myocytes remains unclear. Here we investigated\n"
    "the role of Ca2+ influx in 24 patients with...\"  — fluent, but NOT an answer",
    fs=9)

box(axA, 0.2, 1.6, 2.6, 2.2, "#EEF3F4", TEAL,
    "Same prompt,\nafter SFT", fs=9)
axA.annotate("", xy=(3.6, 2.7), xytext=(2.9, 2.7),
             arrowprops=dict(arrowstyle="->", color=MUT, lw=1.5))
box(axA, 3.7, 1.6, 6.1, 2.2, "#EDF3EE", GREEN,
    "SFT model answers the question:\n\"Depolarization opens voltage-gated Ca2+ channels, triggering\n"
    "Ca2+ influx and downstream signaling such as vesicle release.\"",
    fs=9)
axA.text(0.2, 0.6, "Pretraining taught it P(next token | text) over 1B tokens of papers. Papers are not conversations —\n"
                   "so nothing in that data ever showed it the move \"question in, answer out\". SFT shows it that move.",
         fontsize=9, color=MUT)

# ---------------- Panel B: the mechanism = chat template + loss mask ----------------
axB = fig.add_axes([0.04, 0.30, 0.92, 0.32]); axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
axB.text(0, 9.6, "B — The mechanism: SAME training loop as stage 3, new data format + a loss mask",
         fontsize=12, weight="bold", color=INK)
axB.text(0, 8.7, "Each example is wrapped in a chat template; loss is computed ONLY on the answer tokens:",
         fontsize=9.5, color=MUT)

# token strip
toks = [("<|user|>", "tmpl"), ("What", "q"), ("does", "q"), ("depolar-", "q"), ("ization", "q"),
        ("trigger?", "q"), ("<|asst|>", "tmpl"), ("It", "a"), ("opens", "a"), ("Ca2+", "a"),
        ("channels", "a"), ("...", "a"), ("<|eot|>", "a")]
colors = {"tmpl": ("#E8E6E0", MUT), "q": ("#DCE9EB", TEAL), "a": ("#F4E4D2", ORANGE)}
x = 0.2
for t, kind in toks:
    w = 0.16 + 0.075 * len(t)
    fc, ec = colors[kind]
    box(axB, x, 5.6, w, 1.1, fc, ec, t, fs=7.5, pad=0.012)
    mask = "0" if kind in ("tmpl", "q") else "1"
    axB.text(x + w / 2, 5.0, mask, ha="center", fontsize=9, weight="bold",
             color=MUT if mask == "0" else ORANGE)
    x += w + 0.08
axB.text(0.2, 4.2, "loss mask:  0 = token is INPUT only (model must not be rewarded for parroting the question)   "
                   "1 = token is the TARGET", fontsize=9, color=MUT)

box(axB, 0.2, 1.2, 4.4, 2.0, "#EEF3F4", TEAL,
    "Everything else is UNCHANGED from stage 3:\nsame model, same AdamW + cosine, same bf16.\n"
    "Just: start from final.pt, lower LR (~10x),\nand multiply loss by the mask.", fs=9)
box(axB, 5.0, 1.2, 4.8, 2.0, "#F4EFE6", ORANGE,
    "Skip the mask and the model learns to\ngenerate QUESTIONS as happily as answers —\n"
    "it drifts back toward autocomplete.\n(classic failure: 'the model keeps asking itself')", fs=9)

# ---------------- Panel C: scale ----------------
axC = fig.add_axes([0.04, 0.02, 0.92, 0.24]); axC.set_xlim(0, 10); axC.set_ylim(0, 10); axC.axis("off")
axC.text(0, 9.4, "C — Scale: SFT shapes behavior, it cannot add knowledge",
         fontsize=12, weight="bold", color=INK)

axC.add_patch(plt.Rectangle((0.2, 4.2), 8.0, 1.6, fc=TEAL, ec="none", alpha=0.85))
axC.text(4.2, 5.0, "pretraining: 988M tokens, 30,167 steps, ~62 min", ha="center",
         va="center", fontsize=9.5, color="white", weight="bold")
axC.add_patch(plt.Rectangle((0.2, 2.0), 0.26, 1.6, fc=ORANGE, ec="none"))
axC.text(0.65, 2.8, "SFT: ~20–50k QA pairs ≈ 5–15M tokens, minutes of training",
         fontsize=9.5, color=ORANGE, va="center", weight="bold")
axC.text(0.2, 0.7, "All the cell-biology knowledge is already in final.pt — SFT only rewires HOW it is surfaced.  "
                   "Our SFT data: QA from PMC abstracts + BETSE simulator QA + lab protocols.",
         fontsize=9, color=MUT)

fig.savefig(os.path.join(OUT, "explain_sft.png"), dpi=130, facecolor=BG)
print("saved", os.path.join(OUT, "explain_sft.png"))
