# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 4c: before/after — same prompts through the base model and the SFT model.

The base model sees the exact same chat template, so the comparison isolates
what SFT changed (behavior), not what the prompt changed.
Saves out/before_after.md and out/before_after.png. Greedy decoding (temp 0),
seed 42 recorded in the markdown header.
"""

import os
import sys
import textwrap

os.environ.setdefault("OMP_NUM_THREADS", "10")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))
sys.path.insert(0, os.path.join(HERE, "..", "02_transformer"))
from bpe import BPE                      # noqa: E402
from model import GPT, GPTConfig        # noqa: E402
from sft import EVAL_PROMPTS, PROMPT_FMT  # noqa: E402

OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
BASE = os.path.join(HERE, "..", "03_pretrain", "out", "final.pt")
SFT = os.path.join(OUT, "sft_final.pt")


def load(path, device):
    ck = torch.load(path, weights_only=False)
    m = GPT(GPTConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


@torch.no_grad()
def answer(model, tok, prompt, device, eot):
    ids = torch.tensor([tok.encode(PROMPT_FMT.format(prompt))], device=device)
    out = model.generate(ids, 90, temperature=0.0)[0].tolist()
    ans = out[ids.shape[1]:]
    ans = ans[:ans.index(eot)] if eot in ans else ans
    return tok.decode(ans).strip()


def main():
    device = "cuda"
    torch.manual_seed(42)
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    base, bck = load(BASE, device)
    sft, sck = load(SFT, device)

    rows = []
    for p in EVAL_PROMPTS:
        rows.append((p, answer(base, tok, p, device, eot),
                     answer(sft, tok, p, device, eot)))

    md = [f"# Before/after SFT (greedy, seed 42)\n",
          f"Base: final.pt step {bck['step']} (pretrain val {bck['val_loss']:.4f})  |  "
          f"SFT: sft_final.pt step {sck['step']} (sft val {sck['val_loss']:.4f})\n"]
    for p, b, s in rows:
        md += [f"\n## Prompt\n{p}\n", f"**BASE:** {b}\n", f"**SFT:** {s}\n"]
    with open(os.path.join(OUT, "before_after.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.1 * n), facecolor="#FBFAF7")
    for ax, (p, b, s) in zip(axes, rows):
        ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        wrap = lambda t, w=95: "\n".join(textwrap.wrap(t, w)[:3])
        ax.text(0, 0.97, "PROMPT  " + wrap(p.replace("\n", " "), 90),
                fontsize=9.5, weight="bold", color="#1a2b32", va="top")
        ax.text(0, 0.60, "BASE (autocomplete):", fontsize=9, weight="bold",
                color="#B3403A", va="top")
        ax.text(0.17, 0.60, wrap(b), fontsize=9, color="#1a2b32", va="top")
        ax.text(0, 0.26, "SFT (answers):", fontsize=9, weight="bold",
                color="#3A7D44", va="top")
        ax.text(0.17, 0.26, wrap(s), fontsize=9, color="#1a2b32", va="top")
    fig.suptitle("Same 40M weights family, same prompt — behavior before vs after SFT",
                 fontsize=12, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUT, "before_after.png"), dpi=130, facecolor="#FBFAF7")
    print("saved before_after.md / before_after.png")
    for p, b, s in rows:
        print(f"\nQ: {p[:80]}\n  BASE: {b[:110]}\n  SFT:  {s[:110]}")


if __name__ == "__main__":
    main()
