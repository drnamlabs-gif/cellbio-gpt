# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 5a-eval: did DPO actually fix the SFT failure modes?

Quantitative, on held-out pref_val prompts (never trained on by DPO):
  circular rate  - abbrev answers of the form "X stands for X"
  repetition     - fraction of answers containing a repeated 4-gram
  length         - mean answer words (DPO should not collapse to one word)
Qualitative: probe prompts side by side (SFT vs DPO), incl. the exact
"ATP stands for ATP" bug from stage 4.

Saves out/dpo_vs_sft" + tag + ".json, out/dpo_vs_sft" + tag + ".png, out/dpo_before_after" + tag + ".md.
Greedy decoding, seed 42.
"""

import json
import os
import re
import sys
import textwrap
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))
sys.path.insert(0, os.path.join(HERE, "..", "02_transformer"))
sys.path.insert(0, os.path.join(HERE, "..", "04_sft"))
from bpe import BPE                      # noqa: E402
from model import GPT, GPTConfig        # noqa: E402
from sft import PROMPT_FMT              # noqa: E402

OUT = os.path.join(HERE, "out")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
SFT_MODEL = os.path.join(HERE, "..", "04_sft", "out", "sft_final.pt")
DPO_MODEL = os.path.join(OUT, "dpo_final.pt")

PROBES = [
    "In cell biology, what does ATP stand for?",
    "In cell biology, what does GJ stand for?",
    "What does membrane depolarization trigger?",
    "What is a gap junction?",
    "What is apoptosis?",
]
CIRC_RE = re.compile(r"\b([A-Za-z]{2,6}s?) stands for \1\b", re.I)


def load(path, device):
    ck = torch.load(path, weights_only=False)
    m = GPT(GPTConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def gen_many(model, tok, prompts, device, eot, max_new=80):
    """Greedy, batched by exact prompt length."""
    buckets = defaultdict(list)
    for i, p in enumerate(prompts):
        ids = tok.encode(PROMPT_FMT.format(p))
        if len(ids) <= 400:
            buckets[len(ids)].append((i, ids))
    outs = {}
    for L, items in sorted(buckets.items()):
        for s in range(0, len(items), 64):
            chunk = items[s:s + 64]
            idx = torch.tensor([ids for _, ids in chunk], device=device)
            out = model.generate(idx, max_new, temperature=0.0)
            for (i, _), row in zip(chunk, out.tolist()):
                ans = row[L:]
                ans = ans[:ans.index(eot)] if eot in ans else ans
                outs[i] = tok.decode(ans).strip()
    return [outs.get(i, "") for i in range(len(prompts))]


def rep4(text):
    ws = text.lower().split()
    grams = [tuple(ws[i:i + 4]) for i in range(len(ws) - 3)]
    return len(grams) != len(set(grams)) if grams else False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="",
                    help="checkpoint/output suffix, e.g. _v2")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(42)
    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    sft = load(SFT_MODEL, device)
    dpo = load(os.path.join(OUT, f"dpo_final{args.tag}.pt"), device)
    tag = args.tag

    vex = [json.loads(l) for l in open(os.path.join(OUT, "pref_val.jsonl"),
                                       encoding="utf-8")]
    prompts = [e["prompt"] for e in vex]
    tasks = [e["task"] for e in vex]
    print(f"{len(prompts)} held-out val prompts")

    res = {}
    for name, model in (("SFT", sft), ("DPO", dpo)):
        ans = gen_many(model, tok, prompts, device, eot)
        ab = [a for a, t in zip(ans, tasks) if t == "abbrev"]
        res[name] = {
            "circular_rate": sum(bool(CIRC_RE.search(a)) for a in ab) / max(1, len(ab)),
            "rep4_rate": sum(rep4(a) for a in ans) / len(ans),
            "mean_words": sum(len(a.split()) for a in ans) / len(ans),
            "empty_rate": sum(not a for a in ans) / len(ans),
        }
        print(name, json.dumps(res[name], indent=1))
    json.dump(res, open(os.path.join(OUT, "dpo_vs_sft" + tag + ".json"), "w"), indent=2)

    sft_p = gen_many(sft, tok, PROBES, device, eot)
    dpo_p = gen_many(dpo, tok, PROBES, device, eot)
    with open(os.path.join(OUT, "dpo_before_after" + tag + ".md"), "w", encoding="utf-8") as f:
        f.write("# SFT vs DPO (greedy, seed 42)\n")
        for p, a, b in zip(PROBES, sft_p, dpo_p):
            f.write(f"\n## {p}\n**SFT:** {a}\n\n**DPO:** {b}\n")
            print(f"\nQ: {p}\n  SFT: {a[:100]}\n  DPO: {b[:100]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor="#FBFAF7")
    metrics = [("circular_rate", "circular answers\n(abbrev val prompts)"),
               ("rep4_rate", "repeated 4-gram\n(all val prompts)"),
               ("mean_words", "mean answer length\n(words)")]
    for ax, (k, title) in zip(axes, metrics):
        vals = [res["SFT"][k], res["DPO"][k]]
        bars = ax.bar(["SFT", "DPO"], vals, color=["#B3403A", "#3A7D44"], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if v < 1 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=10, weight="bold")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("DPO vs SFT on held-out prompts (greedy, seed 42)",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "dpo_vs_sft" + tag + ".png"), dpi=130, facecolor="#FBFAF7")
    print("\nsaved dpo_vs_sft" + tag + ".json / dpo_vs_sft" + tag + ".png / dpo_before_after" + tag + ".md")


if __name__ == "__main__":
    main()
