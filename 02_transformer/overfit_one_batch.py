# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage-2 smoke test: the classic architecture check.

A correct GPT must drive loss to ~0 on ONE memorized batch. If it can't,
the architecture has a bug. Also asserts:
  1. manual attention == fused SDPA attention (logit parity),
  2. KV-cache generation == full-recompute generation (same token ids),
then saves loss curve, layer-0 attention map, sample generation, metrics.

Outputs -> 02_transformer/out/
"""

import glob
import gzip
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))
from bpe import BPE                      # noqa: E402
from model import GPT, GPTConfig        # noqa: E402

SEED = 42
OUT = os.path.join(HERE, "out")
CORPUS = os.path.join(HERE, "..", "data", "corpus")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
INK, ACCENT, VOLT = "#1B2528", "#0E7C86", "#C97B1E"


def get_batch(tok, n_docs=8, seq=256):
    docs = []
    path = sorted(glob.glob(os.path.join(CORPUS, "slice_*.jsonl.gz")))[-1]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids = tok.encode(d["title"] + "\n" + d["abstract"])
            if len(ids) >= seq + 1:
                docs.append(ids[: seq + 1])
            if len(docs) == n_docs:
                break
    x = torch.tensor([d[:-1] for d in docs])
    y = torch.tensor([d[1:] for d in docs])
    return x, y


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(10)
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = BPE.load(TOKENIZER)
    cfg = GPTConfig(vocab_size=tok.vocab_size, n_layer=4, n_head=4,
                    d_model=256, max_seq=512)
    model = GPT(cfg).to(dev)
    print(f"device {dev}, params {model.num_params()/1e6:.1f}M")

    x, y = get_batch(tok)
    x, y = x.to(dev), y.to(dev)
    print(f"batch {tuple(x.shape)} = {x.numel()} tokens to memorize")

    # --- check 1: manual vs fused attention parity (before training) ---
    with torch.no_grad():
        logits_manual, _, _ = model(x)
        model.cfg.fused = True
        logits_fused, _, _ = model(x)
        model.cfg.fused = False
    diff = (logits_manual - logits_fused).abs().max().item()
    parity_ok = diff < 1e-3
    print(f"manual-vs-fused max |diff| = {diff:.2e} -> {'OK' if parity_ok else 'FAIL'}")

    # --- overfit ---
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses, t0 = [], time.time()
    for step in range(1000):
        _, loss, _ = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")
        if loss.item() < 0.05:
            break
    print(f"final loss {losses[-1]:.4f} after {len(losses)} steps "
          f"({time.time()-t0:.0f}s)")
    overfit_ok = losses[-1] < 0.05

    # --- check 2: KV-cache == full recompute (greedy, token-exact) ---
    prompt = x[:1, :20]
    g_cache = model.generate(prompt.clone(), 30, use_cache=True)
    g_full = model.generate(prompt.clone(), 30, use_cache=False)
    cache_ok = bool(torch.equal(g_cache, g_full))
    print(f"KV-cache parity: {'OK' if cache_ok else 'FAIL'}")

    # --- memorization sample: continue the first doc from its first 20 tokens ---
    sample = tok.decode(g_cache[0].tolist())
    truth = tok.decode(x[0, :50].tolist())
    with open(os.path.join(OUT, "sample.txt"), "w", encoding="utf-8") as f:
        f.write("GENERATED (greedy, 20-token prompt):\n" + sample
                + "\n\nGROUND TRUTH (first 50 tokens):\n" + truth + "\n")

    # --- figure 1: loss curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(losses, color=ACCENT, lw=2)
    ax.axhline(0.05, color=VOLT, ls="--", lw=1.5)
    ax.annotate("pass threshold 0.05", (len(losses) * 0.55, 0.06),
                color=VOLT, fontsize=9)
    ax.set(xlabel="step", ylabel="cross-entropy loss (log)",
           title="Overfit one batch — architecture sanity")
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overfit_loss.png"), dpi=120)

    # --- figure 2: layer-0 attention maps after training ---
    with torch.no_grad():
        model(x[:1, :64], store_att=True)
    att = model.blocks[0].attn.last_att[0]          # (heads, T, T)
    fig, axes = plt.subplots(1, cfg.n_head, figsize=(3.2 * cfg.n_head, 3.4))
    for h, ax in enumerate(axes):
        ax.imshow(att[h].cpu(), cmap="Blues", aspect="auto")
        ax.set_title(f"head {h}", fontsize=10)
        ax.set_xlabel("key pos")
        if h == 0:
            ax.set_ylabel("query pos")
    fig.suptitle("Layer-0 attention (trained) — upper triangle must be empty (causal)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "attention_map.png"), dpi=120)

    result = {"seed": SEED, "date": "2026-10-23", "device": dev,
              "params": model.num_params(), "batch_shape": list(x.shape),
              "steps": len(losses), "final_loss": losses[-1],
              "parity_max_diff": diff, "parity_ok": parity_ok,
              "cache_ok": cache_ok, "overfit_ok": overfit_ok}
    with open(os.path.join(OUT, "overfit_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    assert parity_ok and cache_ok and overfit_ok, "STAGE 2 SMOKE FAILED"
    print("STAGE 2 SMOKE PASSED")


if __name__ == "__main__":
    main()
