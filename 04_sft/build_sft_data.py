# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 4a: build the SFT dataset from the corpus itself (no external LLM).

Four self-supervised instruction tasks mined from title+abstract pairs:
  qtitle   - papers whose TITLE is a question -> answer = the abstract's last
             sentences (where conclusions live)
  defn     - "X is a/an/the ..." sentence -> "What is X?" -> that sentence
  abbrev   - "long form (ABBR)" (Schwartz-Hearst-style initial alignment)
             -> "What does ABBR stand for?" -> long form
  titlegen - "Write a concise title for this abstract: ..." -> the real title
             (an honest seq2seq task with a verifiable target)

Contamination rule: the 5 temporal-val slices (data/tokens/meta.json) are
EXCLUDED, so SFT never sees text from the pretraining val set.

Output: out/sft_train.jsonl, out/sft_val.jsonl (2%), out/sft_meta.json,
        out/sft_data_stats.png. Seed 42. --smoke = 10 slices.
"""

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "data", "corpus")
TOKENS_META = os.path.join(HERE, "..", "data", "tokens", "meta.json")
OUT = os.path.join(HERE, "out")

SEED = 42
N_WORKERS = 10
CAPS = {"qtitle": 30000, "defn": 25000, "abbrev": 25000, "titlegen": 25000}

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
DEFN_RE = re.compile(
    r"^(?:The |An? )?([A-Z][A-Za-z0-9\-/ ]{2,50}?) (is|are) (?:a|an|the) ")
ABBR_RE = re.compile(r"\(([A-Z][A-Za-z]{1,5}s?)\)")
STOP = {"of", "and", "the", "in", "for", "to", "with", "by", "on", "a", "an"}


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def long_form(words_before, abbr):
    """Shortest word-suffix whose initials spell the abbreviation (stopwords free)."""
    letters = [c for c in abbr.rstrip("s").lower()]
    for start in range(len(words_before) - 1, -1, -1):
        cand = words_before[start:]
        sig = [w[0].lower() for w in cand if w.lower() not in STOP and w[0].isalpha()]
        if sig == letters:
            lf = " ".join(cand).strip(",;:. ")
            if len(lf) >= len(abbr) + 3 and len(lf) <= 70:
                return lf
    return None


def mine(path):
    out = {"qtitle": [], "defn": [], "abbrev": [], "titlegen": []}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            title, abstract = d["title"].strip(), d["abstract"].strip()
            if len(abstract) < 200:
                continue
            sents = sentences(abstract)

            if title.endswith("?") and len(sents) >= 3:
                ans = " ".join(sents[-2:])
                if 60 <= len(ans) <= 450:
                    out["qtitle"].append(
                        {"task": "qtitle", "prompt": title, "answer": ans})

            for s in sents[:4]:
                m = DEFN_RE.match(s)
                if m and len(s) <= 350 and not any(c.isdigit() for c in m.group(1)):
                    verb = "are" if m.group(2) == "are" else "is"
                    q = f"What {verb} {m.group(1).strip()}?"
                    out["defn"].append({"task": "defn", "prompt": q, "answer": s})
                    break

            m = ABBR_RE.search(abstract)
            if m:
                before = abstract[:m.start()].split()
                lf = long_form(before[-10:], m.group(1))
                if lf:
                    out["abbrev"].append(
                        {"task": "abbrev",
                         "prompt": f"In cell biology, what does {m.group(1)} stand for?",
                         "answer": f"{m.group(1)} stands for {lf}."})

            words = abstract.split()
            if len(words) >= 80 and 30 <= len(title) <= 180:
                out["titlegen"].append(
                    {"task": "titlegen",
                     "prompt": "Write a concise title for this abstract:\n"
                               + " ".join(words[:170]),
                     "answer": title})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    val_slices = set(json.load(open(TOKENS_META))["val_slices"])
    files = [f for f in sorted(glob.glob(os.path.join(CORPUS, "slice_*.jsonl.gz")))
             if os.path.basename(f) not in val_slices]
    if args.smoke:
        files = files[::len(files) // 10][:10]
    print(f"{len(files)} slices ({'SMOKE' if args.smoke else 'full'}), "
          f"{len(val_slices)} temporal-val slices excluded")

    pools = {k: [] for k in CAPS}
    with mp.Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(mine, files, chunksize=4)):
            for k, v in res.items():
                pools[k].extend(v)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(files)} slices,",
                      {k: len(v) for k, v in pools.items()}, flush=True)

    rng = np.random.default_rng(SEED)
    data = []
    counts_raw = {k: len(v) for k, v in pools.items()}
    for k, v in pools.items():
        idx = rng.permutation(len(v))[:CAPS[k]]
        data.extend(v[i] for i in idx)
    order = rng.permutation(len(data))
    data = [data[i] for i in order]
    n_val = max(200, int(0.02 * len(data))) if not args.smoke else 100
    val, train = data[:n_val], data[n_val:]

    suff = "_smoke" if args.smoke else ""
    for name, part in (("sft_train", train), ("sft_val", val)):
        with open(os.path.join(OUT, f"{name}{suff}.jsonl"), "w", encoding="utf-8") as f:
            for ex in part:
                f.write(json.dumps(ex) + "\n")
    counts = {k: sum(1 for ex in train if ex["task"] == k) for k in CAPS}
    json.dump({"seed": SEED, "date": "2026-10-23", "slices": len(files),
               "raw_counts": counts_raw, "caps": CAPS,
               "n_train": len(train), "n_val": len(val),
               "train_counts": counts, "val_excluded": sorted(val_slices)},
              open(os.path.join(OUT, f"sft_meta{suff}.json"), "w"), indent=2)
    print(f"raw mined: {counts_raw}")
    print(f"train {len(train)}, val {len(val)}; per-task {counts}")

    for k in CAPS:
        ex = next(e for e in train if e["task"] == k)
        print(f"\n--- sample [{k}] ---\nQ: {ex['prompt'][:200]}\nA: {ex['answer'][:200]}")

    if not args.smoke:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ks = list(CAPS)
        axes[0].bar(ks, [counts[k] for k in ks], color="#0E7C86")
        axes[0].set_title(f"SFT examples per task (train, n={len(train):,})")
        axes[0].grid(alpha=0.25, axis="y")
        lens = [len(ex["prompt"].split()) + len(ex["answer"].split()) for ex in train]
        axes[1].hist(lens, bins=60, color="#C97B1E")
        axes[1].set_title("words per example (prompt+answer)")
        axes[1].axvline(np.percentile(lens, 99), color="#B3403A", ls="--",
                        label=f"p99 = {np.percentile(lens, 99):.0f}")
        axes[1].legend(frameon=False)
        axes[1].grid(alpha=0.25, axis="y")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "sft_data_stats.png"), dpi=120)
        print("\nsaved", os.path.join(OUT, "sft_data_stats.png"))


if __name__ == "__main__":
    main()
