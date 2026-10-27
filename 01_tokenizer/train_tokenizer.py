# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Train the domain BPE tokenizer on a sample of the corpus.

Usage:
  python train_tokenizer.py --smoke            # 2k docs, 500 merges, asserts
  python train_tokenizer.py                    # full: sample docs, 16384 vocab

Seed convention: SEED=42, single seed, recorded in the saved tokenizer meta.
"""

import argparse
import glob
import gzip
import json
import os
import random
import time

from bpe import BPE

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "data", "corpus")
OUT = os.path.join(HERE, "out", "tokenizer.json")


def sample_texts(n_docs, seed=SEED):
    """Reservoir-sample n_docs abstracts across all slices (uniform, seeded)."""
    rng = random.Random(seed)
    sample, seen = [], 0
    for path in sorted(glob.glob(os.path.join(CORPUS, "slice_*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                seen += 1
                if len(sample) < n_docs:
                    sample.append(line)
                else:
                    j = rng.randrange(seen)
                    if j < n_docs:
                        sample[j] = line
    print(f"sampled {len(sample):,} of {seen:,} docs (seed {seed})")
    for line in sample:
        d = json.loads(line)
        yield d["title"] + "\n" + d["abstract"]


def report(tok, texts):
    n_bytes = n_tokens = 0
    for t in texts:
        ids = tok.encode(t)
        assert tok.decode(ids) == t, "round-trip failed"
        n_bytes += len(t.encode("utf-8"))
        n_tokens += len(ids)
    print(f"round-trip OK on {len(texts)} texts; "
          f"compression {n_bytes / n_tokens:.2f} bytes/token")
    for term in ["depolarization", "membrane potential", "gap junction",
                 "regeneration", "bioelectric"]:
        ids = tok.encode(" " + term)
        pieces = [tok.decode([i]) for i in ids]
        print(f"  {term!r:22s} -> {len(ids)} token(s): {pieces}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--docs", type=int, default=400_000)
    ap.add_argument("--vocab", type=int, default=16384)
    args = ap.parse_args()
    if args.smoke:
        args.docs, args.vocab = 2000, 256 + 1 + 500

    texts = list(sample_texts(args.docs))
    t0 = time.time()
    tok = BPE.train(texts, args.vocab)
    print(f"trained {len(tok.merges):,} merges in {time.time() - t0:.0f}s")

    report(tok, texts[:200])
    out = OUT if not args.smoke else OUT.replace(".json", "_smoke.json")
    tok.save(out, meta={"seed": SEED, "docs": args.docs, "vocab": args.vocab,
                        "corpus": CORPUS, "date": "2026-10-22"})
    print(f"saved {out}")
    tok2 = BPE.load(out)
    assert tok2.encode("Vmem depolarization") == tok.encode("Vmem depolarization")
    print("save/load OK")


if __name__ == "__main__":
    main()
