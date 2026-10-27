# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""One-time: encode the whole corpus into flat uint16 token files.

corpus slices -> BPE ids -> data/tokens/train.bin + val.bin (+ meta.json)

- docs are joined into one stream, <|endoftext|> after each doc
- val split is TEMPORAL: the last 5 slices (most recent science) are held out,
  so validation also measures slight forward-in-time generalization
- uint16 is enough: vocab 16,384 < 65,536; 0.8B tokens -> ~1.6GB on disk
- 10 worker processes (Windows thread cap rule)
"""

import glob
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))

CORPUS = os.path.join(HERE, "..", "data", "corpus")
TOKENIZER = os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json")
OUT_DIR = os.path.join(HERE, "..", "data", "tokens")
N_VAL_SLICES = 5
_tok = None


def _init():
    global _tok
    from bpe import BPE
    _tok = BPE.load(TOKENIZER)


def _encode_slice(path):
    ids = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.extend(_tok.encode(d["title"] + "\n" + d["abstract"]))
            ids.append(_tok.eot_id)
    return np.array(ids, dtype=np.uint16)


def encode(files, out_path, workers=10):
    t0, n = time.time(), 0
    with open(out_path, "wb") as out, mp.Pool(workers, initializer=_init) as pool:
        for i, arr in enumerate(pool.imap(_encode_slice, files, chunksize=1)):
            arr.tofile(out)
            n += len(arr)
            if (i + 1) % 25 == 0:
                rate = n / (time.time() - t0) / 1e6
                print(f"  {i + 1}/{len(files)} slices, {n / 1e6:.0f}M tokens "
                      f"({rate:.1f}M tok/s)", flush=True)
    return n


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CORPUS, "slice_*.jsonl.gz")))
    train_files, val_files = files[:-N_VAL_SLICES], files[-N_VAL_SLICES:]
    print(f"{len(train_files)} train slices, {len(val_files)} val slices "
          f"(val = newest, temporal holdout)")

    n_val = encode(val_files, os.path.join(OUT_DIR, "val.bin"))
    print(f"val.bin: {n_val:,} tokens")
    n_train = encode(train_files, os.path.join(OUT_DIR, "train.bin"))
    print(f"train.bin: {n_train:,} tokens")

    from bpe import BPE
    tok = BPE.load(TOKENIZER)
    meta = {"date": "2026-10-23", "tokenizer": TOKENIZER,
            "vocab_size": tok.vocab_size, "eot_id": tok.eot_id,
            "n_train_tokens": n_train, "n_val_tokens": n_val,
            "val_slices": [os.path.basename(p) for p in val_files],
            "dtype": "uint16"}
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("meta.json written. done.")


if __name__ == "__main__":
    main()
