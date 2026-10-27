# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Track B, stage 3 data: the SAME corpus and SAME split as Track A's
03_pretrain/encode_corpus.py, re-encoded with the pretrained model's own
tokenizer (a pretrained model cannot change tokenizer).

- same files: data/corpus/slice_*.jsonl.gz, last 5 slices = temporal val
- same document text: title + "\n" + abstract, one end-of-text id after each
- uint32 (Qwen vocab is ~152k, beyond uint16)
- also counts UTF-8 bytes per split, so held-out loss can be reported as
  bits-per-byte, which is comparable across tokenizers (Track A vs Track B)
Outputs data/tokens_qwen/{train.bin,val.bin,meta.json}. 10 workers.
"""

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "data", "corpus")
N_VAL_SLICES = 5
_tok = None
_eot = None


def _init(tokenizer):
    global _tok, _eot
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(tokenizer)
    _eot = _tok.convert_tokens_to_ids("<|endoftext|>")
    if _eot is None or _eot == _tok.unk_token_id:
        _eot = _tok.eos_token_id


def _encode_slice(path):
    texts = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["title"] + "\n" + d["abstract"])
    ids = []
    for chunk in _tok(texts, add_special_tokens=False)["input_ids"]:
        ids.extend(chunk)
        ids.append(_eot)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    return np.array(ids, dtype=np.uint32), n_bytes, len(texts)


def encode(files, out_path, tokenizer, workers=10):
    t0, n, nb, nd = time.time(), 0, 0, 0
    with open(out_path, "wb") as out, mp.Pool(workers, initializer=_init,
                                               initargs=(tokenizer,)) as pool:
        for i, (arr, b, d) in enumerate(pool.imap(_encode_slice, files, chunksize=1)):
            arr.tofile(out)
            n, nb, nd = n + len(arr), nb + b, nd + d
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(files)} slices, {n / 1e6:.0f}M tokens "
                      f"({n / (time.time() - t0) / 1e6:.2f}M tok/s)", flush=True)
    return n, nb, nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--out", default="tokens_qwen")
    args = ap.parse_args()
    OUT_DIR = os.path.join(HERE, "..", "data", args.out)
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CORPUS, "slice_*.jsonl.gz")))
    train_files, val_files = files[:-N_VAL_SLICES], files[-N_VAL_SLICES:]
    print(f"{len(train_files)} train slices, {len(val_files)} val slices")
    n_val, b_val, d_val = encode(val_files, os.path.join(OUT_DIR, "val.bin"), args.tokenizer)
    print(f"val.bin: {n_val:,} tokens, {b_val:,} bytes, {d_val:,} docs")
    n_tr, b_tr, d_tr = encode(train_files, os.path.join(OUT_DIR, "train.bin"), args.tokenizer)
    print(f"train.bin: {n_tr:,} tokens, {b_tr:,} bytes, {d_tr:,} docs")
    _init(args.tokenizer)
    meta = {"date": "2026-10-24", "tokenizer": args.tokenizer,
            "vocab_size": len(_tok), "eot_id": _eot, "dtype": "uint32",
            "n_train_tokens": n_tr, "n_val_tokens": n_val,
            "n_train_bytes": b_tr, "n_val_bytes": b_val,
            "n_train_docs": d_tr, "n_val_docs": d_val,
            "val_slices": [os.path.basename(p) for p in val_files],
            "mirrors": "03_pretrain/encode_corpus.py (same files, split, text)"}
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("meta.json written. done.")


if __name__ == "__main__":
    main()
