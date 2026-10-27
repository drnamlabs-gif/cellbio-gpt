# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 11a: full-text index over the 3.2M-abstract corpus.

SQLite FTS5 (porter stemmer) over title+abstract — disk-based BM25, no extra
dependencies, runs on a 3.7GB server. One pass over the 483 gzipped
slices. Output: out/fts.db (+ row count printed). --smoke = 3 slices.
"""

import argparse
import glob
import gzip
import json
import os
import sqlite3
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "data", "corpus")
OUT = os.path.join(HERE, "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    db_path = os.path.join(args.out, "fts_smoke.db" if args.smoke else "fts.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE VIRTUAL TABLE docs USING fts5(pmid UNINDEXED, title, "
               "abstract, tokenize='porter unicode61')")
    files = sorted(glob.glob(os.path.join(args.corpus, "*.jsonl.gz")))
    if args.smoke:
        files = files[:3]
    t0, n = time.time(), 0
    for i, fp in enumerate(files):
        rows = []
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("abstract"):
                    rows.append((r["pmid"], r.get("title", ""), r["abstract"]))
        db.executemany("INSERT INTO docs VALUES (?,?,?)", rows)
        n += len(rows)
        if i % 40 == 0 or i == len(files) - 1:
            db.commit()
            print(f"{i + 1}/{len(files)} slices, {n:,} docs, "
                  f"{time.time() - t0:.0f}s", flush=True)
    db.execute("INSERT INTO docs(docs) VALUES ('optimize')")
    db.commit()
    db.close()
    print(f"DONE {n:,} docs -> {db_path} "
          f"({os.path.getsize(db_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
