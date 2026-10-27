# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 6a: fetch PubMedQA (pqa_labeled, 1k expert-labeled yes/no/maybe) and
measure pretraining contamination honestly.

PubMedQA questions are built FROM PubMed abstracts; our pretraining corpus IS
PubMed abstracts. Any benchmark item whose source PMID is in the corpus was
seen during pretraining. We cannot un-train, so we do the next honest thing:
flag every contaminated item and report benchmark scores on the CLEAN subset
alongside the full set.

Outputs (out/):
  pubmedqa.jsonl      one line per item: pubid, question, contexts, label,
                      contaminated (bool)
  contamination.json  counts + label distribution for full vs clean
"""

import gzip
import json
import multiprocessing as mp
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CORPUS = os.path.join(HERE, "..", "data", "corpus")
SEED = 42


def pmids_in_slice(path):
    found = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            # pmid is the first field; avoid full json parse for 3.2M docs
            i = line.find('"pmid": "') + 9
            found.add(line[i:line.index('"', i)])
    return found


def main():
    os.makedirs(OUT, exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    items = [{
        "pubid": str(r["pubid"]),
        "question": r["question"],
        "contexts": list(r["context"]["contexts"]),
        "label": r["final_decision"],
    } for r in ds]
    print(f"PubMedQA pqa_labeled: {len(items)} items")

    slices = sorted(os.path.join(CORPUS, f) for f in os.listdir(CORPUS)
                    if f.endswith(".jsonl.gz"))
    print(f"scanning {len(slices)} corpus slices for PMID overlap ...")
    with mp.Pool(10) as pool:
        corpus_pmids = set()
        for s in pool.imap_unordered(pmids_in_slice, slices, chunksize=8):
            corpus_pmids |= s
    print(f"corpus PMIDs: {len(corpus_pmids):,}")

    n_contam = 0
    for it in items:
        it["contaminated"] = it["pubid"] in corpus_pmids
        n_contam += it["contaminated"]

    with open(os.path.join(OUT, "pubmedqa.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    def dist(sub):
        d = {}
        for it in sub:
            d[it["label"]] = d.get(it["label"], 0) + 1
        return d

    clean = [it for it in items if not it["contaminated"]]
    meta = {
        "seed": SEED, "date": "2026-10-23",
        "n_items": len(items), "n_contaminated": n_contam,
        "n_clean": len(clean),
        "label_dist_full": dist(items), "label_dist_clean": dist(clean),
        "corpus_pmids": len(corpus_pmids),
        "note": "contaminated = benchmark source PMID present in pretraining corpus",
    }
    with open(os.path.join(OUT, "contamination.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
