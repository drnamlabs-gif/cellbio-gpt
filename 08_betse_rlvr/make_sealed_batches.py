# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8g: exam C, twenty sealed test batches for statistical validation.

Exams A and B gave one number each. To compare models with a paired test we
need many independent units, so this builds 20 batches of fresh simulations,
15 per batch (about 60 problems), at points that appear in no training set and
in no earlier exam:
  - at least 0.25 log2 units from every lattice node (as exam B),
  - at least 0.20 from every point of the random training set,
  - at least 0.15 from every exam B point and from every other exam C point.

The real BETSE simulator produces the truths. Truths and problem files are
written OUTSIDE the repo, to a sealed folder, and nothing about the knob
values is published. What goes in the repo is a commitment: the SHA-256 of
each sealed file, written before any model or agent sees a batch. After the
exams, the files are revealed and anyone can check them against the hashes.

  python make_sealed_batches.py --smoke     # 2 sims, nothing committed
  python make_sealed_batches.py             # 300 sims, ~15 min on 8 workers

Seed: 42, stream 2027 (SeedSequence([42, 2027])), so the points differ from
exam B, which used the plain seed 42 stream. CPU only, OMP 1 per worker.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "05_dpo_rlvr"))
import make_sim_problems as msp  # noqa: E402
from make_sim_problems import K_ENV, DMK_MULT, DMNA_MULT, P_NUM, P_POL, run_combo  # noqa: E402
from bioelectric import reward  # noqa: E402

SEED, STREAM = 42, 2027
N_BATCH, PER_BATCH = 20, 15
D_NODE, D_RANDOM, D_EXAMB, D_PAIR = 0.25, 0.20, 0.15, 0.15
POL_MIN_DELTA = 3.0
SEALED = os.path.join(os.path.expanduser("~"), "sealed", "cellbio_exam_c")
PUBLIC = os.path.join(HERE, "exam_c")
OUT = os.path.join(HERE, "out")


def log2d(a, b):
    return math.sqrt(sum((math.log2(x) - math.log2(y)) ** 2 for x, y in zip(a, b)))


def combos(name):
    return [tuple(json.loads(l)["combo"]) for l in open(os.path.join(OUT, name), encoding="utf-8")]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sample_points(n):
    grid = [(k, dk, dn) for k in K_ENV for dk in DMK_MULT for dn in DMNA_MULT]
    rand_train = combos("random_train_truths.jsonl")
    exam_b = combos("offgrid_truths.jsonl")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, STREAM]))
    pts = []
    while len(pts) < n:
        k = round(float(2 ** rng.uniform(math.log2(2), math.log2(40))), 1)
        dk = round(float(2 ** rng.uniform(math.log2(0.25), math.log2(4))), 2)
        dn = round(float(2 ** rng.uniform(math.log2(0.5), math.log2(2))), 2)
        c = (k, dk, dn)
        if min(log2d(c, g) for g in grid) < D_NODE:
            continue
        if min(log2d(c, g) for g in rand_train) < D_RANDOM:
            continue
        if min(log2d(c, g) for g in exam_b) < D_EXAMB:
            continue
        if pts and min(log2d(c, q) for q in pts) < D_PAIR:
            continue
        pts.append(c)
    return pts


def problems_from(truths, vb):
    probs = []
    for t in truths:
        k, dk, dn = t["combo"]
        na, v = round(150 - k, 1), t["mean_vmem_mV"]
        e = round(v)
        for i in range(2):
            probs.append({"kind": "nernst", "combo": [k, dk, dn],
                          "prompt": P_NUM[i].format(k=k, na=na, dk=dk, dn=dn),
                          "answer": f"Mean Vmem = {e:+d} mV.", "truth": {"value": e}})
        if abs(v - vb) >= POL_MIN_DELTA:
            word = "depolarizes" if v > vb else "hyperpolarizes"
            adj = "depolarized" if v > vb else "hyperpolarized"
            for i in range(2):
                probs.append({"kind": "polarity", "combo": [k, dk, dn],
                              "prompt": P_POL[i].format(k=k, na=na, dk=dk, dn=dn),
                              "answer": f"The tissue is {adj}.", "truth": {"cat": word}})
    for p in probs:
        assert reward(p["kind"], p["truth"], p["answer"]) == 1.0, p
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = SEALED + ("_smoke" if args.smoke else "")
    os.makedirs(root, exist_ok=True)
    msp.RUNS = os.path.join(root, "runs")          # keep sim folders out of the repo tree
    os.makedirs(msp.RUNS, exist_ok=True)

    pts = sample_points(N_BATCH * PER_BATCH)
    if args.smoke:
        pts = pts[:2]
    print(f"{len(pts)} points sampled")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        truths = list(ex.map(run_combo, pts))
    bad = [t for t in truths if "error" in t]
    print(f"{len(truths) - len(bad)} ok, {len(bad)} failed in {(time.time() - t0) / 60:.1f} min")
    if bad or args.smoke:
        for t in truths:
            print(t)
        if bad:
            raise SystemExit("a simulation failed; nothing written")
        return

    vb = json.load(open(os.path.join(OUT, "betse_meta.json"), encoding="utf-8"))["baseline_vmem_mV"]
    commit = {"design": {"seed": SEED, "stream": STREAM, "n_batches": N_BATCH,
                         "sims_per_batch": PER_BATCH,
                         "min_dist_log2": {"lattice": D_NODE, "random_train": D_RANDOM,
                                           "exam_b": D_EXAMB, "between_exam_c_points": D_PAIR},
                         "baseline_vmem_mV": vb, "polarity_min_delta_mV": POL_MIN_DELTA},
              "batches": []}
    for b in range(N_BATCH):
        d = os.path.join(root, f"batch_{b:02d}")
        os.makedirs(d, exist_ok=True)
        tb = truths[b::N_BATCH]
        pb = problems_from(tb, vb)
        for name, rows in (("truths.jsonl", tb), ("problems.jsonl", pb)):
            with open(os.path.join(d, name), "w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        commit["batches"].append({
            "batch": b, "n_sims": len(tb), "n_problems": len(pb),
            "sha256_truths": sha256(os.path.join(d, "truths.jsonl")),
            "sha256_problems": sha256(os.path.join(d, "problems.jsonl"))})
    os.makedirs(PUBLIC, exist_ok=True)
    with open(os.path.join(PUBLIC, "commitment.json"), "w", encoding="utf-8") as f:
        json.dump(commit, f, indent=1)
    print("commitment written;", sum(c["n_problems"] for c in commit["batches"]), "problems in",
          N_BATCH, "batches")


if __name__ == "__main__":
    main()
