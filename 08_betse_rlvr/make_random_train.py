# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8e (case 2): a DIVERSE training set — random off-lattice points.

Case 1 (done): train on a 150-node lattice -> the model becomes a soft lookup
table and collapses on the off-grid exam (0.539 < 1-NN 0.793).
Case 2 (this): same 150-sim budget, but training points sampled log-uniformly
across the SAME knob ranges as the off-grid test — decimals everywhere, no
lattice to snap to. Contamination guard: every training point stays >=0.2
log2 from every off-grid TEST point (and >=0.1 from each other).

Outputs out/: random_train_truths.jsonl, betse_train2.jsonl,
random_train_meta.json. Seed 43 (train points must not replay the seed-42
test stream). --smoke = 2 combos.
"""

import argparse
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
from make_sim_problems import OUT, P_NUM, P_POL, run_combo  # noqa: E402
from bioelectric import reward  # noqa: E402

SEED = 43        # NOT 42: test points were drawn with 42 from the same ranges
N_POINTS = 150
MIN_TEST_DIST = 0.20
MIN_PAIR_DIST = 0.10
POL_MIN_DELTA = 3.0


def log2d(a, b):
    return math.sqrt(sum((math.log2(x) - math.log2(y)) ** 2 for x, y in zip(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    test_pts = [tuple(json.loads(l)["combo"]) for l in
                open(os.path.join(OUT, "offgrid_truths.jsonl"), encoding="utf-8")]
    rng = np.random.default_rng(SEED)
    pts = []
    while len(pts) < N_POINTS:
        k = round(float(2 ** rng.uniform(math.log2(2), math.log2(40))), 1)
        dk = round(float(2 ** rng.uniform(math.log2(0.25), math.log2(4))), 2)
        dn = round(float(2 ** rng.uniform(math.log2(0.5), math.log2(2))), 2)
        c = (k, dk, dn)
        if min(log2d(c, t) for t in test_pts) < MIN_TEST_DIST:
            continue
        if pts and min(log2d(c, q) for q in pts) < MIN_PAIR_DIST:
            continue
        pts.append(c)
    d_test = [min(log2d(c, t) for t in test_pts) for c in pts]
    print(f"{len(pts)} random train points; dist-to-nearest-TEST-point "
          f"min {min(d_test):.2f} median {np.median(d_test):.2f}")
    if args.smoke:
        pts = pts[:2]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        truths = list(ex.map(run_combo, pts))
    errs = [t for t in truths if "error" in t]
    truths = [t for t in truths if "error" not in t]
    print(f"{len(truths)} ok, {len(errs)} failed in {(time.time() - t0) / 60:.1f} min")
    for e in errs[:3]:
        print("FAILED", e["combo"], e["error"][-200:])
    with open(os.path.join(OUT, "random_train_truths.jsonl"), "w", encoding="utf-8") as f:
        for t in truths:
            f.write(json.dumps(t) + "\n")
    if args.smoke:
        for t in truths:
            print(t)
        return

    vb = json.load(open(os.path.join(OUT, "betse_meta.json"),
                        encoding="utf-8"))["baseline_vmem_mV"]
    probs = []
    for t in truths:
        k, dk, dn = t["combo"]
        na, v = round(150 - k, 1), t["mean_vmem_mV"]
        e = round(v)
        for i in range(2):
            probs.append({"kind": "nernst", "ion": "K+", "cin": dk, "cout": dn,
                          "cout2": k, "combo": [k, dk, dn],
                          "prompt": P_NUM[i].format(k=k, na=na, dk=dk, dn=dn),
                          "answer": f"Mean Vmem = {e:+d} mV.", "truth": {"value": e}})
        if abs(v - vb) >= POL_MIN_DELTA:
            word = "depolarizes" if v > vb else "hyperpolarizes"
            adj = "depolarized" if v > vb else "hyperpolarized"
            for i in range(2):
                probs.append({"kind": "polarity", "ion": "K+", "cin": dk, "cout": dn,
                              "cout2": k, "combo": [k, dk, dn],
                              "prompt": P_POL[i].format(k=k, na=na, dk=dk, dn=dn),
                              "answer": f"The tissue is {adj}.", "truth": {"cat": word}})
    for p in probs:
        assert reward(p["kind"], p["truth"], p["answer"]) == 1.0, p
    with open(os.path.join(OUT, "betse_train2.jsonl"), "w", encoding="utf-8") as f:
        for p in probs:
            f.write(json.dumps(p) + "\n")
    meta = {"seed": SEED, "n_points": len(truths), "n_problems": len(probs),
            "min_test_dist_log2": MIN_TEST_DIST,
            "dist_to_test": {"min": round(min(d_test), 3),
                             "median": round(float(np.median(d_test)), 3)},
            "vmem_range": [min(t["mean_vmem_mV"] for t in truths),
                           max(t["mean_vmem_mV"] for t in truths)]}
    with open(os.path.join(OUT, "random_train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
