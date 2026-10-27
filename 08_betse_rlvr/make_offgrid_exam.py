# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8e: the OFF-GRID exam — kill the 1-NN crutch by moving the test
points off the training lattice.

The original split hid random grid NODES, so every test point had an adjacent
trained neighbor and a no-LLM 1-NN lookup scored 0.927: that ceiling was a
property of the exam, not of any model. Here we sample 60 knob combos
log-uniformly INSIDE the grid's range but AWAY from every node (min log2
distance 0.25), run the real BETSE sim for each, and emit test-only problems.
Nothing is retrained: same models, same phone book, harder questions.

Falsifiable prediction (recorded before running): 1-NN drops hard; if the
40M model holds, its margin is real interpolation; if it drops in lockstep,
the honest verdict is that it, too, was mostly looking things up.

Outputs out/: offgrid_truths.jsonl, betse_offgrid.jsonl, offgrid_meta.json.
--smoke = 2 combos, no problem files. CPU only (8 workers x OMP 1).
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
from make_sim_problems import (K_ENV, DMK_MULT, DMNA_MULT, OUT, P_NUM, P_POL,
                               run_combo)  # noqa: E402
from bioelectric import reward  # noqa: E402

SEED = 42
N_POINTS = 60
MIN_NODE_DIST = 0.25     # log2 units, to the NEAREST grid node (all 3 knobs)
MIN_PAIR_DIST = 0.15     # between sampled points
POL_MIN_DELTA = 3.0


def log2d(a, b):
    return math.sqrt(sum((math.log2(x) - math.log2(y)) ** 2 for x, y in zip(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    grid = [(k, dk, dn) for k in K_ENV for dk in DMK_MULT for dn in DMNA_MULT]
    rng = np.random.default_rng(SEED)
    pts = []
    while len(pts) < N_POINTS:
        k = round(float(2 ** rng.uniform(math.log2(2), math.log2(40))), 1)
        dk = round(float(2 ** rng.uniform(math.log2(0.25), math.log2(4))), 2)
        dn = round(float(2 ** rng.uniform(math.log2(0.5), math.log2(2))), 2)
        c = (k, dk, dn)
        if min(log2d(c, g) for g in grid) < MIN_NODE_DIST:
            continue
        if pts and min(log2d(c, q) for q in pts) < MIN_PAIR_DIST:
            continue
        pts.append(c)
    d_node = [min(log2d(c, g) for g in grid) for c in pts]
    print(f"{len(pts)} off-grid points; dist-to-nearest-node "
          f"min {min(d_node):.2f} median {np.median(d_node):.2f} max {max(d_node):.2f}")
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
    with open(os.path.join(OUT, "offgrid_truths.jsonl"), "w", encoding="utf-8") as f:
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
    with open(os.path.join(OUT, "betse_offgrid.jsonl"), "w", encoding="utf-8") as f:
        for p in probs:
            f.write(json.dumps(p) + "\n")
    meta = {"seed": SEED, "n_points": len(truths), "n_problems": len(probs),
            "min_node_dist_log2": MIN_NODE_DIST,
            "dist_to_node": {"min": round(min(d_node), 3),
                             "median": round(float(np.median(d_node)), 3),
                             "max": round(max(d_node), 3)},
            "vmem_range": [min(t["mean_vmem_mV"] for t in truths),
                           max(t["mean_vmem_mV"] for t in truths)]}
    with open(os.path.join(OUT, "offgrid_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
