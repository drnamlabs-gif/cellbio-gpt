# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8a: problems whose ground truth is the REAL BETSE simulator.

Pipeline:
  1. GRID: 150 config combos = extracellular K+ (10 values, Na+ compensated to
     keep charge balance) x Dm_K scale (5) x Dm_Na scale (3)
  2. RUN: each combo gets its own run folder (template = validate_one/, which
     is `ion profile: custom`); betse seed+init+sim via subprocess pool
     (8 workers, OMP_NUM_THREADS=1 each — CPU rule: <=12 threads total)
  3. TRUTH: per-combo steady-state mean tissue Vmem (mV) read from the sim
     pickle; sim/init pickles are then DELETED (26MB each, regenerable in
     ~13s from the kept yaml) — truths cached in out/sim_truths.jsonl
  4. PROBLEMS: numeric (predict mean Vmem, kind "nernst" so stage-5b's graded
     reward() applies unchanged) + categorical (depolarized/hyperpolarized vs
     the baseline combo, kind "polarity"); 2 phrasings each; combos with
     |Vmem - baseline| < 3 mV are skipped for the categorical family.
  5. SPLIT: combo-level 15% held-out via md5 (same recipe as stage 5b).
     Self-test: every gold answer must score reward == 1.0.

Outputs out/: sim_truths.jsonl, betse_train.jsonl, betse_val.jsonl,
betse_meta.json. Seed 42. --smoke runs 2 combos end-to-end, no problem files.
"""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "05_dpo_rlvr"))
from bioelectric import reward  # noqa: E402  (reused verbatim)

SEED = 42
OUT = os.path.join(HERE, "out")
RUNS = os.path.join(HERE, "runs")
TEMPLATE = os.path.join(HERE, "validate_one")
BETSE = os.path.join(HERE, "..", ".venv-betse", "Scripts", "betse.exe")

K_ENV = [2, 3, 4, 5, 6, 8, 12, 16, 25, 40]      # mM; Na+ = 150 - K keeps charge balance
DMK_MULT = [0.25, 0.5, 1, 2, 4]                 # scales Dm_K 1.0e-18
DMNA_MULT = [0.5, 1, 2]                         # scales Dm_Na 2.0e-18
BASELINE = (5, 1, 1)
VAL_FRAC, POL_MIN_DELTA = 0.15, 3.0

P_NUM = [
    "A BETSE tissue simulation runs with extracellular K+ = {k} mM, "
    "extracellular Na+ = {na} mM, K+ membrane permeability scaled x{dk} and "
    "Na+ permeability scaled x{dn}. What steady-state mean Vmem (mV) does the tissue reach?",
    "In a bioelectric tissue sim (BETSE), the bath has {k} mM K+ and {na} mM Na+; "
    "membrane Dm_K is scaled x{dk} and Dm_Na x{dn}. Predict the tissue's mean Vmem in mV.",
]
P_POL = [
    "A BETSE tissue simulation runs with extracellular K+ = {k} mM, Na+ = {na} mM, "
    "Dm_K scaled x{dk}, Dm_Na x{dn}. Relative to the baseline run (K+ = 5 mM, "
    "Na+ = 145 mM, default permeabilities), does the tissue sit depolarized or hyperpolarized?",
    "Compared with baseline (5 mM K+, 145 mM Na+, default channel permeabilities), is a "
    "BETSE tissue with {k} mM K+, {na} mM Na+, Dm_K x{dk}, Dm_Na x{dn} depolarized or hyperpolarized?",
]


def combo_tag(k, dk, dn):
    return f"K{k}_dk{dk}_dn{dn}".replace(".", "p")


def make_run_dir(k, dk, dn):
    d = os.path.join(RUNS, combo_tag(k, dk, dn))
    if os.path.exists(os.path.join(d, "truth.json")):
        return d                                   # already simulated
    os.makedirs(d, exist_ok=True)
    for sub in ("geo", "extra_configs"):
        if not os.path.exists(os.path.join(d, sub)):
            shutil.copytree(os.path.join(TEMPLATE, sub), os.path.join(d, sub))
    for sub in ("INITS", "SIMS", "RESULTS"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    y = open(os.path.join(TEMPLATE, "sim2.yaml"), encoding="utf-8").read()
    y = y.replace("extracellular K+ concentration: 5.0",
                  f"extracellular K+ concentration: {float(k)}")
    y = y.replace("extracellular Na+ concentration: 145.0",
                  f"extracellular Na+ concentration: {float(150 - k)}")
    y = y.replace("Dm_Na: 2.0e-18", f"Dm_Na: {2.0 * dn}e-18")
    y = y.replace("Dm_K: 1.0e-18", f"Dm_K: {1.0 * dk}e-18")
    with open(os.path.join(d, "sim.yaml"), "w", encoding="utf-8") as f:
        f.write(y)
    return d


def run_combo(args):
    k, dk, dn = args
    d = make_run_dir(k, dk, dn)
    tf = os.path.join(d, "truth.json")
    if os.path.exists(tf):
        return json.load(open(tf, encoding="utf-8"))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    for phase in ("seed", "init", "sim"):
        r = subprocess.run([BETSE, phase, "sim.yaml"], cwd=d, env=env,
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return {"combo": [k, dk, dn], "error": r.stdout[-400:] + r.stderr[-400:]}
    with gzip.open(os.path.join(d, "SIMS", "sim_1.betse.gz"), "rb") as f:
        sim, cells, p = pickle.load(f)
    vm_mem = np.asarray(sim.vm)
    m2c = np.asarray(cells.mem_to_cells)
    n = len(cells.cell_i)
    vm, cnt = np.zeros(n), np.zeros(n)
    np.add.at(vm, m2c, vm_mem)
    np.add.at(cnt, m2c, 1)
    vm = vm / cnt * 1000.0
    truth = {"combo": [k, dk, dn], "mean_vmem_mV": round(float(vm.mean()), 2),
             "min_mV": round(float(vm.min()), 2), "max_mV": round(float(vm.max()), 2),
             "n_cells": n, "wall_s": round(time.time() - t0, 1), "seed": SEED}
    with open(tf, "w", encoding="utf-8") as f:
        json.dump(truth, f)
    for sub in ("INITS", "SIMS"):                   # 26MB/run, regenerable in ~13s
        for fn in os.listdir(os.path.join(d, sub)):
            if fn.endswith(".betse.gz"):
                os.remove(os.path.join(d, sub, fn))
    return truth


def is_val(k, dk, dn):
    h = hashlib.md5(f"betse|{k}|{dk}|{dn}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < VAL_FRAC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)

    combos = [(k, dk, dn) for k in K_ENV for dk in DMK_MULT for dn in DMNA_MULT]
    if args.smoke:
        combos = [BASELINE, (25, 1, 1)]
    print(f"{len(combos)} combos, {args.workers} workers "
          f"(~{len(combos) * 14 / args.workers / 60:.1f} min)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        truths = list(ex.map(run_combo, combos))
    errs = [t for t in truths if "error" in t]
    truths = [t for t in truths if "error" not in t]
    print(f"{len(truths)} ok, {len(errs)} failed in {(time.time() - t0) / 60:.1f} min")
    for e in errs[:3]:
        print("FAILED", e["combo"], e["error"][-200:])
    with open(os.path.join(OUT, "sim_truths.jsonl"), "w", encoding="utf-8") as f:
        for t in truths:
            f.write(json.dumps(t) + "\n")
    if args.smoke:
        for t in truths:
            print(t)
        return

    base = next(t for t in truths if tuple(t["combo"]) == BASELINE)
    vb = base["mean_vmem_mV"]
    print(f"baseline {BASELINE} mean Vmem {vb} mV")
    train, val = [], []
    for t in truths:
        k, dk, dn = t["combo"]
        na, v = 150 - k, t["mean_vmem_mV"]
        probs = []
        for i in range(2):
            e = round(v)
            probs.append({"kind": "nernst", "ion": "K+", "cin": dk, "cout": dn,
                          "cout2": k, "combo": [k, dk, dn],
                          "prompt": P_NUM[i].format(k=k, na=na, dk=dk, dn=dn),
                          "answer": f"Mean Vmem = {e:+d} mV.", "truth": {"value": e}})
        if abs(v - vb) >= POL_MIN_DELTA and tuple(t["combo"]) != BASELINE:
            word = "depolarizes" if v > vb else "hyperpolarizes"
            adj = "depolarized" if v > vb else "hyperpolarized"
            for i in range(2):
                probs.append({"kind": "polarity", "ion": "K+", "cin": dk, "cout": dn,
                              "cout2": k, "combo": [k, dk, dn],
                              "prompt": P_POL[i].format(k=k, na=na, dk=dk, dn=dn),
                              "answer": f"The tissue is {adj}.", "truth": {"cat": word}})
        for p in probs:
            assert reward(p["kind"], p["truth"], p["answer"]) == 1.0, p
        (val if is_val(k, dk, dn) else train).extend(probs)

    with open(os.path.join(OUT, "betse_train.jsonl"), "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(p) + "\n")
    with open(os.path.join(OUT, "betse_val.jsonl"), "w", encoding="utf-8") as f:
        for p in val:
            f.write(json.dumps(p) + "\n")
    meta = {"seed": SEED, "date": "2026-10-23", "n_combos": len(truths),
            "n_train": len(train), "n_val": len(val),
            "baseline_vmem_mV": vb, "val_frac": VAL_FRAC,
            "grid": {"K_env": K_ENV, "DmK_mult": DMK_MULT, "DmNa_mult": DMNA_MULT},
            "vmem_range": [min(t["mean_vmem_mV"] for t in truths),
                           max(t["mean_vmem_mV"] for t in truths)]}
    with open(os.path.join(OUT, "betse_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
