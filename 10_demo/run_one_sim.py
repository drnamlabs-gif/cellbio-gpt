# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Run ONE real BETSE simulation for the demo AND render what the tissue did:
  vmem_map.png  — final per-cell Vmem map
  vmem_anim.gif — Vmem evolving over simulated time (if time series available)
Usage: python run_one_sim.py <K_mM> <DmK_mult> <DmNa_mult>  (betse venv, 3.11)
Prints one JSON line: truth + image filenames. Cached per combo in runs/<tag>.
"""

import gzip
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "08_betse_rlvr"))
sys.path.insert(0, os.path.join(HERE, "..", "05_dpo_rlvr"))
from make_sim_problems import BETSE, make_run_dir  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

VMIN, VMAX = -75, 0


def cell_mean(arr_mem, m2c, n):
    v, c = np.zeros(n), np.zeros(n)
    np.add.at(v, m2c, arr_mem)
    np.add.at(c, m2c, 1)
    return v / c * 1000.0


def render(d, sim, cells):
    xy = np.asarray(cells.cell_centres) * 1e6
    m2c = np.asarray(cells.mem_to_cells)
    n = len(cells.cell_i)
    vm_final = cell_mean(np.asarray(sim.vm), m2c, n)

    def draw(ax, vm, t_label):
        ax.clear()
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vm, cmap="RdBu_r", vmin=VMIN,
                        vmax=VMAX, s=110, edgecolors="white", linewidths=0.4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"tissue Vmem (mV) — {t_label}", fontsize=10)
        return sc

    fig, ax = plt.subplots(figsize=(4.4, 4.0), facecolor="white")
    sc = draw(ax, vm_final, "steady state")
    fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "vmem_map.png"), dpi=110)
    plt.close(fig)

    frames = []
    vm_t = getattr(sim, "vm_time", None)
    if vm_t is not None and len(vm_t) > 3:
        t = np.asarray(getattr(sim, "time", np.arange(len(vm_t))))
        idx = np.unique(np.linspace(0, len(vm_t) - 1, 20).astype(int))
        frames = [(cell_mean(np.asarray(vm_t[i]), m2c, n), f"t = {t[i]:.2f} s")
                  for i in idx]
    if frames:
        fig2, ax2 = plt.subplots(figsize=(4.4, 4.0), facecolor="white")
        sc2 = draw(ax2, frames[0][0], frames[0][1])
        fig2.colorbar(sc2, ax=ax2, shrink=0.85)

        def update(i):
            s = draw(ax2, frames[i][0], frames[i][1])
            return s,

        anim = FuncAnimation(fig2, update, frames=len(frames))
        anim.save(os.path.join(d, "vmem_anim.gif"), writer=PillowWriter(fps=5))
        plt.close(fig2)

    # Vmem-vs-time trace (mean line + min/max band across cells)
    if vm_t is not None and len(vm_t) > 3:
        t = np.asarray(getattr(sim, "time", np.arange(len(vm_t))))
        means, mins, maxs = [], [], []
        for v in vm_t:
            cv = cell_mean(np.asarray(v), m2c, n)
            means.append(cv.mean()); mins.append(cv.min()); maxs.append(cv.max())
        fig3, ax3 = plt.subplots(figsize=(4.6, 3.2), facecolor="white")
        ax3.fill_between(t[:len(means)], mins, maxs, color="#0E7C86", alpha=0.18,
                         label="cell min-max")
        ax3.plot(t[:len(means)], means, color="#0E7C86", lw=2.2, label="tissue mean")
        ax3.axhline(means[-1], color="#B3403A", ls="--", lw=1.2)
        ax3.annotate(f"{means[-1]:+.1f} mV", (t[len(means) - 1], means[-1]),
                     xytext=(-56, 7), textcoords="offset points",
                     color="#B3403A", fontsize=10, weight="bold")
        ax3.set_xlabel("simulated time (s)", fontsize=9)
        ax3.set_ylabel("Vmem (mV)", fontsize=9)
        ax3.set_title("membrane potential over time", fontsize=10)
        ax3.legend(frameon=False, fontsize=8, loc="best")
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.grid(alpha=0.25, lw=0.5)
        fig3.tight_layout()
        fig3.savefig(os.path.join(d, "vmem_trace.png"), dpi=110)
        plt.close(fig3)
    return bool(frames)


def main():
    k = round(float(sys.argv[1]), 1)
    dk = round(float(sys.argv[2]), 2)
    dn = round(float(sys.argv[3]), 2)
    d = make_run_dir(k, dk, dn)
    tf = os.path.join(d, "truth.json")
    mapf, animf = os.path.join(d, "vmem_map.png"), os.path.join(d, "vmem_anim.gif")
    if os.path.exists(tf) and os.path.exists(mapf):
        out = json.load(open(tf, encoding="utf-8"))
        out["tag"] = os.path.basename(d)
        out["anim"] = os.path.exists(animf)
        print(json.dumps(out))
        return

    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    t0 = time.time()
    pk = os.path.join(d, "SIMS", "sim_1.betse.gz")
    if not os.path.exists(pk):
        for phase in ("seed", "init", "sim"):
            r = subprocess.run([BETSE, phase, "sim.yaml"], cwd=d, env=env,
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                print(json.dumps({"combo": [k, dk, dn],
                                  "error": r.stdout[-300:] + r.stderr[-300:]}))
                return
    with gzip.open(pk, "rb") as f:
        sim, cells, p = pickle.load(f)
    m2c = np.asarray(cells.mem_to_cells)
    n = len(cells.cell_i)
    vm = cell_mean(np.asarray(sim.vm), m2c, n)
    has_anim = render(d, sim, cells)
    truth = {"combo": [k, dk, dn], "mean_vmem_mV": round(float(vm.mean()), 2),
             "min_mV": round(float(vm.min()), 2), "max_mV": round(float(vm.max()), 2),
             "n_cells": n, "wall_s": round(time.time() - t0, 1), "seed": 42}
    with open(tf, "w", encoding="utf-8") as f:
        json.dump(truth, f)
    for sub in ("INITS", "SIMS"):
        sd = os.path.join(d, sub)
        if os.path.isdir(sd):
            for fn in os.listdir(sd):
                if fn.endswith(".betse.gz"):
                    os.remove(os.path.join(sd, fn))
    truth["tag"] = os.path.basename(d)
    truth["anim"] = has_anim
    print(json.dumps(truth))


if __name__ == "__main__":
    main()
