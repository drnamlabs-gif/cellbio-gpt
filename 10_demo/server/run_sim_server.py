# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Server-side single BETSE run + visuals (small CPU server, niced).
Usage: python run_sim_server.py <K> <DmK> <DmNa> -> one JSON line.
Paths are fixed to /opt/cellbio. Cached per combo in runs/<tag>/."""

import gzip
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

BASE = "/opt/cellbio"
BETSE = os.path.join(BASE, "venv", "bin", "betse")
TEMPLATE = os.path.join(BASE, "template")
RUNS = os.path.join(BASE, "runs")
VMIN, VMAX = -75, 0


def tag_of(k, dk, dn):
    return f"K{k}_dk{dk}_dn{dn}".replace(".", "p")


def cell_mean(arr_mem, m2c, n):
    v, c = np.zeros(n), np.zeros(n)
    np.add.at(v, m2c, arr_mem)
    np.add.at(c, m2c, 1)
    return v / c * 1000.0


def render(d, sim, cells):
    xy = np.asarray(cells.cell_centres) * 1e6
    m2c = np.asarray(cells.mem_to_cells)
    n = len(cells.cell_i)

    def draw(ax, vm, t_label):
        ax.clear()
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vm, cmap="RdBu_r", vmin=VMIN,
                        vmax=VMAX, s=110, edgecolors="white", linewidths=0.4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"tissue Vmem (mV) — {t_label}", fontsize=10)
        return sc

    vm_final = cell_mean(np.asarray(sim.vm), m2c, n)
    fig, ax = plt.subplots(figsize=(4.4, 4.0), facecolor="white")
    sc = draw(ax, vm_final, "steady state")
    fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "vmem_map.png"), dpi=110)
    plt.close(fig)

    vm_t = getattr(sim, "vm_time", None)
    has_anim = vm_t is not None and len(vm_t) > 3
    if has_anim:
        t = np.asarray(getattr(sim, "time", np.arange(len(vm_t))))
        idx = np.unique(np.linspace(0, len(vm_t) - 1, 16).astype(int))
        frames = [(cell_mean(np.asarray(vm_t[i]), m2c, n), f"t = {t[i]:.2f} s")
                  for i in idx]
        fig2, ax2 = plt.subplots(figsize=(4.4, 4.0), facecolor="white")
        sc2 = draw(ax2, frames[0][0], frames[0][1])
        fig2.colorbar(sc2, ax=ax2, shrink=0.85)

        def update(i):
            return draw(ax2, frames[i][0], frames[i][1]),

        FuncAnimation(fig2, update, frames=len(frames)).save(
            os.path.join(d, "vmem_anim.gif"), writer=PillowWriter(fps=5))
        plt.close(fig2)

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
        ax3.legend(frameon=False, fontsize=8)
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.grid(alpha=0.25, lw=0.5)
        fig3.tight_layout()
        fig3.savefig(os.path.join(d, "vmem_trace.png"), dpi=110)
        plt.close(fig3)
    return has_anim


def main():
    k = round(float(sys.argv[1]), 1)
    dk = round(float(sys.argv[2]), 2)
    dn = round(float(sys.argv[3]), 2)
    d = os.path.join(RUNS, tag_of(k, dk, dn))
    tf = os.path.join(d, "truth.json")
    if os.path.exists(tf) and os.path.exists(os.path.join(d, "vmem_map.png")):
        out = json.load(open(tf))
        out["tag"] = os.path.basename(d)
        out["anim"] = os.path.exists(os.path.join(d, "vmem_anim.gif"))
        print(json.dumps(out))
        return

    os.makedirs(d, exist_ok=True)
    for sub in ("geo", "extra_configs"):
        if not os.path.exists(os.path.join(d, sub)):
            shutil.copytree(os.path.join(TEMPLATE, sub), os.path.join(d, sub))
    for sub in ("INITS", "SIMS", "RESULTS"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    y = open(os.path.join(TEMPLATE, "sim2.yaml")).read()
    y = y.replace("extracellular K+ concentration: 5.0",
                  f"extracellular K+ concentration: {float(k)}")
    y = y.replace("extracellular Na+ concentration: 145.0",
                  f"extracellular Na+ concentration: {float(round(150 - k, 1))}")
    y = y.replace("Dm_Na: 2.0e-18", f"Dm_Na: {2.0 * dn}e-18")
    y = y.replace("Dm_K: 1.0e-18", f"Dm_K: {1.0 * dk}e-18")
    with open(os.path.join(d, "sim.yaml"), "w") as f:
        f.write(y)

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    for phase in ("seed", "init", "sim"):
        r = subprocess.run(["nice", "-n", "15", BETSE, phase, "sim.yaml"],
                           cwd=d, env=env, capture_output=True, text=True,
                           timeout=600)
        if r.returncode != 0:
            print(json.dumps({"error": (r.stdout + r.stderr)[-300:]}))
            return
    with gzip.open(os.path.join(d, "SIMS", "sim_1.betse.gz"), "rb") as f:
        sim, cells, p = pickle.load(f)
    m2c = np.asarray(cells.mem_to_cells)
    n = len(cells.cell_i)
    vm = cell_mean(np.asarray(sim.vm), m2c, n)
    has_anim = render(d, sim, cells)
    truth = {"combo": [k, dk, dn], "mean_vmem_mV": round(float(vm.mean()), 2),
             "n_cells": n, "wall_s": round(time.time() - t0, 1), "seed": 42}
    with open(tf, "w") as f:
        json.dump(truth, f)
    for sub in ("INITS", "SIMS"):
        sd = os.path.join(d, sub)
        for fn in os.listdir(sd):
            if fn.endswith(".betse.gz"):
                os.remove(os.path.join(sd, fn))
    truth["tag"] = os.path.basename(d)
    truth["anim"] = has_anim
    print(json.dumps(truth))


if __name__ == "__main__":
    main()
