# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8g: paired statistics for exam C, as laid out in exam_c/PLAN.md.

The unit is the batch (n = 20). For every comparison we take the per-batch
difference in mean reward, then report the mean difference, a 95% percentile
bootstrap interval over batches (10,000 resamples, seed 42) and a two-sided
Wilcoxon signed-rank p-value. Nothing is dropped and no comparison is added
after seeing the data: the list below is the planned one.

  python exam_c_stats.py            # writes out/exam_c/stats.json and stats.md
"""

import json
import os

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
SYS_DIR = os.path.join(HERE, "out", "exam_c", "systems")
OUTD = os.path.join(HERE, "out", "exam_c")

PRIMARY = [("mixed_k5", "agent_k5")]
SECONDARY = [(s, "constant") for s in ("grid_k0", "random_k0", "mixed_k0", "mixed_k5",
                                       "agent_k0", "agent_k5")] + \
            [("mixed_k5", "mixed_k0"), ("agent_k5", "agent_k0"),
             ("mixed_k5", "random_k0"), ("agent_k5", "nn1"), ("mixed_k5", "nn1")]


def load(name):
    p = os.path.join(SYS_DIR, name + ".json")
    return json.load(open(p))["batch_means"] if os.path.exists(p) else None


def compare(a, b, rng, n_boot=10000):
    x, y = np.array(load(a)), np.array(load(b))
    d = x - y
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"a": a, "b": b, "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "diff": float(d.mean()), "ci95": [float(lo), float(hi)], "wilcoxon_p": p,
            "a_wins": int((d > 0).sum()), "b_wins": int((d < 0).sum()), "ties": int((d == 0).sum()),
            "n_batches": int(len(d))}


def main():
    rng = np.random.default_rng(42)
    names = sorted(f[:-5] for f in os.listdir(SYS_DIR) if f.endswith(".json"))
    rows = {"primary": [], "secondary": []}
    for kind, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for a, b in pairs:
            if load(a) is None or load(b) is None:
                print("skipped (missing system):", a, b)
                continue
            rows[kind].append(compare(a, b, rng))
    means = {n: float(np.mean(load(n))) for n in names}
    json.dump({"means_of_batch_means": means, **rows}, open(os.path.join(OUTD, "stats.json"), "w"), indent=1)
    lines = ["| system | mean reward over 20 batches |", "|---|---|"]
    lines += [f"| {n} | {m:.3f} |" for n, m in sorted(means.items(), key=lambda t: -t[1])]
    lines += ["", "| comparison | diff | 95% CI | Wilcoxon p | wins a/b |", "|---|---|---|---|---|"]
    for kind in ("primary", "secondary"):
        for r in rows[kind]:
            lines.append(f"| {'**' if kind == 'primary' else ''}{r['a']} vs {r['b']}{'**' if kind == 'primary' else ''} "
                         f"| {r['diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] "
                         f"| {r['wilcoxon_p']:.4f} | {r['a_wins']}/{r['b_wins']} |")
    open(os.path.join(OUTD, "stats.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
