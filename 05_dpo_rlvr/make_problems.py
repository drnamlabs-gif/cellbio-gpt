# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 5b-data: generate the verifiable bioelectric problem sets.

Combo-level split: every (kind, ion, cin, cout[, cout2]) combination is hashed;
15% of COMBOS go to val and never appear in warm-start or GRPO training, so
val accuracy measures real generalization, not recall of a seen question.

Outputs (out/): rlvr_ws.jsonl (warm-start SFT, gold answers included),
rlvr_train.jsonl (GRPO prompts + truth for the verifier), rlvr_val.jsonl,
rlvr_meta.json. Self-test: every gold answer must get reward 1.0.
Seed 42.
"""

import hashlib
import itertools
import json
import os

import numpy as np

from bioelectric import IONS, make_problem, reward

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SEED = 42
VAL_FRAC = 0.15


def combo_is_val(key):
    h = hashlib.md5(key.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < VAL_FRAC


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    train, val = [], []

    for ion, spec in IONS.items():
        for cin, cout in itertools.product(spec["ins"], spec["outs"]):
            key = f"nernst|{ion}|{cin}|{cout}"
            v = int(rng.integers(0, 2))
            p = make_problem("nernst", ion, cin, cout, variant=v)
            (val if combo_is_val(key) else train).append(p)
        for cin in spec["ins"]:
            for a, b in itertools.permutations(spec["outs"], 2):
                for kind in ("deltaE", "polarity"):
                    key = f"{kind}|{ion}|{cin}|{a}|{b}"
                    v = int(rng.integers(0, 2))
                    p = make_problem(kind, ion, cin, a, cout2=b, variant=v)
                    (val if combo_is_val(key) else train).append(p)

    # self-test: gold answers must be scored 1.0 by the verifier
    for p in train[:2000] + val[:500]:
        r = reward(p["kind"], p["truth"], p["answer"])
        assert r == 1.0, (p, r)
    print("self-test PASS: gold answers all score 1.0")

    rng.shuffle(train)
    rng.shuffle(val)
    n_ws = 3000
    ws, grpo_train = train[:n_ws], train   # GRPO may revisit ws combos; val never
    if len(val) > 1000:
        val = val[:1000]

    for name, part in (("rlvr_ws", ws), ("rlvr_train", grpo_train), ("rlvr_val", val)):
        with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for p in part:
                f.write(json.dumps(p) + "\n")
    counts = lambda xs: {k: sum(1 for p in xs if p["kind"] == k)
                         for k in ("nernst", "deltaE", "polarity")}
    meta = {"seed": SEED, "date": "2026-10-23", "val_frac_combos": VAL_FRAC,
            "n_ws": len(ws), "n_grpo_train": len(grpo_train), "n_val": len(val),
            "ws_counts": counts(ws), "train_counts": counts(grpo_train),
            "val_counts": counts(val)}
    json.dump(meta, open(os.path.join(OUT, "rlvr_meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))
    for p in ws[:3]:
        print(f"\n[{p['kind']}] {p['prompt']}\n  GOLD: {p['answer']}")


if __name__ == "__main__":
    main()
