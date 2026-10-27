# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8g: open models (Ollama) sit exam C.

Same prompts and answer format as eval_ollama.py (one-line answers, greedy,
thinking off where supported), same five nearest worked examples as the other
exam C systems, same verifier. Only questions and worked examples reach the
model; truths stay in the sealed folder. Progress is saved per batch so a
stopped run resumes.

  python eval_exam_c_ollama.py --smoke                 # 3 problems per model
  python eval_exam_c_ollama.py [--models a,b] [--conds five,zero]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "05_dpo_rlvr"))
from bioelectric import reward  # noqa: E402
from eval_agents import shots_for, load  # noqa: E402
from eval_exam_c import load_batches, save, OUT  # noqa: E402
from eval_ollama import ask  # noqa: E402

MODELS = {"qwen35": "qwen3.5:latest", "qwen36": "qwen3.6:latest",
          "gptoss20b": "gpt-oss:20b", "deepseekr1_8b": "deepseek-r1:8b"}
PART = os.path.join(OUT, "exam_c", "ollama_partial")


def run(name, tag, cond, batches, pool, limit=None):
    os.makedirs(PART, exist_ok=True)
    pf = os.path.join(PART, f"{name}_{cond}.json")
    done = json.load(open(pf)) if os.path.exists(pf) and not limit else {}
    rews, empty = [], 0
    t0 = time.time()
    for b, probs in enumerate(batches):
        if str(b) in done:
            rews.append(done[str(b)]["r"])
            continue
        r, tx = [], []
        for p in (probs[:limit] if limit else probs):
            try:
                text = ask(tag, p, shots_for(p, pool) if cond == "five" else [])
            except Exception as e:
                text = ""
                print("  ERROR", name, b, str(e)[:80], flush=True)
            r.append(reward(p["kind"], p["truth"], text))
            empty += (text.strip() == "")
            tx.append(text[-120:])
        rews.append(r)
        if limit:
            print(name, cond, "sample:", tx[:3], r[:3], f"{time.time() - t0:.1f}s")
            return None
        done[str(b)] = {"r": r, "empty": sum(1 for x in tx if x.strip() == "")}
        json.dump(done, open(pf, "w"))
        print(f"{name} {cond} batch {b:02d} mean {np.mean(r):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    n_empty = sum(v.get("empty", 0) for v in done.values())
    save(f"{name}_{'k5' if cond == 'five' else 'k0'}", rews, {"model": tag, "k_examples": 5 if cond == "five" else 0, "empty_answers": n_empty, "token_budget": 3000 if "deepseek" in tag else (1024 if "gpt-oss" in tag else 300)})
    return rews


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--conds", default="five,zero")
    a = ap.parse_args()
    batches = load_batches()
    pool = load("betse_train2_tr.jsonl")
    for cond in a.conds.split(","):
        for name in a.models.split(","):
            run(name, MODELS[name], cond, batches[:1] if a.smoke else batches, pool, limit=3 if a.smoke else None)


if __name__ == "__main__":
    main()
