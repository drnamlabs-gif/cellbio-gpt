# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8g: the 40M checkpoints and the no-LLM baselines sit exam C.

Reads the 20 sealed batches (see make_sealed_batches.py and exam_c/PLAN.md),
scores every system with the same verifier, and saves per-batch rewards to
out/exam_c/systems/<name>.json. Truth values are never written out.

Systems:
  grid_k0        bg8_final.pt   trained on the 150 grid simulations, zero-shot
  random_k0      bg8_wsr.pt     trained on 150 random points, zero-shot
  mixed_k5       bw8r_ws.pt     mixed 0-5 worked examples, five in the prompt
  mixed_k0       bw8r_ws.pt     same checkpoint, zero-shot
  constant       most common training answer per question kind
  nn1            answer of the nearest training combination (internal diagnostic)
Worked examples are the five nearest same-kind problems from the random
training set, as in exam B. Seed 42, greedy decoding.
"""

import json
import os
import sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "05_dpo_rlvr"))
from baldwin_betse import (OUT, TOKENIZER, build_prompt_ids, generate_from_ids,
                           load_problems)  # noqa: E402
from bioelectric import reward  # noqa: E402
from bpe import BPE               # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

SEALED = os.path.join(os.path.expanduser("~"), "sealed", "cellbio_exam_c")
SYS_DIR = os.path.join(OUT, "exam_c", "systems")
N_BATCH = 20


def load_batches():
    out = []
    for b in range(N_BATCH):
        p = os.path.join(SEALED, f"batch_{b:02d}", "problems.jsonl")
        out.append([json.loads(l) for l in open(p, encoding="utf-8")])
    return out


def save(name, rewards_by_batch, meta):
    os.makedirs(SYS_DIR, exist_ok=True)
    means = [float(np.mean(r)) for r in rewards_by_batch]
    allr = [x for r in rewards_by_batch for x in r]
    d = {"system": name, "batch_means": means, "n": [len(r) for r in rewards_by_batch],
         "rewards": rewards_by_batch, "mean_all": float(np.mean(allr)), **meta}
    json.dump(d, open(os.path.join(SYS_DIR, name + ".json"), "w"), indent=1)
    print(f"{name:12s} mean over problems {np.mean(allr):.3f} | mean of batch means "
          f"{np.mean(means):.3f} | batch min {min(means):.3f} max {max(means):.3f}", flush=True)


def dist(a, b):
    import math
    return sum((math.log2(x) - math.log2(y)) ** 2 for x, y in zip(a, b))


def main():
    torch.manual_seed(42)
    torch.set_num_threads(10)
    batches = load_batches()
    train = load_problems("betse_train2_tr.jsonl")
    pool = defaultdict(list)
    for p in train:
        pool[p["kind"]].append(p)

    # no-LLM baselines
    mode = {}
    for kind, ps in pool.items():
        key = (lambda p: p["truth"]["value"]) if kind == "nernst" else (lambda p: p["truth"]["cat"])
        mode[kind] = Counter(key(p) for p in ps).most_common(1)[0][0]

    def const_text(kind):
        if kind == "nernst":
            return f"Mean Vmem = {mode[kind]} mV."
        return "The tissue is " + ("depolarized." if mode[kind] == "depolarizes" else "hyperpolarized.")

    save("constant", [[reward(p["kind"], p["truth"], const_text(p["kind"])) for p in b]
                      for b in batches], {"note": "mode of train2 answers per kind"})
    save("nn1", [[reward(p["kind"], p["truth"],
                         min(pool[p["kind"]], key=lambda q: dist(q["combo"], p["combo"]))["answer"])
                  for p in b] for b in batches], {"note": "nearest training combination, internal diagnostic"})

    tok = BPE.load(TOKENIZER)
    eot = tok.vocab_size - 1
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)
    for ckpt, runs in [("bg8_final.pt", [("grid_k0", 0)]),
                       ("bg8_wsr.pt", [("random_k0", 0)]),
                       ("bw8r_ws.pt", [("mixed_k5", 5), ("mixed_k0", 0)])]:
        ck = torch.load(os.path.join(OUT, ckpt), weights_only=False)
        model = GPT(GPTConfig(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        for name, k in runs:
            rews = []
            for b in batches:
                enc = [build_prompt_ids(p, k, pool, tok) for p in b]
                outs = generate_from_ids(model, enc, device, eot, temperature=0.0)
                r = []
                for p, (_, ans) in zip(b, outs):
                    text = tok.decode(ans[:-1] if ans and ans[-1] == eot else ans)
                    r.append(reward(p["kind"], p["truth"], text))
                rews.append(r)
            save(name, rews, {"checkpoint": ckpt, "k_examples": k, "seed": 42})
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
