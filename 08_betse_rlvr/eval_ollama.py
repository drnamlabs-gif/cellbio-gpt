# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8d (part 2): pretrained LLMs (Ollama) take the same BETSE sim exam.

Same 98 held-out problems, same verifier (bioelectric.reward). Two conditions:
  zero-shot : the raw question + a strict answer-format instruction
  5-shot    : plus the 5 nearest TRAIN combos (same kind) as worked Q/A turns —
              the ICL analogue of our k=5 scaffold / the 1-NN phone book.
Greedy (temperature 0), thinking disabled where supported.
Saves out/ollama_baselines.json (+ merged figure via plot afterwards).
"""

import json
import math
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.join(HERE, "..", "05_dpo_rlvr"))
from bioelectric import reward as score  # noqa: E402

MODELS = ["phi3:latest", "mistral:latest", "gemma3:latest",
          "qwen3.5:latest", "gpt-oss:20b", "qwen3.6:latest"]
URL = "http://localhost:11434/api/chat"

SYSTEM = (
    "You are a bioelectricity expert answering exam questions about BETSE "
    "tissue simulations. Answer with EXACTLY ONE line and nothing else.\n"
    "For numeric Vmem questions: 'Mean Vmem = <integer> mV.' (e.g. 'Mean Vmem = -58 mV.')\n"
    "For depolarized/hyperpolarized questions: 'The tissue is depolarized.' or "
    "'The tissue is hyperpolarized.'\n"
    "No reasoning, no explanation, no units other than mV."
)


def load(name):
    return [json.loads(l) for l in open(os.path.join(OUT, name), encoding="utf-8")]


def dist(a, b):
    return sum((math.log2(x) - math.log2(y)) ** 2 for x, y in zip(a, b))


def chat(model, messages, think_field):
    # gpt-oss routes everything through a reasoning channel: give it a low
    # thinking budget + room, and read the final content as usual
    # deepseek-r1 always thinks (500 to 3000 tokens) before the one-line answer
    npred = 3000 if "deepseek" in model else (1024 if "gpt-oss" in model else 300)
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.0, "num_predict": npred},
            "keep_alive": "2m"}
    if think_field is not None:
        body["think"] = think_field
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"]


def ask(model, p, shots):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in shots:
        msgs.append({"role": "user", "content": ex["prompt"]})
        msgs.append({"role": "assistant", "content": ex["answer"]})
    msgs.append({"role": "user", "content": p["prompt"]})
    tries = ("low", None) if "gpt-oss" in model else (False, None)
    for think in tries:  # some models reject the think field
        try:
            return chat(model, msgs, think)
        except Exception as e:
            err = e
    raise err


def main():
    smoke = "--smoke" in sys.argv
    models = MODELS[:2] if smoke else MODELS
    train, val = load("betse_train.jsonl"), load("betse_val.jsonl")
    if smoke:
        val = val[:4]
    # deterministic nearest-5 same-kind shots per val problem (as baldwin eval)
    shots5 = {}
    for i, p in enumerate(val):
        pool = sorted((q for q in train if q["kind"] == p["kind"]),
                      key=lambda q: dist(q["combo"], p["combo"]))
        shots5[i] = pool[:5]

    path = os.path.join(OUT, "ollama_baselines_smoke.json" if smoke
                        else "ollama_baselines.json")
    results = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for model in models:
        for cond, get_shots in [("0-shot", lambda i: []),
                                ("5-shot", lambda i: shots5[i])]:
            key = f"{model} {cond}"
            if key in results:
                print(f"skip {key} (done)")
                continue
            t0 = time.time()
            rs, by, preds = [], {}, []
            for i, p in enumerate(val):
                try:
                    text = ask(model, p, get_shots(i))
                except Exception as e:
                    print(f"  {key} item {i} ERROR {e}")
                    text = ""
                r = score(p["kind"], p["truth"], text)
                rs.append(r)
                by.setdefault(p["kind"], []).append(r)
                preds.append({"i": i, "text": text[-200:], "r": r})
                if i % 20 == 0:
                    print(f"  {key} {i}/98 running mean {sum(rs)/len(rs):.3f}", flush=True)
            results[key] = {
                "val_reward": sum(rs) / len(rs),
                "exact": sum(r == 1.0 for r in rs) / len(rs),
                "by_kind": {k: sum(v) / len(v) for k, v in by.items()},
                "secs": round(time.time() - t0, 1),
                "preds_tail": preds[:3],
            }
            print(f"{key:28s} val {results[key]['val_reward']:.3f} "
                  f"exact {results[key]['exact']:.3f} {results[key]['by_kind']} "
                  f"({results[key]['secs']}s)", flush=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
