# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Track B, stage B0: an untouched open instruct model on Track A's exams.

Scores exam A (98 grid-holdout problems) and exam B (234 sealed off-grid)
with the SAME verifier as Track A (bioelectric.reward), zero-shot, using the
same strict answer-format system prompt the Ollama baselines got. Then shows
free long answers to the long-question set and one multi-turn dialogue, so
the capability gap Track A could not close is visible side by side.

Usage: python b0_baseline.py [--model Qwen/Qwen2.5-1.5B-Instruct]
Saves out/b0_<model-tag>.json. Seed 42, greedy.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SIM_OUT = os.path.join(HERE, "..", "08_betse_rlvr", "out")
sys.path.insert(0, os.path.join(HERE, "..", "05_dpo_rlvr"))
from bioelectric import reward as score  # noqa: E402

SYSTEM = (
    "You are a bioelectricity expert answering exam questions about BETSE "
    "tissue simulations. Answer with EXACTLY ONE line and nothing else.\n"
    "For numeric Vmem questions: 'Mean Vmem = <integer> mV.' (e.g. 'Mean Vmem = -58 mV.')\n"
    "For depolarized/hyperpolarized questions: 'The tissue is depolarized.' or "
    "'The tissue is hyperpolarized.'\n"
    "No reasoning, no explanation, no units other than mV.")
LONG_QS = [
    "If extracellular potassium in a tissue bath rises from 5 mM to 25 mM while "
    "sodium stays constant, how does the resting membrane potential change, and why?",
    "Why do many cancer cells have a more depolarized resting membrane potential "
    "than the normal cells they came from?",
    "How do gap junctions let bioelectric signals spread across a tissue during "
    "regeneration?",
]
DIALOGUE = ["What is bioelectricity?",
            "Explain more: how is it different from nerve signalling?",
            "Give one experiment that shows it matters for regeneration."]


def load(name):
    return [json.loads(l) for l in open(os.path.join(SIM_OUT, name), encoding="utf-8")]


@torch.no_grad()
def chat_batch(model, tok, convs, max_new):
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
             for c in convs]
    outs = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for row in gen:
            outs.append(tok.decode(row[enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip())
    return outs


def exam(model, tok, probs):
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": p["prompt"]}] for p in probs]
    texts = chat_batch(model, tok, convs, 24)
    rs, by = [], defaultdict(list)
    for p, t in zip(probs, texts):
        r = score(p["kind"], p["truth"], t)
        rs.append(r)
        by[p["kind"]].append(r)
    return {"val_reward": sum(rs) / len(rs),
            "exact": sum(r == 1.0 for r in rs) / len(rs),
            "by_kind": {k: sum(v) / len(v) for k, v in by.items()},
            "sample": texts[:3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(10)
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: <=30GB of 32GB
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    res = {"model": args.model, "seed": 42}

    res["exam_A"] = exam(model, tok, load("betse_val.jsonl"))
    print("exam A", json.dumps({k: v for k, v in res["exam_A"].items() if k != "sample"}))
    res["exam_B"] = exam(model, tok, load("betse_offgrid.jsonl"))
    print("exam B", json.dumps({k: v for k, v in res["exam_B"].items() if k != "sample"}))

    longs = chat_batch(model, tok, [[{"role": "user", "content": q}] for q in LONG_QS], 220)
    res["long"] = [{"q": q, "a": a} for q, a in zip(LONG_QS, longs)]
    for q, a in zip(LONG_QS, longs):
        print(f"\nQ: {q}\nA ({len(a.split())} words): {a}")

    conv, turns = [], []
    for u in DIALOGUE:
        conv.append({"role": "user", "content": u})
        a = chat_batch(model, tok, [conv], 180)[0]
        conv.append({"role": "assistant", "content": a})
        turns.append({"user": u, "assistant": a})
        print(f"\n[user] {u}\n[model] {a}")
    res["dialogue"] = turns

    os.makedirs(OUT, exist_ok=True)
    tag = args.model.split("/")[-1]
    with open(os.path.join(OUT, f"b0_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("\nsaved", os.path.join(OUT, f"b0_{tag}.json"))


if __name__ == "__main__":
    main()
