# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8g: build question files for agents on exam C, and score their answers.

  python eval_exam_c_agents.py make  --work DIR     # DIR/in/C<bb>_<cond>.txt, 40 files
  python eval_exam_c_agents.py score --work DIR --system NAME

Each file holds one batch in one condition (zero = no examples, five = the five
nearest same-kind worked examples from the random training set). Truths stay in
the sealed folder; the question files hold only questions and examples.
Scoring writes out/exam_c/systems/<NAME>_k0.json and <NAME>_k5.json in the
same per-batch format the 40M checkpoints use, plus the raw answers and a
list of missing ids under out/exam_c/agent/<NAME>/.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "05_dpo_rlvr"))
from bioelectric import reward  # noqa: E402
from eval_agents import load, parse_answers, shots_for  # noqa: E402
from eval_exam_c import N_BATCH, SEALED, SYS_DIR, load_batches, save  # noqa: E402

REC = os.path.join(HERE, "out", "exam_c", "agent")

RULES_C = """You are a candidate sitting a closed-book exam on BETSE bioelectric
tissue simulations (a 213-cell tissue run to steady state).

Rules, which are part of the exam:
- The only tools you may call are Read (on THIS file, as many pages as needed to
  see every question) and Write (once, for the answer file named below).
- Do NOT call Bash or PowerShell or any other tool. Do NOT start background
  tasks. Do NOT write or run any script or code of any kind, not even to save
  or format the answer file. Do not search, list directories or open other files.
- Work out every answer yourself by reasoning. You may reason before answering.
  Only the answer file is scored.

Answer format, one line per question, in the form  <id>: <answer>
- numeric questions:  <id>: Mean Vmem = <integer> mV.
- polarity questions: <id>: The tissue is depolarized.   or   <id>: The tissue is hyperpolarized.
Write exactly one line for every question id, nothing else in the file."""


def render(bid, answer_path, items, with_shots):
    out = [f"EXAM BATCH {bid}  ({len(items)} questions)", "", RULES_C, "",
           f"Answer file (Write here, nothing else): {answer_path}", "", "=" * 60]
    for idx, p, shots in items:
        out += [f"Question id {idx}", p["prompt"]]
        if with_shots:
            out.append("Worked examples (same simulator, other settings):")
            for s in shots:
                out += [f"  Q: {s['prompt']}", f"  A: {s['answer']}"]
        out.append("-" * 60)
    return "\n".join(out) + "\n"


def make(work):
    batches = load_batches()
    pool = load("betse_train2_tr.jsonl")
    for sub in ("in", "answers"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    for b, probs in enumerate(batches):
        for cond in ("zero", "five"):
            bid = f"C{b:02d}_{cond}"
            ans = os.path.join(work, "answers", bid + ".txt")
            items = [(i, p, shots_for(p, pool) if cond == "five" else [])
                     for i, p in enumerate(probs)]
            open(os.path.join(work, "in", bid + ".txt"), "w", encoding="utf-8").write(
                render(bid, ans, items, cond == "five"))
    print("wrote", 2 * len(batches), "files to", os.path.join(work, "in"))


def score(work, system):
    batches = load_batches()
    os.makedirs(os.path.join(REC, system), exist_ok=True)
    for cond, tag in (("zero", "k0"), ("five", "k5")):
        rews, missing = [], []
        for b, probs in enumerate(batches):
            bid = f"C{b:02d}_{cond}"
            ap = os.path.join(work, "answers", bid + ".txt")
            ans = parse_answers(ap)
            rews.append([reward(p["kind"], p["truth"], ans.get(i, "")) for i, p in enumerate(probs)])
            missing += [f"{bid}:{i}" for i in range(len(probs)) if i not in ans]
            if os.path.exists(ap):
                shutil.copy(ap, os.path.join(REC, system, bid + ".txt"))
        save(f"{system}_{tag}", rews, {"missing": len(missing), "missing_ids": missing[:50]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--system", default="agent")
    a = ap.parse_args()
    make(a.work) if a.cmd == "make" else score(a.work, a.system)
