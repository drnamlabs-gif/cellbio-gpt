# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 8f: LLM agents sit the same simulator exams (A and B), closed book.

Why: the open models in eval_ollama.py ran locally. Hosted models are reached
here through agents that read one question file and write one answer file.
The agent never sees the simulator truths, never runs the simulator, and is
told to use no other tools. Scoring is the same verifier (bioelectric.reward)
and the same bands as every other exam in this repo.

  python eval_agents.py make  --work DIR [--smoke]   # question files + manifest
  python eval_agents.py score --work DIR [--smoke]   # score DIR/answers/*.txt

DIR must not be inside the repo's out/ folders while agents run, so a stray
directory listing cannot reach the truth files. `score` copies inputs, answers
and per-item scores into out/agent_exam/ for the record.

Exam A: 98 held-out grid problems (betse_val.jsonl), shots from betse_train.
Exam B: 234 sealed off-grid problems (betse_offgrid.jsonl), shots from
betse_train2_tr (the random-position training set), as in case 3.
Conditions: zero-shot, and five nearest same-kind worked examples per question.
Seed 42 (batch order is fixed; nothing is sampled here).
"""

import argparse
import datetime
import json
import math
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
REC = os.path.join(OUT, "agent_exam")
sys.path.insert(0, os.path.join(HERE, "..", "05_dpo_rlvr"))
from bioelectric import reward as score_one  # noqa: E402

RULES = """You are a candidate sitting a closed-book exam on BETSE bioelectric
tissue simulations (a 213-cell tissue run to steady state).

Rules, which are part of the exam:
- Use no tools except: Read on THIS file, and Write on the one answer file named below.
- Do not search, do not list directories, do not open any other file, do not run
  code or the simulator. Answer from your own knowledge and reasoning.
- You may reason before answering. Only the answer file is scored.

Answer format, one line per question, in the form  <id>: <answer>
- numeric questions:  <id>: Mean Vmem = <integer> mV.
- polarity questions: <id>: The tissue is depolarized.   or   <id>: The tissue is hyperpolarized.
Write exactly one line for every question id, nothing else in the file."""


def load(name):
    return [json.loads(l) for l in open(os.path.join(OUT, name), encoding="utf-8")]


def dist(a, b):
    return sum((math.log2(x) - math.log2(y)) ** 2 for x, y in zip(a, b))


def shots_for(p, pool, k=5):
    cand = sorted((q for q in pool if q["kind"] == p["kind"]),
                  key=lambda q: dist(q["combo"], p["combo"]))
    return cand[:k]


EXAMS = {
    "A": ("betse_val.jsonl", "betse_train.jsonl"),
    "B": ("betse_offgrid.jsonl", "betse_train2_tr.jsonl"),
}
BATCH = {"zero": 120, "five": 50}


def render(batch_id, answer_path, items, with_shots):
    out = [f"EXAM BATCH {batch_id}  ({len(items)} questions)", "", RULES, "",
           f"Answer file (Write here, nothing else): {answer_path}", "",
           "=" * 60]
    for idx, p, shots in items:
        out += [f"Question id {idx}", p["prompt"]]
        if with_shots:
            out.append("Worked examples (same simulator, other settings):")
            for s in shots:
                out += [f"  Q: {s['prompt']}", f"  A: {s['answer']}"]
        out.append("-" * 60)
    return "\n".join(out) + "\n"


def make(work, smoke):
    inp = os.path.join(work, "in")
    os.makedirs(inp, exist_ok=True)
    os.makedirs(os.path.join(work, "answers"), exist_ok=True)
    manifest = []
    for exam, (qfile, pfile) in EXAMS.items():
        val, pool = load(qfile), load(pfile)
        ids = list(range(len(val)))
        if smoke:
            if exam != "A":
                continue
            ids = [0, 1, 2, len(val) - 2, len(val) - 1]
        for cond, size in BATCH.items():
            if smoke and cond != "zero":
                continue
            chunks = [ids[i:i + size] for i in range(0, len(ids), size)]
            for n, chunk in enumerate(chunks):
                bid = f"{'smoke_' if smoke else ''}{exam}_{cond}_{n}"
                ans = os.path.join(work, "answers", bid + ".txt")
                items = [(i, val[i], shots_for(val[i], pool) if cond == "five" else [])
                         for i in chunk]
                text = render(bid, ans, items, cond == "five")
                path = os.path.join(inp, bid + ".txt")
                open(path, "w", encoding="utf-8").write(text)
                manifest.append({"batch": bid, "exam": exam, "cond": cond,
                                 "ids": chunk, "input": path, "answers": ans,
                                 "qfile": qfile})
    mp = os.path.join(work, "manifest_smoke.json" if smoke else "manifest.json")
    json.dump(manifest, open(mp, "w"), indent=1)
    for m in manifest:
        print(m["batch"], len(m["ids"]), m["input"])


LINE = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(.+?)\s*$")


def parse_answers(path):
    got = {}
    if not os.path.exists(path):
        return got
    for line in open(path, encoding="utf-8"):
        m = LINE.match(line)
        if m:
            got[int(m.group(1))] = m.group(2)
    return got


def score(work, smoke, label):
    mp = os.path.join(work, "manifest_smoke.json" if smoke else "manifest.json")
    manifest = json.load(open(mp))
    os.makedirs(os.path.join(REC, "in"), exist_ok=True)
    os.makedirs(os.path.join(REC, "answers"), exist_ok=True)
    pools = {e: load(q) for e, (q, _) in EXAMS.items()}
    items_out, agg = [], {}
    for m in manifest:
        ans = parse_answers(m["answers"])
        for i in m["ids"]:
            p = pools[m["exam"]][i]
            text = ans.get(i, "")
            r = score_one(p["kind"], p["truth"], text)
            items_out.append({"batch": m["batch"], "exam": m["exam"], "cond": m["cond"],
                              "id": i, "kind": p["kind"], "combo": p["combo"],
                              "truth": p["truth"], "answer": text, "reward": r,
                              "missing": i not in ans})
        for src, sub in (("input", "in"), ("answers", "answers")):
            if os.path.exists(m[src]):
                shutil.copy(m[src], os.path.join(REC, sub, os.path.basename(m[src])))
    for it in items_out:
        k = (it["exam"], it["cond"])
        a = agg.setdefault(k, {"r": [], "by": {}, "missing": 0})
        a["r"].append(it["reward"])
        a["by"].setdefault(it["kind"], []).append(it["reward"])
        a["missing"] += it["missing"]
    res = {}
    for (exam, cond), a in sorted(agg.items()):
        res[f"{exam} {cond}"] = {
            "n": len(a["r"]), "reward": sum(a["r"]) / len(a["r"]),
            "exact": sum(x == 1.0 for x in a["r"]) / len(a["r"]),
            "by_kind": {k: sum(v) / len(v) for k, v in a["by"].items()},
            "missing": a["missing"]}
        print(f"{exam} {cond:5s} n={len(a['r'])} reward {res[f'{exam} {cond}']['reward']:.3f} "
              f"exact {res[f'{exam} {cond}']['exact']:.3f} "
              f"{ {k: round(v, 3) for k, v in res[f'{exam} {cond}']['by_kind'].items()} } "
              f"missing {a['missing']}")
    tag = "smoke_" if smoke else ""
    with open(os.path.join(REC, f"{tag}items.jsonl"), "w", encoding="utf-8") as f:
        for it in items_out:
            f.write(json.dumps(it) + "\n")
    json.dump({"model": label, "date": datetime.date.today().isoformat(), "seed": 42,
               "verifier": "05_dpo_rlvr/bioelectric.reward", "results": res},
              open(os.path.join(REC, f"{tag}results.json"), "w"), indent=2)
    print("saved", REC)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--label", default="unlabeled")
    a = ap.parse_args()
    make(a.work, a.smoke) if a.cmd == "make" else score(a.work, a.smoke, a.label)
