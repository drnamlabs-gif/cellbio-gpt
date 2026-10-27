# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 11e: fair eval of rag_qa.pt — length-bucketed greedy generation (no
padding artifacts), val metrics + live retrieval probes on the finished
index. Saves out/rag_eval.json."""

import json
import os
import re
import sys
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "10")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_tokenizer"))
sys.path.insert(0, os.path.join(HERE, "..", "02_transformer"))
sys.path.insert(0, HERE)
from bpe import BPE               # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from rag_retrieve import open_db, retrieve  # noqa: E402

OUT = os.path.join(HERE, "out")
PROMPT_FMT = "### Instruction:\n{}\n\n### Response:\n"
MAX_NEW = 96
PROBES = ["What is bioelectricity?", "What is membrane voltage?",
          "What does membrane depolarization trigger?", "What is apoptosis?",
          "What are gap junctions?", "What is the resting membrane potential?"]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def f1(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return 0.0
    c = sum((Counter(p) & Counter(g)).values())
    return 0.0 if c == 0 else 2 * c / len(p) * (c / len(g)) / (c / len(p) + c / len(g))


@torch.no_grad()
def gen_bucketed(model, tok, prompts, device, eot):
    enc = [tok.encode(PROMPT_FMT.format(p)) for p in prompts]
    buckets = defaultdict(list)
    for i, e in enumerate(enc):
        buckets[len(e)].append(i)
    outs = [None] * len(enc)
    for L, idxs in sorted(buckets.items()):
        for s in range(0, len(idxs), 64):
            chunk = idxs[s:s + 64]
            x = torch.tensor([enc[i] for i in chunk], device=device)
            out = model.generate(x, MAX_NEW, temperature=0.0)
            for i, row in zip(chunk, out.tolist()):
                ans = row[L:]
                ans = ans[:ans.index(eot)] if eot in ans else ans
                outs[i] = tok.decode(ans).strip()
    return outs


def main():
    torch.manual_seed(42)
    torch.set_num_threads(10)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.90)
    tok = BPE.load(os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json"))
    eot = tok.vocab_size - 1
    ck = torch.load(os.path.join(OUT, "rag_qa.pt"), weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    val = [json.loads(l) for l in open(os.path.join(OUT, "defqa_val.jsonl"),
                                       encoding="utf-8")]
    preds = gen_bucketed(model, tok, [p["prompt"] for p in val], device, eot)
    f1s = [f1(pr, p["answer"]) for pr, p in zip(preds, val)]
    em = float(np.mean([norm(pr) == norm(p["answer"])
                        for pr, p in zip(preds, val)]))
    faith = float(np.mean([" ".join(norm(pr)) in " ".join(norm(p["prompt"]))
                           for pr, p in zip(preds, val)]))
    log = {"token_f1": float(np.mean(f1s)), "exact": em,
           "faithful_substring": faith, "n_val": len(val)}
    print(json.dumps(log))

    db = open_db(os.path.join(OUT, "fts.db"))
    log["probes"] = []
    for q in PROBES:
        hits = retrieve(db, q, k=1)
        if not hits:
            print(f"\nQ: {q}\n  (no retrieval hit)")
            log["probes"].append({"q": q, "hit": None})
            continue
        h = hits[0]
        full = f"Context: {h['ctx']}\n\nAnswer from the context: {q}"
        ans = gen_bucketed(model, tok, [full], device, eot)[0]
        print(f"\nQ: {q}\n  src PMID {h['pmid']}: {h['title'][:72]}\n  A: {ans[:280]}")
        log["probes"].append({"q": q, "pmid": h["pmid"], "title": h["title"],
                              "answer": ans})
    with open(os.path.join(OUT, "rag_eval.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
