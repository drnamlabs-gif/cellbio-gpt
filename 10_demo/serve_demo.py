# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""CellBio-GPT 40M demo server: the closed physics-verified loop, live.

Two lanes:
  qa  — free-text bioelectricity Q&A (textbook model, 05 grpo_final.pt)
  exp — experiment loop: knobs -> 40M predicts mean Vmem (with 5 nearest
        worked examples in context, case-3 recipe) -> the REAL BETSE sim
        runs the experiment (~20 s) -> physics verdict in mV.

Run (GPU venv):  python serve_demo.py   ->  http://localhost:8765
BETSE runs via the 3.11 venv in a subprocess (run_one_sim.py); each combo's
truth is cached in 08_betse_rlvr/runs/. Seed 42. VRAM <=30GB rule applied.
"""

import json
import os
import re
import subprocess
import sys
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("OMP_NUM_THREADS", "10")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "08_betse_rlvr"))
sys.path.insert(0, os.path.join(HERE, "..", "11_rag"))
from baldwin_betse import (OUT, TOKENIZER, build_prompt_ids, generate_from_ids,
                           load_problems, neighbors)  # noqa: E402
from make_sim_problems import P_NUM  # noqa: E402
from bpe import BPE                  # noqa: E402
from model import GPT, GPTConfig     # noqa: E402
from sft import PROMPT_FMT           # noqa: E402
from rag_retrieve import open_db, retrieve, retrieve_def  # noqa: E402

RAG_OUT = os.path.join(HERE, "..", "11_rag", "out")

PORT = 8765
RUNS = os.path.abspath(os.path.join(HERE, "..", "08_betse_rlvr", "runs"))
SIM_FILES = {"vmem_map.png": "image/png", "vmem_anim.gif": "image/gif",
             "vmem_trace.png": "image/png"}
BETSE_PY = os.path.join(HERE, "..", ".venv-betse", "Scripts", "python.exe")
QA_CKPT = os.path.join(HERE, "..", "05_dpo_rlvr", "out", "grpo_final.pt")
EXP_CKPT = os.path.join(OUT, "bw8r_ws.pt")
NUM_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*mV")

torch.manual_seed(42)
torch.set_num_threads(10)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: <=30GB of 32GB

print("loading tokenizer + models ...")
TOK = BPE.load(TOKENIZER)
EOT = TOK.vocab_size - 1
MODELS = {}
for name, path in [("qa", QA_CKPT), ("exp", EXP_CKPT),
                   ("lit", os.path.join(RAG_OUT, "rag_qa.pt"))]:
    ck = torch.load(path, weights_only=False)
    m = GPT(GPTConfig(**ck["cfg"])).to(DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    MODELS[name] = m
FTS = open_db(os.path.join(RAG_OUT, "fts.db"))
POOL = defaultdict(list)
for p in load_problems("betse_train2_tr.jsonl"):
    POOL[p["kind"]].append(p)
GEN_LOCK = threading.Lock()
LIT_LOCK = threading.Lock()
print(f"ready: http://localhost:{PORT}")


def generate(model, ids, max_new=16):
    with GEN_LOCK, torch.no_grad():
        x = torch.tensor([ids], device=DEVICE)
        out = model.generate(x, max_new, temperature=0.0)[0].tolist()
    ans = out[len(ids):]
    ans = ans[:ans.index(EOT)] if EOT in ans else ans
    return TOK.decode(ans).strip()


def handle_ask(body):
    if body["mode"] == "qa":
        q = body["question"].strip()[:400]
        return {"answer": generate(MODELS["qa"], TOK.encode(PROMPT_FMT.format(q)))}
    if body["mode"] == "lit":
        q = body["question"].strip()[:300]
        # the reader was trained on definition questions only; anything else
        # (how/why/what role...) gets an honest decline instead of an
        # on-the-page but off-topic quote
        if not re.match(r"^\s*(what\s+(is|are)\s+(a|an|the)?\s*[\w\-\s]{1,60}\??$"
                        r"|define\s+[\w\-\s]{1,60}\??$)", q, re.I):
            return {"answer": "This version answers definition questions "
                              "(\"What is X?\"). How and why questions are "
                              "outside what its reader was trained on, so it "
                              "declines rather than quote something off-topic.",
                    "sources": []}
        with LIT_LOCK:
            hits = retrieve_def(FTS, q, k=6, max_ctx_words=160) \
                + retrieve(FTS, q, k=5, max_ctx_words=260)
        seen = set()
        hits = [h for h in hits
                if not (h["pmid"] in seen or seen.add(h["pmid"]))]
        # SELECT, don't free-generate: collect definitional candidate
        # sentences from the retrieved abstracts, let the trained reader
        # score them by likelihood, return the winner VERBATIM (a real
        # quote from the paper - grammatical by construction, checkable
        # against the PMID).
        qterms = [t for t in re.findall(r"[a-z][a-z0-9\-]+", q.lower())
                  if t not in ("what", "is", "are", "the", "define", "does",
                               "do", "an")]
        cands, relaxed = [], []
        for h in hits:
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", h["ctx"]):
                s = s.strip()
                if not (8 <= len(s.split()) <= 60):
                    continue
                low = s.lower()
                strict = any(re.search(
                    rf"\b{re.escape(t)}\w*\s+(is|are)\s+(a|an|the|defined)\b",
                    low) for t in qterms)
                loose = any(re.search(
                    rf"\b{re.escape(t)}\w*\s+(is|are)\s+"
                    r"(?!also\b|likely\b|not\b|often\b|still\b|unclear\b|"
                    r"unknown\b|under\b)\w+", low) for t in qterms)
                if strict:
                    cands.append((s, h))
                elif loose:
                    relaxed.append((s, h))
        if not cands:
            cands = relaxed
        if not cands:
            return {"answer": "The 3.2M-paper library has no definitional "
                              "sentence for that term - this model only "
                              "quotes retrieved text, so it declines rather "
                              "than guesses.", "sources": []}
        best, best_src, best_lp = None, None, -1e9
        for s, h in cands[:8]:
            full = f"Context: {h['ctx']}\n\nAnswer from the context: {q}"
            pr = TOK.encode(PROMPT_FMT.format(full))
            ans_ids = TOK.encode(s)
            ids = (pr + ans_ids)[-510:]
            na = len(ans_ids)
            with GEN_LOCK, torch.no_grad():
                x = torch.tensor([ids], device=DEVICE)
                logits, _, _ = MODELS["lit"](x)
                lp = torch.log_softmax(logits[0, -na - 1:-1].float(), dim=-1)
                score_lp = lp.gather(1, torch.tensor(
                    ids[-na:], device=DEVICE).unsqueeze(1)).mean().item()
            if score_lp > best_lp:
                best, best_src, best_lp = s, h, score_lp
        return {"answer": best, "quoted": True,
                "sources": [{"pmid": best_src["pmid"],
                             "title": best_src["title"]}]}
    k = round(float(body["k"]), 1)
    dk = round(float(body["dk"]), 2)
    dn = round(float(body["dn"]), 2)
    prompt = P_NUM[0].format(k=k, na=round(150 - k, 1), dk=dk, dn=dn)
    p = {"kind": "nernst", "combo": [k, dk, dn], "prompt": prompt}
    anchors = [{"combo": e["combo"], "answer": e["answer"]}
               for e in neighbors(p, POOL)[:5]]
    ans = generate(MODELS["exp"], build_prompt_ids(p, 5, POOL, TOK))
    m = NUM_RE.search(ans)
    return {"answer": ans, "prompt": prompt, "anchors": anchors,
            "pred": float(m.group(1)) if m else None}


def handle_verify(body):
    k = round(float(body["k"]), 1)
    dk = round(float(body["dk"]), 2)
    dn = round(float(body["dn"]), 2)
    r = subprocess.run([BETSE_PY, os.path.join(HERE, "run_one_sim.py"),
                        str(k), str(dk), str(dn)],
                       capture_output=True, text=True, timeout=600)
    line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")][-1]
    truth = json.loads(line)
    if "error" in truth:
        return {"error": truth["error"][-300:]}
    tag = truth.get("tag", "")
    rd = os.path.join(RUNS, tag)
    return {"truth_mV": truth["mean_vmem_mV"], "n_cells": truth["n_cells"],
            "wall_s": truth.get("wall_s", 0.0), "tag": tag,
            "map": os.path.exists(os.path.join(rd, "vmem_map.png")),
            "anim": os.path.exists(os.path.join(rd, "vmem_anim.gif")),
            "trace": os.path.exists(os.path.join(rd, "vmem_trace.png"))}


class H(BaseHTTPRequestHandler):
    def _send(self, code, data, ctype="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path.startswith("/sim_img/"):
            parts = self.path.split("?")[0].split("/")
            if len(parts) == 4 and parts[3] in SIM_FILES \
                    and re.fullmatch(r"[A-Za-z0-9_]+", parts[2]):
                fp = os.path.join(RUNS, parts[2], parts[3])
                if os.path.exists(fp):
                    with open(fp, "rb") as f:
                        self._send(200, f.read(), SIM_FILES[parts[3]])
                    return
            self._send(404, {"error": "no image"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/ask":
                self._send(200, handle_ask(body))
            elif self.path == "/verify":
                self._send(200, handle_verify(body))
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": str(e)[:300]})

    def log_message(self, fmt, *a):
        print("[http]", fmt % a, flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
