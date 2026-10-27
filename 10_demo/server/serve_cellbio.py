# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""CellBio-GPT public demo server (small CPU-only server).

Self-contained: needs only bpe.py, model.py, the two checkpoints, tokenizer,
betse_train2_tr.jsonl and run_sim_server.py in /opt/cellbio/app + data.
Binds 127.0.0.1:5062 — nginx proxies it at https://drnamlabs.com/cellbio/.
Guard rails: inputs clamped to the validated ranges, ONE simulation at a
time, 150 fresh sims/day, 1 torch thread, every sim niced.
"""

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

BASE = "/opt/cellbio"
APP = os.path.join(BASE, "app")
RUNS = os.path.join(BASE, "runs")
sys.path.insert(0, APP)
from bpe import BPE        # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from rag_retrieve import open_db, retrieve  # noqa: E402

PORT = 5062
PROMPT_FMT = "### Instruction:\n{}\n\n### Response:\n"
P_NUM = ("A BETSE tissue simulation runs with extracellular K+ = {k} mM, "
         "extracellular Na+ = {na} mM, K+ membrane permeability scaled x{dk} "
         "and Na+ permeability scaled x{dn}. What steady-state mean Vmem (mV) "
         "does the tissue reach?")
MAX_SEQ, MAX_NEW = 512, 16
NUM_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*mV")
SIM_FILES = {"vmem_map.png": "image/png", "vmem_anim.gif": "image/gif",
             "vmem_trace.png": "image/png"}
DAILY_SIM_CAP = 150

torch.manual_seed(42)
torch.set_num_threads(1)

print("loading tokenizer + models (CPU) ...", flush=True)
TOK = BPE.load(os.path.join(BASE, "tokenizer.json"))
EOT = TOK.vocab_size - 1
MODELS = {}
for name, fn in [("qa", "grpo_final.pt"), ("exp", "bw8r_ws.pt"),
                 ("lit", "rag_qa.pt")]:
    path = os.path.join(BASE, fn)
    if not os.path.exists(path):
        continue
    ck = torch.load(path, weights_only=False, map_location="cpu")
    m = GPT(GPTConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    m.eval()
    MODELS[name] = m
POOL = [json.loads(l) for l in open(os.path.join(BASE, "betse_train2_tr.jsonl"))
        if json.loads(l)["kind"] == "nernst"]
FTS_PATH = os.path.join(BASE, "fts.db")
FTS = open_db(FTS_PATH) if os.path.exists(FTS_PATH) else None
GEN_LOCK = threading.Lock()
SIM_LOCK = threading.Lock()
LIT_LOCK = threading.Lock()
SIM_BUDGET = {"day": "", "used": 0}
print(f"ready on 127.0.0.1:{PORT}", flush=True)


def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def dist(a, b):
    return sum((math.log2(x) - math.log2(y)) ** 2 for x, y in zip(a, b))


def generate(model, ids):
    ids = ids[-(MAX_SEQ - MAX_NEW):]
    with GEN_LOCK, torch.no_grad():
        x = torch.tensor([ids])
        out = model.generate(x, MAX_NEW, temperature=0.0)[0].tolist()
    ans = out[len(ids):]
    ans = ans[:ans.index(EOT)] if EOT in ans else ans
    return TOK.decode(ans).strip()


def handle_ask(body):
    if body.get("mode") == "qa":
        q = str(body.get("question", "")).strip()[:400]
        if not q:
            return {"error": "empty question"}
        return {"answer": generate(MODELS["qa"], TOK.encode(PROMPT_FMT.format(q)))}
    if body.get("mode") == "lit":
        if FTS is None or "lit" not in MODELS:
            return {"error": "the literature index is still being built on "
                             "this server — try again later"}
        q = str(body.get("question", "")).strip()[:300]
        if not q:
            return {"error": "empty question"}
        with LIT_LOCK:
            hits = retrieve(FTS, q, k=2)
        if not hits:
            return {"answer": "No matching abstract found in the 3.2M-paper "
                              "library for that phrasing.", "sources": []}
        best, best_src = None, None
        for h in hits:
            full = f"Context: {h['ctx']}\n\nAnswer from the context: {q}"
            ans = generate(MODELS["lit"], TOK.encode(PROMPT_FMT.format(full)))
            if best is None:
                best, best_src = ans, h
            if re.search(r"\b(is|are)\s+(a|an|the|defined)\b", ans) \
                    and len(ans.split()) >= 8:
                best, best_src = ans, h
                break
        return {"answer": best,
                "sources": [{"pmid": best_src["pmid"],
                             "title": best_src["title"]}]}
    k = round(clamp(body.get("k", 5), 2, 40), 1)
    dk = round(clamp(body.get("dk", 1), 0.25, 4), 2)
    dn = round(clamp(body.get("dn", 1), 0.5, 2), 2)
    prompt = P_NUM.format(k=k, na=round(150 - k, 1), dk=dk, dn=dn)
    near = sorted((q for q in POOL if tuple(q["combo"]) != (k, dk, dn)),
                  key=lambda q: dist(q["combo"], (k, dk, dn)))[:5]
    parts = [PROMPT_FMT.format(e["prompt"]) + e["answer"] for e in near]
    ids = TOK.encode("\n\n".join(parts + [PROMPT_FMT.format(prompt)]))
    while len(ids) > MAX_SEQ - MAX_NEW and parts:
        parts = parts[:-1]
        ids = TOK.encode("\n\n".join(parts + [PROMPT_FMT.format(prompt)]))
    ans = generate(MODELS["exp"], ids)
    m = NUM_RE.search(ans)
    return {"answer": ans, "prompt": prompt, "k": k, "dk": dk, "dn": dn,
            "anchors": [{"combo": e["combo"], "answer": e["answer"]} for e in near],
            "pred": float(m.group(1)) if m else None}


def handle_verify(body):
    k = round(clamp(body.get("k", 5), 2, 40), 1)
    dk = round(clamp(body.get("dk", 1), 0.25, 4), 2)
    dn = round(clamp(body.get("dn", 1), 0.5, 2), 2)
    tag = f"K{k}_dk{dk}_dn{dn}".replace(".", "p")
    cached = os.path.exists(os.path.join(RUNS, tag, "truth.json"))
    today = time.strftime("%Y-%m-%d")
    if SIM_BUDGET["day"] != today:
        SIM_BUDGET["day"], SIM_BUDGET["used"] = today, 0
    if not cached and SIM_BUDGET["used"] >= DAILY_SIM_CAP:
        return {"error": "daily simulation budget reached — try tomorrow"}
    if not SIM_LOCK.acquire(timeout=240):
        return {"error": "another experiment is running — try again in a minute"}
    try:
        if not cached:
            SIM_BUDGET["used"] += 1
        r = subprocess.run([os.path.join(BASE, "venv", "bin", "python"),
                            os.path.join(APP, "run_sim_server.py"),
                            str(k), str(dk), str(dn)],
                           capture_output=True, text=True, timeout=480)
        line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")][-1]
        truth = json.loads(line)
    except Exception as e:
        return {"error": str(e)[:200]}
    finally:
        SIM_LOCK.release()
    if "error" in truth:
        return {"error": truth["error"][-300:]}
    rd = os.path.join(RUNS, truth["tag"])
    return {"truth_mV": truth["mean_vmem_mV"], "n_cells": truth["n_cells"],
            "wall_s": truth.get("wall_s", 0.0), "tag": truth["tag"],
            "map": os.path.exists(os.path.join(rd, "vmem_map.png")),
            "anim": os.path.exists(os.path.join(rd, "vmem_anim.gif")),
            "trace": os.path.exists(os.path.join(rd, "vmem_trace.png"))}


class H(BaseHTTPRequestHandler):
    def _send(self, code, data, ctype="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if ctype.startswith("image/"):
            self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(APP, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path.startswith("/sim_img/"):
            parts = path.split("/")
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
            n = int(self.headers.get("Content-Length", 0))
            if n > 4096:
                self._send(413, {"error": "too large"})
                return
            body = json.loads(self.rfile.read(n))
            if self.path.split("?")[0] == "/ask":
                self._send(200, handle_ask(body))
            elif self.path.split("?")[0] == "/verify":
                self._send(200, handle_verify(body))
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": str(e)[:200]})

    def log_message(self, fmt, *a):
        print("[http]", self.client_address[0], fmt % a, flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
