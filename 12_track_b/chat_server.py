# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Track B interactive chat: try every stage's model while training goes on.

Models come from models.json (re-read on every request, so a new stage
appears in the dropdown as soon as its entry is appended). All entries that
share a base model share one copy of its weights; LoRA adapters are loaded
lazily and switched per request.

Two modes:
  chat      plain multi-turn dialogue, closed book
  library   each turn retrieves 5 passages from the 3.2M-abstract index; the
            model must answer from them and cite [n]; every sentence of the
            reply is scored by the same lexical checker as b3_grounded.py
            (supported by its cited source / backed by some source / neither)

Run: python chat_server.py  ->  http://localhost:8766   (VRAM capped at 8 GB
so it can run beside training). Seed 42, greedy decoding.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "11_rag"))
from b3_grounded import RAG_SYSTEM, check, rag_user  # noqa: E402
from rag_retrieve import open_db, retrieve  # noqa: E402

PORT = 8766
REGISTRY = os.path.join(HERE, "models.json")
MAX_NEW = 420

torch.manual_seed(42)
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.25)  # ~8GB: leaves room for training
DB = open_db(os.path.join(ROOT, "11_rag", "out", "fts.db"))
LOCK = threading.Lock()


class Bank:
    """One base model per base name, adapters attached lazily by id.
    Entries with "reload": true point at a checkpoint directory that a
    running training job overwrites; it is reloaded when its files change."""

    def __init__(self):
        self.bases, self.toks, self.loaded, self.stamp = {}, {}, {}, {}

    def get(self, entry):
        base = entry["base"]
        if entry.get("local") and not os.path.isabs(base):
            base = os.path.join(HERE, base)
        if entry.get("reload"):
            cfg = os.path.join(base, "config.json")
            if not os.path.exists(cfg):
                raise RuntimeError("no checkpoint saved yet - the first one "
                                   "appears ~33 min after training starts")
            st = max(os.path.getmtime(os.path.join(base, f)) for f in os.listdir(base))
            if self.stamp.get(base) != st and base in self.bases:
                print("checkpoint changed, reloading", base, flush=True)
                del self.bases[base]
                torch.cuda.empty_cache()
            self.stamp[base] = st
        if base not in self.bases:
            print("loading base", base, flush=True)
            tok = AutoTokenizer.from_pretrained(entry.get("tokenizer", base))
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self.toks[base] = tok
            self.bases[base] = AutoModelForCausalLM.from_pretrained(
                base, dtype=torch.bfloat16, device_map="cuda").eval()
        model = self.bases[base]
        if entry.get("adapter") and entry["id"] not in self.loaded:
            path = os.path.join(HERE, entry["adapter"])
            print("loading adapter", entry["id"], flush=True)
            if isinstance(model, PeftModel):
                model.load_adapter(path, adapter_name=entry["id"])
            else:
                model = PeftModel.from_pretrained(model, path, adapter_name=entry["id"])
                self.bases[base] = model.eval()
            self.loaded[entry["id"]] = True
        return self.bases[base], self.toks[base]


BANK = Bank()


def registry():
    with open(REGISTRY, encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def generate(entry, messages):
    model, tok = BANK.get(entry)
    stop = {}
    if entry.get("system") and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": entry["system"]}] + messages
    if entry.get("eos_token"):
        stop = {"eos_token_id": tok.convert_tokens_to_ids(entry["eos_token"])}
    if entry.get("fmt") == "instr":
        # Track A / Track B SFT template; earlier turns are replayed in the same form
        text = ""
        for m in messages:
            if m["role"] == "system":
                continue
            text += (f"### Instruction:\n{m['content']}\n\n### Response:\n"
                     if m["role"] == "user" else f"{m['content']}\n\n")
        stop = {"eos_token_id": tok.eos_token_id}
    elif tok.chat_template:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in messages) + "Assistant:"
    enc = tok(text, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=MAX_NEW, do_sample=False, repetition_penalty=1.05,
              pad_token_id=tok.pad_token_id, **stop)
    if isinstance(model, PeftModel):
        if entry.get("adapter"):
            model.set_adapter(entry["id"])
            out = model.generate(**enc, **kw)
        else:
            with model.disable_adapter():
                out = model.generate(**enc, **kw)
    else:
        out = model.generate(**enc, **kw)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def handle_chat(body):
    entries = {e["id"]: e for e in registry()}
    entry = entries.get(body.get("model_id"))
    if entry is None:
        return {"error": "unknown model"}
    msgs = [m for m in body.get("messages", [])
            if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()]
    msgs = msgs[-12:]
    if not msgs or msgs[-1]["role"] != "user":
        return {"error": "the last message must be from the user"}
    q = str(msgs[-1]["content"])[:1200]
    if body.get("mode") != "library":
        with LOCK:
            ans = generate(entry, msgs)
        return {"answer": ans, "model": entry["label"]}

    users = [m["content"] for m in msgs if m["role"] == "user"]
    query = q if len(q.split()) >= 6 or len(users) < 2 else users[-2] + " " + q
    with LOCK:
        hits = retrieve(DB, query, k=5, max_ctx_words=110)
    sources = [{"n": i + 1, "pmid": h["pmid"], "title": h["title"], "text": h["ctx"]}
               for i, h in enumerate(hits)]
    if not sources:
        return {"answer": "No passage in the 3.2M-abstract library matched this "
                          "question, so there is nothing to ground an answer in.",
                "model": entry["label"], "sources": [], "sentences": []}
    turn = [{"role": "system", "content": RAG_SYSTEM}] + msgs[:-1] + \
        [{"role": "user", "content": rag_user(q, sources)}]
    with LOCK:
        ans = generate(entry, turn)
    rows, metrics = check(ans, sources)
    return {"answer": ans, "model": entry["label"], "sources": sources,
            "sentences": rows, "metrics": metrics, "query": query}


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
            with open(os.path.join(HERE, "chat.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/models":
            self._send(200, [{"id": e["id"], "label": e["label"]} for e in registry()])
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            if self.path == "/chat":
                self._send(200, handle_chat(body))
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": str(e)[:300]})

    def log_message(self, fmt, *a):
        print("[http]", fmt % a, flush=True)


if __name__ == "__main__":
    print(f"ready: http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
