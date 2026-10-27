# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Track B, stage B3: long answers grounded in our own library, with an
automatic faithfulness check.

For each of 30 how/why/what-role questions, retrieve passages from the
3.2M-abstract FTS index (11_rag/out/fts.db), then answer under three
configurations:
  b0_closed   base model, no retrieval (how much does it invent?)
  b0_rag      base model, numbered sources in the prompt, must cite [n]
  b1_rag      B1 domain-LoRA model, same prompt
The checker (no model involved) splits every answer into sentences and
scores each one:
  cited      the sentence carries at least one [n]
  supported  >= SUPPORT of its content words appear in a cited source
  backed     >= SUPPORT of its content words appear in ANY retrieved source
This is a lexical proxy for entailment, not a proof; per-sentence recalls are
saved so the threshold can be re-chosen later.

Saves out/b3_<model-tag>.json (questions, sources, answers, per-sentence
scores, metrics) and out/b3_<model-tag>.png. --smoke = 3 questions.
"""

import argparse
import json
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "10")
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "11_rag"))
from rag_retrieve import open_db, retrieve  # noqa: E402

SUPPORT = 0.6
K_SOURCES, CTX_WORDS, MAX_NEW = 5, 110, 320
QUESTIONS = [
    "How does extracellular potassium affect the resting membrane potential?",
    "Why do cancer cells often have a depolarized membrane potential?",
    "How do gap junctions coordinate bioelectric signaling between cells?",
    "What role do voltage-gated sodium channels play in the action potential?",
    "How does membrane potential regulate cell proliferation?",
    "What is the role of bioelectric signaling in planarian regeneration?",
    "How do V-ATPase proton pumps influence left-right patterning in embryos?",
    "How does the Na+/K+-ATPase maintain ion gradients across the membrane?",
    "Why does membrane depolarization trigger calcium influx in excitable cells?",
    "How do endogenous electric fields guide cell migration during wound healing?",
    "What is the role of inward rectifier potassium channels in setting the resting potential?",
    "How does membrane potential change during the cell cycle?",
    "How do voltage-sensitive dyes report membrane potential?",
    "What role does bioelectricity play in Xenopus tail regeneration?",
    "How do chloride channels contribute to the resting membrane potential?",
    "Why is the resting membrane potential of neurons close to the potassium equilibrium potential?",
    "How does hyperkalemia affect cardiac muscle excitability?",
    "What is the role of connexin 43 in cardiac conduction?",
    "How do calcium-activated potassium channels regulate membrane potential?",
    "How does ischemia change the membrane potential of cells?",
    "What is the role of KATP channels in insulin secretion?",
    "How do electric fields affect the orientation of cell division?",
    "How does depolarization affect stem cell differentiation?",
    "What role do ion channels play in cancer cell migration?",
    "How do astrocytes buffer extracellular potassium?",
    "How does membrane potential influence gene expression?",
    "How does the Goldman equation describe the resting membrane potential?",
    "What is the role of HCN channels in pacemaker activity?",
    "How do bioelectric gradients pattern the embryonic head?",
    "How does ouabain affect the membrane potential?",
]
RAG_SYSTEM = (
    "You are a careful biology assistant. Answer the question using ONLY the "
    "numbered sources provided. Write 3 to 6 sentences. End every sentence with "
    "the number of the source that supports it, in brackets, like [2] or [1][3]. "
    "Do not add facts that are not in the sources. If the sources do not answer "
    "the question, say so in one sentence.")
STOP = set("""a an the of in on for to and or but is are was were be been being by
with as at from that this these those it its their there which who whom whose
what when where why how than then so such can could may might will would shall
should do does did not no nor also into onto over under between through during
about after before above below up down out off more most less least very much
many some any each other both all only own same just both further here we they
our you your he she his her them i me my one two three via both while thus
however therefore although whereas""".split())


def content(text):
    ws = re.findall(r"[a-z0-9][a-z0-9\-+/]*", text.lower())
    return [w.rstrip("s") if len(w) > 4 else w for w in ws
            if w not in STOP and len(w) > 2]


def sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", text) if s.strip()]


def check(answer, sources):
    src_words = [set(content(s["text"])) for s in sources]
    rows = []
    for s in sentences(answer):
        cites = sorted({int(c) for c in re.findall(r"\[(\d+)\]", s)
                        if 1 <= int(c) <= len(sources)})
        body = re.sub(r"\[\d+\]", "", s)
        cw = content(body)
        if len(cw) < 4:
            continue
        recall = [len([w for w in cw if w in sw]) / len(cw) for sw in src_words]
        cited_best = max([recall[c - 1] for c in cites], default=0.0)
        rows.append({"sentence": s, "cites": cites,
                     "cited_recall": round(cited_best, 3),
                     "any_recall": round(max(recall, default=0.0), 3)})
    n = len(rows) or 1
    return rows, {
        "n_sent": len(rows),
        "cited": sum(bool(r["cites"]) for r in rows) / n,
        "supported": sum(r["cited_recall"] >= SUPPORT for r in rows) / n,
        "backed": sum(r["any_recall"] >= SUPPORT for r in rows) / n,
        "words": len(answer.split())}


def gather_sources(db, q):
    hits = retrieve(db, q, k=K_SOURCES, max_ctx_words=CTX_WORDS)
    return [{"n": i + 1, "pmid": h["pmid"], "title": h["title"], "text": h["ctx"]}
            for i, h in enumerate(hits)]


def rag_user(q, sources):
    lines = [f"[{s['n']}] {s['title']} {s['text']}" for s in sources]
    return "Sources:\n" + "\n\n".join(lines) + f"\n\nQuestion: {q}"


@torch.no_grad()
def generate(model, tok, convs):
    outs = []
    for i in range(0, len(convs), 8):
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                 for c in convs[i:i + 8]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             repetition_penalty=1.05, pad_token_id=tok.pad_token_id)
        for row in gen:
            outs.append(tok.decode(row[enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip())
    return outs


def plot(tag, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUT, BG = "#1a2b32", "#5c6f76", "#FBFAF7"
    cols = {"b0_closed": "#9AA7AC", "b0_rag": "#C97B1E", "b1_rag": "#0E7C86"}
    labels = {"b0_closed": "open model, no sources",
              "b0_rag": "open model + our library",
              "b1_rag": "B1 (our data) + our library"}
    metrics = [("cited", "sentences with a citation"),
               ("supported", "supported by the cited source"),
               ("backed", "backed by any retrieved source")]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor=BG,
                           gridspec_kw={"width_ratios": [2.2, 1]})
    w = 0.26
    for j, cfg in enumerate(cols):
        vals = [summary[cfg][m] for m, _ in metrics]
        xs = [i + (j - 1) * w for i in range(len(metrics))]
        ax[0].bar(xs, vals, w, color=cols[cfg], label=labels[cfg])
        for x, v in zip(xs, vals):
            ax[0].text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax[0].set_xticks(range(len(metrics)))
    ax[0].set_xticklabels([m[1] for m in metrics], fontsize=9)
    ax[0].set_ylim(0, 1.1)
    ax[0].set_ylabel("fraction of answer sentences")
    ax[0].set_title("Faithfulness of long answers (30 questions)")
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    ws = [summary[c]["words"] for c in cols]
    ax[1].bar(range(3), ws, color=list(cols.values()))
    for i, v in enumerate(ws):
        ax[1].text(i, v + 2, f"{v:.0f}", ha="center", fontsize=8.5, color=INK)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(["no sources", "+ library", "B1 + library"], fontsize=9)
    ax[1].set_ylabel("mean words per answer")
    ax[1].set_title("Answer length")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.5, axis="y")
    fig.suptitle(f"Track B · B3 grounded long answers · {tag} · lexical support ≥ {SUPPORT}",
                 fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"b3_{tag}.png"), dpi=130, facecolor=BG,
                bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variant", default="v1")
    args = ap.parse_args()
    qs = QUESTIONS[:3] if args.smoke else QUESTIONS
    mtag = args.model.split("/")[-1]
    tag = mtag + f"_{args.variant}" + ("_smoke" if args.smoke else "")
    torch.manual_seed(42)
    torch.set_num_threads(10)
    torch.cuda.set_per_process_memory_fraction(0.90)  # shared GPU: <=30GB of 32GB

    db = open_db(os.path.join(ROOT, "11_rag", "out", "fts.db"))
    srcs = [gather_sources(db, q) for q in qs]
    print("retrieved:", [len(s) for s in srcs])

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                device_map="cuda")
    model = PeftModel.from_pretrained(base, os.path.join(OUT, f"b1_{mtag}_lora"))
    model.eval()

    closed = [[{"role": "user", "content": q}] for q in qs]
    rag = [[{"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": rag_user(q, s)}] for q, s in zip(qs, srcs)]
    answers = {}
    with model.disable_adapter():
        answers["b0_closed"] = generate(model, tok, closed)
        answers["b0_rag"] = generate(model, tok, rag)
    answers["b1_rag"] = generate(model, tok, rag)

    res = {"model": args.model, "seed": 42, "support_threshold": SUPPORT,
           "k_sources": K_SOURCES, "items": [], "summary": {}}
    for i, q in enumerate(qs):
        item = {"q": q, "sources": srcs[i], "answers": {}}
        for cfg in answers:
            rows, m = check(answers[cfg][i], srcs[i])
            item["answers"][cfg] = {"text": answers[cfg][i], "sentences": rows, **m}
        res["items"].append(item)
    for cfg in answers:
        ms = [it["answers"][cfg] for it in res["items"]]
        res["summary"][cfg] = {k: sum(m[k] for m in ms) / len(ms)
                               for k in ("cited", "supported", "backed", "words")}
        print(cfg, {k: round(v, 3) for k, v in res["summary"][cfg].items()})

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"b3_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    plot(tag, res["summary"])
    it = res["items"][0]
    print("\nQ:", it["q"])
    for s in it["sources"]:
        print(f"  [{s['n']}] PMID {s['pmid']} {s['title'][:80]}")
    for cfg in answers:
        print(f"\n--- {cfg} ---\n{it['answers'][cfg]['text']}")
    print("\nsaved", os.path.join(OUT, f"b3_{tag}.json"))


if __name__ == "__main__":
    main()
