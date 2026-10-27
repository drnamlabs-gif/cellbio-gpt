# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 6d: assemble the release folder — weights, tokenizer, inference
script, and a model card whose numbers are rendered FROM eval_results.json
(the card can never disagree with the harness).

release/cellbio-gpt-40m/
  model_{base,sft,dpo,grpo}.pt   stripped checkpoints: cfg + weights + metadata
  tokenizer.json                 the 16,384-merge BPE trained in stage 1
  generate.py                    standalone greedy/temperature inference
  README.md                      model card (HF-style)
"""

import json
import os
import shutil

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.join(HERE, "release", "cellbio-gpt-40m")
SRC = {
    "base": os.path.join(HERE, "..", "03_pretrain", "out", "final.pt"),
    "sft": os.path.join(HERE, "..", "04_sft", "out", "sft_final.pt"),
    "dpo": os.path.join(HERE, "..", "05_dpo_rlvr", "out", "dpo_final_v2.pt"),
    "grpo": os.path.join(HERE, "..", "05_dpo_rlvr", "out", "grpo_final.pt"),
}
STAGE_DESC = {
    "base": "stage 3 pretrain (988.5M tokens, 1 epoch)",
    "sft": "stage 4 SFT (97.5k corpus-mined instructions, 2 epochs)",
    "dpo": "stage 5a DPO v2 (beta=0.3, on-policy rejected + hard negatives)",
    "grpo": "stage 5b GRPO (bioelectric verifier, 300 steps, KL 0.05)",
}

os.makedirs(REL, exist_ok=True)

for name, path in SRC.items():
    ck = torch.load(path, weights_only=False)
    torch.save({"cfg": ck["cfg"], "model": ck["model"], "stage": STAGE_DESC[name],
                "seed": 42, "date": "2026-10-23",
                "author": "Dr. Nam H. Le — drnamlabs.com"},
               os.path.join(REL, f"model_{name}.pt"))
    print(f"model_{name}.pt  <- {STAGE_DESC[name]}")
cfg = ck["cfg"]

shutil.copy(os.path.join(HERE, "..", "01_tokenizer", "out", "tokenizer.json"),
            os.path.join(REL, "tokenizer.json"))
shutil.copy(os.path.join(HERE, "..", "02_transformer", "model.py"),
            os.path.join(REL, "model.py"))
shutil.copy(os.path.join(HERE, "..", "01_tokenizer", "bpe.py"),
            os.path.join(REL, "bpe.py"))
shutil.copy(os.path.join(HERE, "generate.py"), os.path.join(REL, "generate.py"))

res = json.load(open(os.path.join(HERE, "out", "eval_results.json"), encoding="utf-8"))
con = json.load(open(os.path.join(HERE, "out", "contamination.json"), encoding="utf-8"))
M = res["models"]


def row(n, label):
    r = M[n]
    return (f"| {label} | {r['val_ppl']:.2f} | {r['pubmedqa_full']['acc']:.3f} | "
            f"{r['pubmedqa_clean']['acc']:.3f} | {r['betse_reward']:.3f} |")


card = f"""# cellbio-gpt-40m

Dr. Nam H. Le — [drnamlabs.com](https://drnamlabs.com)

A {cfg['n_layer']}-layer, {cfg['d_model']}-dim GPT ({cfg['n_head']} heads, RoPE, RMSNorm, SwiGLU,
weight tying; ~40.5M parameters, context {cfg['max_seq']}, BPE vocab {cfg['vocab_size']})
built **from scratch** — tokenizer, architecture, pretraining, SFT, DPO, and
GRPO/RLVR all implemented in plain PyTorch, no HF Trainer, no nn.Transformer —
and trained end-to-end on a single RTX 5090 for **cell & tissue biology**.

This is a *pedagogical/portfolio* release: the point is the complete, honest
pipeline at small scale, not SOTA. Repo: https://github.com/drnamlabs-gif/cellbio-gpt.

## The four checkpoints (the post-training ladder)

| file | stage |
|---|---|
| `model_base.pt` | {STAGE_DESC['base']} |
| `model_sft.pt` | {STAGE_DESC['sft']} |
| `model_dpo.pt` | {STAGE_DESC['dpo']} |
| `model_grpo.pt` | {STAGE_DESC['grpo']} |

## Training data

3,198,416 PubMed cell/tissue-biology abstracts (988.5M BPE tokens), fetched
via NCBI E-utilities, date-sliced; the 5 newest slices (9.2M tokens) are a
temporal holdout never seen in training. SFT/DPO data are self-mined from the
same corpus (no external LLM). RLVR rewards come from the Nernst equation —
the membrane law the BETSE bioelectric simulator integrates.

## Evaluation (seed 42, all checkpoints, same three exams)

| model | holdout ppl ↓ | PubMedQA acc (full) | PubMedQA acc (clean) | BETSE reward |
|---|---|---|---|---|
{row('base', 'base')}
{row('sft', 'SFT')}
{row('dpo', 'DPO v2')}
{row('grpo', 'GRPO')}

Baselines: PubMedQA chance 0.333, majority-class ("yes") {res['baselines']['majority_acc']:.3f}.
Contamination audit: {con['n_contaminated']}/{con['n_items']} PubMedQA items ({con['n_contaminated'] / con['n_items']:.1%}) have their
source PMID in the pretraining corpus; the **clean** column excludes them.
BETSE = mean verifier reward on 727 combo-held-out Nernst-physics problems
(numeric mV graded ±2/±10/±25; categorical exact).

## Honest limitations

- 40M parameters: fluent in-domain prose that **confabulates facts**; do not
  use for any biomedical decision.
- The PubMedQA failure is diagnosed, not just reported: all checkpoints carry
  a ~7-nat prior against answering "no" (never an answer in any training
  stage), so raw likelihood scoring measures that prior. Prior-calibrated
  (PMI) scoring frees the "no" class but accuracy FALLS toward chance, and
  deleting the abstract barely moves the raw score — there is no usable
  evidence-reading signal at this scale (`out/pubmedqa_diagnosis.png`).
- The RLVR stage shows the machinery working, and its limit: categorical
  physics rules generalize perfectly to unseen ion/concentration combos,
  but the model memorizes numeric answers — it cannot learn ln() at this scale.
- Post-training costs perplexity (the alignment tax is visible in the table).
- English-only, abstracts-only register.

## Usage

```
python generate.py --model model_grpo.pt --prompt "What is a gap junction?"
```

`generate.py` is standalone (needs only torch + the bundled tokenizer).
Seed 42 everywhere; every number above is reproducible from the repo's
stage scripts and saved artifacts.
"""

with open(os.path.join(REL, "README.md"), "w", encoding="utf-8") as f:
    f.write(card)
print("README.md written")
print(card.split("## Evaluation")[1].split("## Honest")[0])
