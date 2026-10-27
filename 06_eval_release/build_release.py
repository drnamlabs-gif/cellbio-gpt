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
    "sim_grid": os.path.join(HERE, "..", "08_betse_rlvr", "out", "bg8_final.pt"),
    "sim_mixed": os.path.join(HERE, "..", "08_betse_rlvr", "out", "bw8r_ws.pt"),
    "sim_random": os.path.join(HERE, "..", "08_betse_rlvr", "out", "bg8_wsr.pt"),
}
STAGE_DESC = {
    "sim_random": "stage 8e: warm start on 150 randomly placed simulations, single format (exam B)",
    "sim_grid": "stage 8 simulator RL: warm start + GRPO on 150 grid simulations (exam A)",
    "sim_mixed": "stage 8e: 150 randomly placed simulations, mixed 0-5 worked examples (exam B)",
    "base": "stage 3 pretrain (988.5M tokens, 1 epoch)",
    "sft": "stage 4 SFT (97.5k corpus-mined instructions, 2 epochs)",
    "dpo": "stage 5a DPO v2 (beta=0.3, on-policy rejected + hard negatives)",
    "grpo": "stage 5b GRPO (bioelectric verifier, 300 steps, KL 0.05)",
}

os.makedirs(REL, exist_ok=True)

for name, path in SRC.items():
    ck = torch.load(path, weights_only=False, map_location="cpu")
    torch.save({"cfg": ck["cfg"], "model": ck["model"], "stage": STAGE_DESC[name],
                "seed": 42,
                "date": "2026-10-24" if name.startswith("sim") else "2026-10-23",
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


card = f"""---
license: mit
language: en
library_name: pytorch
tags:
- biology
- bioelectricity
- from-scratch
- reinforcement-learning
- verifiable-rewards
---

# cellbio-gpt-40m

Dr. Nam H. Le — [drnamlabs.com](https://drnamlabs.com)

A {cfg['n_layer']}-layer, {cfg['d_model']}-dim GPT ({cfg['n_head']} heads, RoPE, RMSNorm, SwiGLU,
weight tying; ~40.5M parameters, context {cfg['max_seq']}, BPE vocab {cfg['vocab_size']})
built **from scratch** — tokenizer, architecture, pretraining, SFT, DPO, and
GRPO/RLVR all implemented in plain PyTorch, no HF Trainer, no nn.Transformer —
and trained end-to-end on a single RTX 5090 for **cell & tissue biology**.

This is a *pedagogical/portfolio* release: the point is the complete, honest
pipeline at small scale, not SOTA. Repo: https://github.com/drnamlabs-gif/cellbio-gpt.

## Checkpoints

The first four are the post-training ladder; the last three are trained against
the BETSE tissue simulator.

| file | stage |
|---|---|
| `model_base.pt` | {STAGE_DESC['base']} |
| `model_sft.pt` | {STAGE_DESC['sft']} |
| `model_dpo.pt` | {STAGE_DESC['dpo']} |
| `model_grpo.pt` | {STAGE_DESC['grpo']} |
| `model_sim_grid.pt` | {STAGE_DESC['sim_grid']} |
| `model_sim_mixed.pt` | {STAGE_DESC['sim_mixed']} |
| `model_sim_random.pt` | {STAGE_DESC['sim_random']} |

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

## Simulator exams

The model is given three simulator settings (extracellular K+, K+ permeability,
Na+ permeability) and answers with the mean membrane potential in mV. BETSE
scores it: full reward within 2 mV, partial within 10 and 25 mV.

| | Exam A: 98 held-out grid problems | Exam B: 234 sealed off-grid problems |
|---|---|---|
| constant predictor | 0.538 | 0.663 |
| hosted model as agent (Claude Sonnet 5.5), zero-shot | 0.562 | 0.591 |
| hosted model as agent (Claude Sonnet 5.5), five worked examples | **0.955** | 0.808 |
| `model_sim_grid.pt` | **0.945** | 0.539 |
| `model_sim_mixed.pt`, five worked examples in the prompt | not measured | **0.835** |
| `model_sim_mixed.pt`, zero-shot | not measured | 0.666 |
| `model_sim_random.pt`, zero-shot | not measured | 0.772 |

The hosted-model rows above come from a first single test (one agent run). A larger
test followed, exam C: 20 sealed batches of fresh simulations (300 simulations, 1,146
problems) whose SHA-256 hashes were published before any model saw them. Mean reward
over the batches:

| system | reward |
|---|---|
| hosted model as agent (Claude Sonnet 5.5), five examples | 0.852 |
| nearest-neighbour lookup, no model (diagnostic) | 0.827 |
| `model_sim_mixed.pt`, five examples | 0.807 |
| `model_sim_random.pt`, zero-shot | 0.772 |
| constant predictor | 0.708 |
| `model_sim_mixed.pt`, zero-shot | 0.678 |
| hosted model as agent, zero-shot | 0.602 |
| `model_sim_grid.pt`, zero-shot | 0.500 |

The comparison fixed in advance was `model_sim_mixed.pt` with five examples against the
hosted agent with five examples. The agent is ahead by 0.045 (95% bootstrap interval 0.025
to 0.065 over batches, Wilcoxon p = 0.0014) and wins 17 of 20 batches, so the near-tie in
the first test did not hold. Other comparisons are descriptive. One hosted model, one run
per batch; the agents' tool use was checked only through their own reports. Exam C stays
sealed until other model families have sat it. Details are in the dataset repo. Per-question answers and
scores are in the dataset repo.

The grid-trained model is excellent on exam A and falls below the constant
predictor on exam B: it learned a lookup table. Placing the training
simulations at random positions fixed most of that. Prompts follow the problem
format in `08_betse_rlvr/make_sim_problems.py`, and `10_demo/serve_demo.py`
shows how the five worked examples are built. `generate.py` uses the SFT chat
template and is meant for the first four checkpoints.

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
