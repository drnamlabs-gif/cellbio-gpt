# cellbio-gpt-40m

Dr. Nam H. Le — [drnamlabs.com](https://drnamlabs.com)

A 10-layer, 512-dim GPT (8 heads, RoPE, RMSNorm, SwiGLU,
weight tying; ~40.5M parameters, context 512, BPE vocab 16384)
built **from scratch** — tokenizer, architecture, pretraining, SFT, DPO, and
GRPO/RLVR all implemented in plain PyTorch, no HF Trainer, no nn.Transformer —
and trained end-to-end on a single RTX 5090 for **cell & tissue biology**.

This is a *pedagogical/portfolio* release: the point is the complete, honest
pipeline at small scale, not SOTA. Repo: https://github.com/drnamlabs-gif/cellbio-gpt.

## The four checkpoints (the post-training ladder)

| file | stage |
|---|---|
| `model_base.pt` | stage 3 pretrain (988.5M tokens, 1 epoch) |
| `model_sft.pt` | stage 4 SFT (97.5k corpus-mined instructions, 2 epochs) |
| `model_dpo.pt` | stage 5a DPO v2 (beta=0.3, on-policy rejected + hard negatives) |
| `model_grpo.pt` | stage 5b GRPO (bioelectric verifier, 300 steps, KL 0.05) |

## Training data

3,198,416 PubMed cell/tissue-biology abstracts (988.5M BPE tokens), fetched
via NCBI E-utilities, date-sliced; the 5 newest slices (9.2M tokens) are a
temporal holdout never seen in training. SFT/DPO data are self-mined from the
same corpus (no external LLM). RLVR rewards come from the Nernst equation —
the membrane law the BETSE bioelectric simulator integrates.

## Evaluation (seed 42, all checkpoints, same three exams)

| model | holdout ppl ↓ | PubMedQA acc (full) | PubMedQA acc (clean) | BETSE reward |
|---|---|---|---|---|
| base | 11.50 | 0.537 | 0.526 | 0.000 |
| SFT | 14.59 | 0.491 | 0.480 | 0.239 |
| DPO v2 | 14.59 | 0.496 | 0.486 | 0.238 |
| GRPO | 16.24 | 0.460 | 0.453 | 0.977 |

Baselines: PubMedQA chance 0.333, majority-class ("yes") 0.552.
Contamination audit: 59/1000 PubMedQA items (5.9%) have their
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
