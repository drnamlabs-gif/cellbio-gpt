---
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

A 10-layer, 512-dim GPT (8 heads, RoPE, RMSNorm, SwiGLU,
weight tying; ~40.5M parameters, context 512, BPE vocab 16384)
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
| `model_base.pt` | stage 3 pretrain (988.5M tokens, 1 epoch) |
| `model_sft.pt` | stage 4 SFT (97.5k corpus-mined instructions, 2 epochs) |
| `model_dpo.pt` | stage 5a DPO v2 (beta=0.3, on-policy rejected + hard negatives) |
| `model_grpo.pt` | stage 5b GRPO (bioelectric verifier, 300 steps, KL 0.05) |
| `model_sim_grid.pt` | stage 8 simulator RL: warm start + GRPO on 150 grid simulations (exam A) |
| `model_sim_mixed.pt` | stage 8e: 150 randomly placed simulations, mixed 0-5 worked examples (exam B) |
| `model_sim_random.pt` | stage 8e: warm start on 150 randomly placed simulations, single format (exam B) |

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

## Simulator exams

The model is given three simulator settings (extracellular K+, K+ permeability,
Na+ permeability) and answers with the mean membrane potential in mV. BETSE
scores it: full reward within 2 mV, partial within 10 and 25 mV.

| | Exam A: 98 held-out grid problems | Exam B: 234 sealed off-grid problems |
|---|---|---|
| constant predictor | 0.538 | 0.663 |
| hosted model as agent (frontier LLM), zero-shot | 0.562 | 0.591 |
| hosted model as agent (frontier LLM), five worked examples | **0.955** | 0.808 |
| `model_sim_grid.pt` | **0.945** | 0.539 |
| `model_sim_mixed.pt`, five worked examples in the prompt | not measured | **0.835** |
| `model_sim_mixed.pt`, zero-shot | not measured | 0.666 |
| `model_sim_random.pt`, zero-shot | not measured | 0.772 |

The hosted-model rows above come from a first single test (one agent run). A larger
test followed, exam C: 20 sealed batches of fresh simulations (300 simulations, 1,146
problems) whose SHA-256 hashes were published before any model saw them. Mean reward
over the batches:

| System | Zero-shot | Five worked examples |
|---|---|---|
| Frontier hosted LLM, large (agent) | 0.602 | **0.852** |
| Frontier hosted LLM, small (agent) | 0.569 | 0.842 |
| `model_sim_random.pt` | **0.772** | not trained for it |
| `model_sim_mixed.pt` | 0.678 | 0.807 |
| `model_sim_grid.pt` | 0.500 | not trained for it |
| gpt-oss:20b (Ollama) | 0.511 | 0.616 |
| Qwen 3.5 (Ollama) | 0.465 | 0.784 |
| Qwen 3.6 (Ollama) | 0.429 | 0.759 |
| DeepSeek-R1 8B (Ollama) | 0.278 (see note) | 0.431 (see note) |
| Constant predictor | 0.708 | 0.708 |
| Nearest-neighbour lookup, no model (diagnostic) | n/a | 0.827 |

Mean reward over the 20 batches. Zero-shot is the headline column: it shows what a model brings on its own. With worked examples in the prompt the large hosted models have the advantage, and they gain 0.25 to 0.27 from the examples (paired over 20 batches, 20 of 20 batches). DeepSeek-R1 always reasons before answering and was given 3,000 tokens; 679 of its 1,146 zero-shot answers (59%) and 563 with examples (49%) were cut off before a final line and score zero, so its row measures the token budget as much as the model. The constant predictor ignores the question and always answers the most common training answer for that kind of question; it is the floor, and a model below it has not learned the simulator. The open models ran through Ollama with default quantization, greedy decoding and one-line answers.

The comparison fixed in advance was `model_sim_mixed.pt` with five examples against the
hosted agent with five examples. The agent is ahead by 0.045 (95% bootstrap interval 0.025
to 0.065 over batches, Wilcoxon p = 0.0014) and wins 17 of 20 batches, so the near-tie in
the first test did not hold. Other comparisons are descriptive. One hosted model, one run
per batch; the agents' tool use was checked only through their own reports. Exam C stays
sealed for now (GPT and Gemini have not sat it). Details are in the dataset repo. Per-question answers and
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
