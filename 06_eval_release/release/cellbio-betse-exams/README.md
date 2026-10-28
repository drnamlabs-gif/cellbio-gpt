---
license: mit
language: en
tags:
- biology
- bioelectricity
- simulation
- benchmark
size_categories:
- n<1K
---

# cellbio-betse-exams

Text problems about tissue membrane potential, with answers computed by the
[BETSE](https://github.com/betsee/betse) bioelectric simulator. They were
written to train and test a small language model against physics. See the
[code and report](https://github.com/drnamlabs-gif/cellbio-gpt).

Each simulation is one 213-cell tissue run to steady state under three
global settings, listed in the `combo` field in this order:

1. extracellular K+ in mM (extracellular Na+ is lowered by the same amount, so charge stays balanced)
2. K+ membrane permeability, as a multiple of baseline
3. Na+ permeability, as a multiple of baseline

The simulator returns the voltage of every cell. The files keep the mean, the
minimum and the maximum, plus the number of cells and the seed (42).

## Files

| File | Rows | What it is |
|---|---|---|
| `grid_sims.jsonl` | 150 | simulations on a regular grid: 10 K+ levels x 5 K+ permeabilities x 3 Na+ permeabilities |
| `exam_a_train.jsonl`, `exam_a_val.jsonl` | 466 / 98 | problems made from the grid, split by knob combination, so no held-out combination was simulated for training |
| `random_sims.jsonl` | 150 | simulations at random positions in the same space |
| `random_train.jsonl`, `random_val.jsonl` | 472 / 102 | problems made from the random simulations |
| `exam_b_sims.jsonl` | 60 | sealed off-grid simulations, each at least 0.25 (log2 units) from any grid node |
| `exam_b_problems.jsonl` | 234 | the problems made from them |
| `*_meta.json` | | seed, ranges and distances for each set |

Problems ask either for the steady-state mean Vmem in mV (`kind: nernst`; the
name is historical) or for the direction of the shift (`kind: polarity`). The
`truth` field holds the simulator value. A reward function gives full credit
within 2 mV and partial credit within 10 and 25 mV.

## Hosted model as agent

`agent_exam/` holds a first single test: Claude Sonnet 5.5 sat exams A and B closed book
as an agent (it read one question file and wrote one answer file, could reason, had no
simulator or truth access). Questions, answers, per-question scores, summary and run log
are included.

| | Exam A | Exam B |
|---|---|---|
| zero-shot | 0.562 | 0.591 |
| five nearest worked examples | 0.955 | 0.808 |

Four of ten batches were repeated because the agent reported using code, against the
rules; the discarded attempts are in `agent_exam/violations/` and scored within 0.03 of
the repeats.

## Exam C: twenty sealed batches

`exam_c/` holds the pre-registered test that followed: 20 batches of fresh simulations
(300 simulations, 1,146 problems) at points that appear in no training set and no earlier
exam. `commitment.json` has the SHA-256 of every sealed file, published before any model
saw a batch; `PLAN.md` is the analysis plan; `batch_means.json` has each system's mean
reward per batch; `stats.md` and `stats.json` have the paired statistics;
`agent_run_log.jsonl` is the run log. The questions, truths and per-problem answers are
withheld until other model families have sat the exam.

| System | Zero-shot | Five worked examples |
|---|---|---|
| Claude Sonnet 5.5 (agent) | 0.602 | **0.852** |
| Claude Haiku 5.5 (agent, `claude-haiku-5-5`) | 0.569 | 0.842 |
| 40M, trained on random points | **0.772** | not trained for it |
| 40M, mixed warm start | 0.678 | 0.807 |
| 40M, trained on the grid | 0.500 | not trained for it |
| gpt-oss:20b (Ollama) | 0.511 | 0.616 |
| Qwen 3.5 (Ollama) | 0.465 | 0.784 |
| Qwen 3.6 (Ollama) | 0.429 | 0.759 |
| DeepSeek-R1 8B (Ollama) | 0.278 (see note) | 0.431 (see note) |
| Constant predictor | 0.708 | 0.708 |
| Nearest-neighbour lookup, no model (diagnostic) | n/a | 0.827 |

Mean reward over the 20 batches. Zero-shot is the headline column: it shows what a model brings on its own. With worked examples in the prompt the large hosted models have the advantage, and they gain 0.25 to 0.27 from the examples (paired over 20 batches, 20 of 20 batches). DeepSeek-R1 always reasons before answering and was given 3,000 tokens; 679 of its 1,146 zero-shot answers (59%) and 563 with examples (49%) were cut off before a final line and score zero, so its row measures the token budget as much as the model. The constant predictor ignores the question and always answers the most common training answer for that kind of question; it is the floor, and a model below it has not learned the simulator. The open models ran through Ollama with default quantization, greedy decoding and one-line answers.

Planned comparison, 40M with five examples against the hosted agent with five examples:
the agent is ahead by 0.045 (95% bootstrap interval 0.025 to 0.065, Wilcoxon p = 0.0014),
winning 17 of 20 batches. Other comparisons are descriptive. One hosted model, one run
per batch, tool use checked only through the agents' own reports. The zero-shot column became the headline after the first-round results were seen; see PLAN.md.

## Why two exams

A model trained on the grid scores well on held-out grid problems because
every test point sits next to a trained node. Exam B has no such neighbours.
A model that learned a lookup table does well on exam A and falls below the
constant predictor on exam B. Report both.

## Use

These exams are published in full, so they no longer test a model that has
seen them. Keep exam B out of training data.

Simulator: BETSE 1.5.0, Python 3.11 or 3.12. One configuration takes about
13 seconds.
