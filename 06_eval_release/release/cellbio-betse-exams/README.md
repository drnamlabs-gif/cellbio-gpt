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

`agent_exam/` holds one run of Claude Sonnet 5.5 sitting both exams closed book
as an agent: it read one question file and wrote one answer file, could reason
but had no simulator or truth access. Questions (`in/`), answers (`answers/`),
per-question scores (`items.jsonl`), summary (`results.json`) and the run log
(`run_log.jsonl`) are included.

| | Exam A | Exam B |
|---|---|---|
| zero-shot | 0.562 | 0.591 |
| five nearest worked examples | 0.955 | 0.808 |

Four of ten batches were repeated because the agent reported using code, against
the rules. The discarded attempts are in `agent_exam/violations/` and scored
within 0.03 of the repeats. Tool use could not be audited beyond the agents' own
reports. This is a single forward test: one run, one model, no repeated batches, so no statistical validation yet. A follow-up with 20 separate, unseen test batches is planned.

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
