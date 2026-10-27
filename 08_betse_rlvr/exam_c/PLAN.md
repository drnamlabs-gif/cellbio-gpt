# Exam C: twenty sealed batches

Written before any model or agent has seen a batch. Exams A and B each gave one
number per model. This exam is meant to give enough independent units to test
whether the differences between models are real.

## What is fixed in advance

- 20 batches of 15 new BETSE simulations each (300 in total), about 60 problems
  per batch, built the same way as exam B: two numeric questions per
  simulation, and two direction questions when the shift from baseline is at
  least 3 mV.
- Every point lies at least 0.25 log2 units from every lattice node, 0.20 from
  every point of the random training set, 0.15 from every exam B point and
  0.15 from every other exam C point. No model has trained on, or been
  scored on, anything near these points.
- Seed 42, stream 2027. The truths and problem files are kept outside the
  repository. `commitment.json` holds the SHA-256 of each file, published
  before the exams. After the exams the files are released and can be checked
  against the hashes.

## Who sits it, and how

- CellBio-GPT checkpoints (the grid-trained and the mixed-example model),
  with and without five worked examples where the model was trained for them.
- Open models through Ollama, zero-shot and five-shot.
- Hosted models as agents. Each agent is a fresh one, given one batch in one
  condition, and is told about the exam only at the moment it sits it. It reads
  one question file and writes one answer file, and is told in so many words
  to use no Bash, no background tasks and no code. Agents are never shown a
  batch in advance, and none is asked to prepare.
- Worked examples come from the random training set, the five nearest of the
  same kind, as in exam B.
- The constant predictor is computed from the training set, as before.

## How it is analysed

- The unit is the batch, so n = 20 for every model and every comparison is
  paired by batch.
- Primary comparison: the mixed-example 40M model with five examples against
  the hosted agent with five examples. Difference in mean reward per batch,
  95% bootstrap interval over batches (10,000 resamples), and a Wilcoxon
  signed-rank test.
- Secondary: each model against the constant predictor, and zero-shot against
  five-shot for each model.
- One run per model and batch. The report states the number of batches, the
  intervals and the p-values as computed, whichever way they fall, and keeps
  any batch that had to be repeated in the log, as in the first agent run.

## What would count as a surprise

The first agent run suggested the hosted agent with five examples is about
level with the 40M model on exam A and a little behind on exam B. If the
paired interval over 20 batches excludes zero in either direction, the first
result was misleading. If it spans zero, the honest reading is that the two
cannot be told apart at this sample size.

## Addendum, written before the second round of runs

The first round (40M checkpoints and Sonnet 5.5 agents) is finished and its result is
in the report. Before running anything further, these systems and comparisons are fixed.

Systems added, each on all 20 batches, zero-shot and with five examples:
- A Haiku-class agent (the `haiku` alias of the agent tool; the exact model version
  is whatever the harness maps it to and is recorded from the agent's own reply).
  Same rules and prompt as the Sonnet agents.
- Open models through Ollama, greedy decoding, one-line answers as in the earlier
  open-model exam: `qwen3.5:latest`, `qwen3.6:latest`, `gpt-oss:20b`, `deepseek-r1:8b`.

Comparisons, all declared now:
- Each added system against the constant predictor, and five-shot against zero-shot
  for each system.
- With five examples: the 40M mixed warm start against each added system.
- Zero-shot: the 40M model trained on random points against each added system.
- The Haiku agent against the Sonnet agent, with five examples.

The planned primary comparison is unchanged. For this second family, p-values are
reported both raw and Holm-adjusted over the whole secondary family, and the adjusted
value is the one to read. No comparison will be added after the data are seen. Batches
that have to be repeated are logged as before.

## Note on the headline, added after the first-round results were seen

After the 40M and Sonnet results were known, the author decided that the headline
comparison in write-ups is the zero-shot column (no worked examples in the prompt),
because that column measures what a model brings on its own. This choice was made
after seeing the data, so it is recorded here. The comparison fixed in advance
(40M with five examples against the hosted agent with five examples) is unchanged
and is still reported, and the final table always shows both columns. Zero-shot
comparisons were already among the declared secondary ones, so their Holm-adjusted
p-values are the ones to read.
