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
