# Milestone: Track A complete (from scratch), 2026-10-24

This file freezes what the from-scratch track built and measured before the
project opens Track B (an open pretrained model adapted to the same domain).
Nothing below is re-run for Track B; Track B is compared against these numbers.

## What exists

| Stage | What | Key measured result | Where |
|---|---|---|---|
| Corpus | 3.2M PubMed cell-biology abstracts | 59/1000 eval items overlapped (5.9%), reported both ways | `data/corpus/`, `06_eval_release/` |
| Tokenizer | BPE 16k trained on the corpus | domain vocabulary | `01_tokenizer/` |
| Pretrain | 40M decoder, random init, one GPU | held-out ppl 11.50 | `03_pretrain/` |
| SFT, DPO | instruction + preference tuning | answer format learned; ppl 14.59 | `04_sft/`, `05_dpo_rlvr/` |
| RLVR, textbook | Nernst / polarity graded by the equation | held-out reward 0.000 → 0.977 | `05_dpo_rlvr/` |
| RLVR, simulator | BETSE outcome is the reward | exam A 0.945 (numeric 0.912) | `08_betse_rlvr/` |
| Exam design | sealed off-grid exam B | lattice-trained 0.539 → random-trained 0.777 → +5-shot 0.835 | `08_betse_rlvr/out/two_exams.png` |
| Library | FTS5 over 3.2M abstracts + quote-only reader | token-F1 0.990, exact 0.957; declines when no quotable sentence | `11_rag/` |
| Live demo | textbook lane + simulator loop | public at https://drnamlabs.com/cellbio/ (2 lanes) | `10_demo/` |

Comparisons on exam A (98 held-out grid problems, same verifier): constant
0.538; own ladder zero-shot 0.000-0.321; best open chat model 0-shot 0.529,
5-shot 0.904; CellBio-GPT 40M 0.945.

## What it cannot do (measured)

- Long or how/why questions fail on every lane (long-question probe,
  2026-10-24): the chat model writes self-contradicting PubMed boilerplate,
  the RL'd model collapses to 3-word templates, the library lane now declines.
- No multi-turn dialogue: 512-token context, single-turn training data.
- No synthesis across sources; one quoted sentence or a decline.
- Fails on sharp regions of the response surface (the Dm_K cliff).
- PubMedQA at the majority baseline: the prior, not the evidence, answers.

## Why Track B

These limits come from scale and from general-language data that a
3.2M-abstract corpus cannot supply, not from the method. Track B keeps the
two sockets (domain corpus, physics verifier), the exams and the demo, and
swaps the from-scratch 40M for an open pretrained instruct model with
long-form and dialogue ability. Both tracks are scored on identical exams.
