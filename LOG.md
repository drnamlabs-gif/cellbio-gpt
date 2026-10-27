# Development log

Chronological notes kept while building the project. The front page is
`README.md`; the written account is `09_report/tech_report/report.pdf`.

Build the full modern LLM stack from scratch **for cell & tissue biology** —
one runnable stage at a time, ending where industry interviews actually probe:
**post-training (SFT → DPO/RLVR) and evaluation**, not just pretraining.
No toy data: the corpus is PubMed/PMC cell-biology text (see `DATA.md`), and the
RLVR stage uses the BETSE bioelectric simulator as a verifiable reward.

Hardware target: single RTX 5090 (torch 2.11.0+cu128), ≤10–12 CPU threads
(set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `torch.set_num_threads` explicitly),
≤30GB of the 32GB VRAM (`torch.cuda.set_per_process_memory_fraction(30/32)`) — the machine is shared.

## Rules of the road

- One stage = one folder = one runnable artifact with a figure/animation, not just a loss number.
- Prototype with 1 seed (always 42, always recorded in saved metadata); multi-seed only for results being defended.
- Validate on one tiny case before any batch/scale run.
- Everything written to be public (English, clean README per stage) — this repo IS the portfolio.

## Persistence rules — nothing exists unless it is saved

Training time is the scarce resource; nothing expensive is ever recomputed.

- Every stage writes its products to `<stage>/out/`: checkpoints, figures (PNG),
  metrics (JSON/CSV), logs, sample generations. Console output alone counts as lost.
- Checkpoints: save every N steps during pretraining plus `final.pt` and `best.pt`;
  a crashed run must resume from the last checkpoint, never restart.
- Versioned inputs are immutable: `data/corpus/` and `01_tokenizer/tokenizer.json`
  are never overwritten — a new variant gets a new filename (e.g. `tokenizer_v2_32k.json`).
- Every saved artifact carries metadata: seed, date, data snapshot, config.
- Figures are written to disk by the producing script/notebook (explicit paths),
  then referenced from stage READMEs.

## Stages

| # | Folder | Build | Proof it works |
|---|--------|-------|----------------|
| 0 | `00_setup/` | env check: torch + CUDA on 5090, thread caps | matmul benchmark + device report |
| 1 | `01_tokenizer/` | byte-pair encoding from scratch, trained on a PubMed cell-bio sample | round-trip + compression-ratio plot; show domain terms ("depolarization", "Vmem") become single tokens |
| 2 | `02_transformer/` | GPT decoder in pure PyTorch (attention, RoPE, KV-cache) — no `nn.Transformer` | overfit one batch; char-level Shakespeare samples |
| 3 | `03_pretrain/` | 40–100M param model on the PubMed cell-biology corpus (`DATA.md`); mixed precision, cosine LR, grad accum | loss curve vs Chinchilla expectation; coherent in-domain samples |
| 4 | `04_sft/` | instruction-tune the stage-3 model on BETSE QA + PMC-derived QA + protocols; then LoRA on a 1–3B open model (bridge to existing Qwen2.5-VL work) | before/after instruction-following comparisons |
| 5 | `05_dpo_rlvr/` | **the portfolio gap**: DPO from scratch (loss derived + implemented), then GRPO with the BETSE simulator as verifiable reward | win-rate vs SFT checkpoint; reward curves |
| 6 | `06_eval_release/` | eval harness (PubMedQA, LAB-Bench, held-out BETSE + lm-eval-harness), vLLM serving demo, HF release + English writeup | model card, benchmark table vs frontier baselines |
| 7 | `07_baldwin_rlvr/` | **research stage**: phenotype-first RLVR — Baldwinian GRPO (in-context scaffold + assimilation anneal), from the author's evolution-of-learning research program | beats the stage-5b 0.60 held-out ceiling, or a documented refutation |
| 8 | `08_betse_rlvr/` | RLVR against the **real BETSE simulator**: config knobs → batch sims → cached tissue-level truths (mean Vmem, polarity) as the verifier — the law-vs-simulator gap closed | verifier = actual multi-cell physics; held-out-combo reward |

## Why this order

Stages 1–3 are the "I understand the machine" proof (nanoGPT territory — table stakes).
Stages 4–6 are what 2025–26 LLM research-scientist JDs actually ask about and where
few academic applicants have hands-on evidence. Stage 5's verifiable-reward angle
uses the BETSE simulator as a free RLVR signal.

## What "success" means here (the honest verdict after stage 6)

Two different questions, two different answers:

- **Did the pipeline work?** Yes, measurably, at every rung: pretrain took the
  model from token soup to fluent domain prose (ppl 11.5); SFT flipped
  autocomplete into answering; DPO removed the targeted failure (circular
  answers 4.4%→0.0%) without breaking fluency; GRPO taught a brand-new
  verifiable skill (Nernst problems 0→0.977). Same 40M weights, completely
  different behavior — that is everything post-training can do, demonstrated.
- **Is the model smart?** No — **by design**. 40M params is ~4 orders of
  magnitude below frontier; no model at this scale reads evidence and reasons
  (PubMedQA separates from the majority baseline only at billions of params).
  The scale was chosen by the corpus's Chinchilla budget, and the repo's goal
  was never a competitive chatbot: it is proof of building the full stack by
  hand and **measuring honestly what it can and cannot do** — including
  root-causing its own failures (stage 6e) instead of just reporting scores.

## Status

- 2026-10-22: repo created, roadmap set. Stage 0 PASSED on RTX 5090
  (34.2 GB, bf16 OK, 4096³ matmul: 66 TFLOP/s fp32 / 179 TFLOP/s bf16).
- 2026-10-22: domain fixed = cell & tissue biology (`DATA.md`). Validate-one on
  a real baseline file PASSED: 5.0% in-domain yield, 516 tok/doc → ~1.7B-token
  corpus.
- 2026-10-22: corpus fetch running (date-sliced efetch; NCBI caps efetch at the
  first 10k records of a result set, hence the EDAT bisection). 3,199,432 docs
  total. `data/corpus_stats.ipynb` ready to run once the fetch completes.
- 2026-10-22: stage 1 BPE written from scratch (`01_tokenizer/bpe.py`) and
  smoke-PASSED on 2k docs / 500 merges (round-trip exact, save/load OK, 1s).
  Convention: seed 42, recorded in every saved artifact.
- 2026-10-23: corpus COMPLETE (482 slices, 3,198,416 docs, 1.6GB, 0 failures,
  0 dup PMIDs). Stats notebook executed → `data/out/`: 616M words ≈ 0.75–0.8B
  BPE tokens (1.7B projection corrected), p99 = 368 words → context 1024.
  Chinchilla target revised to ~40M params (or 60–80M at 2 epochs).
- 2026-10-23: stage 1 DONE. Full tokenizer (16,384 vocab, 46s train) PASSED:
  " depolarization"→1 token, " regeneration"→1, compression 4.91 bytes/token.
  Next: `02_transformer/` — GPT block from scratch, overfit-one-batch test.
- 2026-10-23: stage 2 DONE. GPT from scratch (RoPE, RMSNorm, SwiGLU, weight
  tying, KV-cache) smoke-PASSED: loss 9.77→0.049 in 68 steps, manual==fused
  attention (2.4e-06), cache==recompute token-exact. Figures in
  `02_transformer/out/`. Next: `03_pretrain/` — 40M params on the full corpus.
- 2026-10-23: stage 3 DONE. Corpus encoded (988.5M train / 9.2M val tokens,
  temporal holdout = 5 newest slices). Smoke 300 steps PASSED (archived in
  `03_pretrain/out/smoke/`), then full pretrain: 40.5M params, 30,167 steps
  = 1 epoch, 123 ms/step, ~62 min on the 5090. **Final val loss 2.4363**
  (ppl 11.4; init = ln(16384) = 9.70). Train ≈ val throughout (single epoch,
  no overfit). Samples went from token soup → fluent (confabulated) methods
  prose about patch clamp / ATPase kinetics. Artifacts: `final.pt`, `best.pt`,
  `ckpt.pt` (resume state), `log.jsonl`, `loss_curve.png`, `samples.txt`.
  Next: `04_sft/` — instruction tuning (BETSE QA + PMC QA + protocols).
- 2026-10-23: stage 4 DONE (v1, corpus-mined data). SFT dataset built from the
  corpus itself, no external LLM: 97.5k examples, 4 self-supervised tasks
  (question-titles, definitions, abbreviations, title generation), temporal-val
  slices excluded. SFT = stage-3 loop + chat template + loss mask (−100 on
  prompt tokens); init final.pt, LR 6e-5, 2 epochs = 3,048 steps ≈ 5 min.
  Masked val 2.316. Before/after (`04_sft/out/before_after.png`): base
  autocompletes, SFT answers — "What is a gap junction?" → correct definition;
  title generation genuinely good. Kept honest failures ("ATP stands for ATP",
  one repetitive answer) — the motivation for stage 5 preference optimization.
  BETSE QA deferred to stage 5 where the simulator becomes the verifier.
  Next: `05_dpo_rlvr/` — DPO from scratch, then GRPO with BETSE reward.
- 2026-10-23: stage 5a DONE — DPO from scratch. Pairs self-mined (10.8k):
  chosen = gold, rejected = the SFT model's own sample (on-policy) + circular
  hard negatives. Loss implemented from the formula; step-0 loss = ln 2 exactly
  (policy≡ref sanity check). v1 (β=0.1) over-optimized — fixed circularity but
  broke fluency (kept as instructive failure). v2 (β=0.3, LR 2e-6, clean tasks):
  circular answers 4.4%→0.0% on held-out prompts, repetition and on-topic
  quality preserved (`05_dpo_rlvr/out/dpo_vs_sft_v2.png`). Honest limit: DPO
  reshapes preferences but cannot add missing knowledge ("ATP" long form).
  Next: stage 5b — GRPO with the BETSE simulator as verifiable reward.
- 2026-10-23: stage 5b DONE — GRPO from scratch with a bioelectric-physics
  verifier (Nernst equation = the membrane law BETSE integrates). Generator +
  checker, no human labels: 3 problem families, combo-level held-out val.
  Warm-start (400 SFT steps) → sign-rule tasks generalize to 1.00; GRPO
  (16×8 samples/step, group advantage, KL leash) climbs train reward
  0.72→0.87 with zero categorical regression; held-out numeric improves
  0.57→0.60 only — honest scale finding: 40M memorizes combos, it cannot
  learn ln(). Figures: `05_dpo_rlvr/out/explain_grpo.png`, `grpo_results.png`.
  Next: `06_eval_release/` — eval harness + model card + HF release.
- 2026-10-23: stage 6 DONE — the ladder scored on exams it never set.
  Contamination audit first: 59/1000 PubMedQA items (5.9%) share a PMID with
  the pretraining corpus → all scores reported full AND clean. Harness runs
  the same 3 exams on every rung (base/SFT/DPO/GRPO): temporal-holdout
  perplexity 11.50→14.59→14.59→16.24; PubMedQA (likelihood scoring)
  0.537→0.491→0.496→0.460, ALL below the majority baseline 0.552 — honest:
  40M can't do biomedical MCQ reasoning, and post-training pays a visible
  alignment tax; BETSE physics 0.00→0.24→0.24→0.977 — what the tax bought.
  clean ≈ full everywhere, so nothing is contamination-inflated.
  `release/cellbio-gpt-40m/`: 4 checkpoints + tokenizer + standalone
  `generate.py` + model card rendered from `eval_results.json`.
  Figures: `06_eval_release/out/explain_eval.png`, `eval_results.png`.
  Remaining (optional): HF upload (needs account login), English writeup/post.
- 2026-10-23: stage 6e — PubMedQA failure root-caused, not just reported.
  All four checkpoints answer "no" exactly 0/1000 times: an empty-prompt probe
  shows a ~7-nat prior gap against "no" (no training stage ever emits yes/no
  answers), so raw likelihood scoring was measuring the prior — which mimics
  the majority class. PMI calibration frees "no" but accuracy falls to chance;
  deleting the abstract moves raw accuracy by only ~2 pts. Verdict: zero
  evidence-reading capability at 40M; the 0.46–0.54 scores were prior
  artifacts. `06_eval_release/out/pubmedqa_diagnosis.png` + model card updated.
- 2026-10-23: stage 7 DESIGNED (`07_baldwin_rlvr/`) — the Adaptation Stack
  reading of the whole repo (pretraining=evolution, ICL=lifetime learning,
  SFT/distillation=culture; GRPO = Lamarckian odd-one-out) recasts the
  stage-6 result: we built an instinct-only creature, and both failures
  (PubMedQA evidence-blindness, the ln() memorization ceiling) are the
  missing middle layer. Proposed algorithm: phenotype-first RLVR — GRPO
  with an in-context worked-example scaffold (plasticity channel) annealed
  k:5→0 (genetic assimilation), selection on the post-learning phenotype.
  Falsifiable predictions P1/P2 + controls specified in the stage README;
  figure `07_baldwin_rlvr/out/explain_stack.png`. Not yet run.
- 2026-10-23: stage 7 RUN (1 seed). Smoke exposed a real precondition:
  scaffolds HURT a single-format model (sens −0.46) — plasticity must be
  taught before Baldwin can act → warm-start v2 with mixed k 0..5. Three
  arms from the same ws2 (val 0.841, nernst 0.321): direct k=0 → 0.514,
  k=5-fixed → 0.517, **anneal 5→0 → 0.564** nernst on held-out combos at
  k=0, with the predicted lag-then-overtake at the assimilation phase and
  lower KL. P1 directionally confirmed (weak: 1 seed, small val); P2 not
  confirmed (gains in weights, not reading); k5-collapse refuted. All in
  `07_baldwin_rlvr/out/baldwin_results.png` + README §5. New machine rule:
  ≤30GB of 32GB VRAM (`set_per_process_memory_fraction`), now in all scripts.
- 2026-10-23: stage 8 validate-one PASSED — the REAL BETSE simulator now runs
  inside the repo (`.venv-betse`, Python 3.11 + betse 1.5.0; 1.5.0 is broken
  on 3.13). Full chain ≈13 s/config: config → seed → init → sim → per-cell
  Vmem readout. Knob→truth verified: extracellular K+ 5→25 mM (charge-
  balanced against Na+) depolarizes the 213-cell tissue −43.76 → −20.69 mV.
  Gotchas recorded in KNOWLEDGE.md (`ion profile: custom` required; charge
  neutrality enforced). Next: problem generator + batch sims → RLVR where
  the verifier is the simulator itself.
- 2026-10-23: stage 8 DONE — the simulator loop closed end-to-end. 150 real
  BETSE sims (6.8 min, 0 failures; Vmem −74.9..−7.2 mV, the Dm_K×4 slice
  non-monotone = emergent multi-cell behavior with no closed form) became
  466/98 combo-held-out RLVR problems. Warm-start + GRPO on the sim-truth
  verifier: held-out val 0.938 → **0.945** (numeric 0.898 → 0.912, polarity
  0.979). First stage where 40M generalizes NUMERICALLY — honest mechanism:
  the 3-knob response surface is smooth enough for memorize-and-interpolate,
  unlike 5b's log-ratio space. Demo on unseen combos: model −46 vs sim −47 mV.
  Artifacts: `08_betse_rlvr/out/` (sim_truths, problems, bg8_*.pt, logs,
  `betse_results.png`).
- 2026-10-24: stage 8c — Baldwin applied to the real-sim problems
  (`baldwin_betse.py`, anchors = nearest-neighbor combos with sim-measured
  Vmem; sens gap +0.106: the plasticity channel genuinely works here).
  Result (300 steps, `baldwin_betse_results.png`): direct RL wins 0.878 vs
  Baldwin-anneal 0.809 (numeric 0.800 vs 0.665); the anneal arm's entire
  gain lands in its 50-step k=0 phase at a ~27% steeper per-step slope —
  smoothing is real, but cannot amortize 250 scaffolded steps when direct RL
  already has partial-credit gradient. Paired with stage 7 (Baldwin won when
  the scaffold was ignored), this brackets the mechanism: Baldwin's niche is
  needle landscapes where direct RL has NO gradient — the sharpened
  falsifiable test (binary sparse reward) is specified in KNOWLEDGE.md.
  v1 run (an anneal-schedule slip giving zero k=0 steps) kept as control:
  scaffolded training alone improves anchor-use, transfers nothing.
- 2026-10-24: stage 8d — baseline table for the sim exam (`eval_baselines.py`,
  `08_betse_rlvr/out/betse_baselines.png`). Generic ladder zero-shot scores
  0.000 (base/SFT/DPO; GRPO-textbook 0.321) vs fine-tuned 0.945 — the sim
  fine-tune is necessary. Honest floor: a no-LLM 1-NN knob lookup scores
  0.927 (numeric 0.856 vs model 0.912), so claims are benchmarked against
  interpolation, not against zero. Also an interface explainer figure
  (`explain_interface.py` → `explain_interface.png`): the LLM never talks to
  BETSE — 150 offline sims → text problems → verifier parses the model's mV.
- 2026-10-24: stage 8d (cont.) — pretrained LLMs sit the same exam via Ollama
  (`eval_ollama.py`, 6 models × 0-shot/5-shot). Best: qwen3.5 5-shot 0.904 <
  40M's 0.945; best 0-shot 0.529 < constant floor. Preprint skeleton
  opened: `09_report/draft.md` — title "Physics-verified reinforcement
  learning for a tiny biology LLM", Nature-template abstract v1, outline,
  figure plan; Baldwin reserved for paper 2.
- 2026-10-24: stage 8e — the OFF-GRID exam (user's challenge: "your exam was
  too predictable"). 60 sealed decimal-knob sims between lattice nodes
  (`make_offgrid_exam.py`): lattice-trained 40M collapses 0.945→0.539, below
  1-NN 0.793 AND constant 0.663 — it was a soft lookup table (17 distinct
  answers, 16 memorized; `diag_offgrid.py`). Case 2: retrain on 150 RANDOM
  points, same budget (`make_random_train.py`) → 0.777 off-grid (+0.238 from
  data design alone), predictions now track truth over nearest-node, but
  still 0 novel output values and still below 1-NN-with-oracle-knobs 0.856.
  Figure: `08_betse_rlvr/out/two_exams.png` — exam design decides what a
  score means.
- 2026-10-24: stage 8e case 3 — the 40M's own 5-shot: mixed-k SFT on the
  random data + 5 nearest worked examples as text in the prompt (same
  language starting line as the pretrained models' 5-shot). Off-grid ladder:
  0.539 (lattice) → 0.777 (random data) → **0.835** (+ in-context examples).
  GRPO on top of anchors slightly hurt off-grid (0.812) — honest note kept.
  Baseline policy fixed: main tables compare language-reading systems only
  (chance/constant floors); 1-NN demoted to internal diagnostic. gpt-oss:20b
  rescored fairly (0.455/0.664) after a thinking-channel harness bug.
- 2026-10-24: stage 10 — LIVE closed-loop demo (`10_demo/`): local web app
  (`serve_demo.py`, http://localhost:8765) with two lanes — free-text
  bioelectricity Q&A (textbook GRPO model) and the simulator experiment loop:
  design an intervention → 40M predicts mean Vmem with 5 retrieved worked
  examples in context (case-3 recipe) → the REAL BETSE sim runs (~19 s,
  cached per combo) → physics verdict ✓≤2 / ≈≤10 / ✗ mV → and the tissue
  itself, rendered: animated per-cell Vmem map + membrane-potential-vs-time
  trace straight from the sim. Demo recording
  `10_demo/out/cellbio_gpt_demo.gif` (slowed, ~50 s) shows the loop catching
  a wrong prediction on the Dm_K cliff (−57 vs −7.06, ✗, tissue flashes red)
  and verifying a correct one (−44 vs −43.9, ✓ 0.1 mV, tissue stays blue).
  Social drafts (EN, not posted): `10_demo/social_drafts.md`.
- 2026-10-24: Track B, B0 + B1 on Qwen2.5-1.5B-Instruct. Untouched: exam A
  0.427 / exam B 0.377 (below the constant floor) but fluent long answers
  and dialogue. After LoRA SFT on our own data only (`12_track_b/
  b1_domain_sft.py`): exam A 0.914, exam B 0.839, reading F1 0.990, with
  fluency and dialogue intact. Free-form explanations remain unsourced and
  sometimes wrong → B3 grounding next. Figure
  `12_track_b/out/b1_Qwen2.5-1.5B-Instruct.png`.
- 2026-10-24: MILESTONE — Track A (from scratch) frozen in
  `MILESTONE_track_A.md`. Long-question probe showed the 40M ceiling (no
  long answers, no dialogue, no synthesis). Track B opened
  (`12_track_b/README.md`): an open pretrained instruct model adapted to the
  same domain through the same two sockets (library + BETSE verifier),
  scored on Track A's exams. Stages B0 baseline → B1 LoRA domain SFT → B2
  BETSE RLVR → B3 grounded long answers with PMIDs → B4 dialogue → B5
  inverse control.
- 2026-10-24: stage 11 — RAG: the from-scratch encyclopedia lane. SQLite
  FTS5 index over all 3.2M abstracts (`11_rag/build_index.py`, 8.2GB);
  29,054 self-mined definition-QA samples, no teacher model
  (`make_defqa.py`: "<term> is a ..." sentences as gold, surrounding
  abstract as context); `rag_sft.py` teaches dpo_final_v2 to READ →
  `rag_qa.pt`: held-out token-F1 **0.990**, exact 0.957, faithful 0.962
  (first eval said 0.39 — a LEFT-PADDING artifact, model never saw pads;
  bucketed generation is mandatory). Demo gains an "Ask the literature"
  tab: BM25 + definition-first rerank → 40M extracts the answer → source
  PMID shown. "What is bioelectricity?" now answers "the electrical
  activity that occurs within living cells and tissues" with a citation.
  Known edges: non-"what is" phrasings and occasional misread stitching.
- 2026-10-24: stage 10 LIVE — the demo is public at
  **https://drnamlabs.com/cellbio/** (user-approved deploy). Server bundle in
  `10_demo/server/` (self-contained: serve_cellbio.py + run_sim_server.py +
  index.html with relative URLs). On a small 2-vCPU server betse 1.5.0 runs
  on py3.12; a niced sim takes ~66 s. Guards: clamped inputs, one sim at a
  time, a daily cap on fresh sims.
