# Knowledge Tree

Dr. Nam H. Le — drnamlabs.com, bunnyquant.com

Index of every saved artifact in this repo: what it is, where it lives, one-line
hook for search. Updated at the end of every step. Nothing exists unless it is
saved AND listed here.

## Plan & explainer
- `LOG.md` — 7-stage roadmap, rules (seed 42 recorded, validate-one-first, persistence), status log
- `DATA.md` — data plan: sources, licensing, validate-one results, corpus projection

## 00_setup — environment
- `00_setup/check_env.py` — GPU check script (threads capped at 10)
- RESULT (2026-10-22, in LOG status): RTX 5090, 34.2 GB, torch 2.11.0+cu128;
  matmul 66 TFLOP/s fp32 / 179 TFLOP/s bf16

## data — corpus
- `data/build/fetch_corpus.py` — PubMed fetcher; EDAT-bisected slices because NCBI
  caps efetch at the first 10k of a result set (retstart≥10000 → HTTP 400)
- `data/build/validate_one.py` — one-baseline-file yield check; gotcha: history-combined
  esearch needs `usehistory=y` or silent "Wrong subquery"
- `data/corpus/slice_*.jsonl.gz` — THE corpus, immutable, COMPLETE 2026-10-23:
  482 slices, 3,198,416 docs, 1.6GB gz, zero failed slices, zero duplicate PMIDs
  (MeSH union: Morphogenesis, Regeneration, Cell Physiological Phenomena,
  Tissue Engineering, Electrophysiological Phenomena)
- `data/corpus_stats.ipynb` — streaming stats notebook (seed 42), EXECUTED with outputs saved
- `data/out/corpus_stats.png` — length histogram + year coverage (mass of corpus is 2000–2026)
- `data/out/corpus_summary.json` — RESULT: 616M words ≈ **0.75–0.8B BPE tokens**
  (earlier 1.7B projection was off: validate-one file had unusually long abstracts;
  full-corpus mean is 193 words/doc, p99 = 368 words ≈ 500 tokens → context 1024 ample)
  → Chinchilla-optimal model ≈ **40M params** (or 60–80M at ~2 epochs; decide in stage 3)
- Validate-one result (2026-10-22, file pubmed26n1200): 5.0% yield, 516 tok/doc mean

## 01_tokenizer — BPE from scratch
- `01_tokenizer/bpe.py` — byte-level BPE: word-type training, incremental pair counts,
  lazy heap, cached encode; GPT-2-style pretokenizer (leading space attached)
- `01_tokenizer/train_tokenizer.py` — CLI: `--smoke` (2k docs/500 merges) or full
  (400k docs sample, 16,384 vocab); saves meta {seed, docs, vocab, date}
- `01_tokenizer/out/tokenizer_smoke.json` — smoke artifact (2026-10-22): round-trip exact,
  2.45 bytes/token at 500 merges, 1s train
- `01_tokenizer/out/tokenizer.json` — FULL tokenizer DONE 2026-10-23 (seed 42 in meta):
  400k-doc sample, 540,516 word types, 16,127 merges in 46s, round-trip exact,
  compression 4.91 bytes/token. PASS: " depolarization"→1 tok, " regeneration"→1 tok,
  " gap junction"→2, " bioelectric"→[bio, electric] (clean morphology)

## 02_transformer — GPT from scratch
- `02_transformer/model.py` — decoder-only GPT, hand-written attention
  softmax(QKᵀ/√d + causal mask)V; RoPE, RMSNorm, SwiGLU, weight tying, KV-cache;
  `fused=True` = F.scaled_dot_product_attention path for stage-3 speed
- `02_transformer/overfit_one_batch.py` — smoke: overfit 8×256 tokens + 2 parity asserts
- `02_transformer/out/` — RESULT 2026-10-23 (seed 42, 7.4M params, cuda): PASSED —
  loss 9.77→0.049 in 68 steps (1s); manual-vs-fused max diff 2.4e-06;
  KV-cache generation token-exact vs full recompute; greedy continuation
  reproduces the memorized abstract verbatim (sample.txt)
- `out/overfit_loss.png` — loss curve, log scale, pass threshold marked
- `out/attention_map.png` — layer-0 heads: upper triangle empty (causality visually
  verified), attention-sink on token 0, near-diagonal locality in head 2
## 03_pretrain — pretraining on the full corpus
- `03_pretrain/encode_corpus.py` — one-time corpus→uint16 binary; 10 workers,
  23.7M tok/s. RESULT 2026-10-23: `data/tokens/train.bin` = 988,531,863 tokens
  (2GB), `val.bin` = 9.2M tokens (TEMPORAL holdout: 5 newest slices, Jul–Dec 2026),
  meta.json records vocab 16,384 / eot 16,383
- `03_pretrain/pretrain.py` — training loop: bf16 autocast, fused SDPA, AdamW
  (decay on matrices only), warmup 500 + cosine, clip 1.0, seed 42; checkpoint
  every 1000 steps with optimizer+rng (crash → `--resume`); val every 1000;
  samples every 5000 → out/samples.txt
- `out/smoke/` — smoke run 2026-10-23: 300 steps, 122 ms/step, loss 9.8→5.1,
  val 5.20 (archived, never overwritten)
- FULL RUN DONE 2026-10-23 (seed 42): 40.5M params, 30,167 steps = 1 epoch over
  988M tokens, 123 ms/step (~62 min total). **Final val loss 2.4363** (ppl ≈ 11.4),
  down from 9.70 = ln(16384) at init. Train ≈ val throughout (1 epoch → no overfit)
- `out/final.pt` (155MB), `out/best.pt` (best-val weights), `out/ckpt.pt`
  (full resume state: model+opt+rng, 464MB), `out/log.jsonl` — all saved
- `out/loss_curve.png` — train per-batch + val markers; smooth cosine-decay tail
- `out/samples.txt` — generation every 5k steps from "Membrane potential": fluent
  patch-clamp/ATPase abstract prose by step 30k (facts confabulated — expected
  for 40M base model; style and terminology are domain-correct)
- `03_pretrain/plot_run.py` — re-plottable curve from log.jsonl
## 04_sft — instruction tuning
- `04_sft/explain_sft.py` → `out/explain_sft.png` — explainer figure (2026-10-23):
  base model = autocomplete vs SFT = answers; chat template + loss mask on prompt
  tokens (the one new mechanism); scale bar: SFT ≈ 5–15M tokens vs 988M pretrain
- `04_sft/build_sft_data.py` — SFT data mined from the corpus itself, no external
  LLM: 4 self-supervised tasks (qtitle = question-titles → abstract conclusions,
  defn = "X is a..." sentences, abbrev = Schwartz-Hearst-lite long forms,
  titlegen = abstract → real title). Temporal-val slices EXCLUDED (contamination).
  RESULT 2026-10-23 (seed 42): 97,553 train / 1,990 val, balanced ~24.5k/task
  (raw mined: 24.5k qtitle, 298k defn, 883k abbrev, 2.9M titlegen → capped)
- `out/sft_train.jsonl` / `sft_val.jsonl` / `sft_meta.json` / `sft_data_stats.png`
- `04_sft/sft.py` — SFT loop: identical to stage 3 except (1) chat template
  "### Instruction / ### Response", (2) loss mask = targets −100 on prompt+pad
  (F.cross_entropy default ignore_index — no model change needed). Init from
  final.pt, LR 6e-5 (10x below pretrain), 2 epochs. Smoke 100 steps PASSED
  (85 ms/step, artifacts *_smoke.*)
- FULL SFT DONE 2026-10-23 (seed 42): 3,048 steps ≈ 5 min, masked val
  2.3309→2.3159 (best 2.3101). `out/sft_final.pt` / `sft_best.pt` / `sft_log.jsonl`
- `04_sft/compare.py` → `out/before_after.md` / `before_after.png` — same prompts,
  base vs SFT, greedy. RESULT: behavior flip is unmistakable (base autocompletes
  "The aim of this study was..."; SFT answers "The gap junction is a highly
  specialized membrane channel..."; titlegen produces a genuinely good title).
  HONEST failures kept in the figure: "ATP stands for ATP" (circular — common
  abbrevs never get expanded in abstracts, so the pattern was unlearnable) and
  degenerate repetition on one open question → motivates stage 5 (DPO/RLVR)
## 05_dpo_rlvr — preference optimization + RLVR
- `05_dpo_rlvr/explain_dpo.py` → `out/explain_dpo.png` — explainer (2026-10-23):
  SFT's blind spot (no concept of WRONG), the DPO loss (4 log-probs + sigmoid,
  β = leash to frozen ref), pair sources, GRPO+BETSE preview
- `05_dpo_rlvr/build_pref_data.py` — preference pairs, self-mined, no external
  LLM: chosen = gold, rejected = SFT model's OWN temp-0.8 sample (on-policy)
  or circular hard negative "X stands for X" (the observed stage-4 bug).
  Batched generation bucketed by exact prompt length (~26 gen/s).
  RESULT 2026-10-23 (seed 42): 10,841 train / 500 val pairs
  (8.9k on-policy + 1.9k circular; tasks qtitle/defn/abbrev)
- `05_dpo_rlvr/dpo.py` — DPO from scratch: masked per-sequence logprobs,
  margin = β[(πθ−πref)(y⁺) − (πθ−πref)(y⁻)], L = −logσ(margin); policy+ref
  both init from sft_final.pt; chosen & rejected in one forward.
  Sanity check built in: step-0 loss = ln 2 = 0.6931 exactly (policy≡ref).
  CLI: --beta --lr --tasks --tag (tags never overwrite earlier versions)
- v1 RESULT (dpo_final.pt, β=0.1, lr 5e-6, all tasks, 677 steps): val acc 0.877,
  margin +2.68 — OVER-OPTIMIZED. Fixed circularity (4.4%→0%) but rep4 0.27→0.46,
  answers drift off-topic. GOTCHA: noisy chosen (qtitle conclusions) + loose β
  → classic DPO fluency collapse. Kept as the instructive failure.
- v2 RESULT (dpo_final_v2.pt, β=0.3, lr 2e-6, defn+abbrev only, 438 steps):
  val acc 0.863, margin +1.93. Held-out metrics (out/dpo_vs_sft_v2.png):
  circular 4.4%→**0.0%**, rep4 0.272→0.276 (unchanged), length 21.5→24 words,
  on-topic answers preserved ("GJ stands for gap junction intercellular
  communication"). Honest remainder: ATP now non-circular but wrong long form —
  DPO cannot add knowledge the base never saw. → stage 5b GRPO/BETSE
- `out/dpo_log*.jsonl`, `dpo_vs_sft*.json/png`, `dpo_before_after*.md` — all saved

### 5b — GRPO + bioelectric verifier (RLVR)
- `05_dpo_rlvr/explain_grpo.py` → `out/explain_grpo.png` — explainer: RLVR idea
  (verifier = Nernst physics, BETSE's core equation), one GRPO step drawn
  (group-relative advantage, no critic/no reward model), 2 practical truths
  (warm-start or no gradient; uniform groups skipped)
- `05_dpo_rlvr/bioelectric.py` — physics ground truth + reward parser.
  3 families: nernst (numeric mV, graded ±2/±10/±25), deltaE (incr/decr),
  polarity (de/hyperpolarize). Gold answers must score 1.0 (self-test)
- `05_dpo_rlvr/make_problems.py` — combo-level 85/15 split (held-out val combos
  NEVER appear in ws or RL). RESULT: 3,000 ws / 4,028 train / 727 val
- `05_dpo_rlvr/warmstart.py` → `out/ws_final.pt` — 400 SFT steps (11s) from
  dpo_final_v2; teaches format. Baseline on held-out combos: deltaE 1.00,
  polarity 1.00 (sign rule GENERALIZES), nernst 0.565 (headroom → RL target)
- `05_dpo_rlvr/grpo.py` — GRPO from scratch: 16 prompts × K=8 samples/step,
  verifier rewards, A=(r−mean)/std per group, token PG loss + KL 0.05 to ws ref,
  uniform groups skipped, nernst oversampled 60%. Smoke 30 steps PASSED.
- FULL RUN 2026-10-23 (300 steps, ~7.5 min, seed 42): train reward 0.72→0.87
  (exact 0.55→0.75ish — RL clearly climbing on seen combos). Held-out combos:
  nernst 0.565→0.599 (best 0.621), categorical stays 1.00 (no regression).
  KL rose 0.00→0.22 by the end — drift = memorization of seen combos.
- HONEST finding: GRPO machinery works end-to-end (reward climbs, verifier
  loop sound, zero regression) but a 40M model does NOT learn ln() — gains on
  unseen combos are small; it memorizes trained combos. The categorical physics
  rules are what generalize. This is the expected scale story for RLVR.
- `out/grpo_log.jsonl`, `grpo_best.pt`, `grpo_final.pt`, `grpo_ckpt.pt`,
  `grpo_results.png` — all saved. v2 idea (not started): full BETSE sim configs
  as problems, precomputed outcomes as truth.
## 06_eval_release — eval harness + model card + release (DONE 2026-10-23)

- `06_eval_release/explain_eval.py` → `out/explain_eval.png` — explainer: why a
  release needs exams the model never set (3 axes: language/knowledge/physics),
  likelihood scoring for MCQ (no answer parsing, honest 1/3 chance), and the
  contamination audit (PubMedQA is built FROM our training corpus)
- `06_eval_release/fetch_pubmedqa.py` — downloads PubMedQA pqa_labeled (1k
  expert yes/no/maybe) + scans all 483 corpus slices for source-PMID overlap.
  RESULT: 59/1000 (5.9%) contaminated → every score reported full AND clean
- `06_eval_release/eval_harness.py` — same 3 exams for the whole ladder
  (base/sft/dpo_v2/grpo): perplexity on temporal-holdout val.bin (full pass,
  512-token windows), PubMedQA via length-normalized option logprob under the
  SFT template, BETSE 727 combo-held-out problems via grpo.eval_val.
  Smoke mode first, then full (~30 s/model on the 5090).
- RESULTS (seed 42, `out/eval_results.json`, figure `out/eval_results.png`):

  | model | holdout ppl | PubMedQA full | clean | BETSE |
  |---|---|---|---|---|
  | base | 11.50 | 0.537 | 0.526 | 0.000 |
  | SFT | 14.59 | 0.491 | 0.480 | 0.239 |
  | DPO v2 | 14.59 | 0.496 | 0.486 | 0.238 |
  | GRPO | 16.24 | 0.460 | 0.453 | 0.977 |

  Baselines: chance 0.333, majority "yes" 0.552.
- HONEST findings: (1) every rung sits BELOW the majority baseline on PubMedQA —
  40M cannot do biomedical reasoning MCQ; base scores highest because likelihood
  scoring favors raw LM calibration. (2) The alignment tax is visible and
  monotone: ppl 11.50→16.24 down the ladder, PubMedQA −8pts, bought BETSE
  0→0.977. Post-training reallocates capacity; it does not add intelligence.
  (3) clean ≈ full everywhere (≤1.1pt gap) — scores are not contamination-inflated.
- `06_eval_release/build_release.py` → `release/cellbio-gpt-40m/` — 4 stripped
  checkpoints (cfg+weights+stage metadata), tokenizer.json, model.py, bpe.py,
  standalone `generate.py` (tested: fluent, confabulating — as the card states),
  README.md = model card RENDERED FROM eval_results.json (card can never
  disagree with the harness). HF upload prepared but NOT pushed (needs owner).
- LAB-Bench skipped deliberately: multimodal/tool-use items are out of scope
  for a 512-ctx 40M text model; noted in roadmap as scale-appropriate honesty.

### 6e — PubMedQA root-cause diagnosis ("never says no")
- Trigger: pred distributions showed "no" chosen 0/1000 by ALL four checkpoints
  (recall on "no"-labeled items = 0.000 everywhere).
- `06_eval_release/diagnose_pubmedqa.py` → `out/pubmedqa_diagnosis.json/png`,
  three hypotheses tested (seed 42):
  - H1 tokenization: "yes"→[y,es], "no"→[no], "maybe"→[m,ay,be] — odd pieces
    (options encoded without leading space) but length-normalized; not the cause.
  - H2 prior bias CONFIRMED: with an EMPTY prompt, mean logprob of "no" is
    ~−13.5 vs "yes" ~−6.4 — a ~7-nat gap. No training stage ever produced
    "no"/"yes" as an answer (SFT answers were definitions/titles, GRPO answers
    were mV values / incr/decr), so the raw MCQ score = prior, which happens to
    mimic the majority class ("yes" = 55%). acc 0.537 ≈ accidental majority.
  - H3 no evidence signal CONFIRMED: PMI calibration (score − empty-prompt
    prior, Zhao et al. 2021) frees the "no" class (129–984 "no" preds) but
    accuracy FALLS to 0.34–0.47 ≈ chance; and removing the abstract entirely
    moves base raw acc only 0.537→0.519. The model does not read the evidence.
- CONCLUSION: the honest mechanism is "the prior was taking the exam, not the
  knowledge." Below-majority PubMedQA is not a near-miss — a 40M
  abstracts-continuation model has zero verdict-reasoning capability, and
  the raw 0.46–0.54 numbers were prior artifacts, not partial understanding.

## 07_baldwin_rlvr — phenotype-first RLVR (DESIGN 2026-10-23, not yet run)

- Bridge stage: connects the repo to the author's evolution-of-learning
  research program (Adaptation Stack book spine, Baldwin/GP papers,
  phenotype-first/phenopoiesis line). Full design in `07_baldwin_rlvr/README.md`,
  figure `out/explain_stack.png`.
- Adaptation Stack mapping: pretraining=evolution (instinct in weights),
  ICL=lifetime learning (ABSENT at 40M — the hole), SFT/DPO=culture
  (transmitted demonstrations/preferences). GRPO = Lamarckian odd-one-out:
  episode-level trial-and-error credited straight into the genome.
- Re-read of stage 6: both headline failures are the same missing middle
  layer. PubMedQA = within-lifetime task, model answers from instinct
  (context ablation ≈ 0 effect). ln() = needle-in-haystack reward landscape,
  direct-to-genome learning can only memorize (0.87 train / 0.60 held-out).
- Algorithm (Baldwinian GRPO): prompts carry k worked NEIGHBOR-combo examples
  (plasticity channel); GRPO selects on the post-learning phenotype;
  anneal k 5→0 = genetic assimilation pressure (Hinton & Nowlan 1987
  landscape-smoothing). Cultural variant 7b: verified traces → SFT next gen.
- Predictions: P1 final k=0 held-out nernst > 0.60 ceiling; P2
  context-sensitivity rises above ~0. Controls: k≡0 (= stage 5b, done),
  k≡5 (crutch never removed → collapse at k=0 test). Seed 42, minutes/arm.
- RUN 2026-10-23 (seed 42). GOTCHA found by smoke: with the single-format
  ws1 warm-start, scaffolds HURT (sens gap −0.46 — few-shot is OOD noise);
  plasticity had to be taught first → `warmstart2.py` ws2 with mixed k 0..5
  (gap +0.04; k=0 baseline drops to val 0.841 / nernst 0.321 — all 3 arms
  restart from ws2, so stage-5b's 0.60 is no longer directly comparable).
- RESULTS (300 steps/arm, `out/baldwin_results.png`, bg_log_*.jsonl):
  nernst k=0: direct 0.514, k5-fixed 0.517, **anneal 0.564** (+0.05, with
  the predicted lag-then-overtake during the k=1→0 assimilation phase, and
  lower KL 0.15 vs 0.25). P1 directionally confirmed but WEAK (1 seed,
  42 nernst val problems ≈ 2 problems of credit). P2 NOT confirmed (sens
  gap stays ±0.02 — gains travel through weights, not reading). k5-collapse
  prediction REFUTED (crutch arm transfers fine; model mostly ignores
  scaffold). Two of three predictions failed and are recorded as such.
  To defend: ≥5 seeds + bigger nernst val, then 1–3B.

## 08_betse_rlvr — REAL BETSE as the verifier (validate-one PASSED 2026-10-23)

- Honest gap it closes: stages 5b/7 verified against the closed-form Nernst
  equation (the law BETSE integrates), but no actual BETSE sim had ever run
  in this repo. Stage 8 = truth from the full multi-cell simulator (gap
  junctions, pumps, tissue geometry — not analytically computable).
- Environment: `.venv-betse` (Python 3.11 + betse 1.5.0). BETSE 1.5.0 is
  BROKEN on Python 3.13 (cached-property AttributeError) — use py -3.11.
- Validate-one chain, all PASSED: `betse config` → seed 2.7s → init 3.5s →
  sim 7s (≈13 s/config total) → read per-cell Vmem via
  `D:\language_to_living_systems\language_cppn_betse\betse_io.py`
  (sim pickles are `SIMS/sim_1.betse.gz`; 213 cells, baseline −43.76 mV).
- Knob→truth VERIFIED: extracellular K+ 5→25 mM (Na+ 145→125 to keep charge
  balance) depolarizes the tissue −43.76 → −20.69 mV. Two real gotchas:
  (1) ion concentrations in "customized ion profile" are IGNORED unless
  `ion profile: custom`; (2) env+cyto ion sets must each sum to zero charge
  or BETSE aborts (swap Na+ for K+, high-K-Ringer style).
- `betse config` on a dir where INITS/RESULTS/SIMS exist dies with
  die_if_path; the haiku error messages hide tracebacks — rerun with
  `--verbose`. extra_configs/ AND geo/ must both sit next to the yaml.
- `make_sim_problems.py` — 150-combo grid (bath K+ 10 values with Na+
  charge-compensated to 150−K, Dm_K ×{0.25..4}, Dm_Na ×{0.5,1,2}), 8 betse
  subprocesses (OMP=1 each), truths cached to out/sim_truths.jsonl, sim/init
  pickles deleted after readout (regenerable ~13s each from kept yaml).
  RESULT: 150/150 ok in 6.8 min, mean Vmem spans −74.9..−7.2 mV; the Dm_K ×4
  slice is NON-monotone in K (emergent multi-cell behavior — no closed form).
  466 train / 98 val problems (combo-held-out md5 split), gold self-test 1.0;
  kinds reuse bioelectric.reward verbatim ("nernst" numeric + "polarity").
- `betse_grpo.py` — warm-start SFT (300 steps on train gold) then GRPO
  (200 steps, 16×8, 60% numeric, KL 0.05). RESULT (seed 42, bg8_log.jsonl,
  out/betse_results.png): warm-start val 0.938 (numeric 0.898!, polarity
  0.979) → GRPO 0.945 (numeric 0.912). Demo on unseen combos: model −46 vs
  sim −47 mV; −70 vs −72 mV.
- KEY finding vs stage 5b: the 40M model DOES generalize numerically here —
  because the 3-knob landscape is low-dimensional and locally smooth, so
  memorize-plus-interpolate between neighboring trained combos lands inside
  the ±2/±10 mV reward bands. Honest framing: it learned the simulator's
  response surface, not symbolic physics; 5b's 2-free-concentration log-ratio
  space (where interpolation fails) is what defeated it. Scale story intact,
  the loop (language → predict sim outcome → simulator grades) closed
  end-to-end.

### 8c — Baldwin on the real-sim problems (2026-10-24, seed 42)
- `baldwin_betse.py` — Baldwin kept as ANALOGY: plasticity = k NEAREST-
  NEIGHBOR anchors (nearby combos + their sim-measured Vmem = real local
  experience), selection on post-plasticity phenotype, assimilation = anneal
  k 5→0. Shared mixed-k warm-start bw8_ws.pt (val k=0 0.772, numeric 0.594;
  sens gap +0.106 — unlike stage 7, the plasticity channel GENUINELY HELPS
  here, anchors are interpolation hints on a smooth surface).
- v1 RUN (200 steps, logs bw8_log_*_v1_noassim.jsonl): DESIGN SLIP — anneal
  window 40×5 = 200 = total steps, so the Baldwin arm got ZERO training at
  k=0. Kept as an informative control: 200 scaffolded steps improved
  anchor-USE (gap 0.106→0.156) and transferred NOTHING to k=0 (0.772→0.764),
  while direct k=0 gained +0.074. Scaffolded training optimizes the plastic
  phenotype, not the innate one.
- v2 RUN (300 steps, 50 at k=0, `out/baldwin_betse_results.png`):
  direct k0 0.772→**0.878** (numeric 0.594→0.800, sens gap →~0);
  anneal 0.772→0.809 (numeric 0.665; all gain inside the 50-step k=0 phase,
  whose per-step slope +0.089/50 is ~27% steeper than direct's first 50
  +0.070/50 — smoothing is real but cannot amortize 250 scaffolded steps);
  k5 frozen at 0.772 innate, anchor-use keeps improving.
- VERDICT — the honest bracket across stages 7 and 8: Baldwin-anneal BEAT
  direct RL when the scaffold was weak/ignored (stage 7, +0.05) and LOST
  under matched budget when the scaffold genuinely works (stage 8, −0.069),
  because scaffolded steps optimize a phenotype that does not transfer while
  direct RL spends every step on the tested behavior. This is Hinton &
  Nowlan's own cost-of-learning tradeoff, reproduced in an LLM: Baldwin's
  niche is NEEDLE landscapes where direct RL has zero gradient — both our
  tasks gave direct RL partial-credit gradient, so it never needed rescuing.
- SHARPENED falsifiable next experiment (not run): make the reward sparse
  (binary, exact-hit only) on a task the warm start almost never solves —
  direct GRPO then gets all-uniform zero groups (no gradient) while
  anchor-scaffolded sampling still occasionally hits → Baldwin should be the
  ONLY arm that learns. That is the Hinton–Nowlan needle, verbatim.

### 8d — baseline table for the sim exam (2026-10-24, seed 42)
- `eval_baselines.py` → out/betse_baselines.json + .png. Same 98 held-out-combo
  exam, same verifier, greedy. Who scores what (val reward / numeric nernst):
  - constant (train-mode answer per kind): 0.538 / 0.454 — the partial-credit
    bands make the floor non-trivial; always report it.
  - 1-NN knob lookup, NO LLM (nearest train combo in log2 knob space, copy its
    answer): **0.927 / 0.856** — the honest headline. Pure interpolation nearly
    matches the fine-tuned model; polarity it solves perfectly (1.0).
    DECLARED CHEAT: 1-NN never reads the question — it is handed the parsed
    knob values (the jsonl "combo" field, an oracle parser). It is a floor for
    the knobs→Vmem sub-task only, not a rival system: the LLM must do
    reading + mapping + formatting from raw text and still beats it.
  - generic ladder zero-shot: base 0.000, +SFT 0.000, +DPO 0.000 — they can't
    even emit the parseable "Mean Vmem = X mV." format. +GRPO-textbook (5b)
    0.321 / 0.150 — format partially transfers from the textbook bioelectric
    problems, numbers don't.
  - sim warm-start SFT 0.938 / 0.898; +GRPO vs simulator 0.945 / **0.912**.
- Reading: the fine-tune is NECESSARY (generic = 0) but the deep claim must be
  modest — the model beats no-LLM 1-NN by +0.018 val, and on the numeric task
  by +0.056 (0.912 vs 0.856), i.e. it interpolates the smooth response surface
  somewhat better than copying the nearest seen combo. Any paper claim about
  "learning the physics" must be benchmarked against this 1-NN floor, not
  against zero. [SUPERSEDED by 8e: the +0.056 did not survive off-grid.]

### 8e — the OFF-GRID exam kills the interpolation claim (2026-10-24, seed 42)
- User's challenge was right: the random-combo split hid NODES, not REGIONS —
  every test point had an adjacent trained neighbor, so 1-NN 0.927 was a
  property of the EXAM. `make_offgrid_exam.py`: 60 knob combos log-uniform
  INSIDE the grid range but ≥0.25 log2 from every node (median 0.43), real
  BETSE sims (60/60 ok, 2.9 min), 234 test-only problems. Nothing retrained.
- Scores collapse and REORDER (`betse_baselines_offgrid.json/.png`):
  constant 0.663 / 1-NN 0.793 / warm-start 0.536 / +GRPO 0.539 — the 40M
  model falls BELOW both non-LLM floors (grid exam had it at 0.945 > 0.927).
- Diagnosis (`diag_offgrid.py`, offgrid_diagnosis.json): NOT a number-format
  break (60/60 parseable, decimals read fine). The model emits only 17
  distinct values, 16 of them memorized train answers — it SNAPS every new
  question onto its answer vocabulary. MAE 11.86 mV vs 1-NN's 6.82;
  corr(pred, truth) 0.516 ≈ corr(pred, nn-node) 0.469. Verdict: the 40M is a
  soft, noisy lookup table; grid-exam "interpolation" was memorize+snap.
- Hypothesis for the fix (next experiment): training on a LATTICE teaches a
  lookup; training on RANDOM points (same 150-sim budget, min 0.2 log2 from
  the off-grid test points) may force the answer head off the memorized
  vocabulary and teach the actual response function. Lattice-trained vs
  random-trained on the SAME off-grid exam = clean attribution.
- CASE 2 RESULT (`make_random_train.py` seed 43, 150/150 sims ok;
  `betse_grpo.py --train betse_train2_tr.jsonl --suffix r`; one-time
  unsealing via `eval_case2.py`; figure `two_exams.png`): random-trained 40M
  scores 0.777 on the off-grid exam vs lattice-trained 0.539 — diversity of
  the TRAINING generator, not the model, was the binding constraint (+0.238
  from data design alone; GRPO adds ~nothing on top of ws here, 0.772→0.777).
  Diagnosis flipped the right way: pred now closer to TRUTH than to the
  nearest node (MAE 8.82 vs 9.45; corr(truth) 0.591 > corr(nn) 0.497) — but
  the answer head STILL emits only memorized train values (0 novel of 19
  distinct), and explicit 1-NN on the same random book stays ahead (0.856).
  Honest summary for the paper: at 40M, data diversity turns a brittle
  lookup into a generalizing one; beating oracle-parsed lookup off-grid
  remains open.
- CASE 3 — the 40M's 5-shot (`baldwin_betse.py --arm k5 --train
  betse_train2_tr.jsonl --wspt bw8r_ws.pt --tag r`; sealed eval
  `eval_case3_offgrid.py`, case3_offgrid.json): mixed-k SFT warm start on the
  random data, 5 nearest train anchors AS TEXT in the prompt (same language
  starting line as pretrained 5-shot). OFF-GRID: **0.835** @k=5 (ws
  checkpoint) vs 0.777 case-2 zero-shot vs 0.539 case-1 — best
  language-reading system on exam B. Ladder: 0.539 → 0.777 (data design) →
  0.835 (in-context examples). Two honest notes: (a) GRPO@k5 slightly HURT
  off-grid (0.835→0.812) — RL on anchors overfits in-pool patterns, the SFT
  ws generalizes better; (b) mixed-k capacity split costs k=0 performance
  (0.666 vs case-2's 0.777), so anchors are a pure-win only when examples
  are available at test time. Figure: two_exams.png (updated, 1-NN bars
  removed per baseline policy).
- BASELINE POLICY (user decision, 2026-10-24): the paper's main tables
  compare only systems that START FROM THE SAME LINE — read the question as
  language, answer as language. Canonical floors: chance + constant/mode.
  The 1-NN lookup is EXCLUDED from all comparison tables: it holds the
  answer book + oracle-parsed knobs (leakage), and it is one arbitrary
  member of an infinite regressor family (inviting "why not k-NN/GP"
  forever). It survives only as an internal diagnostic (it exposed the
  case-1 soft-lookup). If reviewers demand a regressor, do the whole family
  once in an appendix (k-NN k=1,3,5 + linear on log-knobs).
- gpt-oss:20b fair rerun (think:"low", content channel read): 0-shot 0.455,
  5-shot 0.664 — legit scores, leaderboard unchanged (qwen3.5 5-shot 0.904
  still best pretrained).
- Ollama exam (grid exam A, `eval_ollama.py`, ollama_baselines.json): best
  pretrained = qwen3.5 5-shot 0.904, qwen3.6 5-shot 0.894 (numeric 0.892),
  mistral 5-shot 0.887; best 0-shot qwen3.6 0.529 < constant 0.538. All
  below the 40M's 0.945 on exam A. gpt-oss:20b first scored 0.17/0.13 as a
  HARNESS ARTIFACT (all output in the thinking channel, content empty) —
  scores deleted, rerun with think:"low" + num_predict 1024; never publish a
  model's score without reading its raw transcripts first.

## 10 — Live closed-loop demo (2026-10-24)
- `10_demo/serve_demo.py` (GPU venv) serves http://localhost:8765; BETSE runs
  via `run_one_sim.py` under `.venv-betse` in a subprocess (truth cached per
  combo in 08_betse_rlvr/runs/ — delete a combo's truth.json to force a live
  re-sim, e.g. for recordings). Models: qa = 05 grpo_final.pt, exp = case-3
  bw8r_ws.pt with 5 nearest anchors from betse_train2_tr in context.
- Demo GIF `10_demo/out/cellbio_gpt_demo.gif` (v2, 40 frames, ~1.2 s/frame
  slowed via PIL re-timing): QA beat, then the loop CAUGHT a wrong prediction
  live — (7.5, ×1.5, ×0.8) sits on the Dm_K depolarization cliff (truth
  −7.06 mV vs model −57, ✗) with the tissue map flashing RED, then verified
  (4.8, ×0.55, ×0.9) at 0.1 mV (✓) with the tissue staying blue. The ✗ is a
  feature: no plain chatbot can even be wrong in millivolts.
- After every /verify the demo now RENDERS what the tissue did (user request
  "visual tất cả"): per-cell Vmem map (vmem_map.png), animated map over
  simulated time (vmem_anim.gif, 20 frames from sim.vm_time), and a
  mean±min/max Vmem-vs-time trace (vmem_trace.png) — all saved in the
  combo's runs/ dir and served at /sim_img/<tag>/<file>. The trace exposed a
  genuine transient: tissue depolarizes to a peak at t≈0.02 s then relaxes
  to steady state — great teaching material. BETSE's vm_time gives the
  cell-averaged time series via mem_to_cells accumulation.
- Social drafts (X thread + LinkedIn, EN) in `10_demo/social_drafts.md` —
  DRAFTS ONLY, user posts them, never auto-post.
- Open definitional QA ("what is bioelectricity?") gives fluent but
  vacuous/broken answers on ALL ladder checkpoints (probe: scratchpad
  qa_probe) — consistent with 6e: 40M speaks the PubMed dialect but holds no
  retrievable encyclopedia; it only truly knows the RLVR'd quantitative
  families. UI now frames this honestly (tab renamed "Textbook problems",
  guidance note added); real fix = retrieval (RAG over the 3.2M abstracts),
  roadmap item. DPO shows a repetition pathology on open prompts
  ("transient, transient, reversible, irreversible...").
- LIVE DEPLOY (2026-10-24, user-approved): https://drnamlabs.com/cellbio/ on
  a small server. Bundle `10_demo/server/` (self-contained, relative URLs for
  the /cellbio/ proxy prefix). Facts learned: betse 1.5.0 RUNS on Python
  3.12 (3.13 remains broken); a niced 1-thread sim takes ~66 s on a 2-vCPU
  box; guards = input clamps + one sim at a time + a daily cap on fresh
  sims + a memory limit on the service.

## 11 — RAG: the from-scratch encyclopedia (2026-10-24, seed 42)
- Index: SQLite FTS5 (porter), title+abstract, all 3.2M docs, 8.2GB, ~5 min
  build (`11_rag/build_index.py`). Retrieval `rag_retrieve.py`: AND-then-OR
  with a small synonym map (voltage↔potential, UK↔US spellings),
  definition-first rerank (bonus "<term> is a <noun>" excluding evaluative
  adjectives like "important"; penalty "is also/likely/..."), context window
  starts at the definitional sentence, ≤90 words.
- DefQA: 29,054 train / 946 val self-mined ("<term> is/are ..." sentence =
  gold answer, abstract window = context; 23,960 distinct terms). No teacher
  model — pure from-scratch, per the encyclopedia goal.
- `rag_sft.py` (800 steps from dpo_final_v2) → rag_qa.pt. FAIR eval
  (`eval_rag.py`, bucketed generation): token-F1 0.990 / exact 0.957 /
  faithful-substring 0.962. GOTCHA that cost an hour: the in-training eval
  left-padded prompts with EOT and scored 0.39 — the model has never seen
  padding and there is no attention mask; ALWAYS bucket by length (as
  betse_grpo's generate_batched does).
- Demo lane "Ask the literature" (local wired; sqlite needs
  check_same_thread=False + a lock under ThreadingHTTPServer): retrieve k=2,
  generate per context, keep the first answer shaped like a definition.
  Probes: bioelectricity/membrane voltage/apoptosis/gastrulation answer at
  textbook quality WITH PMID receipts. Known edges: questions not of the
  "what is X" family (out of training distribution), and occasional
  misread stitching from a non-definitional context (gap junctions case).
- MECHANISM CHANGE (user: "cơ chế tạo câu trả lời chưa ổn"): free generation
  at 40M stitches fragments ("Each parameter is electricity is...") when the
  context lacks a definition. The lit lane now SELECTS instead of generating:
  candidate sentences are mined from retrieved abstracts (strict tier:
  "<term> is/are a/an/the"; relaxed tier: any "<term> is/are X" minus weak
  adverbs), the trained reader ranks them by token log-likelihood (same
  trick as PubMedQA option scoring), and the winner is returned VERBATIM
  with its PMID — grammatical by construction, checkable word-for-word,
  declines when no quotable sentence exists. Generation remains only where
  a verifier grades it (the sim loop).
- Retrieval upgrade `retrieve_def()`: FTS5 PHRASE query '"<term> is" OR
  "<term> are"' surfaces definition-bearing abstracts that pure BM25
  frequency ranking misses (gap junctions, resting membrane potential);
  demo merges retrieve_def + retrieve pools, dedup by pmid.
- Battery after the change (all verbatim quotes with sources): apoptosis,
  membrane voltage, gastrulation textbook-grade; "what is electricity"
  returns a real quoted sentence; unknown terms (xenobot — outside corpus)
  decline honestly. Remaining quality edge: quotes are sometimes role-
  statements rather than definitions (gap junctions) — a ranking issue,
  not a safety issue.
- LONG-QUESTION PROBE (user asked "can it handle longer questions?", 6
  multi-clause how/why/compute questions; scratchpad long_probe): NO, on
  every lane. SFT chat writes 40-56 words of PubMed boilerplate that
  contradicts itself ("not a direct consequence ... but rather a direct
  consequence"). Textbook-RL collapsed to 3-word templates ("The cell
  depolarizes.") for ANY question, answered a rephrased Nernst question
  with polarity, and called cancer cells hyperpolarized (wrong). The lit
  lane's relaxed tier quoted verbatim but OFF-TOPIC sentences for how/why
  questions, worse than declining → fixed: a definition-question gate;
  everything else gets an explicit decline. Root causes: 40M + 512 ctx,
  single-turn short SFT data, RL narrowing to trained templates, reader
  trained only on "What is X?". Cheapest from-scratch extension: mine
  CAUSAL QA the same way as DefQA ("X leads to / results in / is required
  for Y" → "What does X lead to?"), with 2-3 consecutive sentences as the
  quoted answer.
- Optional next (not run): abstain-training — mix ~15% negatives (context
  without the term, gold = "The context does not define X.") so the READER
  itself learns to decline; currently declining is the pipeline's job.
- Server rollout plan (ON HOLD until local sign-off — user instruction):
  upload the 1.5GB gz corpus and BUILD the index on the box (do not upload
  the 8.2GB db); +160MB model RAM → raise cellbio.service MemoryMax to
  2.6G. A partial server index build was killed and its 2.3GB fts.db
  removed; the public demo still runs the original 2-lane version,
  untouched.

## 12 — Track B: open pretrained model, same domain (opened 2026-10-24)
- Track A frozen in `MILESTONE_track_A.md`; design in `12_track_b/README.md`.
- Env: peft 0.21.2 installed into .venv-gpu (torch 2.11 untouched). Cached:
  Qwen2.5-1.5B-Instruct. Disk 3TB free. Target 7-8B LoRA bf16 fits ≤30GB.
- B0 (Qwen2.5-1.5B untouched, same verifier + strict format prompt as the
  Ollama runs): exam A 0.427, exam B 0.377 — BELOW the constant floor on
  both. But long answers (150-170 words, structured) and 3-turn dialogue
  with correct "explain more" resolution, which Track A cannot do. Confident
  factual errors without sources (invented Vmem equation; "resting
  potential = AP threshold"). The two tracks fail in opposite directions;
  Track B = borrowed fluency + Track A's discipline (PMID-grounded words,
  simulator-graded numbers).
- B1 (LoRA r=16 on Qwen2.5-1.5B, 500 steps, own-data mixture): exam A
  0.427 → 0.914, exam B 0.377 → 0.839 (Track A 0.945 / 0.835), textbook
  0.976, reading F1 0.990 faithful 1.000. Fluency + dialogue preserved.
  But free-form explanations still confidently wrong and unsourced
  (invented Vmem formula) — short-answer SFT teaches answers, not
  explanations → B3 grounding is the next necessity. Exam-B comparison
  caveat: B1 trained on lattice+random, Track A's best on random only.
- CORRECTION (user): Track B must MIRROR Track A stage by stage from a
  pretrained BASE model — continued pretraining on the same PubMed split
  first, then the same SFT/DPO/GRPO/sim/library data. B1 above = an
  exploratory shortcut (skipped PubMed pretraining), kept but not official.
  Track B never trains on Track A model outputs (B1 didn't either: its
  targets were BETSE truths, Nernst answers, verbatim PubMed sentences).
- b03 encode: 1.07B Qwen tokens (vs 988M with our domain BPE, +8%); uint32;
  val bytes 46,163,519 recorded for bits/byte. CPT of Qwen2.5-0.5B base:
  OOM at micro 8 (152k-vocab logits ≈5GB/tensor) → micro 2 × accum 16 +
  expandable_segments; 16.5k tok/s, 12.8GB, 1 epoch ≈ 18h (started
  2026-10-24). Bits/byte on the temporal val: Track A 40M 0.698 vs untouched
  Qwen 0.5B 0.714 — our 40M domain model compresses future PubMed slightly
  better than a 12x-larger general model before domain pretraining.
- B3 v1 baseline (30 how/why questions, lexical support ≥0.6): untouched
  closed-book backed 12%; untouched + library backed 57% (cites only at
  paragraph ends); B1-shortcut + library collapsed to 1 quoted sentence
  (22 words) — its extractive reading data imprinted a one-sentence habit.
  Retrieval for long questions drifts to tangential abstracts (BM25 latches
  onto one rare term) — fix before scoring Track B's library stage.

## Gotchas learned (search here first when something breaks)
- NCBI efetch 10k cap → date-bisect (see data/build/fetch_corpus.py header)
- NCBI sometimes returns HTTP 200 with TRUNCATED XML → retry fetch+parse together;
  one bad slice must never kill the run (log to data/corpus/failed_slices.txt, continue)
- NCBI rate-limit responses are valid JSON WITHOUT "count" → esearch must retry on
  missing count, and keep request rate at ≥0.55s/call without an API key
- esearch + WebEnv needs usehistory=y
- MeSH tree files gone from old nlmpubs URLs → use E-utilities hierarchy explosion instead
- Windows: machine is shared — cap CPU at 12–20 threads total (default ~12),
  set OMP/MKL/torch explicitly (00_setup/check_env.py)
- GPU is shared too — never take the full 32GB VRAM; budget ≤30GB total:
  `torch.cuda.set_per_process_memory_fraction(30/32)` after device setup in
  every training/inference script (PyTorch's caching allocator otherwise
  grows to fill the card)
