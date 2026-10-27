# Track B development log

## CORRECTED PLAN (2026-10-24): mirror Track A stage by stage

Track B repeats Track A's pipeline exactly, with ONE difference: the
starting weights are a pretrained general LLM (BASE model, not instruct)
instead of random init. Every later stage uses the same data and the same
exams as Track A, and no stage ever trains on Track A model outputs.

| Track A stage | Track B mirror | Status |
|---|---|---|
| 01 tokenizer (own BPE) | the base model's own tokenizer (cannot change) | — |
| 03 pretrain, 1 epoch on 988M tokens | `b03_encode_corpus.py` (same files, same temporal split, 1.07B Qwen tokens) + `b03_cpt.py` continued pretraining, 1 epoch | running |
| 04 SFT (97.5k self-mined tasks) | same `04_sft/out/sft_train.jsonl` | next |
| 05 DPO + textbook GRPO | same pref + rlvr data | after |
| 06 eval harness | same: held-out (as bits/byte), PubMedQA, BETSE | after |
| 08 simulator RLVR | same sim problems, same exams A/B | after |
| 11 library | same index, same reader data | after |

Every finished stage is appended to `models.json` and appears in the local
chat (`chat_server.py`, http://localhost:8766).

The B1 run below (Qwen Instruct + task data, no PubMed pretraining) is kept
as an exploratory shortcut, not as the official Track B.

### Stage 3 numbers so far
- Encoding: 1,070,357,739 train tokens (Track A: 988,531,863 with its own
  domain BPE; the general tokenizer spends ~8% more tokens on this text),
  9,840,004 val tokens, 46.16M val bytes.
- Throughput (Qwen2.5-0.5B base, full-parameter, fp32 master + bf16
  autocast, seq 1024, 32k tokens/step): 16.5k tok/s, 12.8GB peak → one epoch
  ≈ 18 h. 1.5B would be ≈ 3x slower and near the 30GB cap.
- Held-out bits/byte, same temporal val: Track A 40M after its epoch =
  0.698; Qwen2.5-0.5B untouched = 0.714 (200k-token subset). The 12x-larger
  general model starts slightly WORSE on future PubMed text than our 40M
  domain model.
- SWITCH (2026-10-24): official Track B base = SmolLM2-135M
  (HuggingFaceTB, Apache 2.0) — size class close to Track A's 40M, so the
  comparison changes essentially one variable: pretrained start vs random
  start. Larger bases (360M, 1.7B, Qwen 1.5B) are later scaling points.
  Encoding `data/tokens_smol/`: 1,059,079,746 train / 9,759,061 val tokens
  (vocab 49,152, eot id 0), same files and split.
  Run: full-parameter, no activation checkpointing, micro 8 × accum 4 =
  32,768 tok/step (Track A ≈ 32.8k), lr 1e-4 warmup + cosine to 10%, 1 epoch
  = 32,320 steps; 63.8k tok/s, 16.6GB → ≈ 4.6 h. Started 2026-10-24.
  Step 0 (untouched): 0.825 bits/byte on the temporal val (1M tokens) — far
  worse than Track A's 40M (0.698) on future PubMed text.
- Val bits/byte (1M tokens): 1k 0.732 · 3k 0.699 (= Track A) · 11k 0.665 ·
  23k 0.648 · 27k 0.646; curve nearly flat after ~75% of the epoch.
- FINAL (step 32,320, 1.06B tokens): full val 0.6395 bits/byte (Track A
  0.698). Matches Track A after ~131M tokens. Figure
  `out/cpt_SmolLM2-135M/val_bpb.png`.
- Why 1 epoch: same data exposure as Track A (one variable changes: the
  start), curve nearly flat, and more domain epochs risk forgetting general
  language. Optional later ablation: a 2nd epoch (~5 h), measuring bits/byte
  AND dialogue retention.
- Chat test on the LIVE checkpoint (2026-10-24): "What is
  bioelectricity?" → the model echoes the question ("What is cancer?" →
  "adir: what is cancer?"). Expected for a base model after pretraining only:
  it continues documents and has never seen a question→answer format. Track
  A's 40M behaved the same after stage 03; stage 04 SFT taught the format.
  This is the observation the SFT stage must fix.

### Diagnosis after stage 4 (2026-10-24): SFT made answers SHORT
- smol_sft answers 14-18 words vs Qwen-1.5B-Instruct 184-225. Cause: the
  97.5k SFT answers have median 15 words, max 86 — the model learns to emit
  eos after one sentence (max_new_tokens is only a ceiling). CPT on abstracts
  also erased the base's web-style Q/A habit (echoing). The base model never
  had real dialogue: that comes from instruct tuning.
- New route: start from SmolLM2-135M-Instruct
  (untouched 0.874 bits/byte, 195-230-word answers, wrong bioelectricity
  facts), freeze it, LoRA r=64 on all linears (19.5M trainable) continued
  pretraining on the same PubMed tokens. No SFT.
- Pilot 2,000 steps (65.5M tokens): 0.758 full-val bits/byte (full-param
  base at the same tokens ≈ 0.710); answers stay long (290-343 words) and
  conversational, in a PubMed register, but details still hallucinated
  (gap junctions "composed of voltage-gated and calcium-dependent channels").
- Full epoch running: `b03_cpt.py --model HuggingFaceTB/SmolLM2-135M-Instruct
  --lora_r 64 --bf16_base --lr 2e-4 --micro 8 --accum 4 --eval_every 1000`,
  53.6k tok/s, 5.5 h. GPU-bound (89% util, CPU 3%); bf16 frozen base gave
  only +2% (52.7k fp32), micro 16 OOMs beside the chat server (49k-vocab
  logits), no triton on Windows so no torch.compile.
- RESULT full LoRA epoch (2026-10-25): full val **0.677** bits/byte (Track A
  0.698, full-param base 0.640); crosses Track A at ~0.43B tokens. Figure
  `out/cpt_routes.png`. Dialogue ERODES with more PubMed even under LoRA:
  answers no longer end on their own (run to the 420-token cap, untouched
  instruct stops at 195-230 words), slide into paper register ("In this
  article, we will discuss...") and then into the next abstract's title;
  repetition ("the transmembrane proteins" x3). Knowledge up, chat format
  down: LoRA forgets less, not nothing.

### Recipe v2 post-training (2026-10-25) — `b04v2_build_data.py`, `b04v2_sft.py`
- Mixture (data/sft_v2, 128.7k train): 80k smol-smoltalk multi-turn
  (median 187-word answers) + 20k "describe the study titled ..." → abstract
  (30% with a follow-up turn) + 10k findings-from-context + 20k Track A short
  tasks. ChatML template, loss on every assistant turn. Init = full-param CPT
  model (0.644 on the 1M-token val). lr 1e-4, 1 epoch, 3,992 steps, 47 min.
- v2 result: stop rate 0.00 → **1.00**, but median answer 15.5 words and
  "Explain more" repeats the previous answer; PubMed 0.644 → **0.694**
  bits/byte (general chat erodes domain knowledge).
- Cause found: our system prompt was on 100% of domain examples and 0% of
  general ones → a MODE SWITCH. With it: 12-58 words, PubMed register.
  Without it: 173-196 words, bullets, stops — but generic and wrong
  ("bioelectricity = electrical stimulation therapy"). The two skills sit in
  separate compartments, not fused.
- v2.1 (data/sft_v2b, out/sft2b_*): every example draws its system prompt
  from the same 1/3-1/3-1/3 mix (none / ours / generic). Only change. The
  original smoltalk task-specific system prompts are replaced. Next if
  needed: lower lr and/or raw PubMed replay against the 0.05 bits/byte loss.
- v2.1 result: switch gone, but now EVERY bio question gets the short
  paper register, with or without a system prompt (median 32.5 / 21 words,
  stop 1.00; "...is discussed"; "Explain more" repeats; "K+ doubles → Vmem
  doubles"). PubMed 0.695. Reading: the cue is now the TOPIC. In the data,
  bio questions only ever get abstract-style answers (our tasks) and long
  chat answers only ever come on non-bio topics (smoltalk). Nothing shows
  the model a bio question answered as a long, conversational explanation,
  so it cannot learn that combination. That bridge data is what real domain
  chat models get from human writers or a stronger teacher model.

### Stage 4 SFT (2026-10-24) — `b04_sft.py`, out/sft_SmolLM2-135M/
- Same 97,559 train / 1,990 val tasks as Track A, 0 skipped at 1024 tok,
  3,048 steps, lr 1e-5, ~20 min.
- Val answer loss 2.526 (stage-3 model, no SFT) → 2.141 @200 → 2.100 final;
  flat after ~1,000 steps. (Not comparable to Track A's 2.316: different
  tokenizers, per-token.)
- Same sample prompts, final answers, Track A vs Track B:
  - ATP: "ATP stands for ATP." vs "ATP stands for adenosine triphosphate."
  - depolarization: generic loop sentence vs "...activation of the Ca2+ channel."
  - "What is bioelectricity?": echoes the question before SFT; after SFT a
    fluent but wrong-in-scope answer ("a key component of the nervous
    system"). Form fixed, content still shallow: the job of stages 5, 8, 11.

- Run 1 (Qwen2.5-0.5B, full-parameter, 1 epoch planned) STOPPED by the author at step ~100
  of 32,664 (≈3.3M tokens, 15.4k tok/s, ETA was 19 h). No checkpoint was
  saved (first one due at step 1000). Its log and step-0 numbers remain in
  `out/cpt_Qwen2.5-0.5B/`. Open decision: full-parameter with val-based early
  stopping vs LoRA continued pretraining (≈30% faster, learns less and
  forgets less).

Track A built every layer from scratch and hit the limits of a 40M model:
no long answers, no dialogue, no synthesis (see `MILESTONE_track_A.md`).
Track B borrows general language ability from an open instruct model and
fits it to the same domain with the same two sockets: the corpus/library
and the physics verifier. Everything is scored on Track A's exams so the
two tracks are directly comparable.

## Base model

- Smoke tests: Qwen2.5-1.5B-Instruct (already cached).
- Target: a 7-8B open instruct model with LoRA in bf16 and gradient
  checkpointing, inside the 30GB VRAM cap. Native long context and a
  multi-turn chat template come with it.

## Stages

| Stage | What | Verified by |
|---|---|---|
| B0 baseline | base model, untouched, on exam A, exam B, the long-question set and a follow-up dialogue set | same verifier as Track A |
| B1 domain SFT (LoRA) | mixture: forward sim problems, inverse problems, definition and causal QA grounded in retrieved abstracts, plus a general-instruction slice to keep chat ability | held-out exams; general-chat regression check |
| B2 RLVR | GRPO, reward = BETSE outcome, same bands as Track A | exams A and B |
| B3 grounded long answers | retrieve from the 3.2M library, answer in several sentences with inline PMIDs | automatic faithfulness: every sentence must be supported by a retrieved passage |
| B4 dialogue | multi-turn follow-ups ("explain more", "why?", "what if K+ doubles?") | judged against the same retrieved sources and the simulator |
| B5 inverse control | goal in words → proposed intervention → BETSE | hit rate within ±2 / ±5 mV on sealed targets |

## What Track B must show to be worth it

1. Long, multi-sentence answers that stay faithful to cited sources.
2. Follow-up dialogue that keeps context.
3. Exam A/B scores at least as good as Track A, so domain grounding is not
   traded away for fluency.
4. The same live loop: every number still goes to the simulator.

## B0 result: Qwen2.5-1.5B-Instruct, untouched (2026-10-24, seed 42)

`b0_baseline.py` → `out/b0_Qwen2.5-1.5B-Instruct.json`.

| | Exam A | Exam B |
|---|---|---|
| constant floor | 0.538 | 0.663 |
| Qwen2.5-1.5B, zero-shot | 0.427 | 0.377 |
| Track A, CellBio-GPT 40M | 0.945 | 0.835 (5-shot) |

- Below the constant floor on both exams: the numbers are guesses.
- Long answers: 150-170 fluent, well-structured words per question, where
  Track A produced boilerplate or 3-word templates.
- Dialogue: three turns, and "Explain more" correctly resolved to the
  previous topic. Track A cannot do this at all.
- But confident errors with no sources: an invented resting-potential
  equation (−½(g_N V_N + g_K V_K)), "resting potential is the lower
  threshold for an action potential", and a regeneration experiment that
  is about nerve dependence and growth factors rather than bioelectricity.

Reading: Track A is right about the physics and cannot talk; the open
model talks and is wrong about the physics. Track B's job is to keep the
second one's fluency and add the first one's discipline: retrieval with
PMIDs for words, the simulator for numbers.

## B1 result: LoRA domain SFT on Qwen2.5-1.5B (2026-10-24, seed 42)

`b1_domain_sft.py --steps 500` (r=16, 18.5M trainable, 0.5 s/step, ~4 min)
→ `out/b1_Qwen2.5-1.5B-Instruct_{log.jsonl,eval.json,lora/}`, figure
`out/b1_Qwen2.5-1.5B-Instruct.png`. Mixture: 2814 sim (lattice + random
train, ×3), 1500 textbook, 3000 reading — all our own data.

| | B0 untouched | B1 + our data | Track A 40M |
|---|---|---|---|
| Exam A | 0.427 | **0.914** | 0.945 (with GRPO) |
| Exam B | 0.377 | **0.839** | 0.835 (5-shot) |
| Textbook val | — | 0.976 (numeric 0.517) | 0.977 |
| Reading F1 / faithful | — | 0.990 / 1.000 | 0.990 / 0.962 |

- Domain SFT alone moves the open model from below the constant floor to
  Track A's level on both exams. Data caveat: B1 saw BOTH the lattice and
  the random train sets; Track A's best exam-B model saw only the random
  set (plus 5 examples in context).
- Long answers and three-turn dialogue survive the fine-tune: still fluent,
  "explain more" still resolves the topic. No capability regression.
- Free-form explanations are still unsourced and still contain confident
  errors (an invented resting-potential formula "−(K+3)/R × R", "nerve
  signalling uses neurotransmitters while bioelectricity uses ions"). SFT
  on short-answer data teaches the answers, not correct explanations. This
  is exactly what B3 (grounded long answers with PMIDs) has to fix.

## Setup notes

- `peft` must be installed into `.venv-gpu` (not present on 2026-10-24).
- Every torch script keeps `set_per_process_memory_fraction(0.90)`.
- Seed 42; smoke test on 1.5B before any 7-8B run.
