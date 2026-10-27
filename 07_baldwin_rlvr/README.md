# Stage 7 — Phenotype-first RLVR (Baldwinian GRPO)

Dr. Nam H. Le — drnamlabs.com, bunnyquant.com

**Status: RUN (1 seed).** This stage turns the repo's main negative result
into a testable training-algorithm proposal, drawn from the author's
evolutionary-biology research program (the Adaptation Stack; phenotype-first
evolution; the Baldwin effect). Figures: `out/explain_stack.png` (design),
`out/baldwin_results.png` (results). Verdict: see §5.

## 1. The Adaptation Stack reading of stages 3–6

Biology adapts on three stacked timescales; LLM training maps onto them
cleanly:

| layer | biology | LLM analog | this repo |
|---|---|---|---|
| evolution | population, slowest; inherits instinct in the genome | pretraining (weights) | stage 3 |
| lifetime learning | one individual, fastest; nothing inherited — lives in brain state | in-context learning / test-time adaptation | **absent at 40M** |
| culture | between individuals; inherits transmitted behavior | SFT on demonstrations, distillation, preference data | stages 4, 5a |

GRPO (stage 5b) is the odd one out: trial-and-error *during* an episode whose
credit is written **straight into the weights** — a Lamarckian shortcut
biology does not have. Nothing in the standard pipeline trains the middle
layer.

## 2. Why this re-reads the stage-6 diagnosis

The eval-harness root-cause analysis (`06_eval_release/out/pubmedqa_diagnosis.png`)
showed: deleting the abstract from PubMedQA prompts moves accuracy ~2 points —
the model answers from its prior, i.e. **from instinct**. PubMedQA is a
within-lifetime task (read *this* evidence, weigh it, answer); we built an
instinct-only creature, so it cannot take that exam at any prior-calibration.

The GRPO ln() failure is the same phenomenon from the other side:
direct-to-genome learning on a numeric task can only memorize combos
(train reward 0.87, held-out 0.60). The reward landscape for "compute
26.73/z·ln(out/in)" is a needle: full credit at the exact circuit, nothing
on the way there.

## 3. The Baldwin effect, and the algorithm it suggests

Hinton & Nowlan (1987), after Baldwin (1896): genotypes that can **learn the
last steps within a lifetime** turn a fitness needle into a slope that
selection can climb; **genetic assimilation** then moves the solution from
plastic to innate — no Lamarckian copying required. This is the mechanism
behind the author's phenotype-first line of work (behavior appears first via
plasticity; heredity follows).

**Phenotype-first RLVR (Baldwinian GRPO):**

1. **Plasticity channel** — each training prompt carries `k` worked examples
   of *neighboring* (ion, concentration) combos with gold answers: an
   in-context scaffold the model can interpolate from. This is the lifetime-
   learning layer, installed by construction.
2. **Selection on the post-learning phenotype** — K samples, verifier reward,
   GRPO group advantage, exactly as stage 5b. The genome is now rewarded for
   *being able to use the scaffold*, not for innately knowing the answer.
3. **Assimilation schedule** — anneal `k: 5 → 0` over training. The scaffold
   is slowly removed, pressuring the circuit from "interpolate from the
   examples in front of you" toward "know it innately".

Cultural rung (same machinery, optional stage 7b): verified-correct traces
become SFT data for the next generation — distillation = culture.

## 4. Falsifiable predictions (seed 42)

- **P1 (assimilation):** final `k=0` held-out nernst reward beats direct
  GRPO's 0.60 ceiling (stage 5b's result = the `k≡0` control, already run).
- **P2 (the model starts reading):** context-sensitivity — the gap between
  with-scaffold and without-scaffold accuracy early in training, and the
  recovery of that gap after anneal — rises above the ~0 measured in stage 6.
- **Controls:** `k≡0` fixed (done, 0.60); `k≡5` fixed (crutch never removed —
  Baldwin predicts high scaffolded reward but no assimilation, so a collapse
  when tested at k=0).
- **Cost:** minutes per arm on the RTX 5090; all artifacts saved per repo rules.

Honest scope note: at 40M this tests the *mechanism* (does Baldwinian
scaffolding + annealing beat direct RL on a verifiable task?), not SOTA. A
positive result here is an argument for the training-algorithm idea, to be
retried at 1–3B on an open model.

## 5. Results (2026-10-23, seed 42)

**Pre-step forced by a smoke run:** with the stage-5b warm-start, worked
examples in context HURT (sens gap −0.46 on the 160-problem val subset) —
a single-format model treats few-shot prompts as noise. Hinton & Nowlan's
genotypes could learn; ours had to be taught how. `warmstart2.py` re-runs the
warm-start with mixed `k ~ U{0..5}` scaffold formatting → `ws2_final.pt`
(scaffold now helps: gap +0.042; k=0 val drops to 0.841 / nernst 0.321, the
price of format diversity — all three arms start from this same checkpoint).

| arm (300 GRPO steps from ws2) | val k=0 | **nernst k=0** | final KL |
|---|---|---|---|
| `k0` fixed — direct GRPO control | 0.841 → 0.867 | 0.321 → 0.514 | +0.25 |
| `anneal` 5→0 — **Baldwin arm** | 0.841 → 0.865 | 0.321 → **0.564** | +0.15 |
| `k5` fixed — crutch never removed | 0.841 → 0.866 | 0.321 → 0.517 | +0.14 |

- **P1 — directionally confirmed, weak evidence.** On the numeric task the
  Baldwin arm ends **+0.05 above both controls** at k=0, with the predicted
  trajectory: it lags the direct arm while the scaffold is in place
  (steps 0–200), then overtakes exactly during the k=1→0 assimilation phase.
  Caveats stated plainly: one seed, 42 nernst val problems (+0.05 ≈ 2
  problems of credit), and the absolute score stays below stage-5b's 0.60
  (which started from a nernst-0.565 warm start — not comparable).
- **P2 — not confirmed.** Context sensitivity never rose above ~±0.02 after
  training; the gain traveled through the weights, not through better
  in-context reading. At 40M the plasticity channel stays shallow.
- **k5-collapse prediction — refuted.** The crutch arm matched the direct
  control at k=0 (0.517 vs 0.514): scaffolded training still transfers at
  this scale, likely because the model mostly ignores the scaffold (small
  sens gap throughout).

Honest summary: one seed of weak-but-clean evidence that annealed scaffolding
(genetic assimilation pressure) beats direct RL on the task both fail at —
and two of three predictions did not survive contact with the experiment,
which is recorded here with equal prominence. To defend P1: multi-seed
(≥5) + a bigger nernst val set, then 1–3B.
