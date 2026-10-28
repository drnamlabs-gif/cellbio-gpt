# Data Plan — Cell & Tissue Biology LLM

Dr. Nam H. Le — drnamlabs.com, bunnyquant.com

Domain: **cellular biology, tissue engineering, morphogenesis, bioelectricity**.
One corpus covers all three "bio" targets; wearable/physiology is a separate
multimodal problem (time-series, PhysioNet) and is deferred — see bottom.

Why this domain and not generic text: the BETSE simulator gives a
**verifiable reward** for stage 5 — a domain-specific RLVR setup no
generic-data toy can have, and no other applicant can copy. Real biological
systems and physical robots are the natural next verifiers once a cheap
simulator stops being enough, but checking against them is expensive, so
they come later in the eval-cost tiering, not first.

## Corpus sizes (verified against PubMed E-utilities, 2026-10-22)

| MeSH query | Articles |
|---|---|
| Morphogenesis | 472,412 |
| Regeneration | 288,342 |
| Cell Physiological Phenomena | 2,868,141 |
| Tissue Engineering | 52,827 |
| Electrophysiological Phenomena | 442,849 |
| Membrane Potentials | 198,051 |
| **Union of first four, with abstract** | **3,049,232** |

At ~250 tokens/abstract → **~0.8–1.0B tokens** of pure in-domain abstracts.
Chinchilla-optimal for a ~40–50M param model; a 100M model takes 2 epochs or
PMC full-text augmentation. Perfect scale for one RTX 5090.

## Validate-one results (2026-10-22, file pubmed26n1200.xml.gz)

30,000 records → 18,853 with MeSH → **1,494 in-domain with abstract (5.0%)**.
Tokens/doc: mean **516**, p50 507, p90 733 — double the initial 250 estimate.
Revised corpus projection: ~3.3M docs × ~516 tok ≈ **1.7B tokens** —
Chinchilla-optimal for ~80M params, comfortable for the 40–100M target.
Samples read as real cell-physiology text (apoptosis pathways, protein
misfolding, nanoparticle therapy) — broad, which is right for pretraining;
the bioelectric core gets upweighted via textbooks/protocols and stage 4–5.

## Sources, in pipeline order

### Stage 3 — pretraining corpus
1. **PubMed via E-utilities history** (`data/build/fetch_corpus.py`):
   one esearch (usehistory=y) pins the ~3.3M-PMID result set server-side,
   then efetch pages through it in 10k batches (~330 calls, ~0.5s apart,
   resumable, ~8GB of XML → jsonl.gz). Chosen over the 50GB baseline
   download because the MeSH-tree file is no longer at its old nlmpubs URL
   and E-utilities applies NLM's own hierarchy explosion for free.
   Gotcha found in validation: history-combined esearch needs `usehistory=y`
   or it returns "Wrong subquery".
2. **PMC Open Access subset** (AWS S3 `pmc-oa-opendata` or FTP) — full text.
   Use `oa_file_list.csv`, keep **CC BY / CC0 only** (clean for model release),
   intersect with the stage-3 PMID set. Adds several B tokens of full text if
   the 100M model needs it.
3. **bioRxiv** TDM bucket (S3, requester-pays) — developmental-biology +
   cell-biology sections. Preprint-fresh coverage of the Levin-field literature.
4. **Open textbooks**: OpenStax Biology 2e, NCBI Bookshelf OA books (CC BY) —
   small but high-quality "spine" text; upsample.
5. **protocols.io** API, CC-BY subset — procedural/intervention language
   ("add X µM blocker, incubate...") that abstracts lack. This register is
   exactly what stages 4–5 need.

### Stage 4 — SFT data (~50–100k pairs)
- Existing BETSE VQA dataset (D:\language_to_living_systems) — text-side QA.
- Synthesized QA from PMC CC-BY full text: frontier model generates
  question/answer grounded in a passage; keep pairs that pass a citation check.
- protocols.io → instruction-following format (goal → numbered steps).

### Stage 5 — RLVR with simulator verifiers (the moat)
- **BETSE verifier**: tasks of the form "given tissue config + intervention,
  predict the Vmem/pattern outcome" or "propose an intervention that produces
  pattern X". Reward = simulated outcome match. Reuse French Flag
  law+commitment tasks as the first curriculum.
- A real biological system or physical robot as a second, expensive verifier
  — apply the eval-cost tiering from generic_discovery (cheap simulator
  verifier first, promote only what survives it).
- DPO variant: chosen/rejected pairs harvested from verifier pass/fail.

### Stage 6 — evals
- **PubMedQA** (1k expert-labeled yes/no/maybe) — standard, instantly legible.
- **LAB-Bench** (FutureHouse) — biology research tasks; the eval the AI-for-science
  labs actually care about.
- SciQ (easy sanity floor).
- Held-out BETSE eval (ours) vs frontier API baselines (GPT/Claude/Gemini).

## Licensing stance
Abstracts: research use OK, standard practice for open models trained on them.
Full text: CC BY/CC0 subset only, so the final model + data recipe can be
released on HF without restriction. Attribute per CC BY in the dataset card.

## Validate-one-first checklist (before any bulk download)
1. Download ONE baseline file (~30MB), parse XML, apply MeSH filter.
2. Report: yield %, tokens/doc distribution, sample 5 abstracts by eye.
3. Only then fetch the full baseline (background job, resumable).
