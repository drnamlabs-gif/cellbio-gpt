# Track B: adapting a pretrained model to the domain

The main project builds a 40M-parameter model from random weights. This folder asks the practical question next to it: what changes if you start from a pretrained model and adapt it to the same domain? SmolLM2-135M (Apache 2.0) stands in for the pretrained start, because its size is close to the from-scratch model and the comparison then changes mostly one thing, the starting weights.

Everything else is held fixed. The corpus, the temporal split, the exams and the 97.5k SFT tasks are the same as in the main project, and no stage trains on the from-scratch model's outputs. The later instruction mixtures include SmolTalk, which larger models wrote.

The day-by-day notes are in `LOG.md`. This page keeps only what was learned.

## What happened

| Step | Model | Held-out bits per byte (lower is better) |
|---|---|---|
| Start | SmolLM2-135M, untouched | 0.825 |
| Reference | CellBio-GPT 40M after one epoch | 0.698 |
| Continued pretraining, full parameters | one epoch, 1.06B tokens | **0.640** |
| Continued pretraining, LoRA r=64 on the Instruct model | one epoch | 0.677 |

Bits per byte is used because the two models have different tokenizers and per-token losses do not compare. Full-parameter training matched the from-scratch model after about 131M tokens, an eighth of one epoch, and the curve was nearly flat over the last quarter. That is the advantage of the pretrained start, and it is large.

## The problem that appeared

After continued pretraining the model knows more PubMed and cannot hold a conversation. It continues documents. Asked "What is cancer?" it echoes the question. Two attempts to fix this produced the most useful results of the track.

**Attempt 1, task SFT.** The same 97.5k tasks as the main project. Answers came out 14 to 18 words long. The tasks have a median answer of 15 words and a maximum of 86, so the model learned to stop after one sentence. Format was fixed and content stayed shallow.

**Attempt 2, LoRA on an Instruct model.** Start from SmolLM2-135M-Instruct, freeze it, train LoRA on the PubMed text. LoRA forgot less than full fine-tuning but still forgot. Over an epoch the answers stopped ending by themselves, drifted into paper register ("In this article, we will discuss") and then into the title of an imagined next abstract.

**Attempt 3, a mixed post-training set (recipe v2).** 80k multi-turn conversations from smol-smoltalk, 30k long domain examples built from abstracts, and 20k short tasks. Stopping was fixed (0% to 100% of probe answers). Domain answers stayed short, and the cause turned out to be the data: the system prompt appeared on every domain example and on no general one, so it worked as a mode switch. With it, 12 to 58 words in paper register. Without it, 173 to 196 words, structured, generic and wrong ("bioelectricity is electrical stimulation therapy"). The two skills sat in separate compartments.

**Attempt 4, decouple the cue (recipe v2.1).** Every example drew its system prompt from the same mixture, so the prompt carried no information. The switch disappeared, but now every biology question got the short paper register, with or without a system prompt. The cue had moved from the system prompt to the topic. In the data, biology questions only ever receive abstract-style answers and long chat answers only ever appear on other topics.

## Diagnosis

The model has never seen a biology question answered as a long, conversational explanation, so it cannot learn that combination. Mixing the two data sources does not create it. What real domain chat models use for this step is bridge data: biology questions with long answers, written by people or by a stronger model. We did not have that and have not generated it. The cheapest honest version would be answers written by an open model from retrieved abstracts, with every sentence checked against its source.

## An exploratory shortcut

`b1_domain_sft.py` fine-tunes Qwen2.5-1.5B-Instruct with LoRA on the project's own simulator and reading data, with no PubMed pretraining. It scores 0.914 on exam A and 0.839 on exam B, against 0.427 and 0.377 untouched, and it keeps fluent answers and multi-turn dialogue. It is a shortcut, not part of the matched comparison: it saw both the grid and the random simulator sets, and its free-form explanations still contain confident errors with no sources.

## Files

| File | What it does |
|---|---|
| `b03_encode_corpus.py` | tokenizes the corpus for the base model's tokenizer, same files and split |
| `b03_cpt.py` | continued pretraining, full parameters or LoRA |
| `b04_sft.py` | SFT on the main project's tasks |
| `b04v2_build_data.py`, `b04v2_sft.py` | mixed post-training sets (v2, v2.1) |
| `b0_baseline.py`, `b1_domain_sft.py`, `b3_grounded.py` | baseline, exploratory shortcut, grounded answers |
| `chat_server.py`, `chat.html`, `models.json` | local chat for every checkpoint |
| `explain_bpb.py`, `b03_plot*.py` | figures |

Scripts cap GPU memory at 90% of the card and use seed 42.
