# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Standalone inference for cellbio-gpt-40m (any of the four checkpoints).

Usage:
  python generate.py --model model_grpo.pt --prompt "What is a gap junction?"
  python generate.py --model model_base.pt --raw --prompt "The membrane potential" \
      --temperature 0.8

Post-trained checkpoints (sft/dpo/grpo) expect the chat template (default);
--raw skips it for the base model's plain continuation mode.
Needs: torch, plus model.py / bpe.py / tokenizer.json from this folder.
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bpe import BPE            # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

PROMPT_FMT = "### Instruction:\n{}\n\n### Response:\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_grpo.pt")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--raw", action="store_true", help="no chat template (base model)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=120)
    args = ap.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPE.load(os.path.join(HERE, "tokenizer.json"))
    eot = tok.vocab_size - 1

    ck = torch.load(os.path.join(HERE, args.model), map_location=device,
                    weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[{args.model}: {ck.get('stage', '?')}]")

    text = args.prompt if args.raw else PROMPT_FMT.format(args.prompt)
    ids = tok.encode(text)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(x, args.max_new, temperature=args.temperature)
    ans = out[0].tolist()[len(ids):]
    if eot in ans:
        ans = ans[:ans.index(eot)]
    print(tok.decode(ans).strip())


if __name__ == "__main__":
    main()
