"""Sample responses to chat prompts with vLLM and write them as predictions for `geoql-env score`.

Prompts are JSON lines {"task", "messages"} from env_prompts.py. On a Colab A100:

  pip install vllm==0.11.0
  python env_generate.py --prompts env_prompts.jsonl --model Melikshah/GeoQL-4B-SFT --samples 16 --out sft-t1.jsonl
  python env_generate.py --prompts env_prompts.jsonl --model Melikshah/GeoQL-4B-SFT --greedy --out sft-greedy.jsonl

Sampling uses only the parameters given here, not the model's generation config. A response cut off at the token
limit is graded on its text like any other, so it is marked "finish": "length", not "truncated", which
`geoql-env score` fails outright.
"""

import argparse
import json

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="", help="a LoRA adapter directory to apply to the model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true", help="one response at temperature 0")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    opts = ap.parse_args()

    rows = [json.loads(line) for line in open(opts.prompts) if line.strip()]
    lora_args = {"enable_lora": True, "max_lora_rank": 64} if opts.adapter else {}
    llm = LLM(model=opts.model, dtype="bfloat16", max_model_len=4096, seed=opts.seed, **lora_args)
    params = SamplingParams(n=1 if opts.greedy else opts.samples,
                            temperature=0.0 if opts.greedy else opts.temperature,
                            max_tokens=opts.max_tokens, seed=opts.seed)
    lora = LoRARequest("adapter", 1, opts.adapter) if opts.adapter else None
    outputs = llm.chat([r["messages"] for r in rows], params, lora_request=lora)

    model = opts.model + (f"+{opts.adapter}" if opts.adapter else "")
    with open(opts.out, "w") as f:
        for row, out in zip(rows, outputs):
            for epoch, sample in enumerate(out.outputs, 1):
                f.write(json.dumps({"task": row["task"], "epoch": epoch, "model": model, "mode": "single_shot",
                                    "response": sample.text, "finish": sample.finish_reason},
                                   ensure_ascii=False) + "\n")
    print(f"wrote {sum(len(o.outputs) for o in outputs)} responses for {len(rows)} prompts to {opts.out}")


if __name__ == "__main__":
    main()
