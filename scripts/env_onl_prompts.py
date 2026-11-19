"""Write the OverpassNL prompts that step 0 samples the SFT model on, with its SFT system prompt: all 1,000 dev
prompts, the first 500 of them and the first 500 GRPO training prompts.

  python scripts/env_onl_prompts.py
"""

import json

dev = [json.loads(line) for line in open("data_out/sft_dev.jsonl")]
train = [json.loads(line) for line in open("data_out/grpo_train.jsonl")]
system = dev[0]["messages"][0]["content"]


def write(path, rows):
    with open(path, "w") as f:
        for task, user in rows:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            f.write(json.dumps({"task": task, "messages": messages}, ensure_ascii=False) + "\n")


write("data_out/onl_dev_prompts.jsonl", [(f"onl-dev-{i:04d}", d["messages"][1]["content"]) for i, d in enumerate(dev)])
write("data_out/onl_dev500_prompts.jsonl",
      [(f"onl-dev-{i:04d}", d["messages"][1]["content"]) for i, d in enumerate(dev[:500])])
write("data_out/onl_train500_prompts.jsonl", [(f"onl-train-{i:04d}", t["nl"]) for i, t in enumerate(train[:500])])
