"""Close the GRPO diagnosis with the SFT model's own samples, scored by the reward grpo_modal.py trained with.

1. Its training prompts at temperature 1.0 (what GRPO sampled at step 0) and greedy (where sharpening ends),
   against the reported training-reward curve: start about 0.25 (format -0.25, KVS 0.50, struct 0.62, chrF 0.58),
   end about 0.60 (format -0.03, KVS 0.70, struct 0.83, chrF 0.78).
2. The same on unseen dev prompts.
3. Greedy dev outputs of the merged SFT model against the stored LoRA-path predictions: the churn that a change of
   inference setup alone causes.

  python scripts/env_onl_check.py
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

from grpo_modal import FENCE_RE, REWARD_FUNCS, REWARD_WEIGHTS, extract_query, normalize_query

NAMES = ["format", "kvs", "struct", "chrf", "em"]
PREDS = Path("preds/env_step0")
train_refs = [json.loads(line)["ref_query"] for line in open("data_out/grpo_train.jsonl")]
dev_refs = [line.strip() for line in open("OverpassNL/dataset/dataset/dataset.dev.query")]


def load(name):
    groups = defaultdict(list)
    for line in open(PREDS / name):
        p = json.loads(line)
        groups[p["task"]].append(p)
    return groups


def reference(task):
    _, kind, index = task.split("-")
    return (train_refs if kind == "train" else dev_refs)[int(index)]


def score(name):
    totals, parts = [], defaultdict(list)
    for task, samples in load(name).items():
        texts = [s["response"] for s in samples]
        values = [f(texts, ref_query=[reference(task)] * len(texts)) for f in REWARD_FUNCS]
        for label, v in zip(NAMES, values):
            parts[label] += v
        totals += [sum(w * v[i] for w, v in zip(REWARD_WEIGHTS, values)) for i in range(len(texts))]
    return statistics.fmean(totals), {k: statistics.fmean(v) for k, v in parts.items()}, len(totals)


def query(text):
    q, err = extract_query(text)
    if q is None and err in ("trailing_text", "multiple_blocks"):
        q = FENCE_RE.findall(text)[0].strip()
    return normalize_query(" ".join(q.split())) if q else None


print("old GRPO reward on the SFT model's samples (mean over samples)")
print(f"  {'':34} {'total':>6} " + " ".join(f"{k:>7}" for k in NAMES))
print(f"  {'reported start of training':34} {0.25:6.2f} {-0.25:7.2f} {0.50:7.2f} {0.62:7.2f} {0.58:7.2f} {'-':>7}")
print(f"  {'reported end of training (500)':34} {0.60:6.2f} {-0.03:7.2f} {0.70:7.2f} {0.83:7.2f} {0.78:7.2f} {'-':>7}")
for label, name in (("train prompts, temperature 1.0", "onl-train-t1.jsonl"),
                    ("train prompts, greedy", "onl-train-greedy.jsonl"),
                    ("dev prompts, temperature 1.0", "onl-dev-t1.jsonl"), ("dev prompts, greedy", "onl-dev-greedy.jsonl")):
    total, parts, n = score(name)
    print(f"  {label:34} {total:6.3f} " + " ".join(f"{parts[k]:7.3f}" for k in NAMES) + f"   n={n}")

merged = {int(t.split("-")[2]): query(s[0]["response"]) for t, s in load("onl-dev-greedy.jsonl").items()}
stored = {r["idx"]: query(r["output"]) for r in map(json.loads, open("preds/r64_20260614/r64_20260614_full_e5_dev.jsonl"))}
changed = sum(merged[i] != stored[i] for i in merged)
print(f"\ngreedy dev outputs, merged SFT model vs the stored LoRA-path predictions: {changed} of {len(merged)} differ "
      f"({100 * changed / len(merged):.1f}%; GRPO step 100 differed in 18.1%)")
