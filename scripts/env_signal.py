"""Step 0 of the environment-reward test: where the SFT model starts on the geoql-env benchmark, and what each
reward would pay for among its own samples.

Inputs: responses sampled at temperature 1.0 (env_generate.py), their verdicts from `geoql-env score --json`, the
greedy run's verdicts (optional) and the benchmark's task files. The old GRPO reward is the one grpo_modal.py
trained with, scored against each task's first reading, as OverpassNL scores against its single reference.

  python scripts/env_signal.py --predictions preds/env_step0/sft-t1.jsonl --run preds/env_step0/sft-t1.json \
      --greedy preds/env_step0/sft-greedy.json --tasks ../geoql-env/tasks
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import yaml

from grpo_modal import FENCE_RE, REWARD_FUNCS, REWARD_WEIGHTS

STYLES = {"ends out;>;out skel qt;": "outskelqt", "[out:json][timeout:25]": "[out:json][timeout:25]",
          "{{geocodeArea:...}}": "{{geocodeArea:", "nwr": "nwr", "out count": "outcount"}


def old_reward(responses, reference):
    parts = [f(responses, ref_query=[reference] * len(responses)) for f in REWARD_FUNCS]
    return [sum(w * p[i] for w, p in zip(REWARD_WEIGHTS, parts)) for i in range(len(responses))]


def pass_at(n, c, k):
    return 1.0 - comb(n - c, k) / comb(n, k)


def auc(scores, right):
    """The chance that a right sample of the task outscores a wrong one, ties counting half."""
    good = [s for s, ok in zip(scores, right) if ok]
    bad = [s for s, ok in zip(scores, right) if not ok]
    return statistics.fmean((g > b) + 0.5 * (g == b) for g in good for b in bad)


def credit_on_wrong(scores, right):
    """The share of a GRPO group's positive advantage that goes to wrong samples, None if no sample has any."""
    mean = statistics.fmean(scores)
    lifts = [(s - mean, ok) for s, ok in zip(scores, right) if s > mean]
    total = sum(a for a, _ in lifts)
    return sum(a for a, ok in lifts if not ok) / total if total else None


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def query(text):
    blocks = FENCE_RE.findall(text)
    return "".join((blocks[0] if blocks else text).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--run", required=True, help="`geoql-env score --json` of the predictions")
    ap.add_argument("--greedy", help="`geoql-env score --json` of the greedy run")
    ap.add_argument("--tasks", required=True, help="geoql-env's tasks/ directory")
    opts = ap.parse_args()

    meta = {t["id"]: t for t in (yaml.safe_load(p.read_text()) for p in Path(opts.tasks).rglob("*.yaml"))}
    lines = Path(opts.predictions).read_text().splitlines()
    texts = {(p["task"], p["epoch"]): p for p in (json.loads(line) for line in lines if line.strip())}
    groups = defaultdict(list)
    for r in sorted(json.loads(Path(opts.run).read_text())["records"], key=lambda r: (r["task"], r["epoch"])):
        p = texts[r["task"], r["epoch"]]
        groups[r["task"]].append({"right": r["category"] == "pass", "real": r["real"], "category": r["category"],
                                  "response": p["response"], "finish": p.get("finish")})
    reference = {t: meta[t]["interpretations"][0]["program"] for t in groups}
    for t, samples in groups.items():
        for s, value in zip(samples, old_reward([s["response"] for s in samples], reference[t])):
            s["old"] = value
    greedy = {}
    if opts.greedy:
        greedy = {r["task"]: r["category"] == "pass" for r in json.loads(Path(opts.greedy).read_text())["records"]}

    n = len(next(iter(groups.values())))
    test = sorted(t for t in groups if meta[t]["split"] == "test")
    print(f"== starting point, {n} samples per task at temperature 1.0")
    for name, ids in (("test", test), ("all", sorted(groups))):
        right = [sum(s["right"] for s in groups[t]) for t in ids]
        real = [sum(s["real"] for s in groups[t]) for t in ids]
        ks = "  ".join(f"pass@{k} {100 * mean(pass_at(n, c, k) for c in right):5.1f}"
                       for k in (1, 2, 4, 8, 16, 32) if k <= n)
        print(f"  {name:4} {len(ids):2} tasks  {ks}")
        greedy_part = f"  greedy {100 * mean(greedy[t] for t in ids):5.1f}" if greedy else ""
        print(f"       pass@1 on the real world only {100 * mean(c / n for c in real):5.1f}{greedy_part}")
        print(f"       tasks with right and wrong samples {sum(0 < c < n for c in right)}, all wrong "
              f"{right.count(0)}, all right {right.count(n)}")

    samples = [s for t in test for s in groups[t]]
    outcomes = Counter(s["category"] for s in samples).most_common()
    print("\n== outcomes of the test samples")
    print("  " + ", ".join(f"{c} {100 * k / len(samples):.1f}%" for c, k in outcomes))
    print(f"  cut off at the token limit: {sum(s['finish'] == 'length' for s in samples)} of {len(samples)}")

    print("\n== what each reward pays among the test samples (the environment reward pays right samples only)")
    paid = [s for s in samples if s["real"]]
    near = [s for s in paid if not s["right"]]
    near_tasks = sum(any(s["real"] and not s["right"] for s in groups[t]) for t in test)
    print(f"  real-world-only reward: {len(near)} of the {len(paid)} samples it pays are wrong on the witness worlds "
          f"({100 * len(near) / max(len(paid), 1):.1f}%), in {near_tasks} tasks")
    mixed = [t for t in test if 0 < sum(s["right"] for s in groups[t]) < n]
    for label, key in (("real-world-only reward", "real"), ("old GRPO reward", "old")):
        pairs = [([float(s[key]) for s in groups[t]], [s["right"] for s in groups[t]]) for t in mixed]
        if not pairs:
            print(f"  {label}: no test task has both right and wrong samples")
            continue
        credit = [x for x in (credit_on_wrong(*p) for p in pairs) if x is not None]
        print(f"  {label}: ranks a right sample above a wrong one {100 * mean(auc(*p) for p in pairs):.1f}% of the "
              f"time (ties half, {len(pairs)} tasks with both); on average {100 * mean(credit):.1f}% of a task's "
              f"positive advantage goes to wrong samples ({len(credit)} tasks with any)")
    has_right = [t for t in test if any(s["right"] for s in groups[t])]
    best_wrong = 0
    for t in has_right:
        top = max(s["old"] for s in groups[t])
        best_wrong += not any(s["right"] for s in groups[t] if s["old"] == top)
    print(f"  old GRPO reward: mean {mean(s['old'] for s in samples if s['right']):.3f} on right samples, "
          f"{mean(s['old'] for s in samples if not s['right']):.3f} on wrong ones; its best-scored samples are all "
          f"wrong in {best_wrong} of the {len(has_right)} tasks that have a right one")

    print("\n== style of the test samples and of the references (first reading)")
    found, refs = [query(s["response"]) for s in samples], ["".join(reference[t].split()) for t in test]
    for label, needle in STYLES.items():
        print(f"  {label:24} samples {100 * mean(needle in q for q in found):5.1f}%  "
              f"references {100 * mean(needle in q for q in refs):5.1f}%")

    print("\n== pass@1 by family (test)")
    by = defaultdict(list)
    for t in test:
        by[meta[t]["family"]].append(mean(s["right"] for s in groups[t]))
    for family, values in sorted(by.items(), key=lambda kv: -mean(kv[1])):
        print(f"  {family:18} {len(values):2} tasks  {100 * mean(values):5.1f}")


if __name__ == "__main__":
    main()
