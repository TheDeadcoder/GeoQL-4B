"""Report of the environment-reward pilot and its decision by the pre-registered rule.

Reads the pilot's outputs (env_pilot.sh): the verdicts of the SFT model and of each arm's checkpoints on the 180
held-out tasks and the 67 benchmark test tasks (greedy G and 8 samples at temperature 1.0, P = their pass rate),
and each arm's training log. Differences are paired over tasks, with 10,000 bootstrap resamples (benchmark tasks
resampled by intent) and 95% intervals.

Decision at step 150. GO: on held-out tasks G(E) - G(SFT) > 0 and P(E) - P(S) > 0 with both intervals above
zero, and on the benchmark both differences at least zero. NO-GO: on held-out tasks G(E) - G(SFT) and
P(E) - P(SFT) both under 2 points, or P(E) - P(S) at most zero. Otherwise inconclusive.

  python scripts/env_pilot_report.py preds/env_pilot ../geoql-env/tasks
"""

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

RESAMPLES, STEPS, FINAL = 10_000, (50, 100, 150), 150


def records(path):
    by = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        by[r["task"]].append(r)
    return by


def metrics(folder, name):
    """Per task: greedy pass (G), sampled pass rate (P), and the same on the real world alone."""
    greedy, sampled = records(folder / f"eval-{name}-greedy-verdicts.jsonl"), records(folder / f"eval-{name}-t1-verdicts.jsonl")
    return {t: {"G": float(greedy[t][0]["category"] == "pass"), "P": statistics.fmean(r["category"] == "pass" for r in rs),
                "Greal": float(greedy[t][0]["real"]), "Preal": statistics.fmean(r["real"] for r in rs)}
            for t, rs in sampled.items()}


def interval(a, b, key, tasks, clusters, seed=0):
    """Mean difference of a minus b over the tasks, with a 95% bootstrap interval resampling clusters."""
    groups = defaultdict(list)
    for t in tasks:
        groups[clusters.get(t, t)].append(a[t][key] - b[t][key])
    keys, rng, means = sorted(groups), random.Random(seed), []
    for _ in range(RESAMPLES):
        picked = [d for k in (rng.choice(keys) for _ in keys) for d in groups[k]]
        means.append(statistics.fmean(picked))
    means.sort()
    total = [d for k in keys for d in groups[k]]
    return 100 * statistics.fmean(total), 100 * means[int(0.025 * RESAMPLES)], 100 * means[int(0.975 * RESAMPLES)]


def main():
    folder, task_dir = Path(sys.argv[1]), Path(sys.argv[2])
    clusters = {}
    for p in task_dir.rglob("*.yaml"):
        t = yaml.safe_load(p.read_text())
        if isinstance(t, dict) and "id" in t:
            clusters[t["id"]] = t.get("intent", t["id"])
    names = ["sft"] + [f"{arm}-{s}" for arm in ("env", "old") for s in STEPS]
    found = {n: metrics(folder, n) for n in names if (folder / f"eval-{n}-t1-verdicts.jsonl").exists()}
    sets = {"held out": sorted(t for t in found["sft"] if t.startswith("gen-gg-")),
            "benchmark": sorted(t for t in found["sft"] if not t.startswith("gen-"))}

    print(f"{'model':10}" + "".join(f"{s + ' ' + k:>18}" for s in sets for k in ("G", "P", "real G", "real P")))
    for n, m in found.items():
        print(f"{n:10}" + "".join(f"{100 * statistics.fmean(m[t][k] for t in ids):18.1f}" for ids in sets.values()
                                  for k in ("G", "P", "Greal", "Preal")))

    final_e, final_s = f"env-{FINAL}", f"old-{FINAL}"
    if final_e not in found or final_s not in found:
        print("\nthe step-150 checkpoints of both arms are not all evaluated yet")
        return
    sft, e, s = found["sft"], found[final_e], found[final_s]
    print(f"\npaired differences at step {FINAL}, points [95% interval]")
    result = {}
    for label, ids in sets.items():
        for name, (a, b, key) in {"G(E)-G(SFT)": (e, sft, "G"), "P(E)-P(SFT)": (e, sft, "P"), "P(E)-P(S)": (e, s, "P"),
                                  "G(E)-G(S)": (e, s, "G"), "G(S)-G(SFT)": (s, sft, "G")}.items():
            result[label, name] = interval(a, b, key, ids, clusters)
            m, lo, hi = result[label, name]
            print(f"  {label:9} {name:12} {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")

    held, bench = (lambda k: result["held out", k]), (lambda k: result["benchmark", k])
    if (held("G(E)-G(SFT)")[1] > 0 and held("P(E)-P(S)")[1] > 0 and bench("G(E)-G(SFT)")[0] >= 0
            and bench("P(E)-P(S)")[0] >= 0):
        verdict = "GO"
    elif (held("G(E)-G(SFT)")[0] < 2 and held("P(E)-P(SFT)")[0] < 2) or held("P(E)-P(S)")[0] <= 0:
        verdict = "NO-GO"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\ndecision by the pre-registered rule: {verdict}")

    for arm in ("env", "old"):
        log = folder / arm / "log_history.json"
        if not log.exists():
            continue
        rows = [r for r in json.loads(log.read_text()) if "rewards/reward_env/mean" in r]
        print(f"\ntraining log, arm {arm} (every 25 steps): environment reward, real-world-only reward, old reward "
              f"total, completion length")
        for k, r in enumerate(rows, 1):
            if (5 * k) % 25 == 0:
                old = sum(w * r.get(f"rewards/reward_{c}/mean", 0)
                          for c, w in zip(("format", "kvs", "struct", "chrf", "em"), (1.0, 0.45, 0.15, 0.20, 0.20)))
                print(f"  step {5 * k:3}: env {r['rewards/reward_env/mean']:.3f}  real-only "
                      f"{r['rewards/reward_real/mean']:.3f}  old {old:.3f}  length {r.get('completions/mean_length', 0):.0f}")


if __name__ == "__main__":
    main()
