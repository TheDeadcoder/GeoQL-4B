"""Grade responses to geoql-env tasks in parallel with the environment's grader: on the real world and every
witness world, as `geoql-env score` grades the benchmark, for generated training tasks too.

  python scripts/env_grade.py PREDICTIONS VERDICTS [TASKS ...] [--keep KEEP]

PREDICTIONS are JSON lines {"task", "epoch", "response"} (env_generate.py); VERDICTS gets one line {"task",
"epoch", "category", "real"} each. TASKS are as for env_prompts.py. --keep writes the ids of the tasks with at
least one response right on the real world and not every response right on every world: the tasks where a
policy's samples differ, so that GRPO has a signal. It imports geoql-env, so run it with that project's environment.
"""

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from env_prompts import load


class Grader:
    """Verdicts of responses to tasks, graded in threads and kept, since every rollout is scored twice."""

    def __init__(self, chosen, threads=16):
        from geoql_env.engine import Engine
        self.tasks, self.engine = {t.id: t for t in chosen}, Engine()
        self.pool, self.cache, self.lock = ThreadPoolExecutor(threads), {}, threading.Lock()

    def _grade(self, key):
        from geoql_env import env, grader
        task = self.tasks[key[0]]
        verdict = grader.grade(task, key[1], self.engine, env.gold(task, self.engine))
        return {"category": verdict.category, "real": verdict.real}

    def verdicts(self, task_ids, responses):
        keys = list(zip(task_ids, responses))
        with self.lock:
            todo = sorted({k for k in keys if k not in self.cache})
        for key, verdict in zip(todo, self.pool.map(self._grade, todo)):
            with self.lock:
                self.cache[key] = verdict
        with self.lock:
            found = [self.cache[k] for k in keys]
            if len(self.cache) > 50_000:
                self.cache.clear()
        return found


def keep(records):
    by = defaultdict(list)
    for r in records:
        by[r["task"]].append(r)
    return sorted(t for t, rs in by.items()
                  if any(r["real"] for r in rs) and not all(r["category"] == "pass" for r in rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("verdicts")
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--keep")
    opts = ap.parse_args()
    rows = [json.loads(line) for line in open(opts.predictions) if line.strip()]
    found = Grader(load(opts.tasks)).verdicts([r["task"] for r in rows], [r["response"] for r in rows])
    records = [{"task": r["task"], "epoch": r.get("epoch", 1), **v} for r, v in zip(rows, found)]
    with open(opts.verdicts, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    passed = sum(r["category"] == "pass" for r in records)
    print(f"graded {len(records)} responses: {passed} pass ({100 * passed / max(len(records), 1):.1f}%)")
    if opts.keep:
        kept = keep(records)
        with open(opts.keep, "w") as f:
            f.writelines(t + "\n" for t in kept)
        print(f"kept {len(kept)} of {len({r['task'] for r in records})} tasks with a signal in {opts.keep}")


if __name__ == "__main__":
    main()
