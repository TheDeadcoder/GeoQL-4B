"""Write geoql-env's single-shot prompts as JSON lines {"task", "messages"}, one per task.

Tasks are task files or folders of them, or `benchmark` (every task of the benchmark, the default) or
`benchmark:<split>`. It imports geoql-env, so run it with that project's environment:

  ../geoql-env/.venv/bin/python scripts/env_prompts.py data_out/env_prompts.jsonl
  python scripts/env_prompts.py train-prompts.jsonl geoql-env/data/train-tasks/mt-malta
"""

import json
import sys
from pathlib import Path

from geoql_env import prompt, tasks


def load(refs):
    chosen = []
    for ref in refs or ["benchmark"]:
        if ref.split(":")[0] == "benchmark":
            split = ref.partition(":")[2]
            chosen += [t for t in tasks.load_all() if split in ("", t.split)]
        else:
            path = Path(ref)
            chosen += [tasks.load(str(p)) for p in (sorted(path.rglob("*.yaml")) if path.is_dir() else [path])]
    return chosen


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        for task in load(sys.argv[2:]):
            f.write(json.dumps({"task": task.id, "messages": prompt.messages(task, "single_shot", 0)},
                               ensure_ascii=False) + "\n")
