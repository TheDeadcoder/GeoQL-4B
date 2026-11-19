"""Write the geoql-env benchmark's single-shot prompts as JSON lines {"task", "messages"}, one per task.

It imports geoql-env, so run it with that project's environment:

  ../geoql-env/.venv/bin/python scripts/env_prompts.py data_out/env_prompts.jsonl
"""

import json
import sys

from geoql_env import prompt, tasks

with open(sys.argv[1], "w") as f:
    for task in tasks.load_all():
        f.write(json.dumps({"task": task.id, "messages": prompt.messages(task, "single_shot", 0)},
                           ensure_ascii=False) + "\n")
