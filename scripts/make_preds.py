"""Convert raw generation jsonl (from sft_modal.py generate) into the official
OverpassNL predictions format: one query per line, aligned with the split.

  python make_preds.py --raw preds/sft_full_e3_dev.jsonl --out preds/sft_full_e3_dev.txt
"""

import argparse
import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward import extract_query, sanity_check

FALLBACK = "node(1);out;"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    opts = ap.parse_args()

    rows = [json.loads(l) for l in open(opts.raw)]
    rows.sort(key=lambda r: r["idx"])
    assert [r["idx"] for r in rows] == list(range(len(rows))), "non-contiguous idx"

    stats = Counter()
    preds = []
    for r in rows:
        q, err = extract_query(r["output"])
        if q is None and err in ("trailing_text", "multiple_blocks"):
            from reward import FENCE_RE
            q = FENCE_RE.findall(r["output"])[0].strip()
            stats[f"lenient_{err}"] += 1
        if q is None:
            stats[f"extract_fail_{err}"] += 1
            q = FALLBACK
        else:
            g = sanity_check(q)
            stats["ok" if g is None else f"sanity_{g}"] += 1
            q = " ".join(q.split())
        preds.append(q)

    with open(opts.out, "w") as f:
        f.write("\n".join(preds) + "\n")
    print(json.dumps({"n": len(preds), "stats": dict(stats)}, indent=2))


if __name__ == "__main__":
    main()
