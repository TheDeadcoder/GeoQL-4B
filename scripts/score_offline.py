"""Convert-free OQS proxy for fast SFT/GRPO checkpoint selection.

proxy_OQS = mean(chrF_exact, kvs_proxy, struct_proxy) * 100   (no /api/convert needed)
  - chrF and EM are exact replicas of the official metric.
  - kvs_proxy / struct_proxy are validated against official KVS/TreeS by
    validate_reward_proxy.py. Use this to RANK checkpoints; confirm the final
    pick with the official convert-based run_evaluation.py.

  python score_offline.py --preds preds/sft_full_e3_dev.txt --split dev
  python score_offline.py --raw   preds/sft_full_e3_dev.jsonl --split dev   # raw generations
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward import (chrf_score, exact_match, kvs_proxy, struct_proxy,
                    extract_query, FENCE_RE, sanity_check)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "OverpassNL", "dataset", "dataset")
FALLBACK = "node(1);out;"


def load_preds_txt(path):
    with open(path) as f:
        return [l.rstrip("\n") for l in f]


def load_raw_jsonl(path):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r["idx"])
    preds = []
    for r in rows:
        q, err = extract_query(r["output"])
        if q is None and err in ("trailing_text", "multiple_blocks"):
            blocks = FENCE_RE.findall(r["output"])
            q = blocks[0].strip() if blocks else None
        preds.append(" ".join(q.split()) if q else FALLBACK)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", help="official-format one-query-per-line file")
    ap.add_argument("--raw", help="raw generations jsonl (alternative to --preds)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--show_worst", type=int, default=0)
    opts = ap.parse_args()

    refs = [l.strip() for l in open(os.path.join(DATA, f"dataset.{opts.split}.query"))]
    if opts.raw:
        preds = load_raw_jsonl(opts.raw)
    elif opts.preds:
        preds = load_preds_txt(opts.preds)
    else:
        ap.error("pass --preds or --raw")
    assert len(preds) == len(refs), f"{len(preds)} preds vs {len(refs)} refs"

    rows = []
    n_fallback = n_gate = 0
    for p, r in zip(preds, refs):
        if p == FALLBACK:
            n_fallback += 1
        em = exact_match(p, r)
        chrf = chrf_score(p, r)
        kvs = kvs_proxy(p, r)
        struct = struct_proxy(p, r)
        oqs = (chrf + kvs + struct) / 3.0
        if sanity_check(p):
            n_gate += 1
        rows.append((oqs, em, chrf, kvs, struct, p, r))

    n = len(rows)
    agg = lambda i: 100.0 * sum(x[i] for x in rows) / n
    print(f"split={opts.split}  n={n}  fallback/empty={n_fallback}  sanity_flagged={n_gate}")
    print(f"  proxy_OQS={agg(0):.2f}  EM={agg(1):.2f}  chrF={agg(2):.2f}  "
          f"KVS*={agg(3):.2f}  TreeS*={agg(4):.2f}")
    print("  (* = offline proxy; confirm final pick with official run_evaluation.py)")

    if opts.show_worst:
        rows.sort(key=lambda x: x[0])
        print(f"\n  {opts.show_worst} worst by proxy_OQS:")
        for oqs, em, chrf, kvs, struct, p, r in rows[: opts.show_worst]:
            print(f"  [oqs={oqs*100:.0f} kvs={kvs*100:.0f}] pred: {p[:80]}")
            print(f"                      gold: {r[:80]}")


if __name__ == "__main__":
    main()