"""Build SFT and GRPO data files from the OverpassNL repo.

Outputs (to --out_dir):
  sft_train.jsonl   chat-format training data (forward + optional comments/reverse tasks)
  sft_dev.jsonl     forward-only dev data (loss tracking; selection uses official metrics)
  grpo_train.jsonl  {nl, ref_query} prompt pool for GRPO reward lookup
  stats.json
"""

import argparse
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SYSTEM_PROMPT = (
    "You translate natural language requests into OverpassQL queries for OpenStreetMap.\n"
    "Rules: reply with exactly one fenced code block labeled overpassql containing only "
    "the query, with nothing before or after it. Preserve overpass-turbo shortcuts: use "
    "{{bbox}} when the request refers to the current view, map or selected window, "
    '{{geocodeArea:"Name"}} for named places searched as areas, and {{date:...}} for '
    "relative dates. Follow OSM tagging conventions precisely.\n"
    "If given an OverpassQL query and asked to explain it, reply with one short plain-English sentence instead."
)

EXPLAIN_TEMPLATES = [
    "Explain what this OverpassQL query does:\n\n```overpassql\n{q}\n```",
    "Describe in one sentence what the following Overpass query returns.\n\n```overpassql\n{q}\n```",
]


def read_split(data_dir, split):
    with open(os.path.join(data_dir, f"dataset.{split}.nl")) as f:
        nls = [l.strip() for l in f]
    with open(os.path.join(data_dir, f"dataset.{split}.query")) as f:
        queries = [l.strip() for l in f]
    assert len(nls) == len(queries)
    return nls, queries


def get_comment_queries(path, comment_max_count=10, comment_min_content_ratio=0.80):
    comment_count = Counter()
    instances = []
    with open(path) as f:
        for line in f:
            inst = json.loads(line)
            for c in inst["comments"]:
                comment_count[c] += 1
            if len(inst["query"]) > 800:
                continue
            instances.append(inst)
    texts, labels = [], []
    for inst in instances:
        keep = []
        for c, ratio in zip(inst["comments"], inst["comment_content_ratio"]):
            if ratio < comment_min_content_ratio or comment_count[c] > comment_max_count:
                continue
            c = c.strip()
            if c and c[-1] != ".":
                c += "."
            keep.append(c)
        if keep:
            text = " ".join(keep)
            if len(text) < 800:
                texts.append(text)
                labels.append(inst["query"])
    return texts, labels


def forward_example(nl, query, task):
    return {
        "task": task,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": nl},
            {"role": "assistant", "content": f"```overpassql\n{query}\n```"},
        ],
    }


def reverse_example(nl, query, rng):
    tmpl = rng.choice(EXPLAIN_TEMPLATES)
    return {
        "task": "reverse",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": tmpl.format(q=query)},
            {"role": "assistant", "content": nl},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "OverpassNL", "dataset", "dataset"))
    ap.add_argument("--comments_file", default=os.path.join(ROOT, "OverpassNL", "dataset", "comments.jsonl"))
    ap.add_argument("--out_dir", default="data_out")
    ap.add_argument("--use_comments", action="store_true")
    ap.add_argument("--comments_max", type=int, default=4000)
    ap.add_argument("--reverse_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    opts = ap.parse_args()
    rng = random.Random(opts.seed)
    os.makedirs(opts.out_dir, exist_ok=True)
    stats = {}

    train_nl, train_q = read_split(opts.data_dir, "train")
    dev_nl, dev_q = read_split(opts.data_dir, "dev")
    test_nl, test_q = read_split(opts.data_dir, "test")

    seen, pairs, dups = set(), [], 0
    for nl, q in zip(train_nl, train_q):
        key = (nl, q)
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        pairs.append((nl, q))
    stats["train_pairs"] = len(pairs)
    stats["train_exact_dups_removed"] = dups

    dev_test_pairs = set(zip(dev_nl, dev_q)) | set(zip(test_nl, test_q))
    leaks = sum(1 for p in pairs if p in dev_test_pairs)
    dev_test_queries = set(dev_q) | set(test_q)
    query_leaks = sum(1 for _, q in pairs if q in dev_test_queries)
    stats["pair_leakage_into_devtest"] = leaks
    stats["query_only_overlap_with_devtest"] = query_leaks

    examples = [forward_example(nl, q, "forward") for nl, q in pairs]

    if opts.use_comments:
        c_texts, c_labels = get_comment_queries(opts.comments_file)
        c_all = list(zip(c_texts, c_labels))
        c_all = [(t, q) for t, q in c_all if q not in dev_test_queries]
        rng.shuffle(c_all)
        c_all = c_all[: opts.comments_max]
        examples += [forward_example(t, q, "comments") for t, q in c_all]
        stats["comments_examples"] = len(c_all)

    n_rev = int(len(pairs) * opts.reverse_frac)
    rev_pairs = rng.sample(pairs, n_rev)
    examples += [reverse_example(nl, q, rng) for nl, q in rev_pairs]
    stats["reverse_examples"] = n_rev

    rng.shuffle(examples)
    with open(os.path.join(opts.out_dir, "sft_train.jsonl"), "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(os.path.join(opts.out_dir, "sft_dev.jsonl"), "w") as f:
        for nl, q in zip(dev_nl, dev_q):
            f.write(json.dumps(forward_example(nl, q, "forward"), ensure_ascii=False) + "\n")

    with open(os.path.join(opts.out_dir, "grpo_train.jsonl"), "w") as f:
        for nl, q in pairs:
            f.write(json.dumps({"nl": nl, "ref_query": q}, ensure_ascii=False) + "\n")

    lens = sorted(len(q) for _, q in pairs)
    nl_lens = sorted(len(n) for n, _ in pairs)
    pct = lambda a, p: a[int(len(a) * p)]
    stats["query_chars"] = {"p50": pct(lens, 0.5), "p90": pct(lens, 0.9),
                            "p99": pct(lens, 0.99), "max": lens[-1]}
    stats["nl_chars"] = {"p50": pct(nl_lens, 0.5), "p90": pct(nl_lens, 0.9),
                         "p99": pct(nl_lens, 0.99), "max": nl_lens[-1]}
    stats["est_completion_tokens_p99"] = int(pct(lens, 0.99) / 3.0) + 12
    stats["total_sft_examples"] = len(examples)

    with open(os.path.join(opts.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
