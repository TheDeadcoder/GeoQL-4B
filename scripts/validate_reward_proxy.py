"""Correlate offline reward proxies (kvs_proxy, struct_proxy) against the
official XML-based metrics (kv_overlap, xml_overlap) from the OverpassNL repo.

Run on a machine with network access to an Overpass /api/convert endpoint:
  python validate_reward_proxy.py --repo ~/OverpassNL --n 150
Use --converter_url http://localhost:12346/api/convert if a local instance is up.
"""

import argparse
import csv
import importlib.util
import os
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from reward import kvs_proxy, struct_proxy, TAG_FILTER_RE, strip_settings


def load_official(repo):
    path = os.path.join(repo, "models", "evaluation.py")
    spec = importlib.util.spec_from_file_location("off_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def perturb_drop_tag(q, rng):
    ms = list(TAG_FILTER_RE.finditer(strip_settings(q)))
    if not ms:
        return None
    m = rng.choice(ms)
    off = len(q) - len(strip_settings(q))
    return q[: m.start() + off] + q[m.end() + off:]


def perturb_swap_value(q, rng, vocab):
    ms = [m for m in TAG_FILTER_RE.finditer(q) if m.group("qval")]
    if not ms:
        return None
    m = rng.choice(ms)
    new_val = rng.choice(vocab)
    s, e = m.span("qval")
    return q[:s] + new_val + q[e:]


def perturb_shuffle(q, rng):
    parts = [p for p in q.split(";") if p.strip()]
    if len(parts) < 3:
        return None
    rng.shuffle(parts)
    return ";".join(parts) + ";"


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(ROOT, "OverpassNL"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--converter_url", default="https://overpass-api.de/api/convert")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--out_csv", default="proxy_validation.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    opts = ap.parse_args()
    rng = random.Random(opts.seed)

    data_dir = os.path.join(opts.repo, "dataset", "dataset")
    with open(os.path.join(data_dir, "dataset.dev.query")) as f:
        refs = [l.strip() for l in f]
    vocab = sorted({m.group("qval") for q in refs for m in TAG_FILTER_RE.finditer(q) if m.group("qval")})
    sample = rng.sample(refs, opts.n)

    rows = []
    for ref in sample:
        cands = {"gold": ref,
                 "drop_tag": perturb_drop_tag(ref, rng),
                 "swap_value": perturb_swap_value(ref, rng, vocab),
                 "shuffle": perturb_shuffle(ref, rng),
                 "cross": rng.choice(refs)}
        for kind, cand in cands.items():
            if cand is None:
                continue
            rows.append({"kind": kind, "ref": ref, "cand": cand,
                         "proxy_kvs": kvs_proxy(cand, ref),
                         "proxy_struct": struct_proxy(cand, ref)})

    if not opts.dry_run:
        off = load_official(opts.repo)
        import requests as _rq
        _HDRS = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) GeoQL-eval/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        _orig_get = _rq.get
        off.requests.get = lambda url, **kw: _orig_get(
            url, **{**kw, "headers": {**_HDRS, **kw.get("headers", {})}})
        cache_dir = os.path.join(opts.repo, "evaluation", "cache")
        converter = off.OverpassXMLConverter(url=opts.converter_url, cache_dir=cache_dir)
        for i, row in enumerate(rows):
            before = converter.api_calls
            ref_xml = off.get_overpass_xml(row["ref"], converter)
            cand_xml = off.get_overpass_xml(row["cand"], converter)
            row["off_kvs"] = off.get_key_value_overlap_score(ref_xml, cand_xml)
            row["off_trees"] = off.get_xml_overlap_score(ref_xml, cand_xml)
            if converter.api_calls > before:
                time.sleep(opts.sleep)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)} scored ({converter.api_calls} api calls)")
        converter.save_cache()

        kinds = sorted({r["kind"] for r in rows})
        print(f"\n{'subset':<12}{'n':>5}{'kvs_pearson':>14}{'kvs_spear':>12}{'struct_pearson':>16}{'struct_spear':>14}")
        for subset in ["ALL"] + kinds:
            rs = rows if subset == "ALL" else [r for r in rows if r["kind"] == subset]
            pk = [r["proxy_kvs"] for r in rs]
            ok = [r["off_kvs"] for r in rs]
            ps = [r["proxy_struct"] for r in rs]
            ot = [r["off_trees"] for r in rs]
            print(f"{subset:<12}{len(rs):>5}{pearson(pk, ok):>14.3f}{spearman(pk, ok):>12.3f}"
                  f"{pearson(ps, ot):>16.3f}{spearman(ps, ot):>14.3f}")
        kvs_mae = float(np.mean([abs(r["proxy_kvs"] - r["off_kvs"]) for r in rows]))
        print(f"\nKVS mean abs error proxy vs official: {kvs_mae:.4f}")

    with open(opts.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {opts.out_csv}")


if __name__ == "__main__":
    main()