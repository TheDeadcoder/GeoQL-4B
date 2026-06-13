import sys, os, time, random
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "OverpassNL", "dataset", "dataset")
OFFICIAL_EVAL = os.path.join(ROOT, "OverpassNL", "evaluation", "evaluation.py")

sys.path.insert(0, HERE)
from reward import (composite_reward, extract_query, extract_key_values,
                    kvs_proxy, sanity_check, chrf_score)


def fence(q):
    return f"```overpassql\n{q}\n```"


def load(split):
    with open(f"{DATA}/dataset.{split}.query") as f:
        return [l.strip() for l in f]


queries = load("train")
random.seed(0)
sample = random.sample(queries, 300)

print("== T1: gold-as-output scores ~1.0 on 300 real queries ==")
lows = []
t0 = time.time()
for q in sample:
    r = composite_reward(fence(q), q)
    if r["reward"] < 0.999:
        lows.append((r["reward"], r["gate"], q[:90]))
dt = time.time() - t0
print(f"   {len(sample)-len(lows)}/300 == 1.0; {len(lows)} below; {300/dt:.0f} rewards/sec")
for r, g, q in lows[:8]:
    print(f"   low={r:.3f} gate={g} :: {q}")

print("== T2: gate behaviour on real-gold sanity_check (should be ~0 flagged) ==")
flagged = [(sanity_check(q), q[:80]) for q in sample if sanity_check(q)]
print(f"   flagged {len(flagged)}/300")
for g, q in flagged[:8]:
    print(f"   {g} :: {q}")

print("== T3: KVS extraction spot check ==")
q = '[out:json][timeout:125];{{geocodeArea:"Forl\u00ec-Cesena"}}->.searchArea;(node["shop"="florist"](area.searchArea);)->.florist;(.florist;);out geom;>;out skel qt;out;>;out skel qt;'
kv = extract_key_values(q)
assert kv == {"shop", "florist", ("shop", "florist")}, kv
print("   pass:", kv)
q2 = '(node["amenity"~"pub|bar"][!"name"];);out;'
kv2 = extract_key_values(q2)
assert ("amenity", "pub|bar") in kv2 and "name" in kv2, kv2
print("   pass:", kv2)

print("== T4: reward ordering ==")
gold = 'area[name="London"];(node["amenity"="cafe"](area););out;'
minor = 'area[name="London"];(node["amenity"="cafe"](area););out center;'
wrong_tag = 'area[name="London"];(node["shop"="bakery"](area););out;'
broken = 'area[name="London";(node["amenity"="cafe"](area);out'
garbage = "SELECT * FROM cafes WHERE city = 'London';"
vals = [composite_reward(fence(x), gold)["reward"] for x in [gold, minor, wrong_tag, broken, garbage]]
print("   gold=%.3f minor=%.3f wrong_tag=%.3f broken=%.3f garbage=%.3f" % tuple(vals))
assert vals[0] > vals[1] > vals[2] > vals[4], vals
assert vals[3] <= 0.10 + 1e-9
no_fence = composite_reward(gold, gold)
assert no_fence["reward"] == 0.0 and no_fence["gate"] == "no_block"
two = composite_reward(fence(gold) + "\n" + fence(gold), gold)
assert two["reward"] == 0.0
trail = composite_reward(fence(gold) + "\nThis query finds cafes.", gold)
assert trail["reward"] == 0.0 and trail["gate"] == "trailing_text"
print("   pass: ordering, gates")

print("== T5: tag stuffing is self-penalizing ==")
stuffed = 'area[name="London"];(node["amenity"="cafe"]["cuisine"="coffee"]["shop"="yes"]["tourism"="yes"](area););out;'
k_gold = kvs_proxy(gold, gold)
k_stuffed = kvs_proxy(stuffed, gold)
print(f"   gold kvs={k_gold:.3f}  stuffed kvs={k_stuffed:.3f}")
assert k_stuffed < k_gold

print("== T6: chrF replication vs official module ==")
spec = importlib.util.spec_from_file_location("off_eval", OFFICIAL_EVAL)
off = importlib.util.module_from_spec(spec)
spec.loader.exec_module(off)
diffs = []
for q in sample[:50]:
    mine = chrf_score(q[: len(q) // 2] + "out;", q)
    theirs = off.get_character_fscore(q, q[: len(q) // 2] + "out;")
    diffs.append(abs(mine - theirs))
print(f"   max |delta| over 50 pairs: {max(diffs):.2e}")
assert max(diffs) < 1e-9
em_mine = composite_reward(fence("[out:csv];node[a=b];out;"), "[out:xml][timeout:25] ;node[a=b];out;")
em_off = off.normalize_query("[out:csv];node[a=b];out;") == off.normalize_query("[out:xml][timeout:25] ;node[a=b];out;")
assert (em_mine["em"] == 1.0) == em_off
print("   pass: EM normalization matches official")

print("\nALL TESTS PASSED")
