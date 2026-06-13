"""Reward functions for GRPO on Text-to-OverpassQL.

Design: training-time reward uses only string-computable signals.
EM and chrF replicate the official OverpassNL implementation exactly.
KVS and structure are offline proxies of the official XML-based
kv_overlap / xml_overlap (those need the Overpass /api/convert endpoint;
official metrics are used for checkpoint selection and reporting only).
"""

import re
from sacrebleu import CHRF

_sb_chrf = CHRF(CHRF.CHAR_ORDER, CHRF.WORD_ORDER, CHRF.BETA,
                lowercase=False, whitespace=False, eps_smoothing=False)

DEFAULT_WEIGHTS = {"kvs": 0.40, "struct": 0.25, "chrf": 0.20, "em": 0.15}

FENCE_RE = re.compile(r"```(?:overpassql|overpass|ql)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

KNOWN_SHORTCUTS = ("{{bbox}}", "{{center}}", "{{user}}", "{{geocodeArea:", "{{nominatimArea:",
                   "{{geocodeId:", "{{geocodeCoords:", "{{geocodeBbox:", "{{date:")
USERVAR_RE = re.compile(r"\{\{\s*\w+\s*(=[^}]*)?\}\}")


def extract_query(model_output: str):
    """Return (query, error). Exactly one fenced block, no trailing prose."""
    blocks = FENCE_RE.findall(model_output)
    if len(blocks) == 0:
        return None, "no_block"
    if len(blocks) > 1:
        return None, "multiple_blocks"
    tail = model_output[model_output.rfind("```") + 3:].strip()
    if len(tail) > 0:
        return None, "trailing_text"
    query = blocks[0].strip()
    if not query:
        return None, "empty_block"
    return query, None


# --- exact replicas of official functions (models/evaluation.py) ---

def normalize_query(s: str) -> str:
    s = re.sub(r"out:[a-z]+", "out:json", s)
    s = re.sub(r"timeout:\d+", "timeout:300", s)
    s = re.sub(r"\s+", "", s)
    return s


def exact_match(out_query: str, ref_query: str) -> float:
    return float(normalize_query(out_query) == normalize_query(ref_query))


def chrf_score(out_query: str, ref_query: str) -> float:
    return _sb_chrf.corpus_score([out_query], [[ref_query]]).score / 100.0


# --- offline proxies ---

TAG_FILTER_RE = re.compile(
    r"""\[\s*
        (?P<neg>!)?\s*
        (?P<keyre>~)?\s*
        (?:"(?P<qkey>[^"]*)"|'(?P<skey>[^']*)'|(?P<ukey>[A-Za-z_][\w:.\-*]*))
        \s*
        (?:(?P<op>!~|!=|~|=)\s*
           (?:"(?P<qval>[^"]*)"|'(?P<sval>[^']*)'|(?P<uval>[^\],\s]+)))?
        \s*(?:,\s*i\s*)?
        \]""",
    re.VERBOSE,
)


LEADING_SETTINGS_RE = re.compile(r"^\s*(\[[^\]]*\]\s*)+;")


def strip_settings(query: str) -> str:
    return LEADING_SETTINGS_RE.sub("", query, count=1)


def extract_key_values(query: str) -> set:
    """Mirror official get_key_values semantics on QL text:
    set of keys, values, and (key, value) pairs from tag filters.
    Settings ([out:json][timeout:25];) become osm-script attributes in the
    official XML, not k/v attribs, so they are stripped here too."""
    kvs = set()
    query = strip_settings(query)
    for m in TAG_FILTER_RE.finditer(query):
        k = m.group("qkey") or m.group("skey") or m.group("ukey")
        v = m.group("qval") or m.group("sval") or m.group("uval")
        if k:
            kvs.add(k)
        if v:
            kvs.add(v)
        if k and v:
            kvs.add((k, v))
    return kvs


def kvs_proxy(out_query: str, ref_query: str) -> float:
    ref_kv = extract_key_values(ref_query)
    out_kv = extract_key_values(out_query)
    if max(len(ref_kv), len(out_kv)) == 0:
        return 1.0
    overlap = len(ref_kv & out_kv)
    if overlap == 0:
        return 0.0
    return overlap / max(len(ref_kv), len(out_kv))


ELEM_RE = re.compile(r"\b(node|way|relation|rel|nwr|nw|wr|nr|area|derived)\b\s*(?=[\[\(\.;])")
OUT_RE = re.compile(r"\bout\b([^;]*);")
SETTING_RE = re.compile(r"^\s*(\[[^\]]+\]\s*)+;", re.DOTALL)
OP_TOKENS = [
    ("around", re.compile(r"\baround\b")),
    ("poly", re.compile(r"\bpoly\s*:")),
    ("newer", re.compile(r"\bnewer\s*:")),
    ("changed", re.compile(r"\bchanged\s*:")),
    ("user", re.compile(r"\buser\s*:")),
    ("uid", re.compile(r"\buid\s*:")),
    ("id", re.compile(r"\bid\s*:")),
    ("if", re.compile(r"\bif\s*:")),
    ("is_closed", re.compile(r"is_closed\s*\(")),
    ("count", re.compile(r"\bcount\b")),
    ("foreach", re.compile(r"\bforeach\b")),
    ("assign", re.compile(r"->\s*\.")),
    ("inputset", re.compile(r"\._")),
    ("recurse_down", re.compile(r">\s*;")),
    ("recurse_down_rel", re.compile(r">>\s*;")),
    ("recurse_up", re.compile(r"<\s*;")),
    ("union_open", re.compile(r"\(")),
    ("stmt_end", re.compile(r";")),
    ("area_filter", re.compile(r"\(\s*area")),
    ("bbox_setting", re.compile(r"\[\s*bbox\s*:")),
    ("out_setting", re.compile(r"\[\s*out\s*:\s*(\w+)")),
    ("timeout_setting", re.compile(r"\[\s*timeout\s*:")),
    ("date_setting", re.compile(r'\[\s*date\s*:')),
    ("shortcut_bbox", re.compile(r"\{\{bbox\}\}")),
    ("shortcut_geocode", re.compile(r"\{\{geocodeArea:")),
]


def structure_features(query: str) -> dict:
    feats = {}
    for m in ELEM_RE.finditer(query):
        key = "elem_" + m.group(1)
        feats[key] = feats.get(key, 0) + 1
    for m in OUT_RE.finditer(query):
        mods = m.group(1).strip().split()
        key = "out_" + ("_".join(sorted(mods)) if mods else "default")
        feats[key] = feats.get(key, 0) + 1
    for name, pat in OP_TOKENS:
        n = len(pat.findall(query))
        if n:
            feats["op_" + name] = n
    return feats


def struct_proxy(out_query: str, ref_query: str) -> float:
    ref_f = structure_features(ref_query)
    out_f = structure_features(out_query)
    if not ref_f and not out_f:
        return 1.0
    inter = sum(min(ref_f.get(k, 0), out_f.get(k, 0)) for k in set(ref_f) | set(out_f))
    total = sum(ref_f.values()) + sum(out_f.values())
    if total == 0:
        return 1.0
    return 2.0 * inter / total


# --- sanity gate (cheap stand-in for parseability; official converter not in training loop) ---

def _outside_strings(query: str) -> str:
    out, in_str, quote, i = [], False, "", 0
    while i < len(query):
        c = query[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                in_str = False
        else:
            if c in ('"', "'"):
                in_str, quote = True, c
            else:
                out.append(c)
        i += 1
    return "".join(out)


def sanity_check(query: str):
    if len(query) > 800:
        return "too_long"
    bare = _outside_strings(query)
    for ch_open, ch_close in [("[", "]"), ("(", ")"), ("{", "}")]:
        if bare.count(ch_open) != bare.count(ch_close):
            return f"unbalanced_{ch_open}{ch_close}"
    if query.count('"') % 2 != 0:
        return "unbalanced_quotes"
    for m in re.finditer(r"\{\{", query):
        rest = query[m.start():]
        if not (rest.startswith(KNOWN_SHORTCUTS) or USERVAR_RE.match(rest)):
            return "unknown_shortcut"
    if not re.search(r"\b(node|way|relation|rel|nwr|nw|wr|nr|area|derived|out)\b", query):
        return "no_statement"
    return None


def composite_reward(model_output: str, ref_query: str, weights: dict = None) -> dict:
    w = weights or DEFAULT_WEIGHTS
    query, err = extract_query(model_output)
    if query is None:
        return {"reward": 0.0, "gate": err, "query": None,
                "kvs": 0.0, "struct": 0.0, "chrf": 0.0, "em": 0.0}
    components = {
        "kvs": kvs_proxy(query, ref_query),
        "struct": struct_proxy(query, ref_query),
        "chrf": chrf_score(query, ref_query),
        "em": exact_match(query, ref_query),
    }
    reward = sum(w[k] * components[k] for k in w)
    gate = sanity_check(query)
    if gate is not None:
        reward = min(reward, 0.10)
    return {"reward": reward, "gate": gate, "query": query, **components}


def grpo_reward_fn(completions, ref_queries, weights: dict = None):
    """Batch API for TRL GRPOTrainer: completions are raw strings."""
    return [composite_reward(c, r, weights)["reward"]
            for c, r in zip(completions, ref_queries)]
