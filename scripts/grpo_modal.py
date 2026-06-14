"""GeoQL-4B GRPO on Modal: RLVR on top of the SFT model with a metric-as-reward.

Loads Melikshah/GeoQL-4B-SFT from HF as base (Modal workspace is disposable),
trains a fresh LoRA with GRPO, dense reward = format gate + weighted
KVS/struct/chrF/EM proxies (validated: KVS Spearman 0.82 vs official). Five
separate reward functions => TRL logs each component, so reward-hacking is visible.

  # 1. GRPO (H100). vLLM rollouts colocated; saves a checkpoint every 100 steps.
  modal run grpo_modal.py --action train --max_steps 500

  # 2. per-checkpoint dev generation (base = the SFT model + GRPO LoRA), then score
  modal run grpo_modal.py --action generate --adapter grpo_r32_lr5e-06_s0/checkpoint-N --name grpo_sN
  #   modal volume get geoql-ckpts preds/grpo_sN_dev.jsonl preds/
  #   python score_offline.py --raw preds/grpo_sN_dev.jsonl --split dev --show_worst 5
  #   then official: make_preds.py -> run_evaluation.py (convert_url localhost:12346)

  # 3. promote the winner (merges GRPO LoRA into the SFT base)
  modal run grpo_modal.py --action merge_push --adapter grpo_r32_lr5e-06_s0/checkpoint-N --repo Melikshah/GeoQL-4B-GRPO

Reward weights lean on KVS (0.45) + exact chrF/EM (0.20 each), struct down to 0.15
(weakest proxy). beta=0 (no KL); if dev OQS diverges from train reward, raise it.
"""

import json
import os
import random
import re

import modal

# ===================== validated reward primitives (inlined from reward.py) =====================
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


_SB_CHRF = None
def chrf_score(out_query: str, ref_query: str) -> float:
    global _SB_CHRF
    if _SB_CHRF is None:
        from sacrebleu import CHRF
        _SB_CHRF = CHRF(6, 0, 2, lowercase=False, whitespace=False, eps_smoothing=False)
    return _SB_CHRF.corpus_score([out_query], [[ref_query]]).score / 100.0


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


# ===================== GRPO reward functions (one per component -> per-component logging) =========
GRPO_WEIGHTS = {"kvs": 0.45, "struct": 0.15, "chrf": 0.20, "em": 0.20}


def _texts(completions):
    out = []
    for c in completions:
        out.append(c[0]["content"] if isinstance(c, list) else c)
    return out


def _component_query(text):
    """Lenient: first fenced block even with trailing text / extra blocks (matches
    make_preds eval extraction). None only if there is no block at all."""
    q, err = extract_query(text)
    if q is not None:
        return q
    if err in ("trailing_text", "multiple_blocks"):
        blocks = FENCE_RE.findall(text)
        return blocks[0].strip() if blocks else None
    return None


def reward_format(completions, **kwargs):
    """Cleanliness shaping (separate from correctness): 0 for one clean valid block,
    -0.25 for a block that is malformed or has trailing/extra content, -1 for none."""
    scores = []
    for t in _texts(completions):
        q, err = extract_query(t)
        if q is not None:
            scores.append(-0.25 if sanity_check(q) is not None else 0.0)
        elif err in ("trailing_text", "multiple_blocks"):
            scores.append(-0.25)
        else:
            scores.append(-1.0)
    return scores


def reward_kvs(completions, ref_query=None, **kwargs):
    out = []
    for t, r in zip(_texts(completions), ref_query):
        q = _component_query(t)
        out.append(kvs_proxy(q, r) if q else 0.0)
    return out


def reward_struct(completions, ref_query=None, **kwargs):
    out = []
    for t, r in zip(_texts(completions), ref_query):
        q = _component_query(t)
        out.append(struct_proxy(q, r) if q else 0.0)
    return out


def reward_chrf(completions, ref_query=None, **kwargs):
    out = []
    for t, r in zip(_texts(completions), ref_query):
        q = _component_query(t)
        out.append(chrf_score(q, r) if q else 0.0)
    return out


def reward_em(completions, ref_query=None, **kwargs):
    out = []
    for t, r in zip(_texts(completions), ref_query):
        q = _component_query(t)
        out.append(exact_match(q, r) if q else 0.0)
    return out


REWARD_FUNCS = [reward_format, reward_kvs, reward_struct, reward_chrf, reward_em]
REWARD_WEIGHTS = [1.0, GRPO_WEIGHTS["kvs"], GRPO_WEIGHTS["struct"],
                  GRPO_WEIGHTS["chrf"], GRPO_WEIGHTS["em"]]

# ===================== Modal app =====================
SFT_MODEL = "Melikshah/GeoQL-4B-SFT"

PIP_GRPO = ["unsloth", "vllm", "trl", "datasets", "wandb", "huggingface_hub",
            "hf_transfer", "sacrebleu"]
PIP_INFER = ["vllm", "transformers", "huggingface_hub", "hf_transfer", "sacrebleu"]
ENV = {"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
       "VLLM_USE_DEEP_GEMM": "0"}

app = modal.App("geoql-grpo")
vol_data = modal.Volume.from_name("geoql-data", create_if_missing=True)
vol_ckpt = modal.Volume.from_name("geoql-ckpts", create_if_missing=True)
vol_hf = modal.Volume.from_name("geoql-hf", create_if_missing=True)
VOLUMES = {"/data": vol_data, "/ckpts": vol_ckpt, "/root/.cache/huggingface": vol_hf}

grpo_image = modal.Image.debian_slim(python_version="3.11").pip_install(*PIP_GRPO).env(ENV)
infer_image = modal.Image.debian_slim(python_version="3.11").pip_install(*PIP_INFER).env(ENV)
SECRETS = [modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")]


@app.function(image=grpo_image, gpu="H100", cpu=16, memory=65536,
              volumes=VOLUMES, secrets=SECRETS, timeout=8 * 3600)
def train(rank: int = 32, lr: float = 5e-6, max_steps: int = 500, num_generations: int = 8,
          gen_bsz: int = 16, accum: int = 4, max_completion: int = 192, max_prompt: int = 512,
          seq_len: int = 1024, temperature: float = 1.0, beta: float = 0.0, seed: int = 0,
          gpu_mem: float = 0.55, save_steps: int = 100, report_to: str = "wandb"):
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    run_name = f"grpo_r{rank}_lr{lr}_s{seed}"
    out_dir = f"/ckpts/{run_name}"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_MODEL, max_seq_length=seq_len, dtype=None, load_in_4bit=False,
        fast_inference=True, max_lora_rank=rank, gpu_memory_utilization=gpu_mem)
    vol_hf.commit()
    model = FastLanguageModel.get_peft_model(
        model, r=rank, lora_alpha=rank, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=seed)

    sys_prompt = json.loads(open("/data/sft_dev.jsonl").readline())["messages"][0]["content"]
    raw = load_dataset("json", data_files="/data/grpo_train.jsonl", split="train")

    def to_prompt(ex):
        return {"prompt": [{"role": "system", "content": sys_prompt},
                           {"role": "user", "content": ex["nl"]}],
                "ref_query": ex["ref_query"]}

    ds = raw.map(to_prompt, remove_columns=raw.column_names)
    print(json.dumps({"config": {"base": SFT_MODEL, "rank": rank, "lr": lr,
                                 "max_steps": max_steps, "G": num_generations,
                                 "gen_bsz": gen_bsz, "accum": accum, "beta": beta,
                                 "prompts": len(ds), "reward_weights": REWARD_WEIGHTS}}))

    cfg = GRPOConfig(
        output_dir=out_dir, learning_rate=lr, lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03, optim="adamw_8bit", weight_decay=0.01, max_grad_norm=0.5,
        per_device_train_batch_size=gen_bsz, gradient_accumulation_steps=accum,
        num_generations=num_generations, max_prompt_length=max_prompt,
        max_completion_length=max_completion, temperature=temperature, beta=beta,
        max_steps=max_steps, save_steps=save_steps, logging_steps=5,
        reward_weights=REWARD_WEIGHTS, bf16=True, seed=seed,
        report_to=report_to, run_name=run_name, use_vllm=True)

    trainer = GRPOTrainer(model=model, processing_class=tokenizer,
                          reward_funcs=REWARD_FUNCS, args=cfg, train_dataset=ds)
    trainer.train()
    model.save_pretrained(f"{out_dir}/final")
    tokenizer.save_pretrained(f"{out_dir}/final")
    vol_ckpt.commit()
    ckpts = sorted(d for d in os.listdir(out_dir) if d.startswith("checkpoint-"))
    print(json.dumps({"run_name": run_name, "out_dir": out_dir, "checkpoints": ckpts}))
    return run_name, ckpts


@app.function(image=infer_image, gpu="L40S", cpu=4, memory=24576,
              volumes=VOLUMES, secrets=SECRETS, timeout=2 * 3600)
def generate(split: str = "dev", adapter: str = "", name: str = "run",
             base_model: str = SFT_MODEL, temperature: float = 0.0,
             max_tokens: int = 192, seed: int = 0):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sys_prompt = json.loads(open("/data/sft_dev.jsonl").readline())["messages"][0]["content"]
    with open(f"/data/dataset.{split}.nl") as f:
        nls = [l.strip() for l in f]
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def build(nl):
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": nl}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [build(nl) for nl in nls]
    use_lora = bool(adapter)
    llm = LLM(model=base_model, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.92, enable_lora=use_lora, max_lora_rank=64)
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, seed=seed)
    lora = LoRARequest("adapter", 1, f"/ckpts/{adapter}") if use_lora else None
    outs = llm.generate(prompts, sp, lora_request=lora)

    os.makedirs("/ckpts/preds", exist_ok=True)
    out_path = f"/ckpts/preds/{name}_{split}.jsonl"
    with open(out_path, "w") as f:
        for i, (nl, o) in enumerate(zip(nls, outs)):
            f.write(json.dumps({"idx": i, "nl": nl, "output": o.outputs[0].text},
                               ensure_ascii=False) + "\n")
    vol_ckpt.commit()
    print(f"wrote {len(nls)} generations -> {out_path}")
    return out_path


@app.function(image=grpo_image, gpu="A10G", cpu=8, memory=32768,
              volumes=VOLUMES, secrets=SECRETS, timeout=2 * 3600)
def merge_push(adapter: str, repo: str, private: bool = True):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"/ckpts/{adapter}", max_seq_length=2048, dtype=None, load_in_4bit=False)
    local = f"/ckpts/merged/{repo.split('/')[-1]}"
    model.save_pretrained_merged(local, tokenizer, save_method="merged_16bit")
    vol_ckpt.commit()
    model.push_to_hub_merged(repo, tokenizer, save_method="merged_16bit",
                             token=os.environ["HF_TOKEN"], private=private)
    print(f"merged -> {local}  pushed -> https://huggingface.co/{repo}")
    return repo


@app.local_entrypoint()
def main(action: str, adapter: str = "", repo: str = "", split: str = "dev", name: str = "run",
         base_model: str = SFT_MODEL, rank: int = 32, lr: float = 5e-6, max_steps: int = 500,
         num_generations: int = 8, gen_bsz: int = 16, accum: int = 4, beta: float = 0.0,
         temperature: float = 0.0, seed: int = 0, report_to: str = "wandb", private: bool = True):
    if action == "train":
        print(train.remote(rank=rank, lr=lr, max_steps=max_steps, num_generations=num_generations,
                           gen_bsz=gen_bsz, accum=accum, beta=beta, seed=seed, report_to=report_to))
    elif action == "generate":
        print(generate.remote(split=split, adapter=adapter, name=name, base_model=base_model,
                              temperature=temperature, seed=seed))
    elif action == "merge_push":
        assert repo, "pass --repo Melikshah/GeoQL-4B-GRPO"
        print(merge_push.remote(adapter=adapter, repo=repo, private=private))
    else:
        raise ValueError(action)