"""GRPO from the GeoQL-4B SFT model on generated geoql-env training tasks, with one of two rewards and otherwise the
settings of grpo_modal.py's run: the environment's reward (1 if the query passes on the real world and on every
witness world, else 0) or the old reward (a format gate plus KVS, structure, chrF and exact-match similarity to the
task's first reading, with the old weights). Every rollout is also scored by the other reward and by a
real-world-only one, at weight 0, for the logs. Changes from the old run, the same in both arms: the benchmark's
single-shot prompt, a 512-token completion limit with truncated completions masked out of the loss, and no
Weights & Biases.

  python scripts/env_grpo.py --reward env --tasks geoql-env/data/train-tasks --keep keep.txt --out runs/env
"""

import argparse
import json
from pathlib import Path

from reward import FENCE_RE, chrf_score, exact_match, extract_query, kvs_proxy, sanity_check, struct_proxy

SFT_MODEL = "Melikshah/GeoQL-4B-SFT"
TRAIN_WORLDS = ("li-liechtenstein", "mt-malta", "im-isle-of-man", "fo-faroe-islands")
OLD_WEIGHTS = (1.0, 0.45, 0.15, 0.20, 0.20)  # format, kvs, struct, chrf, em, as in grpo_modal.py


def _texts(completions):
    return [c[0]["content"] if isinstance(c, list) else c for c in completions]


def _component_query(text):
    q, err = extract_query(text)
    if q is not None:
        return q
    if err in ("trailing_text", "multiple_blocks"):
        blocks = FENCE_RE.findall(text)
        return blocks[0].strip() if blocks else None
    return None


def reward_format(completions, **kwargs):
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
    return [kvs_proxy(q, r) if (q := _component_query(t)) else 0.0 for t, r in zip(_texts(completions), ref_query)]


def reward_struct(completions, ref_query=None, **kwargs):
    return [struct_proxy(q, r) if (q := _component_query(t)) else 0.0 for t, r in zip(_texts(completions), ref_query)]


def reward_chrf(completions, ref_query=None, **kwargs):
    return [chrf_score(q, r) if (q := _component_query(t)) else 0.0 for t, r in zip(_texts(completions), ref_query)]


def reward_em(completions, ref_query=None, **kwargs):
    return [exact_match(q, r) if (q := _component_query(t)) else 0.0 for t, r in zip(_texts(completions), ref_query)]


OLD_FUNCS = [reward_format, reward_kvs, reward_struct, reward_chrf, reward_em]


def environment_rewards(grader):
    """The environment's reward and the real-world-only one, from one grading of each rollout."""

    def reward_env(completions, task_id=None, **kwargs):
        return [float(v["category"] == "pass") for v in grader.verdicts(task_id, _texts(completions))]

    def reward_real(completions, task_id=None, **kwargs):
        return [float(v["real"]) for v in grader.verdicts(task_id, _texts(completions))]

    return [reward_env, reward_real]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=("env", "old"), required=True)
    ap.add_argument("--tasks", required=True, help="geoql-env's data/train-tasks folder")
    ap.add_argument("--keep", help="ids of the tasks to train on, one per line (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--gpu-memory", type=float, default=0.5)
    opts = ap.parse_args()

    import torch
    from unsloth import FastLanguageModel  # before trl, which Unsloth patches
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from env_grade import Grader
    from env_prompts import load
    from geoql_env import prompt

    # Variable-length completions trigger many Dynamo specializations in Unsloth's compiled loss, as in grpo_modal.py.
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 1024

    chosen = [t for w in TRAIN_WORLDS for t in load([str(Path(opts.tasks) / w)])]
    if opts.keep:
        wanted = set(Path(opts.keep).read_text().split())
        chosen = [t for t in chosen if t.id in wanted]
    data = Dataset.from_list([{"prompt": prompt.messages(t, "single_shot", 0), "task_id": t.id,
                               "ref_query": t.interpretations[0].program} for t in chosen])
    funcs = environment_rewards(Grader(chosen)) + OLD_FUNCS
    weights = [1.0, 0.0] + [0.0] * len(OLD_FUNCS) if opts.reward == "env" else [0.0, 0.0, *OLD_WEIGHTS]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_MODEL, max_seq_length=1152, dtype=None, load_in_4bit=False, fast_inference=True,
        max_lora_rank=opts.rank, gpu_memory_utilization=opts.gpu_memory, enforce_eager=True)
    model = FastLanguageModel.get_peft_model(
        model, r=opts.rank, lora_alpha=opts.rank, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=opts.seed)

    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)

    class History(TrainerCallback):
        def on_log(self, args, state, control, **kwargs):
            (out / "log_history.json").write_text(json.dumps(state.log_history))

    config = GRPOConfig(
        output_dir=str(out), learning_rate=opts.lr, lr_scheduler_type="constant_with_warmup", warmup_ratio=0.03,
        optim="adamw_8bit", weight_decay=0.01, max_grad_norm=0.5, per_device_train_batch_size=16,
        gradient_accumulation_steps=4, num_generations=8, max_prompt_length=600, max_completion_length=512,
        mask_truncated_completions=True, temperature=1.0, beta=0.0, max_steps=opts.steps, save_steps=50,
        logging_steps=5, reward_weights=weights, bf16=True, seed=opts.seed, report_to="none", use_vllm=True)
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=funcs, args=config,
                          train_dataset=data, callbacks=[History()])
    trainer.train()
    model.save_pretrained(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history))
    print(json.dumps({"reward": opts.reward, "tasks": len(chosen), "steps": opts.steps, "out": str(out)}))


if __name__ == "__main__":
    main()
