"""Modal app for GeoQL-4B: SFT training and batch generation.

Usage:
  modal run sft_modal.py --action train --variant full
  modal run sft_modal.py --action train --variant noaux
  modal run sft_modal.py --action generate --adapter sft_full/checkpoint-XXX --split dev --name sft_full_e3
  modal run sft_modal.py --action generate --split dev --shots 5 --name base_5shot

Volumes (create once): geoql-data (datasets), geoql-ckpts (runs + preds), geoql-hf (weights cache).
After the first successful image build, freeze versions: modal image list, then pin in PIP_TRAIN/PIP_INFER.
"""

import json
import os
import random

import modal

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TRAIN_GPU = "L40S"
INFER_GPU = "L40S"

PIP_TRAIN = ["unsloth", "trl", "datasets", "wandb", "huggingface_hub"]
PIP_INFER = ["vllm", "transformers", "huggingface_hub"]

app = modal.App("geoql")
vol_data = modal.Volume.from_name("geoql-data", create_if_missing=True)
vol_ckpt = modal.Volume.from_name("geoql-ckpts", create_if_missing=True)
vol_hf = modal.Volume.from_name("geoql-hf", create_if_missing=True)
VOLUMES = {"/data": vol_data, "/ckpts": vol_ckpt, "/root/.cache/huggingface": vol_hf}

train_image = modal.Image.debian_slim(python_version="3.11").pip_install(*PIP_TRAIN)
infer_image = modal.Image.debian_slim(python_version="3.11").pip_install(*PIP_INFER)

SECRETS = [modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")]


@app.function(image=train_image, gpu=TRAIN_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=6 * 3600)
def train(variant: str = "full", lr: float = 1e-4, epochs: int = 3,
          rank: int = 32, seq_len: int = 1024, bsz: int = 16, accum: int = 2,
          seed: int = 0):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    run_name = f"sft_{variant}_r{rank}_lr{lr}_s{seed}"
    out_dir = f"/ckpts/{run_name}"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=seq_len, dtype=None, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=rank, lora_alpha=rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=seed)

    ds = load_dataset("json", data_files="/data/sft_train.jsonl", split="train")
    if variant == "noaux":
        ds = ds.filter(lambda ex: ex["task"] == "forward")

    def fmt(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

    ds = ds.map(fmt, remove_columns=ds.column_names)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            output_dir=out_dir, dataset_text_field="text",
            per_device_train_batch_size=bsz, gradient_accumulation_steps=accum,
            num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine",
            warmup_ratio=0.03, logging_steps=10, save_strategy="epoch",
            bf16=True, seed=seed, report_to="wandb", run_name=run_name,
            max_seq_length=seq_len))
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n")

    stats = trainer.train()
    vol_ckpt.commit()
    ckpts = sorted(d for d in os.listdir(out_dir) if d.startswith("checkpoint-"))
    print(json.dumps({"run_name": run_name, "checkpoints": ckpts,
                      "train_loss": stats.training_loss}))
    return run_name, ckpts


@app.function(image=infer_image, gpu=INFER_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=2 * 3600)
def generate(split: str = "dev", adapter: str = "", name: str = "run",
             shots: int = 0, temperature: float = 0.0, max_tokens: int = 192,
             seed: int = 0):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sys_prompt = json.loads(open("/data/sft_dev.jsonl").readline())["messages"][0]["content"]
    with open(f"/data/dataset.{split}.nl") as f:
        nls = [l.strip() for l in f]

    shot_pairs = []
    if shots > 0:
        pool = [json.loads(l) for l in open("/data/grpo_train.jsonl")]
        shot_pairs = random.Random(seed).sample(pool, shots)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def build(nl):
        msgs = [{"role": "system", "content": sys_prompt}]
        for p in shot_pairs:
            msgs.append({"role": "user", "content": p["nl"]})
            msgs.append({"role": "assistant",
                         "content": f"```overpassql\n{p['ref_query']}\n```"})
        msgs.append({"role": "user", "content": nl})
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)

    prompts = [build(nl) for nl in nls]
    llm = LLM(model=BASE_MODEL, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.92, enable_lora=bool(adapter),
              max_lora_rank=64)
    sp = SamplingParams(temperature=temperature, top_p=1.0,
                        max_tokens=max_tokens, seed=seed)
    lora = LoRARequest("adapter", 1, f"/ckpts/{adapter}") if adapter else None
    outs = llm.generate(prompts, sp, lora_request=lora)

    os.makedirs("/ckpts/preds", exist_ok=True)
    out_path = f"/ckpts/preds/{name}_{split}.jsonl"
    with open(out_path, "w") as f:
        for i, (nl, o) in enumerate(zip(nls, outs)):
            f.write(json.dumps({"idx": i, "nl": nl,
                                "output": o.outputs[0].text}, ensure_ascii=False) + "\n")
    vol_ckpt.commit()
    print(f"wrote {len(nls)} generations -> {out_path}")
    return out_path


@app.local_entrypoint()
def main(action: str, variant: str = "full", adapter: str = "", split: str = "dev",
         name: str = "run", shots: int = 0, lr: float = 1e-4, epochs: int = 3,
         temperature: float = 0.0, seed: int = 0):
    if action == "train":
        print(train.remote(variant=variant, lr=lr, epochs=epochs, seed=seed))
    elif action == "generate":
        print(generate.remote(split=split, adapter=adapter, name=name,
                              shots=shots, temperature=temperature, seed=seed))
    else:
        raise ValueError(action)
