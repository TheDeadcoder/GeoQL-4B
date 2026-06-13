"""GeoQL-4B SFT on Modal: train (Unsloth LoRA) -> generate (vLLM) -> merge_push (HF).

  # 0. baselines (validate loop, ~$1)
  modal run sft_modal.py --action generate --split dev --shots 0 --name base_0shot
  modal run sft_modal.py --action generate --split dev --shots 5 --name base_5shot

  # 1. two SFT variants in parallel (~$6)
  modal run sft_modal.py --action train --variant full
  modal run sft_modal.py --action train --variant noaux

  # 2. per-epoch dev generations for selection (score locally with score_offline.py)
  modal run sft_modal.py --action generate --adapter sft_full_r32_lr0.0001_s0/checkpoint-N --name sft_full_eK

  # 3. promote winner to HF (GRPO then loads it as base; Modal workspace becomes disposable)
  modal run sft_modal.py --action merge_push --adapter sft_full_r32_lr0.0001_s0/checkpoint-N --repo sakibbuet2k19/GeoQL-4B-SFT

Volumes: geoql-data (inputs), geoql-ckpts (outputs), geoql-hf (model cache).
After the first build, pin versions: `modal image list` then freeze PIP_* below.
"""

import json
import os
import random

import modal

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

PIP_TRAIN = ["unsloth", "trl", "datasets", "wandb", "huggingface_hub", "hf_transfer"]
PIP_INFER = ["vllm", "transformers", "huggingface_hub", "hf_transfer"]
# VLLM_USE_FLASHINFER_SAMPLER=0 -> native torch sampler (FlashInfer JITs a CUDA
# kernel needing nvcc, absent from the slim image). Negligible speed impact;
# carried into GRPO too. VLLM_USE_DEEP_GEMM=0 silences a harmless FP8 warning.
ENV = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_USE_DEEP_GEMM": "0",
}

app = modal.App("geoql")
vol_data = modal.Volume.from_name("geoql-data", create_if_missing=True)
vol_ckpt = modal.Volume.from_name("geoql-ckpts", create_if_missing=True)
vol_hf = modal.Volume.from_name("geoql-hf", create_if_missing=True)
VOLUMES = {"/data": vol_data, "/ckpts": vol_ckpt, "/root/.cache/huggingface": vol_hf}

train_image = modal.Image.debian_slim(python_version="3.11").pip_install(*PIP_TRAIN).env(ENV)
infer_image = modal.Image.debian_slim(python_version="3.11").pip_install(*PIP_INFER).env(ENV)
SECRETS = [modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")]


@app.function(image=train_image, gpu="L40S", cpu=8, memory=32768,
              volumes=VOLUMES, secrets=SECRETS, timeout=5 * 3600)
def train(variant: str = "full", lr: float = 1e-4, epochs: int = 3, rank: int = 32,
          seq_len: int = 1024, bsz: int = 32, accum: int = 1, seed: int = 0,
          report_to: str = "wandb"):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    run_name = f"sft_{variant}_r{rank}_lr{lr}_s{seed}"
    out_dir = f"/ckpts/{run_name}"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=seq_len, dtype=None, load_in_4bit=False)
    vol_hf.commit()
    model = FastLanguageModel.get_peft_model(
        model, r=rank, lora_alpha=rank, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=seed)

    ds = load_dataset("json", data_files="/data/sft_train.jsonl", split="train")
    if variant == "noaux":
        ds = ds.filter(lambda ex: ex["task"] == "forward")
    n_examples = len(ds)

    def fmt(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

    ds = ds.map(fmt, remove_columns=ds.column_names)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            output_dir=out_dir, dataset_text_field="text", packing=False,
            per_device_train_batch_size=bsz, gradient_accumulation_steps=accum,
            num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine",
            warmup_ratio=0.03, logging_steps=10, save_strategy="epoch",
            optim="adamw_8bit", weight_decay=0.01, max_grad_norm=1.0,
            bf16=True, seed=seed, report_to=report_to, run_name=run_name,
            max_seq_length=seq_len))
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n")

    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
    ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
    unmasked = tokenizer.decode([t for t, l in zip(ids, labels) if l != -100])
    n_unmasked = sum(1 for l in labels if l != -100)
    print("=== MASKING CHECK ===")
    print(f"examples={n_examples}  unmasked_tokens_ex0={n_unmasked}")
    print("unmasked target (must be ONLY the overpassql block + im_end):")
    print(repr(unmasked[:400]))
    print("=== END MASKING CHECK ===")

    stats = trainer.train()
    vol_ckpt.commit()
    ckpts = sorted(d for d in os.listdir(out_dir) if d.startswith("checkpoint-"))
    print(json.dumps({"run_name": run_name, "out_dir": out_dir,
                      "checkpoints": ckpts, "train_loss": stats.training_loss}))
    return run_name, ckpts


@app.function(image=infer_image, gpu="L40S", cpu=4, memory=24576,
              volumes=VOLUMES, secrets=SECRETS, timeout=2 * 3600)
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
            msgs.append({"role": "assistant", "content": f"```overpassql\n{p['ref_query']}\n```"})
        msgs.append({"role": "user", "content": nl})
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [build(nl) for nl in nls]
    llm = LLM(model=BASE_MODEL, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.92, enable_lora=bool(adapter), max_lora_rank=64)
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, seed=seed)
    lora = LoRARequest("adapter", 1, f"/ckpts/{adapter}") if adapter else None
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


@app.function(image=train_image, gpu="A10G", cpu=8, memory=32768,
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
def main(action: str, variant: str = "full", adapter: str = "", repo: str = "",
         split: str = "dev", name: str = "run", shots: int = 0, lr: float = 1e-4,
         epochs: int = 3, temperature: float = 0.0, seed: int = 0,
         report_to: str = "wandb", private: bool = True):
    if action == "train":
        print(train.remote(variant=variant, lr=lr, epochs=epochs, seed=seed, report_to=report_to))
    elif action == "generate":
        print(generate.remote(split=split, adapter=adapter, name=name,
                              shots=shots, temperature=temperature, seed=seed))
    elif action == "merge_push":
        assert repo, "pass --repo sakibbuet2k19/GeoQL-4B-SFT"
        print(merge_push.remote(adapter=adapter, repo=repo, private=private))
    else:
        raise ValueError(action)