#!/usr/bin/env bash
# Step 0 job as run on the Colab A100, from /content holding env_generate.py and the prompt files.
set -uo pipefail
cd /content
echo "== environment"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "== installing vLLM 0.11.0 in a Python 3.11 environment"
PY=/content/venv/bin/python
{ [ -x $PY ] || uv venv -q --python 3.11 /content/venv; } && uv pip install -q --python $PY vllm==0.11.0 transformers==4.57.2 \
  && $PY -c "import torch, vllm; assert torch.cuda.is_available(); print('vllm', vllm.__version__, 'torch', torch.__version__)" \
  || { echo INSTALL_FAILED; touch step0.done; exit 1; }
gen() { $PY env_generate.py --prompts "$1" --model "$2" --out "$3" "${@:4}" || echo "FAILED $3"; }
BASE=Qwen/Qwen3-4B-Instruct-2507
SFT=Melikshah/GeoQL-4B-SFT
echo "== base model on the benchmark"
gen env_prompts.jsonl $BASE base-t1.jsonl --samples 16
gen env_prompts.jsonl $BASE base-greedy.jsonl --greedy
echo "== waiting for the Hugging Face login"
for i in $(seq 1 240); do $PY -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)" 2>/dev/null && break; sleep 15; done
echo "== SFT model on the benchmark"
gen env_prompts.jsonl $SFT sft-t1.jsonl --samples 16
gen env_prompts.jsonl $SFT sft-greedy.jsonl --greedy
echo "== SFT model on OverpassNL prompts, sampled as the old GRPO rollouts were"
gen onl_dev_prompts.jsonl $SFT onl-dev-greedy.jsonl --greedy --max_tokens 192
gen onl_train500_prompts.jsonl $SFT onl-train-t1.jsonl --samples 8 --max_tokens 192
gen onl_train500_prompts.jsonl $SFT onl-train-greedy.jsonl --greedy --max_tokens 192
gen onl_dev500_prompts.jsonl $SFT onl-dev-t1.jsonl --samples 8 --max_tokens 192
echo STEP0_DONE
touch step0.done
