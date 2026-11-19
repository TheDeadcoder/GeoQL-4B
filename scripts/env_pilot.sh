#!/usr/bin/env bash
# The environment-reward pilot as one detached job on a Colab A100, run from /content holding the env_* scripts,
# reward.py, geoql-env.tar and areas.osm3s. Stages: VM setup; prompts; the SFT model's difficulty profile on the
# training tasks, which keeps the tasks with a signal; its evaluation; a 2-step smoke run; then per arm (environment
# reward, then old reward) GRPO and the evaluation of its checkpoints. Evaluation covers the 180 held-out tasks and
# the 67 benchmark test tasks, 8 samples at temperature 1.0 and greedy. A finished stage leaves a mark in marks/
# and is skipped when the job runs again.
set -uo pipefail
cd /content
STEPS=${STEPS:-150}
PY=/content/venv/bin/python
SFT=Melikshah/GeoQL-4B-SFT
T=/content/geoql-env/data/train-tasks
mkdir -p marks out

run() {
  if [ -e "marks/$1" ]; then echo "== $1 (done before)"; return 0; fi
  echo "== $1"
  "${@:2}" && touch "marks/$1" || { echo "FAILED $1"; exit 1; }
}

prompts() {
  $PY env_prompts.py out/train-prompts.jsonl $T/li-liechtenstein $T/mt-malta $T/im-isle-of-man $T/fo-faroe-islands \
    && $PY env_prompts.py out/eval-prompts.jsonl $T/gg-guernsey-jersey benchmark:test
}

profile() {
  $PY env_generate.py --prompts out/train-prompts.jsonl --model $SFT --samples 8 --out out/profile.jsonl \
    && $PY env_grade.py out/profile.jsonl out/profile-verdicts.jsonl $T --keep out/keep.txt
}

evaluate() {
  local name=$1 adapter=${2:-}
  $PY env_generate.py --prompts out/eval-prompts.jsonl --model $SFT ${adapter:+--adapter $adapter} --samples 8 \
      --out "out/eval-$name-t1.jsonl" \
    && $PY env_generate.py --prompts out/eval-prompts.jsonl --model $SFT ${adapter:+--adapter $adapter} --greedy \
      --out "out/eval-$name-greedy.jsonl" \
    && $PY env_grade.py "out/eval-$name-t1.jsonl" "out/eval-$name-t1-verdicts.jsonl" $T/gg-guernsey-jersey benchmark:test \
    && $PY env_grade.py "out/eval-$name-greedy.jsonl" "out/eval-$name-greedy-verdicts.jsonl" $T/gg-guernsey-jersey benchmark:test
}

smoke() {
  $PY env_grpo.py --reward env --tasks $T --keep out/keep.txt --steps 2 --out out/smoke \
    && $PY env_generate.py --prompts out/eval-prompts.jsonl --model $SFT --adapter out/smoke/final --greedy \
      --out out/smoke-eval.jsonl
}

train() {
  $PY env_grpo.py --reward "$1" --tasks $T --keep out/keep.txt --steps "$STEPS" --out "out/$1"
}

token() {
  for _ in $(seq 1 240); do
    $PY -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)" 2>/dev/null && return 0
    sleep 15
  done
  return 1
}

run setup bash env_vm_setup.sh
export GEOQL_ENGINE=local
run token token
run prompts prompts
run profile profile
run eval-sft evaluate sft
run smoke smoke
for arm in env old; do
  run "train-$arm" train "$arm"
  for step in 50 100 150; do
    run "eval-$arm-$step" evaluate "$arm-$step" "out/$arm/checkpoint-$step"
  done
done
echo PILOT_DONE
touch pilot.done
