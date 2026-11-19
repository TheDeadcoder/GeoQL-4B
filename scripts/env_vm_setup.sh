#!/usr/bin/env bash
# One-time setup of a Colab VM for the environment-reward pilot, run from /content holding geoql-env.tar (the
# geoql-env source with data/train-tasks) and areas.osm3s (the area rules of the pinned engine image).
# Builds the pinned Overpass release into /app, where geoql-env's local engine mode expects it, a Python 3.11
# environment with the old GRPO run's packages and geoql-env, then every world and witness world, and fails unless
# every gold answer equals its lock.
set -euo pipefail
cd /content
OVERPASS=0.7.62.11
RULES_SHA256=e2a1e4d59cc665c6079bed0edd3c0f9a5fb46a29777c3a1e4c57dcdc91434f9b

echo "== system packages"
apt-get -qq update > /dev/null
apt-get -qq install -y osmium-tool liblz4-dev libexpat1-dev zlib1g-dev build-essential > /dev/null

echo "== Overpass $OVERPASS"
if [ ! -x /app/bin/osm3s_query ]; then
  curl -sfL -o osm3s.tar.gz "https://dev.overpass-api.de/releases/osm-3s_v$OVERPASS.tar.gz"
  sha256sum osm3s.tar.gz
  tar xzf osm3s.tar.gz
  (cd "osm-3s_v$OVERPASS" && ./configure --prefix=/app --enable-lz4 > /dev/null && make -j"$(nproc)" > /dev/null \
    && make install > /dev/null)
fi
echo "$RULES_SHA256  areas.osm3s" | sha256sum -c -
mkdir -p /app/etc/rules && cp areas.osm3s /app/etc/rules/areas.osm3s
/app/bin/osm3s_query --version

echo "== Python environment"
mkdir -p geoql-env && tar xf geoql-env.tar -C geoql-env
PY=/content/venv/bin/python
[ -x "$PY" ] || uv venv -q --python 3.11 /content/venv
uv pip install -q --python "$PY" unsloth==2025.11.1 unsloth_zoo==2025.11.2 vllm==0.11.0 trl==0.23.0 \
  transformers==4.57.2 datasets==5.0.0 sacrebleu -e /content/geoql-env
"$PY" -c "import torch, trl, vllm, geoql_env; print('torch', torch.__version__, 'vllm', vllm.__version__, 'trl', trl.__version__)"

echo "== worlds and witness worlds"
export GEOQL_ENGINE=local
cd /content/geoql-env
/content/venv/bin/geoql-env worlds build > /content/worlds.log 2>&1
/content/venv/bin/geoql-env gold > /content/gold-benchmark.log 2>&1 &
benchmark=$!
/content/venv/bin/geoql-env generate --check --jobs "$(nproc)" | tee /content/gold-train.log
wait "$benchmark" || { echo "benchmark gold answers differ from the lock:"; grep -E "DIFFERS|not in the lock" /content/gold-benchmark.log | head; exit 1; }
echo "benchmark gold answers equal the lock"
