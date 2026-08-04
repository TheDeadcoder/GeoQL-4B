# GeoQL-4B: Text-to-OverpassQL with SFT and GRPO

A 4B parameter local model that compiles natural language requests into executable
[OverpassQL](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL) queries for
OpenStreetMap, trained with supervised fine-tuning followed by GRPO with a metric-as-reward
objective.

**Base model:** `Qwen/Qwen3-4B-Instruct-2507`
**Benchmark:** [OverpassNL](https://github.com/raphael-sch/OverpassNL) (Staniek et al., TACL 2024), 6,352 train / 1,000 dev / 1,000 test
**Models:** [`Melikshah/GeoQL-4B-SFT`](https://huggingface.co/Melikshah/GeoQL-4B-SFT)

> **Headline result and honest summary.** SFT worked and produced a usable model at
> **68.49 official dev OQS**. GRPO on top of it moved the score to **68.72 OQS**, a gain of
> **+0.23**, which is within run-to-run noise. The reinforcement learning stage, which was
> the central research bet of this project, did not deliver. This README documents why, with
> the measurements that support each claim, because the negative result is the most useful
> thing this repository contains.

---

## 1. Results

### 1.1 Main table (official OverpassNL evaluation, dev split, n=1000)

| Stage | Model / checkpoint | OQS | EM | chrF | kv_overlap | xml_overlap |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Qwen3-4B-Instruct, 0-shot | 30.93* | 0.00 | 23.39 | 29.02* | 40.37* |
| Baseline | Qwen3-4B-Instruct, 5-shot | 53.09* | 1.90 | 49.94 | 38.55* | 70.77* |
| SFT r32 | `sft_full_r32`, epoch 3 | 66.36 | 14.20 | 69.89 | 60.00 | 69.18 |
| SFT r64 | `sft_noaux_r64` ckpt-796 | 68.24 | 18.20 | 72.36 | 61.60 | 70.77 |
| **SFT r64** | **`sft_full_r64` ckpt-1720** | **68.49** | **18.30** | 72.22 | **61.88** | 71.38 |
| GRPO | `grpo_r32` ckpt-100 | 68.42 | 18.20 | 72.28 | 61.75 | 71.23 |
| GRPO | `grpo_r32` ckpt-200 | 68.55 | 18.40 | 72.41 | 61.78 | 71.47 |
| GRPO | `grpo_r32` ckpt-300 | 68.43 | 17.40 | 72.39 | 61.68 | 71.22 |
| GRPO | `grpo_r32` ckpt-400 | 68.31 | 17.90 | 72.39 | 61.26 | 71.26 |
| **GRPO** | **`grpo_r32` ckpt-500** | **68.72** | 17.40 | 72.83 | 61.69 | 71.63 |

\* Baseline rows are offline proxy scores (see section 3.2), not official convert-based scores.

### 1.2 Against published systems on the same benchmark

| System | Params | Method | OQS | EM |
|---|---:|---|---:|---:|
| OverpassT5 (Staniek et al. 2024) | 582M | full fine-tune | 71.2 | 20.7 |
| GPT-4 + retrieval | n/a | 5-shot ICL | 71.4 | n/a |
| Claude-4-Sonnet + retrieval + tag aug | n/a | 5-shot ICL | 73.2 | n/a |
| OsmT (2025, SOTA) | 582M | SFT + tag retrieval | 74.4 | 24.5 |
| **GeoQL-4B (this work)** | **4B** | **SFT + GRPO, LoRA** | **68.72** | **17.4** |

The project did not beat the published baselines. The gap to OverpassT5 is 2.5 OQS and the
gap to SOTA is 5.7 OQS.

### 1.3 SFT learning curve (rank 64, 5 epochs)

| Variant | Epoch | proxy OQS | EM | chrF | KVS | TreeS |
|---|---:|---:|---:|---:|---:|---:|
| full | 1 | 67.30 | 10.20 | 68.05 | 54.79 | 79.05 |
| full | 2 | 69.40 | 14.00 | 69.00 | 59.74 | 79.47 |
| full | 3 | 70.76 | 16.50 | 70.55 | 60.71 | 81.02 |
| full | 4 | 72.01 | 18.20 | 71.80 | 62.35 | 81.88 |
| full | 5 | **72.22** | **18.30** | 72.22 | 62.14 | 82.30 |
| noaux | 4 | 72.06 | 18.20 | 72.36 | 61.82 | 82.01 |
| noaux | 5 | 71.68 | 18.00 | 71.87 | 61.16 | 82.00 |

> ![image-7](https://ik.imagekit.io/sakib61/geoql/image_7.png)
> ![image-8](https://ik.imagekit.io/sakib61/geoql/image_8.png)

---

## 2. Failure analysis

This is the core of the document. The question is not "did GRPO crash", because it did not.
The question is why a reinforcement learning stage that measurably optimized its own objective
produced no measurable improvement on the evaluation metric.

### 2.1 GRPO optimized its reward successfully

Every reward component rose monotonically over 500 steps. Total reward climbed from about 0.25
to about 0.60. The optimizer was working exactly as intended.

![image_3](https://ik.imagekit.io/sakib61/geoql/image_3.png)
> This shows total reward rising from 0.25 to 0.60, and chrF reward rising from 0.58 to 0.78.

![image-2](https://ik.imagekit.io/sakib61/geoql/image_2.png)
> This shows KVS reward rising 0.50 to 0.70, struct reward rising 0.62 to 0.83, and format
> reward rising from -0.25 to about -0.03.

![image-4](https://ik.imagekit.io/sakib61/geoql/image_4.png)
> KL is flat at zero because beta was set to 0. Gradient norms are stable. Nothing about the
> optimization is broken.

Observed training reward at step 500 versus step 0:

| Reward component | Start | End | Change |
|---|---:|---:|---:|
| total reward | 0.25 | 0.60 | +0.35 |
| reward_kvs | 0.50 | 0.70 | +0.20 |
| reward_struct | 0.62 | 0.83 | +0.21 |
| reward_chrf | 0.58 | 0.78 | +0.20 |
| reward_format | -0.25 | -0.03 | +0.22 |

The corresponding change on the official evaluation was +0.23 OQS. A 0.35 rise in training
reward bought essentially nothing on the benchmark.

### 2.2 What actually changed on the eval set: churn, not improvement

Diffing per-example official scores between the SFT winner and the GRPO winner over all 1,000
dev examples:

| Measurement | Value |
|---|---:|
| Examples where OQS improved | 200 |
| Examples where OQS regressed | 191 |
| Examples unchanged | 609 |
| Total OQS gained across improved examples | +24.61 |
| Total OQS lost across regressed examples | -22.31 |
| Net | +2.30 (that is +0.23 OQS points over 1000) |
| Exact matches gained | 12 |
| Exact matches lost | 21 |
| Catastrophic failures (OQS below 0.10) | 13 to 17 (worse) |

This is the signature of a policy that moved without learning anything the evaluation rewards.
Roughly 39 percent of predictions changed, and the changes were nearly symmetric. GRPO
redistributed errors rather than reducing them. Exact match actually got worse, and
catastrophic failures increased.

### 2.3 Root cause 1: the reward was reference-based, so it was largely redundant with SFT

This is the deepest reason and it was baked into the design from the start.

The reward scored similarity to the gold query: KVS overlap against `ref_query`, chrF against
`ref_query`, exact match against `ref_query`. But supervised fine-tuning had already optimized
toward those exact same gold strings via token level cross entropy. GRPO was therefore
re-optimizing a target that SFT had already extracted most of the signal from. It added
sequence level shaping, which is why format compliance improved at sampling temperature, but
it added almost no information the model had not already been trained on.

Genuine RLVR needs a **verifiable outcome** reward that is independent of the reference string.
For this task that means executing the generated query against an OpenStreetMap database and
comparing returned element sets. That reward can distinguish a query that is textually
different from the gold but semantically correct, which is exactly the signal SFT cannot
provide. We used a proxy of the gold string instead, because planet scale execution was
infeasible (the authors' execution database is a 306 GB download, and the public Overpass API
cannot be used as a training loop).

### 2.4 Root cause 2: the binding constraint is knowledge, and RL cannot inject knowledge

Decomposing the 1,000 dev predictions by tag grounding quality:

| kv_overlap bucket | Count | Share |
|---|---:|---:|
| Exactly 0 (completely wrong tags) | 77 | 7.7% |
| Above 0 but below 0.5 | 302 | 30.2% |
| 0.5 to below 1.0 | 220 | 22.0% |
| Exactly 1.0 (perfect tags) | 401 | 40.1% |

Counterfactual ceilings computed from the same per-example scores:

| Scenario | Resulting OQS | Delta |
|---|---:|---:|
| Actual | 68.73 | baseline |
| If tag grounding (KVS) were perfect | **81.50** | **+12.77** |
| If structure (xml_overlap) were perfect | 78.17 | +9.44 |

Tag grounding is worth nearly 13 OQS points, more than double the 5.7 point gap to the
published state of the art. And 7.7 percent of examples have literally zero tag overlap,
meaning the model chose entirely the wrong OSM keys and values.

This is a knowledge problem. The model does not know that a beekeeper's workplace is
`craft=beekeeper` rather than `workplace=beekeeper`. Reinforcement learning reshapes the
probability distribution over outputs the model can already produce. It cannot add facts that
are not in the weights. Attacking a knowledge bottleneck with a policy gradient method was a
category error, and the eval numbers confirm it: kv_overlap went from 61.88 down to 61.69
across the GRPO run.

### 2.5 Root cause 3: training and evaluation used different decoding regimes

GRPO rollouts were sampled at temperature 1.0. Evaluation used greedy decoding at temperature
0.0. The `reward_format` curve rising from -0.25 to -0.03 means GRPO learned to stop producing
malformed output **when sampling at temperature 1.0**. But greedy decoding was already clean:
the SFT model produced only 14 unparseable outputs out of 1,000 under greedy decoding.

A large share of the measured reward improvement therefore lived in a region of the output
distribution that the evaluation protocol never visits. This is a self-inflicted wound and it
is straightforward to avoid.

### 2.6 Root cause 4: the weakest reward proxy absorbed the optimization pressure

Before training, `validate_reward_proxy.py` measured how well the offline reward components
track the official convert-based metrics on 591 perturbed query pairs:

| Reward component | Pearson vs official | Spearman vs official |
|---|---:|---:|
| kvs_proxy vs kv_overlap | 0.946 | **0.819** |
| struct_proxy vs xml_overlap | 0.835 | **0.513** |

The KVS proxy was faithful. The structure proxy was not. During training, `reward_struct/mean`
rose from 0.62 to 0.83, a gain of 0.21, while the real `xml_overlap` on the evaluation set moved
from 71.38 to 71.63, a gain of 0.25 out of 100. The model found the gap between the proxy and
the true metric and optimized into it. Giving 0.15 weight to a component with 0.51 rank
correlation was a mistake even at that low weight.

![image-1](https://ik.imagekit.io/sakib61/geoql/image_1.png)
> The struct reward standard deviation collapses from 0.38 to about 0.15, showing the policy
> converging onto a narrow structural pattern.

### 2.7 Root cause 5: the learning signal collapsed after roughly 200 steps

GRPO advantages are computed within a group of sampled completions. When all completions in a
group receive similar rewards, the advantage goes to zero and no gradient flows.

- `train/reward_std` fell from 0.65 to about 0.15 over the run.
- `train/frac_reward_zero_std` rose from near 0 toward 0.20, meaning up to a fifth of prompt
  groups produced identical rewards across all 8 samples and contributed nothing.
- Checkpoint 200 already scored 68.55 OQS. Checkpoints 300 and 400 were worse (68.43, 68.31).
  Checkpoint 500 at 68.72 is not a trend, it is noise around a plateau.

Steps 200 to 500 cost GPU time and produced no reliable improvement.

### 2.8 A measurable harm: verbosity drift caused new truncation failures

GRPO made outputs longer, which is a known tendency of RL fine-tuning, and this had a concrete
cost because generation was capped at 192 tokens.

| Measurement | SFT | GRPO | Gold |
|---|---:|---:|---:|
| Mean raw output length (chars) | 245.4 | 257.2 | n/a |
| Mean extracted query length (chars) | 205.8 | 211.8 | **202.9** |
| Unparseable outputs (no closed code fence) | 14 | 18 | n/a |
| Of those, caused by truncation | 14 (100%) | 18 (100%) | n/a |
| New truncation failures introduced by GRPO | | 8 | |
| Truncation failures fixed by GRPO | | 4 | |

Every single parse failure in both models is a truncated generation: the model opened a
```` ```overpassql ```` fence and hit the token limit before closing it. GRPO's verbosity drift
moved the mean query length **away** from the gold mean (202.9 chars) and produced a net 4
additional total failures. Each of these scores near zero OQS, which is why catastrophic
failures rose from 13 to 17.

![image_5](https://ik.imagekit.io/sakib61/geoql/image_5.png)
> Mean completion length rises from about 77 tokens to about 90 tokens over the run.

![image_6](https://ik.imagekit.io/sakib61/geoql/image_6.png)
> `clipped_ratio` hovering between 0.01 and 0.07 shows that generations were hitting the
> 192 token cap throughout training.

### 2.9 A secondary finding: the aux data effect shrank as capacity grew

The `full` variant added roughly 4,000 synthetic tag-grounding examples plus 634 reverse
explanation examples on top of the 6,349 OverpassNL pairs. The `noaux` variant used only the
OverpassNL pairs.

| Setting | full KVS | noaux KVS | Advantage of aux data |
|---|---:|---:|---:|
| LoRA rank 32 | 60.11 | 58.64 | +1.47 |
| LoRA rank 64 | 62.14 | 61.82 | +0.28 |

Added model capacity substituted for explicit tag-grounding data. At rank 64 the aux data was
worth almost nothing. This suggests the synthetic templated tag data was too shallow to teach
what retrieval-based approaches like OsmT achieve at inference time.

### 2.10 What was NOT the cause

To be precise about what was ruled out with direct evidence:

- **Not a masking bug.** The training-time masking check confirmed only the assistant query
  tokens were unmasked, with the system prompt and question correctly ignored by the loss.
- **Not a chat template mismatch.** Qwen3-4B-Instruct-2507 emits no think block, and the
  training text and the generation prompt were verified to align exactly at
  `<|im_start|>assistant\n`.
- **Not a broken reward implementation.** `test_reward.py` verifies that 300 out of 300 real
  gold queries score exactly 1.0, that the chrF and exact match implementations reproduce the
  official OverpassNL functions with zero numerical difference, and that reward ordering is
  correct across perfect, wrong-tag, malformed, and garbage completions.
- **Not an unstable optimization.** Gradient norms, loss, and importance sampling ratios were
  all well behaved for the entire run.

The pipeline was correct. The research hypothesis was wrong.

---

## 3. What we would do differently

Ordered by expected value, based on the measurements above.

### 3.1 Attack the knowledge bottleneck, not the policy

The counterfactual analysis says tag grounding is worth 12.77 OQS. Nothing else comes close.
Three ways to get it, in increasing order of cost:

1. **Inference-time tag retrieval**, which is what OsmT does to reach 74.4. Embed the OSM wiki
   tag descriptions, retrieve candidate tags for the user request, and place them in the
   prompt. This is a documented, working path to state of the art on this benchmark and it
   requires no additional training.
2. **Full fine-tuning instead of LoRA.** OverpassT5 reached 71.2 with a full fine-tune of a
   582M model. LoRA rank 64 on a 4B model reached 68.49. Adapter capacity is plausibly limiting
   memorization of a long tail of tags. The code path exists in `scripts/sft_modal.py` behind
   the `--full_ft` flag but was not run.
3. **Deeper synthetic tag data.** The templated tag data used here was too shallow. Real usage
   examples per tag, mined from the OSM wiki and taginfo with natural phrasings, would be a
   better substitute for retrieval.

### 3.2 If doing RL at all, make the reward execution-grounded

The correct RLVR design for this task rewards whether the query returns the right elements,
not whether it looks like the gold string. Practical route: restrict RL training to prompts
resolvable within a regional OSM extract (a country level Geofabrik extract in a local Overpass
container), and use Jaccard overlap of returned element IDs as the reward. This is a much
smaller training set, but it is real verification and it can reward semantically correct
queries that differ textually from the reference.

### 3.3 Fix the specific mistakes

| Mistake | Fix |
|---|---|
| Rollouts at temperature 1.0, eval greedy | Evaluate at the sampling temperature used in training, or train nearer the eval regime |
| Weak struct proxy in the reward at 0.15 weight | Drop any component whose rank correlation with the true metric is below about 0.7 |
| Fixed 192 token cap causing truncation | Raise the cap to about 320 tokens, or add an explicit penalty for unterminated generations |
| No length control | Add a mild length penalty toward the gold length distribution to counter verbosity drift |
| Ran 500 steps after signal collapsed at 200 | Early stop on `frac_reward_zero_std` and `reward_std`, and evaluate on dev during training rather than only after |
| Single seed | The +0.23 result is meaningless without at least 3 seeds to establish the noise band |

### 3.4 Reconsider the base model

Qwen3 uses byte pair encoding. OverpassT5 and OsmT use ByT5, which is byte level and better
matched to the punctuation dense syntax of OverpassQL. A 4B decoder losing to a 582M byte level
encoder decoder on this task is a real signal about tokenizer fit, not just about scale. Testing
a byte level or character aware model would be a legitimate experiment.

---

## 4. Reproducing this work

### 4.1 Setup

```bash
git clone https://github.com/TheDeadcoder/GeoQL-4B.git && cd GeoQL-4B
uv venv --python 3.10 && source .venv/bin/activate
uv pip install sacrebleu numpy requests modal huggingface_hub

# dataset ships inside the vendored upstream repo
cd OverpassNL/dataset && unzip -o OverpassNL_v1.zip && unzip -o comments.zip && cd ../..

# verify the reward implementation against the official metric
python scripts/test_reward.py          # expect ALL TESTS PASSED, chrF delta 0.00e+00

# build training data
python scripts/prepare_sft_data.py --use_comments --out_dir data_out
```

### 4.2 Modal setup

```bash
modal setup
modal volume create geoql-data && modal volume create geoql-ckpts && modal volume create geoql-hf
modal secret create hf-token HF_TOKEN=hf_yourWriteToken
modal secret create wandb WANDB_API_KEY=your_key

for f in data_out/*.jsonl; do modal volume put geoql-data "$f" "/$(basename $f)"; done
for s in dev test; do for e in nl query bbox; do
  modal volume put geoql-data "OverpassNL/dataset/dataset/dataset.$s.$e" "/dataset.$s.$e"; done; done

modal run scripts/modal_hello.py       # sanity check GPU and volumes
```

### 4.3 Train and evaluate

```bash
# SFT
modal run scripts/sft_modal.py --action train --variant full  --rank 64 --epochs 5
modal run scripts/sft_modal.py --action train --variant noaux --rank 64 --epochs 5

# generate dev predictions for a checkpoint
modal run scripts/sft_modal.py --action generate \
  --adapter sft_full_r64_lr0.0001_s0/checkpoint-1720 --name full64_e5
modal volume get geoql-ckpts preds/full64_e5_dev.jsonl preds/

# fast offline scoring (no convert endpoint needed)
python scripts/score_offline.py --raw preds/full64_e5_dev.jsonl --split dev --show_worst 5

# official scoring
python scripts/make_preds.py --raw preds/full64_e5_dev.jsonl --out preds/full64_e5_dev.txt
cd OverpassNL/evaluation && python run_evaluation.py \
  --ref_file ../dataset/dataset/dataset.dev \
  --model_output_file ../../preds/full64_e5_dev.txt --compute_execution false

# promote to Hugging Face
modal run scripts/sft_modal.py --action merge_push \
  --adapter sft_full_r64_lr0.0001_s0/checkpoint-1720 --repo Melikshah/GeoQL-4B-SFT

# GRPO
modal run scripts/grpo_modal.py --action train --max_steps 500
```

### 4.4 A note on the official convert endpoint

The official evaluator calls an Overpass `/api/convert` endpoint to turn OverpassQL into XML for
the kv_overlap and xml_overlap metrics. Two gotchas cost time here:

1. The upstream converter issues `requests.get(url)` with no headers, and the public
   `overpass-api.de` front end rejects the default `python-requests` user agent with
   HTTP 406. Sending a browser style `User-Agent` and `Accept` header fixes it. This is
   patched in `scripts/validate_reward_proxy.py`.
2. `/api/convert` only parses the query and never reads the database, so a local Overpass
   container initialized with a small regional extract can serve conversions for the entire
   globally distributed dev and test sets:

```bash
docker run -d --name overpass --platform linux/amd64 \
  -e OVERPASS_META=yes -e OVERPASS_MODE=init \
  -e OVERPASS_PLANET_URL=https://download.geofabrik.de/asia/bangladesh-latest.osm.pbf \
  -v $PWD/overpass_db:/db -p 12346:80 wiktorn/overpass-api
# then pass --convert_url http://localhost:12346/api/convert
```

---

## 5. Repository layout

```
scripts/
  reward.py                  reward primitives; EM and chrF exactly replicate the official metric
  test_reward.py             correctness tests for the above, run against the real dataset
  prepare_sft_data.py        builds sft_train.jsonl, sft_dev.jsonl, grpo_train.jsonl
  score_offline.py           fast convert-free OQS proxy for checkpoint selection
  make_preds.py              converts raw generations to official one-query-per-line format
  validate_reward_proxy.py   measures proxy vs official metric correlation
  sft_modal.py               Modal app: SFT train, vLLM generate, merge and push to HF
  grpo_modal.py              Modal app: GRPO train with five separate reward components
  modal_hello.py             GPU and volume sanity check
preds/
  r64_20260614/              SFT rank 64 predictions, per-epoch scores, official results
  grpo_20260615/             GRPO predictions per checkpoint, official results
data_out/stats.json          dataset build statistics
proxy_validation.csv         591 rows of proxy vs official metric measurements
OverpassNL/                  vendored upstream dataset and official evaluation code
```

### Design notes worth reusing

- **Two-tier evaluation.** `score_offline.py` computes a convert-free OQS proxy in seconds for
  ranking checkpoints, and the official `run_evaluation.py` confirms the final pick. This kept
  the iteration loop fast without ever selecting on a self-written metric.
- **Reward components as separate functions.** `grpo_modal.py` registers five reward functions
  rather than one composite, so TRL logs each independently. This is the only reason the
  struct proxy gaming in section 2.6 was visible at all.
- **Proxy validation before training.** `validate_reward_proxy.py` measured proxy fidelity
  against the official metric on perturbed pairs before any GPU time was spent. It correctly
  predicted which component would be unreliable.

---

## 6. Cost and compute

| Stage | Hardware | Approximate time | Approximate cost |
|---|---|---:|---:|
| SFT rank 32, 2 variants | Modal L40S | 3 hours | 6 USD |
| SFT rank 64, 2 variants | Modal L40S | 3 hours | 6 USD |
| Per-checkpoint dev generation | Modal L40S | 2 hours | 4 USD |
| GRPO 500 steps | Modal H100 | 4 hours | 17 USD |
| Merge and push | Modal A10G | 0.5 hours | 1 USD |
| **Total** | | | **about 35 USD** |

Local development and all official evaluation ran on an Apple M4 Max with 36 GB unified memory.

---

## 7. Acknowledgements and citation

The dataset, the evaluation code, and the OQS metric are the work of Staniek et al. This
repository vendors their code under `OverpassNL/` for reproducibility.

```bibtex
@article{staniek2024overpassnl,
  title={Text-to-OverpassQL: A Natural Language Interface for Complex Geodata Querying of OpenStreetMap},
  author={Staniek, Michael and Schumann, Raphael and Riezler, Stefan},
  journal={Transactions of the Association for Computational Linguistics},
  year={2024}
}
```

The upstream repository does not carry a top level license file. The vendored code and dataset
are included here for research reproducibility only. Anyone intending commercial use should
contact the original authors first.

Base model: `Qwen/Qwen3-4B-Instruct-2507` by Alibaba Cloud. Training used
[Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl),
with [Modal](https://modal.com) for compute and [vLLM](https://github.com/vllm-project/vllm) for
batch inference.

---

## 8. Status

The SFT model is released and usable. The GRPO experiment is documented as a negative result.
The most likely path to improving on this work is inference-time tag retrieval combined with a
full fine-tune, not more reinforcement learning against a reference-based reward.