# GeoQL-4B — Text-to-OverpassQL via SFT + GRPO

Small (4B) local model that compiles natural-language requests into executable
OverpassQL queries for OpenStreetMap. SFT + GRPO with metric-as-reward, GGUF on
llama.cpp. Built for the HF build-small hackathon (Backyard AI track).

## Layout
- `scripts/`   reward fn, data prep, Modal train/generate, eval glue
- `OverpassNL/` upstream dataset + official evaluation (cloned, gitignored)
- `data_out/`  generated SFT/GRPO data
- `preds/`     model generations + official-format prediction files
- `docs/`      day-by-day runbooks

## Quickstart
See docs/day0_runbook.md then docs/day1_pipeline.md
