#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

SEED="${SEED:-2024}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
REFRESH_CACHE="${REFRESH_CACHE:-0}"

DATA_DIR="${DATA_DIR:-$ROOT/data/prep_inspired_nv_hg_seeded}"
TRAIN_PATH="${TRAIN_PATH:-$DATA_DIR/coral_nv_hg_seed${SEED}_inspired_train_top100.jsonl}"
VALID_PATH="${VALID_PATH:-$DATA_DIR/coral_nv_hg_seed${SEED}_inspired_valid_top100.jsonl}"
TEST_PATH="${TEST_PATH:-$DATA_DIR/coral_nv_hg_seed${SEED}_inspired_test_top100.jsonl}"
REC_CHECKPOINT_PATH="${REC_CHECKPOINT_PATH:-$ROOT/../rec/checkpoints/inspired/nv_hg_seed${SEED}}"

OUT_DIR="${OUT_DIR:-$ROOT/outputs/inspired_rerank}"
CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/precompute_cache}"
RUN_NAME="${RUN_NAME:-inspired_nv_hg_seed${SEED}_lr${LR}_a0.15_b0.002_t0.50_d0.010_ep${EPOCHS}}"

mkdir -p "$OUT_DIR" "$CACHE_DIR"
cd "$ROOT"

for required_path in "$TRAIN_PATH" "$VALID_PATH" "$TEST_PATH" "$REC_CHECKPOINT_PATH"; do
  if [[ ! -e "$required_path" ]]; then
    echo "missing required path: $required_path" >&2
    exit 1
  fi
done

refresh_args=()
if [[ "$REFRESH_CACHE" == "1" ]]; then
  refresh_args=(--refresh-precompute-cache)
fi

"$PY" -u "$ROOT/train.py" \
  --mode grid-sweep \
  --seed "$SEED" \
  --device "$DEVICE" \
  --data-name inspired \
  --train-path "$TRAIN_PATH" \
  --valid-path "$VALID_PATH" \
  --test-path "$TEST_PATH" \
  --rec-checkpoint-path "$REC_CHECKPOINT_PATH" \
  --reranker-mode soft_deecho \
  --context-source conv \
  --deecho-warm-start-source custom \
  --deecho-init-alpha 0.15 \
  --deecho-init-beta 0.002 \
  --deecho-init-tau 0.50 \
  --deecho-init-delta 0.010 \
  --deecho-apply-mode full_list \
  --deecho-apply-temperature 0 \
  --precompute-cache-dir "$CACHE_DIR" \
  --precompute-cache-format compact \
  "${refresh_args[@]}" \
  --candidate-top-k 50 \
  --batch-size 8 \
  --selection-metrics Recall@10 Recall@5 Recall@1 NDCG@10 \
  --sweep-learning-rates "$LR" \
  --sweep-epochs "$EPOCHS" \
  --sweep-deecho-hidden-dims 64 \
  --sweep-deecho-dropouts 0.0 \
  --sweep-deecho-apply-temperatures 0 \
  --grid-results-path "$OUT_DIR/${RUN_NAME}.json" \
  --reranker-checkpoint-path "$OUT_DIR/${RUN_NAME}_best.pt"
