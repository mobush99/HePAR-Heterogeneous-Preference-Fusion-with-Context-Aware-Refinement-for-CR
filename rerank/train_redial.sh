#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

SEED="${SEED:-2024}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-15}"
LR="${LR:-2e-6}"
REFRESH_CACHE="${REFRESH_CACHE:-0}"

DATA_DIR="${DATA_DIR:-$ROOT/data/prep_redial_nvembed_hg_seeded}"
TRAIN_PATH="${TRAIN_PATH:-$DATA_DIR/coral_nvembed_hg_seed${SEED}_redial_train_top50.jsonl}"
VALID_PATH="${VALID_PATH:-$DATA_DIR/coral_nvembed_hg_seed${SEED}_redial_valid_top50.jsonl}"
TEST_PATH="${TEST_PATH:-$DATA_DIR/coral_nvembed_hg_seed${SEED}_redial_test_top50.jsonl}"
REC_CHECKPOINT_PATH="${REC_CHECKPOINT_PATH:-$ROOT/../rec/checkpoints/redial/nvembed_hg_seed${SEED}}"

OUT_DIR="${OUT_DIR:-$ROOT/outputs/redial_rerank}"
CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/precompute_cache}"
RUN_NAME="${RUN_NAME:-redial_nvembed_hg_b030_seed${SEED}_lr${LR}_a0.25_b0.30_t0.15_d0.010_ep${EPOCHS}}"

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
  --data-name redial \
  --train-path "$TRAIN_PATH" \
  --valid-path "$VALID_PATH" \
  --test-path "$TEST_PATH" \
  --rec-checkpoint-path "$REC_CHECKPOINT_PATH" \
  --base-model-name nvidia/NV-Embed-v2 \
  --rec-mode hypergraph \
  --query-used-info c l d x y \
  --doc-used-info m \
  --item-max-length 128 \
  --n2e-routing-iters 4 \
  --alpha 0.5 \
  --beta 0.2 \
  --delta 0.1 \
  --epsilon 0.1 \
  --reranker-mode soft_deecho \
  --context-source conv \
  --deecho-warm-start-source custom \
  --deecho-init-alpha 0.25 \
  --deecho-init-beta 0.30 \
  --deecho-init-tau 0.15 \
  --deecho-init-delta 0.010 \
  --deecho-apply-mode near_tie \
  --deecho-apply-temperature 0 \
  --precompute-cache-dir "$CACHE_DIR" \
  --precompute-cache-format compact \
  "${refresh_args[@]}" \
  --candidate-top-k 50 \
  --batch-size 8 \
  --selection-metrics Recall@1 Recall@5 NDCG@10 \
  --sweep-learning-rates "$LR" \
  --sweep-epochs "$EPOCHS" \
  --sweep-deecho-hidden-dims 64 \
  --sweep-deecho-dropouts 0.0 \
  --sweep-deecho-apply-temperatures 0 \
  --grid-results-path "$OUT_DIR/${RUN_NAME}.json" \
  --reranker-checkpoint-path "$OUT_DIR/${RUN_NAME}_best.pt"
