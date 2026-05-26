export TOKENIZERS_PARALLELISM=false
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

for seed in 2024; do
  python "$ROOT/train.py" \
    --seed "$seed" \
    --data_name inspired \
    --query_used_info c l d x y \
    --doc_used_info m \
    --item_max_length 128 \
    --base_model_name nvidia/NV-Embed-v2 \
    --mode hypergraph \
    --no_global_edge \
    --flat_gating \
    --n2e_routing_iters 4 \
    --alpha 0.5 \
    --beta 0.2 \
    --delta 0.1 \
    --epsilon 0.1 \
    --epochs 100 \
    --batch_size 10 \
    --accumulation_steps 8 \
    --sampler random \
    --n_negatives 16 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --temperature 0.05 \
    --patience 4 \
    --bf16 \
    --ckpt_dir checkpoints \
    --device cuda
done
