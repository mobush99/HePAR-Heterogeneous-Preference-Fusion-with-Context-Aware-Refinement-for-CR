# HePAR
### : Heterogeneous Preference Fusion with Context-Aware Refinement for Conversational Recommendation

## Overview
<img src=img/overview.png>
HePAR is organized around two components:

- **Preference fusion backbone** (`rec/`): encodes heterogeneous conversational recommendation signals with either static weighting, dynamic weighting, or hypergraph preference fusion.
- **Context-aware refinement** (`rerank/`): applies a context-aware refinement module over backbone candidates using base scores, candidate embeddings, conversation/user context embeddings, and mentioned-item masks.

The repository includes code paths for the ReDial and INSPIRED conversational recommendation datasets.

## Environment
The code was developed with Python 3.10 and PyTorch. Install the provided dependency set from the `rec/` directory:
```bash
cd rec
pip install -r requirements.txt
```
The default scripts use `nvidia/NV-Embed-v2` from Hugging Face and expect CUDA for practical training. Set `TOKENIZERS_PARALLELISM=false` if your environment does not already do so.

## Train the Recommendation Backbone
Run the provided scripts from the repository root or directly from `rec/`.

For ReDial:
```bash
bash rec/run_redial.sh
```
For INSPIRED:
```bash
bash rec/run_inspired.sh
```

The scripts train with `nvidia/NV-Embed-v2`, random negative sampling, and the paper configuration for each dataset. Checkpoints are written under:
```text
rec/checkpoints/<dataset>/<run_name>/
```

The main backbone modes (for ablation)
- `vanilla`: static weighted fusion over signals.
- `dynamic_gating`: sample-wise dynamic weighting over extracted signals.
- `hypergraph`: hypergraph preference fusion with node-to-edge and edge-to-node message passing.

## Train the Context-Aware Reranker

The reranker trains on frozen recommendation-backbone candidates. 
It loads the trained rec checkpoint through `RecEmbeddingProvider`, computes candidate and context embeddings, and caches precomputed batches under `rerank/outputs/precompute_cache/`.

For ReDial:
```bash
bash rerank/train_redial.sh
```
For INSPIRED:
```bash
bash rerank/train_inspired.sh
```

## Evaluation
The training scripts report standard ranking metrics including:
- `Recall@1`
- `Recall@5`
- `Recall@10`
- `Recall@50`
- `NDCG@5`
- `NDCG@10`
- `NDCG@50`

Backbone evaluation is handled in `rec/core/evaluator.py`. Reranker evaluation compares backbone and reranked metrics in `rerank/trainer.py`.
