import math
from typing import Dict, List, Union

import torch
from torch import Tensor


class MetricComputer:
    def __init__(self, cutoffs: List[int]) -> None:
        self.cutoffs = sorted(cutoffs)
        self.max_k = max(cutoffs)

    def compute(
        self,
        scores: Tensor,
        gt_ids: Union[List[int], List[List[int]]],
    ) -> Dict[str, float]:
        if isinstance(gt_ids[0], int):
            gt_ids = [[g] for g in gt_ids]

        ranked = torch.argsort(scores, dim=-1, descending=True)[:, :self.max_k]  # [N, max_k]

        results = {}
        for k in self.cutoffs:
            recalls, ndcgs = [], []
            for i, gt in enumerate(gt_ids):
                recalls.append(self._recall_at_k(ranked[i], gt, k))
                ndcgs.append(self._ndcg_at_k(ranked[i], gt, k))
            results[f"Recall@{k}"] = sum(recalls) / len(recalls)
            if k > 1:
                results[f"NDCG@{k}"] = sum(ndcgs) / len(ndcgs)

        return results

    def _recall_at_k(self, ranked: Tensor, gt: List[int], k: int) -> float:
        gt_set = set(gt)
        top_k = set(ranked[:k].tolist())
        return len(gt_set & top_k) / len(gt_set)

    def _ndcg_at_k(self, ranked: Tensor, gt: List[int], k: int) -> float:
        gt_set = set(gt)
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, item in enumerate(ranked[:k].tolist())
            if item in gt_set
        )
        n_relevant = min(len(gt_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
        return dcg / idcg if idcg > 0 else 0.0
