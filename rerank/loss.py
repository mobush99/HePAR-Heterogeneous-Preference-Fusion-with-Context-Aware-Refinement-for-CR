from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def lambda_loss(final_score: Tensor, label: Tensor, eps: float = 1e-8) -> Tensor:
    if final_score.shape != label.shape:
        raise ValueError("final_score and label must have the same shape [B, K]")
    if final_score.ndim != 2:
        raise ValueError("final_score and label must have shape [B, K]")

    scores = final_score
    labels = label.to(device=scores.device, dtype=scores.dtype)
    batch_size, num_candidates = scores.shape

    score_i = scores.unsqueeze(2)
    score_j = scores.unsqueeze(1)
    label_i = labels.unsqueeze(2)
    label_j = labels.unsqueeze(1)
    valid_pair = label_i > label_j
    pair_loss = F.softplus(-(score_i - score_j))

    with torch.no_grad():
        gain = torch.pow(2.0, labels) - 1.0
        order = torch.argsort(scores, dim=-1, descending=True)
        positions = torch.arange(
            1,
            num_candidates + 1,
            device=scores.device,
            dtype=scores.dtype,
        ).expand(batch_size, -1)
        rank = torch.empty_like(positions)
        rank.scatter_(1, order, positions)
        discount = 1.0 / torch.log2(rank + 1.0)

        ideal_label = torch.sort(labels, dim=-1, descending=True).values
        ideal_gain = torch.pow(2.0, ideal_label) - 1.0
        ideal_discount = 1.0 / torch.log2(positions + 1.0)
        idcg = (ideal_gain * ideal_discount).sum(dim=-1)
        delta_ndcg = (
            torch.abs(
                (gain.unsqueeze(2) - gain.unsqueeze(1))
                * (discount.unsqueeze(2) - discount.unsqueeze(1))
            )
            / (idcg[:, None, None] + eps)
        )
        pair_weight = delta_ndcg * valid_pair.to(delta_ndcg.dtype)

    weighted = pair_loss * pair_weight
    pair_count = valid_pair.sum(dim=(1, 2))
    sample_sum = weighted.sum(dim=(1, 2))
    sample_loss = torch.where(
        pair_count > 0,
        sample_sum / pair_count.clamp_min(1).to(sample_sum.dtype),
        torch.zeros_like(sample_sum),
    )
    return sample_loss.mean()


class PairwiseRankingLoss(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, final_score: Tensor, label: Tensor) -> Tensor:
        return lambda_loss(final_score, label, eps=self.eps)
