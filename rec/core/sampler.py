from abc import ABC, abstractmethod

import torch
from torch import Tensor


class NegativeSampler(ABC):
    @abstractmethod
    def sample(
        self,
        query_emb: Tensor,    # [N, D]
        item_emb: Tensor,     # [M, D]
        gt_mask: Tensor,      # [N, M]  bool
        n_negatives: int,
    ) -> Tensor:
        ...


class RandomNegativeSampler(NegativeSampler):
    def sample(
        self,
        query_emb: Tensor,
        item_emb: Tensor,
        gt_mask: Tensor,
        n_negatives: int,
    ) -> Tensor:
        N, M = gt_mask.shape
        neg_indices = torch.zeros(N, n_negatives, dtype=torch.long)
        for i in range(N):
            candidates = torch.where(~gt_mask[i])[0]
            perm = torch.randperm(len(candidates))[:n_negatives]
            neg_indices[i] = candidates[perm]
        return neg_indices


class HardNegativeSampler(NegativeSampler):
    def sample(
        self,
        query_emb: Tensor,
        item_emb: Tensor,
        gt_mask: Tensor,
        n_negatives: int,
    ) -> Tensor:
        scores = torch.matmul(query_emb, item_emb.T)           # [N, M]
        scores.masked_fill_(gt_mask, float('-inf'))
        probs = torch.softmax(scores, dim=-1)
        neg_indices = torch.multinomial(probs, num_samples=n_negatives, replacement=False)
        return neg_indices                                       # [N, n_negatives]


class MixedNegativeSampler(NegativeSampler):
    def __init__(self, n_hard: int = 4, skip_top: int = 0) -> None:
        self.n_hard = n_hard
        self.skip_top = skip_top

    def sample(
        self,
        query_emb: Tensor,
        item_emb: Tensor,
        gt_mask: Tensor,
        n_negatives: int,
    ) -> Tensor:
        N, M = gt_mask.shape
        n_hard = min(self.n_hard, n_negatives)
        n_rand = n_negatives - n_hard

        scores = torch.matmul(query_emb, item_emb.T)           # [N, M]
        scores_masked = scores.masked_fill(gt_mask, float('-inf'))
        _, top_idx = scores_masked.topk(self.skip_top + n_hard, dim=-1)  # [N, skip+n_hard]
        hard_idx = top_idx[:, self.skip_top:]                  # [N, n_hard]

        neg_indices = torch.zeros(N, n_negatives, dtype=torch.long)
        neg_indices[:, :n_hard] = hard_idx

        if n_rand > 0:
            hard_mask = torch.zeros(N, M, dtype=torch.bool, device=gt_mask.device)
            hard_mask.scatter_(1, hard_idx, True)
            excluded = gt_mask | hard_mask

            for i in range(N):
                candidates = torch.where(~excluded[i])[0]
                if len(candidates) >= n_rand:
                    perm = torch.randperm(len(candidates))[:n_rand]
                    neg_indices[i, n_hard:] = candidates[perm]
                else:
                    all_neg = torch.where(~gt_mask[i])[0]
                    perm = torch.randperm(len(all_neg))[:n_rand]
                    neg_indices[i, n_hard:] = all_neg[perm]

        return neg_indices                                       # [N, n_negatives]
