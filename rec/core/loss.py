from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class RetrievalLoss(ABC):
    @abstractmethod
    def __call__(
        self,
        query_emb: Tensor,   # [B, D]
        item_emb: Tensor,    # [B, 1+n_neg, D]
        labels: Tensor,      # [B]
    ) -> Tensor:
        ...


class InfoNCELoss(RetrievalLoss):

    def __init__(self, temperature: float = 0.05) -> None:
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def __call__(
        self,
        query_emb: Tensor,
        item_emb: Tensor,
        labels: Tensor,
    ) -> Tensor:
        # [B, 1, D] @ [B, D, 1+n_neg] → [B, 1+n_neg]
        scores = torch.bmm(query_emb.unsqueeze(1), item_emb.permute(0, 2, 1)).squeeze(1)
        scores = scores / self.temperature
        return self.ce(scores, labels)
