from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import BaseEncoder
from .metric import MetricComputer
from .logger import Logger


class Evaluator:
    def __init__(
        self,
        model: BaseEncoder,
        query_dataloader: DataLoader,
        item_dataloader: DataLoader,
        metric_computer: MetricComputer,
        logger: Logger,
        bf16: bool = False,
        temperature: float = 0.05,
    ) -> None:
        self.model = model
        self.query_dataloader = query_dataloader
        self.item_dataloader = item_dataloader
        self.metric_computer = metric_computer
        self.logger = logger
        self.bf16 = bf16
        self.temperature = temperature

    @torch.no_grad()
    def encode_items(self, device: str) -> Tensor:
        self.model.eval()
        self.model.to(device)
        embeddings = []
        with autocast(device_type=device, dtype=torch.bfloat16, enabled=self.bf16):
            for batch in tqdm(self.item_dataloader, desc="  [Eval] Encoding items", leave=False):
                batch = {k: v.to(device) if isinstance(v, Tensor) else v
                         for k, v in batch.items()}
                emb = self.model.encode_item(batch)   # [B, D]
                embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def evaluate(
        self,
        gt_ids: Union[List[int], List[List[int]]],
        device: str,
        split: str = "val",
    ) -> Dict[str, float]:
        item_emb = self.encode_items(device).to(device)   # [M, D]

        self.model.eval()
        all_scores = []
        with autocast(device_type=device, dtype=torch.bfloat16, enabled=self.bf16):
            for batch in tqdm(self.query_dataloader, desc=f"  [{split}] Scoring", leave=False):
                _, _, input_dict = batch
                input_dict = {k: v.to(device) if isinstance(v, Tensor) else v
                              for k, v in input_dict.items()}
                query_emb = self.model.encode_query(input_dict)   # [B, D]
                scores = query_emb @ item_emb.T                   # [B, M]

                all_scores.append(scores.cpu())

        all_scores = torch.cat(all_scores, dim=0)                 # [N, M]

        val_loss = self._compute_full_ranking_loss(all_scores, gt_ids)
        metrics = self.metric_computer.compute(all_scores, gt_ids)
        metrics["loss"] = val_loss

        cutoffs = sorted(self.metric_computer.cutoffs)
        recall_str = "  ".join(f"@{k}={metrics.get(f'Recall@{k}', 0.0):.4f}" for k in cutoffs)
        ndcg_str   = "  ".join(f"@{k}={metrics.get(f'NDCG@{k}',   0.0):.4f}" for k in cutoffs if k > 1)
        print(f"\n  [{split}] loss={val_loss:.4f}", flush=True)
        print(f"  Recall : {recall_str}", flush=True)
        print(f"  NDCG   : {ndcg_str}",   flush=True)

        return metrics

    def _compute_full_ranking_loss(
        self,
        all_scores: Tensor,                              # [N, M]  CPU, float
        gt_ids: Union[List[int], List[List[int]]],
    ) -> float:

        scores_scaled = all_scores.float() / self.temperature   # [N, M]

        if isinstance(gt_ids[0], int):
            labels = torch.tensor(gt_ids, dtype=torch.long)    # [N]
            return F.cross_entropy(scores_scaled, labels).item()
        else:
            losses = []
            for i, gts in enumerate(gt_ids):
                for g in gts:
                    losses.append(
                        F.cross_entropy(
                            scores_scaled[i].unsqueeze(0),
                            torch.tensor([g], dtype=torch.long),
                        )
                    )
            return torch.stack(losses).mean().item()
