from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from deecho_controller import DeechoControllerMLP, controller_stats, deecho_controller_score
from loss import PairwiseRankingLoss


def _clamped_logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class SoftDeechoReranker(nn.Module):
    def __init__(
        self,
        *,
        deecho_context_dim: int,
        deecho_hidden_dim: int = 64,
        deecho_dropout: float = 0.1,
        deecho_init_alpha: float = 0.05,
        deecho_init_beta: float = 0.05,
        deecho_init_tau: float = 0.5,
        deecho_init_delta: float = 0.1,
        deecho_delta_radius: float = 0.002,
        deecho_delta_mapping: str = "bounded",
        deecho_param_mode: str = "adaptive_mlp",
        deecho_apply_mode: str = "near_tie",
        deecho_apply_temperature: float = 0.0,
        deecho_force_alpha: Optional[float] = None,
        deecho_force_beta: Optional[float] = None,
    ) -> None:
        super().__init__()
        if deecho_context_dim <= 0:
            raise ValueError("deecho_context_dim must be positive")
        if deecho_delta_radius < 0.0:
            raise ValueError("deecho_delta_radius must be non-negative")
        if deecho_delta_mapping not in {"bounded", "sigmoid"}:
            raise ValueError("deecho_delta_mapping must be either bounded or sigmoid")
        if deecho_param_mode not in {"adaptive_mlp", "global_learnable", "fixed"}:
            raise ValueError("deecho_param_mode must be adaptive_mlp, global_learnable, or fixed")
        if deecho_apply_mode not in {"near_tie", "full_list"} and not (
            deecho_apply_mode.startswith("top") and deecho_apply_mode.endswith("_full_list")
        ):
            raise ValueError(f"unsupported deecho apply mode: {deecho_apply_mode}")

        self.reranker_mode = "soft_deecho"
        self.requires_context_emb = True
        self.deecho_context_dim = int(deecho_context_dim)
        self.deecho_hidden_dim = int(deecho_hidden_dim)
        self.deecho_dropout = float(deecho_dropout)
        self.deecho_init_values = (
            float(deecho_init_alpha),
            float(deecho_init_beta),
            float(deecho_init_tau),
            float(deecho_init_delta),
        )
        self.deecho_delta_radius = float(deecho_delta_radius)
        self.deecho_delta_mapping = deecho_delta_mapping
        self.deecho_param_mode = deecho_param_mode
        self.deecho_apply_mode = deecho_apply_mode
        self.deecho_apply_temperature = float(deecho_apply_temperature)
        self.deecho_force_alpha = deecho_force_alpha
        self.deecho_force_beta = deecho_force_beta

        self.deecho_controller = DeechoControllerMLP(
            context_dim=self.deecho_context_dim,
            hidden_dim=self.deecho_hidden_dim,
            dropout=self.deecho_dropout,
            init_values=self.deecho_init_values,
            delta_radius=self.deecho_delta_radius,
            delta_mapping=self.deecho_delta_mapping,
        )
        raw_alpha = _clamped_logit(deecho_init_alpha)
        raw_beta = _clamped_logit(deecho_init_beta)
        raw_tau = _clamped_logit(float(deecho_init_tau) / 2.0)
        raw_delta = 0.0 if self.deecho_delta_mapping == "bounded" else _clamped_logit(deecho_init_delta)
        self.global_deecho_raw = nn.Parameter(
            torch.tensor([raw_alpha, raw_beta, raw_tau, raw_delta], dtype=torch.float32),
            requires_grad=self.deecho_param_mode in {"global_learnable", "fixed"},
        )
        self.loss_fn = PairwiseRankingLoss()
        self.last_loss_components: Dict[str, float] = {}

    def _global_or_fixed_params(self, batch_size: int, ref: Tensor) -> Dict[str, Tensor]:
        raw = self.global_deecho_raw.to(device=ref.device, dtype=ref.dtype)
        if self.deecho_param_mode == "fixed":
            raw = raw.detach()
        sigmoid = torch.sigmoid(raw)
        if self.deecho_delta_mapping == "bounded":
            delta_value = self.deecho_init_values[3] + self.deecho_delta_radius * torch.tanh(raw[3])
        else:
            delta_value = sigmoid[3]
        return {
            "raw": raw.reshape(1, 4).expand(batch_size, 4),
            "alpha": sigmoid[0].expand(batch_size, 1),
            "beta": sigmoid[1].expand(batch_size, 1),
            "tau": (2.0 * sigmoid[2]).expand(batch_size, 1),
            "delta": delta_value.clamp_min(1e-8).expand(batch_size, 1),
        }

    @staticmethod
    def _validate_inputs(
        base_score: Tensor,
        candidate_emb: Tensor,
        mentioned_mask: Optional[Tensor],
        context_emb: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if base_score.ndim != 2:
            raise ValueError("base_score must have shape [B, K]")
        if candidate_emb.ndim != 3:
            raise ValueError("candidate_emb must have shape [B, K, D]")
        if base_score.shape != candidate_emb.shape[:2]:
            raise ValueError("base_score and candidate_emb batch/candidate dimensions must match")
        if mentioned_mask is None:
            raise ValueError("mentioned_mask is required")
        if context_emb is None:
            raise ValueError("context_emb is required")
        if mentioned_mask.shape != base_score.shape:
            raise ValueError("mentioned_mask must have shape [B, K]")
        if context_emb.shape[0] != base_score.shape[0] or context_emb.ndim != 2:
            raise ValueError("context_emb must have shape [B, D]")
        return mentioned_mask, context_emb

    def forward(
        self,
        base_score: Tensor,
        candidate_emb: Tensor,
        *,
        mentioned_mask: Optional[Tensor] = None,
        context_emb: Optional[Tensor] = None,
        **_: Any,
    ) -> Dict[str, Tensor | Dict[str, Tensor] | Dict[str, float] | bool | str]:
        mentioned_mask, context_emb = self._validate_inputs(
            base_score, candidate_emb, mentioned_mask, context_emb
        )
        base_score = base_score.to(device=candidate_emb.device, dtype=candidate_emb.dtype)
        mentioned_mask = mentioned_mask.to(device=candidate_emb.device, dtype=candidate_emb.dtype)
        context_emb = context_emb.to(device=candidate_emb.device, dtype=candidate_emb.dtype)

        if self.deecho_param_mode == "adaptive_mlp":
            params = self.deecho_controller(context_emb)
        else:
            params = self._global_or_fixed_params(base_score.shape[0], base_score)
        if self.deecho_force_alpha is not None:
            params["alpha"] = torch.full_like(params["alpha"], float(self.deecho_force_alpha))
        if self.deecho_force_beta is not None:
            params["beta"] = torch.full_like(params["beta"], float(self.deecho_force_beta))

        score_debug = deecho_controller_score(
            base_score=base_score,
            candidate_emb=candidate_emb,
            context_emb=context_emb,
            mentioned_mask=mentioned_mask,
            alpha=params["alpha"],
            beta=params["beta"],
            tau=params["tau"],
            delta=params["delta"],
            apply_mode=self.deecho_apply_mode,
            apply_temperature=self.deecho_apply_temperature,
        )
        final_score = score_debug["final_score"]
        if self.deecho_param_mode == "fixed":
            final_score = final_score + 0.0 * self.global_deecho_raw.sum().to(
                device=final_score.device,
                dtype=final_score.dtype,
            )
        zero = candidate_emb.new_zeros(())
        return {
            "final_score": final_score,
            "calibrated_score": final_score,
            "score_for_loss": final_score,
            "soft_score": score_debug["s_soft"],
            "residual_signal": score_debug["z_deecho"],
            "residual_score": final_score - score_debug["s_soft"],
            "mentioned_mask": mentioned_mask,
            "ranking": torch.argsort(final_score, dim=-1, descending=True),
            "controller_enabled": True,
            "deecho_params": params,
            "deecho_controller_stats": controller_stats(params),
            "deecho_s_soft": score_debug["s_soft"],
            "deecho_score": score_debug["deecho_score"],
            "z_deecho": score_debug["z_deecho"],
            "apply_mask": score_debug["apply_mask"],
            "deecho_apply_mode": self.deecho_apply_mode,
            "deecho_apply_temperature": self.deecho_apply_temperature,
            "zero": zero,
            "reranker_mode": self.reranker_mode,
        }

    def compute_loss(self, final_score: Tensor, label: Tensor) -> Tensor:
        loss = self.loss_fn(final_score, label)
        self.last_loss_components = {
            "lambda_loss": float(loss.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
        }
        return loss
