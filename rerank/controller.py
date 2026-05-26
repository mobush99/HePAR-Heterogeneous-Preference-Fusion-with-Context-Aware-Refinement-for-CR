from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch import Tensor


def _clamped_logit(value: float, eps: float = 1e-6) -> float:
    value = min(max(float(value), eps), 1.0 - eps)
    return math.log(value / (1.0 - value))


def masked_zscore_nonmentioned(score: Tensor, mentioned_mask: Tensor, eps: float = 1e-8) -> Tensor:
    mask = (mentioned_mask <= 0).to(dtype=score.dtype)
    count = mask.sum(dim=1, keepdim=True)
    mean = (score * mask).sum(dim=1, keepdim=True) / count.clamp_min(1.0)
    centered = (score - mean) * mask
    var = (centered * centered).sum(dim=1, keepdim=True) / count.clamp_min(1.0)
    z = centered / torch.sqrt(var.clamp_min(eps * eps))
    has_mentioned = (mentioned_mask > 0).any(dim=1, keepdim=True)
    has_nonmentioned = count > 0
    z = torch.where(has_mentioned & has_nonmentioned, z, torch.zeros_like(z))
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def score_weighted_mentioned_reference(
    base_score: Tensor,
    candidate_emb: Tensor,
    mentioned_mask: Tensor,
    tau: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    mentioned = mentioned_mask > 0
    tau = tau.to(device=base_score.device, dtype=base_score.dtype).reshape(-1, 1).clamp_min(eps)
    logits = (base_score / tau).masked_fill(~mentioned, -1e9)
    weights = torch.softmax(logits, dim=1) * mentioned.to(dtype=base_score.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(eps)
    weights = weights / denom
    ref = (candidate_emb * weights.unsqueeze(-1).to(dtype=candidate_emb.dtype)).sum(dim=1)
    has_mentioned = mentioned.any(dim=1, keepdim=True)
    return torch.where(has_mentioned, ref, torch.zeros_like(ref))


def project_out(value: Tensor, ref: Tensor, rho: float = 1.0, eps: float = 1e-8) -> Tensor:
    ref_norm_sq = (ref * ref).sum(dim=-1, keepdim=True)
    if value.dim() == 2:
        coeff = (value * ref).sum(dim=-1, keepdim=True) / (ref_norm_sq + eps)
        projection = coeff * ref
    elif value.dim() == 3:
        coeff = (value * ref.unsqueeze(1)).sum(dim=-1, keepdim=True) / (
            ref_norm_sq.unsqueeze(1) + eps
        )
        projection = coeff * ref.unsqueeze(1)
    else:
        raise ValueError(f"unsupported value shape: {tuple(value.shape)}")
    return value - float(rho) * projection


def _safe_cosine_pairwise(context: Tensor, candidates: Tensor, eps: float = 1e-8) -> Tensor:
    dot = (context.unsqueeze(1) * candidates).sum(dim=-1)
    context_norm = (context * context).sum(dim=-1, keepdim=True).clamp_min(eps * eps).rsqrt()
    candidate_norm = (candidates * candidates).sum(dim=-1).clamp_min(eps * eps).rsqrt()
    return dot * context_norm * candidate_norm


def near_tie_mask(s_soft: Tensor, mentioned_mask: Tensor, delta: Tensor) -> Tensor:
    non_mentioned = mentioned_mask <= 0
    delta = delta.to(device=s_soft.device, dtype=s_soft.dtype).reshape(-1, 1)
    top = s_soft.max(dim=1, keepdim=True).values
    return (((top - s_soft) <= delta) & non_mentioned).to(dtype=s_soft.dtype)


def near_tie_gate(
    s_soft: Tensor,
    mentioned_mask: Tensor,
    delta: Tensor,
    temperature: float = 0.0,
) -> Tensor:
    if float(temperature) <= 0.0:
        return near_tie_mask(s_soft, mentioned_mask, delta)
    non_mentioned = (mentioned_mask <= 0).to(dtype=s_soft.dtype)
    delta = delta.to(device=s_soft.device, dtype=s_soft.dtype).reshape(-1, 1)
    top = s_soft.max(dim=1, keepdim=True).values
    gap = top - s_soft
    gate = torch.sigmoid((delta - gap) / max(float(temperature), 1e-8))
    return gate * non_mentioned


def deecho_apply_gate(
    s_soft: Tensor,
    mentioned_mask: Tensor,
    delta: Tensor,
    mode: str = "near_tie",
    temperature: float = 0.0,
) -> Tensor:
    if mode == "near_tie":
        return near_tie_gate(s_soft, mentioned_mask, delta, temperature)
    if mode == "full_list":
        return (mentioned_mask <= 0).to(dtype=s_soft.dtype)
    if mode.startswith("top") and mode.endswith("_full_list"):
        try:
            top_k = int(mode[len("top") : -len("_full_list")])
        except ValueError as exc:
            raise ValueError(f"unsupported deecho apply mode: {mode}") from exc
        if top_k <= 0:
            raise ValueError(f"top-k full-list mode requires positive k: {mode}")
        non_mentioned = mentioned_mask <= 0
        order = torch.argsort(s_soft, dim=1, descending=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(1, s_soft.shape[1] + 1, device=s_soft.device).unsqueeze(0)
        ranks.scatter_(1, order, rank_values.expand_as(order))
        return ((ranks <= top_k) & non_mentioned).to(dtype=s_soft.dtype)
    raise ValueError(f"unsupported deecho apply mode: {mode}")


class DeechoControllerMLP(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        init_values: Sequence[float] = (0.05, 0.05, 0.5, 0.1),
        delta_radius: float = 0.002,
        delta_mapping: str = "bounded",
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive for DeechoControllerMLP")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive for DeechoControllerMLP")
        if len(init_values) != 4:
            raise ValueError("init_values must contain alpha, beta, tau, delta")
        if delta_radius < 0.0:
            raise ValueError("delta_radius must be non-negative for DeechoControllerMLP")
        if delta_mapping not in {"bounded", "sigmoid"}:
            raise ValueError("delta_mapping must be either 'bounded' or 'sigmoid'")
        self.delta_center = float(init_values[3])
        self.delta_radius = float(delta_radius)
        self.delta_mapping = delta_mapping
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            alpha, beta, tau, delta = [float(value) for value in init_values]
            bias = final.bias.new_tensor(
                [
                    _clamped_logit(alpha),
                    _clamped_logit(beta),
                    _clamped_logit(tau / 2.0),
                    0.0 if self.delta_mapping == "bounded" else _clamped_logit(delta),
                ]
            )
            with torch.no_grad():
                final.bias.copy_(bias)

    def forward(self, context_emb: Tensor) -> Dict[str, Tensor]:
        raw = self.net(context_emb)
        sigmoid = torch.sigmoid(raw)
        if self.delta_mapping == "bounded":
            delta = self.delta_center + self.delta_radius * torch.tanh(raw[:, 3:4])
        else:
            delta = sigmoid[:, 3:4]
        return {
            "raw": raw,
            "alpha": sigmoid[:, 0:1],
            "beta": sigmoid[:, 1:2],
            "tau": 2.0 * sigmoid[:, 2:3],
            "delta": delta.clamp_min(1e-8),
        }


def deecho_controller_score(
    base_score: Tensor,
    candidate_emb: Tensor,
    context_emb: Tensor,
    mentioned_mask: Tensor,
    alpha: Tensor,
    beta: Tensor,
    tau: Tensor,
    delta: Tensor,
    apply_mode: str = "near_tie",
    apply_temperature: float = 0.0,
    eps: float = 1e-8,
) -> Dict[str, Tensor]:
    base_score = base_score.to(device=candidate_emb.device, dtype=candidate_emb.dtype)
    mentioned_mask = mentioned_mask.to(device=candidate_emb.device, dtype=candidate_emb.dtype)
    context_emb = context_emb.to(device=candidate_emb.device, dtype=candidate_emb.dtype)
    alpha = alpha.to(device=candidate_emb.device, dtype=candidate_emb.dtype)
    beta = beta.to(device=candidate_emb.device, dtype=candidate_emb.dtype)
    tau = tau.to(device=candidate_emb.device, dtype=candidate_emb.dtype)
    delta = delta.to(device=candidate_emb.device, dtype=candidate_emb.dtype)

    ref = score_weighted_mentioned_reference(base_score, candidate_emb, mentioned_mask, tau, eps)
    context_deecho = project_out(context_emb, ref, rho=1.0, eps=eps)
    candidate_deecho = project_out(candidate_emb, ref, rho=1.0, eps=eps)
    deecho = _safe_cosine_pairwise(context_deecho, candidate_deecho, eps=eps)
    has_mentioned = (mentioned_mask > 0).any(dim=1, keepdim=True)
    deecho = torch.where(has_mentioned, deecho, torch.zeros_like(deecho))
    z_deecho = masked_zscore_nonmentioned(deecho, mentioned_mask, eps)
    s_soft = base_score - alpha.reshape(-1, 1) * mentioned_mask
    mask = deecho_apply_gate(s_soft, mentioned_mask, delta, apply_mode, apply_temperature)
    final = s_soft + beta.reshape(-1, 1) * mask * z_deecho
    return {
        "final_score": torch.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0),
        "s_soft": s_soft,
        "reference_emb": ref,
        "deecho_score": deecho,
        "z_deecho": z_deecho,
        "apply_mask": mask,
    }


def controller_stats(params: Dict[str, Tensor]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for name in ("alpha", "beta", "tau", "delta"):
        value = params[name].detach().float().reshape(-1).cpu()
        stats[f"{name}_min"] = float(value.min().item())
        stats[f"{name}_mean"] = float(value.mean().item())
        stats[f"{name}_max"] = float(value.max().item())
    return stats
