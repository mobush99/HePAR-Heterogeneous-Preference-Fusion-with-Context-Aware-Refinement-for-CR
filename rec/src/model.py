"""
Source Code for
HePAR: Heterogeneous Preference fusion with context-Aware Refinement
"""

import math
import os
import inspect
from types import MethodType
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
from safetensors.torch import load_file as load_safetensors

from core.model import BaseEncoder


class DynamicGatingNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_signals: int = 5,
        init_biases: Optional[list] = None,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.inter_size = hidden_size // 4
        self.final_size = self.inter_size // 2
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * n_signals, self.inter_size),
            nn.LayerNorm(self.inter_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.inter_size, self.final_size),
            nn.ReLU(),
            nn.Linear(self.final_size, n_signals),
        )
        if init_biases is not None:
            self.mlp[-1].bias.data = torch.tensor(init_biases, dtype=torch.float32)
        else:
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.mlp[0].weight.dtype)
        return torch.sigmoid(self.mlp(x))


class HypergraphGatingNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout_prob: float = 0.1,
        gate_activation: str = "sigmoid",
        init_biases: Optional[list] = None,
        hg_only: bool = False,
    ):
        super().__init__()
        self.gate_activation = gate_activation
        self.hg_only = hg_only
        inter = hidden_size // 4
        final = inter // 2
        self.hg_norm = nn.LayerNorm(hidden_size)

        if not hg_only:
            self.raw_mlp = nn.Sequential(
                nn.Linear(hidden_size * 5, inter),
                nn.LayerNorm(inter), nn.ReLU(), nn.Dropout(dropout_prob),
                nn.Linear(inter, final), nn.ReLU(),
                nn.Linear(final, 5),
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_size * 5, inter),
                nn.LayerNorm(inter), nn.ReLU(), nn.Dropout(dropout_prob),
                nn.Linear(inter, final), nn.ReLU(),
                nn.Linear(final, 1),
            )

        self.hg_mlp = nn.Sequential(
            nn.Linear(hidden_size * 4, inter),
            nn.LayerNorm(inter), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(inter, final), nn.ReLU(),
            nn.Linear(final, 4),
        )

        with torch.no_grad():
            if init_biases is not None and gate_activation == "sigmoid":
                if not hg_only:
                    self.raw_mlp[-1].bias.data = torch.tensor(init_biases[:5], dtype=torch.float32)
                self.hg_mlp[-1].bias.data = torch.tensor(init_biases[5:9], dtype=torch.float32)
            else:
                if not hg_only:
                    nn.init.zeros_(self.raw_mlp[-1].bias)
                nn.init.zeros_(self.hg_mlp[-1].bias)

            if not hg_only:
                for m in self.gate_mlp.modules():
                    if isinstance(m, nn.Linear):
                        if m.out_features == 1:
                            nn.init.normal_(m.weight, mean=0, std=0.01)
                            nn.init.constant_(m.bias, 0.0)  # sigmoid(0)=0.5
                        else:
                            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                            nn.init.zeros_(m.bias)

    def forward(self, raw_conv, raw_like, raw_long, raw_short, raw_dislike,
                hg_conv, hg_like, hg_long, hg_short):
        hg_inputs = [self.hg_norm(x) for x in [hg_conv, hg_like, hg_long, hg_short]]
        hg_features = torch.cat(hg_inputs, dim=-1)     # [B, 4D]

        if self.gate_activation == "sigmoid":
            w_hg = torch.sigmoid(self.hg_mlp(hg_features))      # [B, 4]
        else:
            w_hg = F.softmax(self.hg_mlp(hg_features), dim=-1)

        if self.hg_only:
            B = w_hg.shape[0]
            w_raw = torch.zeros(B, 5, device=w_hg.device, dtype=w_hg.dtype)
            w = torch.cat([w_raw, w_hg], dim=-1)                 # [B, 9]
            split_ratio = torch.ones(B, 1, device=w_hg.device, dtype=w_hg.dtype)
            return w, split_ratio

        raw_inputs = [self.hg_norm(x) for x in [raw_conv, raw_like, raw_long, raw_short, raw_dislike]]
        raw_features = torch.cat(raw_inputs, dim=-1)   # [B, 5D]

        if self.gate_activation == "sigmoid":
            w_raw = torch.sigmoid(self.raw_mlp(raw_features))   # [B, 5]
        else:
            w_raw = F.softmax(self.raw_mlp(raw_features), dim=-1)
        w = torch.cat([w_raw, w_hg], dim=-1)                     # [B, 9]

        split_ratio = torch.sigmoid(self.gate_mlp(raw_features))  # [B, 1]

        return w, split_ratio


class Hypergraph(nn.Module):
    _EDGE_MEMBERS_FULL = [
        [0, 1, 2, 3],   # edge 0: (c, l, a, b)  4-way  — global
        [0, 1],          # edge 1: (c, l)
        [0, 2],          # edge 2: (c, a)
        [0, 3],          # edge 3: (c, b)
        [1, 2],          # edge 4: (l, a)
        [1, 3],          # edge 5: (l, b)
        [2, 3],          # edge 6: (a, b)
        [1, 2, 3],       # edge 7: (l, a, b)
    ]
    _EDGE_MEMBERS_NO_GLOBAL = [
        [0, 1],          # edge 0: (c, l)
        [0, 2],          # edge 1: (c, a)
        [0, 3],          # edge 2: (c, b)
        [1, 2],          # edge 3: (l, a)
        [1, 3],          # edge 4: (l, b)
        [2, 3],          # edge 5: (a, b)
        [1, 2, 3],       # edge 6: (l, a, b)
    ]

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: Optional[int] = None,
        edge_dropout_prob: float = 0.3,
        update_dropout_prob: float = 0.3,
        num_nodes: int = 4,
        num_heads: int = 4,
        num_routing_iters: int = 2,
        use_n2n: bool = True,
        use_global_edge: bool = True,
        use_residual_skip: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_residual_skip = use_residual_skip
        self.bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else hidden_size // 4
        self.num_nodes = num_nodes
        edge_members = self._EDGE_MEMBERS_FULL if use_global_edge else self._EDGE_MEMBERS_NO_GLOBAL
        self.num_edges = len(edge_members)
        self.num_heads = num_heads
        self.head_dim = self.bottleneck_dim // num_heads
        self.edge_dropout_prob = edge_dropout_prob
        self.num_routing_iters = num_routing_iters
        self.use_n2n = use_n2n
        self.use_global_edge = use_global_edge

        # edge_mask[n, e] = 1.0  iff  node n ∈ edge e
        edge_mask = torch.zeros(num_nodes, self.num_edges)
        for e, members in enumerate(edge_members):
            edge_mask[members, e] = 1.0
        self.register_buffer("edge_mask", edge_mask)

        b = self.bottleneck_dim

        # Projection
        self.proj_down = nn.Linear(hidden_size, b)
        self.proj_up = nn.Linear(b, hidden_size, bias=False)
        self.output_norm = nn.LayerNorm(hidden_size)

        # N2N Attention (per-node)
        if self.use_n2n:
            self.n2n_q_per = nn.ModuleList([nn.Linear(b, b) for _ in range(num_nodes)])
            self.n2n_k_per = nn.ModuleList([nn.Linear(b, b) for _ in range(num_nodes)])
            self.n2n_v_per = nn.ModuleList([nn.Linear(b, b) for _ in range(num_nodes)])
            self.n2n_pre_norm = nn.LayerNorm(b)
        self.n2n_norm = nn.LayerNorm(b)

        # N2E Learnable Aggregation (node → edge)
        self.n2e_pre_norm = nn.LayerNorm(b)
        self.n2e_seeds = nn.Parameter(torch.randn(self.num_edges, b))     # [E, b]
        self.n2e_q = nn.Linear(b, b)
        self.n2e_k = nn.Linear(b, b)
        self.n2e_v = nn.Linear(b, b)
        self.n2e_norm = nn.LayerNorm(b)

        # Edge-type embedding
        self.edge_type_emb = nn.Parameter(torch.zeros(self.num_edges, b))

        # Per-Edge MLP
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(b, b), nn.GELU(), nn.Linear(b, b))
            for _ in range(self.num_edges)
        ])
        self.edge_mlp_norm = nn.LayerNorm(b)

        # E2N Attention (node ← edge cross-attention)
        self.e2n_pre_norm_q = nn.LayerNorm(b)
        self.e2n_pre_norm_kv = nn.LayerNorm(b)
        self.e2n_q = nn.Linear(b, b)
        self.e2n_k = nn.Linear(b, b)
        self.e2n_v = nn.Linear(b, b)
        self.e2n_out = nn.Linear(b, b)
        self.e2n_norm = nn.LayerNorm(b)
        self.update_dropout = nn.Dropout(update_dropout_prob)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            nn.init.normal_(self.n2e_seeds, std=1.0)    # [E, b]
            nn.init.normal_(self.edge_type_emb, std=0.1)   # [E, b]

    def edge_dropout_mask(self, edge_mask: torch.Tensor, dropout_prob: float, training: bool):
        if (not training) or dropout_prob <= 0:
            return edge_mask
        N, E = edge_mask.shape
        keep = (torch.rand(E, device=edge_mask.device) > dropout_prob).float()
        if self.use_global_edge:
            keep[0] = 1.0
        return edge_mask * keep.unsqueeze(0)

    def forward(self, conv, like, long, short) -> dict:
        B = conv.shape[0]
        N, E = self.num_nodes, self.num_edges
        H, d_k = self.num_heads, self.head_dim
        b = self.bottleneck_dim

        # Step 1: Projection
        nodes_full = torch.stack([conv, like, long, short], dim=1)   # [B, N, D]
        nodes = self.proj_down(nodes_full)                            # [B, N, b]

        active_mask = self.edge_dropout_mask(
            self.edge_mask, self.edge_dropout_prob, self.training
        )   # [N, E]

        # Step 1.5: N2N Attention (per-node projection)
        if self.use_n2n:
            nodes_normed = self.n2n_pre_norm(nodes)                              # [B, N, b] Pre-LN
            Q = torch.stack([self.n2n_q_per[i](nodes_normed[:, i]) for i in range(N)], dim=1)  # [B, N, b]
            K = torch.stack([self.n2n_k_per[i](nodes_normed[:, i]) for i in range(N)], dim=1)  # [B, N, b]
            V = torch.stack([self.n2n_v_per[i](nodes_normed[:, i]) for i in range(N)], dim=1)  # [B, N, b]

            Q = Q.view(B, N, H, d_k).transpose(1, 2)                           # [B, H, N, d_k]
            K = K.view(B, N, H, d_k).transpose(1, 2)                           # [B, H, N, d_k]
            V = V.view(B, N, H, d_k).transpose(1, 2)                           # [B, H, N, d_k]

            n2n_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, N, N]
            n2n_attn = F.softmax(n2n_scores, dim=-1)                            # [B, H, N, N]

            n2n_out = torch.matmul(n2n_attn, V)                                 # [B, H, N, d_k]
            n2n_out = n2n_out.transpose(1, 2).contiguous().view(B, N, b)        # [B, N, b]
            node_ctx = self.n2n_norm(nodes + n2n_out)                            # residual + LayerNorm
        else:
            n2n_attn = None
            node_ctx = self.n2n_norm(nodes)

        # Step 2: N→E Dynamic Routing
        node_ctx_normed = self.n2e_pre_norm(node_ctx)                      # [B, N, b] Pre-LN
        Kn = self.n2e_k(node_ctx_normed)                                  # [B, N, b]
        Vn = self.n2e_v(node_ctx_normed)                                  # [B, N, b]

        Kn = Kn.view(B, N, H, d_k).transpose(1, 2)                       # [B, H, N, d_k]
        Vn = Vn.view(B, N, H, d_k).transpose(1, 2)                       # [B, H, N, d_k]

        # membership mask: edge_mask [N, E] → transpose → [E, N]
        n2e_mask = active_mask.T.unsqueeze(0).unsqueeze(0)                # [1, 1, E, N]

        # Q: learnable seeds
        q = self.n2e_q(self.n2e_seeds).unsqueeze(0).expand(B, -1, -1)    # [B, E, b]
        q = q.view(B, E, H, d_k).transpose(1, 2)                         # [B, H, E, d_k]

        # Dynamic routing
        _routing_attns = []
        _routing_hyperedges = []
        for r in range(self.num_routing_iters):
            n2e_scores = torch.matmul(q, Kn.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, E, N]
            n2e_scores = n2e_scores.masked_fill(n2e_mask == 0, -1e9)
            n2e_attn = F.softmax(n2e_scores, dim=-1)                     # [B, H, E, N]
            n2e_attn = torch.nan_to_num(n2e_attn, nan=0.0)
            _routing_attns.append(n2e_attn.detach())

            hyperedges = torch.matmul(n2e_attn, Vn)                      # [B, H, E, d_k]
            _routing_hyperedges.append(hyperedges.detach().transpose(1, 2).contiguous().view(B, E, b))

            if r < self.num_routing_iters - 1:
                q = hyperedges

        hyperedges = hyperedges.transpose(1, 2).contiguous().view(B, E, b)  # [B, E, b]
        hyperedges = hyperedges + self.edge_type_emb.unsqueeze(0)
        hyperedges = self.n2e_norm(hyperedges)

        # Step 2.5: Per-Edge MLP
        hyperedges_pre_mlp = hyperedges
        mlp_out = torch.stack([self.edge_mlps[e](hyperedges[:, e]) for e in range(E)], dim=1)  # [B, E, b]
        hyperedges = self.edge_mlp_norm(hyperedges + mlp_out)               # residual + LN

        # Step 3: E2N Cross-Attention (membership masked)
        node_ctx_normed_e2n = self.e2n_pre_norm_q(node_ctx)                # [B, N, b] Pre-LN (Q)
        hyperedges_normed = self.e2n_pre_norm_kv(hyperedges)              # [B, E, b] Pre-LN (K/V)
        Qe = self.e2n_q(node_ctx_normed_e2n)                              # [B, N, b]
        Ke = self.e2n_k(hyperedges_normed)                                 # [B, E, b]
        Ve = self.e2n_v(hyperedges_normed)                                 # [B, E, b]

        Qe = Qe.view(B, N, H, d_k).transpose(1, 2)                      # [B, H, N, d_k]
        Ke = Ke.view(B, E, H, d_k).transpose(1, 2)                       # [B, H, E, d_k]
        Ve = Ve.view(B, E, H, d_k).transpose(1, 2)                       # [B, H, E, d_k]

        scores_e2n = torch.matmul(Qe, Ke.transpose(-2, -1)) / math.sqrt(d_k)   # [B, H, N, E]
        scores_e2n = scores_e2n.masked_fill(active_mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)  # [1,1,N,E]

        attn_e2n = F.softmax(scores_e2n, dim=-1)
        attn_e2n = torch.nan_to_num(attn_e2n, nan=0.0)
        _attn_e2n_clean = attn_e2n.detach()

        nodes_update = torch.matmul(attn_e2n, Ve)                        # [B, H, N, d_k]
        nodes_update = nodes_update.transpose(1, 2).contiguous().view(B, N, b)
        update = self.update_dropout(self.e2n_out(nodes_update))
        nodes_out = self.e2n_norm(node_ctx + update)                      # [B, N, b]

        # Step 4: All 4 nodes → proj_up → output
        nodes_up = self.proj_up(nodes_out)                                 # [B, N, D]
        if self.use_residual_skip:
            nodes_up = self.output_norm(nodes_up) / math.sqrt(self.hidden_size) + nodes_full  # [B, N, D]
        else:
            nodes_up = self.output_norm(nodes_up) / math.sqrt(self.hidden_size)  # [B, N, D]

        h_conv  = nodes_up[:, 0, :]   # [B, D]
        h_like  = nodes_up[:, 1, :]
        h_long  = nodes_up[:, 2, :]
        h_short = nodes_up[:, 3, :]

        # Probing
        with torch.no_grad():
            eps = 1e-9
            n = nodes_full
            names = ["conv", "like", "long", "short"]

            # N2N attention entropy
            if n2n_attn is not None:
                _n2n_attn_clean = n2n_attn.detach()                              # [B, H, N, N]
                p_n2n = _n2n_attn_clean.clamp(min=eps)
                H_n2n = -(p_n2n * p_n2n.log()).sum(dim=-1).mean(dim=(0, 1))     # [N]
                H_n2n_max = math.log(N)
            else:
                H_n2n = torch.zeros(N)
                H_n2n_max = 1.0

            nc_ = F.normalize(node_ctx, p=2, dim=-1)
            n2e_entropy_per_edge = {}
            for r_idx, _r_attn in enumerate(_routing_attns):
                for e in range(E):
                    num_members = active_mask[:, e].sum().item()
                    if num_members < 2:
                        continue
                    p = _r_attn[:, :, e, :].clamp(min=eps)            # [B, H, N]
                    H_e = -(p * p.log()).sum(dim=-1).mean().item()
                    H_max = math.log(num_members)
                    n2e_entropy_per_edge[f"r{r_idx}_e{e}"] = {
                        "H": H_e, "H_max": H_max,
                        "H_ratio": H_e / H_max if H_max > 0 else 1.0,
                    }
            _n2e_attn_clean = _routing_attns[-1]
            e2n_entropy_per_node = {}
            for i, nm in enumerate(names):
                p = _attn_e2n_clean[:, :, i, :].clamp(min=eps)
                H_i = -(p * p.log()).sum(dim=-1).mean().item()
                num_active = active_mask[i].sum().item()
                H_max = math.log(max(num_active, 1))
                e2n_entropy_per_node[nm] = {
                    "H": H_i, "H_max": H_max,
                    "H_ratio": H_i / H_max if H_max > 0 else 1.0,
                }
            e2n_argmax_per_node = {
                nm: int(_attn_e2n_clean[:, :, i, :].mean(dim=(0, 1)).argmax().item())
                for i, nm in enumerate(names)
            }
            hg_direction_change = {
                nm: F.cosine_similarity(
                    n[:, i], nodes_up[:, i, :], dim=-1
                ).mean().item()
                for i, nm in enumerate(names)
            }
            np_ = F.normalize(nodes, p=2, dim=-1)
            probe = {}
            for i, nm in enumerate(names):
                probe[f"n2n_entropy_{nm}"] = (H_n2n[i] / H_n2n_max).item() if H_n2n_max > 0 else 1.0

            probe["n2n_sim_co_li"] = F.cosine_similarity(nc_[:, 0], nc_[:, 1], dim=-1).mean().item()
            probe["n2n_sim_co_lo"] = F.cosine_similarity(nc_[:, 0], nc_[:, 2], dim=-1).mean().item()
            probe["n2n_sim_co_sh"] = F.cosine_similarity(nc_[:, 0], nc_[:, 3], dim=-1).mean().item()

            for key, info in n2e_entropy_per_edge.items():
                probe[f"n2e_entropy_{key}"] = info["H_ratio"]

            for nm, info in e2n_entropy_per_node.items():
                probe[f"e2n_entropy_{nm}"] = info["H_ratio"]

            for nm, idx in e2n_argmax_per_node.items():
                probe[f"e2n_argmax_{nm}"] = float(idx)

            for nm, val in hg_direction_change.items():
                probe[f"direction_change_{nm}"] = val

            probe["proj_sim_co_li"] = F.cosine_similarity(np_[:, 0], np_[:, 1], dim=-1).mean().item()
            probe["proj_sim_co_lo"] = F.cosine_similarity(np_[:, 0], np_[:, 2], dim=-1).mean().item()
            probe["proj_sim_co_sh"] = F.cosine_similarity(np_[:, 0], np_[:, 3], dim=-1).mean().item()
            probe["proj_sim_li_lo"] = F.cosine_similarity(np_[:, 1], np_[:, 2], dim=-1).mean().item()
            probe["proj_sim_li_sh"] = F.cosine_similarity(np_[:, 1], np_[:, 3], dim=-1).mean().item()
            probe["proj_sim_lo_sh"] = F.cosine_similarity(np_[:, 2], np_[:, 3], dim=-1).mean().item()

            probe["input_sim_conv_like"]  = F.cosine_similarity(n[:, 0], n[:, 1], dim=-1).mean().item()
            probe["input_sim_conv_long"]  = F.cosine_similarity(n[:, 0], n[:, 2], dim=-1).mean().item()
            probe["input_sim_conv_short"] = F.cosine_similarity(n[:, 0], n[:, 3], dim=-1).mean().item()
            probe["input_sim_like_long"]  = F.cosine_similarity(n[:, 1], n[:, 2], dim=-1).mean().item()
            probe["input_sim_like_short"] = F.cosine_similarity(n[:, 1], n[:, 3], dim=-1).mean().item()
            probe["input_sim_long_short"] = F.cosine_similarity(n[:, 2], n[:, 3], dim=-1).mean().item()

            for r_idx, r_he in enumerate(_routing_hyperedges):
                rhe = F.normalize(r_he, p=2, dim=-1)
                r_sims = []
                for i in range(E):
                    for j in range(i + 1, E):
                        sim = F.cosine_similarity(rhe[:, i], rhe[:, j], dim=-1).mean().item()
                        r_sims.append(sim)
                        probe[f"he_sim_r{r_idx}_e{i}_e{j}"] = sim
                probe[f"he_sim_r{r_idx}_mean"] = sum(r_sims) / len(r_sims)

            he = F.normalize(hyperedges.detach(), p=2, dim=-1)  # [B, E, b]
            for i in range(E):
                for j in range(i + 1, E):
                    probe[f"he_sim_e{i}_e{j}"] = F.cosine_similarity(
                        he[:, i], he[:, j], dim=-1
                    ).mean().item()

            he_pre = F.normalize(hyperedges_pre_mlp.detach(), p=2, dim=-1)
            pre_sims, post_sims = [], []
            for i in range(E):
                for j in range(i + 1, E):
                    pre_sims.append(F.cosine_similarity(he_pre[:, i], he_pre[:, j], dim=-1).mean().item())
                    post_sims.append(F.cosine_similarity(he[:, i], he[:, j], dim=-1).mean().item())
            probe["he_sim_mean_pre_mlp"] = sum(pre_sims) / len(pre_sims)
            probe["he_sim_mean_post_mlp"] = sum(post_sims) / len(post_sims)

            no = F.normalize(nodes_out.detach(), p=2, dim=-1)
            probe["e2n_out_sim_co_li"] = F.cosine_similarity(no[:, 0], no[:, 1], dim=-1).mean().item()
            probe["e2n_out_sim_co_lo"] = F.cosine_similarity(no[:, 0], no[:, 2], dim=-1).mean().item()
            probe["e2n_out_sim_co_sh"] = F.cosine_similarity(no[:, 0], no[:, 3], dim=-1).mean().item()

            probe["proj_up_norm"] = nodes_up.detach().norm(dim=-1).mean().item()

            self._probe = probe

        return {
            "nodes_updated": {
                "conv": h_conv, "like": h_like, "long": h_long, "short": h_short,
            },
        }


class BackboneEncoder(BaseEncoder):
    VALID_MODES = ("vanilla", "dynamic_gating", "hypergraph")

    _SIGNAL_META = {
        "c": ("conv",    False),
        "l": ("like",    False),
        "d": ("dislike", True),
        "a": ("long",    False),
        "b": ("short",   False),
        "x": ("long",    False),
        "y": ("short",   False),
    }

    def __init__(
        self,
        base_model_name: str,
        mode: str = "vanilla",
        query_used_info: tuple = ("c", "l", "d", "a", "b"),
        alpha: float = 0.5,
        beta: float = 0.2,
        delta: float = 0.2,
        epsilon: float = 0.5,
        seed: int = 42,
        uniform_gate_init: bool = False,
        gate_activation: str = "sigmoid",
        hg_only: bool = False,
        num_routing_iters: int = 2,
        use_n2n: bool = True,
        use_global_edge: bool = True,
        flat_gating: bool = False,
    ) -> None:
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode
        self.hg_only = hg_only
        self.flat_gating = flat_gating
        self.base_model_name = base_model_name
        self._uses_nv_embed_backbone = "nv-embed" in base_model_name.lower()
        base_model_name_l = base_model_name.lower()
        self._uses_stella_backbone = "stella_en_" in base_model_name_l and "_v5" in base_model_name_l
        self._uses_last_token_pooling_backbone = (
            "qwen3-embedding" in base_model_name.lower()
            or "harrier-oss" in base_model_name.lower()
            or "f2llm-v2" in base_model_name.lower()
        )
        self._uses_bidirlm_backbone = "bidirlm" in base_model_name.lower()
        self.stella_vector_dim = 1024

        print(f"[BackboneEncoder] Loading backbone: {base_model_name}")
        self.backbone = AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if self._uses_nv_embed_backbone:
            self._patch_nv_embed_position_embeddings()
        self.backbone_hidden_size = self._infer_backbone_hidden_size()
        self.stella_projection = self._load_stella_projection() if self._uses_stella_backbone else None
        self.hidden_size: int = self.stella_vector_dim if self._uses_stella_backbone else self.backbone_hidden_size

        if mode == "vanilla":
            print("[BackboneEncoder] Mode: vanilla (fixed scalar weights, LoRA only)")
            self._alpha = float(alpha)
            self._beta = float(beta)
            self._delta = float(delta)
            self._epsilon = float(epsilon)

        elif mode == "dynamic_gating":
            print("[BackboneEncoder] Mode: dynamic_gating (MLP gates)")
            self.query_used_info = list(query_used_info)
            n_signals = len(self.query_used_info)
            if uniform_gate_init:
                init_biases = [0.0] * n_signals
            else:
                _vanilla_w = {"c": 1.0, "l": float(alpha), "d": float(beta),
                              "a": float(delta), "b": float(epsilon),
                              "x": float(delta), "y": float(epsilon)}
                init_biases = [
                    torch.logit(torch.tensor(min(max(_vanilla_w[k], 0.05), 0.95))).item()
                    for k in self.query_used_info
                ]
            self.gating_network = DynamicGatingNetwork(
                self.hidden_size, n_signals=n_signals, init_biases=init_biases
            )

        elif mode == "hypergraph":
            hg_label = "HG-only" if hg_only else "N2N-E2N + TwinExpert"
            gating_label = "flat-9" if flat_gating else "split_ratio"
            print(f"[BackboneEncoder] Mode: hypergraph ({hg_label}, gate={gate_activation}, gating={gating_label})")
            self.hypergraph_module = Hypergraph(
                self.hidden_size, num_routing_iters=num_routing_iters,
                use_n2n=use_n2n, use_global_edge=use_global_edge,
                use_residual_skip=not flat_gating,
            )

            _raw_w = [1.0, float(alpha), float(delta), float(epsilon), float(beta)]
            _hg_w = [0.5, 0.5, 0.5, 0.5]
            init_biases = [
                torch.logit(torch.tensor(min(max(v, 0.05), 0.95))).item()
                for v in _raw_w + _hg_w
            ]

            if flat_gating:
                self.gating_network = DynamicGatingNetwork(
                    self.hidden_size, n_signals=9, init_biases=init_biases,
                )
            else:
                self.gating_network = HypergraphGatingNetwork(
                    self.hidden_size,
                    gate_activation=gate_activation,
                    init_biases=init_biases,
                    hg_only=hg_only,
                )

        self._configure_lora(seed)

    @property
    def alpha(self): return self._alpha
    @property
    def beta(self): return self._beta
    @property
    def delta(self): return self._delta
    @property
    def epsilon(self): return self._epsilon

    def _configure_lora(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        for param in self.backbone.parameters():
            param.requires_grad = False

        lora_cfg = LoraConfig(
            target_modules=self._get_lora_target_modules(),
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        if self.mode == "dynamic_gating":
            for param in self.gating_network.parameters():
                param.requires_grad = True
        elif self.mode == "hypergraph":
            for param in self.gating_network.parameters():
                param.requires_grad = True
            for param in self.hypergraph_module.parameters():
                param.requires_grad = True

    def _get_lora_target_modules(self) -> list[str]:
        if self._uses_nv_embed_backbone:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

        module_leaf_names = {name.rsplit(".", 1)[-1] for name, module in self.backbone.named_modules()
                             if isinstance(module, nn.Linear)}
        for candidates in (
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["qkv_proj", "o_proj"],
            ["query", "key", "value", "dense"],
            ["q", "k", "v"],
        ):
            if all(name in module_leaf_names for name in candidates):
                print(f"[BackboneEncoder] Non-NV LoRA target modules: {candidates}")
                return candidates
        raise ValueError(
            f"Could not infer LoRA target modules for non-NV backbone '{self.base_model_name}'. "
            f"Available Linear leaf names include: {sorted(module_leaf_names)[:30]}"
        )

    def _infer_backbone_hidden_size(self) -> int:
        for config in (self.backbone.config, getattr(self.backbone.config, "text_config", None)):
            if config is not None and hasattr(config, "hidden_size"):
                return int(config.hidden_size)
        raise ValueError(f"Could not infer hidden_size for backbone '{self.base_model_name}'")

    def _load_stella_projection(self) -> nn.Linear:
        projection = nn.Linear(self.backbone_hidden_size, self.stella_vector_dim)
        candidate_filenames = [
            f"2_Dense_{self.stella_vector_dim}/model.safetensors",
            f"2_Dense_{self.stella_vector_dim}/pytorch_model.bin",
            "2_Dense/model.safetensors",
            "2_Dense/pytorch_model.bin",
        ]

        if os.path.isdir(self.base_model_name):
            projection_path = None
            for filename in candidate_filenames:
                candidate_path = os.path.join(self.base_model_name, filename)
                if os.path.exists(candidate_path):
                    projection_path = candidate_path
                    break
            if projection_path is None:
                raise FileNotFoundError(
                    f"Could not find Stella projection in '{self.base_model_name}'. "
                    f"Tried: {candidate_filenames}"
                )
        else:
            last_error = None
            for filename in candidate_filenames:
                try:
                    projection_path = hf_hub_download(
                        repo_id=self.base_model_name,
                        filename=filename,
                    )
                    break
                except Exception as exc:
                    last_error = exc
            else:
                raise RuntimeError(
                    f"Could not download Stella projection for '{self.base_model_name}'. "
                    f"Tried: {candidate_filenames}"
                ) from last_error

        if projection_path.endswith(".safetensors"):
            state = load_safetensors(projection_path)
        else:
            state = torch.load(projection_path, map_location="cpu")
        state = {key.replace("linear.", ""): value for key, value in state.items()}
        projection.load_state_dict(state)
        for param in projection.parameters():
            param.requires_grad = False
        print(f"[BackboneEncoder] Stella projection loaded: {self.backbone_hidden_size} -> {self.stella_vector_dim}")
        return projection

    def _patch_nv_embed_position_embeddings(self) -> None:
        """
        NV-Embed-v2 remote code subclasses MistralModel and calls decoder layers
        without the shared RoPE tuple introduced in newer transformers releases.
        Patch only the loaded NV backbone instance so other backbones stay intact.
        """
        embedding_model = getattr(self.backbone, "embedding_model", None)
        rotary_emb = getattr(embedding_model, "rotary_emb", None)
        layers = getattr(embedding_model, "layers", None)
        if embedding_model is None or rotary_emb is None or layers is None:
            return

        embedding_model.config.use_cache = False

        patched = 0
        for layer in layers:
            if getattr(layer, "_backbone_nv_position_patch", False):
                continue
            if "position_embeddings" not in inspect.signature(layer.forward).parameters:
                continue

            layer._backbone_original_forward = layer.forward
            layer._backbone_rotary_emb = rotary_emb

            def _forward_with_position_embeddings(
                layer_self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                **kwargs,
            ):
                if position_embeddings is None:
                    if position_ids is None:
                        if cache_position is None:
                            cache_position = torch.arange(
                                hidden_states.shape[1],
                                device=hidden_states.device,
                            )
                        position_ids = cache_position.unsqueeze(0)
                    position_embeddings = layer_self._backbone_rotary_emb(hidden_states, position_ids)

                return layer_self._backbone_original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            layer.forward = MethodType(_forward_with_position_embeddings, layer)
            layer._backbone_nv_position_patch = True
            patched += 1

        if patched:
            print(f"[BackboneEncoder] Patched NV-Embed RoPE compatibility for {patched} decoder layers")

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(last_hidden_state.device).unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    @staticmethod
    def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_state[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        return last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device),
            sequence_lengths.to(last_hidden_state.device),
        ]

    def _encode_signal(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if input_ids is None:
            return None
        if self._uses_nv_embed_backbone:
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pool_mask=attention_mask,
            )
            pooled = out["sentence_embeddings"]
        else:
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if "sentence_embeddings" in out:
                pooled = out["sentence_embeddings"]
            elif self._uses_last_token_pooling_backbone:
                pooled = self._last_token_pool(out.last_hidden_state, attention_mask)
            else:
                pooled = self._mean_pool(out.last_hidden_state, attention_mask)
            if self.stella_projection is not None:
                pooled = self.stella_projection(pooled.to(self.stella_projection.weight.dtype))
        return F.normalize(pooled, p=2, dim=-1)

    def encode_item(self, batch: dict) -> torch.Tensor:
        return self._encode_signal(
            batch.get("item_input_ids"),
            batch.get("item_attention_mask"),
        )

    def _encode_query_signals(self, batch: dict) -> dict[str, Optional[torch.Tensor]]:
        conv = self._encode_signal(batch.get("conv_input_ids"), batch.get("conv_attention_mask"))
        like = self._encode_signal(batch.get("like_input_ids"), batch.get("like_attention_mask"))
        dislike = self._encode_signal(batch.get("dislike_input_ids"), batch.get("dislike_attention_mask"))
        long_ = self._encode_signal(batch.get("long_input_ids"), batch.get("long_attention_mask"))
        short = self._encode_signal(batch.get("short_input_ids"), batch.get("short_attention_mask"))
        return {
            "conv": conv,
            "like": like,
            "dislike": dislike,
            "long": long_,
            "short": short,
        }

    def encode_query(self, batch: dict) -> torch.Tensor:
        signals = self._encode_query_signals(batch)
        return self._fuse_query_signals(
            signals["conv"],
            signals["like"],
            signals["dislike"],
            signals["long"],
            signals["short"],
        )

    def encode_query_with_components(self, batch: dict) -> dict[str, Optional[torch.Tensor]]:
        signals = self._encode_query_signals(batch)
        user_embedding = self._fuse_query_signals(
            signals["conv"],
            signals["like"],
            signals["dislike"],
            signals["long"],
            signals["short"],
        )
        return {
            "user_embedding": user_embedding,
            "conv_embedding": signals["conv"],
            "like_embedding": signals["like"],
            "dislike_embedding": signals["dislike"],
            "long_embedding": signals["long"],
            "short_embedding": signals["short"],
        }

    def _fuse_query_signals(self, conv, like, dislike, long_, short) -> torch.Tensor:
        if self.mode == "vanilla":
            return self._fuse_vanilla(conv, like, dislike, long_, short)
        if self.mode == "dynamic_gating":
            return self._fuse_dynamic(conv, like, dislike, long_, short)
        if self.mode == "hypergraph":
            return self._fuse_hypergraph(conv, like, dislike, long_, short)
        raise NotImplementedError(f"mode='{self.mode}' not yet implemented")

    def _fuse_vanilla(self, conv, like, dislike, long_, short):
        reference = next(e for e in [conv, like, long_, short, dislike] if e is not None)
        user_emb = torch.zeros_like(reference)
        if conv is not None: user_emb = user_emb + 1.0 * conv
        if like is not None: user_emb = user_emb + self.alpha * like
        if dislike is not None: user_emb = user_emb - self.beta * dislike
        if long_ is not None: user_emb = user_emb + self.delta * long_
        if short is not None: user_emb = user_emb + self.epsilon * short
        return F.normalize(user_emb, p=2, dim=-1)

    def _fuse_dynamic(self, conv, like, dislike, long_, short):
        _all = {"c": conv, "l": like, "d": dislike, "a": long_, "b": short,
                "x": long_, "y": short}
        available = [(key, _all[key]) for key in self.query_used_info]
        x = torch.cat([sig for _, sig in available], dim=-1)
        gates = self.gating_network(x)
        user_emb = torch.zeros_like(available[0][1])
        for i, (key, sig) in enumerate(available):
            _, is_neg = self._SIGNAL_META[key]
            if is_neg:
                user_emb = user_emb - gates[:, i:i+1] * sig
            else:
                user_emb = user_emb + gates[:, i:i+1] * sig
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        with torch.no_grad():
            self._probe_gates = {}
            for i, (key, _) in enumerate(available):
                name, _ = self._SIGNAL_META[key]
                self._probe_gates[f"gate_{name}_mean"] = gates[:, i].mean().item()
                self._probe_gates[f"gate_{name}_std"] = (
                    gates[:, i].std().item() if gates.shape[0] > 1 else 0.0
                )
        return user_emb

    def _fuse_hypergraph(self, conv, like, dislike, long_, short):
        reference = next(e for e in [conv, like, long_, short, dislike] if e is not None)
        zeros = torch.zeros_like(reference)
        raw_conv = conv if conv is not None else zeros
        raw_like = like if like is not None else zeros
        raw_long = long_ if long_ is not None else zeros
        raw_short = short if short is not None else zeros
        raw_dis = dislike if dislike is not None else zeros

        hg_out = self.hypergraph_module(raw_conv, raw_like, raw_long, raw_short)
        conv_h = hg_out["nodes_updated"]["conv"]
        like_h = hg_out["nodes_updated"]["like"]
        long_h = hg_out["nodes_updated"]["long"]
        short_h = hg_out["nodes_updated"]["short"]

        if self.flat_gating:
            all_signals = [raw_conv, raw_like, raw_long, raw_short, raw_dis,
                           conv_h, like_h, long_h, short_h]
            x = torch.cat(all_signals, dim=-1)       # [B, 9*D]
            gates = self.gating_network(x)             # [B, 9] sigmoid

            if hasattr(self, '_zero_hg_gates') and self._zero_hg_gates:
                gates = gates.clone()
                gates[:, 5:9] = 0.0
            user_emb = F.normalize(
                gates[:, 0:1] * raw_conv
                + gates[:, 1:2] * raw_like
                + gates[:, 2:3] * raw_long
                + gates[:, 3:4] * raw_short
                - gates[:, 4:5] * raw_dis
                + gates[:, 5:6] * conv_h
                + gates[:, 6:7] * like_h
                + gates[:, 7:8] * long_h
                + gates[:, 8:9] * short_h,
                p=2, dim=-1,
            )

            with torch.no_grad():
                raw_names = ["conv", "like", "long", "short", "dislike"]
                hg_names = ["hg_conv", "hg_like", "hg_long", "hg_short"]
                B = gates.shape[0]
                self._probe_gates = {
                    **{f"raw_{raw_names[i]}_mean": gates[:, i].mean().item() for i in range(5)},
                    **{f"raw_{raw_names[i]}_std": (gates[:, i].std().item() if B > 1 else 0.0) for i in range(5)},
                    **{f"{hg_names[i]}_mean": gates[:, 5+i].mean().item() for i in range(4)},
                    **{f"{hg_names[i]}_std": (gates[:, 5+i].std().item() if B > 1 else 0.0) for i in range(4)},
                    "user_emb_norm": user_emb.norm(dim=-1).mean().item(),
                    **{f"hg_internal_{k}": v
                       for k, v in getattr(self.hypergraph_module, "_probe", {}).items()},
                }
            return user_emb

        gating_weights, split_ratio = self.gating_network(
            raw_conv, raw_like, raw_long, raw_short, raw_dis,
            conv_h, like_h, long_h, short_h,
        )

        if hasattr(self, '_fixed_split_ratio') and self._fixed_split_ratio is not None:
            split_ratio = torch.full_like(split_ratio, self._fixed_split_ratio)

        hg_expert = (
            gating_weights[:, 5:6] * conv_h
            + gating_weights[:, 6:7] * like_h
            + gating_weights[:, 7:8] * long_h
            + gating_weights[:, 8:9] * short_h
        )
        if self.hg_only:
            user_emb = hg_expert

            with torch.no_grad():
                hg_names = ["hg_conv", "hg_like", "hg_long", "hg_short"]
                B = gating_weights.shape[0]
                self._probe_gates = {
                    **{f"{hg_names[i]}_mean": gating_weights[:, 5+i].mean().item() for i in range(4)},
                    **{f"{hg_names[i]}_std": (gating_weights[:, 5+i].std().item() if B > 1 else 0.0) for i in range(4)},
                    "hg_expert_norm": hg_expert.norm(dim=-1).mean().item(),
                    **{f"hg_internal_{k}": v
                       for k, v in getattr(self.hypergraph_module, "_probe", {}).items()},
                }
        else:
            raw_expert = (
                gating_weights[:, 0:1] * raw_conv
                + gating_weights[:, 1:2] * raw_like
                + gating_weights[:, 2:3] * raw_long
                + gating_weights[:, 3:4] * raw_short
                - gating_weights[:, 4:5] * raw_dis
            )

            raw_expert_normed = F.normalize(raw_expert, p=2, dim=-1)
            hg_expert_normed = F.normalize(hg_expert, p=2, dim=-1)
            user_emb = (1 - split_ratio) * raw_expert_normed + split_ratio * hg_expert_normed

            with torch.no_grad():
                raw_names = ["conv", "like", "long", "short", "dislike"]
                hg_names = ["hg_conv", "hg_like", "hg_long", "hg_short"]
                B = gating_weights.shape[0]
                self._probe_gates = {
                    **{f"raw_{raw_names[i]}_mean": gating_weights[:, i].mean().item() for i in range(5)},
                    **{f"raw_{raw_names[i]}_std": (gating_weights[:, i].std().item() if B > 1 else 0.0) for i in range(5)},
                    **{f"{hg_names[i]}_mean": gating_weights[:, 5+i].mean().item() for i in range(4)},
                    **{f"{hg_names[i]}_std": (gating_weights[:, 5+i].std().item() if B > 1 else 0.0) for i in range(4)},
                    "split_ratio_mean": split_ratio.mean().item(),
                    "split_ratio_std": (split_ratio.std().item() if B > 1 else 0.0),
                    "raw_expert_norm": raw_expert.norm(dim=-1).mean().item(),
                    "hg_expert_norm": hg_expert.norm(dim=-1).mean().item(),
                    "raw_hg_sim": F.cosine_similarity(raw_expert, hg_expert, dim=-1).mean().item(),
                    **{f"hg_internal_{k}": v
                       for k, v in getattr(self.hypergraph_module, "_probe", {}).items()},
                }

        return user_emb

    def get_weight_probe(self) -> dict:
        if self.mode == "vanilla":
            return {
                "model/alpha(like)": self.alpha,
                "model/beta(dislike)": self.beta,
                "model/delta(long)": self.delta,
                "model/epsilon(short)": self.epsilon,
            }
        elif self.mode in ("dynamic_gating", "hypergraph"):
            probe = getattr(self, "_probe_gates", {})
            return {f"model/{k}": v for k, v in probe.items()}
        return {}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path)
        head_state = {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if not name.startswith("backbone.")
        }
        torch.save(head_state, os.path.join(path, "head.pt"))
        print(f"[BackboneEncoder] Saved to {path}  (head keys: {list(head_state.keys())})")

    def load(self, path: str, device: str = "cpu") -> None:
        self.backbone.load_adapter(path, adapter_name="default")
        self.backbone.set_adapter("default")
        head_path = os.path.join(path, "head.pt")
        if os.path.exists(head_path):
            head_state = torch.load(head_path, map_location=device)
            for name, param in self.named_parameters():
                if name in head_state:
                    param.data.copy_(head_state[name])
            print(f"[BackboneEncoder] Loaded head from {head_path}")
        else:
            print(f"[BackboneEncoder] Warning: {head_path} not found, head params unchanged")


