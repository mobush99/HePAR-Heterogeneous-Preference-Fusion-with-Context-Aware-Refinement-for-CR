import math
import os
import random
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import Tensor
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import TrainingConfig
from core.model import BaseEncoder
from core.sampler import HardNegativeSampler, MixedNegativeSampler, NegativeSampler
from core.loss import RetrievalLoss
from core.evaluator import Evaluator
from core.logger import Logger
from core.trainer import BaseTrainer


class Trainer(BaseTrainer):
    LAST_CKPT_DIRNAME = "last"
    TRAINER_STATE_FILENAME = "trainer_state.pt"

    def __init__(
        self,
        model: BaseEncoder,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        train_infer_dataloader: DataLoader,
        item_dataloader: DataLoader,
        val_evaluator: Evaluator,
        test_evaluator: Evaluator,
        val_gt_ids: Union[List[int], List[List[int]]],
        test_gt_ids: Union[List[int], List[List[int]]],
        gt_mask: Tensor,
        sampler: NegativeSampler,
        loss_fn: RetrievalLoss,
        logger: Logger,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.train_infer_dataloader = train_infer_dataloader
        self.item_dataloader = item_dataloader
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        self.val_gt_ids = val_gt_ids
        self.test_gt_ids = test_gt_ids
        self.gt_mask = gt_mask
        self.sampler = sampler
        self.loss_fn = loss_fn
        self.logger = logger

        self.item_dataset = item_dataloader.dataset

    def train(self, device: str) -> Dict:
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer)

        best_metric = -float("inf")
        best_metrics = {}
        patience_counter = 0
        start_epoch = 0
        ckpt_path = self.config.ckpt_dir
        os.makedirs(self.config.ckpt_dir, exist_ok=True)

        n_train = len(self.train_dataloader.dataset)
        n_items = len(self.item_dataset)
        n_val = len(self.val_evaluator.query_dataloader.dataset)
        n_batches_per_epoch = len(self.train_dataloader)
        n_opt_steps_per_epoch = math.ceil(n_batches_per_epoch / self.config.accumulation_steps)
        total_steps = n_opt_steps_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        print(f"\n{'='*64}")
        print(f"[Trainer] Training start")
        print(f"  train queries : {n_train}  |  items : {n_items}  |  val queries : {n_val}")
        print(f"  epochs        : {self.config.epochs}  |  patience : {self.config.patience}"
              f"  |  n_neg : {self.config.n_negatives}")
        print(f"  lr            : {self.config.learning_rate}  |  batch_size : {self.config.batch_size}"
              f"  |  accum : {self.config.accumulation_steps}"
              f"  |  epoch_encoding : {self.config.epoch_encoding}"
              f"  |  warmup_steps : {warmup_steps}/{total_steps}  (warmup_ratio={self.config.warmup_ratio})")
        print(f"  eval_metric   : {self.config.eval_metric}  |  temperature : {self.config.temperature}")
        print(f"  ckpt_dir      : {self.config.ckpt_dir}")
        print(f"{'='*64}\n")

        if self.config.resume:
            resume_state = self.load_checkpoint(
                ckpt_path,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                resume_training=True,
            )
            if resume_state["resume_mode"] == "manual_legacy":
                self._prime_scheduler_to_step(scheduler, resume_state["resume_step"])
            start_epoch = resume_state["start_epoch"]
            best_metric = resume_state["best_metric"]
            best_metrics = resume_state["best_metrics"]
            patience_counter = resume_state["patience_counter"]
            print(f"[Trainer] Resumed from {resume_state['checkpoint_path']}")
            print(f"  next_epoch    : {start_epoch + 1}/{self.config.epochs}")
            print(f"  best_metric   : {best_metric:.4f}")
            print(f"  patience      : {patience_counter}/{self.config.patience}\n")

        if start_epoch == 0:
            print(f"[Trainer] Zero-shot evaluation (before training)")
            self.val_evaluator.evaluate(self.val_gt_ids, device, split="val")

        for epoch in range(start_epoch, self.config.epochs):
            print(f"\n{'─'*64}")
            print(f"[Epoch {epoch + 1}/{self.config.epochs}]"
                  f"  patience: {patience_counter}/{self.config.patience}"
                  f"  best {self.config.eval_metric}: {best_metric:.4f}")
            print(f"{'─'*64}")

            needs_epoch_item_emb = (
                self.config.epoch_encoding
                or isinstance(self.sampler, (HardNegativeSampler, MixedNegativeSampler))
            )
            if needs_epoch_item_emb:
                item_emb = self._encode_items(device)              # [M, D] CPU
                print(f"  [Enc] item_emb  : {tuple(item_emb.shape)}"
                      f"  norm(mean)={item_emb.norm(dim=-1).mean():.4f}")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                item_emb = torch.empty(0)
                print("  [Enc] item_emb  : skipped (random sampler, epoch_encoding=False)")

            if isinstance(self.sampler, (HardNegativeSampler, MixedNegativeSampler)):
                query_emb = self._encode_queries(device)           # [N, D] CPU
                print(f"  [Enc] query_emb : {tuple(query_emb.shape)}"
                      f"  norm(mean)={query_emb.norm(dim=-1).mean():.4f}")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                query_emb = torch.empty(0)

            neg_indices = self.sampler.sample(
                query_emb, item_emb, self.gt_mask, self.config.n_negatives
            )                                                       # [N, n_neg]

            self._probe_neg_sampling(query_emb, item_emb, neg_indices)

            epoch_stats = self._train_one_epoch(
                epoch, neg_indices, item_emb, optimizer, scheduler, device,
            )

            aux_str = f"  aux_loss={epoch_stats['aux_loss']:.4f}" if epoch_stats.get('aux_loss', 0) > 0 else ""
            grad_str = f"  grad_lora={epoch_stats['grad_norm']:.4f}"
            if epoch_stats.get('grad_norm_head', 0) > 0:
                grad_str += f"  grad_head={epoch_stats['grad_norm_head']:.4f}"
            print(f"\n  [Train] loss={epoch_stats['loss']:.4f}{aux_str}"
                  f"  pos_score={epoch_stats['pos_score']:.4f}"
                  f"  neg_score={epoch_stats['neg_score']:.4f}"
                  f"  score_gap={epoch_stats['score_gap']:.4f}"
                  f"{grad_str}"
                  f"  lr={epoch_stats['lr']:.2e}")

            train_log = {
                "train/pos_score": epoch_stats["pos_score"],
                "train/neg_score": epoch_stats["neg_score"],
                "train/score_gap": epoch_stats["score_gap"],
            }

            mode = getattr(self.model, "mode", "")
            if mode == "vanilla":
                weight_probe = self.model.get_weight_probe()
                print(f"  [Weights] alpha(like)={weight_probe.get('model/alpha(like)', 0):.4f}"
                      f"  beta(dislike)={weight_probe.get('model/beta(dislike)', 0):.4f}"
                      f"  delta(long)={weight_probe.get('model/delta(long)', 0):.4f}"
                      f"  epsilon(short)={weight_probe.get('model/epsilon(short)', 0):.4f}")
                train_log.update(weight_probe)

            elif mode == "dynamic_gating":
                gates = epoch_stats.get("gates", {})
                if gates:
                    signals = ["conv", "like", "long", "short", "dislike"]
                    gate_str = "  ".join(
                        f"{s}={gates.get(f'gate_{s}_mean', 0):.4f}"
                        f"(±{gates.get(f'gate_{s}_std', 0):.4f})"
                        for s in signals
                    )
                    print(f"  [Gates]  {gate_str}")
                    for s in signals:
                        train_log[f"gating/{s}_mean"] = gates.get(f"gate_{s}_mean", 0.0)
                        train_log[f"gating/{s}_std"] = gates.get(f"gate_{s}_std", 0.0)

            elif mode == "hypergraph":
                gates = epoch_stats.get("gates", {})
                if gates:
                    raw_signals = ["conv", "like", "long", "short", "dislike"]
                    raw_str = "  ".join(
                        f"{s}={gates.get(f'raw_{s}_mean', 0):.4f}"
                        f"(±{gates.get(f'raw_{s}_std', 0):.4f})"
                        for s in raw_signals
                    )
                    print(f"  [Raw gates]  {raw_str}")

                    hg_signals = ["hg_conv", "hg_like", "hg_long", "hg_short"]
                    hg_str = "  ".join(
                        f"{s}={gates.get(f'{s}_mean', 0):.4f}"
                        f"(±{gates.get(f'{s}_std', 0):.4f})"
                        for s in hg_signals
                    )
                    print(f"  [HG gates]   {hg_str}")

                    sr = gates.get("split_ratio_mean", 0.0)
                    sr_std = gates.get("split_ratio_std", 0.0)
                    raw_n = gates.get("raw_expert_norm", 0.0)
                    hg_n = gates.get("hg_expert_norm", 0.0)
                    rh_sim = gates.get("raw_hg_sim", 0.0)
                    print(f"  [Blend]      split_ratio={sr:.4f}(±{sr_std:.4f})"
                          f"  raw_norm={raw_n:.4f}  hg_norm={hg_n:.4f}"
                          f"  raw_hg_sim={rh_sim:.3f}")

                    nodes = ["conv", "like", "long", "short"]
                    n2n_ent_str = "  ".join(
                        f"{nm}={gates.get(f'hg_internal_n2n_entropy_{nm}', 0):.3f}"
                        for nm in nodes
                    )
                    n2n_sim_str = (
                        f"co-li={gates.get('hg_internal_n2n_sim_co_li', 0):.3f}"
                        f"  co-lo={gates.get('hg_internal_n2n_sim_co_lo', 0):.3f}"
                        f"  co-sh={gates.get('hg_internal_n2n_sim_co_sh', 0):.3f}"
                    )
                    print(f"  [N2N ent]    {n2n_ent_str}")
                    print(f"  [N2N sim]    {n2n_sim_str}")

                    n2e_keys = sorted([k for k in gates if k.startswith("hg_internal_n2e_entropy_")])
                    if n2e_keys:
                        from itertools import groupby
                        def _n2e_round(k):
                            suffix = k.split("hg_internal_n2e_entropy_")[-1]  # "r0_e0"
                            return suffix.split("_")[0]  # "r0"
                        for r_label, g in groupby(n2e_keys, key=_n2e_round):
                            keys = list(g)
                            ent_str = "  ".join(
                                f"e{k.split('hg_internal_n2e_entropy_')[-1].split('_e')[-1]}={gates.get(k, 0):.3f}"
                                for k in keys
                            )
                            print(f"  [N2E {r_label}]   {ent_str}")

                    e2n_str = "  ".join(
                        f"{nm}={gates.get(f'hg_internal_e2n_entropy_{nm}', 0):.3f}"
                        f"(→e{int(gates.get(f'hg_internal_e2n_argmax_{nm}', 0))})"
                        for nm in nodes
                    )
                    print(f"  [E2N ent]    {e2n_str}")

                    dir_str = "  ".join(
                        f"{nm}={gates.get(f'hg_internal_direction_change_{nm}', 0):.3f}"
                        for nm in nodes
                    )
                    print(f"  [Direction]  {dir_str}")

                    r_idx = 0
                    while f"hg_internal_he_sim_r{r_idx}_mean" in gates:
                        print(f"  [HE sim r{r_idx}] mean={gates[f'hg_internal_he_sim_r{r_idx}_mean']:.3f}")
                        r_idx += 1

                    pre_mlp = gates.get("hg_internal_he_sim_mean_pre_mlp")
                    post_mlp = gates.get("hg_internal_he_sim_mean_post_mlp")
                    if pre_mlp is not None and post_mlp is not None:
                        print(f"  [HE sim MLP] pre={pre_mlp:.3f}  post={post_mlp:.3f}  delta={post_mlp - pre_mlp:+.3f}")

                    he_indices = set()
                    for k in gates:
                        if k.startswith("hg_internal_he_sim_e"):
                            parts = k.split("hg_internal_he_sim_e")[-1]  # "0_e3"
                            i_str, j_str = parts.split("_e")
                            he_indices.add(int(i_str))
                            he_indices.add(int(j_str))
                    he_indices = sorted(he_indices)
                    n_he = len(he_indices)
                    if n_he > 0:
                        header = "  [HE sim]     " + "".join(f"   e{j}" for j in he_indices)
                        print(header)
                        for i in he_indices:
                            row = f"             e{i}"
                            for j in he_indices:
                                if j <= i:
                                    row += "      "
                                else:
                                    val = gates.get(f"hg_internal_he_sim_e{i}_e{j}", 0)
                                    row += f" {val:.3f}"
                            print(row)

                    e2n_out_str = (
                        f"co-li={gates.get('hg_internal_e2n_out_sim_co_li', 0):.3f}"
                        f"  co-lo={gates.get('hg_internal_e2n_out_sim_co_lo', 0):.3f}"
                        f"  co-sh={gates.get('hg_internal_e2n_out_sim_co_sh', 0):.3f}"
                    )
                    print(f"  [E2N out]    {e2n_out_str}"
                          f"  proj_up_norm={gates.get('hg_internal_proj_up_norm', 0):.4f}")

                    _sim_pairs = [
                        ("conv","like"), ("conv","long"), ("conv","short"),
                        ("like","long"), ("like","short"), ("long","short"),
                    ]
                    sim_str = "  ".join(
                        f"{a[:2]}-{b[:2]}={gates.get(f'hg_internal_input_sim_{a}_{b}', 0):.3f}"
                        for a, b in _sim_pairs
                    )
                    print(f"  [Signal sim] {sim_str}")

                    proj_str = (
                        f"co-li={gates.get('hg_internal_proj_sim_co_li', 0):.3f}"
                        f"  co-lo={gates.get('hg_internal_proj_sim_co_lo', 0):.3f}"
                        f"  co-sh={gates.get('hg_internal_proj_sim_co_sh', 0):.3f}"
                    )
                    print(f"  [Proj sim]   {proj_str}")

                    for k, v in gates.items():
                        if isinstance(v, (int, float)):
                            train_log[f"hypergraph/{k}"] = v

            val_metrics = self.val_evaluator.evaluate(
                self.val_gt_ids, device, split="val"
            )
            val_log = {f"valid/{k}": v for k, v in val_metrics.items() if k != "loss"}
            self.logger.log({**train_log, **val_log})

            current = val_metrics.get(self.config.eval_metric, -float("inf"))
            if current > best_metric:
                prev_best = best_metric
                best_metric = current
                best_metrics = val_metrics
                patience_counter = 0
                self.save_checkpoint(ckpt_path)
                print(f"\n  ✓ {self.config.eval_metric} improved: {prev_best:.4f} → {current:.4f}"
                      f"  | checkpoint saved → {ckpt_path}")
            else:
                patience_counter += 1
                print(f"\n  ✗ {self.config.eval_metric}: {current:.4f}"
                      f"  (best: {best_metric:.4f}  patience: {patience_counter}/{self.config.patience})")

            self.save_checkpoint(
                ckpt_path,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                best_metrics=best_metrics,
                patience_counter=patience_counter,
                save_as_last=True,
            )

            if patience_counter >= self.config.patience:
                print(f"\n[Trainer] Early stopping at epoch {epoch + 1}")
                break

        print(f"\n{'='*64}")
        print(f"[Trainer] Loading best checkpoint from {ckpt_path}")
        self.load_checkpoint(ckpt_path, device)

        print(f"[Trainer] Running final test evaluation...")
        test_metrics = self.test_evaluator.evaluate(
            self.test_gt_ids, device, split="test"
        )
        self.logger.log({f"test/{k}": v for k, v in test_metrics.items()})
        print(f"{'='*64}\n")

        self.logger.finish()
        return test_metrics

    @torch.no_grad()
    def _encode_items(self, device: str) -> Tensor:

        self.model.eval()
        self.model.to(device)
        embeddings = []
        with autocast(device_type=device, dtype=torch.bfloat16, enabled=self.config.bf16):
            for batch in tqdm(self.item_dataloader, desc="  Encoding items", leave=False):
                batch = {k: v.to(device) if isinstance(v, Tensor) else v
                         for k, v in batch.items()}
                embeddings.append(self.model.encode_item(batch).cpu())
        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def _encode_queries(self, device: str) -> Tensor:
        self.model.eval()
        self.model.to(device)
        embeddings = []
        with autocast(device_type=device, dtype=torch.bfloat16, enabled=self.config.bf16):
            for batch in tqdm(self.train_infer_dataloader, desc="  Encoding queries", leave=False):
                _, _, input_dict = batch
                input_dict = {k: v.to(device) if isinstance(v, Tensor) else v
                              for k, v in input_dict.items()}
                embeddings.append(self.model.encode_query(input_dict).cpu())
        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def _encode_items_for_batch(self, item_indices: Tensor, device: str, chunk_size: int = 40) -> Tensor:
        all_embeddings = []
        for start in range(0, len(item_indices), chunk_size):
            chunk_ids = item_indices[start:start + chunk_size]
            samples = [self.item_dataset[int(i)] for i in chunk_ids]
            batch = self.item_dataset.collate_fn(samples)
            batch = {k: v.to(device) if isinstance(v, Tensor) else v
                     for k, v in batch.items()}
            all_embeddings.append(self.model.encode_item(batch))

        return torch.cat(all_embeddings, dim=0)

    def _log_grad_probe(self, step: int, epoch: int) -> None:
        mode = getattr(self.model, "mode", "")
        modules: Dict[str, torch.nn.Module] = {}
        if mode == "dynamic_gating":
            modules["gating"] = self.model.gating_network
        elif mode == "hypergraph":
            modules["hg_gating"] = self.model.gating_network
            modules["hypergraph"] = self.model.hypergraph_module
        if not modules:
            return

        print(f"\n  [GradProbe] epoch={epoch + 1}  step={step}")
        wandb_log: Dict[str, float] = {}
        for mod_name, module in modules.items():
            print(f"    [{mod_name}]")
            for p_name, param in module.named_parameters():
                if param.grad is not None:
                    gn = param.grad.norm().item()
                    print(f"      {p_name:50s}  {gn:.2e}")
                    wandb_log[f"grad/{mod_name}/{p_name}"] = gn
                else:
                    print(f"      {p_name:50s}  None")
        self.logger.log(wandb_log)

    def _probe_neg_sampling(
        self,
        query_emb: Tensor,
        item_emb: Tensor,
        neg_indices: Tensor,
    ) -> None:

        if query_emb.numel() == 0:
            print(f"  [Neg]  sampler={self.sampler.__class__.__name__}"
                  f"  n_neg={neg_indices.shape[1]}")
            return

        n = min(50, len(query_emb))
        q = query_emb[:n].float()              # [n, D]
        ni = neg_indices[:n]                     # [n, n_neg]
        ne = item_emb[ni].float()               # [n, n_neg, D]
        sim = (q.unsqueeze(1) * ne).sum(-1)      # [n, n_neg] — cosine sim (L2-normalized)

        print(f"  [Neg]  sampler={self.sampler.__class__.__name__}"
              f"  n_neg={neg_indices.shape[1]}"
              f"  sim_mean={sim.mean():.4f}"
              f"  sim_max={sim.max():.4f}"
              f"  sim_min={sim.min():.4f}")

    def _train_one_epoch(
        self,
        epoch: int,
        neg_indices: Tensor,         # [N, n_neg]
        item_emb_cache: Tensor,      # [M, D] CPU — neg sampling / epoch_encoding
        optimizer,
        scheduler,
        device: str,
    ) -> Dict:
        self.model.train()
        self.model.to(device)

        total_loss = 0.0
        total_aux_loss = 0.0
        total_pos_score = 0.0
        total_neg_score = 0.0
        total_grad_norm = 0.0
        total_head_grad_norm = 0.0
        n_steps = 0
        n_opt_steps = 0
        last_grad_norm = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        gate_acc: Dict[str, float] = {}

        optimizer.zero_grad()

        tq = tqdm(self.train_dataloader, desc=f"  Epoch {epoch + 1}", leave=False)
        for step, batch in enumerate(tq):
            query_ids, gt_ids, input_dict = batch
            input_dict = {k: v.to(device) if isinstance(v, Tensor) else v
                          for k, v in input_dict.items()}

            pos_idx = torch.tensor(gt_ids, dtype=torch.long).unsqueeze(1)   # [B, 1]
            neg_idx = neg_indices[query_ids]                                  # [B, n_neg]
            all_idx = torch.cat([pos_idx, neg_idx], dim=1)                   # [B, 1+n_neg]

            unique_ids, inverse = torch.unique(all_idx.flatten(), return_inverse=True)
            inverse = inverse.to(device)
            labels = torch.zeros(len(query_ids), dtype=torch.long, device=device)

            with autocast(device_type=device, dtype=torch.bfloat16, enabled=self.config.bf16):
                if self.config.epoch_encoding:
                    batch_item = item_emb_cache[all_idx.flatten()].to(device).view(
                        all_idx.shape[0], all_idx.shape[1], -1
                    )
                else:
                    unique_embs = self._encode_items_for_batch(unique_ids, device)   # [K, D]
                    batch_item = unique_embs[inverse].view(
                        all_idx.shape[0], all_idx.shape[1], -1
                    )                                                                 # [B, 1+n_neg, D]
                query_emb = self.model.encode_query(input_dict)                  # [B, D]

                batch_item_loss = batch_item
                main_loss = self.loss_fn(query_emb, batch_item_loss, labels)

                ib_loss_val = 0.0
                if self.config.use_inbatch_neg:
                    B_local = batch_item.shape[0]
                    if B_local > 1:
                        pos_emb = batch_item[:, 0]                               # [B, D]
                        ib_sim = (query_emb @ pos_emb.T) / self.config.temperature  # [B, B]
                        pos_item_ids = pos_idx.squeeze(1).to(device)             # [B]
                        same_item = pos_item_ids.unsqueeze(0) == pos_item_ids.unsqueeze(1)  # [B, B]
                        eye = torch.eye(B_local, dtype=torch.bool, device=device)
                        ib_sim = ib_sim.masked_fill(same_item & ~eye, float('-inf'))
                        ib_labels = torch.arange(B_local, device=device)
                        ib_loss = F.cross_entropy(ib_sim, ib_labels)
                        loss = main_loss + ib_loss
                        ib_loss_val = ib_loss.item()
                    else:
                        loss = main_loss
                else:
                    loss = main_loss

                aux_loss_val = 0.0
                hg_out = getattr(self.model, "_hg_out", None)
                if hg_out is not None and self.config.aux_weight > 0:
                    aux_loss = self.loss_fn(hg_out, batch_item_loss, labels)
                    loss = loss + self.config.aux_weight * aux_loss
                    aux_loss_val = aux_loss.item()

                with torch.no_grad():
                    pos_score = (query_emb * batch_item[:, 0]).sum(-1).mean().item()
                    neg_score = (query_emb.unsqueeze(1) * batch_item[:, 1:]).sum(-1).mean().item()

            (loss / self.config.accumulation_steps).backward()

            if step in (0, 100):
                self._log_grad_probe(step, epoch)

            total_loss += main_loss.item()
            total_aux_loss += aux_loss_val
            total_pos_score += pos_score
            total_neg_score += neg_score
            n_steps += 1

            probe = getattr(self.model, "_probe_gates", None)
            if probe:
                for k, v in probe.items():
                    gate_acc[k] = gate_acc.get(k, 0.0) + v

            log_dict = {"train_step_loss": main_loss.item()}
            if aux_loss_val > 0:
                log_dict["train_step_aux_loss"] = aux_loss_val

            if probe and getattr(self.model, "mode", "") == "dynamic_gating":
                for s in ["conv", "like", "long", "short", "dislike"]:
                    log_dict[f"gating/{s}_mean"] = probe.get(f"gate_{s}_mean", 0.0)
                    log_dict[f"gating/{s}_std"] = probe.get(f"gate_{s}_std", 0.0)
            elif probe and getattr(self.model, "mode", "") == "hypergraph":
                for k, v in probe.items():
                    if isinstance(v, (int, float)):
                        log_dict[f"hypergraph/{k}"] = v

            if (step + 1) % self.config.accumulation_steps == 0 or \
               (step + 1) == len(self.train_dataloader):
                lora_gn = torch.nn.utils.clip_grad_norm_(
                    self._lora_params, self.config.max_grad_norm
                ).item() if self._lora_params else 0.0
                head_gn = torch.nn.utils.clip_grad_norm_(
                    self._head_params, self.config.max_grad_norm
                ).item() if self._head_params else 0.0
                last_grad_norm = lora_gn

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_grad_norm += lora_gn
                total_head_grad_norm += head_gn
                n_opt_steps += 1
                current_lr = optimizer.param_groups[0]["lr"]
                log_dict["train/lr"] = current_lr
                log_dict["train/grad_norm_lora"] = lora_gn
                log_dict["train/grad_norm_head"] = head_gn

            self.logger.log(log_dict)

            tq.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.2e}",
                g_norm=f"{last_grad_norm:.3f}",
                pos_s=f"{pos_score:.3f}",
                neg_s=f"{neg_score:.3f}",
            )

        avg_loss = total_loss / max(n_steps, 1)
        avg_aux_loss = total_aux_loss / max(n_steps, 1)
        avg_pos_score = total_pos_score / max(n_steps, 1)
        avg_neg_score = total_neg_score / max(n_steps, 1)
        avg_grad_norm = total_grad_norm / max(n_opt_steps, 1)
        avg_head_grad_norm = total_head_grad_norm / max(n_opt_steps, 1)

        stats = {
            "loss": avg_loss,
            "aux_loss": avg_aux_loss,
            "pos_score": avg_pos_score,
            "neg_score": avg_neg_score,
            "score_gap": avg_pos_score - avg_neg_score,
            "grad_norm": avg_grad_norm,
            "grad_norm_head": avg_head_grad_norm,
            "lr": current_lr,
        }
        if gate_acc:
            stats["gates"] = {k: v / max(n_steps, 1) for k, v in gate_acc.items()}
        return stats

    def _setup_optimizer(self):
        lora_decay, lora_no_decay = [], []
        head_decay, head_no_decay = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = not name.startswith("backbone.")
            is_no_decay = param.ndim <= 1 or name.endswith(".bias")

            if is_head:
                (head_no_decay if is_no_decay else head_decay).append(param)
            else:
                (lora_no_decay if is_no_decay else lora_decay).append(param)

        self._lora_params = lora_decay + lora_no_decay
        self._head_params = head_decay + head_no_decay

        head_lr = self.config.learning_rate * self.config.head_lr_multiplier

        groups = [
            {"params": lora_decay, "weight_decay": self.config.weight_decay, "lr": self.config.learning_rate},
            {"params": lora_no_decay, "weight_decay": 0.0, "lr": self.config.learning_rate},
        ]
        if head_decay or head_no_decay:
            groups.append({"params": head_decay, "weight_decay": self.config.weight_decay, "lr": head_lr})
            groups.append({"params": head_no_decay, "weight_decay": 0.0, "lr": head_lr})
            print(f"  [Optimizer] LoRA lr={self.config.learning_rate:.1e}"
                  f"  |  Head lr={head_lr:.1e} (x{self.config.head_lr_multiplier})")

        return AdamW(groups)

    def _setup_scheduler(self, optimizer):
        n_batches_per_epoch = len(self.train_dataloader)
        n_opt_steps_per_epoch = math.ceil(n_batches_per_epoch / self.config.accumulation_steps)
        total_steps = n_opt_steps_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        return transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    def save_checkpoint(
        self,
        path: str,
        optimizer=None,
        scheduler=None,
        epoch: int | None = None,
        best_metric: float | None = None,
        best_metrics: Dict | None = None,
        patience_counter: int | None = None,
        save_as_last: bool = False,
    ) -> None:
        target_path = self._last_checkpoint_path(path) if save_as_last else path
        self.model.save(target_path)

        if not save_as_last:
            return

        trainer_state = {
            "next_epoch": (epoch + 1) if epoch is not None else 0,
            "best_metric": best_metric if best_metric is not None else -float("inf"),
            "best_metrics": best_metrics or {},
            "patience_counter": patience_counter or 0,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "rng_state": self._capture_rng_state(),
        }
        torch.save(trainer_state, self._trainer_state_path(target_path))
        print(f"[Trainer] Saved resume checkpoint → {target_path}")

    def load_checkpoint(
        self,
        path: str,
        device: str = "cpu",
        optimizer=None,
        scheduler=None,
        resume_training: bool = False,
    ) -> Dict:
        target_path = self._resolve_resume_path(path) if resume_training else path
        self.model.load(target_path, device)

        if not resume_training:
            return {}

        trainer_state_path = self._trainer_state_path(target_path)
        if not os.path.exists(trainer_state_path):
            if self.config.resume_epoch is None:
                raise ValueError(
                    f"{trainer_state_path} not found."
                )
            resume_step = self._epoch_to_step(self.config.resume_epoch)
            print(f"[Trainer] Legacy resumed.")
            print(f"  completed_epochs : {self.config.resume_epoch}")
            print(f"  resumed_step     : {resume_step}")
            return {
                "start_epoch": self.config.resume_epoch,
                "best_metric": (
                    float(self.config.resume_best_metric)
                    if self.config.resume_best_metric is not None
                    else -float("inf")
                ),
                "best_metrics": {},
                "patience_counter": int(self.config.resume_patience_counter),
                "checkpoint_path": target_path,
                "resume_mode": "manual_legacy",
                "resume_step": resume_step,
            }

        trainer_state = torch.load(trainer_state_path, map_location="cpu")

        if optimizer is not None and trainer_state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
        if scheduler is not None and trainer_state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(trainer_state["scheduler_state_dict"])

        self._restore_rng_state(trainer_state.get("rng_state"))
        return {
            "start_epoch": int(trainer_state.get("next_epoch", 0)),
            "best_metric": float(trainer_state.get("best_metric", -float("inf"))),
            "best_metrics": trainer_state.get("best_metrics", {}),
            "patience_counter": int(trainer_state.get("patience_counter", 0)),
            "checkpoint_path": target_path,
            "resume_mode": "full",
            "resume_step": 0,
        }

    def _last_checkpoint_path(self, path: str) -> str:
        return os.path.join(path, self.LAST_CKPT_DIRNAME)

    def _trainer_state_path(self, path: str) -> str:
        return os.path.join(path, self.TRAINER_STATE_FILENAME)

    def _resolve_resume_path(self, path: str) -> str:
        last_path = self._last_checkpoint_path(path)
        if os.path.exists(self._trainer_state_path(last_path)):
            return last_path
        if os.path.exists(self._trainer_state_path(path)):
            return path
        return path

    def _epoch_to_step(self, completed_epochs: int) -> int:
        n_batches_per_epoch = len(self.train_dataloader)
        n_opt_steps_per_epoch = math.ceil(n_batches_per_epoch / self.config.accumulation_steps)
        return completed_epochs * n_opt_steps_per_epoch

    def _prime_scheduler_to_step(self, scheduler, completed_steps: int) -> None:
        if completed_steps <= 0:
            return
        last_lrs = [
            base_lr * lr_lambda(completed_steps)
            for base_lr, lr_lambda in zip(scheduler.base_lrs, scheduler.lr_lambdas)
        ]
        state = scheduler.state_dict()
        state["last_epoch"] = completed_steps
        state["_step_count"] = completed_steps + 1
        state["_last_lr"] = last_lrs
        scheduler.load_state_dict(state)
        for param_group, lr in zip(scheduler.optimizer.param_groups, last_lrs):
            param_group["lr"] = lr

    def _capture_rng_state(self) -> Dict:
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        return rng_state

    def _restore_rng_state(self, rng_state: Dict | None) -> None:
        if not rng_state:
            return
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.random.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
