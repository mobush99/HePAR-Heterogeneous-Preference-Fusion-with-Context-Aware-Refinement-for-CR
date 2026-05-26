from __future__ import annotations

import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor


YEAR_SUFFIX_RE = re.compile(r"\s*\((\d{4})\)\s*$")


def load_jsonl(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def item_key(item: Dict[str, Any]) -> Optional[str]:
    value = item.get("doc_key") or item.get("id") or item.get("title")
    return str(value) if value is not None else None


def _normalize_item_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).strip().lower().split())
    return text or None


def _strip_year_suffix(value: Any) -> Optional[str]:
    text = _normalize_item_text(value)
    if text is None:
        return None
    return YEAR_SUFFIX_RE.sub("", text).strip() or None


def _year_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return str(year) if 1000 <= year <= 9999 else None


def _candidate_identifier_forms(candidate: Dict[str, Any]) -> set[str]:
    forms: set[str] = set()
    for key in ("doc_key", "title", "id"):
        text = _normalize_item_text(candidate.get(key))
        if text:
            forms.add(text)
            stripped = _strip_year_suffix(text)
            if stripped:
                forms.add(stripped)
    year = _year_text(candidate.get("year"))
    title = _strip_year_suffix(candidate.get("title") or candidate.get("doc_key") or candidate.get("id"))
    if title and year:
        forms.add(f"{title} ({year})")
    return forms


def _mentioned_identifier_forms(sample: Dict[str, Any]) -> set[str]:
    forms: set[str] = set()
    mentioned = sample.get("mentioned_movies") or []
    if not isinstance(mentioned, list):
        return forms
    for item in mentioned:
        if not isinstance(item, dict):
            continue
        year = _year_text(item.get("year"))
        title_source = item.get("title") or item.get("doc_key") or item.get("id")
        title = _strip_year_suffix(title_source)
        if title and year:
            forms.add(f"{title} ({year})")
            continue
        for key in ("doc_key", "title", "id"):
            text = _normalize_item_text(item.get(key))
            if text:
                forms.add(text)
                stripped = _strip_year_suffix(text)
                if stripped:
                    forms.add(stripped)
    return forms


def candidate_keys(sample: Dict[str, Any]) -> List[str]:
    keys = []
    for candidate in sample.get("candidates", []):
        key = item_key(candidate)
        if key is not None:
            keys.append(key)
    return keys


def gt_keys(sample: Dict[str, Any]) -> List[str]:
    raw_gt = sample.get("gt")
    items: List[Any]
    if isinstance(raw_gt, list):
        items = raw_gt
    elif isinstance(raw_gt, dict):
        items = [raw_gt]
    else:
        items = []
    train_gt = sample.get("train_gt")
    if isinstance(train_gt, dict):
        items.append(train_gt)

    keys: List[str] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item_key(item)
        if key is not None and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def gt_count_for_sample(sample: Dict[str, Any]) -> int:
    return len(gt_keys(sample))


def labels_for_sample(sample: Dict[str, Any]) -> List[float]:
    positives = set(gt_keys(sample))
    return [1.0 if key in positives else 0.0 for key in candidate_keys(sample)]


def mentioned_mask_for_sample(sample: Dict[str, Any]) -> List[float]:
    mentioned = _mentioned_identifier_forms(sample)
    if not mentioned:
        return [0.0 for _ in sample.get("candidates", [])]
    mask = []
    for candidate in sample.get("candidates", []):
        mask.append(1.0 if _candidate_identifier_forms(candidate) & mentioned else 0.0)
    return mask


def validate_candidate_keys(samples: Sequence[Dict[str, Any]], documents_path: str | Path) -> None:
    path = Path(documents_path)
    with path.open() as handle:
        documents = json.load(handle)
    valid_keys = {str(key) for key in documents.keys()}

    missing: Dict[str, int] = {}
    for sample in samples:
        for key in candidate_keys(sample):
            if key not in valid_keys:
                missing[key] = missing.get(key, 0) + 1
    if missing:
        preview = ", ".join(f"{key} ({count})" for key, count in sorted(missing.items())[:10])
        raise KeyError(f"{len(missing)} unique candidate keys are missing from {path}: {preview}")


def _score_metrics(
    score: Tensor,
    label: Tensor,
    relevant_count: Optional[Tensor] = None,
    cutoffs: Sequence[int] = (1, 3, 5, 10, 50),
) -> Dict[str, float]:
    score = score.detach().float().cpu()
    label = label.detach().float().cpu()
    if relevant_count is None:
        relevant_count = label.sum(dim=-1)
    relevant_count = relevant_count.detach().float().cpu()
    order = torch.argsort(score, dim=-1, descending=True)
    sorted_label = torch.gather(label, 1, order)
    batch_size, num_candidates = label.shape
    metrics: Dict[str, float] = {}

    for cutoff in cutoffs:
        k = min(cutoff, num_candidates)
        top_label = sorted_label[:, :k]
        positions = torch.arange(1, k + 1, dtype=torch.float32)
        discounts = 1.0 / torch.log2(positions + 1.0)
        gains = torch.pow(2.0, top_label) - 1.0
        dcg = (gains * discounts).sum(dim=-1)

        ideal = torch.zeros(batch_size, k, dtype=torch.float32)
        for row, count in enumerate(relevant_count.long().tolist()):
            ideal[row, : min(k, count)] = 1.0
        idcg = ((torch.pow(2.0, ideal) - 1.0) * discounts).sum(dim=-1)
        ndcg = torch.where(idcg > 0, dcg / idcg.clamp_min(1e-8), torch.zeros_like(dcg))
        metrics[f"NDCG@{cutoff}"] = ndcg.mean().item() if batch_size else 0.0

        hits = top_label.sum(dim=-1)
        recall = torch.where(
            relevant_count > 0,
            hits / relevant_count.clamp_min(1.0),
            torch.zeros_like(hits),
        )
        metrics[f"Recall@{cutoff}"] = recall.mean().item() if batch_size else 0.0

    top5 = sorted_label[:, : min(5, num_candidates)]
    reciprocal_ranks = []
    for row in top5:
        rel = torch.nonzero(row > 0, as_tuple=False)
        reciprocal_ranks.append(0.0 if rel.numel() == 0 else 1.0 / float(rel[0].item() + 1))
    metrics["MRR@5"] = float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0
    return metrics


def _selection_score(metrics: Dict[str, float], selection_metric: str) -> float:
    if selection_metric == "BalancedR10":
        return (
            0.25 * float(metrics.get("Recall@1", 0.0))
            + float(metrics.get("Recall@5", 0.0))
            + 0.25 * float(metrics.get("Recall@10", 0.0))
        )
    return float(metrics.get(selection_metric, -math.inf))


def _context_embedding_from_user(user: Dict[str, Any], source: str) -> Tensor:
    if source == "user":
        value = user.get("user_embedding")
    elif source == "conv":
        value = (user.get("components") or {}).get("conv")
    else:
        raise ValueError(f"unsupported context source: {source}")
    if value is None:
        raise ValueError(f"provider did not return context embedding for source '{source}'")
    return value.detach().float().reshape(-1)


def _controller_stats(items: Sequence[Dict[str, Tensor]]) -> Dict[str, float]:
    if not items:
        return {}
    stats: Dict[str, float] = {}
    for name in ("alpha", "beta", "tau", "delta"):
        values = torch.cat([item[name].detach().float().reshape(-1).cpu() for item in items])
        stats[f"{name}_min"] = float(values.min().item())
        stats[f"{name}_mean"] = float(values.mean().item())
        stats[f"{name}_max"] = float(values.max().item())
    return stats


class SoftDeechoTrainer:
    def __init__(
        self,
        model,
        provider,
        device: str = "cpu",
        batch_size: int = 8,
        learning_rate: float = 1e-2,
        max_grad_norm: float = 1.0,
        context_source: str = "conv",
    ) -> None:
        self.model = model.to(device)
        self.provider = provider
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.context_source = context_source
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _iter_batches(
        self,
        samples: Sequence[Dict[str, Any]],
        shuffle: bool = False,
        batch_size: Optional[int] = None,
    ):
        batch_size = batch_size or self.batch_size
        indices = list(range(len(samples)))
        if shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield [samples[index] for index in indices[start:start + batch_size]]

    def prepare_batch(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        encoded = self.provider.encode_samples(samples)
        candidate_embeddings = torch.stack(
            [item["candidate_embeddings"].detach().float().to(self.device) for item in encoded],
            dim=0,
        )
        base_scores = torch.tensor(
            [
                [float(candidate.get("score", 0.0) or 0.0) for candidate in sample.get("candidates", [])]
                for sample in samples
            ],
            dtype=torch.float32,
            device=self.device,
        )
        labels = torch.tensor(
            [labels_for_sample(sample) for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        mentioned_mask = torch.tensor(
            [mentioned_mask_for_sample(sample) for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        relevant_counts = torch.tensor(
            [gt_count_for_sample(sample) for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        context_emb = torch.stack(
            [
                _context_embedding_from_user(item["user"], self.context_source).to(self.device)
                for item in encoded
            ],
            dim=0,
        )
        return {
            "base_score": base_scores,
            "candidate_emb": candidate_embeddings,
            "candidate_keys": [candidate_keys(sample) for sample in samples],
            "label": labels,
            "mentioned_mask": mentioned_mask,
            "relevant_count": relevant_counts,
            "context_emb": context_emb,
        }

    def _batch_to_cpu(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in batch.items()}

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _split_precomputed_batch(self, batch: Dict[str, Any], chunk_size: int) -> List[Dict[str, Any]]:
        batch_size = batch["base_score"].shape[0]
        chunks = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunks.append(
                {
                    key: (
                        value[start:end]
                        if torch.is_tensor(value) and value.shape[:1] == (batch_size,)
                        else value[start:end]
                        if isinstance(value, list) and len(value) == batch_size
                        else value
                    )
                    for key, value in batch.items()
                }
            )
        return chunks

    def _compact_precomputed_batch(
        self,
        batch: Dict[str, Any],
        item_to_index: Dict[str, int],
        item_keys: List[str],
        item_embeddings: List[Tensor],
    ) -> Dict[str, Any]:
        candidate_embeddings = batch["candidate_emb"]
        keys_by_sample = batch["candidate_keys"]
        candidate_indices: List[List[int]] = []
        for row, keys in enumerate(keys_by_sample):
            row_indices = []
            for col, key in enumerate(keys):
                if key not in item_to_index:
                    item_to_index[key] = len(item_keys)
                    item_keys.append(key)
                    item_embeddings.append(candidate_embeddings[row, col].detach().cpu())
                row_indices.append(item_to_index[key])
            candidate_indices.append(row_indices)

        compact = {key: value for key, value in batch.items() if key not in {"candidate_emb", "candidate_keys"}}
        compact["candidate_index"] = torch.tensor(candidate_indices, dtype=torch.long)
        return compact

    def compact_precomputed_batches(
        self,
        batches: Sequence[Dict[str, Any]],
        samples: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        item_to_index: Dict[str, int] = {}
        item_keys: List[str] = []
        item_embeddings: List[Tensor] = []
        compact_batches: List[Dict[str, Any]] = []
        sample_offset = 0
        for batch in batches:
            batch_size = batch["base_score"].shape[0]
            if "candidate_keys" not in batch:
                if samples is None:
                    raise ValueError("samples are required to compact legacy batches without candidate_keys")
                batch = dict(batch)
                batch["candidate_keys"] = [candidate_keys(sample) for sample in samples[sample_offset:sample_offset + batch_size]]
            if "mentioned_mask" not in batch and samples is not None:
                batch = dict(batch)
                batch["mentioned_mask"] = torch.tensor(
                    [mentioned_mask_for_sample(sample) for sample in samples[sample_offset:sample_offset + batch_size]],
                    dtype=torch.float32,
                )
            sample_offset += batch_size
            compact_batches.append(self._compact_precomputed_batch(batch, item_to_index, item_keys, item_embeddings))
        embedding_table = torch.stack(item_embeddings, dim=0).float().cpu() if item_embeddings else torch.empty(0)
        return {
            "format": "compact_v1",
            "item_keys": item_keys,
            "item_embeddings": embedding_table,
            "batches": compact_batches,
        }

    def precompute_batches(
        self,
        samples: Sequence[Dict[str, Any]],
        desc: str = "precompute",
        precompute_batch_size: Optional[int] = None,
        compact: bool = False,
    ):
        from tqdm import tqdm

        encode_batch_size = precompute_batch_size or self.batch_size
        total = math.ceil(len(samples) / encode_batch_size) if samples else 0
        batches = []
        for batch_samples in tqdm(
            self._iter_batches(samples, batch_size=encode_batch_size),
            total=total,
            desc=desc,
        ):
            encoded_batch = self._batch_to_cpu(self.prepare_batch(batch_samples))
            batches.extend(self._split_precomputed_batch(encoded_batch, self.batch_size))
        if compact:
            return self.compact_precomputed_batches(batches)
        return batches

    @staticmethod
    def _is_compact_precomputed(batches: Any) -> bool:
        return isinstance(batches, dict) and batches.get("format") == "compact_v1"

    def _materialize_compact_batch(self, store: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        materialized = {key: value for key, value in batch.items() if key != "candidate_index"}
        materialized["candidate_emb"] = store["item_embeddings"][batch["candidate_index"]]
        return materialized

    def _iter_precomputed_batches(self, batches: Any, shuffle: bool = False):
        batch_items = batches["batches"] if self._is_compact_precomputed(batches) else batches
        indices = list(range(len(batch_items)))
        if shuffle:
            random.shuffle(indices)
        for index in indices:
            batch = batch_items[index]
            if self._is_compact_precomputed(batches):
                batch = self._materialize_compact_batch(batches, batch)
            yield batch

    def _training_loss(self, output: Dict[str, Any], batch: Dict[str, Any]) -> tuple[Tensor, Dict[str, float]]:
        loss = self.model.compute_loss(output["score_for_loss"], batch["label"])
        components = getattr(self.model, "last_loss_components", {})
        return loss, {
            "loss": float(loss.detach().cpu()),
            "lambda_loss": float(components.get("lambda_loss", loss.detach().cpu())),
            "sample_count": float(batch["label"].shape[0]),
        }

    def train_epoch(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        self.model.train()
        losses: List[float] = []
        lambda_losses: List[float] = []
        for batch_samples in self._iter_batches(samples, shuffle=True):
            batch = self.prepare_batch(batch_samples)
            output = self.model(**batch)
            loss, loss_stats = self._training_loss(output, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss: {loss.item()}")
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            losses.append(loss_stats["loss"])
            lambda_losses.append(loss_stats["lambda_loss"])
        return {
            "loss": float(sum(losses) / len(losses)) if losses else 0.0,
            "lambda_loss": float(sum(lambda_losses) / len(lambda_losses)) if lambda_losses else 0.0,
        }

    def train_epoch_precomputed(self, batches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        self.model.train()
        losses: List[float] = []
        lambda_losses: List[float] = []
        for batch in self._iter_precomputed_batches(batches, shuffle=True):
            batch = self._batch_to_device(batch)
            output = self.model(**batch)
            loss, loss_stats = self._training_loss(output, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss: {loss.item()}")
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            losses.append(loss_stats["loss"])
            lambda_losses.append(loss_stats["lambda_loss"])
        return {
            "loss": float(sum(losses) / len(losses)) if losses else 0.0,
            "lambda_loss": float(sum(lambda_losses) / len(lambda_losses)) if lambda_losses else 0.0,
        }

    @torch.no_grad()
    def controller_output_stats(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        self.model.eval()
        snapshots: List[Dict[str, Tensor]] = []
        for batch_samples in self._iter_batches(samples, shuffle=False):
            batch = self.prepare_batch(batch_samples)
            output = self.model(**batch)
            snapshots.append(output["deecho_params"])
        return _controller_stats(snapshots)

    @torch.no_grad()
    def controller_output_stats_precomputed(self, batches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        self.model.eval()
        snapshots: List[Dict[str, Tensor]] = []
        for batch in self._iter_precomputed_batches(batches, shuffle=False):
            batch = self._batch_to_device(batch)
            output = self.model(**batch)
            snapshots.append(output["deecho_params"])
        return _controller_stats(snapshots)

    @torch.no_grad()
    def evaluate(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        self.model.eval()
        base_scores: List[Tensor] = []
        reranked_scores: List[Tensor] = []
        labels: List[Tensor] = []
        relevant_counts: List[Tensor] = []
        controller_snapshots: List[Dict[str, Tensor]] = []
        for batch_samples in self._iter_batches(samples, shuffle=False):
            batch = self.prepare_batch(batch_samples)
            output = self.model(**batch)
            if not torch.isfinite(output["final_score"]).all():
                raise FloatingPointError("non-finite final_score during evaluation")
            base_scores.append(batch["base_score"].detach().cpu())
            reranked_scores.append(output["calibrated_score"].detach().cpu())
            labels.append(batch["label"].detach().cpu())
            relevant_counts.append(batch["relevant_count"].detach().cpu())
            controller_snapshots.append(output["deecho_params"])
        return self._metrics_payload(base_scores, reranked_scores, labels, relevant_counts, controller_snapshots)

    @torch.no_grad()
    def evaluate_precomputed(self, batches: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        self.model.eval()
        base_scores: List[Tensor] = []
        reranked_scores: List[Tensor] = []
        labels: List[Tensor] = []
        relevant_counts: List[Tensor] = []
        controller_snapshots: List[Dict[str, Tensor]] = []
        for batch in self._iter_precomputed_batches(batches, shuffle=False):
            batch = self._batch_to_device(batch)
            output = self.model(**batch)
            if not torch.isfinite(output["final_score"]).all():
                raise FloatingPointError("non-finite final_score during evaluation")
            base_scores.append(batch["base_score"].detach().cpu())
            reranked_scores.append(output["calibrated_score"].detach().cpu())
            labels.append(batch["label"].detach().cpu())
            relevant_counts.append(batch["relevant_count"].detach().cpu())
            controller_snapshots.append(output["deecho_params"])
        return self._metrics_payload(base_scores, reranked_scores, labels, relevant_counts, controller_snapshots)

    @staticmethod
    def _metrics_payload(
        base_scores: Sequence[Tensor],
        reranked_scores: Sequence[Tensor],
        labels: Sequence[Tensor],
        relevant_counts: Sequence[Tensor],
        controller_snapshots: Sequence[Dict[str, Tensor]],
    ) -> Dict[str, Any]:
        if base_scores:
            base_score = torch.cat(list(base_scores), dim=0)
            reranked_score = torch.cat(list(reranked_scores), dim=0)
            label = torch.cat(list(labels), dim=0)
            relevant_count = torch.cat(list(relevant_counts), dim=0)
            backbone_metrics = _score_metrics(base_score, label, relevant_count)
            reranked_metrics = _score_metrics(reranked_score, label, relevant_count)
        else:
            backbone_metrics = {}
            reranked_metrics = {}
        return {
            "backbone": backbone_metrics,
            "reranked": reranked_metrics,
            "scalars": _controller_stats(controller_snapshots),
        }

    def fit(
        self,
        train_samples: Sequence[Dict[str, Any]],
        valid_samples: Sequence[Dict[str, Any]],
        epochs: int,
        selection_metric: str = "NDCG@10",
    ) -> Dict[str, Any]:
        best_state = copy.deepcopy(self.model.state_dict())
        best_metric = -math.inf
        history: List[Dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(train_samples)
            valid_stats = self.evaluate(valid_samples)
            score = _selection_score(valid_stats["reranked"], selection_metric)
            if score > best_metric:
                best_metric = score
                best_state = copy.deepcopy(self.model.state_dict())
            history.append({
                "epoch": epoch,
                "train": train_stats,
                "valid": valid_stats,
                "selection_metric": selection_metric,
                "selection_score": score,
            })
            print(
                f"epoch={epoch} loss={train_stats['loss']:.6f} "
                f"lambda={train_stats.get('lambda_loss', 0.0):.6f} "
                f"valid_{selection_metric}={score:.6f}"
            )
        self.model.load_state_dict(best_state)
        return {"history": history, "best_metric": best_metric, "best_state": best_state}

    def fit_precomputed(
        self,
        train_batches: Sequence[Dict[str, Any]],
        valid_batches: Sequence[Dict[str, Any]],
        epochs: int,
        selection_metric: str = "NDCG@10",
    ) -> Dict[str, Any]:
        best_state = copy.deepcopy(self.model.state_dict())
        best_metric = -math.inf
        history: List[Dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch_precomputed(train_batches)
            valid_stats = self.evaluate_precomputed(valid_batches)
            score = _selection_score(valid_stats["reranked"], selection_metric)
            if score > best_metric:
                best_metric = score
                best_state = copy.deepcopy(self.model.state_dict())
            history.append({
                "epoch": epoch,
                "train": train_stats,
                "valid": valid_stats,
                "selection_metric": selection_metric,
                "selection_score": score,
            })
            print(
                f"epoch={epoch} loss={train_stats['loss']:.6f} "
                f"lambda={train_stats.get('lambda_loss', 0.0):.6f} "
                f"valid_{selection_metric}={score:.6f}"
            )
        self.model.load_state_dict(best_state)
        return {"history": history, "best_metric": best_metric, "best_state": best_state}

    def fit_precomputed_multi_selection(
        self,
        train_batches: Sequence[Dict[str, Any]],
        valid_batches: Sequence[Dict[str, Any]],
        epochs: int,
        selection_metrics: Sequence[str],
        early_stop_metric: Optional[str] = None,
        early_stop_patience: Optional[int] = None,
        early_stop_min_epochs: int = 0,
        early_stop_min_delta: float = 0.0,
    ) -> Dict[str, Any]:
        if not selection_metrics:
            raise ValueError("selection_metrics must not be empty")
        if early_stop_metric is not None and early_stop_metric not in selection_metrics:
            raise ValueError("early_stop_metric must be one of selection_metrics")

        best_by_metric = {
            metric: {
                "best_metric": -math.inf,
                "best_state": copy.deepcopy(self.model.state_dict()),
                "best_epoch": 0,
            }
            for metric in selection_metrics
        }
        early_stop_enabled = early_stop_patience is not None and early_stop_patience > 0
        early_stop_metric = early_stop_metric or selection_metrics[0]
        early_stop_best = -math.inf
        early_stop_bad_epochs = 0
        stopped_early = False
        stop_epoch = epochs
        history: List[Dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch_precomputed(train_batches)
            valid_stats = self.evaluate_precomputed(valid_batches)
            scores = {}
            for metric in selection_metrics:
                score = _selection_score(valid_stats["reranked"], metric)
                scores[metric] = score
                if score > best_by_metric[metric]["best_metric"]:
                    best_by_metric[metric]["best_metric"] = score
                    best_by_metric[metric]["best_state"] = copy.deepcopy(self.model.state_dict())
                    best_by_metric[metric]["best_epoch"] = epoch
            history.append({
                "epoch": epoch,
                "train": train_stats,
                "valid": valid_stats,
                "selection_scores": scores,
            })
            valid_text = " ".join(f"valid_{metric}={score:.6f}" for metric, score in scores.items())
            print(
                f"epoch={epoch} loss={train_stats['loss']:.6f} "
                f"lambda={train_stats.get('lambda_loss', 0.0):.6f} "
                f"{valid_text}"
            )
            if early_stop_enabled:
                early_score = scores[early_stop_metric]
                if early_score > early_stop_best + early_stop_min_delta:
                    early_stop_best = early_score
                    early_stop_bad_epochs = 0
                else:
                    early_stop_bad_epochs += 1
                if epoch >= early_stop_min_epochs and early_stop_bad_epochs >= early_stop_patience:
                    stopped_early = True
                    stop_epoch = epoch
                    break

        primary_metric = selection_metrics[0]
        final_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_by_metric[primary_metric]["best_state"])
        return {
            "history": history,
            "best_by_metric": best_by_metric,
            "final_state": final_state,
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch,
            "early_stop_metric": early_stop_metric if early_stop_enabled else None,
            "early_stop_patience": early_stop_patience if early_stop_enabled else None,
            "early_stop_min_epochs": early_stop_min_epochs if early_stop_enabled else None,
            "early_stop_min_delta": early_stop_min_delta if early_stop_enabled else None,
        }


def save_reranker_checkpoint(path: str | Path, model, metadata: Optional[Dict[str, Any]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata or {}}, path)


def load_reranker_checkpoint_data(path: str | Path, device: str = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"reranker checkpoint does not exist: {path}")
    return torch.load(path, map_location=device)


def load_reranker_checkpoint(path: str | Path, model, device: str = "cpu") -> Dict[str, Any]:
    checkpoint = load_reranker_checkpoint_data(path, device=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    return checkpoint.get("metadata", {})
