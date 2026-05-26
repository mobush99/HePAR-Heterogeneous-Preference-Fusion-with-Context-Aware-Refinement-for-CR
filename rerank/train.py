from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import SoftDeechoReranker
from rec_embedding_provider import RecEmbeddingProvider, RecEmbeddingProviderConfig
from trainer import (
    SoftDeechoTrainer,
    load_jsonl,
    load_reranker_checkpoint_data,
    save_reranker_checkpoint,
    validate_candidate_keys,
)


DEFAULT_TRAIN = "data/prep_inspired_nv_hg_seeded/coral_nv_hg_seed2024_inspired_train_top100.jsonl"
DEFAULT_VALID = "data/prep_inspired_nv_hg_seeded/coral_nv_hg_seed2024_inspired_valid_top100.jsonl"
DEFAULT_TEST = "data/prep_inspired_nv_hg_seeded/coral_nv_hg_seed2024_inspired_test_top100.jsonl"
DEFAULT_REC_ROOT = "../rec"
DEFAULT_REC_CHECKPOINT = "../rec/checkpoints/inspired/nv_hg_seed2024"
DEFAULT_RERANKER_CHECKPOINT = "outputs/soft_deecho.pt"
DEFAULT_DATA_NAME = "inspired"
DEFAULT_QUERY_USED_INFO = ["c", "l", "d", "x", "y"]
DEFAULT_SWEEP_LEARNING_RATES = [1e-4]
DEFAULT_SWEEP_EPOCHS = [15]
DEFAULT_SWEEP_DEECHO_HIDDEN_DIMS = [64]
DEFAULT_SWEEP_DEECHO_DROPOUTS = [0.0]
DEFAULT_SWEEP_DEECHO_APPLY_TEMPERATURES = [0.0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def rec_documents_path(rec_root: str | Path, data_name: str) -> Path:
    return Path(rec_root).resolve() / "data" / data_name / "processed_document.json"


def install_transformers_compat() -> None:
    try:
        from transformers import PreTrainedModel
    except Exception:
        return
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


def build_provider(args) -> RecEmbeddingProvider:
    install_transformers_compat()
    return RecEmbeddingProvider(
        RecEmbeddingProviderConfig(
            checkpoint_path=args.rec_checkpoint_path,
            encoder_impl=args.encoder_impl,
            data_name=args.data_name,
            rec_root=args.rec_root,
            documents_path=str(rec_documents_path(args.rec_root, args.data_name)),
            base_model_name=args.base_model_name,
            device=args.device,
            allow_cpu_fallback=True,
            batch_size=args.provider_batch_size,
            bf16=args.bf16,
            mode=args.rec_mode,
            query_used_info=tuple(args.query_used_info),
            doc_used_info=tuple(args.doc_used_info),
            item_max_length=args.item_max_length,
            n2e_routing_iters=args.n2e_routing_iters,
            use_global_edge=args.use_global_edge,
            flat_gating=args.flat_gating,
            alpha=args.alpha,
            beta=args.beta,
            delta=args.delta,
            epsilon=args.epsilon,
        )
    )


def build_reranker(
    args,
    deecho_hidden_dim: int | None = None,
    deecho_dropout: float | None = None,
    deecho_apply_temperature: float | None = None,
) -> SoftDeechoReranker:
    return SoftDeechoReranker(
        deecho_context_dim=args.deecho_context_dim,
        deecho_hidden_dim=deecho_hidden_dim if deecho_hidden_dim is not None else args.deecho_hidden_dim,
        deecho_dropout=deecho_dropout if deecho_dropout is not None else args.deecho_dropout,
        deecho_init_alpha=args.deecho_init_alpha,
        deecho_init_beta=args.deecho_init_beta,
        deecho_init_tau=args.deecho_init_tau,
        deecho_init_delta=args.deecho_init_delta,
        deecho_delta_radius=args.deecho_delta_radius,
        deecho_delta_mapping=args.deecho_delta_mapping,
        deecho_param_mode=args.deecho_param_mode,
        deecho_apply_mode=args.deecho_apply_mode,
        deecho_apply_temperature=(
            deecho_apply_temperature
            if deecho_apply_temperature is not None
            else args.deecho_apply_temperature
        ),
        deecho_force_alpha=args.deecho_force_alpha,
        deecho_force_beta=args.deecho_force_beta,
    )


def ensure_context_dim(args, provider: RecEmbeddingProvider, samples) -> None:
    if args.deecho_context_dim > 0:
        return
    if not samples:
        raise ValueError("cannot infer context dim from an empty sample set")
    encoded = provider.encode_samples(samples[:1])[0]["user"]
    if args.context_source == "user":
        value = encoded.get("user_embedding")
    elif args.context_source == "conv":
        value = (encoded.get("components") or {}).get("conv")
    else:
        raise ValueError(f"unsupported context source: {args.context_source}")
    if value is None:
        raise ValueError(f"provider did not return context embedding for {args.context_source}")
    args.deecho_context_dim = int(value.reshape(-1).shape[0])


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n[{title}]")
    for group in ("backbone", "reranked"):
        values = metrics.get(group, {})
        text = " ".join(f"{key}={value:.6f}" for key, value in sorted(values.items()))
        print(f"{group}: {text}")
    scalars = metrics.get("scalars", {})
    if scalars:
        print("scalars: " + " ".join(f"{key}={value:.6f}" for key, value in scalars.items()))


def print_controller_output_stats(title: str, stats: dict) -> None:
    if not stats:
        return
    keys = (
        "alpha_mean",
        "alpha_min",
        "alpha_max",
        "beta_mean",
        "beta_min",
        "beta_max",
        "tau_mean",
        "tau_min",
        "tau_max",
        "delta_mean",
        "delta_min",
        "delta_max",
    )
    text = " ".join(f"{key}={stats[key]:.6f}" for key in keys if key in stats)
    print(f"[{title}] {text}")


def reranker_metadata(
    args,
    deecho_hidden_dim: int | None = None,
    deecho_dropout: float | None = None,
    deecho_apply_temperature: float | None = None,
) -> dict[str, Any]:
    return {
        "reranker_model_version": "soft_deecho_v1",
        "reranker_mode": "soft_deecho",
        "context_source": args.context_source,
        "deecho_context_dim": args.deecho_context_dim,
        "deecho_hidden_dim": deecho_hidden_dim if deecho_hidden_dim is not None else args.deecho_hidden_dim,
        "deecho_dropout": deecho_dropout if deecho_dropout is not None else args.deecho_dropout,
        "deecho_init_alpha": args.deecho_init_alpha,
        "deecho_init_beta": args.deecho_init_beta,
        "deecho_init_tau": args.deecho_init_tau,
        "deecho_init_delta": args.deecho_init_delta,
        "deecho_delta_radius": args.deecho_delta_radius,
        "deecho_delta_mapping": args.deecho_delta_mapping,
        "deecho_param_mode": args.deecho_param_mode,
        "deecho_apply_mode": args.deecho_apply_mode,
        "deecho_apply_temperature": (
            deecho_apply_temperature
            if deecho_apply_temperature is not None
            else args.deecho_apply_temperature
        ),
        "deecho_force_alpha": args.deecho_force_alpha,
        "deecho_force_beta": args.deecho_force_beta,
    }


def checkpoint_path_for_metric(path: str | Path, metric: str, is_primary: bool) -> Path:
    path = Path(path)
    if is_primary:
        return path
    safe_metric = metric.replace("@", "").replace("/", "_")
    return path.with_name(f"{path.stem}_{safe_metric}{path.suffix}")


def build_reranker_from_checkpoint(args) -> tuple[SoftDeechoReranker, dict]:
    checkpoint = load_reranker_checkpoint_data(args.reranker_checkpoint_path, device=args.device)
    metadata = checkpoint.get("metadata", {})
    args.deecho_context_dim = int(metadata.get("deecho_context_dim", args.deecho_context_dim))
    args.deecho_hidden_dim = int(metadata.get("deecho_hidden_dim", args.deecho_hidden_dim))
    args.deecho_dropout = float(metadata.get("deecho_dropout", args.deecho_dropout))
    args.deecho_init_alpha = float(metadata.get("deecho_init_alpha", args.deecho_init_alpha))
    args.deecho_init_beta = float(metadata.get("deecho_init_beta", args.deecho_init_beta))
    args.deecho_init_tau = float(metadata.get("deecho_init_tau", args.deecho_init_tau))
    args.deecho_init_delta = float(metadata.get("deecho_init_delta", args.deecho_init_delta))
    args.deecho_delta_radius = float(metadata.get("deecho_delta_radius", args.deecho_delta_radius))
    args.deecho_delta_mapping = str(metadata.get("deecho_delta_mapping", args.deecho_delta_mapping))
    args.deecho_param_mode = str(metadata.get("deecho_param_mode", args.deecho_param_mode))
    args.deecho_apply_mode = str(metadata.get("deecho_apply_mode", args.deecho_apply_mode))
    args.deecho_apply_temperature = float(
        metadata.get("deecho_apply_temperature", args.deecho_apply_temperature)
    )
    args.deecho_force_alpha = metadata.get("deecho_force_alpha", args.deecho_force_alpha)
    args.deecho_force_beta = metadata.get("deecho_force_beta", args.deecho_force_beta)
    args.context_source = metadata.get("context_source", args.context_source)
    model = build_reranker(args)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    return model, metadata


def run_smoke(device: str) -> None:
    device = resolve_device(device)
    args = argparse.Namespace(
        deecho_context_dim=4,
        deecho_hidden_dim=8,
        deecho_dropout=0.0,
        deecho_init_alpha=0.1,
        deecho_init_beta=0.1,
        deecho_init_tau=0.5,
        deecho_init_delta=0.1,
        deecho_delta_radius=0.002,
        deecho_delta_mapping="bounded",
        deecho_param_mode="adaptive_mlp",
        deecho_apply_mode="near_tie",
        deecho_apply_temperature=0.0,
        deecho_force_alpha=None,
        deecho_force_beta=None,
    )
    model = build_reranker(args).to(device)
    base_score = torch.zeros(2, 3, device=device)
    candidate_emb = torch.randn(2, 3, 4, device=device)
    mentioned_mask = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=device)
    context_emb = torch.randn(2, 4, device=device)
    output = model(base_score, candidate_emb, mentioned_mask=mentioned_mask, context_emb=context_emb)
    assert output["final_score"].shape == base_score.shape
    print("SoftDeechoReranker smoke test passed.")


def truncate_candidates(samples, candidate_top_k: int | None):
    if candidate_top_k is None or candidate_top_k <= 0:
        return samples
    truncated = []
    for sample in samples:
        item = dict(sample)
        item["candidates"] = list(sample.get("candidates", []))[:candidate_top_k]
        truncated.append(item)
    return truncated


def load_and_validate(path: str, limit: int | None, documents_path: Path, candidate_top_k: int | None):
    samples = load_jsonl(path, limit=limit)
    samples = truncate_candidates(samples, candidate_top_k)
    validate_candidate_keys(samples, documents_path)
    return samples


def precompute_cache_path(args, split: str, sample_path: str, sample_count: int) -> Path | None:
    if not args.precompute_cache_dir:
        return None
    sample_stem = Path(sample_path).stem
    safe_sample_stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in sample_stem)
    top_k = f"top{args.candidate_top_k}" if args.candidate_top_k else "all"
    signature_fields = [
        "soft_deecho_v1",
        split,
        str(Path(sample_path).resolve()),
        str(sample_count),
        str(args.candidate_top_k),
        args.data_name,
        str(Path(args.rec_checkpoint_path).resolve()),
        str(args.encoder_impl),
        str(args.rec_mode),
        ",".join(args.query_used_info),
        ",".join(args.doc_used_info),
        str(args.item_max_length),
        str(args.n2e_routing_iters),
        str(args.use_global_edge),
        str(args.flat_gating),
        str(args.alpha),
        str(args.beta),
        str(args.delta),
        str(args.epsilon),
        args.context_source,
        args.precompute_cache_format,
        str(args.batch_size),
    ]
    digest = hashlib.sha1("|".join(signature_fields).encode("utf-8")).hexdigest()[:16]
    filename = (
        f"{split}__{safe_sample_stem}__{top_k}__n{sample_count}"
        f"__trainbs{args.batch_size}__{args.rec_mode}__{args.precompute_cache_format}__{digest}.pt"
    )
    return Path(args.precompute_cache_dir) / args.data_name / filename


def _validate_context_cache(batches, split: str) -> None:
    batch_items = batches.get("batches", []) if isinstance(batches, dict) and batches.get("format") == "compact_v1" else batches
    missing = [index for index, batch in enumerate(batch_items) if "context_emb" not in batch]
    if missing:
        raise ValueError(
            f"soft+deecho cache for {split} is missing context_emb in batch "
            f"{missing[0]}; rerun with --refresh-precompute-cache"
        )


def get_precomputed_batches(cache_builder, samples, args, split: str, sample_path: str):
    cache_path = precompute_cache_path(args, split, sample_path, len(samples))
    if cache_path is not None and cache_path.exists() and not args.refresh_precompute_cache:
        print(f"loading precomputed {split} batches: {cache_path}")
        batches = torch.load(cache_path, map_location="cpu", weights_only=False)
        _validate_context_cache(batches, split)
        return batches
    batches = cache_builder.precompute_batches(
        samples,
        desc=f"precompute {split}",
        precompute_batch_size=args.precompute_batch_size,
        compact=args.precompute_cache_format == "compact",
    )
    _validate_context_cache(batches, split)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(batches, cache_path)
        print(f"saved precomputed {split} batches: {cache_path}")
    return batches


def precomputed_batch_count(batches) -> int:
    if isinstance(batches, dict) and batches.get("format") == "compact_v1":
        return len(batches["batches"])
    return len(batches)


def build_trainer(args, provider, learning_rate: float | None = None, model=None) -> SoftDeechoTrainer:
    return SoftDeechoTrainer(
        model or build_reranker(args),
        provider,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=learning_rate if learning_rate is not None else args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        context_source=args.context_source,
    )


def load_splits(args):
    documents_path = rec_documents_path(args.rec_root, args.data_name)
    train_samples = load_and_validate(args.train_path, args.limit_train, documents_path, args.candidate_top_k)
    valid_samples = load_and_validate(args.valid_path, args.limit_valid, documents_path, args.candidate_top_k)
    test_samples = load_and_validate(args.test_path, args.limit_test, documents_path, args.candidate_top_k)
    return documents_path, train_samples, valid_samples, test_samples


def prepare_precomputed(args):
    documents_path, train_samples, valid_samples, test_samples = load_splits(args)
    provider = build_provider(args)
    ensure_context_dim(args, provider, train_samples)
    cache_builder = build_trainer(args, provider)
    print("precomputing frozen provider batches...")
    train_batches = get_precomputed_batches(cache_builder, train_samples, args, "train", args.train_path)
    valid_batches = get_precomputed_batches(cache_builder, valid_samples, args, "valid", args.valid_path)
    test_batches = get_precomputed_batches(cache_builder, test_samples, args, "test", args.test_path)
    print(
        f"precomputed batches: train={precomputed_batch_count(train_batches)} "
        f"valid={precomputed_batch_count(valid_batches)} test={precomputed_batch_count(test_batches)}"
    )
    return documents_path, provider, train_batches, valid_batches, test_batches


def run_train(args) -> None:
    documents_path, train_samples, valid_samples, test_samples = load_splits(args)
    provider = build_provider(args)
    ensure_context_dim(args, provider, train_samples)
    trainer = build_trainer(args, provider)
    print_controller_output_stats(
        "initial train deecho controller outputs before first optimizer step",
        trainer.controller_output_stats(train_samples),
    )
    fit_result = trainer.fit(train_samples, valid_samples, epochs=args.epochs, selection_metric=args.selection_metric)
    test_metrics = trainer.evaluate(test_samples)
    print_metrics("test", test_metrics)
    save_reranker_checkpoint(
        args.reranker_checkpoint_path,
        trainer.model,
        metadata={
            "best_metric": fit_result["best_metric"],
            "selection_metric": args.selection_metric,
            "rec_checkpoint_path": args.rec_checkpoint_path,
            "rec_root": str(Path(args.rec_root).resolve()),
            "data_name": args.data_name,
            "documents_path": str(documents_path),
            "candidate_top_k": args.candidate_top_k,
            **reranker_metadata(args),
        },
    )
    print(f"saved reranker checkpoint: {args.reranker_checkpoint_path}")


def run_eval(args) -> None:
    documents_path = rec_documents_path(args.rec_root, args.data_name)
    test_samples = load_and_validate(args.test_path, args.limit_test, documents_path, args.candidate_top_k)
    provider = build_provider(args)
    model, metadata = build_reranker_from_checkpoint(args)
    if metadata:
        print(f"loaded reranker metadata: {metadata}")
    trainer = build_trainer(args, provider, model=model)
    print_metrics("eval", trainer.evaluate(test_samples))


def run_precompute(args) -> None:
    prepare_precomputed(args)


def run_lr_sweep(args) -> None:
    documents_path, provider, train_batches, valid_batches, test_batches = prepare_precomputed(args)
    results = []
    best_result = None
    best_state = None
    for learning_rate in args.sweep_learning_rates:
        set_seed(args.seed)
        trainer = build_trainer(args, provider, learning_rate=learning_rate)
        print(f"\n[lr-sweep] lr={learning_rate:g} epochs={args.epochs}")
        print_controller_output_stats(
            f"initial train deecho controller outputs before first optimizer step lr={learning_rate:g}",
            trainer.controller_output_stats_precomputed(train_batches),
        )
        fit_result = trainer.fit_precomputed(
            train_batches,
            valid_batches,
            epochs=args.epochs,
            selection_metric=args.selection_metric,
        )
        test_metrics = trainer.evaluate_precomputed(test_batches)
        print_metrics(f"lr={learning_rate:g} test", test_metrics)
        result = {
            "learning_rate": learning_rate,
            "best_valid": fit_result["best_metric"],
            "test": test_metrics,
            "state": trainer.model.state_dict(),
        }
        results.append(result)
        if best_result is None or result["best_valid"] > best_result["best_valid"]:
            best_result = result
            best_state = trainer.model.state_dict()

    if best_result is not None and best_state is not None:
        model = build_reranker(args)
        model.load_state_dict(best_state)
        save_reranker_checkpoint(
            args.reranker_checkpoint_path,
            model,
            metadata={
                "best_metric": best_result["best_valid"],
                "best_learning_rate": best_result["learning_rate"],
                "selection_metric": args.selection_metric,
                "rec_checkpoint_path": args.rec_checkpoint_path,
                "rec_root": str(Path(args.rec_root).resolve()),
                "data_name": args.data_name,
                "documents_path": str(documents_path),
                "candidate_top_k": args.candidate_top_k,
                **reranker_metadata(args),
            },
        )
        print(f"saved best lr-sweep checkpoint: {args.reranker_checkpoint_path}")


def run_grid_sweep(args) -> None:
    documents_path, provider, train_batches, valid_batches, test_batches = prepare_precomputed(args)
    selection_metrics = args.selection_metrics or [args.selection_metric]
    results = []
    best_by_metric = {metric: {"result": None, "state": None} for metric in selection_metrics}
    grid = [
        (learning_rate, epochs, hidden_dim, dropout, apply_temperature)
        for learning_rate in args.sweep_learning_rates
        for epochs in args.sweep_epochs
        for hidden_dim in args.sweep_deecho_hidden_dims
        for dropout in args.sweep_deecho_dropouts
        for apply_temperature in args.sweep_deecho_apply_temperatures
    ]
    for index, (learning_rate, epochs, hidden_dim, dropout, apply_temperature) in enumerate(grid, start=1):
        set_seed(args.seed)
        model = build_reranker(args, hidden_dim, dropout, apply_temperature)
        trainer = build_trainer(args, provider, learning_rate=learning_rate, model=model)
        print(
            f"\n[grid {index}/{len(grid)}] lr={learning_rate:g} epochs={epochs} "
            f"deecho_hidden_dim={hidden_dim} deecho_dropout={dropout:g} "
            f"deecho_apply_mode={args.deecho_apply_mode} "
            f"deecho_apply_temperature={apply_temperature:g}"
        )
        print_controller_output_stats(
            f"initial train deecho controller outputs before first optimizer step grid={index}",
            trainer.controller_output_stats_precomputed(train_batches),
        )
        fit_result = trainer.fit_precomputed_multi_selection(
            train_batches,
            valid_batches,
            epochs=epochs,
            selection_metrics=selection_metrics,
            early_stop_metric=args.early_stop_metric,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_epochs=args.early_stop_min_epochs,
            early_stop_min_delta=args.early_stop_min_delta,
        )
        for metric in selection_metrics:
            metric_state = fit_result["best_by_metric"][metric]["best_state"]
            trainer.model.load_state_dict(metric_state)
            test_metrics = trainer.evaluate_precomputed(test_batches)
            print_metrics(
                f"lr={learning_rate:g} epochs={epochs} deecho_hidden_dim={hidden_dim} "
                f"deecho_dropout={dropout:g} deecho_apply_mode={args.deecho_apply_mode} "
                f"deecho_apply_temperature={apply_temperature:g} selected_by={metric} test",
                test_metrics,
            )
            result = {
                "selection_metric": metric,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "deecho_hidden_dim": hidden_dim,
                "deecho_dropout": dropout,
                "deecho_apply_mode": args.deecho_apply_mode,
                "deecho_apply_temperature": apply_temperature,
                "best_valid": fit_result["best_by_metric"][metric]["best_metric"],
                "best_epoch": fit_result["best_by_metric"][metric]["best_epoch"],
                "stopped_early": fit_result["stopped_early"],
                "stop_epoch": fit_result["stop_epoch"],
                "early_stop_metric": fit_result["early_stop_metric"],
                "early_stop_patience": fit_result["early_stop_patience"],
                "early_stop_min_epochs": fit_result["early_stop_min_epochs"],
                "early_stop_min_delta": fit_result["early_stop_min_delta"],
                "history": fit_result["history"],
                "test": test_metrics,
                "state": metric_state,
            }
            results.append(result)
            current = best_by_metric[metric]["result"]
            if current is None or result["best_valid"] > current["best_valid"]:
                best_by_metric[metric] = {"result": result, "state": metric_state}

    for index, metric in enumerate(selection_metrics):
        best_result = best_by_metric[metric]["result"]
        best_state = best_by_metric[metric]["state"]
        if best_result is None or best_state is None:
            continue
        model = build_reranker(
            args,
            best_result["deecho_hidden_dim"],
            best_result["deecho_dropout"],
            best_result["deecho_apply_temperature"],
        )
        model.load_state_dict(best_state)
        checkpoint_path = checkpoint_path_for_metric(args.reranker_checkpoint_path, metric, index == 0)
        save_reranker_checkpoint(
            checkpoint_path,
            model,
            metadata={
                "best_metric": best_result["best_valid"],
                "best_epoch": best_result["best_epoch"],
                "best_learning_rate": best_result["learning_rate"],
                "best_epochs": best_result["epochs"],
                "best_deecho_hidden_dim": best_result["deecho_hidden_dim"],
                "best_deecho_dropout": best_result["deecho_dropout"],
                "best_deecho_apply_mode": best_result["deecho_apply_mode"],
                "best_deecho_apply_temperature": best_result["deecho_apply_temperature"],
                "selection_metric": metric,
                "rec_checkpoint_path": args.rec_checkpoint_path,
                "rec_root": str(Path(args.rec_root).resolve()),
                "data_name": args.data_name,
                "documents_path": str(documents_path),
                "candidate_top_k": args.candidate_top_k,
                **reranker_metadata(
                    args,
                    best_result["deecho_hidden_dim"],
                    best_result["deecho_dropout"],
                    best_result["deecho_apply_temperature"],
                ),
            },
        )
        print(f"saved best grid-sweep checkpoint: {checkpoint_path} (selection_metric={metric})")

    if args.grid_results_path:
        results_path = Path(args.grid_results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "selection_metrics": selection_metrics,
            "results": [{key: value for key, value in result.items() if key != "state"} for result in results],
            "best_by_metric": {
                metric: {
                    key: value
                    for key, value in (best_by_metric[metric]["result"] or {}).items()
                    if key != "state"
                }
                for metric in selection_metrics
            },
        }
        results_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"saved grid results: {results_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "train", "eval", "precompute", "lr-sweep", "grid-sweep"],
        default="smoke",
    )
    parser.add_argument("--train-path", default=DEFAULT_TRAIN)
    parser.add_argument("--valid-path", default=DEFAULT_VALID)
    parser.add_argument("--test-path", default=DEFAULT_TEST)
    parser.add_argument("--data-name", default=DEFAULT_DATA_NAME)
    parser.add_argument("--rec-root", default=DEFAULT_REC_ROOT)
    parser.add_argument("--rec-checkpoint-path", default=DEFAULT_REC_CHECKPOINT)
    parser.add_argument("--reranker-checkpoint-path", default=DEFAULT_RERANKER_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--provider-batch-size", type=int, default=32)
    parser.add_argument("--precompute-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--selection-metric", default="NDCG@10")
    parser.add_argument("--selection-metrics", nargs="+", default=None)
    parser.add_argument("--early-stop-metric", default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-epochs", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--grid-results-path", default=None)
    parser.add_argument("--precompute-cache-dir", default=None)
    parser.add_argument("--precompute-cache-format", choices=["compact", "full"], default="compact")
    parser.add_argument("--refresh-precompute-cache", action="store_true")
    parser.add_argument("--candidate-top-k", type=int, default=50)
    parser.add_argument("--reranker-mode", choices=["soft_deecho"], default="soft_deecho")
    parser.add_argument("--context-source", choices=["user", "conv"], default="conv")
    parser.add_argument("--deecho-context-dim", type=int, default=0)
    parser.add_argument("--deecho-hidden-dim", type=int, default=64)
    parser.add_argument("--deecho-dropout", type=float, default=0.1)
    parser.add_argument("--sweep-deecho-hidden-dims", type=int, nargs="+", default=DEFAULT_SWEEP_DEECHO_HIDDEN_DIMS)
    parser.add_argument("--sweep-deecho-dropouts", type=float, nargs="+", default=DEFAULT_SWEEP_DEECHO_DROPOUTS)
    parser.add_argument("--deecho-init-alpha", type=float, default=0.05)
    parser.add_argument("--deecho-init-beta", type=float, default=0.05)
    parser.add_argument("--deecho-init-tau", type=float, default=0.5)
    parser.add_argument("--deecho-init-delta", type=float, default=0.1)
    parser.add_argument("--deecho-delta-radius", type=float, default=0.002)
    parser.add_argument("--deecho-delta-mapping", choices=["bounded", "sigmoid"], default="bounded")
    parser.add_argument("--deecho-param-mode", choices=["adaptive_mlp", "global_learnable", "fixed"], default="adaptive_mlp")
    parser.add_argument("--deecho-apply-mode", default="near_tie")
    parser.add_argument("--deecho-apply-temperature", type=float, default=0.0)
    parser.add_argument("--sweep-deecho-apply-temperatures", type=float, nargs="+", default=DEFAULT_SWEEP_DEECHO_APPLY_TEMPERATURES)
    parser.add_argument("--deecho-force-alpha", type=float, default=None)
    parser.add_argument("--deecho-force-beta", type=float, default=None)
    parser.add_argument("--deecho-warm-start-source", default="custom")
    parser.add_argument("--sweep-learning-rates", type=float, nargs="+", default=DEFAULT_SWEEP_LEARNING_RATES)
    parser.add_argument("--sweep-epochs", type=int, nargs="+", default=DEFAULT_SWEEP_EPOCHS)
    parser.add_argument("--encoder-impl", choices=["modern", "legacy"], default="modern")
    parser.add_argument("--base-model-name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--rec-mode", default="hypergraph")
    parser.add_argument("--rec-query-used-info", "--query-used-info", dest="query_used_info", nargs="+", default=DEFAULT_QUERY_USED_INFO)
    parser.add_argument("--doc-used-info", nargs="+", default=["m"])
    parser.add_argument("--item-max-length", type=int, default=128)
    parser.add_argument("--n2e-routing-iters", type=int, default=4)
    parser.add_argument("--use-global-edge", action="store_true")
    parser.add_argument("--no-flat-gating", dest="flat_gating", action="store_false")
    parser.set_defaults(flat_gating=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--bf16", dest="bf16", action="store_true")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"unsupported arguments in upload-clean version: {' '.join(unknown)}")
    return args


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    set_seed(args.seed)
    if args.mode == "smoke":
        run_smoke(args.device)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "precompute":
        run_precompute(args)
    elif args.mode == "lr-sweep":
        run_lr_sweep(args)
    elif args.mode == "grid-sweep":
        run_grid_sweep(args)
    else:
        raise ValueError(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
