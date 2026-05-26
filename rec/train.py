import argparse
import os
import random
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
from wonderwords import RandomWord

from core.config import TrainingConfig
from core.evaluator import Evaluator
from core.logger import build_logger
from core.loss import InfoNCELoss
from core.metric import MetricComputer
from core.sampler import HardNegativeSampler, MixedNegativeSampler, RandomNegativeSampler
from src.trainer import Trainer
from src.data_utils import build_dataloaders, load_rec_datasets
from src.model import BackboneEncoder


def parse_args():
    parser = argparse.ArgumentParser()

    # ── Data ──────────────────────────────────────────────
    parser.add_argument("--data_dir",           type=str,   default="data")
    parser.add_argument("--data_name",          type=str,   required=True,
                        choices=["inspired", "redial"])
    parser.add_argument("--query_used_info",    type=str,   nargs="+",
                        default=["c", "l", "d", "a", "b"],
                        choices=["c", "l", "d", "a", "b", "x", "y"])
    parser.add_argument("--doc_used_info",      type=str,   nargs="+",
                        default=["m"])
    parser.add_argument("--base_model_name",    type=str,   default="nvidia/NV-Embed-v2")
    parser.add_argument("--query_max_length",   type=int,   default=512)
    parser.add_argument("--aux_max_length",     type=int,   default=256)
    parser.add_argument("--item_max_length",    type=int,   default=512)
    parser.add_argument("--num_workers",        type=int,   default=4)

    # ── Device ────────────────────────────────────────────
    parser.add_argument("--device",             type=str,   default="cuda")

    # ── Model ─────────────────────────────────────────────
    parser.add_argument("--mode",               type=str,   default="vanilla",
                        choices=["vanilla", "dynamic_gating", "hypergraph"])
    parser.add_argument("--alpha",              type=float, default=0.5)
    parser.add_argument("--beta",               type=float, default=0.2)
    parser.add_argument("--delta",              type=float, default=0.2)
    parser.add_argument("--epsilon",            type=float, default=0.5)
    parser.add_argument("--uniform_gate_init",  action="store_true",
                        help="DG mode: initialize all gate biases to 0 (sigmoid=0.5)")
    parser.add_argument("--hg_gate_activation", type=str, default="sigmoid",
                        choices=["softmax", "sigmoid"],
                        help="HG gating network activation (softmax=legacy, sigmoid=DG-style)")
    parser.add_argument("--hg_only", action="store_true",
                        help="HG mode: use only HG expert (no raw expert blend)")
    parser.add_argument("--n2e_routing_iters", type=int, default=2,
                        help="N2E dynamic routing iterations (default: 2)")
    parser.add_argument("--no_n2n", action="store_true",
                        help="HG mode: disable N2N attention")
    parser.add_argument("--no_global_edge", action="store_true",
                        help="HG mode: remove global edge (8→7 edges)")
    parser.add_argument("--flat_gating", action="store_true",
                        help="HG mode: flat 9-signal gating (no split_ratio)")
    parser.add_argument("--fixed_split_ratio", type=float, default=None,
                        help="HG mode: override split_ratio with fixed value (test_only sweep)")
    parser.add_argument("--zero_hg_gates", action="store_true",
                        help="HG flat_gating: zero out HG gates at test time (ablation)")

    # ── Training ──────────────────────────────────────────
    parser.add_argument("--seed",               type=int,   default=2024)
    parser.add_argument("--epochs",             type=int,   default=100)
    parser.add_argument("--batch_size",         type=int,   default=16)
    parser.add_argument("--accumulation_steps", type=int,   default=1)
    parser.add_argument("--epoch_encoding",     action="store_true",
                        help="Use epoch-level cached item embeddings for train loss")
    parser.add_argument("--bf16",               action="store_true")

    # ── Optimizer ─────────────────────────────────────────
    parser.add_argument("--learning_rate",      type=float, default=1e-4)
    parser.add_argument("--weight_decay",       type=float, default=0.0)

    # ── Scheduler ─────────────────────────────────────────
    parser.add_argument("--warmup_ratio",       type=float, default=0.1)

    # ── Gradient clipping ─────────────────────────────────
    parser.add_argument("--max_grad_norm",      type=float, default=1.0)
    parser.add_argument("--head_lr_multiplier", type=float, default=1.0,
                        help="HG/gating head lr = learning_rate * multiplier")

    # ── Early stopping ────────────────────────────────────
    parser.add_argument("--patience",           type=int,   default=10)

    # ── Loss ──────────────────────────────────────────────
    parser.add_argument("--temperature",        type=float, default=0.05)
    parser.add_argument("--n_negatives",        type=int,   default=15)
    parser.add_argument("--aux_weight",         type=float, default=0.0,
                        help="HG auxiliary contrastive loss weight (0=disabled)")
    parser.add_argument("--use_inbatch_neg",    action="store_true",
                        help="in-batch negative")

    # ── Sampler ───────────────────────────────────────────
    parser.add_argument("--train_data_ratio",   type=float, default=1.0,
                        help="Ratio of training data to use (0.0-1.0)")
    parser.add_argument("--train_split_seed",   type=int,   default=2024,
                        help="Seed used only for train_data_ratio subsampling")
    parser.add_argument("--sampler",            type=str,   default="hard",
                        choices=["random", "hard", "mixed"])
    parser.add_argument("--n_hard_negatives",   type=int,   default=4,
                        help="(mixed sampler) hard negatives per query")
    parser.add_argument("--skip_top",           type=int,   default=0,
                        help="(mixed sampler) skip this many top-ranked items "
                             "to avoid false hard negatives")

    # ── Evaluation ────────────────────────────────────────
    parser.add_argument("--cutoffs",    type=int, nargs="+", default=[1, 5, 10, 50, 100])
    parser.add_argument("--eval_metric",        type=str,   default="NDCG@10")

    # ── Score fusion (eval-time only) ────────────────────
    parser.add_argument("--score_fusion_weight", type=float, default=0.0,
                        help="λ for score fusion: score = main + λ·hg (0=disabled)")
    parser.add_argument("--score_fusion_search", action="store_true",
                        help="Grid search λ on val set (test_only mode)")

    # ── Checkpoint ────────────────────────────────────────
    parser.add_argument("--ckpt_dir",           type=str,   default="checkpoints")
    parser.add_argument("--test_only",          action="store_true")
    parser.add_argument("--resume",             action="store_true",
                        help="Resume training from the checkpoint directory passed via --ckpt_path")
    parser.add_argument("--resume_epoch",       type=int,   default=None,
                        help="Legacy/manual resume: completed epoch count in the checkpoint (e.g. 20 -> resume from epoch 21)")
    parser.add_argument("--resume_best_metric", type=float, default=None,
                        help="Legacy/manual resume: best validation metric seen before interruption")
    parser.add_argument("--resume_patience_counter", type=int, default=0,
                        help="Legacy/manual resume: early-stopping patience counter before interruption")
    parser.add_argument("--ckpt_path",          type=str,   default=None)
    parser.add_argument("--first_only",         action="store_true")

    # ── Logging (WandB) ───────────────────────────────────
    parser.add_argument("--wandb_project",      type=str,   default=None)
    parser.add_argument("--wandb_entity",       type=str,   default=None)
    parser.add_argument("--wandb_group",        type=str,   default=None)
    parser.add_argument("--wandb_run_name",     type=str,   default=None)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _generate_run_name() -> str:
    rw = RandomWord()
    while True:
        word = rw.random_words(include_parts_of_speech=["noun", "verb"])[0]
        if " " not in word and "-" not in word:
            break
    return f"{word}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.test_only and args.resume:
        raise ValueError("Can't use --test_only and --resume at the same time.")

    # ── Run name / Checkpoint dir ─────────────────────────────────────────────
    if args.resume:
        assert args.ckpt_path is not None, "[Hyperparameter Check] NEED --ckpt_path for --resume"
        ckpt_dir = args.ckpt_path.rstrip("/\\")
        run_name = args.wandb_run_name or os.path.basename(ckpt_dir)
    else:
        run_name = args.wandb_run_name or _generate_run_name()
        ckpt_dir = os.path.join(args.ckpt_dir, args.data_name, run_name)

    # ── Config / Logger ───────────────────────────────────────────────────────
    config = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        epoch_encoding=args.epoch_encoding,
        bf16=args.bf16,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        head_lr_multiplier=args.head_lr_multiplier,
        patience=args.patience,
        temperature=args.temperature,
        n_negatives=args.n_negatives,
        aux_weight=args.aux_weight,
        use_inbatch_neg=args.use_inbatch_neg,
        cutoffs=args.cutoffs,
        eval_metric=args.eval_metric,
        ckpt_dir=ckpt_dir,
        resume=args.resume,
        resume_epoch=args.resume_epoch,
        resume_best_metric=args.resume_best_metric,
        resume_patience_counter=args.resume_patience_counter,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
    )

    logger = build_logger(
        config=asdict(config),
        use_wandb=args.wandb_project is not None,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=run_name,
        group=args.wandb_group,
    )

    print(f"[train] Loading data: {args.data_name}")
    data = load_rec_datasets(
        data_dir=args.data_dir,
        data_name=args.data_name,
        query_used_info=args.query_used_info,
        doc_used_info=args.doc_used_info,
        tokenizer_name=args.base_model_name,
        query_max_length=args.query_max_length,
        aux_max_length=args.aux_max_length,
        item_max_length=args.item_max_length,
        train_data_ratio=args.train_data_ratio,
        train_split_seed=args.train_split_seed,
    )

    train_dl, val_dl, test_dl, train_infer_dl, item_dl = build_dataloaders(
        data=data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.first_only:
        data["test_gt_ids"] = [ids[0] for ids in data["test_gt_ids"]]

    print(f"[train] Initializing BackboneEncoder (mode={args.mode})")
    model = BackboneEncoder(
        base_model_name=args.base_model_name,
        mode=args.mode,
        query_used_info=tuple(args.query_used_info),
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        epsilon=args.epsilon,
        seed=args.seed,
        uniform_gate_init=args.uniform_gate_init,
        gate_activation=args.hg_gate_activation,
        hg_only=args.hg_only,
        num_routing_iters=args.n2e_routing_iters,
        use_n2n=not args.no_n2n,
        use_global_edge=not args.no_global_edge,
        flat_gating=args.flat_gating,
    )

    loss_fn = InfoNCELoss(temperature=args.temperature)
    if args.sampler == "hard":
        sampler = HardNegativeSampler()
    elif args.sampler == "mixed":
        sampler = MixedNegativeSampler(n_hard=args.n_hard_negatives, skip_top=args.skip_top)
    else:
        sampler = RandomNegativeSampler()
    metric_computer = MetricComputer(cutoffs=args.cutoffs)

    val_evaluator = Evaluator(
        model=model,
        query_dataloader=val_dl,
        item_dataloader=item_dl,
        metric_computer=metric_computer,
        logger=logger,
        bf16=args.bf16,
        temperature=args.temperature,
        score_fusion_weight=args.score_fusion_weight,
    )
    test_evaluator = Evaluator(
        model=model,
        query_dataloader=test_dl,
        item_dataloader=item_dl,
        metric_computer=metric_computer,
        logger=logger,
        bf16=args.bf16,
        temperature=args.temperature,
        score_fusion_weight=args.score_fusion_weight,
    )

    if args.test_only:
        assert args.ckpt_path is not None, "--ckpt_path is required for --test_only "
        print(f"[test_only] Loading checkpoint: {args.ckpt_path}")
        model.load(args.ckpt_path, device=args.device)

        if args.fixed_split_ratio is not None:
            model._fixed_split_ratio = args.fixed_split_ratio
            print(f"[test_only] Fixed split_ratio = {args.fixed_split_ratio}")
        else:
            model._fixed_split_ratio = None

        model._zero_hg_gates = args.zero_hg_gates
        if args.zero_hg_gates:
            print("[test_only] HG gates zeroed out (raw-only ablation)")

        print("[test_only] Running test evaluation...")
        test_metrics = test_evaluator.evaluate(
            gt_ids=data["test_gt_ids"],
            device=args.device,
            split="test",
        )
        print("[test_only] Done. Test metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        logger.log({f"test/{k}": v for k, v in test_metrics.items()})
        logger.finish()
        return

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dl,
        train_infer_dataloader=train_infer_dl,
        item_dataloader=item_dl,
        val_evaluator=val_evaluator,
        test_evaluator=test_evaluator,
        val_gt_ids=data["val_gt_ids"],
        test_gt_ids=data["test_gt_ids"],
        gt_mask=data["gt_mask"],
        sampler=sampler,
        loss_fn=loss_fn,
        logger=logger,
    )

    print(f"[train] Starting training on {args.device}")
    test_metrics = trainer.train(device=args.device)
    print("[train] Done. Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
