from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 16
    accumulation_steps: int = 1          # gradient accumulation
    epoch_encoding: bool = False         # use epoch-level cached item embeddings for train loss
    bf16: bool = False                   # bfloat16 mixed precision
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    warmup_ratio: float = 0.1           # warmup steps = total_steps * warmup_ratio

    max_grad_norm: float = 1.0
    head_lr_multiplier: float = 1.0       # HG/gating head lr = learning_rate * multiplier

    patience: int = 10                  # patience epochs without improvement

    temperature: float = 0.05
    n_negatives: int = 16               # negatives per positive item
    use_inbatch_neg: bool = False       # use other positives in the batch as negatives

    cutoffs: List[int] = field(default_factory=lambda: [1, 5, 10, 50, 100])
    eval_metric: str = "NDCG@10"        # early stopping metric key

    ckpt_dir: str = "checkpoints"
    resume: bool = False
    resume_epoch: Optional[int] = None
    resume_best_metric: Optional[float] = None
    resume_patience_counter: int = 0

    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_run_name: Optional[str] = None
