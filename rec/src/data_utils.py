from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from utils import load_document, load_query
from src.dataset import RecItemDataset, RecQueryDataset

def _str_gt_to_int(
    gt_ids_str: List,                      # List[str] | List[List[str]]
    id_to_idx: Dict[str, int],
) -> List:
    if isinstance(gt_ids_str[0], list):
        return [[id_to_idx[g] for g in gs] for gs in gt_ids_str]
    return [id_to_idx[g] for g in gt_ids_str]


def _build_gt_mask(
    gt_ids_per_query: List[List[int]],
    n_items: int,
) -> Tensor:
    mask = torch.zeros(len(gt_ids_per_query), n_items, dtype=torch.bool)
    for i, ids in enumerate(gt_ids_per_query):
        mask[i, ids] = True
    return mask

def load_rec_datasets(
    data_dir: str,
    data_name: str,
    query_used_info: List[str],
    doc_used_info: List[str],
    tokenizer_name: str,
    query_max_length: int = 512,
    aux_max_length: int = 256,
    item_max_length: int = 1024,
    train_data_ratio: float = 1.0,
    train_split_seed: Optional[int] = 2024,
) -> Dict:
    data_dir = Path(data_dir) / data_name

    item_ids, doc_dict = load_document(
        data_dir / "processed_document.json",
        data_name,
        used_info=doc_used_info,
    )
    id_to_idx: Dict[str, int] = {item_id: idx for idx, item_id in enumerate(item_ids)}
    item_dataset = RecItemDataset(
        item_texts=doc_dict["item"],
        tokenizer_name=tokenizer_name,
        max_length=item_max_length,
    )

    train_gt_str, train_conv_gt_str, train_query_dict = load_query(
        data_dir / "augmented_train_v2.jsonl",
        used_info=query_used_info,
    )
    train_gt_ids = _str_gt_to_int(train_gt_str, id_to_idx)          # List[int]
    train_conv_gt_ids = _str_gt_to_int(train_conv_gt_str, id_to_idx)  # List[List[int]]

    if train_data_ratio < 1.0:
        original_size = len(train_gt_ids)
        num_samples = int(original_size * train_data_ratio)
        if train_split_seed is None:
            indices = np.random.choice(original_size, num_samples, replace=False)
            seed_note = "global"
        else:
            rng = np.random.default_rng(train_split_seed)
            indices = rng.choice(original_size, num_samples, replace=False)
            seed_note = str(train_split_seed)
        indices = sorted(indices.tolist())
        train_gt_ids = [train_gt_ids[i] for i in indices]
        train_conv_gt_ids = [train_conv_gt_ids[i] for i in indices]
        for key in train_query_dict:
            train_query_dict[key] = [train_query_dict[key][i] for i in indices]
        print(
            f"[Data] Train subsampling: {num_samples}/{original_size} "
            f"({train_data_ratio*100:.1f}%, split_seed={seed_note})"
        )
    else:
        print(f"[Data] Train full: {len(train_gt_ids)} samples")

    train_dataset = RecQueryDataset(
        signal_texts=train_query_dict,
        gt_ids=train_gt_ids,
        tokenizer_name=tokenizer_name,
        max_length=query_max_length,
        aux_max_length=aux_max_length,
    )

    val_gt_str, _, val_query_dict = load_query(
        data_dir / "augmented_valid_v2.jsonl",
        used_info=query_used_info,
    )
    val_gt_ids = _str_gt_to_int(val_gt_str, id_to_idx)              # List[int]
    val_dataset = RecQueryDataset(
        signal_texts=val_query_dict,
        gt_ids=val_gt_ids,
        tokenizer_name=tokenizer_name,
        max_length=query_max_length,
        aux_max_length=aux_max_length,
    )

    test_gt_str, _, test_query_dict = load_query(
        data_dir / "augmented_test_v2.jsonl",
        used_info=query_used_info,
    )
    test_gt_ids = _str_gt_to_int(test_gt_str, id_to_idx)            # List[List[int]]
    test_dataset = RecQueryDataset(
        signal_texts=test_query_dict,
        gt_ids=test_gt_ids,
        tokenizer_name=tokenizer_name,
        max_length=query_max_length,
        aux_max_length=aux_max_length,
    )

    gt_mask = _build_gt_mask(train_conv_gt_ids, n_items=len(item_dataset))

    return {
        "item_dataset":    item_dataset,
        "train_dataset":   train_dataset,
        "val_dataset":     val_dataset,
        "test_dataset":    test_dataset,
        "train_gt_ids":    train_gt_ids,
        "val_gt_ids":      val_gt_ids,
        "test_gt_ids":     test_gt_ids,
        "gt_mask":         gt_mask,
        "id_to_idx":       id_to_idx,
        "item_ids":        item_ids,
    }



def build_dataloaders(
    data: Dict,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:

    train_ds   = data["train_dataset"]
    val_ds     = data["val_dataset"]
    test_ds    = data["test_dataset"]
    item_ds    = data["item_dataset"]

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
        num_workers=num_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_ds.collate_fn,
        num_workers=num_workers,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_ds.collate_fn,
        num_workers=num_workers,
    )
    train_infer_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_ds.collate_fn,
        num_workers=num_workers,
    )
    item_dl = DataLoader(
        item_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=item_ds.collate_fn,
        num_workers=num_workers,
    )

    return train_dl, val_dl, test_dl, train_infer_dl, item_dl
