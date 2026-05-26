from typing import Dict, List, Tuple, Union

import torch
from transformers import AutoTokenizer

from core.dataset import ItemDataset, QueryDataset

_SIGNAL_PREFIX: Dict[str, str] = {
    "c": "conv",
    "l": "like",
    "d": "dislike",
    "x": "long",
    "y": "short",
}

_STELLA_S2P_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query.\n"
    "Query: "
)

def _uses_stella_tokenizer(tokenizer_name: str) -> bool:
    name = tokenizer_name.lower()
    return "stella_en_" in name and "_v5" in name


def _pad_sequences(seqs: List[torch.Tensor], padding_value: int, padding_side: str) -> torch.Tensor:
    if padding_side == "right":
        return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=padding_value)

    max_len = max(seq.size(0) for seq in seqs)
    out = seqs[0].new_full((len(seqs), max_len), padding_value)
    for idx, seq in enumerate(seqs):
        out[idx, -seq.size(0):] = seq
    return out


class RecItemDataset(ItemDataset):
    def __init__(
        self,
        item_texts: List[str],
        tokenizer_name: str = "nvidia/NV-Embed-v2",
        max_length: int = 1024,
    ) -> None:
        self.item_texts = item_texts
        self.max_length = max_length
        self.item_prefix = _E5_PASSAGE_PREFIX if _uses_e5_tokenizer(tokenizer_name) else ""
        self.padding_side = "left" if _uses_last_token_tokenizer(tokenizer_name) else "right"
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, truncation_side="right", padding_side=self.padding_side, trust_remote_code=True
        )

    def __len__(self) -> int:
        return len(self.item_texts)

    def __getitem__(self, idx: int) -> Dict:
        tok = self.tokenizer(
            self.item_prefix + self.item_texts[idx],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {k: v.squeeze(0) for k, v in tok.items()}

    def collate_fn(self, batch: List[Dict]) -> Dict:
        input_ids = [item["input_ids"] for item in batch]
        attn_masks = [item["attention_mask"] for item in batch]
        return {
            "item_input_ids": _pad_sequences(
                input_ids, padding_value=self.tokenizer.pad_token_id, padding_side=self.padding_side
            ),
            "item_attention_mask": _pad_sequences(
                attn_masks, padding_value=0, padding_side=self.padding_side
            ),
        }


class RecQueryDataset(QueryDataset):
    def __init__(
        self,
        signal_texts: Dict[str, List[str]],
        gt_ids: Union[List[int], List[List[int]]],
        tokenizer_name: str = "nvidia/NV-Embed-v2",
        max_length: int = 512,
        aux_max_length: int = 256,
    ) -> None:
        assert signal_texts, "signal_texts must contain at least one signal"
        unknown = set(signal_texts) - set(_SIGNAL_PREFIX)
        assert not unknown, f"Unknown signal keys: {unknown}. Supported: {set(_SIGNAL_PREFIX)}"

        self.signal_texts = signal_texts
        self.gt_ids = gt_ids
        self.max_length = max_length
        self.aux_max_length = aux_max_length
        if _uses_stella_tokenizer(tokenizer_name):
            self.query_prefix = _STELLA_S2P_QUERY_PREFIX
        elif _uses_qwen3_tokenizer(tokenizer_name):
            self.query_prefix = _QWEN3_QUERY_PREFIX
        elif _uses_harrier_tokenizer(tokenizer_name):
            self.query_prefix = _HARRIER_QUERY_PREFIX
        elif _uses_f2llm_tokenizer(tokenizer_name):
            self.query_prefix = _F2LLM_QUERY_PREFIX
        elif _uses_e5_tokenizer(tokenizer_name):
            self.query_prefix = _E5_QUERY_PREFIX
        else:
            self.query_prefix = ""
        self.padding_side = "left" if _uses_last_token_tokenizer(tokenizer_name) else "right"
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, truncation_side="right", padding_side=self.padding_side, trust_remote_code=True
        )
        self._len = len(next(iter(signal_texts.values())))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple:
        token_dict: Dict = {}
        for key, texts in self.signal_texts.items():
            prefix = _SIGNAL_PREFIX[key]
            tok = self.tokenizer(
                self.query_prefix + texts[idx],
                max_length=self.max_length if key == "c" else self.aux_max_length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            for k, v in tok.items():
                token_dict[f"{prefix}_{k}"] = v.squeeze(0)
        return idx, self.gt_ids[idx], token_dict

    def collate_fn(self, batch: List[Tuple]) -> Tuple:
        query_ids = [item[0] for item in batch]
        gt_ids_out = [item[1] for item in batch]

        input_dict: Dict = {}
        for key in batch[0][2]:
            seqs = [item[2][key] for item in batch]
            pad_val = self.tokenizer.pad_token_id if key.endswith("input_ids") else 0
            input_dict[key] = _pad_sequences(seqs, padding_value=pad_val, padding_side=self.padding_side)

        return query_ids, gt_ids_out, input_dict


# Backward compatibility for existing scripts and external imports.
RecDataset = RecQueryDataset
BackboneItemDataset = RecItemDataset
BackboneQueryDataset = RecQueryDataset
