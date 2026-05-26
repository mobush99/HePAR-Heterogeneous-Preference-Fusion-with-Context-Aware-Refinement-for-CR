from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


COMPONENT_KEYS = ("conv", "like", "dislike", "long", "short")
QUERY_SIGNAL_KEYS = ("c", "l", "d", "a", "b")


@dataclass(frozen=True)
class RecEmbeddingProviderConfig:
    checkpoint_path: str
    encoder_impl: str = "modern"
    data_name: str = "inspired"
    rec_root: str | None = None
    documents_path: str | None = None
    base_model_name: str = "nvidia/NV-Embed-v2"
    query_used_info: Sequence[str] = ("c", "l", "d", "a", "b")
    doc_used_info: Sequence[str] = ("m", "pref")
    query_max_length: int = 512
    aux_max_length: int = 256
    item_max_length: int = 512
    device: str = "cuda"
    allow_cpu_fallback: bool = True
    batch_size: int = 32
    bf16: bool = True
    mode: str = "vanilla"
    alpha: float = 0.5
    beta: float = 0.2
    gamma: float = 0.3
    delta: float = 0.3
    epsilon: float = 0.3
    zeta: float = 0.3
    uniform_gate_init: bool = False
    hg_gate_activation: str = "sigmoid"
    hg_only: bool = False
    n2e_routing_iters: int = 2
    use_n2n: bool = True
    use_global_edge: bool = True
    flat_gating: bool = False
    use_quantization: bool = False
    quantization_bits: int = 4
    use_dynamic_gating: bool = False
    use_cross_attention: bool = False
    use_hypergraph: bool = False


def _default_rec_root() -> Path:
    return Path(__file__).resolve().parents[3] / "rec"


def _item_key(item: Dict[str, Any]) -> str | None:
    value = item.get("doc_key") or item.get("id") or item.get("entity_id")
    return str(value) if value is not None else None


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def sample_to_rec_query(sample: Dict[str, Any]) -> Dict[str, Any]:
    if "input_dialog_history" in sample or "preference" in sample:
        preference = sample.get("preference") or {}
        return {
            "input_dialog_history": sample.get("input_dialog_history", ""),
            "preference": {
                "like": _as_string_list(preference.get("like")),
                "dislike": _as_string_list(preference.get("dislike")),
            },
            "long_term_preferences": sample.get("long_term_preferences", ""),
            "short_term_preferences": sample.get("short_term_preferences", ""),
            "long_new": sample.get("long_new", ""),
            "short_new": sample.get("short_new", ""),
        }

    context = sample.get("user_context") or {}
    return {
        "input_dialog_history": context.get("conv", ""),
        "preference": {
            "like": _as_string_list(context.get("like")),
            "dislike": _as_string_list(context.get("dislike")),
        },
        "long_term_preferences": context.get("long_term_preferences", ""),
        "short_term_preferences": context.get("short_term_preferences", ""),
        "long_new": context.get("long_new", context.get("long_term_preferences", "")),
        "short_new": context.get("short_new", context.get("short_term_preferences", "")),
    }


def candidate_keys(candidates: Iterable[Dict[str, Any]]) -> List[str]:
    keys: List[str] = []
    for candidate in candidates:
        key = _item_key(candidate)
        if key is not None:
            keys.append(key)
    return keys


class RecEmbeddingProvider:
    """On-the-fly bridge from reranker samples to trained rec embeddings.

    The candidate JSONL remains the ranking source of truth. This provider only
    loads the trained rec encoder, caches item embeddings once, and encodes the
    current sample's five user signal components plus final user embedding.
    """

    def __init__(self, config: RecEmbeddingProviderConfig) -> None:
        self.config = config
        self.rec_root = Path(config.rec_root).resolve() if config.rec_root else _default_rec_root()
        self.documents_path = (
            Path(config.documents_path).resolve()
            if config.documents_path
            else self.rec_root / "data" / config.data_name / "processed_document.json"
        )
        self._loaded = False
        self._item_embeddings = None
        self._item_index: Dict[str, int] = {}

    def load(self) -> None:
        if self._loaded:
            return
        self._prepare_rec_imports()
        import torch
        from utils import format_query, load_document

        device = self._resolve_device(torch)
        self.device = device
        self.torch = torch
        self.format_query = format_query
        self.encoder_impl = self.config.encoder_impl

        item_ids, item_text = load_document(
            file_path=self.documents_path,
            data_name=self.config.data_name,
            used_info=list(self.config.doc_used_info),
        )
        self.item_ids = item_ids
        self._item_index = {str(item_id): idx for idx, item_id in enumerate(item_ids)}

        if self.encoder_impl == "legacy":
            self._load_legacy_encoder(item_text)
        elif self.encoder_impl == "modern":
            self._load_modern_encoder(item_text)
        else:
            raise ValueError("encoder_impl must be 'modern' or 'legacy'")

        self.model.to(device)
        self.model.eval()
        self._loaded = True

    def encode_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.encode_samples([sample])[0]

    def encode_samples(self, samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.load()
        users = self.encode_users(samples)
        encoded = []
        for sample, user in zip(samples, users):
            candidates = sample.get("candidates") or []
            keys = candidate_keys(candidates)
            encoded.append(
                {
                    "sample_id": sample.get("sample_id", sample.get("id")),
                    "user": user,
                    "candidate_keys": keys,
                    "candidate_embeddings": self.get_item_embeddings(keys),
                    "candidates": candidates,
                }
            )
        return encoded

    def encode_user(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.encode_users([sample])[0]

    def encode_users(self, samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.load()
        torch = self.torch
        queries = [sample_to_rec_query(sample) for sample in samples]
        signal_texts = {
            key: [self.format_query(query, key) for query in queries]
            for key in self.config.query_used_info
        }
        if self.encoder_impl == "legacy":
            dataset = self.query_dataset_cls(
                user_text=signal_texts,
                gt_ids=list(range(len(samples))),
                base_tokenizer_name=self.config.base_model_name,
            )
        else:
            dataset = self.query_dataset_cls(
                signal_texts=signal_texts,
                gt_ids=list(range(len(samples))),
                tokenizer_name=self.config.base_model_name,
                max_length=self.config.query_max_length,
                aux_max_length=self.config.aux_max_length,
            )
        batch = dataset.collate_fn([dataset[index] for index in range(len(samples))])
        _, _, input_u = batch
        input_u = {
            key: value.to(self.device) if value is not None else None
            for key, value in input_u.items()
        }
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.bf16 and self.device.type == "cuda",
        ):
            if self.encoder_impl == "legacy":
                output = self.model(**input_u)
            else:
                output = self.model.encode_query_with_components(input_u)
        return [self._user_output_to_tensors(output, index=index) for index in range(len(samples))]

    def get_item_embeddings(self, item_keys: Sequence[str]):
        self.load()
        self._ensure_item_cache()
        indices = []
        missing = []
        for key in item_keys:
            if key in self._item_index:
                indices.append(self._item_index[key])
            else:
                missing.append(key)
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"{len(missing)} candidate items missing from rec documents: {preview}")
        index_tensor = self.torch.tensor(indices, dtype=self.torch.long)
        return self._item_embeddings[index_tensor].to(self.device)

    def _ensure_item_cache(self) -> None:
        if self._item_embeddings is not None:
            return
        torch = self.torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        loader = DataLoader(
            self.item_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.item_dataset.collate_fn,
        )
        embeddings = []
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.bf16 and self.device.type == "cuda",
        ):
            for batch in tqdm(loader, desc="Encoding rec items", leave=False):
                batch = {
                    key: value.to(self.device) if value is not None else None
                    for key, value in batch.items()
                }
                if self.encoder_impl == "legacy":
                    output = self.model(**batch)
                    item_embedding = output["item_embedding"]
                else:
                    item_embedding = self.model.encode_item(batch)
                embeddings.append(item_embedding.detach().float().cpu())
        self._item_embeddings = torch.cat(embeddings, dim=0)

    def _load_legacy_encoder(self, item_text: Dict[str, List[str]]) -> None:
        from src.backbone_legacy import BackboneEncoder, LegacyQueryDataset, LegacyItemDataset

        self.query_dataset_cls = LegacyQueryDataset
        self.item_dataset = LegacyItemDataset(
            item_text=item_text,
            base_tokenizer_name=self.config.base_model_name,
            max_length=self.config.item_max_length,
        )
        self.model = BackboneEncoder(
            base_model_name=self.config.base_model_name,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            delta=self.config.delta,
            epsilon=self.config.epsilon,
            zeta=self.config.zeta,
            use_quantization=self.config.use_quantization,
            quantization_bits=self.config.quantization_bits,
            use_dynamic_gating=self.config.use_dynamic_gating,
            use_cross_attention=self.config.use_cross_attention,
            use_hypergraph=self.config.use_hypergraph,
        )
        self.model.load_best_checkpoint(ckpt_save_path=Path(self.config.checkpoint_path))

    def _load_modern_encoder(self, item_text: Dict[str, List[str]]) -> None:
        from src.dataset import RecItemDataset, RecQueryDataset
        from src.model import BackboneEncoder

        self.query_dataset_cls = RecQueryDataset
        self.item_dataset = RecItemDataset(
            item_texts=item_text["item"],
            tokenizer_name=self.config.base_model_name,
            max_length=self.config.item_max_length,
        )
        self.model = BackboneEncoder(
            base_model_name=self.config.base_model_name,
            mode=self.config.mode,
            query_used_info=tuple(self.config.query_used_info),
            alpha=self.config.alpha,
            beta=self.config.beta,
            delta=self.config.delta,
            epsilon=self.config.epsilon,
            uniform_gate_init=self.config.uniform_gate_init,
            gate_activation=self.config.hg_gate_activation,
            hg_only=self.config.hg_only,
            num_routing_iters=self.config.n2e_routing_iters,
            use_n2n=self.config.use_n2n,
            use_global_edge=self.config.use_global_edge,
            flat_gating=self.config.flat_gating,
        )
        self.model.load(self.config.checkpoint_path, device=str(self.device))

    def _user_output_to_tensors(self, output: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        def take(name: str):
            value = output.get(name)
            if value is None:
                return None
            return value.detach()[index].float()

        result = {
            "components": {
                "conv": take("conv_embedding"),
                "like": take("like_embedding"),
                "dislike": take("dislike_embedding"),
                "long": take("long_embedding"),
                "short": take("short_embedding"),
            },
            "user_embedding": take("user_embedding"),
            "component_order": list(COMPONENT_KEYS),
        }
        gating_weights = take("gating_weights")
        if gating_weights is not None:
            result["gating_weights"] = gating_weights
        split_ratio = take("layer_scale")
        if split_ratio is not None and split_ratio.numel() > 0:
            result["split_ratio"] = split_ratio.reshape(-1)[0]
        return result

    def _prepare_rec_imports(self) -> None:
        rec_root_str = str(self.rec_root)
        loaded_src = sys.modules.get("src")
        if loaded_src is not None:
            src_file = getattr(loaded_src, "__file__", "")
            if src_file and not str(Path(src_file).resolve()).startswith(rec_root_str):
                raise ImportError(
                    "A non-rec module named 'src' is already loaded; start a fresh process "
                    "or load RecEmbeddingProvider before importing reranker src packages."
                )
        if rec_root_str not in sys.path:
            sys.path.insert(0, rec_root_str)

    def _resolve_device(self, torch):
        requested = self.config.device
        if requested == "cuda" and not torch.cuda.is_available():
            if not self.config.allow_cpu_fallback:
                raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
            requested = "cpu"
        return torch.device(requested)
