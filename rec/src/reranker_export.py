from __future__ import annotations

from typing import Any, Dict, Optional

import torch


RERANKER_USER_COMPONENTS = {
    "conv": "conv_embedding",
    "like": "like_embedding",
    "dislike": "dislike_embedding",
    "long": "long_embedding",
    "short": "short_embedding",
}


def _row_tensor_to_list(value: Any, row_index: int) -> Optional[list[float]]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor or None, got {type(value)!r}")
    tensor = value.detach().float().cpu()
    if tensor.dim() == 1:
        row = tensor
    elif tensor.dim() >= 2:
        row = tensor[row_index]
    else:
        raise ValueError("Scalar tensors cannot be exported as embeddings")
    return row.reshape(-1).tolist()


def extract_user_embedding_payload(model_output: Dict[str, Any], row_index: int) -> Dict[str, Any]:
    components = {
        name: _row_tensor_to_list(model_output.get(output_key), row_index)
        for name, output_key in RERANKER_USER_COMPONENTS.items()
    }
    user_embedding = _row_tensor_to_list(model_output.get("user_embedding"), row_index)

    payload: Dict[str, Any] = {
        "schema_version": "rec_user_embedding_payload.v1",
        "component_order": list(RERANKER_USER_COMPONENTS.keys()),
        "components": components,
        "user_embedding": user_embedding,
    }

    gating_weights = model_output.get("gating_weights")
    if isinstance(gating_weights, torch.Tensor):
        payload["gating_weights"] = _row_tensor_to_list(gating_weights, row_index)

    layer_scale = model_output.get("layer_scale")
    if isinstance(layer_scale, torch.Tensor):
        layer_scale_row = _row_tensor_to_list(layer_scale, row_index)
        if layer_scale_row:
            payload["split_ratio"] = layer_scale_row[0]

    if user_embedding is not None:
        payload["dim"] = len(user_embedding)

    return payload
