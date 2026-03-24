"""
src/interventions/hooks.py
--------------------------
PyTorch forward-hook infrastructure for inference-time concept steering.

Usage (context-manager):
    direction: np.ndarray, shape (D,) — unit concept vector at one layer
    with SteeringContext(model, layer_idx=15, direction=v_c, alpha=1.0):
        outputs = model(**inputs, output_hidden_states=True)

The hook intercepts the OUTPUT of the specified transformer layer and adds
alpha * direction to every token-level hidden state (i.e., the entire
residual stream at that position), then re-normalises if needed.

Architecture auto-detection:
    - BERT-style:    model.encoder.layer[i]
    - XLM-R style:   model.encoder.layer[i]   (same — both RobertaModel)
    - Both inherit from BertLayer / RobertaLayer; the hook targets the block
      output tensor (first element of the layer's return tuple).
"""

from __future__ import annotations

import contextlib
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _get_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """Return the transformer block at index *layer_idx* for BERT/XLM-R."""
    # Both bert-base-multilingual-cased and xlm-roberta expose
    # model.encoder.layer[i]
    try:
        return model.encoder.layer[layer_idx]
    except AttributeError:
        pass
    # Fallback: model.transformer.layer[i] (DistilBERT-style)
    try:
        return model.transformer.layer[layer_idx]
    except AttributeError:
        pass
    raise RuntimeError(
        f"Cannot find transformer layer {layer_idx} in model of type "
        f"{type(model).__name__}. Expected model.encoder.layer[i] or "
        f"model.transformer.layer[i]."
    )


class ConceptSteeringHook:
    """
    Adds ``alpha * direction`` to the hidden states output of a single
    transformer layer at every token position.

    Args:
        direction: np.ndarray of shape (D,) — the unit concept vector.
        alpha:     Scaling factor (intervention strength).
    """

    def __init__(self, direction: np.ndarray, alpha: float) -> None:
        self.direction: torch.Tensor  # set on first forward pass
        self._dir_np = direction.astype(np.float32)
        self.alpha = alpha
        self._handle: Optional[torch.utils.hooks.RemovableHook] = None
        self._device: Optional[torch.device] = None

    def _ensure_dir(self, device: torch.device, dtype: torch.dtype) -> None:
        if not hasattr(self, "direction") or self._device != device:
            self.direction = torch.from_numpy(self._dir_np).to(device=device, dtype=dtype)
            self._device = device

    def hook_fn(self, module: nn.Module, input: Tuple, output):
        """Forward hook: intercept layer output and add alpha*v_c."""
        # Layer outputs can be a tuple (hidden_state, ...) or just a tensor
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        self._ensure_dir(hidden.device, hidden.dtype)
        # hidden: (B, T, D)  direction: (D,)
        hidden = hidden + self.alpha * self.direction.view(1, 1, -1)

        if rest is not None:
            return (hidden,) + rest
        return hidden

    def register(self, layer: nn.Module) -> None:
        self._handle = layer.register_forward_hook(self.hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextlib.contextmanager
def SteeringContext(
    model: nn.Module,
    layer_idx: int,
    direction: np.ndarray,
    alpha: float,
):
    """Context manager: apply concept steering to a single layer during its scope."""
    hook = ConceptSteeringHook(direction=direction, alpha=alpha)
    layer = _get_layer(model, layer_idx)
    hook.register(layer)
    try:
        yield hook
    finally:
        hook.remove()


@contextlib.contextmanager
def MultiLayerSteeringContext(
    model: nn.Module,
    layer_idxs: List[int],
    direction: np.ndarray,
    alpha: float,
):
    """Apply the same steering hook to multiple layers simultaneously."""
    hooks = []
    for li in layer_idxs:
        h = ConceptSteeringHook(direction=direction, alpha=alpha)
        h.register(_get_layer(model, li))
        hooks.append(h)
    try:
        yield hooks
    finally:
        for h in hooks:
            h.remove()
