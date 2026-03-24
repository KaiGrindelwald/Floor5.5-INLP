from __future__ import annotations

from typing import Literal

import torch


PoolType = Literal["cls", "mean"]


def pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pool: PoolType,
) -> torch.Tensor:
    """Pool token-level hidden states into a single vector per example.

    Args:
        hidden: (B, T, D) hidden states from some layer.
        attention_mask: (B, T) with 1s for real tokens.
        pool: "cls" => take token 0; "mean" => masked mean over tokens.
    Returns:
        pooled: (B, D)
    """
    if pool == "cls":
        return hidden[:, 0, :]
    if pool == "mean":
        # masked mean: sum(h * mask) / sum(mask)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B,T,1)
        denom = mask.sum(dim=1).clamp_min(1.0)                # (B,1)
        return (hidden * mask).sum(dim=1) / denom
    raise ValueError(f"Unknown pool={pool}")
