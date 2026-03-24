"""
src/analysis/ablations.py
--------------------------
Ablation control experiments for the concept-steering intervention pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.interventions.readout import ProbeReadout
from src.interventions.sweep import layer_sweep
from src.utils.labeling import LABEL2ID, LABELS_6


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)


def random_direction_control(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    directions: np.ndarray,
    readout: ProbeReadout,
    target_class: int,
    alpha: float = 1.0,
    batch_size: int = 16,
    max_length: int = 192,
    seed: int = 42,
) -> Dict:
    L, D = directions.shape
    rng = np.random.default_rng(seed)
    random_dirs = rng.standard_normal((L, D)).astype(np.float32)
    for l in range(L):
        random_dirs[l] = _unit(random_dirs[l])

    result = layer_sweep(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        directions=random_dirs,
        readout=readout,
        target_class=target_class,
        alpha=alpha,
        batch_size=batch_size,
        max_length=max_length,
        description="layer sweep (random direction control)",
    )
    result["control_type"] = "random_direction"
    return result


def shuffled_concept_control(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    directions_dir: str,
    readout: ProbeReadout,
    target_label: str,
    surrogate_label: Optional[str],
    alpha: float = 1.0,
    batch_size: int = 16,
    max_length: int = 192,
) -> Dict:
    if surrogate_label is None:
        idx = LABELS_6.index(target_label)
        surrogate_label = LABELS_6[(idx + 1) % len(LABELS_6)]

    surrogate_path = Path(directions_dir) / f"{surrogate_label}.npy"
    if not surrogate_path.exists():
        raise FileNotFoundError(f"Surrogate direction not found: {surrogate_path}")

    surrogate_dirs = np.load(str(surrogate_path))
    target_class = LABEL2ID[target_label]

    result = layer_sweep(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        directions=surrogate_dirs,
        readout=readout,
        target_class=target_class,
        alpha=alpha,
        batch_size=batch_size,
        max_length=max_length,
        description=f"layer sweep (shuffled concept: {surrogate_label}→{target_label})",
    )
    result["control_type"] = "shuffled_concept"
    result["surrogate_label"] = surrogate_label
    return result
