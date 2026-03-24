"""
src/interventions/sweep.py
--------------------------
Layer-sweep and alpha-sweep logic for concept-steering experiments.

This implementation uses a frozen encoder plus a trained linear probe readout,
which makes the intervention results meaningful even when the base encoder does
not ship with a task-specific classification head.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.interventions.hooks import SteeringContext
from src.interventions.readout import ProbeReadout
from src.utils.pooling import pool_hidden


@torch.no_grad()
def _batch_pooled_reps(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    readout_layer: int,
    pool: str,
    batch_size: int = 16,
    max_length: int = 192,
    hook_layer_idx: Optional[int] = None,
    direction: Optional[np.ndarray] = None,
    alpha: float = 0.0,
) -> np.ndarray:
    all_reps = []
    model.eval()

    if hook_layer_idx is None or direction is None or alpha == 0.0:
        ctx = contextlib.nullcontext()
    else:
        ctx = SteeringContext(model, layer_idx=hook_layer_idx, direction=direction, alpha=alpha)

    with ctx:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            attention_mask = enc["attention_mask"].to(device)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, return_dict=True)
            hidden_states = list(out.hidden_states[1:])
            layer_idx = readout_layer if readout_layer >= 0 else len(hidden_states) - 1
            pooled = pool_hidden(hidden_states[layer_idx], attention_mask, pool=pool)
            all_reps.append(pooled.float().cpu())

    return torch.cat(all_reps, dim=0).numpy()


def _batch_logits(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    readout: ProbeReadout,
    batch_size: int = 16,
    max_length: int = 192,
) -> np.ndarray:
    reps = _batch_pooled_reps(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        readout_layer=readout.readout_layer,
        pool=readout.pool,
        batch_size=batch_size,
        max_length=max_length,
    )
    return readout.logits(reps)


def _logits_with_hook(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    readout: ProbeReadout,
    layer_idx: int,
    direction: np.ndarray,
    alpha: float,
    batch_size: int = 16,
    max_length: int = 192,
) -> np.ndarray:
    reps = _batch_pooled_reps(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        readout_layer=readout.readout_layer,
        pool=readout.pool,
        batch_size=batch_size,
        max_length=max_length,
        hook_layer_idx=layer_idx,
        direction=direction,
        alpha=alpha,
    )
    return readout.logits(reps)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def layer_sweep(
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
    description: str = "layer sweep",
) -> Dict:
    """Sweep the hook layer while keeping the readout fixed."""
    L = directions.shape[0]

    logits_base = _batch_logits(model, tokenizer, texts, device, readout, batch_size, max_length)
    probs_base = _softmax(logits_base)

    results: Dict = {
        "n_layers": int(L),
        "alpha": float(alpha),
        "target_class": int(target_class),
        "n_examples": int(len(texts)),
        "readout_layer": int(readout.readout_layer),
        "pool": readout.pool,
        "by_layer": [],
    }

    for l in tqdm(range(L), desc=description):
        logits_int = _logits_with_hook(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            readout=readout,
            layer_idx=l,
            direction=directions[l],
            alpha=alpha,
            batch_size=batch_size,
            max_length=max_length,
        )
        probs_int = _softmax(logits_int)

        delta_logit = logits_int[:, target_class] - logits_base[:, target_class]
        delta_prob = probs_int[:, target_class] - probs_base[:, target_class]
        n_improved = int((probs_int[:, target_class] > probs_base[:, target_class]).sum())

        results["by_layer"].append({
            "layer": int(l),
            "delta_logit_mean": float(delta_logit.mean()),
            "delta_logit_std": float(delta_logit.std()),
            "delta_prob_mean": float(delta_prob.mean()),
            "delta_prob_std": float(delta_prob.std()),
            "success_rate": float(n_improved / len(texts)),
            "ate": float(delta_prob.mean()),
        })

    results["delta_logit_mean"] = [r["delta_logit_mean"] for r in results["by_layer"]]
    results["delta_prob_mean"] = [r["delta_prob_mean"] for r in results["by_layer"]]
    results["success_rate"] = [r["success_rate"] for r in results["by_layer"]]
    return results


def alpha_sweep(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    direction: np.ndarray,
    readout: ProbeReadout,
    layer_idx: int,
    target_class: int,
    alphas: Optional[List[float]] = None,
    batch_size: int = 16,
    max_length: int = 192,
    description: str = "alpha sweep",
) -> Dict:
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

    logits_base = _batch_logits(model, tokenizer, texts, device, readout, batch_size, max_length)
    probs_base = _softmax(logits_base)

    results: Dict = {
        "layer_idx": int(layer_idx),
        "target_class": int(target_class),
        "n_examples": int(len(texts)),
        "alphas": [float(a) for a in alphas],
        "readout_layer": int(readout.readout_layer),
        "pool": readout.pool,
        "by_alpha": [],
    }

    for alpha in tqdm(alphas, desc=description):
        if float(alpha) == 0.0:
            logits_int = logits_base
        else:
            logits_int = _logits_with_hook(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                device=device,
                readout=readout,
                layer_idx=layer_idx,
                direction=direction,
                alpha=float(alpha),
                batch_size=batch_size,
                max_length=max_length,
            )
        probs_int = _softmax(logits_int)
        delta_logit = logits_int[:, target_class] - logits_base[:, target_class]
        delta_prob = probs_int[:, target_class] - probs_base[:, target_class]
        n_improved = int((probs_int[:, target_class] > probs_base[:, target_class]).sum())
        results["by_alpha"].append({
            "alpha": float(alpha),
            "delta_logit_mean": float(delta_logit.mean()),
            "delta_prob_mean": float(delta_prob.mean()),
            "success_rate": float(n_improved / len(texts)),
            "ate": float(delta_prob.mean()),
        })

    results["delta_logit_means"] = [r["delta_logit_mean"] for r in results["by_alpha"]]
    results["delta_prob_means"] = [r["delta_prob_mean"] for r in results["by_alpha"]]
    results["success_rates"] = [r["success_rate"] for r in results["by_alpha"]]
    return results
