"""
src/report/plot_interventions.py
---------------------------------
Dedicated CLI for visualising intervention results.

Plots produced:
  1. Layer-sweep Δlogit and Δp curves (one line per concept/language)
  2. Alpha-sweep curves (monotonicity check) — Δp vs α for a fixed layer
  3. Ablation comparison bar chart (true vs random vs shuffled vs within-lang)
  4. Per-concept ATE comparison across languages at a fixed layer

Usage:
    python -m src.report.plot_interventions \\
        --layer_sweep_jsons artifacts/results/intv_xlmr_hi_CH_layer.json
                             artifacts/results/intv_xlmr_hi_FC_layer.json \\
        --alpha_sweep_json  artifacts/results/intv_xlmr_hi_CH_alpha.json \\
        --out_dir           artifacts/figs/interventions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import read_json

# ── palette ──────────────────────────────────────────────────────────────────
_PALETTE     = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
                 "#00BCD4", "#795548", "#607D8B"]
_LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def _figax(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _save(fig, path: Path, dpi=150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  ↳ {path}")


# ── individual plot functions ─────────────────────────────────────────────────

def plot_layer_sweep_curves(
    sweep_jsons: List[str],
    out_dir: Path,
    metric: str = "delta_logit_mean",  # or "delta_prob_mean" or "success_rate"
) -> None:
    """One line per sweep JSON (= one concept/language combination)."""
    fig, ax = _figax()
    ylabel_map = {
        "delta_logit_mean": "Mean Δlogit",
        "delta_prob_mean":  "Mean Δp",
        "success_rate":     "Success rate",
    }
    for idx, json_path in enumerate(sweep_jsons):
        data = read_json(json_path)
        sweep = data.get("layer_sweep", {})
        vals  = sweep.get(metric, [])
        if not vals:
            print(f"  [warn] {json_path} has no key '{metric}' in layer_sweep")
            continue
        label = f"{data.get('target_label','?')} {data.get('lang','?')}"
        col   = _PALETTE[idx % len(_PALETTE)]
        ls    = _LINE_STYLES[idx % len(_LINE_STYLES)]
        ax.plot(range(len(vals)), vals, label=label, color=col, linestyle=ls,
                linewidth=1.8, marker="o", markersize=3)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel(ylabel_map.get(metric, metric), fontsize=11)
    ax.set_title(f"Intervention {ylabel_map.get(metric, metric)} vs Layer", fontsize=13,
                 fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, ncol=2)
    _save(fig, out_dir / f"layer_sweep_{metric}.png")


def plot_alpha_sweep(
    alpha_json: str,
    out_dir: Path,
) -> None:
    """Δp vs α for a fixed best layer (monotonicity check)."""
    data   = read_json(alpha_json)
    sweep  = data.get("alpha_sweep", {})
    alphas = sweep.get("alphas", []) or data.get("alphas", [])
    dp     = sweep.get("delta_prob_means", []) or sweep.get("delta_prob_mean", [])
    dl     = sweep.get("delta_logit_means", []) or sweep.get("delta_logit_mean", [])
    sr     = sweep.get("success_rates", []) or sweep.get("success_rate", [])

    label  = f"{data.get('target_label','?')} | layer {data.get('best_layer','?')}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, title, ylabel in zip(
        axes,
        [dp, dl, sr],
        ["Δp vs α", "Δlogit vs α", "Success rate vs α"],
        ["Mean Δp", "Mean Δlogit", "Success rate"],
    ):
        ax.plot(alphas, vals, marker="o", color="#2196F3", linewidth=2, markersize=6)
        ax.set_xlabel("α (steering strength)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.grid(True, linewidth=0.4, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Alpha sweep — {label}", fontsize=14, fontweight="bold")
    _save(fig, out_dir / "alpha_sweep.png")


def plot_ablation_comparison(
    true_json: str,
    random_json: Optional[str],
    shuffled_json: Optional[str],
    within_json: Optional[str],
    out_dir: Path,
    layer_idx: int = -1,
    metric: str = "delta_prob_mean",
) -> None:
    """
    Bar chart comparing true intervention vs controls at a specific layer.
    *layer_idx* = -1 means use the best layer (max |δ|).
    """
    def _get_val(json_path, layer):
        if not json_path:
            return None
        data  = read_json(json_path)
        sweep = data.get("layer_sweep", {})
        vals  = sweep.get(metric, [])
        if not vals:
            return None
        i = layer if layer >= 0 else int(np.argmax(np.abs(vals)))
        return vals[i], i

    entries = [
        ("True intervention",  true_json,    "#2196F3"),
        ("Random direction",   random_json,  "#9E9E9E"),
        ("Shuffled concept",   shuffled_json,"#FF9800"),
        ("Within-language",    within_json,  "#4CAF50"),
    ]

    labels_bar, heights, colors = [], [], []
    best_layer = None
    for name, path, col in entries:
        if not path:
            continue
        result = _get_val(path, layer_idx)
        if result is None:
            continue
        val, li = result
        if best_layer is None:
            best_layer = li
        labels_bar.append(name)
        heights.append(val)
        colors.append(col)

    if not heights:
        print("[plot_interventions] No ablation data to plot.")
        return

    fig, ax = _figax(w=8, h=5)
    bars = ax.bar(labels_bar, heights, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002 * np.sign(h),
                f"{h:.4f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    ax.set_ylabel(f"Mean {metric.replace('_', ' ')} at layer {best_layer}", fontsize=11)
    ax.set_title("Ablation comparison: true vs controls", fontsize=13, fontweight="bold")
    _save(fig, out_dir / "ablation_comparison.png")


def main():
    ap = argparse.ArgumentParser(description="Plot intervention sweep results.")
    ap.add_argument("--layer_sweep_jsons", type=str, nargs="*",
                    help="One or more layer-sweep JSON files to overlay.")
    ap.add_argument("--alpha_sweep_json",  type=str, default=None)
    ap.add_argument("--true_json",         type=str, default=None,
                    help="True intervention JSON for ablation bar chart.")
    ap.add_argument("--random_json",       type=str, default=None)
    ap.add_argument("--shuffled_json",     type=str, default=None)
    ap.add_argument("--within_json",       type=str, default=None)
    ap.add_argument("--out_dir",           type=str, required=True)
    ap.add_argument("--metric",            type=str, default="delta_prob_mean",
                    choices=["delta_logit_mean", "delta_prob_mean", "success_rate"])
    ap.add_argument("--best_layer",        type=int, default=-1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.layer_sweep_jsons:
        for m in ["delta_logit_mean", "delta_prob_mean", "success_rate"]:
            plot_layer_sweep_curves(args.layer_sweep_jsons, out_dir, metric=m)

    if args.alpha_sweep_json:
        plot_alpha_sweep(args.alpha_sweep_json, out_dir)

    if args.true_json:
        plot_ablation_comparison(
            true_json=args.true_json,
            random_json=args.random_json,
            shuffled_json=args.shuffled_json,
            within_json=args.within_json,
            out_dir=out_dir,
            layer_idx=args.best_layer,
            metric=args.metric,
        )

    print("[plot_interventions] Done.")


if __name__ == "__main__":
    main()
