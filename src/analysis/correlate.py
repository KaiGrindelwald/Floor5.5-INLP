"""
src/analysis/correlate.py
--------------------------
Correlate cross-lingual alignment metrics with intervention effects by layer.

Produces:
  - Pearson and Spearman r between CKA / cosine and Δlogit / ATE by layer
  - A scatter plot PNG
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.utils.io import read_json, write_json


def compute_correlations(
    align_vals: List[float],
    intv_vals: List[float],
    name_x: str = "alignment",
    name_y: str = "ATE",
) -> Dict:
    """Compute Pearson and Spearman correlation between two layer-series."""
    x = np.array(align_vals, dtype=np.float64)
    y = np.array(intv_vals,  dtype=np.float64)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        "x_name":    name_x,
        "y_name":    name_y,
        "n":         len(x),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def _plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    layer_labels: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, c=range(len(x)), cmap="viridis", s=60, zorder=3)
    if layer_labels:
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(str(i), (xi, yi), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, alpha=0.7)
    # Regression line
    coeffs = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 100)
    ax.plot(xfit, np.polyval(coeffs, xfit), "r--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ↳ {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Correlate layer-wise alignment metrics with intervention effects."
    )
    ap.add_argument("--alignment_json",    type=str, required=True,
                    help="Alignment JSON from run_alignment.py")
    ap.add_argument("--intervention_json", type=str, required=True,
                    help="Layer-sweep intervention JSON from evaluate_intervention.py")
    ap.add_argument("--out_json",  type=str, required=True)
    ap.add_argument("--out_dir",   type=str, default=None,
                    help="Directory to write scatter plots. Defaults to dir of out_json.")
    ap.add_argument("--lang_pair", type=str, default=None,
                    help="Which language pair key to use from alignment JSON "
                         "(e.g. 'en_vs_hi'). Defaults to first non-en_vs_en key.")
    args = ap.parse_args()

    align = read_json(args.alignment_json)
    intv  = read_json(args.intervention_json)

    out_json = Path(args.out_json)
    out_dir  = Path(args.out_dir) if args.out_dir else out_json.parent

    # ── Pick alignment series ─────────────────────────────────────────────────
    cos_pairs = list(align["cosine"].keys())
    cka_pairs = list(align["cka"].keys())

    lang_pair = args.lang_pair
    if lang_pair is None:
        # Auto-pick: first pair that matches intervention language
        intv_lang = intv.get("lang", "")
        candidates = [k for k in cos_pairs if intv_lang in k]
        lang_pair = candidates[0] if candidates else cos_pairs[0]

    cos_means = np.array(align["cosine"][lang_pair]["mean"], dtype=np.float64)
    cka_vals  = np.array(align["cka"][lang_pair], dtype=np.float64)

    # ── Pick intervention series ──────────────────────────────────────────────
    sweep_data = intv.get("layer_sweep", {})
    ate_vals = np.array(sweep_data.get("delta_prob_mean", []), dtype=np.float64)
    dl_vals  = np.array(sweep_data.get("delta_logit_mean", []), dtype=np.float64)

    L = min(len(cos_means), len(cka_vals), len(ate_vals))
    cos_means = cos_means[:L]
    cka_vals  = cka_vals[:L]
    ate_vals  = ate_vals[:L]
    dl_vals   = dl_vals[:L]
    x_layers  = np.arange(L, dtype=np.float64)

    results = {
        "lang_pair":        lang_pair,
        "intervention_lang": intv.get("lang"),
        "target_label":     intv.get("target_label"),
        "n_layers":         int(L),
        "correlations": {
            "cos_vs_ate":     compute_correlations(cos_means.tolist(), ate_vals.tolist(), "cosine",  "ATE"),
            "cka_vs_ate":     compute_correlations(cka_vals.tolist(),  ate_vals.tolist(), "CKA",     "ATE"),
            "cos_vs_dlogit":  compute_correlations(cos_means.tolist(), dl_vals.tolist(),  "cosine",  "Δlogit"),
            "cka_vs_dlogit":  compute_correlations(cka_vals.tolist(),  dl_vals.tolist(),  "CKA",     "Δlogit"),
        }
    }

    write_json(str(out_json), results)
    print(f"[correlate] Wrote {out_json}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_scatter(
        cos_means, ate_vals,
        xlabel=f"Cosine similarity ({lang_pair})",
        ylabel="Mean Δp (ATE)",
        title=f"Cosine alignment vs intervention ATE — {intv.get('target_label','')}",
        out_path=out_dir / "scatter_cos_vs_ate.png",
    )
    _plot_scatter(
        cka_vals, ate_vals,
        xlabel=f"CKA ({lang_pair})",
        ylabel="Mean Δp (ATE)",
        title=f"CKA vs intervention ATE — {intv.get('target_label','')}",
        out_path=out_dir / "scatter_cka_vs_ate.png",
    )

    print("[correlate] Done.")


if __name__ == "__main__":
    main()
