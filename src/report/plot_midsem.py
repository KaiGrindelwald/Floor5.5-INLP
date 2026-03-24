from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

from src.utils.io import read_json
from src.utils.labeling import LABELS_6

# ── Colour palette consistent across plots ─────────────────────────────────
_PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
_LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def _get_figax(w: int = 10, h: int = 5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(True, linewidth=0.4, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _save(fig, out_path: Path, dpi: int = 150) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  ↳ {out_path}")


def plot_lines(
    x,
    series: Dict[str, List[float]],
    title: str,
    ylabel: str,
    out_path: Path,
    stds: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Line plot, optionally with shaded ± std bands."""
    fig, ax = _get_figax()
    for idx, (name, y) in enumerate(series.items()):
        col = _PALETTE[idx % len(_PALETTE)]
        ls = _LINE_STYLES[idx % len(_LINE_STYLES)]
        ax.plot(x, y, label=name, color=col, linestyle=ls, linewidth=1.8, marker="o", markersize=3)
        if stds and name in stds:
            std = np.array(stds[name])
            ya = np.array(y)
            ax.fill_between(x, ya - std, ya + std, alpha=0.15, color=col)
    ax.set_xlabel("Layer (0 = first transformer layer after embeddings)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, out_path)


def plot_f1_heatmap(
    probe: Dict,
    out_dir: Path,
) -> None:
    """Per-class F1 heatmap: rows = layers, cols = classes, one plot per language."""
    mb = probe.get("metrics_by_layer", {})
    for lang, layer_list in mb.items():
        # Check if per_class_f1 is available (enhanced probe output)
        if not layer_list or "per_class_f1" not in layer_list[0]:
            continue
        L = len(layer_list)
        n_cls = len(LABELS_6)
        mat = np.zeros((L, n_cls), dtype=np.float32)
        for l, m in enumerate(layer_list):
            for c, lab in enumerate(LABELS_6):
                mat[l, c] = m["per_class_f1"].get(lab, 0.0)

        fig, ax = plt.subplots(figsize=(10, max(4, int(L * 0.35))))
        if _HAS_SNS:
            sns.heatmap(
                mat.T,
                ax=ax,
                xticklabels=list(range(L)),
                yticklabels=LABELS_6,
                cmap="viridis",
                vmin=0, vmax=1,
                annot=False,
                linewidths=0.3,
            )
        else:
            im = ax.imshow(mat.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(L))
            ax.set_yticks(range(n_cls))
            ax.set_yticklabels(LABELS_6)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Class", fontsize=11)
        ax.set_title(f"Per-class F1 by layer — lang={lang}", fontsize=13, fontweight="bold")
        _save(fig, out_dir / f"probe_f1_heatmap_{lang}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment_json", type=str, required=True)
    ap.add_argument("--probe_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--intervention_json", type=str, default=None,
                    help="Optional: intervention sweep JSON to overlay Δlogit curve.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    align = read_json(args.alignment_json)
    probe = read_json(args.probe_json)

    print(f"[plot_midsem] Writing plots → {out_dir}")

    # ── Cosine alignment with std bands ────────────────────────────────────
    cos = align["cosine"]
    L = len(next(iter(cos.values()))["mean"])
    x = list(range(L))

    plot_lines(
        x,
        {k: v["mean"] for k, v in cos.items()},
        title="Cosine alignment vs layer (test split)",
        ylabel="Mean cosine similarity",
        out_path=out_dir / "cosine_alignment.png",
        stds={k: v["std"] for k, v in cos.items()},
    )

    # ── Linear CKA ─────────────────────────────────────────────────────────
    cka = align["cka"]
    plot_lines(
        x,
        {k: v for k, v in cka.items()},
        title="Linear CKA vs layer (test split)",
        ylabel="CKA",
        out_path=out_dir / "cka.png",
    )

    # ── Probe transfer accuracy ─────────────────────────────────────────────
    mb = probe["metrics_by_layer"]
    plot_lines(
        x,
        {lang: [m["acc"] for m in arr] for lang, arr in mb.items()},
        title="Probe transfer accuracy vs layer (train EN → test language)",
        ylabel="Accuracy",
        out_path=out_dir / "probe_acc.png",
    )

    plot_lines(
        x,
        {lang: [m["macro_f1"] for m in arr] for lang, arr in mb.items()},
        title="Probe transfer macro-F1 vs layer (train EN → test language)",
        ylabel="Macro-F1",
        out_path=out_dir / "probe_macro_f1.png",
    )

    # ── Per-class F1 heatmap (only if per_class_f1 is in probe output) ────
    plot_f1_heatmap(probe, out_dir)

    # ── Optional: overlay Δlogit from intervention ─────────────────────────
    if args.intervention_json:
        try:
            intv = read_json(args.intervention_json)
            sweep = intv.get("layer_sweep", {})
            if sweep:
                plot_lines(
                    x,
                    {f"Δlogit({intv['target_label']}) {intv['lang']}": sweep.get("delta_logit_mean", [0]*L)},
                    title=f"Intervention Δlogit vs layer (α={intv.get('alpha', '?')})",
                    ylabel="Mean Δlogit",
                    out_path=out_dir / "intervention_delta_logit.png",
                )
        except Exception as e:
            print(f"[plot_midsem] Warning: could not plot intervention JSON: {e}")

    print(f"[plot_midsem] Done.")


if __name__ == "__main__":
    main()
