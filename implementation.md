# Cross-Lingual Concept Transfer via Representation Engineering
## Full Implementation Guide

---

## 1. Overview

This codebase implements a complete research pipeline for the proposal:
**Important update:** intervention evaluation now uses a frozen encoder plus an English-trained linear probe readout, rather than a randomly initialised sequence-classification head. That keeps the methodology faithful to the proposal's encoder-centric setting and makes the resulting steering curves analysable.

**"Cross-Lingual Concept Transfer via Representation Engineering"**

The core idea: **moral-foundation concepts** (Care/Harm, Fairness/Cheating, Loyalty/Betrayal, Authority/Subversion, Purity/Degradation, and Non-Moral) learned as geometric directions in a multilingual model's English representation space can be transferred cross-lingually and used to *causally steer* the model's reasoning in Hindi and Welsh.

### Research Questions
| # | Question | Pipeline Step |
|---|----------|--------------|
| RQ1 | How aligned are concepts across EN/HI/LR? | Cosine + CKA alignment |
| RQ2 | Can an EN concept direction steer the model in other languages? | Intervention sweep |
| RQ3 | Which layers are most effective for cross-lingual transfer? | Layer-sweep analysis |

---

## 2. Dataset

- **Source**: [`MLNTeam-Unical/MoralTextManipulation`](https://huggingface.co/datasets/MLNTeam-Unical/MoralTextManipulation) (`unconditioned` config, `revise` split)
- **Labels**: 6 classes — `CH` (Care/Harm), `FC` (Fairness), `LB` (Loyalty), `AS` (Authority), `PD` (Purity), `NM` (Non-Moral)
- **Languages**: EN (original), HI (translated via `Helsinki-NLP/opus-mt-en-hi`), + LR (default: Welsh `cy` via `Helsinki-NLP/opus-mt-en-cy`)
- **Size**: Default 500 examples/class × 6 = 3,000 total (stratified 80/10/10 train/dev/test split)

### Format: `artifacts/data/parallel.jsonl`
```json
{
  "id": "ex_000042",
  "label_id": 0,
  "label": "CH",
  "text_en": "It is wrong to harm innocent people.",
  "text_hi": "...",
  "text_cy": "..."
}
```

---

## 3. Models

| Model | HuggingFace ID | Layers | Hidden dim |
|-------|---------------|--------|------------|
| XLM-RoBERTa large | `xlm-roberta-large` | 24 | 1024 |
| mBERT | `bert-base-multilingual-cased` | 12 | 768 |

---

## 4. Module Architecture

```
src/
├── data/
│   ├── build_parallel_dataset.py   # Download + stratified sample + MT translation
│   └── export_gold_for_manual.py   # CSV export for manual annotation
│
├── encode/
│   └── cache_reps.py               # Layer-wise representation caching → HDF5
│
├── concepts/
│   └── extract_directions.py       # Concept vectors: mean_diff, probe_weight, pca_residual
│
├── metrics/
│   └── run_alignment.py            # Cosine + CKA alignment by layer
│
├── probing/
│   └── probe_transfer.py           # Train EN probe → test EN/HI/LR by layer
│
├── interventions/                   # [POST-MIDSEM] Causal steering
│   ├── hooks.py                    # PyTorch forward hooks (SteeringContext)
│   ├── sweep.py                    # layer_sweep() and alpha_sweep()
│   └── evaluate_intervention.py    # CLI: run sweeps, output JSON
│
├── analysis/                        # [POST-MIDSEM] Controls & correlations
│   ├── ablations.py                # Random dir, shuffled concept, within-lang controls
│   ├── correlate.py                # Pearson/Spearman: alignment ↔ ATE
│   └── run_ablations.py            # CLI: run all ablation controls
│
├── evaluation/
│   ├── classification.py           # full_eval() with per-class F1 + confusion matrix
│   └── causal_metrics.py           # delta_logit, delta_prob, ATE, success_rate
│
├── report/
│   ├── plot_midsem.py              # Cosine/CKA/probe plots with std bands + F1 heatmap
│   └── plot_interventions.py       # Δlogit curves, alpha-sweep, ablation bar charts
│
└── utils/
    ├── hf.py                       # EncoderBundle, TranslatorBundle, load/translate
    ├── io.py                       # read/write JSON and JSONL
    ├── labeling.py                 # LABELS_6, LABEL2ID, extract_label_from_row
    ├── metrics.py                  # cosine_similarity_rows, linear_cka
    ├── pooling.py                  # pool_hidden (CLS / masked mean)
    └── seed.py                     # set_seed (Python/NumPy/PyTorch)
```

---

## 5. HDF5 Representation File Structure

`artifacts/reps/xlmr_large.h5`:
```
root/
  attrs: model_id, pool, max_length, dtype, n_layers, dim, ...
  train/
    en/reps   (N_train, L, D)   float16
    hi/reps   (N_train, L, D)   float16
    cy/reps   (N_train, L, D)   float16
  dev/  ...
  test/ ...
```

---

## 6. Directions File Structure

`artifacts/directions/xlmr_large/`:
```
mean_diff/
    CH.npy   (L, D)   float32   unit concept vectors, one per layer
    FC.npy   (L, D)
    ...
probe_weight/
    CH.npy   (L, D)
    ...
pca_residual/
    CH.npy   (L, D)
meta.json
```

---

## 7. Step-by-Step Running Guide

### 7.1 Setup (Windows + conda or venv)

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows PowerShell
# OR: source .venv/bin/activate  (bash/Linux/Mac)

# Install all dependencies
pip install -r requirements.txt

# Verify GPU (optional but strongly recommended for speed)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 7.2 Run Unit Tests First

```powershell
pytest tests/ -v --tb=short
```

All tests should pass without GPU or real data.

### 7.3 Full Pipeline (Recommended)

**On Linux/Mac/WSL:**
```bash
bash run_full.sh                     # Full run (≈3-6h with GPU)
bash run_full.sh --small             # Quick test with 50 examples/class
bash run_full.sh --force_cpu --small # CPU-only quick test
bash run_full.sh --lr_lang sw        # Use Swahili instead of Welsh
```

**On Windows (PowerShell — step by step):**
Use the commands in Section 7.4 below directly.

### 7.4 Manual Step-by-Step (Windows PowerShell)

#### Step 1: Build parallel dataset
```powershell
python -m src.data.build_parallel_dataset `
  --out_dir artifacts/data `
  --n_per_class 500 `
  --seed 42 `
  --lr_lang cy `
  --dataset_name MLNTeam-Unical/MoralTextManipulation `
  --dataset_config unconditioned `
  --dataset_split revise `
  --translator_en_hi Helsinki-NLP/opus-mt-en-hi `
  --translator_en_lr Helsinki-NLP/opus-mt-en-cy
```
> ⚠️ Downloads ~2 GB of MT models on first run. Uses HuggingFace cache thereafter.

#### Step 2a: Cache XLM-R representations
```powershell
python -m src.encode.cache_reps `
  --data_jsonl artifacts/data/parallel.jsonl `
  --splits_json artifacts/data/splits.json `
  --model_id xlm-roberta-large `
  --out_h5 artifacts/reps/xlmr_large.h5 `
  --pool cls --batch_size 16 --max_length 192 --dtype float16
```

#### Step 2b: Cache mBERT representations
```powershell
python -m src.encode.cache_reps `
  --data_jsonl artifacts/data/parallel.jsonl `
  --splits_json artifacts/data/splits.json `
  --model_id bert-base-multilingual-cased `
  --out_h5 artifacts/reps/mbert.h5 `
  --pool cls --batch_size 32 --max_length 192 --dtype float16
```

#### Step 3: Alignment diagnostics
```powershell
python -m src.metrics.run_alignment `
  --data_jsonl artifacts/data/parallel.jsonl `
  --splits_json artifacts/data/splits.json `
  --reps_h5 artifacts/reps/xlmr_large.h5 `
  --out_json artifacts/results/xlmr_alignment.json --cka_max_n 1200
```

#### Step 4: Probe transfer (cross-lingual)
```powershell
python -m src.probing.probe_transfer `
  --data_jsonl artifacts/data/parallel.jsonl `
  --splits_json artifacts/data/splits.json `
  --reps_h5 artifacts/reps/xlmr_large.h5 `
  --out_json artifacts/results/xlmr_probe_transfer.json `
  --max_train 6000
```

#### Step 5: Extract concept directions
```powershell
python -m src.concepts.extract_directions `
  --data_jsonl artifacts/data/parallel.jsonl `
  --splits_json artifacts/data/splits.json `
  --reps_h5 artifacts/reps/xlmr_large.h5 `
  --out_dir artifacts/directions/xlmr_large `
  --method mean_diff probe_weight pca_residual
```

#### Step 6: Intervention layer sweep
```powershell
python -m src.interventions.evaluate_intervention `
  --data_jsonl artifacts/data/parallel.jsonl `
  --splits_json artifacts/data/splits.json `
  --reps_h5 artifacts/reps/xlmr_large.h5 `
  --reps_h5 artifacts/reps/xlmr_large.h5 `
  --directions_dir artifacts/directions/xlmr_large/mean_diff `
  --model_id xlm-roberta-large `
  --lang hi --target_label CH `
  --sweep_type layer --alpha 1.0 `
  --out_json artifacts/results/intv_xlmr_hi_CH_layer.json
```

#### Step 6b: Intervention alpha sweep
```powershell
python -m src.interventions.evaluate_intervention `
  --data_jsonl artifacts/data/parallel.jsonl `
  --splits_json artifacts/data/splits.json `
  --reps_h5 artifacts/reps/xlmr_large.h5 `
  --reps_h5 artifacts/reps/xlmr_large.h5 `
  --directions_dir artifacts/directions/xlmr_large/mean_diff `
  --model_id xlm-roberta-large `
  --lang hi --target_label CH `
  --sweep_type alpha `
  --alphas 0 0.25 0.5 1.0 1.5 2.0 3.0 `
  --out_json artifacts/results/intv_xlmr_hi_CH_alpha.json
```

#### Step 7: Ablation controls
```powershell
python -m src.analysis.run_ablations `
  --data_jsonl artifacts/data/parallel.jsonl `
  --splits_json artifacts/data/splits.json `
  --reps_h5 artifacts/reps/xlmr_large.h5 `
  --directions_dir artifacts/directions/xlmr_large/mean_diff `
  --model_id xlm-roberta-large `
  --lang hi --target_label CH --alpha 1.0 `
  --out_dir artifacts/results/ablations/xlmr_hi_CH
```

#### Step 8: Generate all plots
```powershell
# Midsem plots  
python -m src.report.plot_midsem `
  --alignment_json artifacts/results/xlmr_alignment.json `
  --probe_json artifacts/results/xlmr_probe_transfer.json `
  --intervention_json artifacts/results/intv_xlmr_hi_CH_layer.json `
  --out_dir artifacts/figs/xlmr_large

# Intervention plots
python -m src.report.plot_interventions `
  --layer_sweep_jsons artifacts/results/intv_xlmr_hi_CH_layer.json `
  --alpha_sweep_json  artifacts/results/intv_xlmr_hi_CH_alpha.json `
  --true_json    artifacts/results/intv_xlmr_hi_CH_layer.json `
  --random_json  artifacts/results/ablations/xlmr_hi_CH/random_direction.json `
  --shuffled_json artifacts/results/ablations/xlmr_hi_CH/shuffled_concept.json `
  --out_dir artifacts/figs/interventions
```

---

## 8. Output Files Reference

| Path | Description |
|------|-------------|
| `artifacts/data/parallel.jsonl` | All 3000 examples with EN/HI/LR text |
| `artifacts/data/splits.json` | Train/dev/test IDs |
| `artifacts/reps/xlmr_large.h5` | Layer-wise reps for all splits + languages |
| `artifacts/reps/mbert.h5` | Same for mBERT |
| `artifacts/directions/xlmr_large/mean_diff/CH.npy` | (L, D) unit concept vector |
| `artifacts/results/xlmr_alignment.json` | Cosine + CKA by layer |
| `artifacts/results/xlmr_probe_transfer.json` | Probe acc/F1/per-class F1 by layer |
| `artifacts/results/intv_xlmr_hi_CH_layer.json` | Layer sweep intervention results |
| `artifacts/results/intv_xlmr_hi_CH_alpha.json` | Alpha sweep results |
| `artifacts/results/ablations/xlmr_hi_CH/random_direction.json` | Random control |
| `artifacts/results/ablations/xlmr_hi_CH/shuffled_concept.json` | Shuffled concept control |
| `artifacts/figs/xlmr_large/cosine_alignment.png` | Cosine plot with std bands |
| `artifacts/figs/xlmr_large/cka.png` | CKA by layer |
| `artifacts/figs/xlmr_large/probe_acc.png` | Probe accuracy EN→{EN,HI,LR} |
| `artifacts/figs/xlmr_large/probe_macro_f1.png` | Probe macro-F1 per layer |
| `artifacts/figs/xlmr_large/probe_f1_heatmap_{lang}.png` | Per-class F1 heatmap |
| `artifacts/figs/interventions/layer_sweep_delta_prob_mean.png` | Δp by layer |
| `artifacts/figs/interventions/alpha_sweep.png` | Δlogit/Δp/success vs α |
| `artifacts/figs/interventions/ablation_comparison.png` | True vs controls bar chart |

---

## 9. Expected Results (Research-Paper Quality)

### Alignment (RQ1)
- **XLM-R large**: cosine similarity peaks around layers 18-22 for EN↔HI (expected ≥0.85), lower for EN↔LR
- **mBERT**: alignment typically peaks at layers 8-10, lower overall — good comparative data point

### Probe Transfer (RQ1 + RQ3)
- EN→EN upper bound: ~85-90% accuracy
- EN→HI: ~70-80% at best layers (demonstrates concept transfer)
- EN→LR (Welsh): ~50-65% — harder, since Welsh is lower-resource

### Interventions (RQ2 + RQ3)
- At the best layer, Δp for target class should be significantly positive (>0.05)
- Success rate should be >0.6 for well-aligned language pairs
- Alpha sweep should show monotonic increase then saturation (supports the RepE hypothesis)

### Ablation Controls
- Random direction control: Δp ≈ 0 (baseline)
- Shuffled concept: Δp small or negative (concept-specific effect)
- If true intervention >> both controls → strong evidence for concept-specific steering

---

## 10. Tips for Best Results

1. **Use GPU** — Step 1 (translation) and Step 2 (encoding) will take 4-8× longer on CPU
2. **Run XLM-R first** — It has more layers (24) and higher alignment, making it the primary model for the paper
3. **Sweep all 6 labels** — Run Steps 5-7 for CH, FC, LB, AS, PD, NM to get a full picture
4. **Try all 3 direction methods** — `mean_diff` is fastest; `pca_residual` often gives cleaner directions
5. **Use `--small` first** — Verify the full pipeline runs before committing to a full 3000-example run
6. **HuggingFace cache** — Set `HF_HOME=path/to/big/drive` if your C: drive is small

---

## 11. Bugs Fixed

| File | Bug | Fix |
|------|-----|-----|
| `src/concepts/extract_directions.py` | `json_dump` called before definition (NameError) | Moved helper to top of file |
| `src/report/plot_midsem.py` | No `figsize`, no DPI, unreadable plots | Rewrote with proper styling, std bands, heatmaps |
| `requirements.txt` | Missing `seaborn`, `pandas` | Added both |

## 12. New Files Added

| File | Purpose |
|------|---------|
| `src/interventions/hooks.py` | PyTorch forward hook (SteeringContext) |
| `src/interventions/sweep.py` | Layer sweep + alpha sweep logic |
| `src/interventions/evaluate_intervention.py` | CLI for running interventions |
| `src/analysis/ablations.py` | Random/shuffled/within-lang controls |
| `src/analysis/correlate.py` | Alignment ↔ ATE Pearson/Spearman + scatter plots |
| `src/analysis/run_ablations.py` | CLI for ablation controls |
| `src/evaluation/classification.py` | full_eval with per-class F1 + confusion matrix |
| `src/evaluation/causal_metrics.py` | Δlogit, Δp, ATE, success_rate |
| `src/report/plot_interventions.py` | Layer sweep, alpha sweep, ablation bar charts |
| `tests/test_utils.py` | 25+ unit tests for all utility modules |
| `run_full.sh` | Full end-to-end pipeline script |
| `run_interventions.sh` | Standalone intervention pipeline |
