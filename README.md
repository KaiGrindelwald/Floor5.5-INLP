# Cross-Lingual Concept Transfer (Full Research Pipeline)

**Proposal:** Cross-Lingual Concept Transfer via Representation Engineering  
**Full guide:** See [`implementation.md`](implementation.md) for the runbook.

## What this codebase now supports

This version is designed so that you can finish the pipeline first, then run it once and directly obtain analysable artifacts.

### Core stages
- `src/data/` — build balanced EN/HI/LR dataset from MoralTextManipulation and export a gold-check subset
- `src/encode/` — cache layer-wise pooled states in HDF5 for XLM-R and mBERT
- `src/metrics/` — cosine + CKA alignment by layer
- `src/probing/` — EN-trained probe transfer to EN/HI/LR with per-class F1
- `src/concepts/` — concept directions via `mean_diff`, `probe_weight`, and `pca_residual`
- `src/interventions/` — **causal steering with a frozen encoder + trained linear probe readout**
- `src/analysis/` — random-direction and shuffled-concept ablations + alignment↔ATE correlation
- `src/report/` — plotting scripts for both mid-sem and post-midsem figures

## Important design fix

The intervention stage no longer relies on a randomly initialised sequence-classification head. Instead, it fits a probe on cached English encoder representations and uses that probe as the readout during hooked forward passes. This makes the intervention outputs meaningful for your proposal's encoder-only setting.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scriptsctivate
pip install -r requirements.txt
pytest tests/ -v
bash run_full.sh --small
```

## Main outputs

```text
artifacts/data/                 parallel dataset, splits, summary
artifacts/reps/                 cached encoder representations
artifacts/results/              alignment, probe, intervention, ablation, correlation JSON
artifacts/directions/           per-label concept vectors
artifacts/readouts/             cached probe readouts used for interventions
artifacts/figs/                 paper-ready plots
```

## Recommended default experiment

```bash
bash run_full.sh                  # full pipeline
bash run_interventions.sh   --model xlmr --lang hi --label CH --method mean_diff
```

## Models

| Model | HF ID | Layers | Hidden dim |
|---|---|---:|---:|
| XLM-R large | `xlm-roberta-large` | 24 | 1024 |
| mBERT | `bert-base-multilingual-cased` | 12 | 768 |

## Lower-resource language

The pipeline is parameterized. Default is Welsh (`cy`) for the repository run scripts, but you can swap to another OPUS-supported language such as Swahili (`sw`) via `--lr_lang sw`.
