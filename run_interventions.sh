#!/usr/bin/env bash
set -euo pipefail

MODEL="xlmr"
LANG="hi"
LABEL="CH"
METHOD="mean_diff"
ALPHA="1.0"
FORCE_CPU=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)  MODEL="$2"; shift 2 ;;
    --lang)   LANG="$2"; shift 2 ;;
    --label)  LABEL="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --alpha)  ALPHA="$2"; shift 2 ;;
    --force_cpu) FORCE_CPU="--force_cpu"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$MODEL" == "xlmr" ]]; then
  MODEL_ID="xlm-roberta-large"
  REPS_H5="artifacts/reps/xlmr_large.h5"
  DIRS_BASE="artifacts/directions/xlmr_large"
  READOUT_JSON="artifacts/readouts/xlmr_readout.json"
else
  MODEL_ID="bert-base-multilingual-cased"
  REPS_H5="artifacts/reps/mbert.h5"
  DIRS_BASE="artifacts/directions/mbert"
  READOUT_JSON="artifacts/readouts/mbert_readout.json"
fi

DIRS_DIR="${DIRS_BASE}/${METHOD}"
OUT_PREFIX="artifacts/results/intv_${MODEL}_${LANG}_${LABEL}"
ABL_DIR="artifacts/results/ablations/${MODEL}_${LANG}_${LABEL}"
FIG_DIR="artifacts/figs/interventions/${MODEL}_${LANG}_${LABEL}"

python -m src.concepts.extract_directions   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 "$REPS_H5"   --out_dir "$DIRS_BASE"   --method "$METHOD"

python -m src.interventions.evaluate_intervention   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 "$REPS_H5"   --directions_dir "$DIRS_DIR"   --model_id "$MODEL_ID"   --lang "$LANG"   --target_label "$LABEL"   --readout_json "$READOUT_JSON"   --sweep_type layer   --alpha "$ALPHA"   --out_json "${OUT_PREFIX}_layer.json"   $FORCE_CPU

python -m src.interventions.evaluate_intervention   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 "$REPS_H5"   --directions_dir "$DIRS_DIR"   --model_id "$MODEL_ID"   --lang "$LANG"   --target_label "$LABEL"   --readout_json "$READOUT_JSON"   --sweep_type alpha   --alphas 0 0.25 0.5 1.0 1.5 2.0 3.0   --out_json "${OUT_PREFIX}_alpha.json"   $FORCE_CPU

python -m src.analysis.run_ablations   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 "$REPS_H5"   --directions_dir "$DIRS_DIR"   --model_id "$MODEL_ID"   --lang "$LANG"   --target_label "$LABEL"   --readout_json "$READOUT_JSON"   --alpha "$ALPHA"   --out_dir "$ABL_DIR"   $FORCE_CPU

python -m src.report.plot_interventions   --layer_sweep_jsons "${OUT_PREFIX}_layer.json"   --alpha_sweep_json "${OUT_PREFIX}_alpha.json"   --true_json "${OUT_PREFIX}_layer.json"   --random_json "${ABL_DIR}/random_direction.json"   --shuffled_json "${ABL_DIR}/shuffled_concept.json"   --out_dir "$FIG_DIR"

echo "Done. Figures in: ${FIG_DIR}"
