#!/usr/bin/env bash
set -euo pipefail

N_PER_CLASS=500
FORCE_CPU=""
LR_LANG="cy"
LR_TRANSLATOR="Helsinki-NLP/opus-mt-en-cy"
TARGET_LABEL="CH"

while [[ $# -gt 0 ]]; do
  case $1 in
    --small)     N_PER_CLASS=50; shift ;;
    --force_cpu) FORCE_CPU="--force_cpu"; shift ;;
    --lr_lang)   LR_LANG="$2"; LR_TRANSLATOR="Helsinki-NLP/opus-mt-en-$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  Cross-Lingual Concept Transfer — Full Pipeline"
echo "  n_per_class=${N_PER_CLASS}  force_cpu=${FORCE_CPU:-none}  lr_lang=${LR_LANG}"
echo "============================================================"

python -m src.data.build_parallel_dataset   --out_dir artifacts/data   --n_per_class "$N_PER_CLASS"   --seed 42   --lr_lang "$LR_LANG"   --dataset_name MLNTeam-Unical/MoralTextManipulation   --dataset_config unconditioned   --dataset_split revise   --translator_en_hi Helsinki-NLP/opus-mt-en-hi   --translator_en_lr "$LR_TRANSLATOR"   $FORCE_CPU

python -m src.encode.cache_reps   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --model_id xlm-roberta-large   --out_h5 artifacts/reps/xlmr_large.h5   --pool cls --batch_size 16 --max_length 192 --dtype float16   $FORCE_CPU

python -m src.encode.cache_reps   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --model_id bert-base-multilingual-cased   --out_h5 artifacts/reps/mbert.h5   --pool cls --batch_size 32 --max_length 192 --dtype float16   $FORCE_CPU

python -m src.metrics.run_alignment   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 artifacts/reps/xlmr_large.h5   --out_json artifacts/results/xlmr_alignment.json   --cka_max_n 1200

python -m src.metrics.run_alignment   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 artifacts/reps/mbert.h5   --out_json artifacts/results/mbert_alignment.json   --cka_max_n 1200

python -m src.probing.probe_transfer   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 artifacts/reps/xlmr_large.h5   --out_json artifacts/results/xlmr_probe_transfer.json   --max_train 6000

python -m src.probing.probe_transfer   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 artifacts/reps/mbert.h5   --out_json artifacts/results/mbert_probe_transfer.json   --max_train 6000

python -m src.concepts.extract_directions   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 artifacts/reps/xlmr_large.h5   --out_dir artifacts/directions/xlmr_large   --method mean_diff probe_weight pca_residual

python -m src.concepts.extract_directions   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 artifacts/reps/mbert.h5   --out_dir artifacts/directions/mbert   --method mean_diff probe_weight

python -m src.interventions.evaluate_intervention   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 artifacts/reps/xlmr_large.h5   --directions_dir artifacts/directions/xlmr_large/mean_diff   --model_id xlm-roberta-large   --lang hi   --target_label "$TARGET_LABEL"   --readout_json artifacts/readouts/xlmr_readout.json   --sweep_type layer   --alpha 1.0   --out_json "artifacts/results/intv_xlmr_hi_${TARGET_LABEL}_layer.json"   $FORCE_CPU

python -m src.interventions.evaluate_intervention   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 artifacts/reps/xlmr_large.h5   --directions_dir artifacts/directions/xlmr_large/mean_diff   --model_id xlm-roberta-large   --lang hi   --target_label "$TARGET_LABEL"   --readout_json artifacts/readouts/xlmr_readout.json   --sweep_type alpha   --alphas 0 0.25 0.5 1.0 1.5 2.0 3.0   --out_json "artifacts/results/intv_xlmr_hi_${TARGET_LABEL}_alpha.json"   $FORCE_CPU

python -m src.analysis.run_ablations   --data_jsonl artifacts/data/parallel.jsonl   --splits_json artifacts/data/splits.json   --reps_h5 artifacts/reps/xlmr_large.h5   --directions_dir artifacts/directions/xlmr_large/mean_diff   --model_id xlm-roberta-large   --lang hi   --target_label "$TARGET_LABEL"   --readout_json artifacts/readouts/xlmr_readout.json   --alpha 1.0   --out_dir "artifacts/results/ablations/xlmr_hi_${TARGET_LABEL}"   $FORCE_CPU

python -m src.analysis.correlate   --alignment_json artifacts/results/xlmr_alignment.json   --intervention_json "artifacts/results/intv_xlmr_hi_${TARGET_LABEL}_layer.json"   --out_json "artifacts/results/correlations_xlmr_hi_${TARGET_LABEL}.json"   --out_dir artifacts/figs/xlmr_large

python -m src.report.plot_midsem   --alignment_json artifacts/results/xlmr_alignment.json   --probe_json artifacts/results/xlmr_probe_transfer.json   --intervention_json "artifacts/results/intv_xlmr_hi_${TARGET_LABEL}_layer.json"   --out_dir artifacts/figs/xlmr_large

python -m src.report.plot_midsem   --alignment_json artifacts/results/mbert_alignment.json   --probe_json artifacts/results/mbert_probe_transfer.json   --out_dir artifacts/figs/mbert

python -m src.report.plot_interventions   --layer_sweep_jsons "artifacts/results/intv_xlmr_hi_${TARGET_LABEL}_layer.json"   --alpha_sweep_json "artifacts/results/intv_xlmr_hi_${TARGET_LABEL}_alpha.json"   --true_json "artifacts/results/intv_xlmr_hi_${TARGET_LABEL}_layer.json"   --random_json "artifacts/results/ablations/xlmr_hi_${TARGET_LABEL}/random_direction.json"   --shuffled_json "artifacts/results/ablations/xlmr_hi_${TARGET_LABEL}/shuffled_concept.json"   --out_dir artifacts/figs/interventions

echo "Done. See artifacts/{data,reps,results,directions,readouts,figs}."
