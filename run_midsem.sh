#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end midsem run (XLM-R). Adjust n_per_class for your compute.
python -m src.data.build_parallel_dataset \
  --out_dir artifacts/data \
  --n_per_class 500 \
  --seed 42 \
  --lr_lang cy \
  --dataset_name MLNTeam-Unical/MoralTextManipulation \
  --dataset_config unconditioned \
  --dataset_split revise \
  --translator_en_hi Helsinki-NLP/opus-mt-en-hi \
  --translator_en_lr Helsinki-NLP/opus-mt-en-cy

python -m src.encode.cache_reps \
  --data_jsonl artifacts/data/parallel.jsonl \
  --splits_json artifacts/data/splits.json \
  --model_id xlm-roberta-large \
  --out_h5 artifacts/reps/xlmr_large.h5 \
  --pool cls \
  --batch_size 16 \
  --max_length 192 \
  --dtype float16

python -m src.metrics.run_alignment \
  --data_jsonl artifacts/data/parallel.jsonl \
  --splits_json artifacts/data/splits.json \
  --reps_h5 artifacts/reps/xlmr_large.h5 \
  --out_json artifacts/results/xlmr_alignment.json \
  --cka_max_n 1200

python -m src.probing.probe_transfer \
  --data_jsonl artifacts/data/parallel.jsonl \
  --splits_json artifacts/data/splits.json \
  --reps_h5 artifacts/reps/xlmr_large.h5 \
  --out_json artifacts/results/xlmr_probe_transfer.json \
  --max_train 6000

python -m src.concepts.extract_directions \
  --data_jsonl artifacts/data/parallel.jsonl \
  --splits_json artifacts/data/splits.json \
  --reps_h5 artifacts/reps/xlmr_large.h5 \
  --out_dir artifacts/directions/xlmr_large \
  --method mean_diff probe_weight

python -m src.report.plot_midsem \
  --alignment_json artifacts/results/xlmr_alignment.json \
  --probe_json artifacts/results/xlmr_probe_transfer.json \
  --out_dir artifacts/figs/xlmr_large

echo "DONE. See artifacts/results and artifacts/figs."
