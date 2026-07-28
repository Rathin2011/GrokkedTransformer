#!/bin/bash -l
#
# Resume/extend a previously-trained composition-task checkpoint past its
# original max_steps. Sibling to train_pilot_scc.sh: identical except it
# resumes from RESUME_FROM instead of the base "gpt2" weights, and drops
# --init_weights/--add_tokens (must NOT re-init weights or re-extend an
# already-extended vocab when resuming).
#
# Usage:
#   mkdir -p logs
#   qsub -N gt_extend -v RESUME_FROM=<output_dir>/checkpoint-<N>,DATASET_NAME=...,MAX_STEPS=... train_extend_scc.sh
#
#$ -P buinlp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=V100|A40|A100|L40S|A6000|P100
#$ -l h_rt=48:00:00
#$ -j y
#$ -m beas
#$ -o logs/$JOB_ID_$JOB_NAME.log

set -euo pipefail

REPO="/projectnb2/buinlp/rathin/GrokkedTransformer"
cd "${REPO}"
mkdir -p logs

: "${RESUME_FROM:?RESUME_FROM must be set to a checkpoint-<N> directory}"
DATASET_NAME="${DATASET_NAME:-composition.2000.200.3.6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
N_LAYERS="${N_LAYERS:-8}"
MAX_STEPS="${MAX_STEPS:-250000}"
SAVE_STEP="${SAVE_STEP:-5000}"
SAVE_STEP_DENSE="${SAVE_STEP_DENSE:--1}"
SAVE_STEP_DENSE_INTERVAL="${SAVE_STEP_DENSE_INTERVAL:-2000}"

OUTPUT_DIR="${REPO}/outputs/${DATASET_NAME}_wd${WEIGHT_DECAY}_L${N_LAYERS}_pilot"

module load miniconda
conda activate grokked

python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

echo "RESUME_FROM=${RESUME_FROM} DATASET_NAME=${DATASET_NAME} MAX_STEPS=${MAX_STEPS} OUTPUT_DIR=${OUTPUT_DIR}"

python main.py \
  --data_dir "data/${DATASET_NAME}/" \
  --model_name_or_path "${RESUME_FROM}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --max_seq_length 10 \
  --max_length 10 \
  --block_size 10 \
  --train_batch_size 512 \
  --eval_batch_size 512 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 1 \
  --save_step "${SAVE_STEP}" \
  --save_step_dense "${SAVE_STEP_DENSE}" \
  --save_step_dense_interval "${SAVE_STEP_DENSE_INTERVAL}" \
  --max_steps "${MAX_STEPS}" \
  --do_train \
  --scheduler constant_schedule_with_warmup \
  --fp16 \
  --evaluate_during_training \
  --predict_during_training \
  --n_layer "${N_LAYERS}"

echo "DONE"
