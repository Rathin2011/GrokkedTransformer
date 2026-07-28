#!/bin/bash -l
#
# Pilot training run for the composition task (Wang et al., arXiv:2405.15071).
# Bounded step count to confirm the environment/pipeline work end-to-end
# before committing to anything near the paper's full 1.5M-step runs.
#
# Usage:
#   mkdir -p logs
#   qsub -N gt_pilot train_pilot_scc.sh
#
#$ -P buinlp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=V100|A40|A100|L40S|A6000|P100
#$ -l h_rt=08:00:00
#$ -j y
#$ -m beas
#$ -o logs/$JOB_ID_$JOB_NAME.log

set -euo pipefail

REPO="/projectnb2/buinlp/rathin/GrokkedTransformer"
cd "${REPO}"
mkdir -p logs

DATASET_NAME="${DATASET_NAME:-composition.2000.200.3.6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
N_LAYERS="${N_LAYERS:-8}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEP="${SAVE_STEP:-5000}"

OUTPUT_DIR="${REPO}/outputs/${DATASET_NAME}_wd${WEIGHT_DECAY}_L${N_LAYERS}_pilot"

module load miniconda
conda activate grokked

python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

echo "DATASET_NAME=${DATASET_NAME} N_LAYERS=${N_LAYERS} MAX_STEPS=${MAX_STEPS} OUTPUT_DIR=${OUTPUT_DIR}"

python main.py \
  --data_dir "data/${DATASET_NAME}/" \
  --model_name_or_path gpt2 \
  --weight_decay "${WEIGHT_DECAY}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_seq_length 10 \
  --max_length 10 \
  --block_size 10 \
  --train_batch_size 512 \
  --eval_batch_size 512 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 1 \
  --save_step "${SAVE_STEP}" \
  --max_steps "${MAX_STEPS}" \
  --do_train \
  --scheduler constant_schedule_with_warmup \
  --fp16 \
  --evaluate_during_training \
  --predict_during_training \
  --init_weights \
  --add_tokens \
  --n_layer "${N_LAYERS}"

echo "DONE"
