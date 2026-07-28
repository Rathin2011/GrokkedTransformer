#!/bin/bash -l
#$ -P buinlp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=V100|A40|A100|L40S|A6000|P100
#$ -l h_rt=04:00:00
#$ -j y
#$ -m beas
#$ -o logs/$JOB_ID_$JOB_NAME.log

set -euo pipefail

REPO="/projectnb2/buinlp/rathin/GrokkedTransformer"
cd "${REPO}"
mkdir -p logs

module load miniconda
conda activate grokked

python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

python causal_tracing_composition_fixed.py \
  --dataset composition.500.50.3.6 \
  --model_dir outputs \
  --checkpoint checkpoint-230000 \
  --num_layer 8 \
  --wd 0.1 \
  --save_path causal_tracing_D500_checkpoint230000.json

echo "DONE"
