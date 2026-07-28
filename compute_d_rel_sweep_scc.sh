#!/bin/bash -l
#$ -P buinlp
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_type=V100|A40|A100|L40S|A6000|P100
#$ -l h_rt=04:00:00
#$ -j y
#$ -o logs/$JOB_ID_$JOB_NAME.log

set -euo pipefail
D="${1:?D required}"
R=$((D / 10))
cd /projectnb2/buinlp/rathin/GrokkedTransformer
module load miniconda
conda activate grokked
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

python compute_d_rel.py \
  --dataset "composition.${D}.${R}.3.6" \
  --model_dir outputs \
  --n_examples 150 \
  --save_path "d_rel_D${D}.json"
echo DONE
