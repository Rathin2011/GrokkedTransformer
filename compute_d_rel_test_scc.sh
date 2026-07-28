#!/bin/bash -l
#$ -P buinlp
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_type=V100|A40|A100|L40S|A6000|P100
#$ -l h_rt=00:20:00
#$ -j y
#$ -o logs/$JOB_ID_$JOB_NAME.log

set -euo pipefail
cd /projectnb2/buinlp/rathin/GrokkedTransformer
module load miniconda
conda activate grokked
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

python compute_d_rel.py \
  --dataset composition.500.50.3.6 \
  --model_dir outputs \
  --checkpoint checkpoint-230000 \
  --n_examples 150 \
  --save_path d_rel_test_D500_230000.json
echo DONE
