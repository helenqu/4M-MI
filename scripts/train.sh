#!/bin/bash

#SBATCH -p gpu
#SBATCH --exclude=workergpu156
#SBATCH -C h100
#SBATCH -t 12:00:00
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=hqu@flatironinstitute.org
#SBATCH --mail-type=END,FAIL

# OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_training_4m.py \
# --config cfgs/default/4m/models/<config>.yaml

ml python
source ~/.venv/fourm/bin/activate
cd /mnt/home/hqu10/ml-4m
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_training_4m.py \
    --config cfgs/default/4m/models/main/helen_test.yaml
# srun python run_training_4m.py --config cfgs/default/4m/models/main/helen_test.yaml