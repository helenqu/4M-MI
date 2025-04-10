#!/bin/bash

# OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_training_4m.py \
# --config cfgs/default/4m/models/<config>.yaml

ml python
source ~/.venv/fourm/bin/activate
cd /mnt/home/hqu10/ml-4m
srun python run_training_4m.py --config cfgs/default/4m/models/main/helen_test.yaml