#!/bin/bash

#SBATCH -p gpu
#SBATCH --exclude=workergpu156
#SBATCH -C a100-80gb
#SBATCH -t 12:00:00
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=hqu@flatironinstitute.org
#SBATCH --mail-type=END,FAIL

NUM_GPUS=4

# ml python
# activate fourm

srun torchrun --nproc_per_node=$NUM_GPUS \
    run_training_vqvae.py \
    --config cfgs/default/tokenization/vqvae/rgb/mi_rgb1.yaml \
    --data_path /mnt/ceph/users/hqu10/Case2_rho0999 \
    --output_dir /mnt/ceph/users/hqu10/mi_outputs/rgb1_tokenizer_case2_rho0999