#!/bin/bash

#SBATCH -p gpu
#SBATCH --exclude=workergpu156
#SBATCH -C a100-80gb
#SBATCH -t 4:00:00
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=hqu@flatironinstitute.org
#SBATCH --mail-type=END,FAIL

ml python
activate fourm

srun ~/.venv/fourm/bin/python run_training_vqvae.py --config cfgs/default/tokenization/vqvae/CLIP-B16/ViTB-ViTB_8k_224.yaml