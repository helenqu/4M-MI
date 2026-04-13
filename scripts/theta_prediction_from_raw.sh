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

# ml python
# activate fourm

srun python raw_vs_tokenized_theta.py