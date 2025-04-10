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
source ~/.venv/fourm/bin/activate

srun ~/.venv/fourm/bin/python save_vq_tokens.py \
    --tokenizer_id CLIP-B16/ViTB-ViTB_8k_224/checkpoint-final \
    --tokenizers_root /mnt/home/hqu10/ceph/4m_outputs/tokenization/vqvae \
    --data_root /mnt/home/hqu10/ceph/test_4m_dataset \
    --folder_suffix tokenized_wds