#!/bin/bash

#SBATCH --time=4:00:00 
#SBATCH --job-name=wav2vec
#SBATCH --account=PAS2301

#SBATCH --mem=32gb
#SBATCH -o testing/results.out

#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4

module load miniconda3
source activate vit_env
python test.py