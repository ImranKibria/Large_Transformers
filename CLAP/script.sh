#!/bin/bash

#SBATCH --time=12:00:00 
#SBATCH --job-name=clap
#SBATCH --account=PAS2301

#SBATCH --mem=64gb
#SBATCH -o testing/results.out

#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4

module load miniconda3
source activate vit_env
python test.py