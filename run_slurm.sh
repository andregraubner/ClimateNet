#!/bin/sh
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24GB
#SBATCH --mem-per-cpu=24GB
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate tcd



python ~/ai4good/ClimateNet_AI4Good/segment_from_scratch_cl.py


exit 0
