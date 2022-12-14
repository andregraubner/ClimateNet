#!/bin/sh
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32GB
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=48:00:00

source ~/.bashrc 
source activate
conda deactivate 

conda activate tcd



python ~/ai4good/ClimateNet_AI4Good/segment_from_scratch.py


exit 0
