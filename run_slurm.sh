#!/bin/sh
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24GB
#SBATCH --mem-per-cpu=24GB
#SBATCH --time=24:00:00

source ~/.bashrc 
source activate
conda deactivate 

conda activate tcd



python ~/ai4good/ClimateNet_AI4Good/segment_from_scratch.py


exit 0
