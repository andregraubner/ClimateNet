#!/bin/sh
#SBATCH --gpus=1
#SBATCH --gres=gpumen:24GB
#SBATCH --mem-per-cpu=24GB
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate tcd



python ~/ai4good/Climatenet_AI4Good/segment_form_scetch_cl.py


exit 0
