source ~/.bashrc
conda activate climatenet

python ~/ClimateNet_AI4Good/run.py example \
    --data_dir /cluster/work/igp_psr/ai4good/group-1b/data \
    --checkpoint_path /cluster/scratch/atabin/checkpoints/example/

echo "Done!"