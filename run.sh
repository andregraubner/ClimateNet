source ~/.bashrc
conda activate climatenet

USER = "atabin"
MODEL = "example"
SCRATCH_FOLDER = "/cluster/scratch/$USER/"
python ~/ClimateNet_AI4Good/run.py $MODEL \
    --data_dir /cluster/work/igp_psr/ai4good/group-1b/data/ \
    --checkpoint_path $SCRATCH_FOLDER/checkpoints/$MODEL/

echo "Done!"