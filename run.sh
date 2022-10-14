source ~/.bashrc
conda activate climatenet

user="atabin"
model="example"
scratch_folder="/cluster/scratch/"
echo "Running with following parameters : "
echo $user
echo $model
echo $scratch_folder
python ~/ClimateNet_AI4Good/run.py $model \
    --data_dir /cluster/work/igp_psr/ai4good/group-1b/data/ \
    --checkpoint_path $scratch_folder/checkpoints/$model/ \
    --scratch_folder $scratch_folder \
    --user $user

echo "Done!"