source ~/.bashrc
conda activate climatenet

user="atabin"
model="trained_cgnet"

scratch_folder="/cluster/scratch/$user"

save_dir="$scratch_folder/data/$model/"
checkpoint_path="$scratch_folder/checkpoints/$model/"
echo "Running with following parameters : "
echo $user
echo $model
echo $save_dir
python ~/ClimateNet_AI4Good/run.py $model \
    --data_dir /cluster/work/igp_psr/ai4good/group-1b/data/ \
    --checkpoint_path $checkpoint_path \
    --save_dir $save_dir

echo "Done!"