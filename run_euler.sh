cd ~/ClimateNet_AI4Good/
git pull
TAG=$(git rev-parse --short HEAD)


# command to run on Euler cluster : 
# This command took ages
# bsub -n 40 -B -N -R "rusage[mem=4500,ngpus_excl_p=8]" -R "select[gpu_model0==GeForceGTX1080Ti]" "python example.py"
# This one is working straight away 
# bsub -n 40 -B -N -R "rusage[mem=4500,ngpus_excl_p=8]" "python example.py"
# To put the output direct in outputs folder
# bsub -n 40 -oo "outputs/output" -B -N -R "rusage[mem=4500,ngpus_excl_p=8]" "python example.py"

bsub -n 1 -W 12:00 -J $TAG -B -N -R "rusage[mem=4096,ngpus_excl_p=8]" -R "select[gpu_mtotal0>=8192]" <~/ClimateNet_AI4Good/run.sh