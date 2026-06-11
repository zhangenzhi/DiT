#!/bin/sh
#PBS -N round_trip
#PBS -q sg
#PBS -l select=1:ngpus=1:mpiprocs=1
#PBS -l walltime=00:20:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
module load gcc ompi
__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else . "/home/c30746/miniconda3/etc/profile.d/conda.sh"; fi
conda activate DiT
export HF_HOME=/work/c30636/dataset/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DINOV3_REPO_DIR=/work/c30636/dinov3 DINOV3_CKPT_DIR=/work/c30636/RAEv2/pretrained_models/encoders/dinov3
export PYTHONPATH=/work/c30636/RAEv2/src:$PYTHONPATH
cd /work/c30636/DiT
python round_trip.py
