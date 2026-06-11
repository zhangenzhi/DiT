#!/bin/sh
#PBS -N fid_partial
#PBS -q sg
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
# Early FID on the first N PNGs already produced by the running 50k job. vars: DIR, N
__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else . "/home/c30746/miniconda3/etc/profile.d/conda.sh"; fi
unset __conda_setup
cd /work/c30636/DiT; conda activate DiT
export HF_HOME=/work/c30636/dataset/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${N:=24000}"
python fid_partial.py --dir "$DIR" --n "$N"
