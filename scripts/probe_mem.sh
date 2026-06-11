#!/bin/sh
#PBS -N probe_mem
#PBS -q sg
#PBS -l select=1:ngpus=1:mpiprocs=1
#PBS -l walltime=00:15:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else . "/home/c30746/miniconda3/etc/profile.d/conda.sh"; fi
conda activate DiT
export PYTHONPATH=/work/c30636/RAEv2/src:$PYTHONPATH
cd /work/c30636/DiT
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python probe_mem.py
