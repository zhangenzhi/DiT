#!/bin/sh
#PBS -N grid
#PBS -q sg
#PBS -l select=1:ngpus=1:mpiprocs=1
#PBS -l walltime=00:30:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
# Single-GPU 4x4 sample grid from a RAE-flow ckpt. vars: CKPT ARCH CFG OUT (+ optional AG_CKPT AG_SCALE)
module load gcc ompi
__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else . "/home/c30746/miniconda3/etc/profile.d/conda.sh"; fi
unset __conda_setup
conda activate DiT
export HF_HOME=/work/c30636/dataset/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DINOV3_REPO_DIR=/work/c30636/dinov3
export DINOV3_CKPT_DIR=/work/c30636/RAEv2/pretrained_models/encoders/dinov3
export PYTHONPATH=/work/c30636/RAEv2/src:$PYTHONPATH
cd /work/c30636/DiT
: "${ARCH:=dit_rope}"; : "${CFG:=1.8}"; : "${OUT:=/work/c30636/DiT/outputs/grid.png}"
AG_ARGS=""; [ -n "$AG_CKPT" ] && AG_ARGS="--ag-ckpt $AG_CKPT --ag-scale ${AG_SCALE:-2.0}"
echo "GRID: CKPT=$CKPT ARCH=$ARCH CFG=$CFG OUT=$OUT $AG_ARGS"
python sample_grid.py --ckpt "$CKPT" --arch "$ARCH" --cfg-scale "$CFG" --out "$OUT" $AG_ARGS
echo "done -> $OUT"
