#!/bin/sh
#PBS -N extr_dino
#PBS -q sg
#PBS -l select=1:ngpus=4:mpiprocs=4
#PBS -l walltime=04:00:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
module load gcc ompi
. /home/c30746/miniconda3/etc/profile.d/conda.sh
conda activate DiT
export HF_HOME=/work/c30636/dataset/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DINOV3_REPO_DIR=/work/c30636/dinov3
export DINOV3_CKPT_DIR=/work/c30636/RAEv2/pretrained_models/encoders/dinov3
export PYTHONPATH=/work/c30636/RAEv2/src:$PYTHONPATH
cd /work/c30636/DiT
: "${NGPU:=4}"; : "${BS:=16}"
: "${DATA_PATH:=/work/c30636/dataset/doclaynet/images}"; : "${OUT:=/work/c30636/dataset/doclaynet_dinov3_768}"; : "${MAX_IMGS:=0}"
echo "EXTRACT DINOv3@768: NGPU=$NGPU BS=$BS DATA=$DATA_PATH OUT=$OUT MAX=$MAX_IMGS"
torchrun --standalone --nproc_per_node=$NGPU extract_doclaynet_dinov3.py --bs "$BS" \
  --data-path "$DATA_PATH" --out "$OUT" --max-imgs "$MAX_IMGS"
echo "EXTRACT DONE"
