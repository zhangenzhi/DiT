#!/bin/sh
#PBS -N rae_recipe_lg
#PBS -q lg
#PBS -l select=4:ngpus=4:mpiprocs=4
#PBS -l walltime=14:00:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
# RAEv2-recipe run on our DiT_RoPE (uniform 1152): Muon + full LR schedule
# (hold 2e-4 for 25ep -> linear decay to 2e-5 by 50ep -> hold to 80ep) +
# global batch 1024 (16 GPU x 64) + flip augmentation (precomputed flipped latents).
# 80ep @ 1024 = 100k steps. Steps/epoch = 1250.
module load gcc ompi
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE); export MASTER_PORT=29516
export DETECTED_IFNAME=$(ip -o -4 addr show | awk '!/^[0-9]+: lo|docker|169\.254/ {print $2}' | head -n 1)
if [ -z "$DETECTED_IFNAME" ]; then export NCCL_SOCKET_IFNAME=^lo,docker0,usb0,virbr0; export GLOO_SOCKET_IFNAME=^lo,docker0,usb0,virbr0;
else export NCCL_SOCKET_IFNAME=$DETECTED_IFNAME; export GLOO_SOCKET_IFNAME=$DETECTED_IFNAME; fi
export NCCL_IB_HCA=mlx5; export NCCL_IB_DISABLE=0; export NCCL_P2P_DISABLE=0; export NCCL_DEBUG=WARN
__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else . "/home/c30746/miniconda3/etc/profile.d/conda.sh"; fi
unset __conda_setup
conda activate DiT
export HF_HOME=/work/c30636/dataset/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/work/c30636/RAEv2/src:$PYTHONPATH
cd /work/c30636/DiT
LATDIR=/work/c30636/dataset/rae_dinov3l_k7_latents/train
FLIPDIR=/work/c30636/dataset/rae_dinov3l_k7_latents_flip/train
N_FLIP=$(ls "$FLIPDIR"/*.npy 2>/dev/null | wc -l)
echo "RAE-RECIPE-LG: Muon + LRsched(31250/62500, 2e-4->2e-5) + GBS=1024 + flip($N_FLIP cls)"
[ "$N_FLIP" -ge 1000 ] || { echo "ERROR: flipped latents incomplete ($N_FLIP/1000)"; exit 1; }
mpirun -np 4 --map-by ppr:1:node --bind-to none \
  -x HF_HOME -x HF_HUB_OFFLINE -x TRANSFORMERS_OFFLINE -x PYTHONPATH \
  -x MASTER_ADDR -x MASTER_PORT -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME \
  -x NCCL_IB_HCA -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x NCCL_DEBUG -x PATH \
  torchrun --nnodes=4 --nproc_per_node=4 --rdzv_id=$PBS_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  ./train_rae_flow.py --latent-dir "$LATDIR" --latent-flip-dir "$FLIPDIR" --arch dit_rope \
  --optimizer muon --learning-rate 2e-4 --lr-final 2e-5 --lr-hold-steps 31250 --lr-decay-end-steps 62500 \
  --global-batch-size 1024 --ckpt-every 10000 --max-steps 100000
