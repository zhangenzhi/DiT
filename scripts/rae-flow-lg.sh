#!/bin/sh
#PBS -N rae_flow_lg
#PBS -q lg
#PBS -l select=2:ngpus=4:mpiprocs=4
#PBS -l walltime=72:00:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
# Path A on lg 8 GPU: our DiT in RAE (DINOv3-K7) latent + flow velocity + shift,
# trained on PRECOMPUTED latents (--latent-dir). vars: MAXSTEPS(0=full), GBS(256)
module load gcc ompi
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE); export MASTER_PORT=29507
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
: "${MAXSTEPS:=0}"; : "${GBS:=256}"; : "${REPA_LAMBDA:=0}"
LATDIR=/work/c30636/dataset/rae_dinov3l_k7_latents/train
echo "RAE-FLOW-LG (Path A, offline): MAXSTEPS=$MAXSTEPS GBS=$GBS REPA_LAMBDA=$REPA_LAMBDA latent=$LATDIR"
mpirun -np 2 --map-by ppr:1:node --bind-to none \
  -x HF_HOME -x HF_HUB_OFFLINE -x TRANSFORMERS_OFFLINE -x PYTHONPATH \
  -x MASTER_ADDR -x MASTER_PORT -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME \
  -x NCCL_IB_HCA -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x NCCL_DEBUG -x PATH \
  torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=$PBS_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  ./train_rae_flow.py --latent-dir "$LATDIR" \
  --global-batch-size "$GBS" --learning-rate 1e-4 --ckpt-every 10000 --max-steps "$MAXSTEPS" \
  --repa-lambda "$REPA_LAMBDA"
