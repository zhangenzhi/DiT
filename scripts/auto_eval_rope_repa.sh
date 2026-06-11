#!/bin/bash
# Auto-eval REPA-on-RoPE (exp 009, DiT_RoPE_REPA). FID-10k per 50k ckpt -> fid_rae_rope_repa.txt
CKDIR=/work/c30636/DiT/results/009-DiT-XL-2-rae-flow/checkpoints
SUB=/work/c30636/DiT/.rope009repa_submitted
OUT=/work/c30636/DiT/outputs/fid_rae_rope_repa.txt
JOB=553746
touch "$SUB"
while true; do
  for pt in "$CKDIR"/*.pt; do
    [ -e "$pt" ] || continue
    s=$(basename "$pt" .pt); n=$((10#$s))
    [ $((n % 50000)) -eq 0 ] || continue
    grep -qx "$s" "$SUB" && continue
    cd /work/c30636/DiT
    jid=$(qsub -v "CKPT=$pt,ARCH=dit_rope,NSAMP=10000,CFG=1.8,OUTFILE=$OUT" scripts/fid-rae-flow.sh 2>&1)
    echo "$(date +%H:%M) submitted eval $s -> $jid"; echo "$s" >> "$SUB"
  done
  qstat $JOB >/dev/null 2>&1 || { echo "training $JOB ended; final sweep"; sleep 30;
    for pt in "$CKDIR"/*.pt; do s=$(basename "$pt" .pt); n=$((10#$s));
      [ $((n % 50000)) -eq 0 ] || continue; grep -qx "$s" "$SUB" && continue;
      cd /work/c30636/DiT; qsub -v "CKPT=$pt,ARCH=dit_rope,NSAMP=10000,CFG=1.8,OUTFILE=$OUT" scripts/fid-rae-flow.sh; echo "$s" >> "$SUB"; done
    break; }
  sleep 300
done
echo "REPA-RoPE(009) AUTO-EVAL DONE"
