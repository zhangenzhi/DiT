#!/bin/bash
# Watch exp 011 (DiT_RoPE) checkpoints; submit an FID-10k eval (ARCH=dit_rope) for each
# new multiple-of-50k checkpoint. Results -> outputs/fid_rae_rope.txt. Exits after job 552884 ends.
CKDIR=/work/c30636/DiT/results/011-DiT-XL-2-rae-flow/checkpoints
SUB=/work/c30636/DiT/.rope011_submitted
OUT=/work/c30636/DiT/outputs/fid_rae_rope.txt
touch "$SUB"
while true; do
  for pt in "$CKDIR"/*.pt; do
    [ -e "$pt" ] || continue
    s=$(basename "$pt" .pt); n=$((10#$s))
    [ $((n % 50000)) -eq 0 ] || continue
    grep -qx "$s" "$SUB" && continue
    cd /work/c30636/DiT
    jid=$(qsub -v "CKPT=$pt,ARCH=dit_rope,NSAMP=10000,CFG=1.8,OUTFILE=$OUT" scripts/fid-rae-flow.sh 2>&1)
    echo "$(date +%H:%M) submitted eval $s -> $jid"
    echo "$s" >> "$SUB"
  done
  qstat 552884 >/dev/null 2>&1 || { echo "training 552884 ended; final sweep then exit"; sleep 30;
    for pt in "$CKDIR"/*.pt; do s=$(basename "$pt" .pt); n=$((10#$s));
      [ $((n % 50000)) -eq 0 ] || continue; grep -qx "$s" "$SUB" && continue;
      cd /work/c30636/DiT; qsub -v "CKPT=$pt,ARCH=dit_rope,NSAMP=10000,CFG=1.8,OUTFILE=$OUT" scripts/fid-rae-flow.sh; echo "$s" >> "$SUB"; done
    break; }
  sleep 300
done
echo "ROPE(011) AUTO-EVAL DONE"
