"""P2P copy stress for one GPU pair. Run under `timeout`; a hang = exit 124.
Usage: python pair_copy_stress.py <src> <dst> <seconds>
"""
import sys, time
import torch

src, dst, secs = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
a = torch.randn(256 * 1024 * 1024 // 4, device=f"cuda:{src}")  # 256MB
b = torch.empty_like(a, device=f"cuda:{dst}")
p2p = torch.cuda.can_device_access_peer(src, dst)
t0 = time.time(); n = 0; nbytes = a.numel() * 4
while time.time() - t0 < secs:
    b.copy_(a, non_blocking=True)
    a.copy_(b, non_blocking=True)
    n += 2
    if n % 200 == 0:
        torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
gbs = n * nbytes / (time.time() - t0) / 1e9
print(f"pair {src}->{dst}: p2p={p2p} {n} copies, {gbs:.0f} GB/s aggregate", flush=True)
