"""NCCL allreduce stress test to reproduce the ar17n12 hang.
Mimics training traffic: continuous allreduce of the same bucket size DDP used
(8,414,208 fp32 elements) plus a large 677M-param-sized burst every 50 iters.
Prints heartbeat every 500 iters; a hang shows as heartbeat stopping while
GPUs sit at 100% (then the 5-min NCCL watchdog kills us with a traceback).

Run: torchrun --standalone --nproc_per_node=4 nccl_stress.py --minutes 15
"""
import os, time, argparse
import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--minutes", type=float, default=15)
args = p.parse_args()

os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
dist.init_process_group("nccl", timeout=__import__("datetime").timedelta(seconds=300))
rank = dist.get_rank()
dev = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(dev)

bucket = torch.randn(8_414_208, device="cuda")          # DDP bucket from the crash
big = torch.randn(677_000_000 // 4, device="cuda")      # ~677M params / 4 buckets
t0 = time.time(); i = 0
while time.time() - t0 < args.minutes * 60:
    dist.all_reduce(bucket)
    if i % 50 == 0:
        dist.all_reduce(big)
    if i % 500 == 0:
        torch.cuda.synchronize()
        if rank == 0:
            print(f"iter {i} ok, {time.time()-t0:.0f}s elapsed", flush=True)
    i += 1
torch.cuda.synchronize()
if rank == 0:
    print(f"PASSED: {i} allreduces in {time.time()-t0:.0f}s with no hang", flush=True)
dist.destroy_process_group()
