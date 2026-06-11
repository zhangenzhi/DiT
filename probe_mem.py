import torch
from models_rope import DiT_RoPE_DDT
dev=torch.device("cuda")
p=torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}  total={p.total_memory/1e9:.1f} GB")
m=DiT_RoPE_DDT(input_size=16,patch_size=1,in_channels=1024,enc_hidden=1152,dec_hidden=2048,
               enc_depth=28,dec_depth=2,enc_heads=16,dec_heads=16,num_classes=1000,learn_sigma=False).to(dev)
opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
for bs in [32,64,96,128,160,192,256]:
    try:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        x=torch.randn(bs,1024,16,16,device=dev); t=torch.rand(bs,device=dev); y=torch.randint(0,1000,(bs,),device=dev)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            v=m(x,t,y); loss=((v-torch.randn_like(v))**2).mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        peak=torch.cuda.max_memory_allocated()/1e9
        print(f"  per-GPU bs={bs:4d}  peak={peak:5.1f} GB  ({peak/(p.total_memory/1e9)*100:.0f}% of {p.total_memory/1e9:.0f}GB)")
    except RuntimeError as e:
        print(f"  per-GPU bs={bs:4d}  OOM"); break
