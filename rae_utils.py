"""
Helpers for the external RAEv2 repo (stage1 RAE encoder/decoder).

The repo location is NOT hardcoded. Set it one of three ways (first hit wins):
  1. --rae-root CLI flag
  2. rae-root key in the YAML passed via --config
  3. RAE_ROOT environment variable
"""
import os
import sys

ENC_NAME = "dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]"
LATENT = (1024, 16, 16)


def add_rae_root_arg(parser):
    parser.add_argument("--rae-root", type=str, default=os.environ.get("RAE_ROOT"),
                        help="Path to the RAEv2 repo checkout (default: $RAE_ROOT)")


def rae_paths(rae_root):
    if not rae_root:
        raise SystemExit("RAEv2 location not set: pass --rae-root, put rae-root in the "
                         "YAML config, or export RAE_ROOT")
    if not os.path.isdir(rae_root):
        raise SystemExit(f"--rae-root does not exist: {rae_root}")
    src = os.path.join(rae_root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    stage1 = os.path.join(rae_root, "pretrained_models/stage1/imagenet/dinov3l-k7")
    return {
        "dec_cfg": os.path.join(rae_root, "configs/decoder/ViTXL"),
        "dec_pt": os.path.join(stage1, "decoder.pt"),
        "stats": os.path.join(stage1, "stats.pt"),
    }


def load_rae(rae_root, device, noise_tau=0.0, resolution=256, decoder=True):
    p = rae_paths(rae_root)
    from stage1 import RAE  # imported late: needs rae_root on sys.path
    return RAE(encoder_name=ENC_NAME, resolution=resolution,
               decoder_config_path=p["dec_cfg"], decoder_patch_size=16,
               pretrained_decoder_path=p["dec_pt"] if decoder else None,
               noise_tau=noise_tau,
               normalization_stat_path=p["stats"]).to(device)
