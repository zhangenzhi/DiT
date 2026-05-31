# REPA-augmented DiT.
# Adds an auxiliary "representation alignment" projection head on top of the
# vanilla DiT (see models.py). During training the model exposes the hidden
# state of an intermediate transformer block, projected to the dimensionality
# of a frozen vision encoder (DINOv2). The training loop aligns this projection
# with the encoder's patch features.
#
# Reference: "Representation Alignment for Generation: Training Diffusion
# Transformers Is Easier Than You Think" (Yu et al., ICLR 2025).

import torch
import torch.nn as nn

from models import DiT


def build_repa_mlp(in_dim, hidden_dim, out_dim):
    """3-layer MLP projector with SiLU, matching the REPA paper."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class DiT_REPA(DiT):
    """
    DiT with a representation-alignment projection head.

    forward() returns:
      - (out, zs) when self.training is True, where `zs` is the projected
        hidden state after block `align_depth` (shape [N, T, z_dim]).
      - out only when in eval mode, so sampling / forward_with_cfg / EMA are
        unaffected and existing inference code keeps working unchanged.
    """
    def __init__(self, *args, z_dim=768, proj_dim=2048, align_depth=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_dim = z_dim
        self.align_depth = align_depth  # align after this many blocks (1-indexed)
        assert 1 <= align_depth <= len(self.blocks), \
            f"align_depth {align_depth} out of range for depth {len(self.blocks)}"
        hidden_size = self.x_embedder.proj.out_channels  # == DiT hidden_size
        self.repa_projector = build_repa_mlp(hidden_size, proj_dim, z_dim)
        # Initialize the projector (DiT.initialize_weights already ran in super
        # before this head existed, so do it here).
        for m in self.repa_projector:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t, y):
        """
        x: (N, C, H, W) latents
        t: (N,) timesteps
        y: (N,) class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)                    # (N, D)
        y = self.y_embedder(y, self.training)     # (N, D)
        c = t + y                                 # (N, D)
        zs = None
        for i, block in enumerate(self.blocks):
            x = block(x, c)                       # (N, T, D)
            if self.training and (i + 1) == self.align_depth:
                zs = self.repa_projector(x)       # (N, T, z_dim)
        x = self.final_layer(x, c)                # (N, T, patch ** 2 * out_channels)
        x = self.unpatchify(x)                    # (N, out_channels, H, W)
        if self.training:
            return x, zs
        return x


#################################################################################
#                                REPA DiT Configs                               #
#################################################################################

def DiT_REPA_XL_2(**kwargs):
    return DiT_REPA(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_REPA_L_2(**kwargs):
    return DiT_REPA(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_REPA_B_2(**kwargs):
    return DiT_REPA(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_REPA_S_2(**kwargs):
    return DiT_REPA(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


DiT_REPA_models = {
    'DiT-XL/2': DiT_REPA_XL_2,
    'DiT-L/2':  DiT_REPA_L_2,
    'DiT-B/2':  DiT_REPA_B_2,
    'DiT-S/2':  DiT_REPA_S_2,
}
