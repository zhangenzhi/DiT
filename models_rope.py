"""
DiT_RoPE: our plain DiT backbone with the three RAEv2-style primitives swapped in —
  LayerNorm        -> RMSNorm
  GELU MLP         -> SwiGLU FFN  (hidden = int(2/3 * mlp_ratio * D) to match param count)
  sin-cos pos_embed-> 2D RoPE (applied to q,k inside attention)
Everything else (adaLN-Zero conditioning, TimestepEmbedder/LabelEmbedder/PatchEmbed,
velocity head, unpatchify) is kept identical to models.DiT so this isolates the
norm/ffn/posenc change. RMSNorm/SwiGLU/RoPE/NormAttention mirror RAEv2's model_utils.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TimestepEmbedder, LabelEmbedder, PatchEmbed, modulate
from models_repa import build_repa_mlp


def rotate_half(x):
    # [..., (d r)] with r=2 -> interleave (-x_odd, x_even); no einops dependency
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RoPE(nn.Module):
    """2D rotary position embedding for a square grid of `vis_len` patches."""
    def __init__(self, dim, vis_len, cond_len=0, theta=10000.):
        super().__init__()
        d, T = dim // 2, int(vis_len ** 0.5)
        assert T * T == vis_len, "RoPE expects a square patch grid"
        vis_freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))   # [d//2]
        base = torch.outer(torch.arange(T).float(), vis_freqs)            # [T, d//2]
        vis_angles = torch.cat([
            base[:, None].expand(-1, T, -1),
            base[None, :].expand(T, -1, -1),
        ], dim=-1).reshape(vis_len, d)                                    # [vis_len, d]
        cond_angles = torch.zeros(cond_len, d)
        angles = torch.cat([vis_angles, cond_angles], dim=0).repeat_interleave(2, dim=-1)  # [L, dim]
        self.register_buffer("freqs_cos", angles.cos())
        self.register_buffer("freqs_sin", angles.sin())

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return n.type_as(x) * self.weight


class NormAttention(nn.Module):
    """MHSA with QK-RMSNorm and RoPE applied to q,k (RAEv2 style)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads, self.dim, self.head_dim = num_heads, dim, dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope):
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        return self.proj(out)


class RoPEBlock(nn.Module):
    """DiT block with adaLN-Zero, but RMSNorm + SwiGLU + RoPE-attention."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, rope):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT_RoPE(nn.Module):
    def __init__(self, input_size=16, patch_size=1, in_channels=1024, hidden_size=1152,
                 depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1,
                 num_classes=1000, learn_sigma=False):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.rope = RoPE(hidden_size // num_heads, num_patches)   # buffers, no params

        self.blocks = nn.ModuleList([
            RoPEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y):
        x = self.x_embedder(x)                   # (N, T, D), NO additive pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c, self.rope)
        x = self.final_layer(x, c)
        return self.unpatchify(x)

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps = model_out
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        w = cfg_scale
        if cfg_interval is not None:
            lo, hi = cfg_interval
            t0 = t[0].item() if torch.is_tensor(t) else float(t)
            if not (lo <= t0 <= hi):
                w = 1.0
        half_eps = uncond_eps + w * (cond_eps - uncond_eps)
        return torch.cat([half_eps, half_eps], dim=0)


def _modulate_pt(x, shift, scale):
    """Per-token adaLN modulate: shift/scale are [N, T, H] (not broadcast)."""
    return x * (1 + scale) + shift


class DDTEncBlock(nn.Module):
    """Plain pre-norm transformer block (RMSNorm + RoPE attn + SwiGLU), NO adaLN.
    Conditioning enters the encoder via concatenated t/class tokens, not modulation."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * hidden_size * mlp_ratio))

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.mlp(self.norm2(x))
        return x


class DDTDecBlock(nn.Module):
    """Wide decoder block with PER-TOKEN adaLN conditioning from the encoder output."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * hidden_size * mlp_ratio))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c, rope):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.adaLN(c).chunk(6, dim=-1)
        x = x + g_msa * self.attn(_modulate_pt(self.norm1(x), s_msa, sc_msa), rope)
        x = x + g_mlp * self.mlp(_modulate_pt(self.norm2(x), s_mlp, sc_mlp))
        return x


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(_modulate_pt(self.norm(x), shift, scale))


class DiT_RoPE_DDT(nn.Module):
    """Faithful two-stream DDT (RAEv2-style) on our RoPE primitives.
      - narrow ENCODER (enc_hidden) = plain ViT over [patch tokens + t-token + class-token],
        no adaLN; builds a per-token semantic representation.
      - bridge: encoder patch output (+ base t-embed) -> Linear -> dec_hidden = per-token condition.
      - wide, shallow DECODER (dec_hidden) re-embeds the noisy latent and denoises with
        PER-TOKEN adaLN from the encoder output. This is the decoupling, not naive scaling."""
    def __init__(self, input_size=16, patch_size=1, in_channels=1024,
                 enc_hidden=1152, dec_hidden=2048, enc_depth=28, dec_depth=2,
                 enc_heads=16, dec_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1,
                 num_classes=1000, learn_sigma=False, num_t_tokens=4, num_c_tokens=8):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_t_tokens = num_t_tokens
        self.num_c_tokens = num_c_tokens

        self.s_embedder = PatchEmbed(input_size, patch_size, in_channels, enc_hidden, bias=True)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dec_hidden, bias=True)
        self.t_embedder = TimestepEmbedder(enc_hidden)
        self.y_embedder = LabelEmbedder(num_classes, enc_hidden, class_dropout_prob)
        # multiple learnable cond-token slots (RAEv2: 4 t-tokens + 8 c-tokens)
        self.t_token_emb = nn.Parameter(torch.zeros(num_t_tokens, enc_hidden))
        self.c_token_emb = nn.Parameter(torch.zeros(num_c_tokens, enc_hidden))
        self.s_projector = nn.Linear(enc_hidden, dec_hidden)
        num_patches = self.s_embedder.num_patches
        self.num_patches = num_patches
        self.num_cond_tokens = num_t_tokens + num_c_tokens
        self.enc_rope = RoPE(enc_hidden // enc_heads, num_patches, cond_len=self.num_cond_tokens)
        self.dec_rope = RoPE(dec_hidden // dec_heads, num_patches)

        self.enc_blocks = nn.ModuleList([
            DDTEncBlock(enc_hidden, enc_heads, mlp_ratio) for _ in range(enc_depth)])
        self.dec_blocks = nn.ModuleList([
            DDTDecBlock(dec_hidden, dec_heads, mlp_ratio) for _ in range(dec_depth)])
        self.final_layer = DDTFinalLayer(dec_hidden, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)
        for emb in (self.s_embedder, self.x_embedder):
            w = emb.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(emb.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_token_emb, std=0.02)
        nn.init.normal_(self.c_token_emb, std=0.02)
        for block in self.dec_blocks:                 # zero-init decoder adaLN -> identity start
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y):
        t_vec = self.t_embedder(t)                          # (N, enc_hidden)
        y_vec = self.y_embedder(y, self.training)           # (N, enc_hidden)
        t_tok = t_vec.unsqueeze(1) + self.t_token_emb.unsqueeze(0)   # (N, num_t_tokens, H)
        c_tok = y_vec.unsqueeze(1) + self.c_token_emb.unsqueeze(0)   # (N, num_c_tokens, H)
        seq = torch.cat([self.s_embedder(x), t_tok, c_tok], dim=1)   # (N, 256 + 12, H)
        for block in self.enc_blocks:
            seq = block(seq, self.enc_rope)
        patches = seq[:, :self.num_patches, :]              # drop cond tokens
        cond = self.s_projector(F.silu(t_vec.unsqueeze(1) + patches))     # (N, 256, dec_hidden)

        h = self.x_embedder(x)                              # (N, 256, dec_hidden) re-embed
        for block in self.dec_blocks:
            h = block(h, cond, self.dec_rope)
        h = self.final_layer(h, cond)
        return self.unpatchify(h)

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, t, y)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        w = cfg_scale
        if cfg_interval is not None:
            lo, hi = cfg_interval
            t0 = t[0].item() if torch.is_tensor(t) else float(t)
            if not (lo <= t0 <= hi):
                w = 1.0
        half_eps = uncond_eps + w * (cond_eps - uncond_eps)
        return torch.cat([half_eps, half_eps], dim=0)


class DiT_RoPE_WideHead(nn.Module):
    """Fair "wide-head" test: a 1152-wide RoPE encoder (enc_depth blocks) followed by a
    narrow stack of WIDE (2048) decoder blocks + a 2048 final layer. Keeps the encoder at
    1152 (so the backbone matches DiT_RoPE) and only widens the last `dec_depth` blocks --
    isolating the value of a wide output head vs RAEv2's full [1440,2048] config.
    Serial (single-stream): tokens flow enc(1152) -> proj -> dec(2048) -> final."""
    def __init__(self, input_size=16, patch_size=1, in_channels=1024,
                 enc_hidden=1152, dec_hidden=2048, enc_depth=28, dec_depth=2,
                 enc_heads=16, dec_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1,
                 num_classes=1000, learn_sigma=False):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, enc_hidden, bias=True)
        self.t_embedder = TimestepEmbedder(enc_hidden)
        self.y_embedder = LabelEmbedder(num_classes, enc_hidden, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.enc_rope = RoPE(enc_hidden // enc_heads, num_patches)
        self.dec_rope = RoPE(dec_hidden // dec_heads, num_patches)

        self.enc_blocks = nn.ModuleList([
            RoPEBlock(enc_hidden, enc_heads, mlp_ratio=mlp_ratio) for _ in range(enc_depth)])
        self.head_proj = nn.Linear(enc_hidden, dec_hidden)          # lift tokens to head width
        self.c_proj = nn.Linear(enc_hidden, dec_hidden)             # lift adaLN condition
        self.dec_blocks = nn.ModuleList([
            RoPEBlock(dec_hidden, dec_heads, mlp_ratio=mlp_ratio) for _ in range(dec_depth)])
        self.final_layer = FinalLayer(dec_hidden, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in list(self.enc_blocks) + list(self.dec_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y):
        x = self.x_embedder(x)
        c = self.t_embedder(t) + self.y_embedder(y, self.training)   # (N, enc_hidden)
        for block in self.enc_blocks:
            x = block(x, c, self.enc_rope)
        x = self.head_proj(x)                                        # (N, T, dec_hidden)
        c2 = self.c_proj(c)                                          # (N, dec_hidden)
        for block in self.dec_blocks:
            x = block(x, c2, self.dec_rope)
        x = self.final_layer(x, c2)
        return self.unpatchify(x)

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, t, y)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        w = cfg_scale
        if cfg_interval is not None:
            lo, hi = cfg_interval
            t0 = t[0].item() if torch.is_tensor(t) else float(t)
            if not (lo <= t0 <= hi):
                w = 1.0
        half_eps = uncond_eps + w * (cond_eps - uncond_eps)
        return torch.cat([half_eps, half_eps], dim=0)


class DiT_RoPE_REPA(DiT_RoPE):
    """DiT_RoPE + REPA alignment head. Identical to DiT_RoPE plus a 3-layer MLP
    projector on the hidden state after `align_depth` blocks. Training-mode forward
    returns (out, zs); eval-mode returns out only (sampling/EMA unaffected).
    Mirrors models_repa.DiT_REPA but on the RMSNorm+SwiGLU+RoPE backbone."""
    def __init__(self, *args, z_dim=1024, proj_dim=2048, align_depth=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_dim = z_dim
        self.align_depth = align_depth
        assert 1 <= align_depth <= len(self.blocks)
        hidden_size = self.x_embedder.proj.out_channels
        self.repa_projector = build_repa_mlp(hidden_size, proj_dim, z_dim)
        for m in self.repa_projector:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t, y):
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        zs = None
        for i, block in enumerate(self.blocks):
            x = block(x, c, self.rope)
            if self.training and (i + 1) == self.align_depth:
                zs = self.repa_projector(x)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.training:
            return x, zs
        return x
