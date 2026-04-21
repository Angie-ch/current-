"""
统一DiT (Diffusion Transformer) 骨干网络 — FM和DM共用

核心架构:
  - SinusoidalTimeEmbedding: 正弦时间步嵌入
  - AdaLayerNorm: 自适应层归一化 (时间步条件注入)
  - PatchEmbed: 2D Patch 嵌入层
  - Unpatchify: 逆Patch操作
  - DiTBlock: DiT核心块 (AdaLN + Self-Attn + Cross-Attn + FFN)
  - UnifiedConditionEncoder: 统一条件编码器 (支持FM的Conv3D时间聚合)
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 时间嵌入
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码 + MLP投影"""

    def __init__(self, dim: int = 256, time_emb_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) int/float 时间步
        return: (B, time_emb_dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


# ============================================================
# 自适应层归一化 (AdaLN)
# ============================================================

class AdaLayerNorm(nn.Module):
    """
    自适应层归一化 — 通过时间步嵌入生成scale/shift/gate参数

    h = gate * (scale * LayerNorm(x) + shift)
    """

    def __init__(self, d_model: int, time_emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 6 * d_model),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        """
        x: (B, N, D)
        time_emb: (B, time_emb_dim)
        返回: (γ₁, β₁, α₁, γ₂, β₂, α₂) 每个 shape (B, 1, D)
        """
        params = self.adaLN_modulation(time_emb).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)
        return gamma1, beta1, alpha1, gamma2, beta2, alpha2

    def modulate(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """应用AdaLN: scale * LN(x) + shift"""
        return (1 + gamma) * self.norm(x) + beta


# ============================================================
# Patch嵌入 / 反Patch
# ============================================================

class PatchEmbed(nn.Module):
    """将2D图像切成patch并线性嵌入"""

    def __init__(self, in_channels: int, d_model: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, D, Hp, Wp = x.shape
        x = x.reshape(B, D, Hp * Wp).permute(0, 2, 1)
        return x


class Unpatchify(nn.Module):
    """将token序列还原为2D图像"""

    def __init__(self, out_channels: int, patch_size: int = 4, grid_size: int = 40):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.grid_h = grid_size // patch_size
        self.grid_w = grid_size // patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        p = self.patch_size
        C = self.out_channels
        x = x.reshape(B, self.grid_h, self.grid_w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, self.grid_h * p, self.grid_w * p)
        return x


# ============================================================
# DiT Block
# ============================================================

class DiTBlock(nn.Module):
    """
    DiT核心块:
      a) AdaLN → Self-Attention → 残差连接
      b) Cross-Attention (Q=噪声token, K/V=条件token) → 残差连接
      c) AdaLN → FFN → 残差连接

    时间步t通过AdaLN注入
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        time_emb_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.adaln = AdaLayerNorm(d_model, time_emb_dim)

        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_proj = nn.Linear(d_model, d_model)

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_kv = nn.Linear(d_model, 2 * d_model)
        self.cross_attn_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(x, time_emb)

        # (a) Self-Attention
        h = self.adaln.modulate(x, gamma1, beta1)
        qkv = self.self_attn_qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        attn_out = self.self_attn_proj(attn_out)
        x = x + alpha1 * self.dropout(attn_out)

        # (b) Cross-Attention
        h_cross = self.cross_attn_norm(x)
        q_cross = h_cross.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        N_cond = cond_tokens.shape[1]
        kv_cross = self.cross_attn_kv(cond_tokens).reshape(B, N_cond, 2, self.n_heads, self.head_dim)
        kv_cross = kv_cross.permute(2, 0, 3, 1, 4)
        k_cross, v_cross = kv_cross[0], kv_cross[1]
        cross_out = F.scaled_dot_product_attention(q_cross, k_cross, v_cross)
        cross_out = cross_out.permute(0, 2, 1, 3).reshape(B, N, D)
        cross_out = self.cross_attn_proj(cross_out)
        x = x + self.dropout(cross_out)

        # (c) FFN
        h_ffn = self.adaln.modulate(x, gamma2, beta2)
        ffn_out = self.ffn(h_ffn)
        x = x + alpha2 * ffn_out

        return x


# ============================================================
# 统一条件编码器 (兼容FM和DM)
# ============================================================

class UnifiedConditionEncoder(nn.Module):
    """
    统一条件编码器 — FM和DM共用此编码器

    支持:
    - DM模式: 标准2D卷积 → Patch → Self-Attn
    - FM模式: 额外支持3D时间卷积聚合 (Conv3D → AdaptiveAvgPool → 残差)
    - 兼容模式: 使用简单编码器 (SimpleConditionEncoder) 加载外部 checkpoint
    """

    def __init__(
        self,
        cond_channels: int = 144,
        d_model: int = 384,
        n_heads: int = 6,
        n_cond_layers: int = 3,
        ff_mult: int = 4,
        patch_size: int = 4,
        grid_size: int = 40,
        dropout: float = 0.1,
        use_grouped_conv: bool = True,
        num_var_groups: int = 3,
        history_steps: int = 16,
        use_temporal_agg: bool = True,
        use_simple_encoder: bool = False,  # 新增: 使用简单编码器
    ):
        super().__init__()
        self.cond_channels = cond_channels  # save for forward pass
        self.d_model = d_model
        self.use_grouped_conv = use_grouped_conv
        self.num_var_groups = num_var_groups
        self.history_steps = history_steps
        self.use_temporal_agg = use_temporal_agg
        self.use_simple_encoder = use_simple_encoder
        self.C_per_step = cond_channels // history_steps  # channels per timestep (e.g. 45//5=9)
        num_patches = (grid_size // patch_size) ** 2

        # 如果启用简单编码器，直接使用 SimpleConditionEncoder
        if use_simple_encoder:
            from .adapter import SimpleConditionEncoder
            self.simple_encoder = SimpleConditionEncoder(
                cond_channels=cond_channels,
                d_model=d_model,
                n_heads=n_heads,
                n_cond_layers=n_cond_layers,
                ff_mult=ff_mult,
                patch_size=patch_size,
                grid_size=grid_size,
                dropout=dropout,
            )
            # 将 simple_encoder 的参数直接作为本模块参数
            self.temporal_conv3d = None
            self.wind_conv = None
            self.z_conv = None
            self.fusion_conv = None
            self.local_conv = None
            self.patch_embed = self.simple_encoder.patch_embed
            self.pos_embed = self.simple_encoder.pos_embed
            self.layers = self.simple_encoder.layers
            self.norm_out = self.simple_encoder.norm_out
        else:
            self.simple_encoder = None

        # 分组卷积 (u/v/z 分组处理)
        if self.use_grouped_conv and cond_channels >= num_var_groups and not use_simple_encoder:
            group_dim = d_model // num_var_groups
            self.wind_conv = nn.Sequential(
                nn.Conv2d(6, 6, 3, padding=1, groups=6),
                nn.Conv2d(6, group_dim, 1),
                nn.GroupNorm(1, group_dim),
                nn.SiLU(),
                nn.Conv2d(group_dim, group_dim, 3, padding=1),
                nn.GroupNorm(min(32, group_dim), group_dim),
                nn.SiLU(),
            )
            self.z_conv = nn.Sequential(
                nn.Conv2d(3, 3, 3, padding=1, groups=3),
                nn.Conv2d(3, group_dim, 1),
                nn.GroupNorm(1, group_dim),
                nn.SiLU(),
                nn.Conv2d(group_dim, group_dim, 3, padding=1),
                nn.GroupNorm(min(32, group_dim), group_dim),
                nn.SiLU(),
            )
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(2 * group_dim, d_model, 1),
                nn.SiLU(),
            )
            self.local_conv = None

            # Temporal aggregation for grouped conv: use Conv2d (per-timestep) + average
            if self.use_temporal_agg:
                C_per_step = cond_channels // history_steps
                self.temporal_conv2d = nn.Sequential(
                    nn.Conv2d(C_per_step, d_model, kernel_size=3, padding=1),
                    nn.GroupNorm(min(32, d_model), d_model),
                    nn.SiLU(),
                    nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                    nn.GroupNorm(min(32, d_model), d_model),
                    nn.SiLU(),
                )
                self.temporal_conv3d = None
            else:
                self.temporal_conv2d = None
                self.temporal_conv3d = None
        else:
            # 标准2D卷积
            self.local_conv = nn.Sequential(
                nn.Conv2d(cond_channels, d_model, 3, padding=1),
                nn.GroupNorm(min(32, d_model), d_model),
                nn.SiLU(),
                nn.Conv2d(d_model, d_model, 3, padding=1),
                nn.GroupNorm(min(32, d_model), d_model),
                nn.SiLU(),
            )
            self.wind_conv = None
            self.z_conv = None
            self.fusion_conv = None

        # Temporal aggregation: Conv3D over (B, C, T, H, W) → (B, C, H, W)
        # Conv3D: per-timestep processing with kernel=(1,3,3) — spatial only.
        # After Conv3D + mean(dim=2): (B, C_per_step=9, H, W).
        # We then expand to d_model via a 1×1 conv, so local_conv is NOT needed.
        if self.use_temporal_agg:
            Cp = self.C_per_step
            self.temporal_conv3d = nn.Sequential(
                nn.Conv3d(Cp, Cp, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.GroupNorm(Cp, Cp),
                nn.SiLU(),
                nn.Conv3d(Cp, Cp, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.GroupNorm(Cp, Cp),
                nn.SiLU(),
                nn.Conv3d(Cp, Cp, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.GroupNorm(Cp, Cp),
                nn.SiLU(),
            )
            # Expand channels from C_per_step → d_model
            self.temporal_expand = nn.Sequential(
                nn.Conv2d(Cp, d_model, kernel_size=1),
                nn.GroupNorm(min(32, d_model), d_model),
                nn.SiLU(),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                nn.GroupNorm(min(32, d_model), d_model),
                nn.SiLU(),
            )
            self.temporal_conv2d = None
            self.local_conv = None  # handled by temporal_expand
        else:
            self.temporal_conv2d = None
            self.temporal_conv3d = None
            self.temporal_expand = None

        self.patch_embed = PatchEmbed(d_model, d_model, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_cond_layers):
            self.layers.append(CondSelfAttnBlock(d_model, n_heads, ff_mult, dropout))
        self.norm_out = nn.LayerNorm(d_model)

    def _group_forward(self, condition: torch.Tensor) -> torch.Tensor:
        B, C, H, W = condition.shape
        n_levels = C // self.num_var_groups
        group_dim = self.d_model // self.num_var_groups
        u = condition[:, :n_levels]
        v = condition[:, n_levels:2*n_levels]
        z = condition[:, 2*n_levels:]
        wind = torch.cat([u, v], dim=1)
        wind = self.wind_conv(wind)
        z = self.z_conv(z)
        h = torch.cat([wind, z], dim=1)
        h = self.fusion_conv(h)
        return h

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        condition: (B, T*C, H, W) — 时间维已展平的条件输入（标准格式）
                   或 (B, T, C, H, W) — 原始5D格式

        返回: (B, num_patches, D) 条件token序列
        """
        # 简单编码器模式
        if self.use_simple_encoder and self.simple_encoder is not None:
            return self.simple_encoder(condition)

        input_is_5d = condition.ndim == 5
        B = condition.shape[0]
        H, W = condition.shape[-2:]

        if input_is_5d:
            B5, T, C, H5, W5 = condition.shape
            condition = condition.reshape(B5, T, C, H5, W5)  # keep 5D

        B, TC, H, W = condition.shape
        T = self.history_steps
        C = self.cond_channels // T

        # Temporal aggregation via Conv3D over time.
        # Input: (B, TC, H, W) = (B, 45, 40, 40)
        # Reshape: (B, T, C, H, W) → Conv3D expects (B, C, T, H, W) → permute.
        if self.use_temporal_agg and self.temporal_conv3d is not None:
            cond_5d = condition.reshape(B, T, C, H, W)       # (B, T, C, H, W)
            cond_5d = cond_5d.permute(0, 2, 1, 3, 4)         # (B, C, T, H, W) — Conv3D format
            h_t = self.temporal_conv3d(cond_5d)              # (B, C, T, H, W)
            h = h_t.mean(dim=2)                               # (B, C, H, W) — temporal mean
            h = self.temporal_expand(h)                       # (B, d_model, H, W)
        elif self.use_temporal_agg and self.temporal_conv2d is not None:
            cond_5d = condition.reshape(B, T, C, H, W)
            cond_BT = cond_5d.reshape(B * T, C, H, W)
            h_t = self.temporal_conv2d(cond_BT)
            h_t = h_t.reshape(B, T, C, H, W)
            h = h_t.mean(dim=1)                   # (B, C, H, W)
            h = self.temporal_expand(h)             # (B, d_model, H, W)
        else:
            if self.use_grouped_conv and self.wind_conv is not None:
                h = self._group_forward(condition)
            else:
                h = self.local_conv(condition)

        tokens = self.patch_embed(h)
        tokens = tokens + self.pos_embed
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm_out(tokens)


class CondSelfAttnBlock(nn.Module):
    """条件编码器中的Self-Attention块"""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + self.dropout(self.proj(attn_out))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 统一DiT骨干网络 (FM和DM共用)
# ============================================================

class UnifiedDiT(nn.Module):
    """
    统一DiT骨干网络 — Flow Matching和Diffusion共用

    输入: x_t (B, in_ch, H, W), t (B,), condition (B, cond_ch, H, W)
    输出: 预测目标 (B, in_ch, H, W)
         - FM: 速度场 v = x_1 - x_0
         - DM: 噪声 ε 或 velocity v

    唯一区别在forward的t和condition的处理方式，骨干网络完全相同
    """

    def __init__(
        self,
        in_channels: int = 9,
        cond_channels: int = 144,
        d_model: int = 384,
        n_heads: int = 6,
        n_dit_layers: int = 12,
        n_cond_layers: int = 3,
        ff_mult: int = 4,
        patch_size: int = 4,
        grid_size: int = 40,
        dropout: float = 0.1,
        time_embedding_scale: float = 1000.0,
        use_grouped_conv: bool = True,
        num_var_groups: int = 3,
        history_steps: int = 16,
        use_temporal_agg: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_size = patch_size
        self.time_embedding_scale = time_embedding_scale
        num_patches = (grid_size // patch_size) ** 2
        time_emb_dim = d_model

        self.time_emb = SinusoidalTimeEmbedding(d_model, time_emb_dim)

        self.cond_encoder = UnifiedConditionEncoder(
            cond_channels=cond_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_cond_layers=n_cond_layers,
            ff_mult=ff_mult,
            patch_size=patch_size,
            grid_size=grid_size,
            dropout=dropout,
            use_grouped_conv=use_grouped_conv,
            num_var_groups=num_var_groups,
            history_steps=history_steps,
            use_temporal_agg=use_temporal_agg,
        )

        self.patch_embed = PatchEmbed(in_channels, d_model, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_mult=ff_mult,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
            )
            for _ in range(n_dit_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * d_model),
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        self.final_linear = nn.Linear(d_model, in_channels * patch_size * patch_size)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        self.unpatchify = Unpatchify(in_channels, patch_size, grid_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, in_ch, H, W) — 噪声/时间插值数据
        t: (B,) — 时间步 (连续FM: t∈[0,1]; 离散DM: t∈[0,T])
        condition: (B, cond_ch, H, W) — 条件输入

        返回: (B, in_ch, H, W) — 预测目标
        """
        t_input = t * self.time_embedding_scale
        time_emb = self.time_emb(t_input)
        cond_tokens = self.cond_encoder(condition)
        x = self.patch_embed(x_t)
        x = x + self.pos_embed
        for block in self.dit_blocks:
            x = block(x, time_emb, cond_tokens)
        final_params = self.final_adaLN(time_emb).unsqueeze(1)
        gamma, beta = final_params.chunk(2, dim=-1)
        x = (1 + gamma) * self.final_norm(x) + beta
        x = self.final_linear(x)
        x = self.unpatchify(x)
        return x


# ============================================================
# Z-Predictor: learns wind → geopotential height mapping
# ============================================================

class ZPredictor(nn.Module):
    """
    Lightweight U-Net that predicts geopotential height (z) from wind fields (u, v).

    Physics basis: geostrophic balance
        u_geo = -(g/f) * ∂z/∂y
        v_geo =  (g/f) * ∂z/∂x
    Integrating geostrophic wind gives z. This module learns the residual between
    the raw geostrophic estimate and the true z.

    Architecture:
        - Encoder: downsamples wind features at 3 pressure levels (u_850,v_850 / u_500,v_500 / u_250,v_250)
        - Geostrophic prior: computes z_grad from u/v via finite differences + Coriolis
        - Decoder: upsamples and fuses wind + geostrophic prior → residual z prediction
        - Output: 3 z channels (z_850, z_500, z_250)

    Inputs:
        uv: (B, 6, H, W) — concatenated [u_850,v_850, u_500,v_500, u_250,v_250]
        center_lats: (B,) — center latitude per sample for Coriolis f = 2Ωsin(lat)
    Output:
        z_pred: (B, 3, H, W) — predicted [z_850, z_500, z_250] in normalized space
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.g = 9.80665
        self.Omega = 7.2921e-5
        self.R = 6.371e6
        self.lat_res = 0.25  # degrees per grid cell

        # Wind encoder (processes 6-channel u/v at 3 pressure levels)
        self.wind_conv_in = nn.Sequential(
            nn.Conv2d(6, d_model, 3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
        )
        # Multi-scale encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, 3, stride=2, padding=1),
            nn.GroupNorm(16, d_model * 2),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model * 2, 3, padding=1),
            nn.GroupNorm(16, d_model * 2),
            nn.SiLU(),
            nn.Conv2d(d_model * 2, d_model * 2, 3, padding=1),
            nn.GroupNorm(16, d_model * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model * 4, 3, stride=2, padding=1),
            nn.GroupNorm(32, d_model * 4),
            nn.SiLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model * 4, 3, padding=1),
            nn.GroupNorm(32, d_model * 4),
            nn.SiLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(d_model * 4, d_model * 2, 2, stride=2),
            nn.GroupNorm(16, d_model * 2),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model * 2, 3, padding=1),
            nn.GroupNorm(16, d_model * 2),
            nn.SiLU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(d_model * 2, d_model, 2, stride=2),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
        )
        # Geostrophic prior: z from u/v via ∂z/∂x = f/g * v, ∂z/∂y = -f/g * u
        self.geo_mlp = nn.Sequential(
            nn.Linear(4, 32),   # input: [dzdx_norm, dzdy_norm] per level
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # Fusion: wind features + geostrophic prior → z
        self.fusion = nn.Sequential(
            nn.Conv2d(d_model + 3, d_model, 3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
        )
        # Output head
        self.head = nn.Conv2d(d_model, 3, 1)   # 3 z channels

    def _compute_geo_prior(self, uv: torch.Tensor, center_lats: torch.Tensor) -> torch.Tensor:
        """
        Compute geostrophic height anomaly from wind via f-plane balance.
        Returns z_geo per pressure level in normalized space (approximate).
        uv: (B, 6, H, W) = [u_850,v_850, u_500,v_500, u_250,v_250]
        Returns: (B, 3, H, W) geostrophic z anomaly
        """
        B, _, H, W = uv.shape
        device = uv.device

        # Build latitude grid (B, H)
        half_span = (H * self.lat_res) / 2.0
        lat_offsets = torch.linspace(-half_span, half_span, H, device=device)
        lats = center_lats.unsqueeze(1) + lat_offsets.unsqueeze(0)  # (B, H)

        # Coriolis parameter f = 2Ωsin(lat), shape (B, H)
        f = 2.0 * self.Omega * torch.sin(torch.deg2rad(lats))
        f = torch.where(f.abs() < 1e-5, torch.sign(f).clamp(min=1) * 1e-5, f)
        f = f.unsqueeze(2)  # (B, H, 1)

        # Grid spacing (meters)
        dx_m = self.R * torch.cos(torch.deg2rad(lats)) * (math.pi / 180) * self.lat_res
        dx_m = dx_m.unsqueeze(2)  # (B, H, 1)
        dy_m = self.R * (math.pi / 180) * self.lat_res  # scalar (same for all rows)

        # dv/dx = (v[:, :, :, 2:] - v[:, :, :, :-2]) / (2*dx_m[:, :-2])
        # Use padding instead of slicing to keep dimensions
        def grad_x(field):
            """Central diff in X, output (B, C, H, W)"""
            padded = F.pad(field, (2, 2, 0, 0), mode='replicate')  # (B, C, H, W+4)
            diff = padded[:, :, :, 4:] - padded[:, :, :, :-4]       # (B, C, H, W)
            # dx_m: (B, H) → broadcast to (B, 1, H, 1)
            dx_bc = dx_m.unsqueeze(1).unsqueeze(-1)
            return diff / (8 * dx_bc.clamp(min=1e-8))

        def grad_y(field):
            """Central diff in Y, output (B, C, H, W)"""
            padded = F.pad(field, (0, 0, 2, 2), mode='replicate')  # (B, C, H+4, W)
            diff = padded[:, :, 4:, :] - padded[:, :, :-4, :]       # (B, C, H, W)
            dy_scalar = dy_m  # scalar
            return diff / (8 * dy_scalar + 1e-8)

        # du/dy per level: shape (B, 3, H, W)
        du_dy = grad_y(uv[:, 0::2])
        # dv/dx per level: shape (B, 3, H, W)
        dv_dx = grad_x(uv[:, 1::2])

        # g/f: shape (B, H, 1)
        g_over_f = (self.g / f).unsqueeze(-1)
        # Broadcast to (B, 1, H, 1) for elementwise with (B, 3, H, W)
        g_over_f = g_over_f.unsqueeze(1)

        # du/dy per level: shape (B, 3, H, W)
        du_dy = grad_y(uv[:, 0::2])
        # dv/dx per level: shape (B, 3, H, W)
        dv_dx = grad_x(uv[:, 1::2])

        # Geostrophic z: z ≈ -(g/f) * du/dy (from u_geo = -g/f * dz/dy)
        z_from_u = -(g_over_f * du_dy)
        # Geostrophic z: z ≈ (g/f) * dv/dx (from v_geo = g/f * dz/dx)
        z_from_v = g_over_f * dv_dx

        # Average both estimates
        z_geo = 0.5 * (z_from_v + z_from_u)  # (B, 3, H, W)

        # Normalize to unit-ish variance so it's comparable to the U-Net features
        z_geo = z_geo / (z_geo.std(dim=(2, 3), keepdim=True).clamp(min=1e-4) + 1e-8)
        return z_geo

    def forward(self, uv: torch.Tensor, center_lats: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            uv: (B, 6, H, W) — wind at 3 pressure levels
            center_lats: (B,) optional — for Coriolis; defaults to 20°N
        Returns:
            z_pred: (B, 3, H, W) — z at 3 pressure levels
        """
        B, _, H, W = uv.shape
        if center_lats is None:
            center_lats = torch.full((B,), 20.0, device=uv.device)

        # Wind encoding
        h = self.wind_conv_in(uv)
        h1 = self.enc1(h)       # (B, d, H, W)
        h2 = self.down1(h1)     # (B, 2d, H/2, W/2)
        h2 = self.enc2(h2)
        h3 = self.down2(h2)     # (B, 4d, H/4, W/4)
        h3 = self.bottleneck(h3)
        h3_up = self.up2(h3)    # (B, 2d, H/2, W/2)
        h2_cat = torch.cat([h3_up, h2], dim=1)
        h2_dec = self.dec2(h2_cat)
        h2_up = self.up1(h2_dec) # (B, d, H, W)
        h1_cat = torch.cat([h2_up, h1], dim=1)
        h1_dec = self.dec1(h1_cat)  # (B, d, H, W)

        # Simple geostrophic signal: raw spatial gradients of u/v per level,
        # stacked and normalized. Shape: (B, 3, H, W) — 1 scalar per spatial point per level.
        # This is a rough physical prior; the U-Net refines it.
        def grad_x(field):
            padded = F.pad(field, (2, 2, 0, 0), mode='replicate')
            diff = padded[:, :, :, 4:] - padded[:, :, :, :-4]
            return diff / 8.0

        def grad_y(field):
            padded = F.pad(field, (0, 0, 2, 2), mode='replicate')
            diff = padded[:, :, 4:, :] - padded[:, :, :-4, :]
            return diff / 8.0

        # dz/dx from v, dz/dy from u, per level — 2 features per level = 6 total
        geo_raw = torch.cat([grad_x(uv[:, 1::2]), grad_y(uv[:, 0::2])], dim=1)  # (B, 6, H, W)
        geo_raw = geo_raw / (geo_raw.std(dim=(2, 3), keepdim=True).clamp(min=1e-4) + 1e-8)
        # Pool to (B, 3, H, W) by averaging the 2 features per level
        geo_prior = geo_raw[:, 0::2] * 0.5 + geo_raw[:, 1::2] * 0.5   # (B, 3, H, W)

        # Fusion: U-Net features + geostrophic prior
        fused = torch.cat([h1_dec, geo_prior], dim=1)  # (B, d+3, H, W)
        fused = self.fusion(fused)                       # (B, d, H, W)
        z_pred = self.head(fused)                        # (B, 3, H, W)
        return z_pred

