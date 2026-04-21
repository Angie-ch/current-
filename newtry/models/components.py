"""
DiT (Diffusion Transformer) 核心组件:
  - SinusoidalTimeEmbedding: 正弦时间步嵌入
  - AdaLayerNorm: 自适应层归一化 (时间步条件注入)
  - PatchEmbed: 2D Patch 嵌入层
  - Unpatchify: 逆 Patch 操作
  - DiTBlock: DiT 核心块 (AdaLN + Self-Attn + Cross-Attn + FFN)
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
    """正弦位置编码 + MLP 投影"""

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
        t: (B,) int 时间步
        return: (B, time_emb_dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return self.mlp(emb)


# ============================================================
# 自适应层归一化 (AdaLN)
# ============================================================

class AdaLayerNorm(nn.Module):
    """
    自适应层归一化 — 通过时间步嵌入生成 scale/shift/gate 参数

    h = gate * (scale * LayerNorm(x) + shift)

    每个 DiTBlock 需要两组 AdaLN 参数 (Self-Attn 前 + FFN 前)，
    共需 6 个参数向量: (γ₁, β₁, α₁, γ₂, β₂, α₂)
    """

    def __init__(self, d_model: int, time_emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # 生成 6 个参数: (γ₁, β₁, α₁, γ₂, β₂, α₂)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 6 * d_model),
        )
        # 初始化为零 → 初始时 gate=1, scale=1, shift=0 (恒等变换)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        """
        x: (B, N, D)
        time_emb: (B, time_emb_dim)

        返回: (γ₁, β₁, α₁, γ₂, β₂, α₂) 每个 shape (B, 1, D)
        """
        params = self.adaLN_modulation(time_emb).unsqueeze(1)  # (B, 1, 6D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)
        return gamma1, beta1, alpha1, gamma2, beta2, alpha2

    def modulate(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """应用 AdaLN: scale * LN(x) + shift"""
        return (1 + gamma) * self.norm(x) + beta


# ============================================================
# Patch 嵌入 / 反 Patch
# ============================================================

class PatchEmbed(nn.Module):
    """
    将 2D 图像切成 patch 并线性嵌入

    输入: (B, C, H, W)
    输出: (B, num_patches, d_model)
    """

    def __init__(self, in_channels: int, d_model: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H/p, W/p)
        B, D, Hp, Wp = x.shape
        x = x.reshape(B, D, Hp * Wp).permute(0, 2, 1)  # (B, N, D)
        return x


class Unpatchify(nn.Module):
    """
    将 token 序列还原为 2D 图像

    输入: (B, num_patches, patch_size² × out_channels)
    输出: (B, out_channels, H, W)
    """

    def __init__(self, out_channels: int, patch_size: int = 4, grid_size: int = 40):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.grid_h = grid_size // patch_size
        self.grid_w = grid_size // patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, p*p*C)
        """
        B, N, _ = x.shape
        p = self.patch_size
        C = self.out_channels
        # (B, N, p*p*C) -> (B, Hp, Wp, p, p, C)
        x = x.reshape(B, self.grid_h, self.grid_w, p, p, C)
        # (B, Hp, Wp, p, p, C) -> (B, C, Hp, p, Wp, p)
        x = x.permute(0, 5, 1, 3, 2, 4)
        # (B, C, Hp*p, Wp*p)
        x = x.reshape(B, C, self.grid_h * p, self.grid_w * p)
        return x


# ============================================================
# DiT Block
# ============================================================

class DiTBlock(nn.Module):
    """
    DiT 核心块:
      a) AdaLN → Self-Attention → 残差连接
      b) Cross-Attention(Q=噪声token, K/V=条件token) → 残差连接
      c) AdaLN → FFN(D → 4D → D) → 残差连接

    时间步 t 通过 AdaLN 注入 (调制 Self-Attn 和 FFN)
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
        assert d_model % n_heads == 0, f"d_model {d_model} 必须被 n_heads {n_heads} 整除"

        # AdaLN (生成 6 个参数: γ₁, β₁, α₁, γ₂, β₂, α₂)
        self.adaln = AdaLayerNorm(d_model, time_emb_dim)

        # Self-Attention
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_proj = nn.Linear(d_model, d_model)

        # Cross-Attention
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_kv = nn.Linear(d_model, 2 * d_model)
        self.cross_attn_proj = nn.Linear(d_model, d_model)

        # FFN
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
        """
        x: (B, N, D) 噪声 token 序列
        time_emb: (B, time_emb_dim) 时间步嵌入
        cond_tokens: (B, N_cond, D) 条件 token 序列

        返回: (B, N, D)
        """
        B, N, D = x.shape

        # --- AdaLN 参数 ---
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(x, time_emb)

        # --- (a) Self-Attention ---
        h = self.adaln.modulate(x, gamma1, beta1)
        qkv = self.self_attn_qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        attn_out = self.self_attn_proj(attn_out)
        x = x + alpha1 * self.dropout(attn_out)

        # --- (b) Cross-Attention ---
        h_cross = self.cross_attn_norm(x)
        q_cross = self.cross_attn_q(h_cross).reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        N_cond = cond_tokens.shape[1]
        kv_cross = self.cross_attn_kv(cond_tokens).reshape(B, N_cond, 2, self.n_heads, self.head_dim)
        kv_cross = kv_cross.permute(2, 0, 3, 1, 4)  # (2, B, heads, N_cond, head_dim)
        k_cross, v_cross = kv_cross[0], kv_cross[1]
        cross_out = F.scaled_dot_product_attention(q_cross, k_cross, v_cross)  # (B, heads, N, head_dim)
        cross_out = cross_out.permute(0, 2, 1, 3).reshape(B, N, D)
        cross_out = self.cross_attn_proj(cross_out)
        x = x + self.dropout(cross_out)

        # --- (c) FFN ---
        h_ffn = self.adaln.modulate(x, gamma2, beta2)
        ffn_out = self.ffn(h_ffn)
        x = x + alpha2 * ffn_out

        return x
