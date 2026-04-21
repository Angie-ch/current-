"""
Flow Matching 核心模型 (DiT 架构)

基于 ICLR 2023 "Flow Matching for Generative Modeling" (Lipman et al.)
"""
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from newtry.models.components import (
    SinusoidalTimeEmbedding,
    AdaLayerNorm,
    PatchEmbed,
    Unpatchify,
    DiTBlock,
)


class ConditionEncoder(nn.Module):
    """条件编码器 — 将历史气象条件编码为 token 序列"""

    def __init__(
        self,
        cond_channels: int = 60,
        d_model: int = 384,
        n_heads: int = 6,
        n_cond_layers: int = 3,
        ff_mult: int = 4,
        patch_size: int = 4,
        grid_size: int = 40,
        dropout: float = 0.1,
        use_grouped_conv: bool = False,
        num_var_groups: int = 3,
        history_steps: int = 16,
        use_temporal_agg: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_grouped_conv = use_grouped_conv
        self.num_var_groups = num_var_groups
        self.history_steps = history_steps
        self.use_temporal_agg = use_temporal_agg
        num_patches = (grid_size // patch_size) ** 2

        if self.use_temporal_agg:
            self.temporal_conv3d = nn.Sequential(
                nn.Conv3d(cond_channels, cond_channels, kernel_size=(3, 3, 3),
                          padding=(1, 1, 1), stride=(2, 1, 1)),
                nn.GroupNorm(1, cond_channels),
                nn.SiLU(),
                nn.Conv3d(cond_channels, cond_channels, kernel_size=(3, 3, 3),
                          padding=(1, 1, 1), stride=(2, 1, 1)),
                nn.GroupNorm(1, cond_channels),
                nn.SiLU(),
                nn.Conv3d(cond_channels, cond_channels, kernel_size=(3, 3, 3),
                          padding=(1, 1, 1), stride=(2, 1, 1)),
                nn.GroupNorm(1, cond_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool3d((1, grid_size, grid_size)),
            )

        if use_grouped_conv and cond_channels >= num_var_groups:
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
        else:
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
        if condition.ndim == 5:
            B, T, C, H, W = condition.shape
            condition_3d = condition.permute(0, 2, 1, 3, 4)
            condition = self.temporal_conv3d(condition_3d).squeeze(2)
            last_frame = condition_3d[:, :, -1, :, :]
            condition = condition + last_frame

        if self.use_grouped_conv and self.local_conv is None:
            h = self._group_forward(condition)
        else:
            h = self.local_conv(condition)

        tokens = self.patch_embed(h)
        tokens = tokens + self.pos_embed
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm_out(tokens)


class CondSelfAttnBlock(nn.Module):
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


class CFMDiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        cond_channels: int = 45,
        d_model: int = 384,
        n_heads: int = 6,
        n_dit_layers: int = 12,
        n_cond_layers: int = 3,
        ff_mult: int = 4,
        patch_size: int = 4,
        grid_size: int = 40,
        dropout: float = 0.1,
        time_embedding_scale: float = 1000.0,
        use_grouped_conv: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_size = patch_size
        self.time_embedding_scale = time_embedding_scale
        num_patches = (grid_size // patch_size) ** 2
        time_emb_dim = d_model

        self.time_emb = SinusoidalTimeEmbedding(d_model, time_emb_dim)
        self.cond_encoder = ConditionEncoder(
            cond_channels=cond_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_cond_layers=n_cond_layers,
            ff_mult=ff_mult,
            patch_size=patch_size,
            grid_size=grid_size,
            dropout=dropout,
            use_grouped_conv=use_grouped_conv,
            history_steps=16,
            use_temporal_agg=True,
        )
        self.patch_embed = PatchEmbed(in_channels, d_model, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dit_blocks = nn.ModuleList([
            DiTBlock(d_model=d_model, n_heads=n_heads, ff_mult=ff_mult,
                     time_emb_dim=time_emb_dim, dropout=dropout)
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


class DivergenceLoss(nn.Module):
    def __init__(self, wind_pairs: List[Tuple[int, int]]):
        super().__init__()
        self.wind_pairs = wind_pairs

    def forward(self, x0_pred: torch.Tensor) -> torch.Tensor:
        total_div = 0.0
        count = 0
        for u_idx, v_idx in self.wind_pairs:
            u = x0_pred[:, u_idx]
            v = x0_pred[:, v_idx]
            du_dx = u[:, :, 1:] - u[:, :, :-1]
            dv_dy = v[:, 1:, :] - v[:, :-1, :]
            du_dx = du_dx[:, :-1, :]
            dv_dy = dv_dy[:, :, :-1]
            divergence = du_dx + dv_dy
            total_div = total_div + (divergence ** 2).mean()
            count += 1
        return total_div / max(count, 1)


class VelocitySolenoidalLoss(nn.Module):
    def __init__(self, wind_pairs: List[Tuple[int, int]]):
        super().__init__()
        self.wind_pairs = wind_pairs

    def forward(self, velocity_field: torch.Tensor) -> torch.Tensor:
        total_div_loss = 0.0
        count = 0
        for u_idx, v_idx in self.wind_pairs:
            u = velocity_field[:, u_idx]
            v = velocity_field[:, v_idx]
            du_dx = u[:, :, 1:] - u[:, :, :-1]
            dv_dy = v[:, 1:, :] - v[:, :-1, :]
            du_dx = du_dx[:, :-1, :]
            dv_dy = dv_dy[:, :, :-1]
            divergence = du_dx + dv_dy
            div_loss = (divergence ** 2).mean()
            total_div_loss = total_div_loss + div_loss
            count += 1
        return total_div_loss / max(count, 1)


class VorticityCurlLoss(nn.Module):
    def __init__(self, data_cfg):
        super().__init__()
        n_pl = len(data_cfg.pressure_levels)
        pl_vars = data_cfg.pressure_level_vars
        u_idx_in_pl = pl_vars.index("u")
        v_idx_in_pl = pl_vars.index("v")
        vo_idx_in_pl = pl_vars.index("vo")
        triplets = []
        for t_step in range(data_cfg.forecast_steps):
            base = t_step * data_cfg.num_channels
            for lev in range(n_pl):
                u_ch = base + u_idx_in_pl * n_pl + lev
                v_ch = base + v_idx_in_pl * n_pl + lev
                vo_ch = base + vo_idx_in_pl * n_pl + lev
                triplets.append((u_ch, v_ch, vo_ch))
        self.triplets = triplets

    def forward(self, x0_pred: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        count = 0
        for u_ch, v_ch, vo_ch in self.triplets:
            u = x0_pred[:, u_ch]
            v = x0_pred[:, v_ch]
            vo = x0_pred[:, vo_ch]
            dv_dx = v[:, :, 1:] - v[:, :, :-1]
            du_dy = u[:, 1:, :] - u[:, :-1, :]
            dv_dx = dv_dx[:, :-1, :]
            du_dy = du_dy[:, :, :-1]
            curl_from_wind = dv_dx - du_dy
            vo_inner = vo[:, :-1, :-1]
            total_loss = total_loss + F.mse_loss(vo_inner, curl_from_wind)
            count += 1
        return total_loss / max(count, 1)


class ChannelWeightedMSE(nn.Module):
    """
    逐通道加权 RMSE 归一化损失

    核心问题:
    - Z 通道归一化后值域极小 (~0.1-1.8)，MSE 贡献 ~2%
    - UV 通道值域大 (~5-8)，MSE 贡献 ~98%
    - 即使提高 Z 权重，也无法平衡量级差异

    解决方案: 逐通道 RMSE 归一化
    - 每通道 RMSE = sqrt(mean((pred - target)²))
    - 归一化: rmse_i / sqrt(mean(target²)) → 消除量级差异
    - 乘以权重: final_loss = sum(w_i * normalized_i)

    效果: 所有通道贡献比例由权重决定，而非数据量级
    """

    def __init__(self, channel_weights: torch.Tensor, pressure_level_weights: torch.Tensor = None,
                 use_normalized: bool = True):
        super().__init__()
        w = channel_weights / channel_weights.mean()
        if pressure_level_weights is not None:
            w = w * (pressure_level_weights / pressure_level_weights.mean())
        self.register_buffer("weights", w)
        self.use_normalized = use_normalized

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = pred.shape[1]
        w = self.weights[:C]

        if self.use_normalized:
            # 逐通道 RMSE 归一化
            diff = pred - target  # (B, C, H, W)
            # 每通道 RMSE: sqrt(mean over B,H,W)
            channel_rmse = torch.sqrt((diff ** 2).mean(dim=(0, 2, 3)) + 1e-8)  # (C,)
            # 除以目标场的 RMS 消除量级差异
            target_rms = torch.sqrt((target ** 2).mean(dim=(0, 2, 3)) + 1e-8)  # (C,)
            normalized = channel_rmse / target_rms  # (C,) - 每通道独立归一化
            # 加权求和后再取平均，使损失值 ~1.0 范围
            return (w[:C] * normalized).mean()
        else:
            w = w[:C].reshape(1, C, 1, 1)
            return (w * (pred - target) ** 2).mean()


class ERA5FlowMatchingModel(nn.Module):
    def __init__(self, model_cfg, data_cfg, train_cfg=None):
        super().__init__()
        self.dit = CFMDiT(
            in_channels=model_cfg.in_channels,
            cond_channels=model_cfg.cond_channels,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_dit_layers=model_cfg.n_dit_layers,
            n_cond_layers=model_cfg.n_cond_layers,
            ff_mult=model_cfg.ff_mult,
            patch_size=model_cfg.patch_size,
            grid_size=data_cfg.grid_size,
            dropout=model_cfg.dropout,
            time_embedding_scale=getattr(model_cfg, 'time_embedding_scale', 1000.0),
            use_grouped_conv=getattr(model_cfg, 'use_grouped_conv', True),
        )
        self.cond_encoder = self.dit.cond_encoder

        wind_pairs = data_cfg.get_wind_channel_indices()
        self.div_loss = DivergenceLoss(wind_pairs)
        self.sol_loss = VelocitySolenoidalLoss(wind_pairs)

        if "vo" in data_cfg.pressure_level_vars:
            self.curl_loss = VorticityCurlLoss(data_cfg)
        else:
            self.curl_loss = None

        if train_cfg is not None and train_cfg.use_channel_weights:
            weights = torch.tensor(train_cfg.channel_weights, dtype=torch.float32)
            if data_cfg.forecast_steps > 1:
                per_step = weights[:data_cfg.num_channels]
                weights = per_step.repeat(data_cfg.forecast_steps)
            pl_weights = getattr(train_cfg, 'pressure_level_weights', None)
            if pl_weights is not None:
                pl_weights = torch.tensor(pl_weights, dtype=torch.float32)
                n_vars = len(data_cfg.pressure_level_vars)
                pl_weights = pl_weights.repeat(n_vars)
                if data_cfg.forecast_steps > 1:
                    pl_weights = pl_weights.repeat(data_cfg.forecast_steps)
            use_norm = getattr(train_cfg, 'use_channel_wise_normalized_loss', True)
            self.channel_mse = ChannelWeightedMSE(weights, pl_weights, use_normalized=use_norm)
        else:
            self.channel_mse = None

        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

        if "z" in data_cfg.pressure_level_vars:
            z_var_idx = data_cfg.pressure_level_vars.index("z")
            n_levels = len(data_cfg.pressure_levels)
            self.z_channel_indices = list(range(
                z_var_idx * n_levels, (z_var_idx + 1) * n_levels
            ))
        else:
            self.z_channel_indices = []

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = target.device
        B = target.shape[0]

        m = torch.randn(B, device=device)
        t = torch.sigmoid(m)
        t = torch.clamp(t, min=1e-5, max=1 - 1e-5)
        x_1 = torch.randn_like(target)
        t_reshaped = t.reshape(-1, 1, 1, 1)
        x_t = (1 - t_reshaped) * target + t_reshaped * x_1
        target_v = x_1 - target

        if self.training and self.train_cfg is not None:
            pert_prob = getattr(self.train_cfg, 'path_perturb_prob', 0.0)
            if pert_prob > 0 and torch.rand(1, device=device).item() < pert_prob:
                pert_sigma = getattr(self.train_cfg, 'path_perturb_sigma', 0.05)
                x_t = x_t + torch.randn_like(x_t) * pert_sigma

        v_pred = self.dit(x_t, t, condition)

        if self.train_cfg is not None and self.train_cfg.velocity_clamp is not None:
            v_pred = v_pred.clamp(*self.train_cfg.velocity_clamp)

        if self.channel_mse is not None:
            loss_mse = self.channel_mse(v_pred, target_v)
        else:
            loss_mse = F.mse_loss(v_pred, target_v)

        if self.train_cfg is not None and self.train_cfg.velocity_loss_scale != 1.0:
            loss_mse = loss_mse * self.train_cfg.velocity_loss_scale

        with torch.no_grad():
            x0_pred = x_t - t_reshaped * v_pred

        if self.channel_mse is not None:
            loss_x0 = self.channel_mse(x0_pred, target)
        else:
            loss_x0 = F.mse_loss(x0_pred, target)

        # 计算逐通道损失用于监控
        with torch.no_grad():
            diff = x0_pred - target
            C = target.shape[1]
            channel_rmse = torch.sqrt((diff ** 2).mean(dim=(0, 2, 3)) + 1e-8)
            channel_names = ['u_850','u_500','u_250','v_850','v_500','v_250','z_850','z_500','z_250']
            channel_losses = {channel_names[i]: channel_rmse[i].item() for i in range(min(C, 9))}

        loss_div_raw = self.div_loss(x0_pred)
        loss_div = torch.clamp(loss_div_raw, max=10.0)
        loss_sol_raw = self.sol_loss(v_pred)
        loss_sol = torch.clamp(loss_sol_raw, max=10.0)
        loss_curl_raw = (
            self.curl_loss(x0_pred) if self.curl_loss is not None
            else torch.tensor(0.0, device=device)
        )
        loss_curl = (
            torch.clamp(loss_curl_raw, max=10.0) if self.curl_loss is not None
            else loss_curl_raw
        )

        return {
            "loss_mse": loss_mse,
            "loss_x0": loss_x0,
            "loss_div": loss_div,
            "loss_sol": loss_sol,
            "loss_curl": loss_curl,
            "v_pred": v_pred,
            "v_true": target_v,
            "x0_pred": x0_pred,
            "channel_losses": channel_losses,
        }
