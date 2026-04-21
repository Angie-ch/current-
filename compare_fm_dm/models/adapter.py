"""
外部模型适配器 — 加载 newtry 项目的 checkpoint 到 compare_fm_dm 评估管道

核心: 创建与 newtry 架构完全一致的模型，直接加载其 checkpoint
使用:
    from models.adapter import load_newtry_checkpoint
    model = load_newtry_checkpoint("newtry/checkpoints/best_eps.pt", data_cfg, model_cfg, device)
"""
import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .components import SinusoidalTimeEmbedding, AdaLayerNorm, PatchEmbed, Unpatchify, DiTBlock


# ============================================================
# 复用 newtry 的组件 (与 newtry/models/components.py 完全一致)
# ============================================================

class ConditionEncoder(nn.Module):
    """与 newtry 完全一致的条件编码器"""

    def __init__(self, cond_channels=45, d_model=384, n_heads=6, n_cond_layers=3,
                 ff_mult=4, patch_size=4, grid_size=40, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        num_patches = (grid_size // patch_size) ** 2

        self.local_conv = nn.Sequential(
            nn.Conv2d(cond_channels, d_model, 3, padding=1),
            nn.GroupNorm(min(32, d_model), d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GroupNorm(min(32, d_model), d_model),
            nn.SiLU(),
        )

        self.patch_embed = PatchEmbed(d_model, d_model, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_cond_layers):
            self.layers.append(CondSelfAttnBlockSimple(d_model, n_heads, ff_mult, dropout))
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        h = self.local_conv(condition)
        tokens = self.patch_embed(h)
        tokens = tokens + self.pos_embed
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm_out(tokens)


class CondSelfAttnBlockSimple(nn.Module):
    """newtry 风格的自注意力块 (无 Cross-Attention)"""

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
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + self.dropout(self.proj(attn_out))
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleDiT(nn.Module):
    """与 newtry 的 ERA5DiT 完全一致的骨干网络"""

    def __init__(self, in_channels=9, cond_channels=45, d_model=384, n_heads=6,
                 n_dit_layers=12, n_cond_layers=3, ff_mult=4, patch_size=4,
                 grid_size=40, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        num_patches = (grid_size // patch_size) ** 2
        time_emb_dim = d_model

        self.time_emb = SinusoidalTimeEmbedding(d_model, time_emb_dim)
        self.cond_encoder = ConditionEncoder(
            cond_channels=cond_channels, d_model=d_model, n_heads=n_heads,
            n_cond_layers=n_cond_layers, ff_mult=ff_mult,
            patch_size=patch_size, grid_size=grid_size, dropout=dropout
        )
        self.patch_embed = PatchEmbed(in_channels, d_model, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dit_blocks = nn.ModuleList([
            SimpleDiTBlock(d_model, n_heads, ff_mult, time_emb_dim, dropout)
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

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_emb(t)
        cond_tokens = self.cond_encoder(condition)
        x = self.patch_embed(x_noisy) + self.pos_embed
        for block in self.dit_blocks:
            x = block(x, time_emb)
        final_params = self.final_adaLN(time_emb).unsqueeze(1)
        gamma, beta = final_params.chunk(2, dim=-1)
        x = (1 + gamma) * self.final_norm(x) + beta
        x = self.final_linear(x)
        return self.unpatchify(x)


class SimpleDiTBlock(nn.Module):
    """newtry 风格的 DiT Block (仅 Self-Attn + FFN)"""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4,
                 time_emb_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.adaln = AdaLayerNorm(d_model, time_emb_dim)
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(x, time_emb)
        h = self.adaln.modulate(x, gamma1, beta1)
        qkv = self.self_attn_qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + alpha1 * self.dropout(self.self_attn_proj(attn_out))
        h_ffn = self.adaln.modulate(x, gamma2, beta2)
        x = x + alpha2 * self.ffn(h_ffn)
        return x


# ============================================================
# 损失函数 (复用 unified_model 的实现)
# ============================================================

class DivergenceLoss(nn.Module):
    """风场散度正则化"""

    def __init__(self, wind_pairs):
        super().__init__()
        self.wind_pairs = wind_pairs

    def forward(self, x0_pred: torch.Tensor) -> torch.Tensor:
        total = 0.0
        count = 0
        for u_idx, v_idx in self.wind_pairs:
            u = x0_pred[:, u_idx]
            v = x0_pred[:, v_idx]
            du_dx = u[:, :, 1:] - u[:, :, :-1]
            dv_dy = v[:, 1:, :] - v[:, :-1, :]
            du_dx = du_dx[:, :-1, :]
            dv_dy = dv_dy[:, :, :-1]
            div = du_dx + dv_dy
            total += (div ** 2).mean()
            count += 1
        return total / max(count, 1)


class ChannelWeightedMSE(nn.Module):
    """逐通道加权 MSE"""

    def __init__(self, channel_weights: torch.Tensor):
        super().__init__()
        w = channel_weights / channel_weights.mean()
        self.register_buffer("weights", w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = pred.shape[1]
        w = self.weights[:C].reshape(1, C, 1, 1)
        return (w * (pred - target) ** 2).mean()


class VorticityCurlLoss(nn.Module):
    """旋度一致性约束"""

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
            total_loss += torch.nn.functional.mse_loss(vo_inner, curl_from_wind)
            count += 1
        return total_loss / max(count, 1)


# ============================================================
# Diffusion Scheduler (与 newtry 一致)
# ============================================================

class DiffusionScheduler:
    """与 newtry 相同的余弦调度器"""

    def __init__(self, num_steps=1000, schedule="cosine", s=0.008,
                 ddim_steps=50, clamp_range=(-5.0, 5.0), prediction_type="eps"):
        self.num_steps = num_steps
        self.ddim_steps = ddim_steps
        self.clamp_range = clamp_range
        self.prediction_type = prediction_type

        steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
        f_t = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f_t / f_t[0]
        alphas_cumprod = alphas_cumprod.clamp(min=1e-8, max=1.0)
        self.alphas_cumprod = alphas_cumprod.float()
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.ddim_timesteps = self._make_ddim_timesteps(ddim_steps, num_steps)

    def _make_ddim_timesteps(self, ddim_steps: int, total_steps: int) -> torch.Tensor:
        step_size = total_steps // ddim_steps
        timesteps = torch.arange(0, total_steps, step_size)
        return timesteps.flip(0)

    def to(self, device: torch.device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        return self

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_noisy, noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha.clamp(min=1e-8)

    def predict_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, condition: torch.Tensor,
                    shape: Tuple[int, ...], device: torch.device, eta: float = 0.0,
                    z_channel_indices: Optional[list] = None,
                    z_clamp_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        B = shape[0]
        x = torch.randn(shape, device=device)
        timesteps = self.ddim_timesteps

        for i in range(len(timesteps)):
            t_current = timesteps[i]
            t_batch = torch.full((B,), t_current, device=device, dtype=torch.long)
            model_output = model(x, t_batch, condition)

            if self.prediction_type == "v":
                x0_pred = self.predict_x0_from_v(x, t_batch, model_output)
            else:
                x0_pred = self.predict_x0_from_eps(x, t_batch, model_output)

            x0_pred = x0_pred.clamp(*self.clamp_range)
            if z_channel_indices and z_clamp_range:
                x0_pred[:, z_channel_indices] = x0_pred[:, z_channel_indices].clamp(*z_clamp_range)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t = self.alphas_cumprod[t_current + 1]
                alpha_next = self.alphas_cumprod[t_next + 1]
                sigma = eta * torch.sqrt(
                    (1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next)
                )
                if self.prediction_type == "v":
                    eps_pred = self.predict_eps_from_v(x, t_batch, model_output)
                else:
                    eps_pred = model_output
                pred_dir = torch.sqrt(1 - alpha_next - sigma ** 2) * eps_pred
                x = torch.sqrt(alpha_next) * x0_pred + pred_dir
                if sigma > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                x = x0_pred
        return x

    def predict_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_one_minus_alpha * x_t + sqrt_alpha * v


import math


# ============================================================
# AdaptedDiffusionModel — 与 newtry 架构完全一致
# ============================================================

class AdaptedDiffusionModel(nn.Module):
    """
    适配的扩散模型 — 完全兼容 newtry 的 ERA5DiffusionModel

    架构:
      - SimpleDiT (无 Cross-Attention, 无 temporal_conv3d)
      - DiffusionScheduler (cosine schedule)
      - 物理损失 (div_loss, curl_loss)
      - Channel-weighted MSE
    """

    def __init__(self, model_cfg, data_cfg, train_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

        # 核心模型
        self.dit = SimpleDiT(
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
        )

        # Diffusion 调度器
        self.scheduler = DiffusionScheduler(
            num_steps=getattr(model_cfg, 'num_diffusion_steps', 1000),
            schedule=getattr(model_cfg, 'noise_schedule', 'cosine'),
            ddim_steps=getattr(model_cfg, 'ddim_sampling_steps', 50),
            prediction_type=getattr(model_cfg, 'prediction_type', 'eps'),
        )

        # 损失函数
        wind_pairs = data_cfg.get_wind_channel_indices()
        self.div_loss = DivergenceLoss(wind_pairs)

        if "vo" in data_cfg.pressure_level_vars:
            self.curl_loss = VorticityCurlLoss(data_cfg)
        else:
            self.curl_loss = None

        if train_cfg is not None and getattr(train_cfg, 'use_channel_weights', False):
            weights = torch.tensor(train_cfg.channel_weights, dtype=torch.float32)
            if data_cfg.forecast_steps > 1:
                per_step = weights[:data_cfg.num_channels]
                weights = per_step.repeat(data_cfg.forecast_steps)
            self.channel_mse = ChannelWeightedMSE(weights)
        else:
            self.channel_mse = None

        # z 通道索引
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
        self.scheduler.to(device)

        t = torch.randint(0, self.model_cfg.num_diffusion_steps, (B,), device=device)
        x_noisy, noise = self.scheduler.q_sample(target, t)
        model_output = self.dit(x_noisy, t, condition)

        if self.scheduler.prediction_type == "v":
            v_target = self.scheduler.compute_v_target(target, t, noise)
            loss_mse = self.channel_mse(model_output, v_target) if self.channel_mse else torch.nn.functional.mse_loss(model_output, v_target)
            x0_pred = self.scheduler.predict_x0_from_v(x_noisy, t, model_output)
        else:
            loss_mse = self.channel_mse(model_output, noise) if self.channel_mse else torch.nn.functional.mse_loss(model_output, noise)
            x0_pred = self.scheduler.predict_x0_from_eps(x_noisy, t, model_output)

        loss_div = torch.clamp(self.div_loss(x0_pred), max=10.0)
        loss_curl = torch.clamp(self.curl_loss(x0_pred), max=10.0) if self.curl_loss else torch.tensor(0.0, device=device)

        return {
            "loss_mse": loss_mse,
            "loss_div": loss_div,
            "loss_curl": loss_curl,
            "eps_pred": model_output,
            "eps_true": noise,
            "x0_pred": x0_pred,
            "t": t.float() / self.model_cfg.num_diffusion_steps,
        }

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, device: torch.device,
               ddim_steps: int = 50, clamp_range: Tuple[float, float] = (-5.0, 5.0),
               z_clamp_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        self.scheduler.to(device)
        self.scheduler.ddim_steps = ddim_steps
        self.scheduler.clamp_range = clamp_range
        B = condition.shape[0]
        shape = (B, self.model_cfg.in_channels, self.data_cfg.grid_size, self.data_cfg.grid_size)
        return self.scheduler.ddim_sample(
            self.dit, condition, shape, device,
            z_channel_indices=self.z_channel_indices if self.z_channel_indices else None,
            z_clamp_range=z_clamp_range,
        )


# ============================================================
# 加载函数
# ============================================================

def load_newtry_checkpoint(
    checkpoint_path: str,
    data_cfg,
    model_cfg,
    train_cfg=None,
    device: torch.device = torch.device("cpu"),
) -> AdaptedDiffusionModel:
    """
    加载 newtry checkpoint 并返回 AdaptedDiffusionModel

    由于架构完全一致，直接加载 state_dict 即可
    """
    model = AdaptedDiffusionModel(model_cfg, data_cfg, train_cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # newtry 的参数名前缀是 "dit."，与 AdaptedDiffusionModel 完全一致
    # 直接加载
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"⚠️  缺失参数: {missing}")
    if unexpected:
        print(f"⚠️  多余参数: {unexpected[:5]}")

    epoch = ckpt.get('epoch', 'N/A')
    print(f"✅ 成功加载 newtry checkpoint (epoch {epoch})")

    return model
