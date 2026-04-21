"""
统一模型 — Flow Matching和Diffusion双模式

核心设计原则:
- 共享骨干: 同一套UnifiedDiT架构，参数完全相同
- 区别仅在: 训练目标函数和采样策略

Flow Matching模式:
  - 训练: MSE(v_pred, x_1 - x_0)，线性插值路径
  - 推理: Euler ODE求解器 (1-10步)

Diffusion模式:
  - 训练: MSE(eps_pred, noise) 或 MSE(v_pred, velocity)
  - 推理: DDIM采样 (50步)

物理约束损失 (FM和DM共用):
  - 散度损失: L_div = ||∇·v||²
  - 涡度损失: vo ≈ ∂v/∂x - ∂u/∂y
  - 地转平衡损失: 预测风场与地转风的偏差
"""
import math
import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import UnifiedDiT, ZPredictor


# ============================================================
# 物理约束损失 (FM和DM共用)
# ============================================================

class DivergenceLoss(nn.Module):
    """风场散度正则化: L_div = ||∂u/∂x + ∂v/∂y||²"""

    def __init__(self, wind_pairs: List[Tuple[int, int]]):
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
            total = total + (div ** 2).mean()
            count += 1
        return total / max(count, 1)


class VelocitySolenoidalLoss(nn.Module):
    """速度场无散损失: 强制速度场接近无散"""

    def __init__(self, wind_pairs: List[Tuple[int, int]]):
        super().__init__()
        self.wind_pairs = wind_pairs

    def forward(self, velocity_field: torch.Tensor) -> torch.Tensor:
        total = 0.0
        count = 0
        for u_idx, v_idx in self.wind_pairs:
            u = velocity_field[:, u_idx]
            v = velocity_field[:, v_idx]
            du_dx = u[:, :, 1:] - u[:, :, :-1]
            dv_dy = v[:, 1:, :] - v[:, :-1, :]
            du_dx = du_dx[:, :-1, :]
            dv_dy = dv_dy[:, :, :-1]
            div = du_dx + dv_dy
            total = total + (div ** 2).mean()
            count += 1
        return total / max(count, 1)


class VorticityCurlLoss(nn.Module):
    """
    旋度一致性损失: 鼓励从u/v计算得到的旋度与地转旋度接近

    当数据中有vo通道时: vo_pred ≈ ∂v/∂x - ∂u/∂y
    当数据中无vo通道时: 计算相对旋度作为正则项
    """

    def __init__(self, data_cfg):
        super().__init__()
        self.forecast_steps = data_cfg.forecast_steps
        self.num_channels = data_cfg.num_channels
        pl_vars = data_cfg.pressure_level_vars
        n_pl = len(data_cfg.pressure_levels)

        # 只在有vo通道时才进行vo一致性比较
        if "vo" in pl_vars:
            vo_idx = pl_vars.index("vo")
            u_idx = pl_vars.index("u")
            v_idx = pl_vars.index("v")
            triplets = []
            for t_step in range(self.forecast_steps):
                base = t_step * self.num_channels
                for lev in range(n_pl):
                    u_ch = base + u_idx * n_pl + lev
                    v_ch = base + v_idx * n_pl + lev
                    vo_ch = base + vo_idx * n_pl + lev
                    triplets.append((u_ch, v_ch, vo_ch))
            self.triplets = triplets
        else:
            # 无vo通道：计算u/v旋度作为正则（鼓励接近地转旋度）
            self.triplets = None

    def forward(self, x0_pred: torch.Tensor) -> torch.Tensor:
        total = 0.0
        count = 0

        if self.triplets is not None:
            # 有vo通道：vo_pred ≈ ∂v/∂x - ∂u/∂y
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
                total = total + F.mse_loss(vo_inner, curl_from_wind)
                count += 1
        else:
            # 无vo通道：计算相对旋度作为正则项
            pl_vars = ["u", "v", "z"]
            n_pl = len(pl_vars)
            u_idx = pl_vars.index("u")
            v_idx = pl_vars.index("v")
            for t_step in range(self.forecast_steps):
                base = t_step * self.num_channels
                for lev in range(n_pl):
                    u_ch = base + u_idx * n_pl + lev
                    v_ch = base + v_idx * n_pl + lev
                    u = x0_pred[:, u_ch]
                    v = x0_pred[:, v_ch]
                    dv_dx = v[:, :, 1:] - v[:, :, :-1]
                    du_dy = u[:, 1:, :] - u[:, :-1, :]
                    dv_dx = dv_dx[:, :-1, :]
                    du_dy = du_dy[:, :, :-1]
                    curl = dv_dx - du_dy
                    total = total + curl.pow(2).mean()
                    count += 1

        return total / max(count, 1)


class GeostrophicBalanceLoss(nn.Module):
    """
    Geostrophic balance loss in NORMALIZED space.

    u_geo = -(g/f) * dz/dy
    v_geo =  (g/f) * dz/dx

    Requires per-batch center latitude because typhoon patches are extracted
    at different latitudes across the domain (Western Pacific, 0°-60°N).
    The Coriolis parameter f = 2Ωsin(lat) varies ~4× across this range.

    Inputs (x1_pred) are in NORMALIZED space. They are denormalized to
    physical units (m/s for u/v, m²/s² for z) before computing the loss.

    Args:
        norm_std: training-set std tensor shape (9,) — channels [0:3]=u, [3:6]=v, [6:9]=z
        norm_mean: training-set mean tensor shape (9,)
        grid_size: spatial grid dimension (40)
        lat_res: grid resolution in degrees (0.25)
        weight: loss scale factor
    """

    def __init__(
        self,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        grid_size: int = 40,
        lat_res: float = 0.25,
        weight: float = 0.01,
    ):
        super().__init__()
        self.weight = weight
        self.grid_size = grid_size
        self.lat_res = lat_res
        self.g = 9.80665
        self.Omega = 7.2921e-5
        self.R = 6.371e6
        self.dy = self.R * math.pi / 180 * lat_res  # meters per grid row (scalar)

        # Register norm stats: [u_850,u_500,u_250, v_850,v_500,v_250, z_850,z_500,z_250]
        self.register_buffer('norm_mean', norm_mean.float())
        self.register_buffer('norm_std', norm_std.float())

    def _denorm_single(self, x: torch.Tensor, ch: int) -> torch.Tensor:
        """Denormalize a single channel (B,1,H,W) to physical units."""
        return x * self.norm_std[ch] + self.norm_mean[ch]

    def forward(
        self,
        x1_pred: torch.Tensor,
        center_lats: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x1_pred: (B, 9, H, W) — normalized predicted state
            center_lats: (B,) — center latitude in degrees per sample; defaults to 20°N
        Returns:
            Scalar loss
        """
        B, C, H, W = x1_pred.shape
        device = x1_pred.device

        if center_lats is None:
            center_lats = torch.full((B,), 20.0, device=device)

        # Build per-batch Coriolis parameter f grid
        # Patch spans grid_size * lat_res degrees around center
        half_span = (self.grid_size * self.lat_res) / 2.0  # 5.0° for 40×0.25°
        lat_offsets = torch.linspace(-half_span, half_span, self.grid_size, device=device)
        # lat per row: (B, H) — varies by latitude row
        lats = center_lats.unsqueeze(1) + lat_offsets.unsqueeze(0)   # (B, H)
        f = 2.0 * self.Omega * torch.sin(torch.deg2rad(lats))         # (B, H)
        f = torch.where(f.abs() < 1e-5, f.sign().clamp(min=1) * 1e-5, f)  # guard near-equator
        f = f.unsqueeze(1).unsqueeze(3)  # (B, 1, H, 1)

        # dx varies by latitude (cosine factor); shape (B, 1, H, 1)
        dx = self.R * torch.cos(torch.deg2rad(lats)) * (math.pi / 180) * self.lat_res
        dx = dx.unsqueeze(1).unsqueeze(3)  # (B, 1, H, 1)

        # Channel layout: [u_850,u_500,u_250, v_850,v_500,v_250, z_850,z_500,z_250]
        pressure_levels = [0, 1, 2]

        total = torch.tensor(0.0, device=device)
        count = 0

        for lev in pressure_levels:
            u_norm = x1_pred[:, lev:lev+1]       # (B,1,H,W), normalized
            v_norm = x1_pred[:, 3+lev:4+lev]     # (B,1,H,W), normalized
            z_norm = x1_pred[:, 6+lev:7+lev]     # (B,1,H,W), normalized

            # Denormalize to physical units
            u_phys = self._denorm_single(u_norm, lev)
            v_phys = self._denorm_single(v_norm, 3 + lev)
            z_phys = self._denorm_single(z_norm, 6 + lev)

            # Central-difference gradients on inner region (H-2, W-2)
            # dz/dx: central diff along W axis
            dzdx = (z_phys[:, :, 1:-1, 2:] - z_phys[:, :, 1:-1, :-2]) \
                   / (2.0 * dx[:, :, 1:-1, :] + 1e-8)  # (B,1,H-2,W-2)
            # dz/dy: central diff along H axis
            dzdy = (z_phys[:, :, 2:, 1:-1] - z_phys[:, :, :-2, 1:-1]) \
                   / (2.0 * self.dy + 1e-8)             # (B,1,H-2,W-2)

            # f on inner H region: (B,1,H-2,1)
            f_inner = f[:, :, 1:-1, :]

            # Geostrophic wind (physical units)
            v_geo = (self.g / f_inner) * dzdx
            u_geo = (-self.g / f_inner) * dzdy

            # Wind residual on inner region, normalized by channel std
            u_std = self.norm_std[lev].clamp(min=1.0)
            v_std = self.norm_std[3 + lev].clamp(min=1.0)
            residual = (
                ((u_phys[:, :, 1:-1, 1:-1] - u_geo) / u_std) ** 2
                + ((v_phys[:, :, 1:-1, 1:-1] - v_geo) / v_std) ** 2
            )
            total = total + residual.mean()
            count += 1

        return self.weight * total / max(count, 1)


class ChannelWeightedMSE(nn.Module):
    """
    Scale-invariant per-channel weighted loss.

    Each channel's error is normalized by its target std (detached), so all channels
    contribute equally in raw prediction-error space regardless of their data magnitude.
    Channel weights then act as explicit importance multipliers (optional — set to 1.0
    for purely scale-invariant loss).

    Fixes: z channels with large velocity magnitudes no longer dominate loss purely
    because of scale — weights control relative importance explicitly.
    """

    def __init__(
        self,
        channel_weights: torch.Tensor,
        pressure_level_weights: torch.Tensor = None,
        use_normalized: bool = True,
        norm_std: torch.Tensor = None,
    ):
        super().__init__()
        w = channel_weights / channel_weights.mean()
        if pressure_level_weights is not None:
            w = w * (pressure_level_weights / pressure_level_weights.mean())
        self.register_buffer("weights", w)
        self.use_normalized = use_normalized

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W)
            target: (B, C, H, W)
        Returns:
            scalar loss
        """
        C = pred.shape[1]
        w = self.weights[:C]  # (C,)

        # Per-channel RMSE (spatial mean over B×H×W)
        diff_sq = (pred - target) ** 2
        error_sq_per_ch = diff_sq.mean(dim=(0, 2, 3))   # (C,)
        error_rms_per_ch = torch.sqrt(error_sq_per_ch + 1e-8)

        if self.use_normalized:
            # Normalize by target std per channel — makes loss scale-invariant.
            # Detach target_std so gradients only flow through pred (not target).
            target_std_per_ch = target.std(dim=(0, 2, 3)).detach().clamp(min=1e-4)
            normalized_error = error_rms_per_ch / target_std_per_ch
            # Apply channel weights on top
            return (w * normalized_error).mean()

        # Fallback: plain weighted MSE (no normalization)
        return (w * error_sq_per_ch).mean()


# ============================================================
# Diffusion调度器
# ============================================================

class DiffusionScheduler:
    """
    扩散过程调度器: Cosine Schedule + DDIM采样
    支持eps-prediction和v-prediction两种模式
    """

    def __init__(
        self,
        num_steps: int = 1000,
        schedule: str = "cosine",
        s: float = 0.008,
        ddim_steps: int = 50,
        clamp_range: Tuple[float, float] = None,  # was (-5.0, 5.0) — kills z variance at inference
        prediction_type: str = "v",
    ):
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

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_noisy, noise

    def compute_v_target(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x_start

    def predict_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        x0 = (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha.clamp(min=1e-8)
        return x0

    def predict_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_one_minus_alpha * x_t + sqrt_alpha * v

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: Tuple[int, ...],
        device: torch.device,
        eta: float = 0.0,
        z_channel_indices: Optional[List[int]] = None,
        z_clamp_range: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
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


# ============================================================
# 统一模型 (FM + DM)
# ============================================================

class UnifiedModel(nn.Module):
    """
    统一模型 — Flow Matching和Diffusion双模式

    使用方法:
        model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="fm")
        model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="dm")

    FM模式:
        - 训练: forward_fm() — 线性路径 x_t = (1-t)*x_0 + t*x_1, 预测v = x_1 - x_0
        - 推理: sample_fm() — Euler ODE求解器

    DM模式:
        - 训练: forward_dm() — 扩散路径, 预测噪声ε或速度v
        - 推理: sample_dm() — DDIM采样
    """

    def __init__(
        self,
        model_cfg,
        data_cfg,
        train_cfg=None,
        method: str = "fm",
    ):
        super().__init__()
        self.method = method  # "fm" or "dm"

        self.dit = UnifiedDiT(
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
            num_var_groups=getattr(model_cfg, 'num_var_groups', 3),
            history_steps=data_cfg.history_steps,
            use_temporal_agg=True,
        )
        self.cond_encoder = self.dit.cond_encoder

        # Diffusion调度器 (DM模式专用)
        self.scheduler = DiffusionScheduler(
            num_steps=getattr(model_cfg, 'num_diffusion_steps', 1000),
            schedule=getattr(model_cfg, 'noise_schedule', 'cosine'),
            ddim_steps=getattr(model_cfg, 'ddim_sampling_steps', 50),
            prediction_type=getattr(model_cfg, 'prediction_type', 'v'),
        )

        # 物理损失
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
            # use_norm=True: scale-invariant loss (normalize by target std, detached).
            # This makes channel weights the ONLY control over relative importance.
            use_norm = getattr(train_cfg, 'use_channel_wise_normalized_loss', True)

            self.channel_mse = ChannelWeightedMSE(
                weights,
                pressure_level_weights=pl_weights,
                use_normalized=use_norm,
            )
        else:
            self.channel_mse = None

        # 地转平衡损失
        if train_cfg is not None and getattr(train_cfg, 'geostrophic_weight', 0) > 0:
            norm_mean = norm_std = None
            stats_path = getattr(data_cfg, 'norm_stats_path', '')
            if stats_path and os.path.exists(stats_path):
                _stats = torch.load(stats_path, map_location='cpu', weights_only=True)
                norm_mean = _stats['mean'].float()
                norm_std = _stats['std'].float()
            self.geo_loss = GeostrophicBalanceLoss(
                norm_mean=norm_mean,
                norm_std=norm_std,
                grid_size=data_cfg.grid_size,
                lat_res=data_cfg.lat_res,
                weight=1.0,  # trainer's w_geo controls the scale; this is the raw loss
            )
        else:
            self.geo_loss = None

        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.z_channel_indices = data_cfg.z_channel_indices

        # Z-Predictor auxiliary network (Option 2)
        self.z_predictor = None
        if train_cfg is not None and getattr(train_cfg, 'use_z_predictor', False):
            z_d_model = getattr(train_cfg, 'z_predictor_d_model', 128)
            self.z_predictor = ZPredictor(d_model=z_d_model)
            z_pred_weight = getattr(train_cfg, 'z_predictor_weight', 1.0)
            self.z_predictor_weight = z_pred_weight
            self.z_channel_weight_override = getattr(train_cfg, 'z_channel_weight_override', 1.0)

    def forward_fm(self, condition: torch.Tensor, target: torch.Tensor,
                   batch: Dict = None) -> Dict[str, torch.Tensor]:
        """
        Flow Matching训练前向传播

        路径: x_t = (1-t)*x_0 + t*x_1, t ∈ [0,1]
        x_0 = condition的最后一个时间步 (当前状态)
        x_1 = target (预报目标)
        目标: v = x_1 - x_0 (从当前状态到预报目标的速度)
        预测: v_pred = model(x_t, t, condition)
        """
        device = target.device
        B, TC, H, W = condition.shape
        C = self.data_cfg.num_channels
        T = self.data_cfg.history_steps  # number of history timesteps

        # x_0 = condition的最后一个时间步 (当前状态, 9ch)
        x_0 = condition[:, -C:, :, :]
        # x_1 = target (预报目标, 9ch)
        x_1 = target

        # 采样时间步
        t = torch.sigmoid(torch.randn(B, device=device))
        t = torch.clamp(t, min=1e-5, max=1 - 1e-5)
        t_reshaped = t.reshape(-1, 1, 1, 1)

        # 线性插值路径
        x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1
        target_v = x_1 - x_0  # velocity: from current to forecast

        # 路径扰动 (Scheduled Sampling)
        if self.training and self.train_cfg is not None:
            pert_prob = getattr(self.train_cfg, 'path_perturb_prob', 0.0)
            if pert_prob > 0 and torch.rand(1, device=device).item() < pert_prob:
                pert_sigma = getattr(self.train_cfg, 'path_perturb_sigma', 0.05)
                x_t = x_t + torch.randn_like(x_t) * pert_sigma

        v_pred = self.dit(x_t, t, condition)

        # 速度场clamp
        if self.train_cfg is not None and self.train_cfg.velocity_clamp is not None:
            v_pred = v_pred.clamp(*self.train_cfg.velocity_clamp)

        # 计算损失
        if self.channel_mse is not None:
            loss_mse = self.channel_mse(v_pred, target_v)
        else:
            loss_mse = F.mse_loss(v_pred, target_v)

        if self.train_cfg is not None and self.train_cfg.velocity_loss_scale != 1.0:
            loss_mse = loss_mse * self.train_cfg.velocity_loss_scale

        x1_pred = x_t + (1 - t_reshaped) * v_pred

        if self.channel_mse is not None:
            loss_x1 = self.channel_mse(x1_pred, target)
        else:
            loss_x1 = F.mse_loss(x1_pred, target)

        # Detached x0 estimate — useful for diagnostics but detached to avoid
        # double-backprop through the physics losses that also use x1_pred
        with torch.no_grad():
            x0_est = x_t - t_reshaped * v_pred

        loss_div_raw = self.div_loss(x1_pred)
        loss_div = torch.clamp(loss_div_raw, max=10.0)
        loss_sol_raw = self.sol_loss(v_pred)
        loss_sol = torch.clamp(loss_sol_raw, max=10.0)

        loss_curl_raw = (
            self.curl_loss(x1_pred) if self.curl_loss is not None
            else torch.tensor(0.0, device=device)
        )
        loss_curl = (
            torch.clamp(loss_curl_raw, max=10.0) if self.curl_loss is not None
            else loss_curl_raw
        )

        loss_geo_raw = (
            self.geo_loss(x1_pred, center_lats=batch.get("center_lats") if batch else None)
            if self.geo_loss is not None
            else torch.tensor(0.0, device=device)
        )
        loss_geo = torch.clamp(loss_geo_raw, max=10.0) if self.geo_loss is not None else loss_geo_raw

        loss_z_pred = torch.tensor(0.0, device=device)

        # ── Z-Predictor auxiliary branch (Option 2) ──────────────────────
        # ZPredictor learns wind→z mapping via geostrophic balance.
        # It takes u/v from x_t and predicts z channels; its loss is blended
        # with the main model's z MSE so z gradients get ~5× stronger signal.
        if self.z_predictor is not None:
            # x_t: (B, 9, H, W) = [u_850,v_850,u_500,v_500,u_250,v_250, z_850,z_500,z_250]
            # uv for ZPredictor: (B, 6, H, W) = [u_850,v_850, u_500,v_500, u_250,v_250]
            uv_t = torch.cat([x_t[:, 0:2], x_t[:, 2:4], x_t[:, 4:6]], dim=1)
            # z_gt: (B, 3, H, W) = [z_850, z_500, z_250]
            z_gt = x_1[:, 6:9]
            # ZPredictor takes x_0 (current state) for u/v, not x_t
            uv_x0 = torch.cat([x_0[:, 0:2], x_0[:, 2:4], x_0[:, 4:6]], dim=1)
            z_pred = self.z_predictor(uv_x0, center_lats=batch.get("center_lats") if batch else None)
            loss_z_pred = F.mse_loss(z_pred, z_gt)

            # Blend: replace z channels in x1_pred with ZPredictor output
            # This gives the physics losses (div, curl, geo) a better z signal too
            z_blend_alpha = getattr(self.train_cfg, 'z_blend_alpha', 0.5)
            x1_pred_z = x1_pred.clone()
            x1_pred_z[:, 6:9] = z_blend_alpha * z_pred + (1 - z_blend_alpha) * x1_pred[:, 6:9]
            x1_pred = x1_pred_z

        return {
            "loss_mse": loss_mse,
            "loss_x1": loss_x1,
            "loss_div": loss_div,
            "loss_sol": loss_sol,
            "loss_curl": loss_curl,
            "loss_geo": loss_geo,
            "loss_z_pred": loss_z_pred,
            "v_pred": v_pred,
            "v_true": target_v,
            "x1_pred": x1_pred,
            "x0_est": x0_est,
            "t": t,
        }

    def forward_dm(self, condition: torch.Tensor, target: torch.Tensor,
                   batch: Dict = None) -> Dict[str, torch.Tensor]:
        """
        Diffusion训练前向传播

        路径: x_t = √(ᾱ_t)*x_0 + √(1-ᾱ_t)*ε
        目标: ε 或 v
        预测: eps_pred = model(x_t, t, condition)
        """
        device = target.device
        B = target.shape[0]
        self.scheduler.to(device)

        t = torch.randint(0, self.model_cfg.num_diffusion_steps, (B,), device=device)
        x_noisy, noise = self.scheduler.q_sample(target, t)
        model_output = self.dit(x_noisy, t, condition)

        prediction_type = self.scheduler.prediction_type
        if prediction_type == "v":
            v_target = self.scheduler.compute_v_target(target, t, noise)
            if self.channel_mse is not None:
                loss_mse = self.channel_mse(model_output, v_target)
            else:
                loss_mse = F.mse_loss(model_output, v_target)
            x0_pred = self.scheduler.predict_x0_from_v(x_noisy, t, model_output)
        else:
            if self.channel_mse is not None:
                loss_mse = self.channel_mse(model_output, noise)
            else:
                loss_mse = F.mse_loss(model_output, noise)
            x0_pred = self.scheduler.predict_x0_from_eps(x_noisy, t, model_output)

        loss_div_raw = self.div_loss(x0_pred)
        loss_div = torch.clamp(loss_div_raw, max=10.0)
        loss_sol_raw = self.sol_loss(model_output)
        loss_sol = torch.clamp(loss_sol_raw, max=10.0)

        loss_curl_raw = (
            self.curl_loss(x0_pred) if self.curl_loss is not None
            else torch.tensor(0.0, device=device)
        )
        loss_curl = torch.clamp(loss_curl_raw, max=10.0) if self.curl_loss is not None else loss_curl_raw

        loss_geo_raw = (
            self.geo_loss(x0_pred, center_lats=batch.get("center_lats") if batch else None)
            if self.geo_loss is not None
            else torch.tensor(0.0, device=device)
        )
        loss_geo = torch.clamp(loss_geo_raw, max=10.0) if self.geo_loss is not None else loss_geo_raw

        if self.channel_mse is not None:
            loss_x1 = self.channel_mse(x0_pred, target)
        else:
            loss_x1 = F.mse_loss(x0_pred, target)

        loss_z_pred = torch.tensor(0.0, device=device)

        # ── Z-Predictor auxiliary branch (Option 2) ──────────────────────
        if self.z_predictor is not None:
            x0 = condition[:, -self.data_cfg.num_channels:, :, :]
            uv_x0 = torch.cat([x0[:, 0:2], x0[:, 2:4], x0[:, 4:6]], dim=1)
            z_gt = target[:, 6:9]
            z_pred = self.z_predictor(uv_x0, center_lats=batch.get("center_lats") if batch else None)
            loss_z_pred = F.mse_loss(z_pred, z_gt)
            z_blend_alpha = getattr(self.train_cfg, 'z_blend_alpha', 0.5)
            x0_pred_z = x0_pred.clone()
            x0_pred_z[:, 6:9] = z_blend_alpha * z_pred + (1 - z_blend_alpha) * x0_pred[:, 6:9]
            x0_pred = x0_pred_z

        return {
            "loss_mse": loss_mse,
            "loss_x1": loss_x1,
            "loss_div": loss_div,
            "loss_sol": loss_sol,
            "loss_curl": loss_curl,
            "loss_geo": loss_geo,
            "loss_z_pred": loss_z_pred,
            "eps_pred": model_output,
            "eps_true": noise,
            "x0_pred": x0_pred,
            "x1_pred": x0_pred,
            "t": t.float() / self.model_cfg.num_diffusion_steps,
        }

    def forward(self, condition: torch.Tensor, target: torch.Tensor,
                batch: Dict = None) -> Dict[str, torch.Tensor]:
        """根据method自动选择FM或DM前向"""
        if self.method == "fm":
            return self.forward_fm(condition, target, batch=batch)
        else:
            return self.forward_dm(condition, target, batch=batch)

    def _prepare_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """
        预处理条件输入，将展平格式转为模型所需格式

        dataset返回: condition (B, T*C, H, W) — 时间维展平到通道维
        模型需要:   (B, T, C, H, W) 或 (B, cond_channels, H, W)

        策略:
        - 如果已经是5D，直接返回
        - 如果是4D，检查cond_channels是否等于T*C:
            - 是: reshape为5D，让encoder的Conv3D处理
            - 否: 已经是单帧条件，直接返回
        """
        if condition.ndim == 5:
            return condition

        B, TC, H, W = condition.shape
        expected_C = self.data_cfg.num_channels
        T = TC // expected_C
        C = expected_C

        if T > 1 and TC == T * C:
            # reshape: (B, T*C, H, W) -> (B, T, C, H, W)
            cond_5d = condition.view(B, T, C, H, W)
            return cond_5d

        # 单帧条件 (T=1)，直接返回
        return condition

    @torch.no_grad()
    def sample_fm(
        self,
        condition: torch.Tensor,
        device: torch.device,
        euler_steps: int = 4,
        euler_mode: str = "midpoint",
        clamp_range: Tuple[float, float] = None,  # was (-5.0, 5.0) — kills z variance at inference
        z_clamp_range: Tuple[float, float] = None,  # was (-3.0, 3.0) — kills z variance at inference
    ) -> torch.Tensor:
        """
        Flow Matching采样 — Forward ODE integration

        路径: x_t = (1-t)*x_0 + t*x_1, t ∈ [0,1]
        x_0 = condition的最后一个时间步 (当前状态)
        x_1 = 预报目标
        模型预测: v = x_1 - x_0

        采样: 从x_0出发，沿着速度场正向积分到x_1
        x_{t+dt} = x_t + v_pred * dt
        """
        C = self.data_cfg.num_channels
        B_orig = condition.shape[0]
        H, W = condition.shape[2], condition.shape[3]

        # x_0 = condition的最后一个时间步 (当前状态)
        x_t = condition[:, -C:, :, :].clone()
        t = 0.0
        dt = 1.0 / euler_steps

        # Use original condition (27ch) for DIT — cond_proc reshapes to wrong channels
        for step in range(euler_steps):
            t_next = t + dt

            t_tensor = torch.full((B_orig,), t, device=device, dtype=torch.float32)
            # 模型预测: v = x_1 - x_0 (从当前到预报目标的速度)
            v_pred = self.dit(x_t, t_tensor, condition)

            if euler_mode == "heun":
                x_temp = x_t + dt * v_pred
                t_tensor2 = torch.full((B_orig,), t_next, device=device, dtype=torch.float32)
                v_pred2 = self.dit(x_temp, t_tensor2, condition)
                x_next = x_t + dt / 2 * (v_pred + v_pred2)
            elif euler_mode == "midpoint":
                x_mid = x_t + dt / 2 * v_pred
                t_mid = t + dt / 2
                t_tensor2 = torch.full((B_orig,), t_mid, device=device, dtype=torch.float32)
                v_pred2 = self.dit(x_mid, t_tensor2, condition)
                x_next = x_t + dt * v_pred2
            else:  # euler
                x_next = x_t + dt * v_pred

            x_t = x_next
            if clamp_range is not None:
                x_t = torch.clamp(x_t, *clamp_range)
            # z clamp only at final step — don't kill variance during integration
            if step == euler_steps - 1:
                if self.z_channel_indices and z_clamp_range:
                    x_t[:, self.z_channel_indices] = x_t[:, self.z_channel_indices].clamp(*z_clamp_range)
            t = t_next

        return x_t

    @torch.no_grad()
    def sample_dm(
        self,
        condition: torch.Tensor,
        device: torch.device,
        ddim_steps: int = 50,
        clamp_range: Tuple[float, float] = None,  # was (-5.0, 5.0) — kills z variance at inference
        z_clamp_range: Tuple[float, float] = None,  # was (-3.0, 3.0) — kills z variance at inference
    ) -> torch.Tensor:
        """Diffusion采样 — DDIM"""
        self.scheduler.to(device)
        self.scheduler.ddim_steps = ddim_steps
        self.scheduler.clamp_range = clamp_range

        B_orig, _, H, W = condition.shape
        cond_proc = self._prepare_condition(condition)

        if cond_proc.ndim == 5:
            # _prepare_condition返回5D (B, T, C, H, W)，需要reshape为4D给scheduler
            B5, T_c, C, H5, W5 = cond_proc.shape
            cond_proc = cond_proc.view(B5, T_c * C, H5, W5)
        # cond_proc现在是标准4D格式 (B, cond_channels, H, W)

        shape = (B_orig, self.data_cfg.num_channels, H, W)
        return self.scheduler.ddim_sample(
            self.dit, cond_proc, shape, device,
            z_channel_indices=self.z_channel_indices if self.z_channel_indices else None,
            z_clamp_range=z_clamp_range,
        )

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        """根据method自动选择采样方式"""
        if self.method == "fm":
            return self.sample_fm(condition, device, **kwargs)
        else:
            return self.sample_dm(condition, device, **kwargs)


def create_model(model_cfg, data_cfg, train_cfg, method: str) -> UnifiedModel:
    """工厂函数: 创建FM或DM模型"""
    model = UnifiedModel(model_cfg, data_cfg, train_cfg, method=method)
    return model
