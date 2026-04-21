"""
ERA5-Diffusion 核心模型 (DiT 架构):
  1. ConditionEncoder   - 条件编码器 (Conv + Patch + Self-Attention)
  2. ERA5DiT            - DiT 去噪网络 (替代 ConditionalUNet)
  3. DiffusionScheduler - Cosine 噪声调度 + DDIM 采样
  4. ERA5DiffusionModel - 组合模型（训练 + 推理接口）
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .components import (
    SinusoidalTimeEmbedding,
    AdaLayerNorm,
    PatchEmbed,
    Unpatchify,
    DiTBlock,
)


# ============================================================
# 条件编码器
# ============================================================

class ConditionEncoder(nn.Module):
    """
    条件编码器 — 将历史气象条件编码为 token 序列

    输入: (B, cond_channels, 40, 40)  例如 (B, 60, 40, 40)

    流程:
      a) 2 层 Conv3×3 混合通道: cond_channels → D (在原始分辨率提取局部模式)
      b) Patch 化: (B, D, 40, 40) → stride=4 的 Conv → (B, 100, D)
      c) + 2D 可学习位置编码 (10×10 网格)
      d) n_cond_layers 层 Self-Attention → 建立条件 token 间的全局关联

    输出: (B, 100, D) 条件 token 序列
    """

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
    ):
        super().__init__()
        self.d_model = d_model
        num_patches = (grid_size // patch_size) ** 2  # 100

        # (a) 局部特征提取: 2 层 Conv3×3
        self.local_conv = nn.Sequential(
            nn.Conv2d(cond_channels, d_model, 3, padding=1),
            nn.GroupNorm(min(32, d_model), d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GroupNorm(min(32, d_model), d_model),
            nn.SiLU(),
        )

        # (b) Patch 化
        self.patch_embed = PatchEmbed(d_model, d_model, patch_size=patch_size)

        # (c) 2D 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # (d) Self-Attention 层
        self.layers = nn.ModuleList()
        for _ in range(n_cond_layers):
            self.layers.append(CondSelfAttnBlock(d_model, n_heads, ff_mult, dropout))

        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        condition: (B, cond_channels, H, W)
        返回: (B, num_patches, D) 条件 token 序列
        """
        # (a) 局部卷积
        h = self.local_conv(condition)  # (B, D, H, W)

        # (b) Patch 化
        tokens = self.patch_embed(h)  # (B, N, D)

        # (c) 加位置编码
        tokens = tokens + self.pos_embed

        # (d) Self-Attention
        for layer in self.layers:
            tokens = layer(tokens)

        return self.norm_out(tokens)


class CondSelfAttnBlock(nn.Module):
    """条件编码器中的 Self-Attention 块 (Pre-LN Transformer)"""

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

        # Self-Attention
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + self.dropout(self.proj(attn_out))

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x


# ============================================================
# ERA5DiT 去噪网络
# ============================================================

class ERA5DiT(nn.Module):
    """
    ERA5 DiT 去噪网络 — 替代 ConditionalUNet

    输入: x_noisy (B, in_ch, 40, 40), t (B,), condition (B, cond_ch, 40, 40)

    流程:
      1. 条件编码: ConditionEncoder(condition) → (B, 100, D)
      2. Patch 嵌入: Conv(in_ch, D, k=4, s=4) → (B, 100, D)
      3. + 2D 位置编码
      4. N 层 DiT Block (AdaLN + SelfAttn + CrossAttn + FFN)
      5. Final: AdaLN → Linear(D, in_ch × patch_size²)
      6. Unpatchify: (B, 100, in_ch×16) → (B, in_ch, 40, 40)

    输出: 预测噪声 ε̂ (B, in_ch, 40, 40)
    """

    def __init__(
        self,
        in_channels: int = 12,
        cond_channels: int = 60,
        d_model: int = 384,
        n_heads: int = 6,
        n_dit_layers: int = 12,
        n_cond_layers: int = 3,
        ff_mult: int = 4,
        patch_size: int = 4,
        grid_size: int = 40,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_size = patch_size
        num_patches = (grid_size // patch_size) ** 2  # 100
        time_emb_dim = d_model  # 时间嵌入维度 = D

        # 时间嵌入
        self.time_emb = SinusoidalTimeEmbedding(d_model, time_emb_dim)

        # 条件编码器
        self.cond_encoder = ConditionEncoder(
            cond_channels=cond_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_cond_layers=n_cond_layers,
            ff_mult=ff_mult,
            patch_size=patch_size,
            grid_size=grid_size,
            dropout=dropout,
        )

        # Patch 嵌入 (噪声输入)
        self.patch_embed = PatchEmbed(in_channels, d_model, patch_size=patch_size)

        # 2D 可学习位置编码 (噪声 tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # DiT Blocks
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

        # Final projection: AdaLN → Linear → Unpatchify
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * d_model),  # γ, β
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        self.final_linear = nn.Linear(d_model, in_channels * patch_size * patch_size)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

        # Unpatchify
        self.unpatchify = Unpatchify(in_channels, patch_size, grid_size)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_noisy: (B, in_ch, 40, 40) 加噪后的目标
        t: (B,) 扩散时间步
        condition: (B, cond_ch, 40, 40) 历史气象条件

        返回: (B, in_ch, 40, 40) 预测的噪声 ε̂
        """
        # 1. 时间嵌入
        time_emb = self.time_emb(t)  # (B, D)

        # 2. 条件编码
        cond_tokens = self.cond_encoder(condition)  # (B, N, D)

        # 3. Patch 嵌入 + 位置编码
        x = self.patch_embed(x_noisy)  # (B, N, D)
        x = x + self.pos_embed

        # 4. DiT Blocks
        for block in self.dit_blocks:
            x = block(x, time_emb, cond_tokens)

        # 5. Final projection
        final_params = self.final_adaLN(time_emb).unsqueeze(1)  # (B, 1, 2D)
        gamma, beta = final_params.chunk(2, dim=-1)
        x = (1 + gamma) * self.final_norm(x) + beta
        x = self.final_linear(x)  # (B, N, in_ch * p * p)

        # 6. Unpatchify
        x = self.unpatchify(x)  # (B, in_ch, H, W)

        return x


# ============================================================
# 扩散调度器
# ============================================================

class DiffusionScheduler:
    """
    扩散过程调度器: Cosine Schedule + DDIM 采样
    支持 eps-prediction 和 v-prediction 两种模式

    v-prediction 优势:
      x₀ = √(ᾱ_t) × x_t - √(1-ᾱ_t) × v  (无除法, 数值稳定)
      vs eps-prediction:
      x₀ = (x_t - √(1-ᾱ_t) × ε) / √(ᾱ_t)  (除法放大误差)
    """

    def __init__(
        self,
        num_steps: int = 1000,
        schedule: str = "cosine",
        s: float = 0.008,
        ddim_steps: int = 50,
        clamp_range: Tuple[float, float] = (-5.0, 5.0),
        prediction_type: str = "v",
    ):
        self.num_steps = num_steps
        self.ddim_steps = ddim_steps
        self.clamp_range = clamp_range
        self.prediction_type = prediction_type

        # 计算 ᾱ_t
        if schedule == "cosine":
            steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
            f_t = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = f_t / f_t[0]
            alphas_cumprod = alphas_cumprod.clamp(min=1e-8, max=1.0)
        elif schedule == "linear":
            beta_start, beta_end = 1e-4, 0.02
            betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            # 在前面加一个 1.0 用于后续计算
            alphas_cumprod = torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cumprod])
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # 转为 float32 注册
        self.alphas_cumprod = alphas_cumprod.float()  # (T+1,)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # DDIM 时间步子序列
        self.ddim_timesteps = self._make_ddim_timesteps(ddim_steps, num_steps)

    def _make_ddim_timesteps(self, ddim_steps: int, total_steps: int) -> torch.Tensor:
        """生成 DDIM 均匀子序列时间步"""
        step_size = total_steps // ddim_steps
        timesteps = torch.arange(0, total_steps, step_size)
        return timesteps.flip(0)  # 从大到小

    def to(self, device: torch.device):
        """将参数搬到指定设备"""
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
        """
        前向加噪: x_t = sqrt(ᾱ_t) · x_0 + sqrt(1-ᾱ_t) · ε

        x_start: (B, C, H, W)
        t: (B,) 时间步 (0 ~ num_steps-1)
        返回: (x_noisy, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)

        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_noisy, noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """从 x_t 和预测噪声 ε̂ 反推 x̂_0 (eps-prediction)"""
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        x0 = (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha.clamp(min=1e-8)
        return x0

    def compute_v_target(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """计算 v-prediction 的训练目标: v = √(ᾱ_t) × ε - √(1-ᾱ_t) × x₀"""
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x_start

    def predict_x0_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """从 x_t 和预测 v 恢复 x̂_0: x₀ = √(ᾱ_t) × x_t - √(1-ᾱ_t) × v (无除法!)"""
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v

    def predict_eps_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """从 x_t 和预测 v 恢复 ε: ε = √(1-ᾱ_t) × x_t + √(ᾱ_t) × v"""
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
        """
        DDIM 采样: 从纯噪声生成预测

        model: 去噪网络
        condition: (B, cond_ch, 40, 40)
        shape: 输出 shape (B, in_ch, 40, 40)
        eta: 0 = 完全确定性, 1 = DDPM
        z_channel_indices: z 通道索引列表 (用于逐通道 clamp)
        z_clamp_range: z 通道的更紧 clamp 范围

        返回: 预测的 x_0 (B, in_ch, 40, 40)
        """
        B = shape[0]
        x = torch.randn(shape, device=device)

        timesteps = self.ddim_timesteps  # 从大到小

        for i in tqdm(range(len(timesteps)), desc="DDIM采样", unit="步", leave=False):
            t_current = timesteps[i]
            t_batch = torch.full((B,), t_current, device=device, dtype=torch.long)

            # 模型输出 (eps or v, 取决于 prediction_type)
            model_output = model(x, t_batch, condition)

            # 恢复 x0 和 eps
            if self.prediction_type == "v":
                x0_pred = self.predict_x0_from_v(x, t_batch, model_output)
                eps_pred = self.predict_eps_from_v(x, t_batch, model_output)
            else:
                x0_pred = self.predict_x0_from_eps(x, t_batch, model_output)
                eps_pred = model_output
            # 全通道 clamp
            x0_pred = x0_pred.clamp(*self.clamp_range)
            # z 通道使用更紧的 clamp，防止自回归崩溃
            if z_channel_indices and z_clamp_range:
                x0_pred[:, z_channel_indices] = x0_pred[:, z_channel_indices].clamp(*z_clamp_range)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                # DDIM 更新
                alpha_t = self.alphas_cumprod[t_current + 1]
                alpha_next = self.alphas_cumprod[t_next + 1]

                sigma = eta * torch.sqrt(
                    (1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next)
                )

                # 预测方向
                pred_dir = torch.sqrt(1 - alpha_next - sigma ** 2) * eps_pred

                x = torch.sqrt(alpha_next) * x0_pred + pred_dir
                if sigma > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                x = x0_pred

        return x


# ============================================================
# 物理约束损失
# ============================================================

class DivergenceLoss(nn.Module):
    """
    风场散度正则化损失: L_div = ||∇·v||² = ||du/dx + dv/dy||²

    对预测的干净样本 x̂_0 中的 u/v 风场分量计算
    使用有限差分近似空间偏导数
    """

    def __init__(self, wind_pairs: List[Tuple[int, int]]):
        """
        wind_pairs: [(u_channel_idx, v_channel_idx), ...]
                    5 对: 10m 风场 + 4 个气压层
        """
        super().__init__()
        self.wind_pairs = wind_pairs

    def forward(self, x0_pred: torch.Tensor) -> torch.Tensor:
        """
        x0_pred: (B, C, 40, 40) 预测的干净样本
        返回: 标量损失
        """
        total_div = 0.0
        count = 0

        for u_idx, v_idx in self.wind_pairs:
            u = x0_pred[:, u_idx]  # (B, H, W)
            v = x0_pred[:, v_idx]

            # 有限差分: du/dx ≈ (u[:,1:] - u[:,:-1]), dv/dy ≈ (v[1:,:] - v[:-1,:])
            du_dx = u[:, :, 1:] - u[:, :, :-1]  # (B, H, W-1)
            dv_dy = v[:, 1:, :] - v[:, :-1, :]  # (B, H-1, W)

            # 对齐尺寸（取内部区域）
            du_dx = du_dx[:, :-1, :]  # (B, H-1, W-1)
            dv_dy = dv_dy[:, :, :-1]  # (B, H-1, W-1)

            divergence = du_dx + dv_dy
            total_div = total_div + (divergence ** 2).mean()
            count += 1

        return total_div / max(count, 1)


class VorticityCurlLoss(nn.Module):
    """
    旋度一致性约束: vo_pred ≈ ∂v/∂x - ∂u/∂y

    强制扩散模型预测的涡度场与风场在物理上自洽
    涡度不再独立学习，而是通过风场梯度来约束
    """

    def __init__(self, data_cfg):
        """
        根据 DataConfig 自动计算 u/v/vo 在目标张量中的通道索引
        """
        super().__init__()
        n_pl = len(data_cfg.pressure_levels)
        pl_vars = data_cfg.pressure_level_vars

        u_idx_in_pl = pl_vars.index("u")
        v_idx_in_pl = pl_vars.index("v")
        vo_idx_in_pl = pl_vars.index("vo")

        # 构建 (u_ch, v_ch, vo_ch) 三元组，每个气压层一组
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
        """
        x0_pred: (B, C, H, W) 预测的干净样本（归一化空间）
        返回: 旋度一致性损失（标量）
        """
        total_loss = 0.0
        count = 0

        for u_ch, v_ch, vo_ch in self.triplets:
            u = x0_pred[:, u_ch]   # (B, H, W)
            v = x0_pred[:, v_ch]
            vo = x0_pred[:, vo_ch]

            # 有限差分计算旋度: curl = ∂v/∂x - ∂u/∂y
            dv_dx = v[:, :, 1:] - v[:, :, :-1]  # (B, H, W-1)
            du_dy = u[:, 1:, :] - u[:, :-1, :]  # (B, H-1, W)

            # 对齐尺寸
            dv_dx = dv_dx[:, :-1, :]  # (B, H-1, W-1)
            du_dy = du_dy[:, :, :-1]  # (B, H-1, W-1)

            curl_from_wind = dv_dx - du_dy

            # vo 也裁剪到内部区域
            vo_inner = vo[:, :-1, :-1]  # (B, H-1, W-1) 左上对齐

            # L2 一致性损失
            total_loss = total_loss + F.mse_loss(vo_inner, curl_from_wind)
            count += 1

        return total_loss / max(count, 1)


class ChannelWeightedMSE(nn.Module):
    """
    逐通道加权 MSE 损失

    给精度差的通道更高权重，强迫模型关注 vo/850风/地面风
    权重在归一化空间中起作用，不受物理量级影响
    """

    def __init__(self, channel_weights: torch.Tensor):
        """
        channel_weights: (C,) 每个通道的权重，会被自动归一化使均值为 1.0
        """
        super().__init__()
        # 归一化使均值为 1.0（这样总 loss 量级不变）
        w = channel_weights / channel_weights.mean()
        self.register_buffer("weights", w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, C, H, W)
        返回: 加权 MSE 标量
        """
        C = pred.shape[1]
        w = self.weights[:C].reshape(1, C, 1, 1)  # (1, C, 1, 1)
        return (w * (pred - target) ** 2).mean()


# ============================================================
# 组合模型
# ============================================================

class ERA5DiffusionModel(nn.Module):
    """
    ERA5 条件扩散模型完整封装

    训练: forward() 返回 (loss_dict, eps_pred, eps_true, x0_pred)
    推理: sample() / autoregressive_predict() 返回预测结果
    """

    def __init__(self, model_cfg, data_cfg, train_cfg=None):
        super().__init__()

        # DiT 去噪网络
        self.dit = ERA5DiT(
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

        # 扩散调度器
        self.scheduler = DiffusionScheduler(
            num_steps=model_cfg.num_diffusion_steps,
            schedule=model_cfg.noise_schedule,
            ddim_steps=model_cfg.ddim_sampling_steps,
            prediction_type=getattr(model_cfg, 'prediction_type', 'eps'),
        )

        # 物理约束损失
        wind_pairs = data_cfg.get_wind_channel_indices()
        self.div_loss = DivergenceLoss(wind_pairs)

        # 旋度一致性约束 (vo = ∂v/∂x - ∂u/∂y)
        if "vo" in data_cfg.pressure_level_vars:
            self.curl_loss = VorticityCurlLoss(data_cfg)
        else:
            self.curl_loss = None

        # 逐通道加权 MSE
        if train_cfg is not None and train_cfg.use_channel_weights:
            weights = torch.tensor(train_cfg.channel_weights, dtype=torch.float32)
            # forecast_steps > 1 时需要重复权重
            if data_cfg.forecast_steps > 1:
                per_step = weights[:data_cfg.num_channels]
                weights = per_step.repeat(data_cfg.forecast_steps)
            self.channel_mse = ChannelWeightedMSE(weights)
        else:
            self.channel_mse = None

        self.model_cfg = model_cfg
        self.data_cfg = data_cfg

        # 预计算 z 通道索引（用于逐通道 clamp）
        if "z" in data_cfg.pressure_level_vars:
            z_var_idx = data_cfg.pressure_level_vars.index("z")
            n_levels = len(data_cfg.pressure_levels)
            self.z_channel_indices = list(range(
                z_var_idx * n_levels, (z_var_idx + 1) * n_levels
            ))
        else:
            self.z_channel_indices = []

    def forward(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向传播

        condition: (B, cond_ch, 40, 40)
        target: (B, in_ch, 40, 40) 干净的目标 x_0

        返回: {
            'loss_mse': MSE 噪声预测损失,
            'loss_div': 风场散度正则化损失,
            'eps_pred': 预测噪声,
            'eps_true': 真实噪声,
            'x0_pred': 反推的干净样本,
        }
        """
        device = target.device
        B = target.shape[0]

        # 确保调度器在正确设备上
        self.scheduler.to(device)

        # 随机采样时间步
        t = torch.randint(0, self.model_cfg.num_diffusion_steps, (B,), device=device)

        # 前向加噪
        x_noisy, noise = self.scheduler.q_sample(target, t)

        # 模型预测 (输出含义取决于 prediction_type)
        model_output = self.dit(x_noisy, t, condition)

        # 计算损失和恢复 x0
        prediction_type = self.scheduler.prediction_type
        if prediction_type == "v":
            # v-prediction: 目标 v = √(ᾱ_t)×ε - √(1-ᾱ_t)×x₀
            v_target = self.scheduler.compute_v_target(target, t, noise)
            if self.channel_mse is not None:
                loss_mse = self.channel_mse(model_output, v_target)
            else:
                loss_mse = F.mse_loss(model_output, v_target)
            x0_pred = self.scheduler.predict_x0_from_v(x_noisy, t, model_output)
        else:
            # eps-prediction (原始模式)
            if self.channel_mse is not None:
                loss_mse = self.channel_mse(model_output, noise)
            else:
                loss_mse = F.mse_loss(model_output, noise)
            x0_pred = self.scheduler.predict_x0_from_eps(x_noisy, t, model_output)

        # 物理约束损失
        loss_div_raw = self.div_loss(x0_pred)
        # Clamp div_loss 防止极端尖峰导致梯度爆炸
        # 原始 div_loss 可达数千, clamp 到 10.0 后乘以权重 0.0001 = 最大贡献 0.001
        loss_div = torch.clamp(loss_div_raw, max=10.0)

        # 旋度一致性损失
        loss_curl_raw = self.curl_loss(x0_pred) if self.curl_loss is not None else torch.tensor(0.0, device=device)
        loss_curl = torch.clamp(loss_curl_raw, max=10.0) if self.curl_loss is not None else loss_curl_raw

        return {
            "loss_mse": loss_mse,
            "loss_div": loss_div,
            "loss_curl": loss_curl,
            "eps_pred": model_output,  # 实际为 v 或 eps，取决于 prediction_type
            "eps_true": noise,
            "x0_pred": x0_pred,
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        device: torch.device,
        z_clamp_range: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        """
        DDIM 采样生成单步预测

        condition: (B, cond_ch, 40, 40)
        z_clamp_range: z 通道的更紧 clamp 范围（可选，覆盖 scheduler 默认值）
        返回: (B, in_ch, 40, 40) 预测的 x_0
        """
        self.scheduler.to(device)
        B = condition.shape[0]
        shape = (B, self.model_cfg.in_channels, self.data_cfg.grid_size, self.data_cfg.grid_size)
        return self.scheduler.ddim_sample(
            self.dit, condition, shape, device,
            z_channel_indices=self.z_channel_indices if self.z_channel_indices else None,
            z_clamp_range=z_clamp_range,
        )

    @torch.no_grad()
    def fast_sample(
        self,
        condition: torch.Tensor,
        device: torch.device,
        ddim_steps: int = 10,
    ) -> torch.Tensor:
        """
        快速 DDIM 采样 (用于训练中的 Scheduled Sampling)

        使用少量 DDIM 步数快速生成一个合理的预测，
        不追求精度，只需要近似模拟模型自回归推理时的预测分布

        condition: (B, cond_ch, 40, 40)
        ddim_steps: DDIM 采样步数 (10 步远快于推理的 50 步)
        返回: (B, in_ch, 40, 40)
        """
        self.scheduler.to(device)
        B = condition.shape[0]
        H = W = self.data_cfg.grid_size
        in_ch = self.model_cfg.in_channels
        shape = (B, in_ch, H, W)

        # 创建临时的 DDIM 时间步子序列 (不修改 scheduler 的状态)
        total_steps = self.scheduler.num_steps
        step_size = total_steps // ddim_steps
        timesteps = torch.arange(0, total_steps, step_size, device=device).flip(0)

        x = torch.randn(shape, device=device)

        for i in range(len(timesteps)):
            t_current = timesteps[i]
            t_batch = torch.full((B,), t_current, device=device, dtype=torch.long)

            model_output = self.dit(x, t_batch, condition)

            # 恢复 x0 和 eps
            if self.scheduler.prediction_type == "v":
                x0_pred = self.scheduler.predict_x0_from_v(x, t_batch, model_output)
                eps_pred = self.scheduler.predict_eps_from_v(x, t_batch, model_output)
            else:
                x0_pred = self.scheduler.predict_x0_from_eps(x, t_batch, model_output)
                eps_pred = model_output
            x0_pred = x0_pred.clamp(*self.scheduler.clamp_range)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t = self.scheduler.alphas_cumprod[t_current + 1]
                alpha_next = self.scheduler.alphas_cumprod[t_next + 1]
                pred_dir = torch.sqrt(1 - alpha_next) * eps_pred
                x = torch.sqrt(alpha_next) * x0_pred + pred_dir
            else:
                x = x0_pred

        return x
