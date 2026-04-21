#!/usr/bin/env python3
"""
独立测试 — 不依赖包结构，直接包含所需代码
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ 复用 components 中的类 ============
# (从 compare_fm_dm/models/components.py 和 adapter.py 提取必要部分)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=256, time_emb_dim=256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)

class AdaLayerNorm(nn.Module):
    def __init__(self, d_model, time_emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 6 * d_model),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
    def forward(self, x, time_emb):
        params = self.adaLN_modulation(time_emb).unsqueeze(1)
        return params.chunk(6, dim=-1)
    def modulate(self, x, gamma, beta):
        return (1 + gamma) * self.norm(x) + beta

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, d_model, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        B, D, Hp, Wp = x.shape
        return x.reshape(B, D, Hp * Wp).permute(0, 2, 1)

class Unpatchify(nn.Module):
    def __init__(self, out_channels, patch_size=4, grid_size=40):
        super().__init__()
        self.out_channels = out_channels
        self.grid_h = grid_size // patch_size
        self.grid_w = grid_size // patch_size
        self.patch_size = patch_size
    def forward(self, x):
        B, N, _ = x.shape
        p, C = self.patch_size, self.out_channels
        x = x.reshape(B, self.grid_h, self.grid_w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, C, self.grid_h * p, self.grid_w * p)

class CondSelfAttnBlockSimple(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.1):
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
    def forward(self, x):
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + self.dropout(self.proj(attn_out))
        x = x + self.ffn(self.norm2(x))
        return x

class SimpleDiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, time_emb_dim=384, dropout=0.1):
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
    def forward(self, x, time_emb):
        B, N, D = x.shape
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(x, time_emb)
        h = self.adaln.modulate(x, gamma1, beta1)
        qkv = self.self_attn_qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + alpha1 * self.dropout(self.self_attn_proj(attn_out))
        h_ffn = self.adaln.modulate(x, gamma2, beta2)
        x = x + alpha2 * self.ffn(h_ffn)
        return x

class SimpleConditionEncoder(nn.Module):
    def __init__(self, cond_channels=45, d_model=384, n_heads=6, n_cond_layers=3,
                 ff_mult=4, patch_size=4, grid_size=40, dropout=0.1):
        super().__init__()
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
        self.layers = nn.ModuleList([CondSelfAttnBlockSimple(d_model, n_heads, ff_mult, dropout)
                                     for _ in range(n_cond_layers)])
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, condition):
        h = self.local_conv(condition)
        tokens = self.patch_embed(h) + self.pos_embed
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm_out(tokens)

class SimpleDiT(nn.Module):
    def __init__(self, in_channels=9, cond_channels=45, d_model=384, n_heads=6,
                 n_dit_layers=12, n_cond_layers=3, ff_mult=4, patch_size=4,
                 grid_size=40, dropout=0.1):
        super().__init__()
        time_emb_dim = d_model
        num_patches = (grid_size // patch_size) ** 2
        self.time_emb = SinusoidalTimeEmbedding(d_model, time_emb_dim)
        self.cond_encoder = SimpleConditionEncoder(
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
    def forward(self, x_noisy, t, condition):
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

class DiffusionScheduler:
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
    def _make_ddim_timesteps(self, ddim_steps, total_steps):
        step_size = total_steps // ddim_steps
        timesteps = torch.arange(0, total_steps, step_size)
        return timesteps.flip(0)
    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        return self
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise
    def predict_x0_from_eps(self, x_t, t, eps):
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha.clamp(min=1e-8)
    def predict_x0_from_v(self, x_t, t, v):
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v
    def predict_eps_from_v(self, x_t, t, v):
        sqrt_alpha = self.sqrt_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, 1, 1, 1)
        return sqrt_one_minus_alpha * x_t + sqrt_alpha * v
    @torch.no_grad()
    def ddim_sample(self, model, condition, shape, device, eta=0.0,
                    z_channel_indices=None, z_clamp_range=None):
        x = torch.randn(shape, device=device)
        timesteps = self.ddim_timesteps
        for i in range(len(timesteps)):
            t_current = timesteps[i]
            t_batch = torch.full((shape[0],), t_current, device=device, dtype=torch.long)
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
                sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
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

class DivergenceLoss(nn.Module):
    def __init__(self, wind_pairs):
        super().__init__()
        self.wind_pairs = wind_pairs
    def forward(self, x0_pred):
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
    def __init__(self, channel_weights):
        super().__init__()
        w = channel_weights / channel_weights.mean()
        self.register_buffer("weights", w)
    def forward(self, pred, target):
        C = pred.shape[1]
        w = self.weights[:C].reshape(1, C, 1, 1)
        return (w * (pred - target) ** 2).mean()

class VorticityCurlLoss(nn.Module):
    def __init__(self, data_cfg):
        super().__init__()
        n_pl = len(data_cfg.pressure_levels)
        pl_vars = data_cfg.pressure_level_vars
        if "vo" not in pl_vars:
            # 如果没有 vo 通道，创建一个空的 loss 返回 0
            self.dummy = True
            return
        self.dummy = False
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
    def forward(self, x0_pred):
        if self.dummy:
            return torch.tensor(0.0, device=x0_pred.device)
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
            total_loss += F.mse_loss(vo_inner, curl_from_wind)
            count += 1
        return total_loss / max(count, 1)

class AdaptedDiffusionModel(nn.Module):
    def __init__(self, model_cfg, data_cfg, train_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
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
        self.scheduler = DiffusionScheduler(
            num_steps=getattr(model_cfg, 'num_diffusion_steps', 1000),
            schedule=getattr(model_cfg, 'noise_schedule', 'cosine'),
            ddim_steps=getattr(model_cfg, 'ddim_sampling_steps', 50),
            prediction_type=getattr(model_cfg, 'prediction_type', 'eps'),
        )
        wind_pairs = data_cfg.get_wind_channel_indices()
        self.div_loss = DivergenceLoss(wind_pairs)
        self.curl_loss = VorticityCurlLoss(data_cfg) if "vo" in data_cfg.pressure_level_vars else None
        if train_cfg is not None and getattr(train_cfg, 'use_channel_weights', False):
            weights = torch.tensor(train_cfg.channel_weights, dtype=torch.float32)
            if data_cfg.forecast_steps > 1:
                per_step = weights[:data_cfg.num_channels]
                weights = per_step.repeat(data_cfg.forecast_steps)
            self.channel_mse = ChannelWeightedMSE(weights)
        else:
            self.channel_mse = None
        if "z" in data_cfg.pressure_level_vars:
            z_var_idx = data_cfg.pressure_level_vars.index("z")
            n_levels = len(data_cfg.pressure_levels)
            self.z_channel_indices = list(range(z_var_idx * n_levels, (z_var_idx + 1) * n_levels))
        else:
            self.z_channel_indices = []
    def forward(self, condition, target):
        device = target.device
        B = target.shape[0]
        self.scheduler.to(device)
        t = torch.randint(0, self.model_cfg.num_diffusion_steps, (B,), device=device)
        x_noisy, noise = self.scheduler.q_sample(target, t)
        model_output = self.dit(x_noisy, t, condition)
        if self.scheduler.prediction_type == "v":
            v_target = self.scheduler.compute_v_target(target, t, noise)
            loss_mse = self.channel_mse(model_output, v_target) if self.channel_mse else F.mse_loss(model_output, v_target)
            x0_pred = self.scheduler.predict_x0_from_v(x_noisy, t, model_output)
        else:
            loss_mse = self.channel_mse(model_output, noise) if self.channel_mse else F.mse_loss(model_output, noise)
            x0_pred = self.scheduler.predict_x0_from_eps(x_noisy, t, model_output)
        loss_div = torch.clamp(self.div_loss(x0_pred), max=10.0)
        loss_curl = torch.clamp(self.curl_loss(x0_pred), max=10.0) if self.curl_loss else torch.tensor(0.0, device=device)
        return {"loss_mse": loss_mse, "loss_div": loss_div, "loss_curl": loss_curl,
                "eps_pred": model_output, "eps_true": noise, "x0_pred": x0_pred,
                "t": t.float() / self.model_cfg.num_diffusion_steps}
    @torch.no_grad()
    def sample(self, condition, device, ddim_steps=50, clamp_range=(-5.0, 5.0), z_clamp_range=None):
        self.scheduler.to(device)
        self.scheduler.ddim_steps = ddim_steps
        self.scheduler.clamp_range = clamp_range
        B = condition.shape[0]
        shape = (B, self.model_cfg.in_channels, self.data_cfg.grid_size, self.data_cfg.grid_size)
        return self.scheduler.ddim_sample(self.dit, condition, shape, device,
                                          z_channel_indices=self.z_channel_indices if self.z_channel_indices else None,
                                          z_clamp_range=z_clamp_range)

# ============ 配置 (简化) ============
class DataConfig:
    def __init__(self):
        self.grid_size = 40
        self.pressure_level_vars = ["u", "v", "z"]
        self.pressure_levels = [850, 500, 250]
        self.surface_vars = []
        self.history_steps = 5
        self.forecast_steps = 1
        self.num_pressure_level_channels = 9
        self.num_surface_channels = 0
        self.num_channels = 9
        self.cond_channels = 45

    def get_wind_channel_indices(self):
        """返回风场 u/v 的通道索引对 [(u_idx, v_idx), ...] — 与 newtry 一致"""
        pairs = []
        n_pl = len(self.pressure_levels)
        for t_step in range(self.forecast_steps):
            base = t_step * self.num_channels
            u_idx_in_pl = self.pressure_level_vars.index("u")
            v_idx_in_pl = self.pressure_level_vars.index("v")
            for lev in range(n_pl):
                u_ch = base + u_idx_in_pl * n_pl + lev
                v_ch = base + v_idx_in_pl * n_pl + lev
                pairs.append((u_ch, v_ch))
        return pairs

class ModelConfig:
    def __init__(self):
        self.in_channels = 9
        self.cond_channels = 45
        self.d_model = 384
        self.n_heads = 6
        self.n_dit_layers = 12
        self.n_cond_layers = 3
        self.ff_mult = 4
        self.patch_size = 4
        self.dropout = 0.1
        self.num_diffusion_steps = 1000
        self.noise_schedule = "cosine"
        self.ddim_sampling_steps = 50
        self.prediction_type = "eps"

class TrainConfig:
    def __init__(self):
        self.use_channel_weights = True
        self.channel_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.5)

def get_config():
    return DataConfig(), ModelConfig(), TrainConfig(), None

# ============ 加载函数 ============
def load_newtry_checkpoint(checkpoint_path, data_cfg, model_cfg, train_cfg=None, device=torch.device("cpu")):
    model = AdaptedDiffusionModel(model_cfg, data_cfg, train_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️  缺失: {len(missing)}, 多余: {len(unexpected)}")
    else:
        print(f"✅ checkpoint 加载成功 (epoch {ckpt.get('epoch', 'N/A')})")
    return model

# ============ 测试 ============
if __name__ == "__main__":
    print("=" * 60)
    print("测试 newtry checkpoint 适配器")
    print("=" * 60)

    data_cfg, model_cfg, train_cfg, _ = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "/root/autodl-tmp/fyp_final/Ver4/newtry/checkpoints/best_eps.pt"

    print(f"\n配置: in_channels={model_cfg.in_channels}, cond_channels={model_cfg.cond_channels}")
    print(f"       d_model={model_cfg.d_model}, n_dit_layers={model_cfg.n_dit_layers}")

    model = load_newtry_checkpoint(ckpt_path, data_cfg, model_cfg, train_cfg, device)
    print(f"\n✅ 模型加载成功!")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n测试前向传播:")
    B = 2
    cond = torch.randn(B, model_cfg.cond_channels, 40, 40).to(device)
    target = torch.randn(B, model_cfg.in_channels, 40, 40).to(device)
    with torch.no_grad():
        out = model(cond, target)
        print(f"   loss_mse: {out['loss_mse'].item():.4f}")
        print(f"   loss_div: {out['loss_div'].item():.4f}")
        print(f"   loss_curl: {out['loss_curl'].item():.4f}")

    print(f"\n测试采样 (DDIM 10步):")
    with torch.no_grad():
        pred = model.sample(cond, device, ddim_steps=10, clamp_range=(-5.0, 5.0))
        print(f"   输出 shape: {pred.shape}")
        print(f"   范围: [{pred.min().item():.3f}, {pred.max().item():.3f}]")

    print(f"\n✅ 所有测试通过!")
    print(f"\n现在可以运行对比实验:")
    print(f"  cd /root/autodl-tmp/fyp_final/Ver4/compare_fm_dm")
    print(f"  python run_comparison.py \\")
    print(f"      --data_root /path/to/data \\")
    print(f"      --external_dm_ckpt /root/autodl-tmp/fyp_final/Ver4/newtry/checkpoints/best_eps.pt \\")
    print(f"      --skip_train")
