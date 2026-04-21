"""
Flow Matching vs Diffusion 对比实验配置

统一 FM 和 DM 的架构配置，确保公平对比
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os


@dataclass
class DataConfig:
    """数据配置

    Year-based split (no data leakage):
      - Train:  1950-2016 (67 years)
      - Val:    2017-2018 (2 years) — hyperparameter tuning
      - Test:   2019-2021 (3 years) — held-out evaluation

    Normalization: (variable - mean) / std, computed over TRAINING SET ONLY.
    """
    # Data directory
    data_root: str = "/root/autodl-tmp/fyp_final/era5_data"
    era5_dir: str = "/root/autodl-tmp/fyp_final/era5_data"
    preprocessed_dir: str = "/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5"

    # CSV 统计文件 (用于划分台风)
    stats_csv: str = ""

    # 网格
    grid_size: int = 40

    # 气象变量 (9通道: u/v/z 各3个气压层)
    pressure_level_vars: List[str] = field(default_factory=lambda: ["u", "v", "z"])
    surface_vars: List[str] = field(default_factory=lambda: [])
    pressure_levels: List[int] = field(default_factory=lambda: [850, 500, 250])

    # 时间窗口
    history_steps: int = 5       # 历史步数 (5 × 3h = 15h)
    forecast_steps: int = 1      # 预测步数 (1 × 3h = 3h)
    time_interval_hours: int = 3

    # Year-based data split (no random — prevents data leakage)
    train_years: Tuple[int, int] = (1950, 2016)   # inclusive
    val_years:   Tuple[int, int] = (2017, 2018)  # inclusive
    test_years:  Tuple[int, int] = (2019, 2021)   # inclusive

    # DataLoader
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True

    # 归一化统计路径 — recomputed for new year-based split
    norm_stats_path: str = "/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5/norm_stats_year_split.pt"

    # 台风轨迹CSV — 用于 GeostrophicBalanceLoss 的中心纬度
    # Per-sample lat metadata enables physically correct f = 2Ωsin(lat) per typhoon
    track_csv_path: str = "/root/autodl-tmp/fyp_final/Ver4/Trajectory/processed_typhoon_tracks.csv"
    default_center_lat: float = 20.0   # fallback, Western Pacific typhoon belt

    # 地理参数 (用于物理损失)
    lat_range: Tuple[float, float] = (0.0, 60.0)
    lon_range: Tuple[float, float] = (95.0, 185.0)
    lat_res: float = 0.25
    lon_res: float = 0.25

    def __post_init__(self):
        if not self.stats_csv:
            self.stats_csv = os.path.join(self.data_root, "typhoon_organization_stats_1950_2021.csv")

        # 计算 z 通道索引 (用于 z-clamp)
        if "z" in self.pressure_level_vars:
            z_var_idx = self.pressure_level_vars.index("z")
            n_levels = len(self.pressure_levels)
            self.z_channel_indices = list(range(
                z_var_idx * n_levels, (z_var_idx + 1) * n_levels
            ))
        else:
            self.z_channel_indices = []

        # 变量名列表 (用于可视化标签)
        self.var_names = self.pressure_level_vars * len(self.pressure_levels) + self.surface_vars

    def year_based_split(self) -> Tuple[List[int], List[int], List[int]]:
        """Return year-based train/val/test typhoon year lists from filenames.

        Typhoon IDs encode the birth year in the first 4 digits (e.g. 2019001N09151).
        Train: years 1950-2016
        Val:   years 2017-2018  (hyperparameter tuning)
        Test:  years 2019-2021  (held-out evaluation)
        """
        train_y, val_y, test_y = self.train_years, self.val_years, self.test_years
        train_yrs = list(range(train_y[0], train_y[1] + 1))
        val_yrs   = list(range(val_y[0],   val_y[1]   + 1))
        test_yrs  = list(range(test_y[0],  test_y[1]  + 1))
        return train_yrs, val_yrs, test_yrs

    @property
    def num_pressure_level_channels(self) -> int:
        return len(self.pressure_level_vars) * len(self.pressure_levels)

    @property
    def num_surface_channels(self) -> int:
        return len(self.surface_vars)

    @property
    def num_channels(self) -> int:
        return self.num_pressure_level_channels + self.num_surface_channels

    @property
    def condition_channels(self) -> int:
        return self.num_channels * self.history_steps

    @property
    def target_channels(self) -> int:
        return self.num_channels * self.forecast_steps

    def get_wind_channel_indices(self) -> List[Tuple[int, int]]:
        """获取风场 u/v 的通道索引对 (用于散度损失)"""
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


@dataclass
class ModelConfig:
    """模型架构配置 — 统一 DiT 骨干"""
    d_model: int = 384
    n_heads: int = 6
    n_dit_layers: int = 12
    n_cond_layers: int = 3
    ff_mult: int = 4
    patch_size: int = 4
    grid_size: int = 40
    dropout: float = 0.1

    # 扩散参数 (DM 模式使用)
    num_diffusion_steps: int = 1000
    noise_schedule: str = "cosine"
    ddim_sampling_steps: int = 50

    # 预测目标类型
    # "eps": 预测噪声 (DM 标准)
    # "v": 预测速度 (DM 替代方案，数值更稳定，与FM的velocity prediction更公平)
    prediction_type: str = "v"

    # 输入/输出通道
    in_channels: int = 9
    cond_channels: int = 45

    # 时间嵌入缩放 (参考 DiT)
    time_embedding_scale: float = 1000.0

    # 分组卷积 (FM 模式专用)
    # CRITICAL: Must be False to match trained checkpoint architecture (temporal_conv3d, no grouped conv)
    use_grouped_conv: bool = False
    num_var_groups: int = 3


@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_epochs: int = 400

    # 优化器
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    # 学习率调度
    warmup_steps: int = 200
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6

    # 混合精度
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    # EMA
    ema_decay: float = 0.999
    ema_start_step: int = 0

    # 梯度裁剪
    max_grad_norm: float = 1.0

    # ── Auxiliary reconstruction loss (shared FM/DM) ────────────────
    # FM: loss_x1 = MSE(x_t + (1-t)*v_pred, x1)
    # DM: loss_x1 = MSE(predict_x0(target), target)
    # Both use the same weight — symmetry fix for fair FM/DM comparison
    aux_loss_weight: float = 0.5

    # ── Physics losses (symmetric FM/DM) ────────────────────────────
    physics_warmup_steps: int = 0          # was FM=10000, DM=0 (asymmetric)
    physics_target_weight: float = 0.1    # was defaulting to 1.0 (too large)
    geostrophic_weight: float = 0.01     # physics regularizer, needs center_lats in batch

    # 通道加权 MSE
    use_channel_weights: bool = True
    channel_weights: Tuple[float, ...] = (
        1.0, 1.0, 1.0,    # u: 850/500/250
        1.0, 1.0, 1.0,    # v: 850/500/250
        1.0, 1.0, 1.0,    # z: 850/500/250 — was 2/5/10; aggressive weights + velocity_clamp caused z velocity collapse
    )
    # Z-channel weight override — applied AFTER channel_weights on z channels only.
    # Setting this to >1.0 (e.g. 5.0) is the simpler fix (Option 1).
    z_channel_weight_override: float = 5.0  # 1.0 = no override; 5.0 = 5× z emphasis
    use_channel_wise_normalized_loss: bool = True  # scale-invariant: normalize by target std per channel

    # Z-Predictor auxiliary network (Option 2)
    # Trains a separate lightweight U-Net that learns wind→z mapping via geostrophic balance
    use_z_predictor: bool = False          # set True to enable ZPredictor
    z_predictor_weight: float = 1.0        # loss weight for z-predictor branch
    velocity_clamp: Optional[Tuple[float, float]] = None  # was (-3.0, 3.0) — clamp kills z velocity learning
    velocity_loss_scale: float = 1.0  # scale the MSE loss on velocity

    # 条件噪声 (DM 模式专用)
    condition_noise_sigma: float = 0.30
    condition_noise_rampup_epochs: int = 100
    condition_noise_prob: float = 0.5
    condition_noise_spatial_smooth: bool = True
    condition_noise_smooth_kernel: int = 5

    # Scheduled Sampling (DM 自回归训练)
    scheduled_sampling_enabled: bool = False
    scheduled_sampling_start_epoch: int = 50
    scheduled_sampling_max_prob: float = 0.3
    scheduled_sampling_rampup_epochs: int = 100
    scheduled_sampling_max_replace: int = 2
    scheduled_sampling_ddim_steps: int = 10

    # 验证与 Early Stopping
    eval_every: int = 10
    early_stopping_patience: int = 50
    save_top_k: int = 3

    # 日志
    log_every: int = 20
    use_tensorboard: bool = True

    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    use_compile: bool = False
    cudnn_benchmark: bool = True
    seed: int = 42


@dataclass
class InferenceConfig:
    """推理配置"""
    # FM 推理
    euler_steps: int = 4
    euler_mode: str = "midpoint"  # "euler", "midpoint", "heun"

    # DM 推理
    ddim_steps: int = 50

    # 裁剪范围 — None 禁用 clamp，避免压制 z 通道方差
    clamp_range: Optional[Tuple[float, float]] = None  # was (-5.0, 5.0)
    z_clamp_range: Optional[Tuple[float, float]] = None  # was (-3.0, 3.0)

    # 自回归
    autoregressive_steps: int = 24
    autoregressive_noise_sigma: float = 0.05
    ar_ensemble_per_step: int = 1

    # 集合预报
    ensemble_size: int = 10

    checkpoint_path: str = ""
    output_dir: str = "outputs"
    device: str = "cuda"


@dataclass
class ComparisonConfig:
    """对比实验配置"""
    run_fm: bool = True
    run_dm: bool = True
    skip_train: bool = False
    compute_psd: bool = True
    num_eval_samples: int = 100
    output_dir: str = "comparison_results"
    device: str = "cuda"


def get_config(
    data_root: Optional[str] = None,
    preprocess_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[DataConfig, ModelConfig, TrainConfig, InferenceConfig]:
    """获取完整配置"""
    data_cfg = DataConfig()
    if data_root:
        data_cfg.data_root = data_root
        data_cfg.era5_dir = data_root
    if preprocess_dir:
        data_cfg.preprocessed_dir = preprocess_dir

    model_cfg = ModelConfig(
        in_channels=data_cfg.num_channels,
        cond_channels=data_cfg.condition_channels,
    )

    train_cfg = TrainConfig()
    if checkpoint_dir:
        train_cfg.checkpoint_dir = checkpoint_dir

    infer_cfg = InferenceConfig()

    return data_cfg, model_cfg, train_cfg, infer_cfg


def get_comparison_config() -> ComparisonConfig:
    """获取对比实验配置"""
    return ComparisonConfig()
