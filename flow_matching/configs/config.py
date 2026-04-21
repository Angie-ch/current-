"""
Flow Matching 配置文件 — 三阶段训练策略
针对最低RMSE优化的三阶段训练配置

阶段1: 预热期 (Epoch 1-20) - 只训练loss_mse
阶段2: 物理炼金期 (Epoch 21-80) - 线性引入物理约束
阶段3: 微调收敛期 (Epoch 81-120) - 学习率下调，稳定收敛
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os


@dataclass
class DataConfig:
    """数据配置 — 三阶段训练专用"""
    # 数据目录
    era5_dir: str = "/root/autodl-tmp/fyp_final/preprocessed_9ch_40x40"
    csv_path: str = "/root/autodl-tmp/fyp_final/VER3_original/VER3/Trajectory/processed_typhoon_tracks.csv"
    
    # 通道配置
    era5_channels: int = 9
    grid_size: int = 40
    history_steps: int = 16  # 16步历史
    forecast_steps: int = 24  # 24步预测
    
    # 数据集划分 (ICLR 2024标准)
    train_years: List[int] = field(default_factory=lambda: list(range(1950, 2016)))
    val_years: List[int] = field(default_factory=lambda: list(range(2016, 2019)))
    test_years: List[int] = field(default_factory=lambda: list(range(2019, 2022)))

    num_workers: int = 8
    pin_memory: bool = True
    
    # 归一化统计量路径
    norm_stats_path: str = "/root/autodl-tmp/fyp_final/preprocessed_9ch_40x40/norm_stats.pt"
    
    # 地理范围和分辨率
    lat_range: Tuple[float, float] = (0.0, 60.0)   # 纬度范围
    lon_range: Tuple[float, float] = (95.0, 185.0)  # 经度范围
    lat_res: float = 0.25  # 纬度分辨率 (度)
    lon_res: float = 0.25  # 经度分辨率 (度)
    
    # 气压层和变量配置（用于物理损失计算）
    pressure_levels: List[int] = field(default_factory=lambda: [850, 500, 250])
    pressure_level_vars: List[str] = field(default_factory=lambda: ["u", "v", "z"])
    
    @property
    def num_channels(self) -> int:
        """通道总数"""
        return len(self.pressure_level_vars) * len(self.pressure_levels)
    
    @property
    def condition_channels(self) -> int:
        """条件通道数 = 通道数 × 历史步数"""
        return self.num_channels * self.history_steps
    
    @property
    def target_channels(self) -> int:
        """目标通道数 = 通道数 × 预测步数"""
        return self.num_channels * self.forecast_steps
    
    def get_wind_channel_indices(self) -> List[Tuple[int, int]]:
        """
        获取风场通道索引 (u, v) 对
        
        假设通道顺序: [u_850, u_500, u_250, v_850, v_500, v_250, z_850, z_500, z_250]
        即: u通道在 0-2, v通道在 3-5, z通道在 6-8
        
        注意: 只考虑当前时间步的通道 (forecast_steps=1)
        """
        pairs = []
        n_pl = len(self.pressure_levels)  # 3 (气压层数量)
        
        # u 通道索引: 0, 1, 2
        # v 通道索引: 3, 4, 5
        u_indices = list(range(n_pl))  # [0, 1, 2]
        v_indices = [n_pl + i for i in range(n_pl)]  # [3, 4, 5]
        
        # 只生成当前时间步的索引 (忽略 forecast_steps 的多步预测)
        base = 0
        for i in range(n_pl):
            u_ch = base + u_indices[i]
            v_ch = base + v_indices[i]
            pairs.append((u_ch, v_ch))
        return pairs


@dataclass
class ModelConfig:
    """模型架构配置 — CFM-DiT (预测速度场 v)"""
    d_model: int = 384
    n_heads: int = 6
    n_dit_layers: int = 12
    n_cond_layers: int = 3
    ff_mult: int = 4
    patch_size: int = 4
    grid_size: int = 40
    dropout: float = 0.1

    # Flow Matching 不需要离散噪声调度，但保留此参数作为文档说明
    num_diffusion_steps: int = 1000  # (仅文档意义，不用于训练)

    # 输入/输出通道
    in_channels: int = 9
    cond_channels: int = 45

    # ===== P1: 时间嵌入优化 =====
    # 将连续时间 t∈[0,1] 缩放至更大整数区间以增强时间嵌入区分度
    # 参考 Stable Diffusion / DiT 的离散化思路
    time_embedding_scale: float = 1000.0  # 推荐范围: 100~10000

    # ===== P5: 通道分组处理 =====
    # 将气象变量按物理量分组处理，增强特征提取能力
    use_grouped_conv: bool = False  # 分组卷积仅适配 cond_channels=45, 禁用以支持任意时间步  # 是否启用分组卷积


@dataclass
class TrainConfig:
    """训练配置 — 三阶段训练策略（最低RMSE优化）"""
    
    # ===== 基础训练参数 =====
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_epochs: int = 200  # 三阶段总训练轮数
    
    # ===== 学习率调度 - 三阶段 =====
    # 第一、二阶段: 2e-4
    # 第三阶段: 余弦退火至 1e-5
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    warmup_steps: int = 200
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-5  # 第三阶段最低学习率
    
    # ===== 三阶段学习率控制 =====
    # 第一阶段 (Epoch 1-100): 2e-4
    # 第二阶段 (Epoch 101-160): 2e-4
    # 第三阶段 (Epoch 161-200): 余弦退火至 1e-5
    phase3_lr_start_epoch: int = 160  # 第三阶段开始的epoch
    
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    
    ema_decay: float = 0.999
    ema_start_step: int = 0
    
    max_grad_norm: float = 1.0

    # ===== Flow Matching 特有的训练参数 =====
    # CFM 的线性路径使得训练更稳定，不需要物理损失来稳定训练
    # 但保留物理损失用于精细调控
    physics_loss_weight: float = 0.0
    vorticity_loss_weight: float = 0.05  # 增加涡度约束，平滑高层风场

    # ===== 逐通道加权 MSE + 气压层敏感度权重 =====
    # 核心问题: Z 通道值域大(14230~106233)，归一化后 MSE 量级远小于 UV
    # 联合优化时 Z 主导梯度，导致 UV250/U500 RMSE 高
    #
    # 解决方案: 逐通道归一化 (每个通道独立 MSE 后 normalize)
    # 保证 Z 和 UV 对总损失的贡献比例由权重决定，而非数据量级决定
    use_channel_weights: bool = True
    channel_weights: Tuple[float, ...] = (
        2.5, 2.5, 3.0,   # u_850, u_500, u_250 → 高层更需要优化
        2.5, 2.5, 3.0,   # v_850, v_500, v_250 → 高层更需要优化
        1.5, 2.0, 2.5,   # z → 适当降低，避免 Z 主导
    )

    # 气压层敏感度权重: 高层风速大、变化剧烈，需要更高权重
    use_pressure_level_weights: bool = True
    pressure_level_weights: Tuple[float, ...] = (
        1.0,   # 850 hPa (低层，稳定)
        1.1,   # 500 hPa (中层，从1.3降到1.1)
        1.3,   # 250 hPa (高层，从2.0降到1.3)
    )

    # 逐通道归一化: 消除 Z(大值域) vs UV(小值域) 之间的量级差异
    # 联合权重决定贡献比例，而非数据量级
    use_channel_wise_normalized_loss: bool = True  # 启用逐通道 RMSE 归一化

    # 条件噪声增强 (与 Diffusion 版本兼容)
    condition_noise_sigma: float = 0.30
    condition_noise_rampup_epochs: int = 100
    condition_noise_prob: float = 0.5
    condition_noise_spatial_smooth: bool = True
    condition_noise_smooth_kernel: int = 5

    # ===== Early Stopping - 防止80 Epoch后过拟合 =====
    # Flow Matching 容易在80 Epoch左右出现过拟合
    eval_every: int = 5              # 更频繁验证
    early_stopping_patience: int = 15  # 缩短耐心值（从50降到15）

    log_every: int = 20
    use_tensorboard: bool = True

    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    use_compile: bool = False
    cudnn_benchmark: bool = True
    seed: int = 42

    # ===== 速度场预测的额外配置 =====
    # velocity_loss_scale: 速度场的 L2 损失量级通常比噪声预测大，
    # 可以用此因子缩放以平衡与物理损失的量级
    velocity_loss_scale: float = 1.0

    # 速度场预测的 clamp 范围（防止异常速度）
    velocity_clamp: Optional[Tuple[float, float]] = None  # e.g., (-5.0, 5.0)

    # ===== 三阶段物理损失调度 (P2) =====
    # 阶段1 (Epoch 1-100): physics_weight = 0 (纯 MSE warmup)
    # 阶段2 (Epoch 101-160): 线性增加 physics_weight 到目标值
    # 阶段3 (Epoch 161-200): 锁定 physics_weight
    
    # 物理损失从第101个epoch开始启动
    physics_warmup_start_epoch: int = 100  # Epoch 101开始引入物理约束
    physics_warmup_end_epoch: int = 160   # Epoch 160时达到目标权重
    physics_warmup_steps: int = 25000    # 约200 epochs × 125 steps
    physics_warmup_type: str = "linear"  # "linear" | "cosine"
    physics_target_weight: float = 0.05   # 降低物理权重，减少对 MSE 的干扰
    
    # ===== 新的地转平衡物理损失配置 =====
    # 2026-04-12: 禁用物理损失，专注于纯 MSE + x0 监督
    use_geostrophic_physics: bool = False  # 禁用地转平衡物理损失
    geostrophic_weight: float = 0.0
    divergence_weight: float = 0.0
    vorticity_weight: float = 0.0
    
    # ===== 多指标综合评估权重 =====
    # 用于 checkpoint 选择的综合评估
    checkpoint_weights: Tuple[float, float] = (0.7, 0.3)  # (rmse_weight, physics_weight)
    checkpoint_top_k: int = 5             # 保存 top-k checkpoints
    
    # ===== 路径扰动 (Scheduled Sampling) - 第二阶段启用 =====
    # 阶段1: 禁用
    # 阶段2: 20%概率启用扰动
    # 阶段3: 禁用
    path_perturb_prob: float = 0.2   # 第二阶段扰动概率
    path_perturb_sigma: float = 0.05  # 5%强度的扰动噪声

    # ===== x0 直接监督损失权重 =====
    # loss_x0 直接作用于反推的 x0，与评估指标一致
    x0_loss_weight: float = 0.1  # 建议比例 1:0.1 (mse:x0)

    # ===== Z通道专属物理约束权重 (Z优化训练专用) =====
    # 2026-04-13: 针对 z_850 RMSE 最高问题，给Z通道的物理损失额外放大
    z_phys_weight: float = 1.0  # 默认1.0，train_z_optimized.py 中设为2.0


@dataclass
class InferenceConfig:
    """推理配置 — Euler 高速采样"""
    # ===== 核心区别：Euler 步数 vs DDIM 步数 =====
    # DDIM (Diffusion): 50步 × 1000步扩散 = 50000 步等效
    # Euler (CFM): 1~10 步 = 极高速度
    euler_steps: int = 1  # 推荐: 1步(Euler)或4步(Heun)，1步通常精度足够

    # Euler 步长模式
    # "uniform": dt = 1.0 / steps，均匀分布
    # "midpoint": 中点法（二阶，精度高于Euler，推荐）
    # "heun": Heun方法（二阶，与Midpoint精度相当）
    # "euler": 简单欧拉法（一阶，速度最快）
    euler_mode: str = "midpoint"  # P3: 默认使用中点法

    # 自适应步长的误差阈值（仅 euler_mode="adaptive" 时使用）
    adaptive_tol: float = 0.01

    clamp_range: Tuple[float, float] = (-5.0, 5.0)

    # ===== P4: Z-clamp配置 (类似 Diffusion) =====
    use_z_clamp: bool = True  # 是否启用 z 通道 clamp
    z_clamp_range: Tuple[float, float] = (-1.0, 1.0)  # z 通道 clamp 范围 (归一化空间)

    # 自回归
    autoregressive_steps: int = 24
    autoregressive_noise_sigma: float = 0.05

    # 逐步集合（与 Diffusion 版本兼容）
    ar_ensemble_per_step: int = 5

    # 集合预报
    ensemble_size: int = 10

    checkpoint_path: str = ""
    output_dir: str = "outputs"
    device: str = "cuda"


def get_config(
    era5_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[DataConfig, ModelConfig, TrainConfig, InferenceConfig]:
    """获取完整配置"""
    data_cfg = DataConfig()
    if era5_dir:
        data_cfg.era5_dir = era5_dir

    # 模型配置
    model_cfg = ModelConfig(
        in_channels=data_cfg.era5_channels,
        cond_channels=data_cfg.era5_channels * data_cfg.history_steps,
    )

    train_cfg = TrainConfig()
    if checkpoint_dir:
        train_cfg.checkpoint_dir = checkpoint_dir

    infer_cfg = InferenceConfig()

    return data_cfg, model_cfg, train_cfg, infer_cfg
