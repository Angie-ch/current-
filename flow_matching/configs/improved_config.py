"""
改进的训练配置 - 针对Z通道和UV风场预测优化
基于原有config.py的改进版本
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os


@dataclass
class ImprovedTrainConfig:
    """改进的训练配置"""
    
    # ===== 基础训练参数 =====
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_epochs: int = 200
    
    # ===== 学习率 =====
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 200
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-5
    
    # ===== AMP =====
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    
    # ===== EMA =====
    ema_decay: float = 0.9999
    ema_start_step: int = 0
    
    # ===== 梯度裁剪 =====
    max_grad_norm: float = 1.0
    
    # ===== ★ 改进1: 提高x0_loss权重 =====
    # 原配置 x0_loss_weight: 0.1 → 改为 0.5
    # x0直接监督对RMSE影响最大
    x0_loss_weight: float = 0.5
    
    # ===== ★ 改进2: 提高物理损失权重 =====
    # 原配置 physics_target_weight: 0.05 → 改为 0.3
    # 地转平衡约束对风场方向很重要
    physics_target_weight: float = 0.3
    
    # ===== ★ 改进3: 分离UV和Z的损失权重 =====
    # U/V通道需要更高权重来改善风场方向
    uv_loss_weight: float = 2.0   # UV通道权重
    z_loss_weight: float = 1.0    # Z通道权重
    
    # ===== ★ 改进4: 气压层敏感度权重 =====
    # 高层(250hPa)风速变化大，需要更高权重
    pressure_level_weights: Tuple[float, ...] = (
        1.0,   # 850 hPa
        1.2,   # 500 hPa  
        1.5,   # 250 hPa (从1.3提升到1.5)
    )
    
    # ===== ★ 改进5: 增加条件噪声增强 =====
    # 提高推理时的鲁棒性
    condition_noise_sigma: float = 0.35  # 从0.30提升
    condition_noise_rampup_epochs: int = 100
    condition_noise_prob: float = 0.6    # 从0.5提升
    condition_noise_spatial_smooth: bool = True
    condition_noise_smooth_kernel: int = 5
    
    # ===== 物理损失配置 =====
    use_geostrophic_physics: bool = True   # 启用地转平衡
    geostrophic_weight: float = 0.15      # 地转平衡权重
    divergence_weight: float = 0.05       # 散度约束
    vorticity_weight: float = 0.10         # 涡度平滑
    
    # ===== 物理损失调度 =====
    physics_warmup_start_epoch: int = 50   # 从100改为50，提前启动
    physics_warmup_end_epoch: int = 150    # 从160改为150
    physics_warmup_type: str = "linear"
    
    # ===== 通道损失 =====
    use_channel_weights: bool = True
    channel_weights: Tuple[float, ...] = (
        # U风: 提高250hPa权重
        2.0, 2.0, 3.0,   # u_850, u_500, u_250
        # V风: 提高250hPa权重
        2.0, 2.0, 3.0,   # v_850, v_500, v_250
        # Z: 略微降低，避免主导
        1.0, 1.2, 1.5,    # z_850, z_500, z_250
    )
    
    # ===== 逐通道归一化损失 =====
    use_channel_wise_normalized_loss: bool = True
    
    # ===== 路径扰动 (Scheduled Sampling) =====
    path_perturb_prob: float = 0.15   # 降低扰动概率
    path_perturb_sigma: float = 0.03   # 降低扰动强度
    
    # ===== 速度场限制 =====
    velocity_loss_scale: float = 1.0
    velocity_clamp: Optional[Tuple[float, float]] = None
    
    # ===== 评估配置 =====
    eval_every: int = 5
    early_stopping_patience: int = 20
    log_every: int = 20
    use_tensorboard: bool = True
    
    # ===== Checkpoint =====
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    checkpoint_top_k: int = 5
    checkpoint_weights: Tuple[float, float] = (0.8, 0.2)  # 提高RMSE权重
    
    # ===== 其他 =====
    use_compile: bool = False
    cudnn_benchmark: bool = True
    seed: int = 42
