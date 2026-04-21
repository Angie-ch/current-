"""
配置文件：存储所有超参数和路径配置
Video2Video 条件扩散模型 - 台风路径预测
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import torch

@dataclass
class DataConfig:
    """数据相关配置 - LT3P风格：3小时分辨率，72小时预测"""
    # 路径配置
    csv_path: str = "processed_typhoon_tracks.csv"
    era5_dir: str = "C:\\Users\\fyp\\Desktop\\Typhoon_data_final"

    # 坐标归一化范围（扩大以覆盖高纬度和跨日期变更线台风）
    lat_range: Tuple[float, float] = (0.0, 60.0)
    lon_range: Tuple[float, float] = (95.0, 185.0)

    # 网格尺寸 - 与 ERA5-Diffusion 扩散模型保持一致 (40×40)
    grid_height: int = 40
    grid_width: int = 40

    # 时间分辨率（小时）- 3小时
    time_resolution_hours: int = 3

    # 压力层（hPa）- 与 ERA5-Diffusion 扩散模型一致：850, 500
    pressure_levels: List[int] = field(default_factory=lambda: [850, 500, 250])

    # 3D ERA5 变量 - 风场 + 位势高度（与扩散模型一致，去掉 vo）
    # 注意: vo(涡度) 已移除 — 40×40 粗网格无法解析涡度空间梯度，
    #       扩散模型也不预测 vo，保持两端通道完全一致避免零填充问题
    era5_3d_vars: List[str] = field(default_factory=lambda: ['u', 'v', 'z'])

    # 2D ERA5 变量 - 已清空，与扩散模型一致（仅使用气压层变量）
    era5_2d_vars: List[str] = field(default_factory=lambda: [])


@dataclass
class ModelConfig:
    """模型相关配置 - LT3P风格：48小时输入，72小时预测"""
    # === 序列长度（LT3P风格）===
    # 输入：过去48小时轨迹 @ 3小时间隔 = 16个时间步
    t_history: int = 16     # 历史轨迹长度（48h / 3h = 16）
    # 输入：未来72小时气象场 @ 3小时间隔 = 24个时间步
    t_future_era5: int = 24  # 未来ERA5长度（72h / 3h = 24）
    # 输出：未来72小时轨迹 @ 3小时间隔 = 24个时间步
    t_future: int = 24      # 预测轨迹长度（72h / 3h = 24）

    # === 输入维度 ===
    # 轨迹坐标维度
    coord_dim: int = 2      # lat, lon（论文只用经纬度）
    # 输出维度
    output_dim: int = 2     # lat, lon

    # 条件特征维度（环境特征）
    cond_feature_dim: int = 32

    # === ERA5 物理条件编码器配置（LT3P的R(·)）===
    # 输入通道数：3个气压层变量(u,v,z) × 3个压力层 = 9通道
    # 与扩散模型完全一致，不使用地面变量
    era5_channels: int = 9
    era5_base_channels: int = 64
    era5_out_dim: int = 256

    # === 轨迹编码器配置 ===
    coord_embed_dim: int = 128

    # === Transformer 配置 ===
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 8
    transformer_ff_dim: int = 2048
    dropout: float = 0.1

    # === 辅助 Heatmap Head 配置 ===
    use_heatmap_head: bool = False  # LT3P不使用heatmap
    heatmap_loss_weight: float = 0.0
    gaussian_sigma: float = 2.0

    # === Diffusion 配置 ===
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine"


@dataclass
class TrainConfig:
    """训练相关配置 - 针对 RTX 4090 24GB 优化"""
    batch_size: int = 32          # 由于序列更长，适当减小batch
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    num_epochs: int = 200

    # 数据加载优化
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = False

    # 是否对真实帧和插值帧使用不同权重
    use_sample_weights: bool = True
    real_sample_weight: float = 1.0
    interp_sample_weight: float = 0.5

    # 早停机制
    early_stopping: bool = True
    patience: int = 25

    # 保存和日志
    save_interval: int = 10
    log_interval: int = 100
    checkpoint_dir: str = "checkpoints/"

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_by: str = "storm_id"
    
    # 学习率调度
    lr_scheduler: str = "cosine_warmup"
    warmup_epochs: int = 20
    
    # 梯度累积
    gradient_accumulation_steps: int = 1
    
    # 最小台风持续时间（小时）- 必须 >= 48 + 72 = 120小时
    min_typhoon_duration_hours: int = 120


@dataclass
class SampleConfig:
    """采样相关配置"""
    num_samples: int = 10
    use_ddim: bool = True
    ddim_steps: int = 50
    eta: float = 0.0
    guidance_scale: float = 1.0


# 全局配置实例
data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
sample_cfg = SampleConfig()

