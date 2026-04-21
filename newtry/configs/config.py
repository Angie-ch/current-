"""
ERA5-Diffusion 项目配置模块
所有超参数通过 dataclass 集中管理，适配 40×40 网格 + RTX 5090 Ti 32GB
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os


@dataclass
class DataConfig:
    """数据相关配置"""
    # 数据路径
    data_root: str = "/Volumes/T7 Shield/Typhoon_data_final"
    stats_csv: str = ""  # 自动设为 data_root/typhoon_organization_stats_1950_2021.csv

    # 空间网格
    grid_size: int = 40  # 40×40 网格

    # 气象变量定义
    # 仅保留气压层变量: 风场(u,v) + 位势高度(z)
    # 台风轨迹预测核心: 引导气流(u/v) + 副热带高压形态(z)
    # 已移除: vo(涡度, 粗网格无法解析), u10m/v10m(地面风, 对轨迹贡献小), msl(主要反映强度)
    # 气压层变量（按NC文件中的变量名）
    pressure_level_vars: List[str] = field(
        default_factory=lambda: ["u", "v", "z"]  # 风场 + 位势高度
    )
    # 地面变量（已清空 — 台风轨迹仅需气压层信息）
    surface_vars: List[str] = field(
        default_factory=lambda: []  # 不使用地面变量
    )
    # 气压层（hPa）
    pressure_levels: List[int] = field(
        default_factory=lambda: [850, 500, 250]  # 850hPa（引导气流）+ 500hPa（副热带高压）+ 250hPa（高层转向）
    )

    # 时间窗口
    # 每个时间步通道数: 3×3 = 9 (u,v,z各3层)
    # 条件: 5×9 = 45 通道
    # 目标: 1×9 = 9 通道
    history_steps: int = 5       # 历史窗口：5 步 × 3h = 15h
    forecast_steps: int = 1      # 预测窗口：1 步 × 3h = 3h
    time_interval_hours: int = 3  # 时间间隔：3小时

    # 数据划分比例
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # DataLoader
    num_workers: int = 16
    prefetch_factor: int = 4
    pin_memory: bool = True

    # 预处理目录 (NC -> NPY 加速)
    # 设置后训练时使用内存映射读取, I/O 提速 50-100 倍
    preprocessed_dir: Optional[str] = None

    # 归一化统计文件路径（训练时计算并保存，推理时加载）
    norm_stats_path: str = ""  # 自动设为项目目录下的 norm_stats.pt

    def __post_init__(self):
        if not self.stats_csv:
            self.stats_csv = os.path.join(
                self.data_root, "typhoon_organization_stats_1950_2021.csv"
            )

    @property
    def num_pressure_level_channels(self) -> int:
        """气压层变量总通道数"""
        return len(self.pressure_level_vars) * len(self.pressure_levels)

    @property
    def num_surface_channels(self) -> int:
        """地面变量通道数"""
        return len(self.surface_vars)

    @property
    def num_channels(self) -> int:
        """每个时间步的总通道数"""
        return self.num_pressure_level_channels + self.num_surface_channels  # 3×3 + 0 = 9

    @property
    def condition_channels(self) -> int:
        """条件输入的总通道数 = C × T_history"""
        return self.num_channels * self.history_steps  # 9 × 5 = 45

    @property
    def target_channels(self) -> int:
        """预测目标的总通道数 = C × T_forecast"""
        return self.num_channels * self.forecast_steps  # 9 × 1 = 9

    def get_wind_channel_indices(self) -> List[Tuple[int, int]]:
        """
        获取风场 u/v 分量在目标张量中的通道索引对（用于物理约束损失）
        返回: [(u_idx, v_idx), ...] 共 3 对（3个气压层）
        """
        pairs = []
        n_pl = len(self.pressure_levels)

        for t_step in range(self.forecast_steps):
            base = t_step * self.num_channels
            # 气压层 u/v：在 pressure_level_vars 中找到 u 和 v 的索引
            u_idx_in_pl = self.pressure_level_vars.index("u")
            v_idx_in_pl = self.pressure_level_vars.index("v")
            for lev in range(n_pl):
                u_ch = base + u_idx_in_pl * n_pl + lev
                v_ch = base + v_idx_in_pl * n_pl + lev
                pairs.append((u_ch, v_ch))

            # 地面 u10m/v10m（仅在 surface_vars 包含时）
            if "u10m" in self.surface_vars and "v10m" in self.surface_vars:
                u10_idx = self.surface_vars.index("u10m")
                v10_idx = self.surface_vars.index("v10m")
                u10_ch = base + self.num_pressure_level_channels + u10_idx
                v10_ch = base + self.num_pressure_level_channels + v10_idx
                pairs.append((u10_ch, v10_ch))

        return pairs


@dataclass
class ModelConfig:
    """模型架构配置 — DiT (Diffusion Transformer)"""
    # DiT 核心参数
    d_model: int = 384           # Transformer 隐藏维度
    n_heads: int = 6             # 注意力头数
    n_dit_layers: int = 12       # DiT Block 层数
    n_cond_layers: int = 3       # 条件编码器 Self-Attention 层数
    ff_mult: int = 4             # FFN 倍率 (D -> ff_mult*D -> D)
    patch_size: int = 4          # Patch 大小 (40/4 = 10 -> 100 tokens)
    dropout: float = 0.1         # Dropout

    # 扩散过程
    num_diffusion_steps: int = 1000
    noise_schedule: str = "cosine"  # cosine schedule
    # DDIM 采样步数
    ddim_sampling_steps: int = 50
    # 预测目标: "eps" = 预测噪声, "v" = 预测 velocity (对 z 等高量级变量更稳定)
    # v-prediction 恢复 x₀ 时无需除法，消除了 eps-prediction 的数值放大问题
    # 注意: 必须与训练时使用的 prediction_type 一致！eps 模型用 "eps", v 模型用 "v"
    prediction_type: str = "eps"

    # 输入/输出通道（由 DataConfig 计算，这里设默认值）
    in_channels: int = 9       # 预测目标通道数 = 9 × 1
    cond_channels: int = 45    # 条件通道数 = 9 × 5


@dataclass
class TrainConfig:
    """训练配置 - 适配 RTX 5090 Ti 32GB"""
    # 批次与梯度累积
    # 96ch 模型约 40M 参数，适当减小 batch_size 以适配 32GB 显存
    # 滑动窗口缩小(5+1=6步)后样本数会增加
    batch_size: int = 48
    gradient_accumulation_steps: int = 1  # 可改为 2 以等效 batch=48
    max_epochs: int = 2000  # 对标 TammyLing 的长训练策略

    # 优化器
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    # 学习率调度
    warmup_steps: int = 200
    warmup_start_lr: float = 1e-6    # warmup 起始 lr
    min_lr: float = 1e-6             # cosine annealing 最低 lr

    # 混合精度
    # BF16: Ampere 及以上架构 (A100/3090/4090/5090)
    # FP16: 更老的架构 (V100/T4 等)
    # AutoDL 上根据实际 GPU 选择，不确定就用 float16 更安全
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    # EMA
    # 总优化器步数~5000, decay=0.999 约需 1000 步收敛, 更适合小数据集
    ema_decay: float = 0.999
    ema_start_step: int = 0

    # 梯度裁剪
    max_grad_norm: float = 1.0

    # 物理约束
    # div_loss 反向传播经过 predict_x0_from_eps 的除法运算，
    # 当 t 接近 T 时 sqrt(ᾱ_t)→0 导致 x0_pred 爆炸，
    # 进而引发梯度爆炸，已导致三次训练崩溃。
    # 设为 0 彻底禁用，模型可从数据中隐式学习物理一致性。
    physics_loss_weight: float = 0.0  # 风场散度正则化权重 λ (已禁用)

    # 旋度一致性约束 (已禁用 — vo 已从预测通道中移除)
    vorticity_loss_weight: float = 0.0  # 旋度约束权重 (无 vo 通道，设为 0)

    # 逐通道损失加权 (给 z 更高权重，防止自回归模式坍缩)
    # 通道顺序: u_850,u_500,u_250, v_850,v_500,v_250,
    #           z_850,z_500,z_250
    use_channel_weights: bool = True
    channel_weights: Tuple[float, ...] = (
        1.0, 1.0, 1.0,    # u: 850/500/250
        1.0, 1.0, 1.0,    # v: 850/500/250
        2.0, 2.0, 2.5,    # z: 850/500/250 (加权, 250最弱需最高权重)
    )

    # 条件噪声增强 (解决自回归分布偏移问题)
    # 训练时给条件输入加高斯噪声，模拟自回归推理时的不完美输入
    # 噪声从 0 线性增长到 condition_noise_sigma，前 rampup_epochs 个 epoch 完成爬升
    # 注意：此噪声仅在训练时加到条件上，推理 +3h 时输入仍是真实数据，单步精度不受影响
    # 随机噪声策略: 每个样本以 condition_noise_prob 概率注入噪声，其余保持干净
    # 这样模型在干净样本上保持单步精度，在噪声样本上学习鲁棒性
    condition_noise_sigma: float = 0.30   # 最终噪声标准差 (增大到0.30, 配合结构化噪声模拟自回归误差)
    condition_noise_rampup_epochs: int = 100  # 噪声爬升期
    condition_noise_prob: float = 0.5     # 每个样本注入噪声的概率 (50%干净+50%噪声)
    # 结构化噪声: 在白噪声基础上混合空间平滑噪声
    # 模拟自回归推理时的结构化偏差（z 场整体偏移等），而不仅仅是像素级白噪声
    condition_noise_spatial_smooth: bool = True
    condition_noise_smooth_kernel: int = 5  # 平滑卷积核大小

    # ---- Scheduled Sampling (计划采样) ----
    # 核心思想: 训练时偶尔用模型自身的预测替换 condition 中的帧，
    # 让模型学会处理自己的"不完美输出"，直接解决自回归分布偏移
    # (随机噪声无法模拟模型真实的结构化预测误差，这才是 z 通道 +33h 崩溃的根因)
    #
    # 流程:
    #   1. 以概率 ss_prob 触发 scheduled sampling
    #   2. 用当前模型做多步自回归 DDIM 采样，生成一条预测链
    #   3. Detach 预测帧，逐帧替换 condition 窗口的后 K 帧
    #   4. 用这个被"污染"的 condition 做正常训练前向传播
    #   5. 模型被迫学会在自己的预测结果上继续准确预测
    scheduled_sampling_enabled: bool = True
    scheduled_sampling_start_epoch: int = 50       # 更早启用 SS (原100, 让模型尽早学自回归)
    scheduled_sampling_max_prob: float = 0.3       # 降低触发概率 (原0.7→0.3, 平衡速度与效果)
    scheduled_sampling_rampup_epochs: int = 100    # 概率从0爬升到max_prob的epoch数
    scheduled_sampling_max_replace: int = 2        # 最多替换 2 帧 (原4→2, 大幅减少 DDIM 次数)
    scheduled_sampling_ddim_steps: int = 10        # DDIM 步数 (原25→10, SS不需要高质量采样)

    # 验证与 Early Stopping
    eval_every: int = 10        # 每 10 个 epoch 验证一次
    early_stopping_patience: int = 50  # 连续 50 次验证不改善则停止（长训练需要更多耐心）
    save_top_k: int = 3          # 保留最好的 k 个 checkpoint

    # 日志
    log_every: int = 20          # 每 20 步记录一次（方便观察收敛情况）
    use_tensorboard: bool = True

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None  # 恢复训练的 checkpoint 路径

    # torch.compile (RTX 5090 Blackwell 架构暂不兼容，关闭)
    use_compile: bool = False

    # cuDNN
    cudnn_benchmark: bool = True

    # 随机种子
    seed: int = 42


@dataclass
class InferenceConfig:
    """推理配置"""
    # DDIM 采样
    ddim_steps: int = 50
    # x_start 裁剪范围
    clamp_range: Tuple[float, float] = (-5.0, 5.0)
    # z 通道使用更紧的 clamp 范围，防止自回归累积偏移
    # z 的物理量级远大于 u/v，归一化空间中 ±3σ 足以覆盖正常变动
    # 同时阻止 DDIM 采样产生极端 z 值引发正反馈崩溃
    z_clamp_range: Tuple[float, float] = (-3.0, 3.0)
    # 自回归
    autoregressive_steps: int = 24   # 24 轮 → 72h
    autoregressive_noise_sigma: float = 0.05  # 自回归噪声注入 (增大以抑制误差累积)
    # 自回归逐步集合平均: 每步做 N 次 DDIM 采样取平均，降低方差，防止 z 通道崩溃
    ar_ensemble_per_step: int = 5   # 每步集合数 (1=不集合, 5=推荐值)
    # 集合预报
    ensemble_size: int = 10  # 集合成员数
    # Checkpoint
    checkpoint_path: str = ""
    # 输出
    output_dir: str = "outputs"
    # 设备
    device: str = "cuda"


def get_config(
    data_root: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[DataConfig, ModelConfig, TrainConfig, InferenceConfig]:
    """获取完整配置（方便一次性创建所有配置）"""
    data_cfg = DataConfig()
    if data_root:
        data_cfg.data_root = data_root
        data_cfg.__post_init__()

    model_cfg = ModelConfig(
        in_channels=data_cfg.target_channels,
        cond_channels=data_cfg.condition_channels,
    )

    train_cfg = TrainConfig()
    if checkpoint_dir:
        train_cfg.checkpoint_dir = checkpoint_dir

    infer_cfg = InferenceConfig()

    return data_cfg, model_cfg, train_cfg, infer_cfg
