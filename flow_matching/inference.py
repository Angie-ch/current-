"""
Flow Matching 推理模块 — 自回归多步预测

支持:
1. 单步预测 (single-step)
2. 自回归多步预测 (autoregressive, 24步)
3. 多种 ODE 求解器 (Euler, Midpoint, Heun)

使用方法:
    from inference import CFMInferencer
    
    inferencer = CFMInferencer(model, data_cfg, infer_cfg)
    
    # 单步预测
    pred = inferencer.predict_single(condition)
    
    # 自回归24步预测
    preds = inferencer.predict_autoregressive(condition, num_steps=24)
"""
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class CFMInferencer:
    """
    Flow Matching 推理器
    
    支持:
    - 单步采样
    - 自回归多步预测
    - 多种 Euler 模式
    """
    
    def __init__(
        self,
        model,
        data_cfg,
        infer_cfg=None,
        norm_mean: np.ndarray = None,
        norm_std: np.ndarray = None,
        device: torch.device = None,
    ):
        """
        Args:
            model: ERA5FlowMatchingModel
            data_cfg: DataConfig
            infer_cfg: InferenceConfig (可选)
            norm_mean: 归一化均值 (可选)
            norm_std: 归一化标准差 (可选)
            device: 设备 (可选)
        """
        self.model = model
        self.data_cfg = data_cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置
        if infer_cfg is not None:
            self.euler_steps = getattr(infer_cfg, 'euler_steps', 4)
            self.euler_mode = getattr(infer_cfg, 'euler_mode', 'midpoint')
            self.autoregressive_steps = getattr(infer_cfg, 'autoregressive_steps', 24)
            self.autoregressive_noise_sigma = getattr(infer_cfg, 'autoregressive_noise_sigma', 0.05)
            self.clamp_range = getattr(infer_cfg, 'clamp_range', (-5.0, 5.0))
            # ===== P4: z 通道 clamp =====
            self.use_z_clamp = getattr(infer_cfg, 'use_z_clamp', True)
            self.z_clamp_range = getattr(infer_cfg, 'z_clamp_range', (-1.0, 1.0))
        else:
            self.euler_steps = 4
            self.euler_mode = 'midpoint'
            self.autoregressive_steps = 24
            self.autoregressive_noise_sigma = 0.05
            self.clamp_range = (-5.0, 5.0)
            self.use_z_clamp = True
            self.z_clamp_range = (-1.0, 1.0)
        
        # z 通道索引: 通道顺序 [u_850, u_500, u_250, v_850, v_500, v_250, z_850, z_500, z_250]
        # z 通道在索引 6, 7, 8
        self.z_channel_indices = [6, 7, 8]
        
        # 归一化参数
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        
        self.model.eval()
    
    @torch.no_grad()
    def _sample_step(
        self,
        x_t: torch.Tensor,
        t: float,
        condition: torch.Tensor,
        return_velocity: bool = False,
    ) -> torch.Tensor:
        """
        单步采样 (Flow Matching)
        
        路径: x_t = (1-t) * x_0 + t * x_1
        速度: v = dx_t/dt = x_1 - x_0
        预测: v_pred = model(x_t, t, condition)
        更新: x_{t-dt} = x_t - dt * v_pred (当 t > dt 时)
             x_1 = x_t + v_pred (当 t ≈ 1 时)
        
        Args:
            x_t: 当前状态 (B, C, H, W)
            t: 当前时间 (0 ~ 1, 1=终点)
            condition: 条件 (B, T, C, H, W) 或 (B, C, H, W)
            return_velocity: 是否返回速度场
        
        Returns:
            x_{t-dt} (下一步状态) 或 (x_1, v_pred)
        """
        # 预处理 condition
        if condition.ndim == 5:
            # (B, T, C, H, W) -> (B, C, H, W) via Conv3D
            B, T, C, H, W = condition.shape
            condition_3d = condition.permute(0, 2, 1, 3, 4)
            cond_processed = self.model.cond_encoder.temporal_conv3d(condition_3d)
            cond_processed = cond_processed.squeeze(2)
            # 残差连接
            last_frame = condition_3d[:, :, -1, :, :]
            cond_processed = cond_processed + last_frame
        else:
            cond_processed = condition
        
        # 预测速度
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=x_t.dtype)
        v_pred = self.model.dit(x_t, t_tensor, cond_processed)
        
        if return_velocity:
            return v_pred
        
        # Euler 更新
        if t >= 1.0 - 1e-6:
            # t ≈ 1: 终点的下一个是 x_1
            x_next = x_t + v_pred
        else:
            # t < 1: x_{t-dt} = x_t - dt * v_pred
            dt = 1.0 / self.euler_steps
            x_next = x_t - dt * v_pred
        
        return x_next
    
    @torch.no_grad()
    def _sample_heun(
        self,
        x_start: torch.Tensor,
        t_start: float,
        t_end: float,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Heun 方法 (二阶 ODE 求解器)
        
        比简单 Euler 更精确，推荐用于高质量预测
        
        Steps:
            k1 = f(x, t)
            k2 = f(x + dt*k1, t + dt)
            x_next = x + dt/2 * (k1 + k2)
        """
        dt = t_end - t_start
        
        # k1
        k1 = self._sample_step(x_start, t_start, condition, return_velocity=True)
        
        # x + dt * k1
        x_temp = x_start + dt * k1
        
        # k2
        k2 = self._sample_step(x_temp, t_end, condition, return_velocity=True)
        
        # Heun 更新
        x_next = x_start + dt / 2 * (k1 + k2)
        
        return x_next
    
    @torch.no_grad()
    def _sample_midpoint(
        self,
        x_start: torch.Tensor,
        t_start: float,
        t_end: float,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Midpoint 方法 (二阶 ODE 求解器)
        
        Steps:
            k1 = f(x, t)
            k2 = f(x + dt/2 * k1, t + dt/2)
            x_next = x + dt * k2
        """
        dt = t_end - t_start
        t_mid = t_start + dt / 2
        
        # k1
        k1 = self._sample_step(x_start, t_start, condition, return_velocity=True)
        
        # x + dt/2 * k1
        x_mid = x_start + dt / 2 * k1
        
        # k2
        k2 = self._sample_step(x_mid, t_mid, condition, return_velocity=True)
        
        # Midpoint 更新
        x_next = x_start + dt * k2
        
        return x_next
    
    @torch.no_grad()
    def _sample_euler(
        self,
        x_start: torch.Tensor,
        t_start: float,
        t_end: float,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        简单 Euler 方法 (一阶 ODE 求解器)
        
        Steps:
            x_next = x + dt * f(x, t)
        """
        dt = t_end - t_start
        k1 = self._sample_step(x_start, t_start, condition, return_velocity=True)
        x_next = x_start + dt * k1
        return x_next
    
    @torch.no_grad()
    def _sample_single(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        单步完整采样 (从 x_1 到 x_0)
        
        使用指定的 ODE 求解器进行多步积分
        
        Args:
            x_start: 起始状态 (B, C, H, W) - 通常是 x_1 (带噪声的终点)
            condition: 条件 (B, T, C, H, W)
        
        Returns:
            x_0 预测 (B, C, H, W)
        """
        x_t = x_start
        t = 1.0
        dt = 1.0 / self.euler_steps
        
        for step in range(self.euler_steps):
            t_next = max(t - dt, 0.0)
            
            if self.euler_mode == 'heun':
                x_t = self._sample_heun(x_t, t, t_next, condition)
            elif self.euler_mode == 'midpoint':
                x_t = self._sample_midpoint(x_t, t, t_next, condition)
            else:  # euler
                x_t = self._sample_euler(x_t, t, t_next, condition)
            
            t = t_next
            
            # Clamp
            if self.clamp_range is not None:
                x_t = torch.clamp(x_t, *self.clamp_range)
        
        return x_t
    
    def predict_single(
        self,
        condition: torch.Tensor,
        x_start: torch.Tensor = None,
        return_velocity: bool = False,
    ) -> torch.Tensor:
        """
        单步预测 (用于单帧 ERA5 预测)
        
        Args:
            condition: 条件 (B, T, C, H, W) 或 (B, C, H, W)
            x_start: 起始状态 (可选，默认使用随机 x_1)
            return_velocity: 是否返回速度场
        
        Returns:
            x_0 预测 或 (x_0, v_pred)
        """
        if condition.ndim == 5:
            B, T, C, H, W = condition.shape
        else:
            B, C, H, W = condition.shape
            T = None
        
        device = condition.device
        
        # 如果没有提供 x_start，使用随机起点
        if x_start is None:
            x_start = torch.randn(B, C, H, W, device=device)
        
        # 完整采样
        x_0_pred = self._sample_single(x_start, condition)
        
        if return_velocity:
            # 返回预测的速度场
            if condition.ndim == 5:
                cond_processed = self._prepare_condition(condition)
            else:
                cond_processed = condition
            t_tensor = torch.ones(B, device=device)
            v_pred = self.model.dit(x_0_pred, t_tensor, cond_processed)
            return x_0_pred, v_pred
        
        return x_0_pred
    
    def _prepare_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """预处理条件输入"""
        if condition.ndim == 5:
            B, T, C, H, W = condition.shape
            condition_3d = condition.permute(0, 2, 1, 3, 4)
            cond_processed = self.model.cond_encoder.temporal_conv3d(condition_3d)
            cond_processed = cond_processed.squeeze(2)
            last_frame = condition_3d[:, :, -1, :, :]
            cond_processed = cond_processed + last_frame
            return cond_processed
        return condition
    
    def predict_autoregressive(
        self,
        condition: torch.Tensor,
        num_steps: int = 24,
        noise_sigma: float = 0.0,
    ) -> List[torch.Tensor]:
        """
        自回归多步预测
        
        每次预测下一帧，然后将预测结果加入条件窗口
        
        Args:
            condition: 初始条件 (B, T_history, C, H, W)
                      例如: (B, 16, 9, 40, 40) - 16帧历史
            num_steps: 预测步数 (默认24步 = 72小时)
            noise_sigma: 噪声扰动 (可选，用于集成)
        
        Returns:
            predictions: List of (B, C, H, W), 长度 = num_steps
        """
        assert condition.ndim == 5, f"condition 需要是 5D (B, T, C, H, W), 得到 {condition.ndim}D"
        
        B, T_history, C, H, W = condition.shape
        device = condition.device
        
        # 初始化: 从 x_1 (带噪声的终点) 开始
        x_t = torch.randn(B, C, H, W, device=device)
        
        # 当前条件窗口 (滑动)
        current_condition = condition.clone()  # (B, T_history, C, H, W)
        
        predictions = []
        
        # z 通道的 Delta Clamp: 限制相邻步之间 z 的逐像素变化量
        z_delta_max = 0.5  # 归一化空间中每步最大变化
        z_prev = None  # 上一步的 z 通道预测
        
        logger.info(f"自回归预测: {num_steps} 步, noise_sigma={noise_sigma}")
        
        for step in range(num_steps):
            # 添加可选噪声
            if noise_sigma > 0:
                x_t_noisy = x_t + torch.randn_like(x_t) * noise_sigma
            else:
                x_t_noisy = x_t
            
            # 采样得到 x_0_pred
            x_0_pred = self._sample_single(x_t_noisy, current_condition)
            
            # 全局 Clamp
            if self.clamp_range is not None:
                x_0_pred = torch.clamp(x_0_pred, *self.clamp_range)
            
            # ===== Z 通道 Delta Clamp: 防止累积误差 =====
            if self.use_z_clamp and self.z_channel_indices and z_prev is not None:
                z_new = x_0_pred[:, self.z_channel_indices]
                z_delta = z_new - z_prev
                z_delta_clamped = z_delta.clamp(-z_delta_max, z_delta_max)
                x_0_pred = x_0_pred.clone()
                x_0_pred[:, self.z_channel_indices] = z_prev + z_delta_clamped
            
            # 更新 z_prev (用于下一步的 delta clamp)
            if self.z_channel_indices:
                z_prev = x_0_pred[:, self.z_channel_indices].clone()
            
            # z 通道额外 clamp
            if self.z_channel_indices and self.z_clamp_range:
                x_0_pred[:, self.z_channel_indices] = x_0_pred[:, self.z_channel_indices].clamp(
                    *self.z_clamp_range
                )
            
            predictions.append(x_0_pred)
            
            # 更新条件窗口: 滑动 (扔掉最老的一帧, 加入新预测的一帧)
            if T_history > 1:
                # current_condition: (B, T_history, C, H, W)
                # 滑动窗口更新
                current_condition = torch.cat([
                    current_condition[:, 1:, :, :, :],  # 去掉最老的一帧
                    x_0_pred.unsqueeze(1),  # 加入新预测的一帧
                ], dim=1)
            
            # 更新 x_t 为预测的 x_1 (下一轮的起点)
            x_t = x_0_pred
            
            if (step + 1) % 8 == 0 or step == 0 or step == num_steps - 1:
                logger.info(f"  Step {step+1}/{num_steps} 完成")
        
        logger.info(f"自回归预测完成, 生成了 {len(predictions)} 帧")
        
        return predictions
    
    def predict_autoregressive_batch(
        self,
        conditions: torch.Tensor,
        num_steps: int = 24,
        noise_sigma: float = 0.0,
    ) -> torch.Tensor:
        """
        自回归多步预测 (批量版本)
        
        Args:
            conditions: 初始条件 (B, T_history, C, H, W)
            num_steps: 预测步数
            noise_sigma: 噪声扰动
        
        Returns:
            predictions: (B, num_steps, C, H, W)
        """
        B = conditions.shape[0]
        preds = self.predict_autoregressive(conditions, num_steps, noise_sigma)
        return torch.stack(preds, dim=1)  # (B, num_steps, C, H, W)
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        反归一化
        
        Args:
            data: (B, C, H, W) 或 (B, T, C, H, W) 归一化数据
        
        Returns:
            物理值数据
        """
        if self.norm_mean is None or self.norm_std is None:
            logger.warning("未提供归一化参数，跳过反归一化")
            return data
        
        mean = torch.from_numpy(self.norm_mean).float().to(data.device)
        std = torch.from_numpy(self.norm_std).float().to(data.device)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        
        if data.ndim == 5:
            mean = mean.reshape(1, 1, -1, 1, 1)
            std = std.reshape(1, 1, -1, 1, 1)
        elif data.ndim == 4:
            mean = mean.reshape(1, -1, 1, 1)
            std = std.reshape(1, -1, 1, 1)
        
        return data * std + mean


class CFMPredictor:
    """
    Flow Matching 预测器 (兼容 ERA5Predictor 接口)
    
    提供与 Diffusion 模型类似的 predict_autoregressive 接口
    """
    
    def __init__(
        self,
        model,
        data_cfg,
        infer_cfg=None,
        norm_mean: np.ndarray = None,
        norm_std: np.ndarray = None,
        device: torch.device = None,
    ):
        self.inferencer = CFMInferencer(
            model, data_cfg, infer_cfg, norm_mean, norm_std, device
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @torch.no_grad()
    def predict_autoregressive(
        self,
        condition: torch.Tensor,
        num_steps: int = 24,
        noise_sigma: float = 0.05,
    ) -> List[torch.Tensor]:
        """
        自回归多步预测 (兼容 ERA5Predictor 接口)
        
        Args:
            condition: (B, T, C, H, W) 初始条件
            num_steps: 预测步数
            noise_sigma: 噪声扰动
        
        Returns:
            List of (B, C, H, W), 长度 = num_steps
        """
        return self.inferencer.predict_autoregressive(
            condition, num_steps, noise_sigma
        )
    
    @torch.no_grad()
    def predict_single(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        单步预测
        
        Args:
            condition: (B, T, C, H, W) 条件
        
        Returns:
            (B, C, H, W) 预测
        """
        return self.inferencer.predict_single(condition)


# ============================================================
# 工具函数
# ============================================================

def compute_rmse(pred: torch.Tensor, target: torch.Tensor, channel_weights: np.ndarray = None) -> float:
    """
    计算 RMSE
    
    Args:
        pred: (B, C, H, W)
        target: (B, C, H, W)
        channel_weights: (C,) 可选通道权重
    
    Returns:
        RMSE 值
    """
    if channel_weights is not None:
        weights = torch.from_numpy(channel_weights).float().to(pred.device)
        mse = ((pred - target) ** 2 * weights.view(1, -1, 1, 1)).mean()
    else:
        mse = ((pred - target) ** 2).mean()
    return torch.sqrt(mse).item()


def compute_channel_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    计算各通道的 RMSE
    
    Returns:
        Dict[str, float]: 各变量的 RMSE
    """
    channel_names = ['u_850', 'u_500', 'u_250', 'v_850', 'v_500', 'v_250', 
                     'z_850', 'z_500', 'z_250']
    
    results = {}
    for i, name in enumerate(channel_names[:pred.shape[1]]):
        mse = ((pred[:, i] - target[:, i]) ** 2).mean()
        results[name] = torch.sqrt(mse).item()
    
    return results
