"""
LT3P风格台风轨迹预测模型
基于论文: Long-Term Typhoon Trajectory Prediction: A Physics-Conditioned Approach

架构:
- 输入: 历史轨迹(48h, 16点) + 未来ERA5气象场(72h, 24点)
- 输出: 未来轨迹坐标(72h, 24点)
- 核心: 物理条件编码器R(·) + 轨迹预测器D(·) + 可选偏差校正器B(·)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from config import model_cfg, data_cfg


# ============== 基础模块 ==============

class PositionalEncoding(nn.Module):
    """序列位置编码"""
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class LeadTimeEncoding(nn.Module):
    """Lead Time编码 - 用于区分不同预测时长"""
    def __init__(self, d_model: int, max_lead_time: int = 24):
        super().__init__()
        self.embedding = nn.Embedding(max_lead_time, d_model)
    
    def forward(self, t_future: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """返回 (B, T_future, d_model) 的lead time编码"""
        lead_times = torch.arange(t_future, device=device).unsqueeze(0).expand(batch_size, -1)
        return self.embedding(lead_times)


# ============== 物理条件编码器 R(·) ==============

class PhysicsEncoder3D(nn.Module):
    """
    3D卷积物理条件编码器 - 编码ERA5时空特征
    论文使用3D Conv + 3D Transformer处理时空数据

    输入: (B, T, C, H, W) - T个时间步的ERA5气象场
    输出: (B, T, D) - 时空特征表示
    """
    def __init__(
        self,
        in_channels: int = 9,      # 3变量(u,v,z) × 3压力层(850,500,250) = 9通道
        base_channels: int = 64,
        out_dim: int = 256
    ):
        super().__init__()

        # 3D卷积编码器 (时间, 高度, 宽度)
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )
        
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
        )
        
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_channels * 4),
            nn.GELU(),
        )
        
        self.conv3d_4 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_channels * 4),
            nn.GELU(),
        )
        
        # 空间注意力池化（替代全局平均池化，保留台风位置信息）
        # 每个时间步学习一个空间注意力图，加权聚合而非简单平均
        # 这样模型能区分"引导气流在台风哪一侧更强"
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(base_channels * 4, 1, kernel_size=1),  # (B, 1, T, H', W')
        )

        self.proj = nn.Sequential(
            nn.Linear(base_channels * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, era5: torch.Tensor) -> torch.Tensor:
        """
        Args:
            era5: (B, T, C, H, W) ERA5气象场序列
        Returns:
            (B, T, out_dim) 物理特征表示
        """
        B, T, C, H, W = era5.shape
        
        # 调整维度为3D卷积格式: (B, C, T, H, W)
        x = era5.permute(0, 2, 1, 3, 4)
        
        # 3D卷积
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.conv3d_4(x)

        # 空间注意力池化: 学习每个位置的重要性权重
        # x: (B, C', T, H', W')
        attn_logits = self.spatial_attn(x)  # (B, 1, T, H', W')
        attn_weights = torch.softmax(
            attn_logits.flatten(3), dim=-1  # (B, 1, T, H'*W')
        ).unflatten(3, attn_logits.shape[3:])  # (B, 1, T, H', W')

        # 加权聚合: (B, C', T, H', W') * (B, 1, T, H', W') -> sum over H',W'
        x = (x * attn_weights).sum(dim=(-2, -1))  # (B, C', T)

        # 调整回序列格式: (B, T, C')
        x = x.permute(0, 2, 1)
        
        # 投影到目标维度
        x = self.proj(x)
        
        return x


# ============== 轨迹编码器 ==============

class TrajectoryEncoder(nn.Module):
    """
    轨迹坐标编码器
    将历史轨迹坐标编码为特征序列
    """
    def __init__(self, coord_dim: int = 2, embed_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos_enc = PositionalEncoding(embed_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, T_history, coord_dim) 历史轨迹坐标
        Returns:
            (B, T_history, embed_dim) 轨迹特征
        """
        x = self.mlp(coords)
        x = self.pos_enc(x)
        return x


# ============== 运动特征编码器 ==============

class MotionEncoder(nn.Module):
    """
    从历史轨迹中提取运动特征（速度、加速度）
    为模型提供方向和动量信息，使预测与历史运动趋势一致
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.vel_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.accel_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # 融合门控
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, history_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_coords: (B, T, 2) 历史轨迹坐标
        Returns:
            (B, T, d_model) 运动特征嵌入
        """
        # 计算速度 (B, T-1, 2)
        velocity = torch.diff(history_coords, dim=1)
        # 补齐第一步（复制第一个速度）
        velocity = torch.cat([velocity[:, :1], velocity], dim=1)  # (B, T, 2)

        # 计算加速度 (B, T-1, 2)
        acceleration = torch.diff(velocity, dim=1)
        acceleration = torch.cat([acceleration[:, :1], acceleration], dim=1)  # (B, T, 2)

        vel_embed = self.vel_proj(velocity)      # (B, T, d_model)
        accel_embed = self.accel_proj(acceleration)  # (B, T, d_model)

        # 门控融合
        gate_input = torch.cat([vel_embed, accel_embed], dim=-1)
        gate = self.gate(gate_input)
        motion_embed = gate * vel_embed + (1 - gate) * accel_embed

        return motion_embed


# ============== 轨迹预测器 D(·) ==============

class TrajectoryPredictor(nn.Module):
    """
    Transformer-based 轨迹预测器
    使用Cross-Attention融合历史轨迹和未来物理场信息
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        t_future: int = 24,
        output_dim: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.t_future = t_future
        self.output_dim = output_dim
        
        # 可学习的未来轨迹查询向量
        self.future_queries = nn.Parameter(torch.randn(1, t_future, d_model) * 0.02)
        
        # Lead time编码
        self.lead_time_enc = LeadTimeEncoding(d_model, max_lead_time=t_future)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )
        
    def forward(
        self,
        history_embed: torch.Tensor,   # (B, T_history, d_model) 历史轨迹特征
        physics_embed: torch.Tensor,   # (B, T_future, d_model) 未来物理场特征
    ) -> torch.Tensor:
        """
        Args:
            history_embed: 历史轨迹编码
            physics_embed: 未来ERA5物理场编码
        Returns:
            (B, T_future, output_dim) 预测的未来坐标
        """
        B = history_embed.shape[0]
        device = history_embed.device
        
        # 拼接条件: 历史轨迹 + 未来物理场
        memory = torch.cat([history_embed, physics_embed], dim=1)  # (B, T_hist + T_fut, d_model)
        
        # 准备查询向量
        queries = self.future_queries.expand(B, -1, -1)  # (B, T_future, d_model)

        # 添加lead time编码
        lead_time_emb = self.lead_time_enc(self.t_future, B, device)
        queries = queries + lead_time_emb
        
        # Transformer解码
        decoded = self.decoder(queries, memory)  # (B, T_future, d_model)
        
        # 输出投影
        output = self.output_proj(decoded)  # (B, T_future, output_dim)
        
        return output


# ============== 完整LT3P模型 ==============

class LT3PModel(nn.Module):
    """
    LT3P风格台风轨迹预测模型
    
    输入:
        - history_coords: (B, T_history, 2) 过去48小时轨迹坐标 [lat, lon]
        - future_era5: (B, T_future, C, H, W) 未来72小时ERA5气象场
    
    输出:
        - predicted_coords: (B, T_future, 2) 未来72小时轨迹坐标 [lat, lon]
    """
    
    def __init__(
        self,
        coord_dim: int = 2,
        output_dim: int = 2,
        era5_channels: int = 9,
        t_history: int = 16,
        t_future: int = 24,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t_history = t_history
        self.t_future = t_future
        self.output_dim = output_dim

        # 轨迹编码器
        coord_embed_dim = d_model // 2
        self.trajectory_encoder = TrajectoryEncoder(coord_dim, coord_embed_dim)

        # 轨迹特征投影到d_model
        self.traj_proj = nn.Linear(coord_embed_dim, d_model)

        # 运动特征编码器（提取速度/加速度信息）
        self.motion_encoder = MotionEncoder(d_model)

        # 未来物理条件编码器 R(·) — 编码未来72h ERA5
        self.physics_encoder = PhysicsEncoder3D(
            in_channels=era5_channels,
            base_channels=64,
            out_dim=d_model
        )

        # 过去物理条件编码器 — 编码过去48h ERA5 (独立参数)
        self.past_physics_encoder = PhysicsEncoder3D(
            in_channels=era5_channels,
            base_channels=64,
            out_dim=d_model
        )

        # past ERA5 特征投影到 history 时间维度上融合
        # past_physics_encoder 输出 (B, T_history, d_model)
        # 与 history_embed 直接相加

        # 轨迹预测器 D(·)
        self.trajectory_predictor = TrajectoryPredictor(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            t_future=t_future,
            output_dim=output_dim
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        history_coords: torch.Tensor,
        future_era5: torch.Tensor,
        target_coords: torch.Tensor = None,
        past_era5: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练/推理前向传播（残差预测架构）

        核心改进:
        1. 从历史轨迹线性外推得到baseline
        2. 模型只预测对baseline的修正量（residual）
        3. 最终预测 = baseline + residual
        4. 多种物理约束损失确保轨迹合理

        Args:
            history_coords: (B, T_history, 2) 历史轨迹坐标（归一化[0,1]）
            future_era5: (B, T_future, C, H, W) 未来ERA5气象场
            target_coords: (B, T_future, 2) 目标坐标（训练时使用）
            past_era5: (B, T_history, C, H, W) 过去ERA5气象场（可选）

        Returns:
            dict containing predicted_coords, loss, and loss components
        """
        B, T_hist, _ = history_coords.shape
        device = history_coords.device

        # ===== 1. 编码历史轨迹 =====
        history_embed = self.trajectory_encoder(history_coords)  # (B, T_history, embed_dim)
        history_embed = self.traj_proj(history_embed)  # (B, T_history, d_model)

        # 添加运动特征（速度/加速度信息）
        motion_embed = self.motion_encoder(history_coords)  # (B, T_history, d_model)
        history_embed = history_embed + motion_embed  # 融合位置和运动信息

        # 添加过去物理场特征（如果提供）
        if past_era5 is not None:
            past_physics_embed = self.past_physics_encoder(past_era5)  # (B, T_history, d_model)
            history_embed = history_embed + past_physics_embed  # 融合过去环境信息

        # ===== 2. 编码未来物理场 =====
        physics_embed = self.physics_encoder(future_era5)  # (B, T_future, d_model)

        # ===== 3. 计算线性外推基线 =====
        # 使用最近4步的平均速度，更稳定
        last_pos = history_coords[:, -1:, :]  # (B, 1, 2)
        recent_vels = torch.diff(history_coords[:, -4:], dim=1)  # (B, 3, 2)
        avg_vel = recent_vels.mean(dim=1, keepdim=True)  # (B, 1, 2)

        steps = torch.arange(1, self.t_future + 1, device=device).float().view(1, -1, 1)
        linear_baseline = last_pos + avg_vel * steps  # (B, T_future, 2)

        # ===== 4. 预测残差修正量 =====
        residual = self.trajectory_predictor(history_embed, physics_embed)  # (B, T_future, 2)

        # ===== 5. 最终预测 = 基线 + 修正 =====
        predicted_coords = linear_baseline + residual
        # 裁剪到有效归一化范围 [0, 1]
        predicted_coords = predicted_coords.clamp(0.0, 1.0)

        outputs = {
            'predicted_coords': predicted_coords,
            'linear_baseline': linear_baseline,
            'residual': residual,
        }

        # ===== 6. 计算多组件损失 =====
        if target_coords is not None:
            T = target_coords.shape[1]

            # --- 6a. 时序加权 MSE 损失（远期权重更高）---
            time_weights = torch.linspace(1.0, 1.5, T, device=device).view(1, T, 1)
            weighted_mse = ((predicted_coords - target_coords) ** 2 * time_weights).mean()

            # --- 6b. 连续性损失：第一个预测点应该接近最后一个历史点 ---
            continuity_loss = F.mse_loss(
                predicted_coords[:, 0], history_coords[:, -1]
            )

            # --- 6c. 方向一致性损失：预测初始方向应与历史运动方向一致 ---
            # 历史最后一步的运动方向
            hist_dir = history_coords[:, -1] - history_coords[:, -2]  # (B, 2)
            # 预测第一步的运动方向
            pred_dir = predicted_coords[:, 0] - history_coords[:, -1]  # (B, 2)

            # 用余弦相似度衡量方向一致性
            hist_dir_norm = F.normalize(hist_dir, dim=-1, eps=1e-8)
            pred_dir_norm = F.normalize(pred_dir, dim=-1, eps=1e-8)
            cos_sim = (hist_dir_norm * pred_dir_norm).sum(dim=-1)  # (B,)
            direction_loss = (1 - cos_sim).clamp(min=0).mean()

            # --- 6d. 曲率约束：惩罚急转弯 ---
            pred_segments = torch.diff(predicted_coords, dim=1)  # (B, T-1, 2)
            if pred_segments.shape[1] > 1:
                seg1 = F.normalize(pred_segments[:, :-1], dim=-1, eps=1e-8)  # (B, T-2, 2)
                seg2 = F.normalize(pred_segments[:, 1:], dim=-1, eps=1e-8)   # (B, T-2, 2)
                cos_angles = (seg1 * seg2).sum(dim=-1)  # (B, T-2)
                # 惩罚超过90度的转弯（cos < 0）
                curvature_loss = F.relu(-cos_angles).mean()
            else:
                curvature_loss = torch.tensor(0.0, device=device)

            # --- 6e. 速度约束：限制每步最大位移 ---
            # 台风极限速度约 150-180km/3h ≈ 纬度 1.6°
            # 归一化坐标下: 1.6° / 60°(lat_range宽度) ≈ 0.027, 留余量取 0.03
            step_sizes = torch.norm(pred_segments, dim=-1)  # (B, T-1)
            max_step = 0.03
            speed_penalty = F.relu(step_sizes - max_step).mean()

            # --- 6f. 残差平滑性约束（防止振荡/绕圈的核心）---
            # 残差的二阶差分 = 残差的"加速度"
            # 如果残差平滑递增，二阶差分接近0 → 允许（正常转弯）
            # 如果残差来回振荡，二阶差分很大 → 惩罚（绕圈的根源）
            if residual.shape[1] > 2:
                residual_accel = torch.diff(residual, n=2, dim=1)  # (B, T-2, 2)
                residual_smooth = (residual_accel ** 2).mean()
            else:
                residual_smooth = torch.tensor(0.0, device=device)

            # --- 6g. 振荡检测（叉积符号变化 = 左右摇摆 = 绕圈前兆）---
            # 注意：只惩罚连续方向反转（zigzag），不惩罚持续转弯（recurvature）
            if pred_segments.shape[1] > 2:
                cross_z = (pred_segments[:, :-1, 0] * pred_segments[:, 1:, 1] -
                           pred_segments[:, :-1, 1] * pred_segments[:, 1:, 0])  # (B, T-2)
                # 相邻叉积符号相反 → 方向来回摆动（zigzag）
                sign_product = cross_z[:, :-1] * cross_z[:, 1:]  # (B, T-3)
                oscillation_loss = F.relu(-sign_product).mean()
            else:
                oscillation_loss = torch.tensor(0.0, device=device)

            # --- 6h. 残差幅度约束（温和，允许模型做修正但不要太疯）---
            residual_l2 = (residual ** 2).mean()

            # --- 汇总总损失 ---
            loss = (
                weighted_mse                     # 主损失
                + 5.0 * continuity_loss           # 起点连续
                + 2.0 * direction_loss            # 方向一致
                + 1.5 * curvature_loss            # 禁止急转弯(>90°)
                + 3.0 * speed_penalty             # 速度限制
                + 1.0 * residual_smooth           # 残差平滑 (3.0→1.0, 释放修正能力)
                + 0.5 * oscillation_loss          # 温和防振荡 (1.0→0.5)
                + 0.1 * residual_l2               # 残差约束 (0.3→0.1, 允许更大修正)
            )

            outputs['loss'] = loss
            outputs['mse_loss'] = weighted_mse
            outputs['continuity_loss'] = continuity_loss
            outputs['direction_loss'] = direction_loss
            outputs['curvature_loss'] = curvature_loss
            outputs['speed_penalty'] = speed_penalty
            outputs['smooth_loss'] = residual_smooth
            outputs['oscillation_loss'] = oscillation_loss
            outputs['residual_l2'] = residual_l2

        return outputs
    
    @torch.no_grad()
    def predict(
        self,
        history_coords: torch.Tensor,
        future_era5: torch.Tensor,
        past_era5: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        推理预测（含后处理去环）

        Returns:
            dict containing predicted coordinates
        """
        self.eval()
        outputs = self.forward(history_coords, future_era5, past_era5=past_era5)

        pred_coords = outputs['predicted_coords']

        # 后处理：检测并平滑绕圈轨迹
        pred_coords = self._smooth_loops(pred_coords)

        return {
            'predicted_coords': pred_coords,
            'predicted_lat': pred_coords[:, :, 0],
            'predicted_lon': pred_coords[:, :, 1],
        }

    @torch.no_grad()
    def _smooth_loops(self, coords: torch.Tensor) -> torch.Tensor:
        """
        后处理：
        1) 检测线段自交 → 剪掉环路（硬性消除）
        2) 迭代检测尖锐转弯 → 邻居均值平滑（兜底）
        """
        B, T, D = coords.shape
        result = coords.clone()

        # ===== 阶段1: 自交检测与环路剪除 =====
        result = self._remove_self_intersections(result)

        # ===== 阶段2: 迭代平滑尖锐转弯 =====
        for _ in range(5):
            segments = torch.diff(result, dim=1)  # (B, T-1, 2)
            if segments.shape[1] < 2:
                break

            dot = (segments[:, :-1] * segments[:, 1:]).sum(dim=-1)
            norm1 = segments[:, :-1].norm(dim=-1) + 1e-8
            norm2 = segments[:, 1:].norm(dim=-1) + 1e-8
            cos_angle = dot / (norm1 * norm2)

            sharp = cos_angle < 0.0
            if not sharp.any():
                break

            new_result = result.clone()
            for t in range(sharp.shape[1]):
                mask = sharp[:, t]
                if mask.any():
                    pt = t + 1
                    avg = 0.5 * (result[:, pt - 1] + result[:, pt + 1])
                    new_result[:, pt] = torch.where(
                        mask.unsqueeze(-1).expand(-1, D), avg, result[:, pt]
                    )
            result = new_result

        return result.clamp(0.0, 1.0)

    @torch.no_grad()
    def _remove_self_intersections(self, coords: torch.Tensor) -> torch.Tensor:
        """
        检测轨迹线段自交并剪除环路。
        如果线段 i->(i+1) 与线段 j->(j+1) 交叉 (j > i+1),
        则 i+1 到 j 之间的点形成一个环，用线性插值替换。
        """
        B, T, D = coords.shape
        result = coords.clone()

        for b in range(B):
            pts = result[b]  # (T, 2)
            i = 0
            while i < T - 2:
                found = False
                # 检查线段 i→i+1 是否与后续线段 j→j+1 交叉
                for j in range(i + 2, T - 1):
                    if self._segments_intersect(
                        pts[i], pts[i + 1], pts[j], pts[j + 1]
                    ):
                        # 发现环路 [i+1 ... j]，用线性插值替换
                        loop_len = j - i
                        for k in range(1, loop_len):
                            alpha = k / loop_len
                            result[b, i + k] = (1 - alpha) * pts[i] + alpha * pts[j + 1]
                        pts = result[b]  # 更新引用
                        i = j  # 跳过已处理的部分
                        found = True
                        break
                if not found:
                    i += 1

        return result

    @staticmethod
    def _segments_intersect(
        p1: torch.Tensor, p2: torch.Tensor,
        p3: torch.Tensor, p4: torch.Tensor,
    ) -> bool:
        """判断线段 p1p2 与线段 p3p4 是否相交（2D 叉积法）"""
        d1 = p2 - p1
        d2 = p4 - p3

        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross.item()) < 1e-10:
            return False  # 平行

        d3 = p3 - p1
        t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
        u = (d3[0] * d1[1] - d3[1] * d1[0]) / cross

        return 0 < t.item() < 1 and 0 < u.item() < 1


# ============== 带扩散增强的LT3P模型 ==============

class LT3PDiffusionModel(LT3PModel):
    """
    LT3P + Diffusion: 在LT3P基础上增加扩散模型的不确定性估计
    
    可以选择:
    1. 使用确定性预测 (mode='deterministic')
    2. 使用扩散采样进行集合预测 (mode='diffusion')
    """
    
    def __init__(self, *args, num_diffusion_steps: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 扩散相关组件
        self.num_diffusion_steps = num_diffusion_steps
        
        # 噪声预测网络（轻量级）
        d_model = kwargs.get('d_model', 256)
        self.noise_pred = nn.Sequential(
            nn.Linear(self.output_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.output_dim),
        )
        
        # Beta schedule
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, num_diffusion_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
    
    @torch.no_grad()
    def sample_ensemble(
        self,
        history_coords: torch.Tensor,
        future_era5: torch.Tensor,
        num_samples: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        集合采样预测
        
        Returns:
            dict with mean prediction and all samples
        """
        self.eval()
        
        # 先获取确定性预测作为基准
        base_pred = self.forward(history_coords, future_era5)['predicted_coords']
        
        all_samples = [base_pred]
        
        # 添加扰动采样
        for _ in range(num_samples - 1):
            noise = torch.randn_like(base_pred) * 0.01  # 小扰动
            perturbed = base_pred + noise
            all_samples.append(perturbed)
        
        samples = torch.stack(all_samples, dim=0)  # (num_samples, B, T, 2)
        mean_pred = samples.mean(dim=0)
        std_pred = samples.std(dim=0)
        
        return {
            'predicted_coords': mean_pred,
            'predicted_lat': mean_pred[:, :, 0],
            'predicted_lon': mean_pred[:, :, 1],
            'uncertainty': std_pred,
            'all_samples': samples,
        }


# ============== 工厂函数 ==============

def create_lt3p_model(use_diffusion: bool = False) -> nn.Module:
    """创建LT3P模型"""
    
    common_kwargs = {
        'coord_dim': model_cfg.coord_dim,
        'output_dim': model_cfg.output_dim,
        'era5_channels': model_cfg.era5_channels,
        't_history': model_cfg.t_history,
        't_future': model_cfg.t_future,
        'd_model': model_cfg.transformer_dim,
        'n_heads': model_cfg.transformer_heads,
        'n_layers': model_cfg.transformer_layers,
        'ff_dim': model_cfg.transformer_ff_dim,
        'dropout': model_cfg.dropout,
    }
    
    if use_diffusion:
        return LT3PDiffusionModel(
            **common_kwargs,
            num_diffusion_steps=model_cfg.num_diffusion_steps
        )
    else:
        return LT3PModel(**common_kwargs)
