"""
改进的 Flow Matching 损失函数
针对Z通道和UV风场分离优化的损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UVZSeparatedLoss(nn.Module):
    """
    分离UV和Z通道的损失函数
    """
    
    def __init__(
        self,
        uv_weight: float = 2.0,
        z_weight: float = 1.0,
        pressure_weights: tuple = (1.0, 1.2, 1.5),
    ):
        super().__init__()
        self.uv_weight = uv_weight
        self.z_weight = z_weight
        self.pressure_weights = torch.tensor(pressure_weights)
        
    def forward(self, pred, target, return_components=False):
        diff = pred - target
        B, C, H, W = diff.shape
        
        # UV通道 (0-5)
        uv_diff = diff[:, :6]
        pl_weights = self.pressure_weights.repeat(2).reshape(1, 6, 1, 1).to(diff.device)
        uv_mse = ((uv_diff ** 2) * pl_weights).mean()
        uv_loss = self.uv_weight * uv_mse
        
        # Z通道 (6-8)
        z_diff = diff[:, 6:9]
        z_pl_weights = self.pressure_weights.reshape(1, 3, 1, 1).to(diff.device)
        z_mse = ((z_diff ** 2) * z_pl_weights).mean()
        z_loss = self.z_weight * z_mse
        
        total_loss = uv_loss + z_loss
        
        if return_components:
            return total_loss, uv_loss, z_loss
        return total_loss


class GeostrophicBalanceLoss(nn.Module):
    """
    地转平衡损失 - 强制U/V与Z保持地转平衡
    """
    
    def __init__(self, lat_grid, lon_res=0.25, lat_res=0.25, weight=0.3):
        super().__init__()
        R, Omega = 6.371e6, 7.2921e-5
        f = 2 * Omega * torch.sin(lat_grid)
        f = torch.where(f.abs() < 1e-5, torch.sign(f + 1e-10) * 1e-5, f)
        self.register_buffer('f', f)
        self.dy = R * torch.pi / 180 * lat_res
        dx = R * torch.pi / 180 * lon_res * torch.cos(lat_grid)
        self.register_buffer('dx', dx)
        self.weight = weight
        
    def forward(self, u_pred, v_pred, z_pred):
        if u_pred.ndim == 4:
            B, n_pl, H, W = u_pred.shape
            losses = []
            for i in range(n_pl):
                loss = self._compute_geostrophic_loss(
                    u_pred[:, i], v_pred[:, i], z_pred[:, i]
                )
                losses.append(loss)
            return self.weight * torch.stack(losses).mean()
        else:
            return self.weight * self._compute_geostrophic_loss(u_pred, v_pred, z_pred)
    
    def _compute_geostrophic_loss(self, u, v, z):
        f = self.f.to(u.device)
        dx = self.dx.to(u.device)
        dy = self.dy
        
        dzdx = (z[:, :, 2:] - z[:, :, :-2]) / (2 * dx[:, :, 1:-1] + 1e-8)
        dzdy = (z[:, 2:, :] - z[:, :-2, :]) / (2 * dy + 1e-8)
        
        f_mid = f[1:-1, 1:-1].to(u.device)
        v_geo = dzdx[:, 1:-1, :] / (f_mid + 1e-8)
        u_geo = -dzdy[:, :, 1:-1] / (f_mid + 1e-8)
        
        u_crop = u[:, 1:-1, 1:-1]
        v_crop = v[:, 1:-1, 1:-1]
        
        loss = ((u_crop - u_geo)**2 + (v_crop - v_geo)**2).mean()
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    时间一致性损失 - 减少相邻时间步之间的剧烈变化
    """
    
    def __init__(self, weight=0.1, z_weight=0.2, uv_weight=0.1):
        super().__init__()
        self.weight = weight
        self.z_weight = z_weight
        self.uv_weight = uv_weight
        
    def forward(self, x0_pred, x0_prev=None):
        if x0_prev is None:
            return torch.tensor(0.0, device=x0_pred.device)
        
        diff = x0_pred - x0_prev
        
        uv_diff = diff[:, :6]
        uv_loss = (uv_diff[:, :, 1:] - uv_diff[:, :, :-1]).square().mean()
        uv_loss += (uv_diff[:, 1:, :] - uv_diff[:, :-1, :]).square().mean()
        
        z_diff = diff[:, 6:9]
        z_loss = (z_diff[:, :, 1:] - z_diff[:, :, :-1]).square().mean()
        z_loss += (z_diff[:, 1:, :] - z_diff[:, :-1, :]).square().mean()
        
        loss = self.weight * (self.uv_weight * uv_loss + self.z_weight * z_loss)
        return loss


class ChannelWiseRMSEWithUncertainty(nn.Module):
    """
    带不确定性的逐通道RMSE损失
    """
    
    def __init__(self, channel_weights, pressure_level_weights=None):
        super().__init__()
        w = torch.tensor(channel_weights, dtype=torch.float32)
        w = w / w.mean()
        self.register_buffer('channel_weights', w)
        
        if pressure_level_weights is not None:
            pl = torch.tensor(pressure_level_weights, dtype=torch.float32)
            pl = pl / pl.mean()
            self.register_buffer('pressure_weights', pl)
        else:
            self.register_buffer('pressure_weights', torch.ones(3))
            
    def forward(self, pred, target):
        diff = pred - target
        B, C, H, W = diff.shape
        
        channel_rmse = torch.sqrt((diff ** 2).mean(dim=(0, 2, 3)) + 1e-8)
        
        pl_weights = self.pressure_weights.reshape(1, 3).repeat(3, 1).reshape(-1)[:C].to(diff.device)
        channel_weights = self.channel_weights[:C].to(diff.device)
        
        weighted_rmse = channel_rmse * channel_weights * pl_weights
        
        return weighted_rmse.mean()
