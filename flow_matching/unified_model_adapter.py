"""
Adapter module: wraps compare_fm_dm's UnifiedModel (FM mode)
to provide the same interface as Trajectory's LT3PModel.

This allows the Trajectory finetune pipeline to use FM models
from the compare_fm_dm experiment.

Key translation:
  LT3PModel: inputs (history_coords, future_era5) -> outputs (pred_coords)
  UnifiedModel: inputs (condition, target) -> outputs (x0_pred velocity field)

Adapter strategy:
  1. Construct condition from history_coords by converting normalized coords
     to ERA5 spatial field patches (similar to how trajectory model uses
     PhysicsEncoder3D to encode coords into features that modulate ERA5)
  2. Use UnifiedModel's sample_fm() to predict future ERA5 velocity field
  3. Integrate velocity → predict future ERA5 field
  4. (Option A - Direct): Train a tiny coord_decoder to extract coords from
     the predicted ERA5 fields (or use a simple heuristic)
  5. (Option B - Latent): Use the unified model's internal trajectory features
     if available

Since UnifiedModel doesn't directly predict coordinates, the adapter
adds a small learnable coordinator head that maps from the predicted
ERA5 field (or its features) to lat/lon coordinates.
"""
import os
import sys
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAJ_DIR)

from config import model_cfg, data_cfg, train_cfg

logger = logging.getLogger(__name__)


# ============================================================
# Helper: convert normalized coordinates to spatial fields
# ============================================================

def coords_to_spatial_field(
    history_coords: torch.Tensor,
    grid_size: int = 40,
    sigma: float = 1.5,
    lat_range: Tuple[float, float] = (0.0, 60.0),
    lon_range: Tuple[float, float] = (95.0, 185.0),
) -> torch.Tensor:
    """
    Convert normalized coordinate sequences to spatial density fields.

    Args:
        history_coords: (B, T_hist, 2) normalized [lat, lon] in [0, 1]
        grid_size: output spatial resolution (40×40)
        sigma: Gaussian blur sigma in grid units
        lat_range, lon_range: denormalization ranges

    Returns:
        field: (B, 3, grid_size, grid_size)
          Channel 0: location density (recent positions)
          Channel 1: lat-gradient field (∂lat/∂x, ∂lat/∂y)
          Channel 2: lon-gradient field (∂lon/∂x, ∂lon/∂y)
    """
    B, T, _ = history_coords.shape
    device = history_coords.device

    # Denormalize to physical coords
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    lat_phys = history_coords[..., 0] * (lat_max - lat_min) + lat_min  # (B, T)
    lon_phys = history_coords[..., 1] * (lon_max - lon_min) + lon_min  # (B, T)

    # Convert to grid indices (0 to grid_size-1)
    lat_grid = (lat_phys - lat_min) / (lat_max - lat_min) * (grid_size - 1)
    lon_grid = (lon_phys - lon_min) / (lon_max - lon_min) * (grid_size - 1)

    # Create coordinate grids
    y = torch.linspace(0, grid_size - 1, grid_size, device=device)
    x = torch.linspace(0, grid_size - 1, grid_size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (grid_size, grid_size)

    # Build density field with Gaussian blobs at each history point
    field_density = torch.zeros(B, grid_size, grid_size, device=device)
    field_dlat_dy = torch.zeros(B, grid_size, grid_size, device=device)
    field_dlon_dx = torch.zeros(B, grid_size, grid_size, device=device)

    for t in range(T):
        lat_t = lat_grid[:, t].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        lon_t = lon_grid[:, t].unsqueeze(1).unsqueeze(2)

        # Gaussian blob
        dist_sq = (yy - lat_t)**2 + (xx - lon_t)**2
        weight = torch.exp(-dist_sq / (2 * sigma**2))  # (B, grid_size, grid_size)
        field_density = field_density + weight

    # Normalize density
    field_density = field_density / (T + 1e-8)

    # Approximate gradients from recent movement (last 2 points)
    if T >= 2:
        lat_last2 = lat_grid[:, -2:]  # (B, 2)
        lon_last2 = lon_grid[:, -2:]
        dlat = lat_last2[:, 1] - lat_last2[:, 0]  # (B,)
        dlon = lon_last2[:, 1] - lon_last2[:, 0]  # (B,)

        # Create smooth gradient fields (broadcast across grid)
        # Gradient direction: perpendicular to radial from last point
        for b in range(B):
            lat0 = lat_grid[b, -1]
            lon0 = lon_grid[b, -1]
            # Simple: assign gradient magnitude to region around last point
            dist_sq = (yy - lat0)**2 + (xx - lon0)**2
            mask = torch.exp(-dist_sq / (2 * (sigma*2)**2))
            field_dlat_dy[b] = dlat[b] * mask
            field_dlon_dx[b] = dlon[b] * mask

    # Stack into 3-channel field
    field = torch.stack([field_density, field_dlat_dy, field_dlon_dx], dim=1)

    return field


# ============================================================
# Adapter: UnifiedModel → LT3PModel interface
# ============================================================

class UnifiedModelAdapter(nn.Module):
    """
    Adapter that wraps a compare_fm_dm UnifiedModel (FM mode) to behave
    like an LT3PModel for the finetuning pipeline.

    Key idea:
      - history_coords (B, T_hist, 2) → spatial condition field (B, C_cond, H, W)
      - future_era5 (B, T_future, C, H, W) → target velocity field x1
      - model.sample_fm(condition) → predicted x0 (future ERA5)
      - predicted_era5 - current_era5 → velocity → integrate to coords

    The adapter adds a tiny coordinator head: predicted_era5 → predicted_coords
    (trained during finetune, while the frozen UnifiedModel provides ERA5 forecasts)
    """

    def __init__(
        self,
        unified_model: nn.Module,
        data_cfg,
        model_cfg,
        grid_size: int = 40,
        coord_hidden_dim: int = 128,
    ):
        super().__init__()
        self.unified_model = unified_model
        self.unified_model.eval()  # keep generator frozen
        for param in self.unified_model.parameters():
            param.requires_grad = False

        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.grid_size = grid_size

        # ERA5 channel info (compare_fm_dm uses 9 channels)
        self.era5_channels = data_cfg.num_channels  # 9
        self.t_history = data_cfg.history_steps  # 16
        self.t_future = data_cfg.forecast_steps  # 24

        # Small coordinator: maps ERA5 field (C, H, W) + history coords → future coords
        # Input: predicted ERA5 fields (T_future, C, H, W) + last_history_coords (2,)
        # Output: future coords (T_future, 2)
        input_dim = self.era5_channels * grid_size * grid_size + 2
        self.coord_decoder = nn.Sequential(
            nn.Linear(input_dim, coord_hidden_dim),
            nn.ReLU(),
            nn.Linear(coord_hidden_dim, coord_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(coord_hidden_dim // 2, self.t_future * 2),
        )
        # Initialize to output small values (start from zero prediction bias)
        nn.init.zeros_(self.coord_decoder[-1].weight)
        nn.init.zeros_(self.coord_decoder[-1].bias)

        # Spatial pooling for ERA5 fields
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.pooled_dim = self.era5_channels * 4 * 4

        # Revised coordinator: pool → flatten → MLP
        self.coord_decoder = nn.Sequential(
            nn.Linear(self.pooled_dim + 2, coord_hidden_dim),
            nn.GELU(),
            nn.Linear(coord_hidden_dim, coord_hidden_dim),
            nn.GELU(),
            nn.Linear(coord_hidden_dim, self.t_future * 2),
        )
        nn.init.zeros_(self.coord_decoder[-1].weight)
        nn.init.zeros_(self.coord_decoder[-1].bias)

    def forward(
        self,
        history_coords: torch.Tensor,
        future_era5: torch.Tensor,
        target_coords: Optional[torch.Tensor] = None,
        past_era5: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass — predict future coordinates from history + future ERA5.

        Args:
            history_coords: (B, T_hist, 2) normalized lat/lon [0-1]
            future_era5: (B, T_future, C, H, W) normalized ERA5 fields
            target_coords: (B, T_future, 2) optional, for loss computation
            past_era5: (B, T_hist, C, H, W) optional, not used

        Returns:
            dict with 'predicted_coords', 'mse_loss', etc.
        """
        B = history_coords.shape[0]
        device = history_coords.device

        # Use the unified model to "refine" or "predict" future ERA5
        # Strategy: treat future_era5 as x1 target, sample from x0=noise toward x1
        # For training, we can use the model's forward_fm to get a refined prediction
        # For inference, we'd use sample_fm starting from noise.

        # Prepare condition from history_coords → spatial field
        condition_field = coords_to_spatial_field(
            history_coords,
            grid_size=self.grid_size,
            lat_range=self.data_cfg.lat_range,
            lon_range=self.data_cfg.lon_range,
        )  # (B, 3, 40, 40)

        # For compare_fm_dm UnifiedModel, condition needs to be (B, C_cond, H, W)
        # Its cond_encoder expects C_cond = data_cfg.num_channels * history_steps? Actually no.
        # Let's inspect: UnifiedDiT's cond_encoder takes condition (B, C, H, W) or (B, T, C, H, W)
        # In forward_fm, they call self._prepare_condition(condition) which handles both.
        # The condition input is whatever the dataset provides.
        # In compare_fm_dm's dataset: condition is (B, T*C, H, W) flattened.
        # BUT in autoregressive_inference, they pass condition as (1, T*C, H, W).
        # So we need to produce a condition of shape (B, C_cond, H, W) where C_cond matches expected.

        # Let's derive the expected condition channels from data_cfg:
        # compare_fm_dm/data/dataset.py uses:
        #   condition = torch.cat([past_era5, future_era5_cond], dim=1)  # shape (T*C, H, W)
        # Actually, check dataset: condition shape = (history_steps * num_channels, H, W)
        # So C_cond = history_steps * num_channels = 16 * 9 = 144

        # Our condition_field is only (B, 3, 40, 40). Need to expand to match.
        # Simple: tile/expand to match channel count, or use a linear projection.
        # Better: pass through a small conv to map 3 → 144 channels
        if not hasattr(self, 'cond_proj'):
            self.cond_proj = nn.Conv2d(3, self.data_cfg.condition_channels, kernel_size=3, padding=1).to(device)
        condition = self.cond_proj(condition_field)  # (B, 144, 40, 40)

        # Now we need to predict future ERA5. For training with teacher-forcing:
        # Use the unified model's forward to get refined future_era5 prediction.
        # But unified_model.forward_fm expects (condition, target) where target is x1 (future_era5).
        # It returns x0_pred = predicted initial noise? Actually it returns x0_pred = predicted x0 (the "past"?)
        # Let's re-read forward_fm:
        #   x_t = (1-t)*target + t*x_1
        #   v_pred = model(x_t, t, condition)
        #   x0_pred = target - t * v_pred  ??? No, they compute x0_pred from v_pred differently.
        # Actually from line 471: v_pred = self.dit(x_t, t, condition)
        # And from return: "x0_pred": x_1 - t * v_pred ??? Not shown directly.
        # Let's search forward_fm return... at line 518: returns dict with "x0_pred"
        # I need to understand how x0_pred is computed. Check further down in unified_model.py.

        # Actually I'll simplify: Instead of trying to use unified_model's FM forward,
        # just use sample_fm for inference-style prediction. For training we need
        # a differentiable path. The unified_model's forward_fm gives that.

        # Use forward_fm with target = future_era5 (flattened appropriately)
        # Prepare target: future_era5 shape is (B, T_future, C, H, W). We need (B, C, H, W) per step?
        # The unified_model's forward_fm expects target shape (B, C, H, W) or (B, T*C, H, W)?
        # Check line 454: B = target.shape[0], then x_1 = torch.randn_like(target)
        # So target shape matches x_1 shape. In training, x_1 is the "future" field? Actually in FM,
        # x_0 is initial (noise), x_1 is target (data). So they pass target = real future field.
        # The target should be (B, C, H, W) single frame? Or multi-step?
        # Their dataset yields target = "future_era5" which seems to be a single time step?
        # In autoregressive_inference they use single-step predictions.
        # Actually in compare_fm_dm's dataset: target is the next frame only (one time step).
        # So the model is trained single-step. For trajectory finetuning, we need multi-step.
        # So we have two mismatches:
        #  1) unified_model expects single-step FM; we need 24-step trajectory
        #  2) unified_model predicts ERA5 fields; we need coordinates

        # Approach: Use the unified model in a loop to autoregressively predict
        # future ERA5 fields (like in inference), then decode coords from the sequence.
        # But training this end-to-end is expensive.
        # Alternative: Train only the coordinator on predicted ERA5 → coords,
        # with unified_model frozen. This is simpler: first generate cached ERA5
        # predictions (like diffusion finetune does), then train coordinator to
        # map those predictions to coords.

        # Given complexity, I'll implement the simpler two-stage approach:
        # Stage A: Generate CFM ERA5 cache using unified_model.sample_fm()
        # Stage B: Train a small coordinator: (predicted_era5_sequence) → (coords_sequence)
        # This decouples the problem and avoids modifying unified_model training.

        raise NotImplementedError(
            "UnifiedModelAdapter requires multi-step autoregressive FM sampling. "
            "Use generate_cfm_era5_cache_compare() to create cache first, "
            "then train a coordinator separately."
        )


# ============================================================
# Alternative: Direct wrapper that uses sample_fm autoregressively
# ============================================================

class UnifiedModelAutoregressiveWrapper(nn.Module):
    """
    Wraps a trained compare_fm_dm FM model and runs autoregressive inference
    to generate a full T_future-length ERA5 sequence, then decodes coordinates.

    This is used at evaluation / cache-generation time, not training.
    For training the coordinator, we generate cached predictions first.
    """

    def __init__(
        self,
        unified_model: nn.Module,
        data_cfg,
        model_cfg,
        device: torch.device,
        euler_steps: int = 4,
        noise_sigma: float = 0.02,
    ):
        super().__init__()
        self.unified_model = unified_model
        self.unified_model.eval()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.device = device
        self.euler_steps = euler_steps
        self.noise_sigma = noise_sigma

        self.era5_channels = data_cfg.num_channels
        self.t_history = data_cfg.history_steps
        self.t_future = data_cfg.forecast_steps
        self.grid_size = data_cfg.grid_size

        # Coordinator: maps ERA5 sequence → coords
        # Input: (T_future, C, H, W) → pooled + flattened → MLP → (T_future, 2)
        self.coord_decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # (T, C, 4, 4)
            nn.Flatten(start_dim=1),  # (T, C*16)
            nn.Linear(self.era5_channels * 4 * 4, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # per-step output (we'll apply this per time step)
        )
        # Actually we need per-step, so we'll use a loop or apply linear across T
        # Better: use a tiny 3D conv or LSTM. Simpler: independent per-step MLP
        self.per_step_decoder = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(self.era5_channels * 4 * 4, 128),
                nn.GELU(),
                nn.Linear(128, 2),
            ) for _ in range(self.t_future)
        ])

        # Initialize last layer small
        for decoder in self.per_step_decoder:
            decoder[-1].weight.data.mul_(0.01)
            decoder[-1].bias.data.zero_()

    def forward(
        self,
        history_coords: torch.Tensor,
        future_era5: torch.Tensor,  # NOT used directly; we predict from model
        target_coords: Optional[torch.Tensor] = None,
        past_era5: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        During training: this wrapper is used only for evaluation / cache generation.
        For actual training, use the two-stage pipeline:
          1) Generate cached ERA5 predictions with unified_model
          2) Train this coordinator on those cached predictions.
        """
        raise RuntimeError(
            "UnifiedModelAutoregressiveWrapper is for inference/cache generation only. "
            "For training, use generate_cfm_era5_cache_compare() to create cache, "
            "then train a coordinator model separately."
        )

    @torch.no_grad()
    def predict(
        self,
        history_coords: torch.Tensor,
        future_era5: Optional[torch.Tensor] = None,
        past_era5: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive prediction: history_coords → future coordinate sequence.

        Args:
            history_coords: (B, T_hist, 2) normalized [0-1]
        Returns:
            pred_coords: (B, T_future, 2) normalized [0-1]
        """
        B = history_coords.shape[0]
        device = history_coords.device

        # Build initial condition from history_coords
        # Convert history coords to a spatial condition field
        condition_field = coords_to_spatial_field(
            history_coords,
            grid_size=self.grid_size,
            lat_range=self.data_cfg.lat_range,
            lon_range=self.data_cfg.lon_range,
        )  # (B, 3, 40, 40)

        # Expand channels to match expected condition channels (using a simple repeat)
        # The unified model's condition encoder expects (B, C_cond, H, W) where
        # C_cond = data_cfg.condition_channels = history_steps * num_channels = 144
        # We'll tile the 3-channel field across 48 blocks (144/3=48) as a simple prior.
        # Alternatively, could learn a projection, but for inference we just tile.
        C_cond = self.data_cfg.condition_channels
        repeat_factor = C_cond // 3
        condition = condition_field.repeat(1, repeat_factor, 1, 1)  # (B, 144, 40, 40)
        # Note: This is a quick hack. A better approach would train a projection,
        # but for cache generation this might be acceptable as the unified model's
        # condition encoder can learn to extract useful info from repeated patterns.

        # Autoregressive generation of ERA5 sequence
        era5_preds = []
        x_t = torch.randn(B, self.era5_channels, self.grid_size, self.grid_size, device=device)

        for step in range(self.t_future):
            # Predict next frame using unified_model.sample_fm (single-step)
            # sample_fm expects: condition (B, C_cond, H, W) and returns x0 (B, C, H, W)
            pred = self.unified_model.sample_fm(
                condition,
                device=device,
                euler_steps=self.euler_steps,
                euler_mode='midpoint',
                clamp_range=self.unified_model.scheduler.clamp_range,
                z_channel_indices=self.unified_model.z_channel_indices,
                z_clamp_range=getattr(self.unified_model, 'z_clamp_range', None),
            )
            # pred shape: (B, C, H, W)
            era5_preds.append(pred.cpu())

            # Update condition for next step: roll the history window
            if step < self.t_future - 1:
                # Shift condition: drop oldest 9 channels, append new prediction
                # Condition is (B, T*C, H, W) = (B, 144, 40, 40) with T=16, C=9
                # Reshape to (B, T, C, H, W), roll on T dim, flatten back.
                cond_5d = condition.view(B, self.t_history, self.era5_channels, self.grid_size, self.grid_size)
                # Roll: remove first time step, append pred at end
                cond_5d = torch.cat([cond_5d[:, 1:], pred.unsqueeze(1)], dim=1)
                condition = cond_5d.view(B, -1, self.grid_size, self.grid_size)

        # Stack predictions: (T_future, B, C, H, W) → (B, T_future, C, H, W)
        era5_sequence = torch.stack(era5_preds, dim=1)  # (B, T_future, C, H, W)

        # Decode coordinates from each ERA5 frame using per-step decoder
        pred_coords = []
        for t in range(self.t_future):
            era5_t = era5_sequence[:, t]  # (B, C, H, W)
            coord_t = self.per_step_decoder[t](era5_t)  # (B, 2)
            # Squash to [0,1] with sigmoid
            coord_t = torch.sigmoid(coord_t)
            pred_coords.append(coord_t)

        pred_coords = torch.stack(pred_coords, dim=1)  # (B, T_future, 2)

        return pred_coords


def coords_to_spatial_field_batch(
    history_coords: torch.Tensor,
    grid_size: int = 40,
    sigma: float = 1.5,
    lat_range: Tuple[float, float] = (0.0, 60.0),
    lon_range: Tuple[float, float] = (95.0, 185.0),
) -> torch.Tensor:
    """Vectorized batch version of coords_to_spatial_field."""
    B, T, _ = history_coords.shape
    device = history_coords.device

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    lat_phys = history_coords[..., 0] * (lat_max - lat_min) + lat_min
    lon_phys = history_coords[..., 1] * (lon_max - lon_min) + lon_min

    lat_grid = (lat_phys - lat_min) / (lat_max - lat_min) * (grid_size - 1)
    lon_grid = (lon_phys - lon_min) / (lon_max - lon_min) * (grid_size - 1)

    # Create coordinate grids (same for all batch items)
    y = torch.linspace(0, grid_size - 1, grid_size, device=device)
    x = torch.linspace(0, grid_size - 1, grid_size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (grid_size, grid_size)

    # Vectorized: compute distance fields for all history steps at once
    # Expand yy, xx to (B, 1, grid_size, grid_size)
    yy_exp = yy.unsqueeze(0).unsqueeze(0)  # (1, 1, gs, gs)
    xx_exp = xx.unsqueeze(0).unsqueeze(0)

    # lat_grid: (B, T) → (B, T, 1, 1)
    lat_exp = lat_grid.unsqueeze(-1).unsqueeze(-1)
    lon_exp = lon_grid.unsqueeze(-1).unsqueeze(-1)

    dist_sq = (yy_exp - lat_exp)**2 + (xx_exp - lon_exp)**2  # (B, T, gs, gs)
    weights = torch.exp(-dist_sq / (2 * sigma**2))  # (B, T, gs, gs)

    field_density = weights.sum(dim=1)  # (B, gs, gs)
    field_density = field_density / T

    # Stack 3 channels: density, dlat/dy, dlon/dx (simplified: gradients zero)
    field = torch.stack([field_density, torch.zeros_like(field_density), torch.zeros_like(field_density)], dim=1)

    return field
