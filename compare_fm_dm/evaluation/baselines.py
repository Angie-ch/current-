"""
Baseline Forecasts — Persistence and Climatology

Provides baseline forecasting methods for comparison:
1. Persistence: tomorrows weather = todays weather
2. Climatology: use the temporal mean as the forecast
3. Linear Trend: extrapolate from recent history
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


def persistence_forecast(
    history: torch.Tensor,
    num_steps: int = 1,
) -> torch.Tensor:
    """
    Persistence forecast: the simplest baseline.

    tomorrow's weather = today's weather

    Args:
        history: (B, C, H, W) or (B, T, C, H, W) — most recent observation(s)
        num_steps: number of forecast steps

    Returns:
        Persistence forecast: (B, C, H, W) repeated num_steps times
                           or list of tensors if num_steps > 1
    """
    if history.ndim == 5:
        # (B, T, C, H, W) — use the last time step
        forecast = history[:, -1]  # (B, C, H, W)
    elif history.ndim == 4:
        forecast = history  # (B, C, H, W)
    else:
        raise ValueError(f"Expected 4D or 5D input, got {history.ndim}D")

    if num_steps == 1:
        return forecast
    else:
        return [forecast.clone() for _ in range(num_steps)]


def climatology_forecast(
    climatology_mean: np.ndarray,
    num_steps: int = 1,
    batch_size: int = 1,
    device: str = "cpu",
) -> List[torch.Tensor]:
    """
    Climatology forecast: use the climatological mean as the forecast.

    Args:
        climatology_mean: (C, H, W) or (num_channels, H, W) climatological mean
        num_steps: number of forecast steps
        batch_size: number of samples in a batch
        device: torch device

    Returns:
        List of (B, C, H, W) tensors, one per step
    """
    if isinstance(climatology_mean, np.ndarray):
        clim_tensor = torch.from_numpy(climatology_mean).float()
    else:
        clim_tensor = climatology_mean

    clim_tensor = clim_tensor.to(device).unsqueeze(0)  # (1, C, H, W)

    return [clim_tensor.expand(batch_size, -1, -1, -1).clone() for _ in range(num_steps)]


def linear_trend_forecast(
    history: torch.Tensor,
    num_steps: int = 1,
    fit_window: int = 5,
) -> List[torch.Tensor]:
    """
    Linear trend forecast: fit a linear trend from recent history
    and extrapolate.

    Args:
        history: (B, T, C, H, W) — historical observations
        num_steps: number of forecast steps
        fit_window: number of recent steps to use for fitting trend

    Returns:
        List of (B, C, H, W) forecast steps
    """
    if history.ndim != 5:
        raise ValueError(f"Expected 5D input (B, T, C, H, W), got {history.ndim}D")

    B, T, C, H, W = history.shape
    history_np = history.cpu().numpy()

    forecasts = []
    for step in range(num_steps):
        forecast = torch.zeros((B, C, H, W))

        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        y = history_np[b, -fit_window:, c, h, w]
                        x = np.arange(len(y))
                        if len(y) > 1:
                            slope, intercept = np.polyfit(x, y, 1)
                            # Extrapolate to step+1 (relative to last observed point)
                            next_val = intercept + slope * (len(x) + step)
                        else:
                            next_val = y[-1]

                        forecast[b, c, h, w] = next_val

        forecasts.append(forecast)

    return forecasts


class BaselineForecaster:
    """
    Unified baseline forecaster supporting multiple strategies.

    Usage:
        forecaster = BaselineForecaster(climatology_mean=clim_mean)
        persistence_pred = forecaster.persist(condition)  # (B, C, H, W)
        climatology_pred = forecaster.climatology(batch_size=B)  # list of (B, C, H, W)
    """

    def __init__(
        self,
        climatology_mean: Optional[np.ndarray] = None,
        device: str = "cpu",
    ):
        self.climatology_mean = climatology_mean
        self.device = device

    def persist(
        self,
        condition: torch.Tensor,
        num_steps: int = 1,
    ) -> List[torch.Tensor]:
        """Persistence forecast."""
        return persistence_forecast(condition, num_steps=num_steps)

    def climatology(
        self,
        num_steps: int = 1,
        batch_size: int = 1,
    ) -> List[torch.Tensor]:
        """Climatology forecast."""
        if self.climatology_mean is None:
            raise ValueError("Climatology mean not set. Provide it to __init__.")
        return climatology_forecast(
            self.climatology_mean, num_steps=num_steps,
            batch_size=batch_size, device=self.device
        )

    def linear_trend(
        self,
        history: torch.Tensor,
        num_steps: int = 1,
    ) -> List[torch.Tensor]:
        """Linear trend forecast."""
        return linear_trend_forecast(history, num_steps=num_steps)

    def evaluate_baselines(
        self,
        ground_truth: List[torch.Tensor],
        conditions: List[torch.Tensor],
        climatology_mean: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all baseline methods against ground truth.

        Args:
            ground_truth: List of (B, C, H, W) tensors
            conditions: List of (B, C, H, W) or (B, T, C, H, W) conditions
            climatology_mean: Optional (C, H, W) climatological mean

        Returns:
            Dict mapping method name -> metric dict
        """
        results = {}

        # Persistence
        pers_errors = []
        for cond, gt in zip(conditions, ground_truth):
            pers_pred = cond if cond.ndim == 4 else cond[:, -1]
            pers_pred = pers_pred[:gt.shape[0]] if pers_pred.shape[0] > gt.shape[0] else pers_pred
            # Pad if needed
            if pers_pred.shape != gt.shape:
                min_b = min(pers_pred.shape[0], gt.shape[0])
                pers_pred = pers_pred[:min_b]
                gt_slice = gt[:min_b]

            mse = ((pers_pred - gt_slice) ** 2).mean().item()
            pers_errors.append(mse)

        results["persistence"] = {
            "mse_mean": np.mean(pers_errors),
            "mse_std": np.std(pers_errors),
            "rmse_mean": np.sqrt(np.mean(pers_errors)),
        }

        # Climatology
        if climatology_mean is not None:
            clim_errors = []
            clim_tensor = torch.from_numpy(climatology_mean).float()
            for gt in ground_truth:
                B = gt.shape[0]
                clim_pred = clim_tensor.unsqueeze(0).expand(B, -1, -1, -1)
                # Handle channel count mismatch
                if clim_pred.shape[1] != gt.shape[1]:
                    clim_pred = clim_pred[:, :gt.shape[1]]
                mse = ((clim_pred - gt) ** 2).mean().item()
                clim_errors.append(mse)

            results["climatology"] = {
                "mse_mean": np.mean(clim_errors),
                "mse_std": np.std(clim_errors),
                "rmse_mean": np.sqrt(np.mean(clim_errors)),
            }

        return results
