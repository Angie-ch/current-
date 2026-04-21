"""
Typhoon Case Study Visualization

Generates publication-ready figures for a specific typhoon case study,
showing how FM and DM capture specific weather phenomena.

Usage:
    python -c "from typhoon_case_study import TyphoonCaseStudy; tc = TyphoonCaseStudy(...); tc.generate(...)"
"""
import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Circle
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    ccrs = None
    cfeature = None

logger = logging.getLogger(__name__)

# Nice colormaps for meteorological fields
WIND_CMAP = LinearSegmentedColormap.from_list('wind', ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'])
Z_CMAP = LinearSegmentedColormap.from_list('z', ['#762a83', '#9970ab', '#c2a5cf', '#f7f7f7', '#a6dba0', '#5aae61', '#1b7837'])


def find_typhoon_center(z_field: np.ndarray) -> Tuple[float, float]:
    """
    Find typhoon center from geopotential height field.

    Typhoon center = minimum of 850 hPa geopotential height (low pressure system)

    Args:
        z_field: (H, W) geopotential height anomaly

    Returns:
        (row_idx, col_idx) of typhoon center
    """
    # Find the minimum (typhoon center)
    min_idx = np.argmin(z_field)
    center_row = min_idx // z_field.shape[1]
    center_col = min_idx % z_field.shape[1]
    return float(center_row), float(center_col)


def plot_typhoon_case_study(
    gt_fields: List[Dict[str, np.ndarray]],
    fm_preds: List[Dict[str, np.ndarray]],
    dm_preds: List[Dict[str, np.ndarray]],
    lead_times: List[int],
    save_path: str,
    typhoon_name: str = "Typhoon",
    lat_range: Tuple[float, float] = (0.0, 60.0),
    lon_range: Tuple[float, float] = (95.0, 185.0),
    channel_names: List[str] = None,
) -> None:
    """
    Generate the main typhoon case study figure.

    Structure:
        Rows: Lead times (+24h, +48h, +72h)
        Columns: ERA5 (GT), FM, DM, Error maps (FM, DM)

    Args:
        gt_fields: List of dicts with 'u', 'v', 'z' arrays for ground truth
        fm_preds: List of dicts with 'u', 'v', 'z' arrays for FM predictions
        dm_preds: List of dicts with 'u', 'v', 'z' arrays for DM predictions
        lead_times: List of forecast lead times in hours
        save_path: Where to save the figure
        typhoon_name: Name of the typhoon
        lat_range, lon_range: Geographic bounds
        channel_names: Variable names
    """
    if channel_names is None:
        channel_names = ['u_850', 'u_500', 'u_250', 'v_850', 'v_500', 'v_250', 'z_850', 'z_500', 'z_250']

    n_lead = len(lead_times)
    fig = plt.figure(figsize=(5 * (n_lead + 1), 4 * 3))
    gs = gridspec.GridSpec(3, n_lead + 1, figure=fig, hspace=0.3, wspace=0.25)

    for col, (lead, gt, fm, dm) in enumerate(zip(lead_times, gt_fields, fm_preds, dm_preds)):
        # Row 0: Z field (850 hPa geopotential height — typhoon eye)
        ax_z = fig.add_subplot(gs[0, col])
        plot_z_field(ax_z, gt.get('z_850', gt.get('z', None)),
                     fm.get('z_850', fm.get('z', None)),
                     dm.get('z_850', dm.get('z', None)), col == 0,
                     f"+{lead}h Z850")

        # Row 1: Wind speed magnitude
        ax_wind = fig.add_subplot(gs[1, col])
        u_gt = gt.get('u_850', gt.get('u', None))
        v_gt = gt.get('v_850', gt.get('v', None))
        u_fm = fm.get('u_850', fm.get('u', None))
        v_fm = fm.get('v_850', fm.get('v', None))
        u_dm = dm.get('u_850', dm.get('v', None))
        v_dm = dm.get('v_850', dm.get('v', None))
        if u_gt is not None and v_gt is not None:
            plot_wind_field(ax_wind, u_gt, v_gt, u_fm, v_fm, u_dm, v_dm, col == 0,
                          f"+{lead}h Wind")

        # Row 2: Error maps
        ax_err = fig.add_subplot(gs[2, col])
        plot_error_map(ax_err, gt, fm, dm, col == 0, f"+{lead}h RMSE")

    # Column headers
    for col, label in enumerate(["ERA5 (GT)"] + [f"FM" if col == 0 else f"DM" for col in range(n_lead)]):
        ax_header = fig.add_subplot(gs[0, col])
        ax_header.text(0.5, 1.15, label, transform=ax_header.transAxes,
                      fontsize=14, fontweight='bold', ha='center', va='bottom')

    # Remove axis ticks for clean look
    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"Typhoon Case Study: {typhoon_name}\n"
                 f"Flow Matching vs Diffusion Model — Northwest Pacific ERA5",
                 fontsize=16, fontweight='bold', y=0.98)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Typhoon case study saved: {save_path}")


def plot_z_field(
    ax,
    gt_z: np.ndarray,
    fm_z: np.ndarray,
    dm_z: np.ndarray,
    show_colorbar: bool = False,
    title: str = "",
):
    """Plot geopotential height field comparison."""
    if gt_z is None:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
        return

    # Plot GT
    vmin, vmax = np.percentile(gt_z, [2, 98])
    im = ax.imshow(gt_z, cmap=Z_CMAP, vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Mark typhoon center
    center_row, center_col = find_typhoon_center(gt_z)
    circle = Circle((center_col, center_row), radius=2,
                   fill=False, color='white', linewidth=2)
    ax.add_patch(circle)
    ax.plot(center_col, center_row, 'w*', markersize=12, markeredgecolor='black')


def plot_wind_field(
    ax,
    u_gt: np.ndarray,
    v_gt: np.ndarray,
    u_fm: np.ndarray,
    v_fm: np.ndarray,
    u_dm: np.ndarray,
    v_dm: np.ndarray,
    show_colorbar: bool = False,
    title: str = "",
):
    """Plot wind speed field comparison."""
    if u_gt is None or v_gt is None:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
        return

    # Compute wind speed
    ws_gt = np.sqrt(u_gt**2 + v_gt**2)
    ws_fm = np.sqrt(u_fm**2 + v_fm**2)
    ws_dm = np.sqrt(u_dm**2 + v_dm**2)

    # Use GT for reference levels
    vmin, vmax = np.percentile(ws_gt, [2, 98])
    im = ax.imshow(ws_gt, cmap='hot', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Wind barbs (subsample)
    step = max(1, ws_gt.shape[0] // 10)
    y, x = np.mgrid[0:ws_gt.shape[0]:step, 0:ws_gt.shape[1]:step]
    ax.quiver(x, y, u_gt[::step, ::step], v_gt[::step, ::step],
              color='white', alpha=0.6, scale=50, width=0.003)


def plot_error_map(
    ax,
    gt: Dict,
    fm: Dict,
    dm: Dict,
    show_colorbar: bool = False,
    title: str = "",
):
    """Plot RMSE error map between FM/DM and GT."""
    # Compute combined error (average over available variables)
    errors = []
    for key in gt.keys():
        if key in fm and key in dm:
            err_fm = np.abs(fm[key] - gt[key])
            err_dm = np.abs(dm[key] - gt[key])
            errors.append((err_fm + err_dm) / 2)

    if not errors:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
        return

    combined_error = np.mean(errors, axis=0)
    vmax = np.percentile(combined_error, [98])[0]
    im = ax.imshow(combined_error, cmap='Reds', vmin=0, vmax=vmax, origin='lower')
    ax.set_title(title, fontsize=11, fontweight='bold')


def plot_typhoon_intensity_evolution(
    gt_sequences: Dict[str, np.ndarray],
    fm_sequences: Dict[str, np.ndarray],
    dm_sequences: Dict[str, np.ndarray],
    lead_times: List[int],
    save_path: str,
    typhoon_name: str = "Typhoon",
):
    """
    Plot typhoon intensity (minimum MSLP, maximum wind) evolution over time.

    Shows how well each model captures the intensification/weakening phases.

    Args:
        gt_sequences: Dict with 'mslp' or 'z_850', 'u_850', 'v_850' sequences
        fm_sequences: FM predicted sequences
        dm_sequences: DM predicted sequences
        lead_times: List of lead times
        save_path: Where to save
        typhoon_name: Name for title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Minimum Z850 (proxy for central pressure)
    ax = axes[0]
    if 'z_850' in gt_sequences:
        gt_min_z = [np.min(z) for z in gt_sequences['z_850']]
        fm_min_z = [np.min(z) for z in fm_sequences['z_850']]
        dm_min_z = [np.min(z) for z in dm_sequences['z_850']]

        ax.plot(lead_times, gt_min_z, 'k-o', linewidth=2, markersize=6, label='ERA5 (GT)')
        ax.plot(lead_times, fm_min_z, 'b--s', linewidth=1.5, markersize=5, label='FM')
        ax.plot(lead_times, dm_min_z, 'r--^', linewidth=1.5, markersize=5, label='DM')
        ax.set_xlabel('Lead Time (h)', fontsize=12)
        ax.set_ylabel('Min Z850 (gpm)', fontsize=12)
        ax.set_title('Typhoon Central Pressure Proxy\n(Lower = Stronger)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    # 2. Maximum Wind Speed
    ax = axes[1]
    if 'u_850' in gt_sequences and 'v_850' in gt_sequences:
        gt_wind = [np.max(np.sqrt(u**2 + v**2)) for u, v in
                   zip(gt_sequences['u_850'], gt_sequences['v_850'])]
        fm_wind = [np.max(np.sqrt(u**2 + v**2)) for u, v in
                   zip(fm_sequences['u_850'], fm_sequences['v_850'])]
        dm_wind = [np.max(np.sqrt(u**2 + v**2)) for u, v in
                   zip(dm_sequences['u_850'], dm_sequences['v_850'])]

        ax.plot(lead_times, gt_wind, 'k-o', linewidth=2, markersize=6, label='ERA5 (GT)')
        ax.plot(lead_times, fm_wind, 'b--s', linewidth=1.5, markersize=5, label='FM')
        ax.plot(lead_times, dm_wind, 'r--^', linewidth=1.5, markersize=5, label='DM')
        ax.set_xlabel('Lead Time (h)', fontsize=12)
        ax.set_ylabel('Max Wind Speed (m/s)', fontsize=12)
        ax.set_title('Maximum Sustained Wind Speed', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    # 3. RMSE over lead time
    ax = axes[2]
    if 'z_850' in gt_sequences:
        fm_rmse = [np.sqrt(np.mean((fm - gt)**2)) for fm, gt in
                   zip(fm_sequences['z_850'], gt_sequences['z_850'])]
        dm_rmse = [np.sqrt(np.mean((dm - gt)**2)) for dm, gt in
                   zip(dm_sequences['z_850'], gt_sequences['z_850'])]

        ax.plot(lead_times, fm_rmse, 'b-o', linewidth=2, markersize=6, label='FM RMSE')
        ax.plot(lead_times, dm_rmse, 'r--s', linewidth=2, markersize=5, label='DM RMSE')
        ax.set_xlabel('Lead Time (h)', fontsize=12)
        ax.set_ylabel('Z850 RMSE', fontsize=12)
        ax.set_title('Forecast Error Growth', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Typhoon Intensity Evolution: {typhoon_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Typhoon intensity evolution saved: {save_path}")


class TyphoonCaseStudy:
    """
    Typhoon case study generator.

    Usage:
        tc = TyphoonCaseStudy(model_fm, model_dm, dataset, device)
        tc.generate(
            typhoon_id="meihua_2022",
            output_dir="./case_studies",
            lead_times=[24, 48, 72],
        )
    """

    def __init__(
        self,
        model_fm,
        model_dm,
        dataset,
        device: str = "cuda",
    ):
        self.model_fm = model_fm
        self.model_dm = model_dm
        self.dataset = dataset
        self.device = device

    def extract_typhoon_sequence(
        self,
        typhoon_id: str,
        start_idx: int,
        num_steps: int = 5,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Extract a sequence of fields for a specific typhoon.

        Returns:
            Dict mapping variable -> list of (H, W) arrays
        """
        sequences = {}

        for step in range(num_steps):
            sample = self.dataset[start_idx + step]
            target = sample["target"]  # (C, H, W)

            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()

            # Parse into variables
            # Expected: [u_850, u_500, u_250, v_850, v_500, v_250, z_850, z_500, z_250]
            for i, var_name in enumerate(['u_850', 'u_500', 'u_250',
                                          'v_850', 'v_500', 'v_250',
                                          'z_850', 'z_500', 'z_250']):
                if var_name not in sequences:
                    sequences[var_name] = []
                sequences[var_name].append(target[i])

        return sequences

    def predict_sequence(
        self,
        model,
        condition: torch.Tensor,
        num_steps: int = 5,
        method: str = "fm",
    ) -> Dict[str, List[np.ndarray]]:
        """
        Generate predictions for a sequence using the model.
        """
        model.eval()
        sequences = {}

        current_window = condition  # (1, T*C, H, W)

        with torch.no_grad():
            for step in range(num_steps):
                if method == "fm":
                    pred = model.sample_fm(
                        current_window, torch.device(self.device),
                        euler_steps=4, euler_mode="midpoint"
                    )
                else:
                    pred = model.sample_dm(
                        current_window, torch.device(self.device),
                        ddim_steps=20
                    )

                if isinstance(pred, torch.Tensor):
                    pred = pred[0].cpu().numpy()  # (C, H, W)

                for i, var_name in enumerate(['u_850', 'u_500', 'u_250',
                                              'v_850', 'v_500', 'v_250',
                                              'z_850', 'z_500', 'z_250']):
                    if var_name not in sequences:
                        sequences[var_name] = []
                    sequences[var_name].append(pred[i])

        return sequences

    def generate(
        self,
        typhoon_id: str,
        output_dir: str,
        lead_times: List[int] = None,
        num_steps: int = 5,
    ):
        """
        Generate complete typhoon case study figures.
        """
        if lead_times is None:
            lead_times = [24, 48, 72]

        # Find the typhoon in dataset
        start_idx = None
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            if sample.get("storm_id") == typhoon_id or sample.get("typhoon_id") == typhoon_id:
                start_idx = idx
                break

        if start_idx is None:
            logger.warning(f"Typhoon {typhoon_id} not found in dataset")
            return

        # Extract GT sequence
        gt_sequences = self.extract_typhoon_sequence(typhoon_id, start_idx, num_steps)

        # Generate FM predictions
        sample = self.dataset[start_idx]
        condition = sample["condition"].unsqueeze(0).to(self.device)
        fm_sequences = self.predict_sequence(
            self.model_fm, condition, num_steps, method="fm"
        )

        # Generate DM predictions
        dm_sequences = self.predict_sequence(
            self.model_dm, condition, num_steps, method="dm"
        )

        # Generate main figure
        gt_fields = [{k: v[i] for k, v in gt_sequences.items()} for i in range(num_steps)]
        fm_fields = [{k: v[i] for k, v in fm_sequences.items()} for i in range(num_steps)]
        dm_fields = [{k: v[i] for k, v in dm_sequences.items()} for i in range(num_steps)]

        plot_typhoon_case_study(
            gt_fields, fm_fields, dm_fields,
            lead_times=[24 * (i + 1) for i in range(num_steps)],
            save_path=os.path.join(output_dir, f"{typhoon_id}_case_study.png"),
            typhoon_name=typhoon_id,
        )

        # Generate intensity evolution
        plot_typhoon_intensity_evolution(
            gt_sequences, fm_sequences, dm_sequences,
            lead_times=[3 * (i + 1) for i in range(num_steps)],
            save_path=os.path.join(output_dir, f"{typhoon_id}_intensity.png"),
            typhoon_name=typhoon_id,
        )

        logger.info(f"Case study for {typhoon_id} generated in {output_dir}")
