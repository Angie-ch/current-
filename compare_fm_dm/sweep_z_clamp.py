"""
Sweep script: test z clamp range and Euler steps on z variance quality.
Runs all experiments in sequence and prints a summary table.
"""
import sys, torch, numpy as np
sys.path.insert(0, '.')
from configs import get_config
from data.dataset import build_dataloaders
from models.unified_model import UnifiedModel
from models.trainer import EMA

device = 'cuda'
data_cfg, model_cfg, train_cfg, _ = get_config()
_, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)

model = UnifiedModel(model_cfg, data_cfg, train_cfg, method='fm').to(device)
ckpt = torch.load('multi_seed_results_20ep/seed_42/checkpoints_fm/best.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
ema = EMA(model, decay=0.999)
ema.load_state_dict(ckpt.get('ema_state_dict', {}))
ema.apply_shadow(model)
model.eval()

nm = norm_mean.numpy() if isinstance(norm_mean, torch.Tensor) else norm_mean
std_arr = norm_std.numpy() if isinstance(norm_std, torch.Tensor) else norm_std
std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)

channel_names = ['u_850', 'u_500', 'u_250', 'v_850', 'v_500', 'v_250', 'z_850', 'z_500', 'z_250']

def collect_samples(n_batches=5, **sample_kwargs):
    """Collect predictions and GTs using model.sample_fm with given kwargs."""
    fm_preds, gts = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= n_batches:
                break
            c = batch['condition'].to(device)
            t = batch['target']
            p = model.sample_fm(c, device, **sample_kwargs)
            fm_preds.append(p.cpu().numpy())
            gts.append(t.cpu().numpy())
    return np.concatenate(fm_preds, axis=0), np.concatenate(gts, axis=0)

def compute_metrics(fm_preds, gts):
    """Compute per-channel metrics."""
    metrics = {}
    for ch in range(9):
        gt_dn = gts[:, ch] * std_arr[ch] + nm[ch]
        p_dn  = fm_preds[:, ch] * std_arr[ch] + nm[ch]

        spatial_std_gt  = gt_dn.std(axis=(1,2)).mean()
        spatial_std_p   = p_dn.std(axis=(1,2)).mean()
        ratio           = spatial_std_p / spatial_std_gt if spatial_std_gt > 0 else 0

        errs = (fm_preds[:, ch] - gts[:, ch]) * std_arr[ch]
        rmse = np.sqrt((errs**2).mean())
        rmse_pct = rmse / abs(nm[ch]) * 100 if ch in [6,7,8] else None

        corr = np.corrcoef(p_dn.flatten(), gt_dn.flatten())[0, 1]

        metrics[channel_names[ch]] = {
            'pred_std': spatial_std_p,
            'gt_std': spatial_std_gt,
            'ratio': ratio,
            'rmse': rmse,
            'rmse_pct': rmse_pct,
            'corr': corr,
        }
    return metrics

# === Experiments ===
experiments = [
    {
        'name': 'BASELINE (z_clamp=-2, steps=4, midpoint)',
        'kwargs': {'euler_steps': 4, 'euler_mode': 'midpoint',
                   'clamp_range': (-3.0, 3.0), 'z_clamp_range': (-2.0, 2.0)},
    },
    {
        'name': 'Exp 1 (z_clamp=None, steps=4, midpoint)',
        'kwargs': {'euler_steps': 4, 'euler_mode': 'midpoint',
                   'clamp_range': (-3.0, 3.0), 'z_clamp_range': None},
    },
    {
        'name': 'Exp 2 (z_clamp=-4, steps=8, midpoint)',
        'kwargs': {'euler_steps': 8, 'euler_mode': 'midpoint',
                   'clamp_range': (-3.0, 3.0), 'z_clamp_range': (-4.0, 4.0)},
    },
    {
        'name': 'Exp 3 (z_clamp=None, steps=8, midpoint)',
        'kwargs': {'euler_steps': 8, 'euler_mode': 'midpoint',
                   'clamp_range': (-3.0, 3.0), 'z_clamp_range': None},
    },
    {
        'name': 'Exp 4 (z_clamp=None, steps=8, heun)',
        'kwargs': {'euler_steps': 8, 'euler_mode': 'heun',
                   'clamp_range': (-3.0, 3.0), 'z_clamp_range': None},
    },
]

results = []
for exp in experiments:
    name = exp['name']
    kwargs = exp['kwargs']
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"  kwargs: {kwargs}")
    fm_preds, gts = collect_samples(n_batches=5, **kwargs)
    metrics = compute_metrics(fm_preds, gts)
    results.append({'name': name, 'kwargs': kwargs, 'metrics': metrics})

    print(f"\n  Z Channel Results:")
    for ch_idx, ch_name in enumerate(['z_850', 'z_500', 'z_250']):
        m = metrics[ch_name]
        rmse_str = f"RMSE={m['rmse']:.0f} ({m['rmse_pct']:.1f}%)" if m['rmse_pct'] else f"RMSE={m['rmse']:.1f}"
        print(f"    {ch_name}: r={m['corr']:.4f} | ratio={m['ratio']:.3f} | {rmse_str}")

    print(f"  U/V Correlations (avg):")
    u_corrs = [metrics[cn]['corr'] for cn in ['u_850','u_500','u_250']]
    v_corrs = [metrics[cn]['corr'] for cn in ['v_850','v_500','v_250']]
    print(f"    u: avg r={np.mean(u_corrs):.4f}  |  v: avg r={np.mean(v_corrs):.4f}")

# === Summary Table ===
print(f"\n\n{'='*70}")
print("SUMMARY TABLE: Z Channel Variance Ratio (pred_std / gt_std)")
print(f"{'='*70}")
print(f"{'Experiment':<45} {'z_850':>8} {'z_500':>8} {'z_250':>8} {'avg':>8}")
print(f"{'-'*70}")
for r in results:
    z_vals = [r['metrics'][cn]['ratio'] for cn in ['z_850','z_500','z_250']]
    vals_str = ' '.join(f'{v:.3f}' for v in z_vals)
    print(f"{r['name']:<45} {vals_str}   {np.mean(z_vals):.3f}")

print(f"\n{'='*70}")
print("SUMMARY TABLE: Z Channel Correlation")
print(f"{'='*70}")
print(f"{'Experiment':<45} {'z_850':>8} {'z_500':>8} {'z_250':>8} {'avg':>8}")
print(f"{'-'*70}")
for r in results:
    z_vals = [r['metrics'][cn]['corr'] for cn in ['z_850','z_500','z_250']]
    vals_str = ' '.join(f'{v:.4f}' for v in z_vals)
    print(f"{r['name']:<45} {vals_str}  {np.mean(z_vals):.4f}")

print(f"\n{'='*70}")
print("SUMMARY TABLE: Z Channel RMSE (% of mean)")
print(f"{'='*70}")
print(f"{'Experiment':<45} {'z_850':>8} {'z_500':>8} {'z_250':>8} {'avg':>8}")
print(f"{'-'*70}")
for r in results:
    z_vals = [r['metrics'][cn]['rmse_pct'] for cn in ['z_850','z_500','z_250']]
    vals_str = ' '.join(f'{v:.1f}%' for v in z_vals)
    print(f"{r['name']:<45} {vals_str}  {np.mean(z_vals):.1f}%")

print(f"\n{'='*70}")
print("SUMMARY TABLE: U/V Correlation (avg)")
print(f"{'='*70}")
print(f"{'Experiment':<45} {'u_avg':>8} {'v_avg':>8}")
print(f"{'-'*70}")
for r in results:
    u_avg = np.mean([r['metrics'][cn]['corr'] for cn in ['u_850','u_500','u_250']])
    v_avg = np.mean([r['metrics'][cn]['corr'] for cn in ['v_850','v_500','v_250']])
    print(f"{r['name']:<45} {u_avg:.4f}   {v_avg:.4f}")
