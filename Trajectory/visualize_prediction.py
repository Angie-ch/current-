"""
可视化台风路径预测结果 - LT3P模型版本
同时显示真实路径和预测路径
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from torch.utils.data import DataLoader

from config import model_cfg, data_cfg
from model import LT3PModel
from dataset import LT3PDataset, split_storms_by_id, filter_short_storms
from data_processing import load_tyc_storms


def load_model_and_data():
    """加载模型和测试数据"""
    print('Loading data...')
    storm_samples = load_tyc_storms(
        csv_path=data_cfg.csv_path,
        era5_base_dir=data_cfg.era5_dir
    )
    
    # 过滤太短的台风
    storm_samples = filter_short_storms(storm_samples, 120, 3)
    
    train_s, val_s, test_s = split_storms_by_id(storm_samples, 0.7, 0.15, seed=42)
    
    print(f'Test storms: {len(test_s)}')
    for s in test_s[:5]:  # 只显示前5个
        print(f'  - {s.storm_id}: {len(s.times)} timesteps')
    if len(test_s) > 5:
        print(f'  ... and {len(test_s) - 5} more')
    
    test_ds = LT3PDataset(test_s, stride=model_cfg.t_future)
    print(f'Test samples: {len(test_ds)}')
    
    if len(test_ds) == 0:
        raise ValueError("No test samples available!")
    
    # 加载模型
    print('\nLoading best model...')
    checkpoint_path = 'checkpoints/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    # 获取ERA5通道数
    sample = test_ds[0]
    era5_channels = sample['future_era5'].shape[1]
    
    model = LT3PModel(
        coord_dim=model_cfg.coord_dim,
        output_dim=model_cfg.output_dim,
        era5_channels=era5_channels,
        t_history=model_cfg.t_history,
        t_future=model_cfg.t_future,
        d_model=model_cfg.transformer_dim,
        n_heads=model_cfg.transformer_heads,
        n_layers=model_cfg.transformer_layers,
        ff_dim=model_cfg.transformer_ff_dim,
        dropout=model_cfg.dropout,
    ).cuda()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Loaded from epoch {checkpoint["epoch"]+1}')
    
    return model, test_ds, test_s


def predict_and_visualize(model, test_ds, test_storms, output_dir='predictions', max_samples=1, random_select=True):
    """预测并可视化结果"""
    import random
    import time
    
    # 使用当前时间作为随机种子，确保每次运行结果不同
    random.seed(int(time.time() * 1000) % 2**32)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    lat_range, lon_range = data_cfg.lat_range, data_cfg.lon_range
    
    # 选择要可视化的样本索引
    total_samples = len(test_ds)
    if random_select and total_samples > max_samples:
        indices = random.sample(range(total_samples), max_samples)
    else:
        indices = list(range(min(max_samples, total_samples)))
    
    print(f"Visualizing samples: {indices}")
    
    all_results = []
    
    with torch.no_grad():
        for idx in indices:
            batch = test_ds[idx]
            
            # 添加batch维度
            history_coords = batch['history_coords'].unsqueeze(0).cuda()
            future_era5 = batch['future_era5'].unsqueeze(0).cuda()
            target_lat = batch['target_lat_raw'].numpy()
            target_lon = batch['target_lon_raw'].numpy()
            history_lat = batch['history_lat_raw'].numpy()
            history_lon = batch['history_lon_raw'].numpy()
            storm_id = batch['storm_id']
            
            # 预测
            outputs = model.predict(history_coords, future_era5)
            pred_coords = outputs['predicted_coords'].cpu().numpy()[0]
            
            # 反归一化预测坐标
            pred_lat = pred_coords[:, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
            pred_lon = pred_coords[:, 1] * (lon_range[1] - lon_range[0]) + lon_range[0]
            
            # 计算误差
            lat_err = (pred_lat - target_lat) * 111
            lon_err = (pred_lon - target_lon) * 111 * np.cos(np.radians(target_lat))
            dist_err = np.sqrt(lat_err**2 + lon_err**2)
            
            all_results.append({
                'storm_id': storm_id,
                'sample_idx': idx,
                'history_lat': history_lat,
                'history_lon': history_lon,
                'target_lat': target_lat,
                'target_lon': target_lon,
                'pred_lat': pred_lat,
                'pred_lon': pred_lon,
                'error_km': dist_err
            })
            
            # 绘制单个预测
            plot_single_prediction(all_results[-1], output_dir)
    
    # 绘制汇总图
    if len(all_results) > 1:
        plot_all_predictions(all_results, output_dir)
    
    return all_results


def plot_single_prediction(result, output_dir):
    """绘制单个预测结果"""
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 计算地图范围
    all_lats = np.concatenate([result['history_lat'], result['target_lat'], result['pred_lat']])
    all_lons = np.concatenate([result['history_lon'], result['target_lon'], result['pred_lon']])
    
    lat_margin = max(3, (all_lats.max() - all_lats.min()) * 0.2)
    lon_margin = max(3, (all_lons.max() - all_lons.min()) * 0.2)
    
    extent = [
        max(100, all_lons.min() - lon_margin), 
        min(180, all_lons.max() + lon_margin),
        max(0, all_lats.min() - lat_margin), 
        min(60, all_lats.max() + lat_margin)
    ]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # 添加地图特征
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # 绘制历史轨迹（输入）
    ax.plot(result['history_lon'], result['history_lat'], 'b-', linewidth=2.5,
            transform=ccrs.PlateCarree(), label='History (48h)', zorder=5)
    ax.scatter(result['history_lon'], result['history_lat'], c='blue', s=30,
               transform=ccrs.PlateCarree(), zorder=6)
    
    # 绘制真实未来轨迹
    full_true_lon = np.concatenate([[result['history_lon'][-1]], result['target_lon']])
    full_true_lat = np.concatenate([[result['history_lat'][-1]], result['target_lat']])
    ax.plot(full_true_lon, full_true_lat, 'g-', linewidth=2.5,
            transform=ccrs.PlateCarree(), label='Ground Truth (72h)', zorder=7)
    ax.scatter(result['target_lon'], result['target_lat'], c='green', s=30,
               transform=ccrs.PlateCarree(), zorder=8)
    
    # 绘制预测轨迹
    full_pred_lon = np.concatenate([[result['history_lon'][-1]], result['pred_lon']])
    full_pred_lat = np.concatenate([[result['history_lat'][-1]], result['pred_lat']])
    ax.plot(full_pred_lon, full_pred_lat, 'r--', linewidth=2.5,
            transform=ccrs.PlateCarree(), label='Prediction (72h)', zorder=9)
    ax.scatter(result['pred_lon'], result['pred_lat'], c='red', s=30, marker='x',
               transform=ccrs.PlateCarree(), zorder=10)
    
    # 标记起始点
    ax.scatter(result['history_lon'][-1], result['history_lat'][-1], c='black', s=100, marker='*',
               transform=ccrs.PlateCarree(), label='Forecast Start', zorder=11)
    
    # 标题和图例
    mean_err = result['error_km'].mean()
    err_24h = result['error_km'][7] if len(result['error_km']) > 7 else result['error_km'][-1]  # +24h
    err_72h = result['error_km'][-1]  # +72h
    
    plt.title(f"Typhoon {result['storm_id']} - Sample {result['sample_idx']}\n"
              f"Mean Error: {mean_err:.1f} km | +24h: {err_24h:.1f} km | +72h: {err_72h:.1f} km",
              fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    # 保存
    save_path = output_dir / f"pred_{result['storm_id']}_sample{result['sample_idx']:02d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_all_predictions(all_results, output_dir):
    """绘制所有预测结果的汇总图"""
    n_samples = len(all_results)
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    if n_samples == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(all_results):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # 计算地图范围
        all_lats = np.concatenate([result['history_lat'], result['target_lat'], result['pred_lat']])
        all_lons = np.concatenate([result['history_lon'], result['target_lon'], result['pred_lon']])
        
        lat_margin = max(2, (all_lats.max() - all_lats.min()) * 0.15)
        lon_margin = max(2, (all_lons.max() - all_lons.min()) * 0.15)
        
        extent = [all_lons.min() - lon_margin, all_lons.max() + lon_margin,
                  all_lats.min() - lat_margin, all_lats.max() + lat_margin]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        
        # 绘制轨迹
        ax.plot(result['history_lon'], result['history_lat'], 'b-', linewidth=1.5, transform=ccrs.PlateCarree())
        
        full_true_lon = np.concatenate([[result['history_lon'][-1]], result['target_lon']])
        full_true_lat = np.concatenate([[result['history_lat'][-1]], result['target_lat']])
        ax.plot(full_true_lon, full_true_lat, 'g-', linewidth=1.5, transform=ccrs.PlateCarree())
        
        full_pred_lon = np.concatenate([[result['history_lon'][-1]], result['pred_lon']])
        full_pred_lat = np.concatenate([[result['history_lat'][-1]], result['pred_lat']])
        ax.plot(full_pred_lon, full_pred_lat, 'r--', linewidth=1.5, transform=ccrs.PlateCarree())
        
        ax.scatter(result['history_lon'][-1], result['history_lat'][-1], c='black', s=50, marker='*',
                   transform=ccrs.PlateCarree())
        
        mean_err = result['error_km'].mean()
        ax.set_title(f"{result['storm_id']}: {mean_err:.0f} km", fontsize=10)
    
    # 隐藏空白子图
    for idx in range(n_samples, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='History (48h)'),
        Line2D([0], [0], color='green', linewidth=2, label='Ground Truth'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Prediction'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)
    
    plt.suptitle('LT3P Typhoon Track Prediction Results (72h Forecast)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / 'all_predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def main():
    print('='*60)
    print('LT3P Typhoon Track Prediction Visualization')
    print('48h History + 72h ERA5 → 72h Trajectory')
    print('='*60)
    
    model, test_ds, test_storms = load_model_and_data()
    results = predict_and_visualize(model, test_ds, test_storms, max_samples=3, random_select=True)
    
    # 打印误差统计
    print('\n' + '='*60)
    print('Error Statistics (km)')
    print('='*60)
    all_errors = np.array([r['error_km'] for r in results])
    print(f'Overall Mean: {all_errors.mean():.1f} km')
    print(f'Overall Std: {all_errors.std():.1f} km')
    print('\nBy forecast hour:')
    for t in range(all_errors.shape[1]):
        hours = (t + 1) * 3  # 3小时间隔
        print(f'  +{hours:2d}h: {all_errors[:, t].mean():.1f} ± {all_errors[:, t].std():.1f} km')
    
    print(f'\nVisualization saved to: predictions/')


if __name__ == '__main__':
    main()

