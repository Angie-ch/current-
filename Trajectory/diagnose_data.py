"""
数据质量诊断脚本 - 快速定位异常数据
输出异常台风ID列表，便于快速定位
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import data_cfg, model_cfg
from data_processing import load_tyc_storms
from dataset import LT3PDataset, split_storms_by_id, filter_short_storms


def main():
    print("="*60)
    print("台风数据质量诊断 - 快速定位异常")
    print("="*60)
    
    # 加载数据
    print("\n加载数据...")
    storm_samples = load_tyc_storms(
        csv_path=data_cfg.csv_path,
        era5_base_dir=data_cfg.era5_dir
    )
    
    # ============ 1. 诊断原始台风数据 ============
    print("\n" + "="*60)
    print("【异常台风列表】")
    print("="*60)
    
    abnormal_storms = []
    
    for sample in tqdm(storm_samples, desc="检查中"):
        issues = []
        
        # 检查坐标范围
        if sample.track_lat.min() < 5 or sample.track_lat.max() > 45:
            issues.append("纬度越界")
        if sample.track_lon.min() < 100 or sample.track_lon.max() > 180:
            issues.append("经度越界")
        
        # 检查NaN
        if np.isnan(sample.track_lat).any() or np.isnan(sample.track_lon).any():
            issues.append("轨迹含NaN")
        
        # 检查移动速度
        if len(sample.track_lat) > 1:
            dlat = np.diff(sample.track_lat)
            dlon = np.diff(sample.track_lon)
            dist_km = np.sqrt((dlat * 111) ** 2 + (dlon * 111 * np.cos(np.radians(sample.track_lat[:-1]))) ** 2)
            speed_kmh = dist_km / 3
            if speed_kmh.max() > 150:
                issues.append(f"速度异常({speed_kmh.max():.0f}km/h)")
        
        # 检查ERA5
        if sample.era5_array is not None:
            if np.isnan(sample.era5_array).any():
                issues.append("ERA5含NaN")
            zero_frames = (np.abs(sample.era5_array).sum(axis=(1,2,3)) == 0).sum()
            if zero_frames > 0:
                issues.append(f"ERA5有{zero_frames}个空帧")
        else:
            issues.append("无ERA5")
        
        if issues:
            abnormal_storms.append({
                'storm_id': sample.storm_id,
                'issues': ', '.join(issues)
            })
    
    # 打印异常列表
    if abnormal_storms:
        print(f"\n发现 {len(abnormal_storms)} 个异常台风:\n")
        for s in abnormal_storms:
            print(f"  {s['storm_id']}: {s['issues']}")
        
        # 保存到文件
        df = pd.DataFrame(abnormal_storms)
        df.to_csv('abnormal_storms.csv', index=False)
        print(f"\n已保存到: abnormal_storms.csv")
    else:
        print("\n✅ 未发现异常台风")
    
    # ============ 2. 诊断预测误差异常 ============
    print("\n" + "="*60)
    print("【预测误差异常样本】")
    print("="*60)
    
    try:
        import torch
        from model import LT3PModel
        
        # 创建测试集
        storm_samples = filter_short_storms(storm_samples, 120, 3)
        _, _, test_s = split_storms_by_id(storm_samples, 0.7, 0.15, seed=42)
        test_ds = LT3PDataset(test_s, stride=model_cfg.t_future)
        
        # 加载模型
        checkpoint = torch.load('checkpoints/best.pt', map_location='cuda', weights_only=False)
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
        
        # 计算每个样本的误差
        lat_range, lon_range = data_cfg.lat_range, data_cfg.lon_range
        errors = []
        
        with torch.no_grad():
            for idx in tqdm(range(len(test_ds)), desc="预测中"):
                batch = test_ds[idx]
                history_coords = batch['history_coords'].unsqueeze(0).cuda()
                future_era5 = batch['future_era5'].unsqueeze(0).cuda()
                target_lat = batch['target_lat_raw'].numpy()
                target_lon = batch['target_lon_raw'].numpy()
                
                outputs = model.predict(history_coords, future_era5)
                pred = outputs['predicted_coords'].cpu().numpy()[0]
                pred_lat = pred[:, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
                pred_lon = pred[:, 1] * (lon_range[1] - lon_range[0]) + lon_range[0]
                
                lat_err = (pred_lat - target_lat) * 111
                lon_err = (pred_lon - target_lon) * 111 * np.cos(np.radians(target_lat))
                dist_err = np.sqrt(lat_err**2 + lon_err**2)
                
                errors.append({
                    'idx': idx,
                    'storm_id': batch['storm_id'],
                    'mean_error': dist_err.mean(),
                    'error_72h': dist_err[-1]
                })
        
        df_err = pd.DataFrame(errors)
        threshold = df_err['mean_error'].quantile(0.95)
        outliers = df_err[df_err['mean_error'] > threshold].sort_values('mean_error', ascending=False)
        
        print(f"\n误差 > {threshold:.0f} km 的异常样本 (共{len(outliers)}个):\n")
        for _, row in outliers.iterrows():
            print(f"  {row['storm_id']} (idx={row['idx']}): 平均误差 {row['mean_error']:.0f} km, +72h {row['error_72h']:.0f} km")
        
        # 保存
        outliers.to_csv('abnormal_predictions.csv', index=False)
        print(f"\n已保存到: abnormal_predictions.csv")
        
    except Exception as e:
        print(f"\n⚠️ 无法加载模型: {e}")
    
    print("\n" + "="*60)
    print("诊断完成!")
    print("="*60)


if __name__ == '__main__':
    main()
