r"""
方法A模型评估脚本
Evaluate Method A (LT3P) with Real ERA5 and Diffusion ERA5 inputs

用途：
  在测试集上评估方法A（真实ERA5训练的模型），分别使用：
    1. 真实ERA5输入 → 方法A的理想上界
    2. 扩散模型生成的ERA5输入 → 方法A+B级联（方法B的实际表现）

运行示例：
  python evalA.py ^
      --checkpoint checkpoints/best.pt ^
      --diffusion_code C:\Users\fyp\Desktop\newtry ^
      --diffusion_ckpt C:\Users\fyp\Desktop\newtry\checkpoints\best.pt ^
      --norm_stats C:\Users\fyp\Desktop\newtry\norm_stats.pt ^
      --data_root C:\Users\fyp\Desktop\Typhoon_data_final ^
      --cache_dir diffusion_era5_cache

哈雷酱出品 (￣▽￣)／
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# 确保能找到本目录的模块
TRAJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAJ_DIR)

from config import model_cfg, data_cfg, train_cfg
from model import LT3PModel
from dataset import (
    LT3PDataset,
    filter_short_storms,
    filter_out_of_range_storms,
    split_storms_by_id,
)
from data_processing import load_tyc_storms
from train import evaluate_on_test

# 复用 finetune_train.py 中已有的组件
from finetune_train import (
    DiffusionERA5Dataset,
    generate_diffusion_era5_cache,
    data_root_to_era5_dir,
)


def load_method_a_model(checkpoint_path: str, device: str) -> LT3PModel:
    """加载方法A的 LT3P 模型（优先使用EMA权重）"""
    print(f"\n[加载] 方法A模型: {checkpoint_path}")

    model = LT3PModel(
        coord_dim=model_cfg.coord_dim,
        output_dim=model_cfg.output_dim,
        era5_channels=model_cfg.era5_channels,
        t_history=model_cfg.t_history,
        t_future=model_cfg.t_future,
        d_model=model_cfg.transformer_dim,
        n_heads=model_cfg.transformer_heads,
        n_layers=model_cfg.transformer_layers,
        ff_dim=model_cfg.transformer_ff_dim,
        dropout=model_cfg.dropout,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 优先使用EMA权重
    if 'ema_model_state_dict' in ckpt:
        print("[加载] 使用 EMA 模型权重")
        state_dict = ckpt['ema_model_state_dict']
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    elif 'model_state_dict' in ckpt:
        print("[加载] 使用标准模型权重")
        state_dict = ckpt['model_state_dict']
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        raise KeyError("Checkpoint 中找不到 model_state_dict 或 ema_model_state_dict")

    epoch = ckpt.get('epoch', '?')
    best_val = ckpt.get('best_val_loss', '?')
    print(f"[加载] Epoch: {epoch}, Best Val Loss: {best_val}")

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[加载] 参数量: {num_params:,}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="方法A模型评估 - 真实ERA5 vs 扩散ERA5"
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                        help='方法A模型 checkpoint 路径')
    parser.add_argument('--diffusion_code', type=str, required=True,
                        help='扩散模型代码目录 (e.g. C:\\Users\\fyp\\Desktop\\newtry)')
    parser.add_argument('--diffusion_ckpt', type=str, required=True,
                        help='扩散模型 checkpoint 路径')
    parser.add_argument('--norm_stats', type=str, required=True,
                        help='扩散模型归一化统计量路径 (norm_stats.pt)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='ERA5 数据根目录')
    parser.add_argument('--track_csv', type=str, default='processed_typhoon_tracks.csv',
                        help='台风轨迹 CSV 文件')
    parser.add_argument('--cache_dir', type=str, default='diffusion_era5_cache',
                        help='扩散ERA5缓存目录')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='DDIM 采样步数')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备 (默认自动检测)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='评估 batch size')
    parser.add_argument('--preprocess_dir', type=str, default=None,
                        help='扩散模型预处理NPY目录')
    args = parser.parse_args()

    # 设备
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("  方法A模型评估")
    print("  真实ERA5 vs 扩散ERA5")
    print("  哈雷酱出品 (￣▽￣)／")
    print("=" * 60)
    print(f"[设备] {device}")

    # ---- 步骤1: 加载台风轨迹数据 ----
    print("\n[1/5] 加载台风轨迹数据...")
    track_csv = args.track_csv
    if not os.path.isabs(track_csv):
        track_csv = os.path.join(TRAJ_DIR, track_csv)

    storm_samples = load_tyc_storms(
        csv_path=track_csv,
        era5_base_dir=data_root_to_era5_dir(args.data_root),
    )
    if not storm_samples:
        print("错误: 没有找到台风数据!")
        return

    # ---- 步骤2: 过滤 + 划分数据集 (与 train.py 一致) ----
    print("\n[2/5] 过滤和划分数据集...")
    storm_samples = filter_short_storms(
        storm_samples,
        min_duration_hours=train_cfg.min_typhoon_duration_hours,
    )
    storm_samples = filter_out_of_range_storms(storm_samples)

    train_storms, val_storms, test_storms = split_storms_by_id(
        storm_samples,
        train_ratio=train_cfg.train_ratio,
        val_ratio=train_cfg.val_ratio,
        seed=42,
    )
    print(f"训练台风: {len(train_storms)}, 验证台风: {len(val_storms)}, 测试台风: {len(test_storms)}")

    if not test_storms:
        print("错误: 测试集为空!")
        return

    # ---- 步骤3: 加载方法A模型 ----
    print("\n[3/5] 加载方法A模型...")
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(TRAJ_DIR, ckpt_path)
    model = load_method_a_model(ckpt_path, device)

    # ---- 步骤4: 真实ERA5输入评估 ----
    print("\n[4/5] 评估: 真实ERA5输入...")
    real_test_ds = LT3PDataset(test_storms, stride=model_cfg.t_future)
    real_test_loader = DataLoader(
        real_test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    print(f"真实ERA5测试样本数: {len(real_test_ds)}")

    print("\n--- 测试结果 (真实ERA5输入) ---")
    real_results = evaluate_on_test(model, real_test_loader, device)

    # ---- 步骤5: 扩散ERA5输入评估 ----
    print(f"\n[5/5] 评估: 扩散ERA5输入...")

    # 加载或生成扩散ERA5缓存
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = Path(TRAJ_DIR) / cache_dir
    cache_file = cache_dir / 'era5_cache.npz'

    if cache_file.exists():
        print(f"[缓存] 加载已有缓存: {cache_file}")
        cache_data = np.load(cache_file, allow_pickle=True)
        diffusion_cache = {k: cache_data[k] for k in cache_data.files}
        print(f"[缓存] 加载了 {len(diffusion_cache)} 个台风的缓存")

        # 检查测试集台风是否都在缓存中
        missing = [s for s in test_storms if s.storm_id not in diffusion_cache]
        if missing:
            print(f"[缓存] 测试集有 {len(missing)} 个台风不在缓存中，补充生成...")
            extra_cache = generate_diffusion_era5_cache(
                storm_samples=missing,
                diffusion_code=args.diffusion_code,
                diffusion_ckpt=args.diffusion_ckpt,
                norm_stats_path=args.norm_stats,
                data_root=args.data_root,
                device=device,
                ddim_steps=args.ddim_steps,
                preprocess_dir=args.preprocess_dir,
            )
            diffusion_cache.update(extra_cache)
            np.savez_compressed(cache_file, **diffusion_cache)
            print(f"[缓存] 缓存已更新: {len(diffusion_cache)} 个台风")
    else:
        print(f"[缓存] 未找到缓存，开始生成...")
        # 为所有台风生成缓存（训练+验证+测试），方便后续复用
        all_storms = train_storms + val_storms + test_storms
        diffusion_cache = generate_diffusion_era5_cache(
            storm_samples=all_storms,
            diffusion_code=args.diffusion_code,
            diffusion_ckpt=args.diffusion_ckpt,
            norm_stats_path=args.norm_stats,
            data_root=args.data_root,
            device=device,
            ddim_steps=args.ddim_steps,
            preprocess_dir=args.preprocess_dir,
        )

        # 保存缓存
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, **diffusion_cache)
        print(f"[缓存] 已保存到: {cache_file}")

    # 创建扩散ERA5数据集和评估
    diff_test_ds = DiffusionERA5Dataset(
        test_storms,
        diffusion_cache,
        stride=model_cfg.t_future,
    )

    if len(diff_test_ds) > 0:
        diff_test_loader = DataLoader(
            diff_test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        print(f"扩散ERA5测试样本数: {len(diff_test_ds)}")

        print("\n--- 测试结果 (扩散ERA5输入) ---")
        diff_results = evaluate_on_test(model, diff_test_loader, device)
    else:
        print("警告: 扩散ERA5测试集为空，跳过评估")
        diff_results = {}

    # ---- 汇总结果 ----
    print("\n" + "=" * 60)
    print("  评估结果汇总")
    print("=" * 60)

    if real_results:
        print(f"\n[真实ERA5] (方法A的理想上界)")
        print(f"  平均误差: {real_results.get('mean_error_km', 'N/A'):.2f} km")
        print(f"  平均误差(去异常值): {real_results.get('mean_error_km_filtered', 'N/A'):.2f} km")
        for key_hour in [24, 48, 72]:
            h_key = f"{key_hour}h"
            if h_key in real_results.get('error_by_hour', {}):
                err = real_results['error_by_hour'][h_key]['mean_km']
                print(f"  +{key_hour}h: {err:.2f} km")

    if diff_results:
        print(f"\n[扩散ERA5] (方法A+B级联)")
        print(f"  平均误差: {diff_results.get('mean_error_km', 'N/A'):.2f} km")
        print(f"  平均误差(去异常值): {diff_results.get('mean_error_km_filtered', 'N/A'):.2f} km")
        for key_hour in [24, 48, 72]:
            h_key = f"{key_hour}h"
            if h_key in diff_results.get('error_by_hour', {}):
                err = diff_results['error_by_hour'][h_key]['mean_km']
                print(f"  +{key_hour}h: {err:.2f} km")

    if real_results and diff_results:
        real_err = real_results.get('mean_error_km', 0)
        diff_err = diff_results.get('mean_error_km', 0)
        gap = diff_err - real_err
        print(f"\n[性能差距] 扩散ERA5 - 真实ERA5 = {gap:+.2f} km")

    # 保存结果到JSON
    results_path = Path(TRAJ_DIR) / 'evalA_results.json'
    combined_results = {
        'real_era5': real_results,
        'diffusion_era5': diff_results,
        'config': {
            'checkpoint': args.checkpoint,
            'diffusion_ckpt': args.diffusion_ckpt,
            'ddim_steps': args.ddim_steps,
            'device': device,
            'num_test_storms': len(test_storms),
            'test_storm_ids': [s.storm_id for s in test_storms],
        }
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 结果已保存到: {results_path}")

    print("\n评估完成！ o(￣▽￣)ｄ")


if __name__ == '__main__':
    main()
