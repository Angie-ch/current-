"""
诊断 ERA5 数据加载过程
逐步检查每个环节，找出 ERA5 全零的真正原因
"""
import os
import glob
import numpy as np
from pathlib import Path
import pandas as pd
import xarray as xr
from datetime import timedelta

from config import data_cfg, model_cfg
from data_processing import (
    load_typhoon_csv, parse_nc_filename_time, load_era5_frame,
    load_tyc_storms
)
from dataset import LT3PDataset, split_storms_by_id, filter_short_storms


def main():
    print("=" * 70)
    print("ERA5 数据加载诊断")
    print("=" * 70)

    era5_dir = data_cfg.era5_dir
    csv_path = data_cfg.csv_path
    print(f"\n  era5_dir: {era5_dir}")
    print(f"  csv_path: {csv_path}")

    # === Step 1: 检查 era5_dir 是否存在 ===
    print(f"\n[Step 1] 检查 ERA5 目录是否存在...")
    era5_base = Path(era5_dir)
    if not era5_base.exists():
        print(f"  ❌ 目录不存在: {era5_dir}")
        print(f"     这就是 ERA5 全零的原因！")
        return
    print(f"  ✅ 目录存在")

    # === Step 2: 列出子文件夹 ===
    print(f"\n[Step 2] 列出 ERA5 子文件夹...")
    subfolders = [f for f in era5_base.iterdir() if f.is_dir()]
    print(f"  共 {len(subfolders)} 个子文件夹")
    if subfolders:
        print(f"  前 5 个: {[f.name for f in subfolders[:5]]}")
    else:
        print(f"  ❌ 没有子文件夹！这就是问题所在")
        return

    # === Step 3: 检查 CSV 中的 storm_id 格式 ===
    print(f"\n[Step 3] 检查 CSV storm_id 格式...")
    track_df = load_typhoon_csv(csv_path)
    csv_storm_ids = track_df['storm_id'].unique()
    print(f"  CSV 中共 {len(csv_storm_ids)} 个台风")
    print(f"  前 5 个 storm_id: {csv_storm_ids[:5].tolist()}")

    # === Step 4: 检查 storm_id 和文件夹名的匹配 ===
    print(f"\n[Step 4] 检查 storm_id 与文件夹名匹配...")
    folder_names = set()
    for f in subfolders:
        sid = f.name.replace('_chazhi_finetuned', '')
        folder_names.add(sid)

    matched_ids = [sid for sid in csv_storm_ids if sid in folder_names]
    print(f"  文件夹名（去后缀）: {list(folder_names)[:5]}")
    print(f"  匹配到的 storm_id: {len(matched_ids)} / {len(csv_storm_ids)}")

    if len(matched_ids) == 0:
        print(f"\n  ❌ 没有任何匹配！")
        print(f"     CSV storm_id 示例: {csv_storm_ids[:3].tolist()}")
        print(f"     文件夹名示例: {list(folder_names)[:3]}")
        print(f"     可能是命名格式不一致（比如 '195001' vs '1950-01'）")
        return

    print(f"  匹配的前5个: {matched_ids[:5]}")

    # === Step 5: 深入检查一个台风的 NC 文件 ===
    test_sid = matched_ids[0]
    print(f"\n[Step 5] 深入检查台风 {test_sid} 的 NC 文件...")

    # 找文件夹
    era5_folder = era5_base / test_sid
    if not era5_folder.exists():
        era5_folder = era5_base / f"{test_sid}_chazhi_finetuned"
    print(f"  ERA5 文件夹: {era5_folder}")
    print(f"  文件夹存在: {era5_folder.exists()}")

    if era5_folder.exists():
        all_files = list(era5_folder.iterdir())
        print(f"  文件夹内共 {len(all_files)} 个文件")
        if all_files:
            print(f"  前 3 个文件: {[f.name for f in all_files[:3]]}")

        # 检查 NC 文件 (glob 模式)
        nc_pattern = str(era5_folder / "era5_merged_*_fused.nc")
        nc_files = sorted(glob.glob(nc_pattern))
        print(f"\n  glob 模式: {nc_pattern}")
        print(f"  匹配到的 NC 文件数: {len(nc_files)}")

        if nc_files:
            print(f"  前 3 个: {[Path(f).name for f in nc_files[:3]]}")
        else:
            # 尝试其他模式
            print(f"  ❌ 没有匹配到 'era5_merged_*_fused.nc' 模式的文件")
            nc_any = sorted(glob.glob(str(era5_folder / "*.nc")))
            print(f"  尝试 *.nc: 找到 {len(nc_any)} 个")
            if nc_any:
                print(f"  NC 文件名示例: {[Path(f).name for f in nc_any[:3]]}")
                print(f"  ⚠️ NC 文件存在但命名格式不匹配！")
            return

        # === Step 6: 检查时间对齐 ===
        print(f"\n[Step 6] 检查时间对齐...")
        nc_times = []
        parse_fails = 0
        for nc_file in nc_files:
            nc_time = parse_nc_filename_time(nc_file)
            if nc_time is not None:
                nc_times.append((nc_time, nc_file))
            else:
                parse_fails += 1

        print(f"  NC 时间解析成功: {len(nc_times)}, 失败: {parse_fails}")
        if nc_times:
            print(f"  NC 时间范围: {nc_times[0][0]} ~ {nc_times[-1][0]}")

        # 轨迹时间
        storm_track = track_df[track_df['storm_id'] == test_sid].sort_values('time')
        track_times = storm_track['time'].values
        print(f"  轨迹时间步数: {len(track_times)}")
        if len(track_times) > 0:
            print(f"  轨迹时间范围: {track_times[0]} ~ {track_times[-1]}")

        # 检查对齐
        matched_count = 0
        unmatched_samples = []
        for idx, row in storm_track.iterrows():
            track_time = pd.Timestamp(row['time'])
            best_diff = timedelta(days=999)
            for nc_time, _ in nc_times:
                diff = abs(track_time - nc_time)
                if diff < best_diff:
                    best_diff = diff
            if best_diff <= timedelta(minutes=30):
                matched_count += 1
            else:
                unmatched_samples.append((track_time, best_diff))

        print(f"  时间匹配成功: {matched_count} / {len(storm_track)}")
        if unmatched_samples and len(unmatched_samples) <= 5:
            for t, d in unmatched_samples:
                print(f"    未匹配: {t}, 最近差距: {d}")
        elif unmatched_samples:
            print(f"    前3个未匹配:")
            for t, d in unmatched_samples[:3]:
                print(f"      {t}, 最近差距: {d}")

        # === Step 7: 尝试实际加载一个 NC 文件 ===
        if nc_files:
            print(f"\n[Step 7] 尝试加载第一个 NC 文件...")
            test_nc = nc_files[0]
            print(f"  文件: {Path(test_nc).name}")

            # 先看一下文件内容
            try:
                ds = xr.open_dataset(test_nc)
                print(f"  变量: {list(ds.data_vars)}")
                print(f"  维度: {dict(ds.dims)}")
                if 'pressure_level' in ds.dims:
                    print(f"  气压层: {ds['pressure_level'].values}")
                elif 'level' in ds.dims:
                    print(f"  气压层: {ds['level'].values}")
                else:
                    print(f"  ⚠️ 没有 pressure_level/level 维度！")
                print(f"  坐标: {list(ds.coords)}")

                # 检查配置中的变量是否存在
                for var in data_cfg.era5_3d_vars:
                    if var in ds:
                        print(f"  ✅ 3D变量 '{var}' 存在, shape={ds[var].shape}")
                    else:
                        print(f"  ❌ 3D变量 '{var}' 不存在!")

                for var in data_cfg.era5_2d_vars:
                    if var in ds:
                        print(f"  ✅ 2D变量 '{var}' 存在, shape={ds[var].shape}")
                    else:
                        print(f"  ❌ 2D变量 '{var}' 不存在!")

                ds.close()
            except Exception as e:
                print(f"  ❌ 打开文件出错: {e}")

            # 尝试用 load_era5_frame 加载
            frame = load_era5_frame(test_nc)
            if frame is not None:
                print(f"\n  load_era5_frame 结果:")
                print(f"    shape: {frame.shape}")
                print(f"    值范围: [{frame.min():.4f}, {frame.max():.4f}]")
                print(f"    均值: {frame.mean():.4f}")
                print(f"    全零: {np.abs(frame).sum() == 0}")
                if np.abs(frame).sum() == 0:
                    print(f"  ⚠️ 加载的 frame 全为零！")
                else:
                    print(f"  ✅ ERA5 数据有效!")
            else:
                print(f"  ❌ load_era5_frame 返回 None!")

    # === Step 8: 检查完整加载后的 StormSample ===
    print(f"\n[Step 8] 检查完整加载流程...")
    print(f"  调用 load_tyc_storms() 加载数据...")
    storm_samples = load_tyc_storms(
        csv_path=csv_path,
        era5_base_dir=era5_dir
    )

    if not storm_samples:
        print(f"  ❌ 没有加载到任何台风样本！")
        return

    # 检查 era5_array 状态
    n_has_era5 = 0
    n_era5_zero = 0
    n_era5_none = 0
    for s in storm_samples:
        if s.era5_array is not None:
            n_has_era5 += 1
            if np.abs(s.era5_array).sum() == 0:
                n_era5_zero += 1
        else:
            n_era5_none += 1

    print(f"\n  StormSample 统计:")
    print(f"    总样本数: {len(storm_samples)}")
    print(f"    era5_array 有值: {n_has_era5}")
    print(f"    era5_array 有值但全零: {n_era5_zero}")
    print(f"    era5_array 为 None: {n_era5_none}")
    print(f"    era5_dataset 有值: {sum(1 for s in storm_samples if s.era5_dataset is not None)}")

    if n_era5_none == len(storm_samples):
        print(f"\n  ❌ 所有样本的 era5_array 都是 None！")
        print(f"     dataset._get_era5_video() 会返回全零张量")
        print(f"     这就是 ERA5 全零的原因！")
    elif n_era5_none > 0:
        print(f"\n  ⚠️ {n_era5_none}/{len(storm_samples)} 个样本没有 ERA5 数据")

    # === Step 9: 检查 Dataset 层面 ===
    print(f"\n[Step 9] 检查 Dataset 层面的 ERA5 数据...")
    storm_samples = filter_short_storms(storm_samples, 120, 3)
    _, _, test_s = split_storms_by_id(storm_samples, 0.7, 0.15, seed=42)

    if test_s:
        test_ds = LT3PDataset(test_s, stride=model_cfg.t_future)

        # 抽样检查几个
        n_check = min(5, len(test_ds))
        n_nonzero = 0
        for i in range(n_check):
            batch = test_ds[i]
            era5 = batch['future_era5']
            is_zero = (era5.abs().sum() == 0).item()
            if not is_zero:
                n_nonzero += 1
            print(f"    样本 {i}: ERA5 shape={era5.shape}, "
                  f"值范围=[{era5.min():.4f}, {era5.max():.4f}], "
                  f"全零={'是' if is_zero else '否'}")

        print(f"\n  结果: {n_nonzero}/{n_check} 个样本有非零 ERA5")

    print(f"\n{'=' * 70}")
    print("诊断完成")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
