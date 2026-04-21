"""
预处理：将每个台风的多个 ERA5 NC 文件合并为单个 .npy 文件
运行一次即可，之后所有脚本秒加载

输入: Typhoon_data_final/{storm_id}/era5_merged_*.nc
输出: preprocessed_era5/{storm_id}.npy  (T, 12, 40, 40)
      preprocessed_era5/{storm_id}_times.npy  (T,) 对应时间戳
"""
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import data_cfg
from data_processing import (
    load_typhoon_csv, parse_nc_filename_time, load_era5_frame
)


def preprocess_all(
    era5_base_dir: str = None,
    csv_path: str = None,
    output_dir: str = "preprocessed_era5",
):
    era5_base_dir = era5_base_dir or data_cfg.era5_dir
    csv_path = csv_path or data_cfg.csv_path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 加载 CSV 获取 storm_id 列表
    track_df = load_typhoon_csv(csv_path)
    print(f"CSV: {len(track_df)} records, {track_df['storm_id'].nunique()} storms")

    # 扫描 ERA5 文件夹
    era5_base = Path(era5_base_dir)
    storm_folders = []
    for folder in era5_base.iterdir():
        if folder.is_dir():
            sid = folder.name.replace('_chazhi_finetuned', '')
            if sid in track_df['storm_id'].values:
                storm_folders.append((sid, folder))

    print(f"Found {len(storm_folders)} storms with ERA5 data")
    print(f"Output: {output_path.absolute()}")
    print()

    success = 0
    failed = 0
    skipped = 0
    total_size_mb = 0

    for sid, folder in tqdm(storm_folders, desc="Converting NC → NPY"):
        npy_path = output_path / f"{sid}.npy"
        times_path = output_path / f"{sid}_times.npy"

        # 跳过已处理的
        if npy_path.exists():
            skipped += 1
            continue

        # 获取所有 NC 文件
        nc_files = sorted(glob.glob(str(folder / "era5_merged_*.nc")))
        if not nc_files:
            failed += 1
            continue

        # 解析时间并加载
        frames = []
        timestamps = []
        for nc_file in nc_files:
            nc_time = parse_nc_filename_time(nc_file)
            if nc_time is None:
                continue
            frame = load_era5_frame(nc_file)
            if frame is not None:
                frames.append(frame)
                timestamps.append(nc_time.value)  # 存为 int64 纳秒

        if not frames:
            failed += 1
            continue

        # 堆叠并保存
        era5_array = np.stack(frames, axis=0)  # (T, C, H, W)
        times_array = np.array(timestamps, dtype=np.int64)

        np.save(npy_path, era5_array)
        np.save(times_path, times_array)

        total_size_mb += era5_array.nbytes / 1024 / 1024
        success += 1

    print(f"\n{'=' * 50}")
    print(f"Done!")
    print(f"  Success: {success}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total new data: {total_size_mb:.0f} MB")
    print(f"  Output dir: {output_path.absolute()}")


if __name__ == '__main__':
    preprocess_all()
