"""
数据处理模块：读取 CSV 和 ERA5 数据，进行对齐
Video2Video 条件扩散模型 - 台风路径预测
"""
import os
import glob
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta, datetime

from config import data_cfg
from data_structures import StormSample


def load_typhoon_csv(csv_path: str) -> pd.DataFrame:
    """
    读取台风轨迹 CSV 文件

    支持两种格式:
    1. 标准格式: storm_id, time, lat, lon, vmax, pmin
    2. TYC格式: typhoon_id, typhoon_name, time, year, lat, lon, wind, pressure
    """
    df = pd.read_csv(csv_path)

    # 检测并转换列名（适配TYC格式）
    column_mapping = {
        'typhoon_id': 'storm_id',
        'wind': 'vmax',
        'pressure': 'pmin'
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})

    # 处理时间格式
    if 'time' in df.columns:
        time_sample = df['time'].iloc[0]
        # 检查时间是否为整数格式（如60500表示第6天05:00）
        if isinstance(time_sample, (int, np.integer)) or (isinstance(time_sample, str) and time_sample.isdigit()):
            # TYC格式: 时间为整数，如60500表示第6天05:00，需要配合year列
            df = _convert_tyc_time_format(df)
        else:
            # 标准datetime格式
            df['time'] = pd.to_datetime(df['time'])

    # 按 storm_id 分组（保留组内原始行顺序，避免跨月time字段排序错乱）
    df = df.sort_values(['storm_id'], kind='mergesort').reset_index(drop=True)

    return df


def _convert_tyc_time_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    将TYC时间格式转换为标准datetime

    TYC格式: time为整数，格式为 DDMMHH
    例如: 60500 = 06日 05月 00时 = 5月6日 00:00
         150712 = 15日 07月 12时 = 7月15日 12:00
    """
    def parse_tyc_time(row):
        time_val = int(row['time'])
        year = int(row['year']) if 'year' in row else 2020

        # 解析 DDMMHH 格式
        # 从右往左: HH(2位) MM(2位) DD(剩余位)
        hour = time_val % 100
        time_val //= 100
        month = time_val % 100
        day = time_val // 100

        # 处理边界情况
        if day == 0:
            day = 1
        if month == 0:
            month = 1

        try:
            result = pd.Timestamp(year=year, month=month, day=day, hour=hour)
            return result
        except:
            return pd.NaT

    df['time'] = df.apply(parse_tyc_time, axis=1)

    return df


def load_era5_for_storm(storm_id: str, era5_dir: str) -> Optional[xr.Dataset]:
    """
    加载指定台风的 ERA5 数据集
    
    假设文件命名为：{storm_id}.nc 或 {storm_id}.zarr
    """
    nc_path = Path(era5_dir) / f"{storm_id}.nc"
    zarr_path = Path(era5_dir) / f"{storm_id}.zarr"
    
    if nc_path.exists():
        return xr.open_dataset(nc_path)
    elif zarr_path.exists():
        return xr.open_zarr(zarr_path)
    else:
        print(f"Warning: ERA5 data not found for storm {storm_id}")
        return None


def mark_real_vs_interpolated(times: np.ndarray, original_resolution_hours: float = 3.0) -> np.ndarray:
    """
    标记哪些时间步是真实观测（3h 间隔），哪些是插值数据
    
    假设原始数据是 3 小时间隔，0, 3, 6, 9, ... 点整为真实数据
    """
    is_real = np.zeros(len(times), dtype=bool)
    
    for i, t in enumerate(times):
        # 转换为 pandas Timestamp 以便处理
        ts = pd.Timestamp(t)
        # 检查是否是 3 小时的整点（0, 3, 6, 9, 12, 15, 18, 21）
        if ts.hour % 3 == 0 and ts.minute == 0:
            is_real[i] = True
    
    return is_real


def align_track_and_era5(
    track_df: pd.DataFrame,
    era5_ds: xr.Dataset,
    time_tolerance_minutes: int = 5
) -> Tuple[pd.DataFrame, xr.Dataset]:
    """
    对齐轨迹数据和 ERA5 数据的时间
    
    返回时间对齐后的 DataFrame 和 Dataset
    """
    # 获取两者的时间范围
    track_times = track_df['time'].values
    era5_times = era5_ds['valid_time'].values
    
    # 找到共同的时间范围
    common_start = max(track_times.min(), era5_times.min())
    common_end = min(track_times.max(), era5_times.max())
    
    # 筛选
    track_df = track_df[
        (track_df['time'] >= common_start) & 
        (track_df['time'] <= common_end)
    ].copy()
    
    era5_ds = era5_ds.sel(
        valid_time=slice(common_start, common_end)
    )
    
    return track_df, era5_ds


def create_storm_sample(
    storm_id: str,
    track_df: pd.DataFrame,
    era5_ds: Optional[xr.Dataset] = None
) -> StormSample:
    """
    从轨迹 DataFrame 和 ERA5 Dataset 创建 StormSample
    """
    # 提取轨迹数据
    storm_data = track_df[track_df['storm_id'] == storm_id].copy()
    # 不按time排序，保留原始行顺序（IBTrACS原始数据已按时间正序）

    times = storm_data['time'].values
    track_lat = storm_data['lat'].values.astype(np.float32)
    track_lon = storm_data['lon'].values.astype(np.float32)
    track_vmax = storm_data['vmax'].values.astype(np.float32)
    track_pmin = storm_data['pmin'].values.astype(np.float32) if 'pmin' in storm_data else None
    
    # 标记真实 vs 插值
    is_real = mark_real_vs_interpolated(times)
    
    # 获取网格坐标
    lat_grid = None
    lon_grid = None
    if era5_ds is not None:
        lat_grid = era5_ds['latitude'].values
        lon_grid = era5_ds['longitude'].values
    
    # 获取年份
    year = pd.Timestamp(times[0]).year if len(times) > 0 else None
    
    # 获取 basin（如果有）
    basin = storm_data['basin'].iloc[0] if 'basin' in storm_data else None
    
    return StormSample(
        storm_id=storm_id,
        times=times,
        track_lat=track_lat,
        track_lon=track_lon,
        track_vmax=track_vmax,
        track_pmin=track_pmin,
        era5_dataset=era5_ds,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        is_real=is_real,
        basin=basin,
        year=year
    )


def load_all_storms(
    csv_path: str = None,
    era5_dir: str = None,
    storm_ids: Optional[List[str]] = None
) -> List[StormSample]:
    """
    加载所有台风数据，返回 StormSample 列表
    """
    csv_path = csv_path or data_cfg.csv_path
    era5_dir = era5_dir or data_cfg.era5_dir
    
    # 加载 CSV
    track_df = load_typhoon_csv(csv_path)
    
    # 获取所有 storm_id
    if storm_ids is None:
        storm_ids = track_df['storm_id'].unique().tolist()
    
    samples = []
    for sid in storm_ids:
        # 加载对应的 ERA5 数据
        era5_ds = load_era5_for_storm(sid, era5_dir)
        
        # 如果有 ERA5 数据，进行对齐
        storm_track = track_df[track_df['storm_id'] == sid]
        if era5_ds is not None:
            storm_track, era5_ds = align_track_and_era5(storm_track, era5_ds)
        
        # 创建样本
        sample = create_storm_sample(sid, storm_track, era5_ds)
        
        if len(sample) > 0:
            samples.append(sample)
    
    print(f"Loaded {len(samples)} storm samples")
    return samples


def load_tyc_storms(
    csv_path: str = None,
    era5_base_dir: str = None,
    storm_ids: Optional[List[str]] = None,
) -> List[StormSample]:
    """
    加载TYC格式的台风数据（带ERA5 NC文件）
    如果存在 preprocessed_era5/ 目录下的 .npy 文件，优先从 npy 加载（秒读）

    TYC格式特点：
    - ERA5文件夹命名：{storm_id}/ 或 {storm_id}_chazhi_finetuned/
    - NC文件命名：era5_merged_{YYYYMMDDHH}_{storm_id}.nc
    - CSV时间格式：DDMMHH
    """
    csv_path = csv_path or data_cfg.csv_path
    era5_base_dir = era5_base_dir or data_cfg.era5_dir

    # 加载CSV
    track_df = load_typhoon_csv(csv_path)
    print(f"Loaded CSV with {len(track_df)} records, {track_df['storm_id'].nunique()} unique storms")

    # 如果没有指定storm_ids，从ERA5目录自动检测可用的台风
    if storm_ids is None:
        storm_ids = []
        era5_base = Path(era5_base_dir)
        if era5_base.exists():
            for folder in era5_base.iterdir():
                if folder.is_dir():
                    # 支持两种命名格式：{storm_id} 或 {storm_id}_chazhi_finetuned
                    sid = folder.name.replace('_chazhi_finetuned', '')
                    if sid in track_df['storm_id'].values:
                        storm_ids.append(sid)
        print(f"Found {len(storm_ids)} storms with ERA5 data: {storm_ids[:10]}..." if len(storm_ids) > 10 else f"Found {len(storm_ids)} storms with ERA5 data: {storm_ids}")

    if not storm_ids:
        print("No storms found with ERA5 data, falling back to CSV-only mode")
        storm_ids = track_df['storm_id'].unique().tolist()[:5]  # 取前5个作为测试

    samples = []
    skipped = 0
    from tqdm import tqdm
    for sid in tqdm(storm_ids, desc="Loading storms"):
        sample = load_single_tyc_storm(sid, track_df, era5_base_dir, verbose=False)
        if sample is not None and len(sample) >= 24:  # 至少需要24个时间步
            samples.append(sample)
        else:
            skipped += 1

    print(f"Total loaded: {len(samples)} storm samples (skipped {skipped} due to insufficient data)")
    return samples


def load_single_tyc_storm(
    storm_id: str,
    track_df: pd.DataFrame,
    era5_base_dir: str,
    verbose: bool = True,
    npy_dir: str = "preprocessed_era5",
) -> Optional[StormSample]:
    """加载单个TYC格式的台风（优先读 .npy 缓存）"""
    # 获取该台风的轨迹数据
    storm_data = track_df[track_df['storm_id'] == storm_id].copy()
    if len(storm_data) == 0:
        if verbose:
            print(f"  Warning: No track data for {storm_id}")
        return None

    # 不按time排序，保留原始行顺序（避免跨月time排序错乱）

    era5_array = None
    lat_grid = None
    lon_grid = None

    # === 优先从 .npy 加载（秒读） ===
    npy_path = Path(npy_dir) / f"{storm_id}.npy"
    times_npy_path = Path(npy_dir) / f"{storm_id}_times.npy"

    if npy_path.exists() and times_npy_path.exists():
        era5_all = np.load(npy_path)  # (T_era5, C, H, W)
        era5_times_ns = np.load(times_npy_path)  # (T_era5,) int64 纳秒
        era5_timestamps = [pd.Timestamp(t) for t in era5_times_ns]

        # 轨迹时间与 ERA5 时间对齐
        matched_data = []
        matched_era5_idx = []

        for idx, row in storm_data.iterrows():
            track_time = pd.Timestamp(row['time'])
            best_idx = None
            min_diff = timedelta(minutes=31)

            for i, era5_t in enumerate(era5_timestamps):
                diff = abs(track_time - era5_t)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i

            if best_idx is not None and min_diff <= timedelta(minutes=30):
                matched_data.append(row)
                matched_era5_idx.append(best_idx)

        if matched_data:
            storm_data = pd.DataFrame(matched_data)
            era5_array = era5_all[matched_era5_idx]  # (T_matched, C, H, W)

    else:
        # === 回退：从 NC 文件加载 ===
        era5_folder = Path(era5_base_dir) / storm_id
        if not era5_folder.exists():
            era5_folder = Path(era5_base_dir) / f"{storm_id}_chazhi_finetuned"

        if era5_folder.exists():
            nc_files = sorted(glob.glob(str(era5_folder / "era5_merged_*.nc")))

            if nc_files:
                nc_times = []
                for nc_file in nc_files:
                    nc_time = parse_nc_filename_time(nc_file)
                    if nc_time is not None:
                        nc_times.append((nc_time, nc_file))

                if nc_times:
                    nc_times.sort(key=lambda x: x[0])

                    matched_data = []
                    matched_files = []

                    for idx, row in storm_data.iterrows():
                        track_time = pd.Timestamp(row['time'])
                        best_match = None
                        min_diff = timedelta(minutes=31)

                        for nc_time, nc_file in nc_times:
                            diff = abs(track_time - nc_time)
                            if diff < min_diff:
                                min_diff = diff
                                best_match = (nc_time, nc_file)

                        if best_match is not None and min_diff <= timedelta(minutes=30):
                            matched_data.append(row)
                            matched_files.append(best_match[1])

                    if matched_data:
                        storm_data = pd.DataFrame(matched_data)

                        era5_frames = []
                        for nc_file in matched_files:
                            frame = load_era5_frame(nc_file)
                            if frame is not None:
                                era5_frames.append(frame)

                        if era5_frames and len(era5_frames) == len(matched_files):
                            era5_array = np.stack(era5_frames, axis=0)
                            ds = xr.open_dataset(matched_files[0])
                            lat_grid = ds['latitude'].values
                            lon_grid = ds['longitude'].values
                            ds.close()

    if len(storm_data) == 0:
        return None

    # 创建StormSample
    times = storm_data['time'].values
    track_lat = storm_data['lat'].values.astype(np.float32)
    track_lon = storm_data['lon'].values.astype(np.float32)
    track_vmax = storm_data['vmax'].values.astype(np.float32)
    track_pmin = storm_data['pmin'].values.astype(np.float32) if 'pmin' in storm_data.columns else None

    # 标记真实vs插值
    is_real = mark_real_vs_interpolated(times)

    year = pd.Timestamp(times[0]).year if len(times) > 0 else None

    return StormSample(
        storm_id=storm_id,
        times=times,
        track_lat=track_lat,
        track_lon=track_lon,
        track_vmax=track_vmax,
        track_pmin=track_pmin,
        era5_array=era5_array,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        is_real=is_real,
        year=year
    )


def parse_nc_filename_time(nc_path: str) -> Optional[pd.Timestamp]:
    """从NC文件名解析时间

    支持两种格式:
    1. era5_merged_{YYYYMMDD}_{YYYYMMDD}_{HHMM}_fused.nc (旧 fused 格式)
    2. era5_merged_{YYYYMMDDHH}_{storm_id}.nc (新格式)
       例: era5_merged_1950050600_1950126N09151.nc -> 1950-05-06 00:00
    """
    filename = Path(nc_path).name
    try:
        core = filename.replace('era5_merged_', '').replace('.nc', '')
        parts = core.split('_')

        if 'fused' in parts:
            # 旧格式: {YYYYMMDD}_{YYYYMMDD}_{HHMM}_fused
            date_str = parts[0]  # YYYYMMDD
            time_str = parts[2]  # HHMM

            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[:2])
            minute = int(time_str[2:4]) if len(time_str) >= 4 else 0
        else:
            # 新格式: {YYYYMMDDHH}_{storm_id}
            datetime_str = parts[0]  # YYYYMMDDHH (10位)
            year = int(datetime_str[:4])
            month = int(datetime_str[4:6])
            day = int(datetime_str[6:8])
            hour = int(datetime_str[8:10])
            minute = 0

        return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
    except Exception as e:
        pass
    return None


def load_era5_frame(nc_path: str, target_size: tuple = None) -> Optional[np.ndarray]:
    """加载单个ERA5 NC文件并转换为numpy数组 (C, H, W)

    通道顺序与 ERA5-Diffusion 扩散模型一致:
    按 era5_3d_vars 的顺序, 每个变量按 pressure_levels 逐层排列
    例: [u_850, u_500, u_250, v_850, v_500, v_250, z_850, z_500, z_250]  (9通道)

    Args:
        nc_path: NC文件路径
        target_size: 目标空间尺寸 (H, W)，默认使用 config 中的 grid_height/grid_width
    """
    if target_size is None:
        target_size = (data_cfg.grid_height, data_cfg.grid_width)

    try:
        ds = xr.open_dataset(nc_path)
        channels = []

        # 3D 变量（按配置的变量和气压层提取）
        for var in data_cfg.era5_3d_vars:
            if var in ds:
                for level in data_cfg.pressure_levels:
                    try:
                        # 优先用 xarray 的 sel 选择气压层
                        if 'pressure_level' in ds[var].dims:
                            data = ds[var].sel(pressure_level=level).values
                        elif 'level' in ds[var].dims:
                            data = ds[var].sel(level=level).values
                        else:
                            # 没有气压层维度, 跳过
                            continue

                        # 处理可能的时间维度: (time, lat, lon) -> (lat, lon)
                        if data.ndim == 3:
                            data = data[0]
                        channels.append(data[np.newaxis].astype(np.float32))
                    except (KeyError, ValueError):
                        # 该气压层不在文件中, 尝试最近的
                        try:
                            if 'pressure_level' in ds[var].dims:
                                avail = ds[var].pressure_level.values
                            elif 'level' in ds[var].dims:
                                avail = ds[var].level.values
                            else:
                                continue
                            nearest = avail[np.argmin(np.abs(avail - level))]
                            data = ds[var].sel(
                                **{('pressure_level' if 'pressure_level' in ds[var].dims else 'level'): nearest}
                            ).values
                            if data.ndim == 3:
                                data = data[0]
                            channels.append(data[np.newaxis].astype(np.float32))
                        except Exception:
                            pass

        # 2D 变量
        for var in data_cfg.era5_2d_vars:
            if var in ds:
                data = ds[var].values
                if data.ndim == 3:
                    data = data[0]  # 去掉时间维度
                channels.append(data[np.newaxis].astype(np.float32))

        ds.close()

        if channels:
            result = np.concatenate(channels, axis=0)
            # 调整空间尺寸
            if result.shape[-2:] != target_size:
                result = _resize_era5_array(result, target_size)
            return result
        else:
            return None
    except Exception as e:
        print(f"  Error loading {nc_path}: {e}")
    return None


def _resize_era5_array(arr: np.ndarray, target_size: tuple) -> np.ndarray:
    """将ERA5数组调整到目标尺寸

    使用简单的填充或裁剪方式，优先保持数据中心对齐
    Args:
        arr: 输入数组，形状 (C, H, W)
        target_size: 目标尺寸 (target_H, target_W)
    """
    c, h, w = arr.shape
    th, tw = target_size

    # 创建目标数组，用边界值填充
    result = np.zeros((c, th, tw), dtype=arr.dtype)

    # 计算填充/裁剪的起始位置（居中对齐）
    src_h_start = max(0, (h - th) // 2)
    src_w_start = max(0, (w - tw) // 2)
    dst_h_start = max(0, (th - h) // 2)
    dst_w_start = max(0, (tw - w) // 2)

    # 复制的范围
    copy_h = min(h, th)
    copy_w = min(w, tw)

    result[:, dst_h_start:dst_h_start+copy_h, dst_w_start:dst_w_start+copy_w] = \
        arr[:, src_h_start:src_h_start+copy_h, src_w_start:src_w_start+copy_w]

    return result

