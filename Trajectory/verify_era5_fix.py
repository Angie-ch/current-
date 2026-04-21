"""
快速验证 ERA5 修复是否生效
"""
import numpy as np
from pathlib import Path
from config import data_cfg
from data_processing import parse_nc_filename_time, load_era5_frame, load_tyc_storms
from dataset import LT3PDataset, split_storms_by_id, filter_short_storms
from config import model_cfg
import glob


def main():
    print("=" * 60)
    print("ERA5 修复验证")
    print("=" * 60)

    era5_dir = data_cfg.era5_dir
    test_folder = Path(era5_dir) / "1950126N09151"

    # 1. 验证 glob 模式
    nc_files = sorted(glob.glob(str(test_folder / "era5_merged_*.nc")))
    print(f"\n[1] glob 匹配: {len(nc_files)} 个 NC 文件")

    # 2. 验证时间解析
    if nc_files:
        for f in nc_files[:3]:
            t = parse_nc_filename_time(f)
            print(f"    {Path(f).name} -> {t}")

    # 3. 验证 NC 文件内容和加载
    if nc_files:
        print(f"\n[2] 加载第一个 NC 文件...")
        import xarray as xr
        ds = xr.open_dataset(nc_files[0])
        print(f"    变量: {list(ds.data_vars)}")
        print(f"    维度: {dict(ds.dims)}")
        for var in data_cfg.era5_3d_vars + data_cfg.era5_2d_vars:
            status = "✅" if var in ds else "❌"
            print(f"    {status} 变量 '{var}' {'存在' if var in ds else '不存在'}")
        ds.close()

        frame = load_era5_frame(nc_files[0])
        if frame is not None:
            print(f"\n    load_era5_frame: shape={frame.shape}, "
                  f"范围=[{frame.min():.3f}, {frame.max():.3f}], "
                  f"全零={np.abs(frame).sum() == 0}")
        else:
            print(f"    ❌ load_era5_frame 返回 None!")

    # 4. 完整加载测试
    print(f"\n[3] 完整加载（仅前 5 个台风验证）...")
    storm_samples = load_tyc_storms(
        csv_path=data_cfg.csv_path,
        era5_base_dir=era5_dir,
        storm_ids=["1950126N09151", "1950174N19127", "1950195N17139",
                    "1950196N21144", "1950206N22143"]
    )

    for s in storm_samples:
        has_era5 = s.era5_array is not None
        era5_info = ""
        if has_era5:
            era5_info = (f"shape={s.era5_array.shape}, "
                        f"范围=[{s.era5_array.min():.3f}, {s.era5_array.max():.3f}]")
        else:
            era5_info = "None"
        print(f"    {s.storm_id}: len={len(s)}, era5={era5_info}")

    # 5. Dataset 层面检查
    if storm_samples:
        print(f"\n[4] Dataset 层面...")
        ds = LT3PDataset(storm_samples[:2], stride=24)
        if len(ds) > 0:
            batch = ds[0]
            era5 = batch['future_era5']
            print(f"    future_era5: shape={era5.shape}, "
                  f"范围=[{era5.min():.3f}, {era5.max():.3f}], "
                  f"全零={era5.abs().sum() == 0}")

    print(f"\n{'=' * 60}")
    print("验证完成")


if __name__ == '__main__':
    main()
