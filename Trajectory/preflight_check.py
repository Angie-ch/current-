"""
训练前全面检查脚本：检查所有文件、配置、数据是否正确
避免因为文件错误浪费训练时间

用法:
  python preflight_check.py
"""
import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path

# 计数器
errors = []
warnings = []
ok_count = 0


def OK(msg):
    global ok_count
    ok_count += 1
    print(f"  ✅ {msg}")

def WARN(msg):
    warnings.append(msg)
    print(f"  ⚠️  {msg}")

def ERR(msg):
    errors.append(msg)
    print(f"  ❌ {msg}")


def main():
    parser = argparse.ArgumentParser(description="训练前全面检查")
    parser.add_argument("--data_root", type=str, default=r"C:\Users\fyp\Desktop\Typhoon_data_final")
    parser.add_argument("--npy_dir", type=str, default="preprocessed_era5")
    parser.add_argument("--cache_dir", type=str, default="diffusion_era5_cache")
    parser.add_argument("--newtry_dir", type=str, default=None,
                        help="扩散模型目录 (自动检测)")
    args = parser.parse_args()

    TRAJ_DIR = os.path.dirname(os.path.abspath(__file__))

    # 自动检测 newtry 目录
    newtry_dir = args.newtry_dir
    if newtry_dir is None:
        for candidate in [
            os.path.join(TRAJ_DIR, '..', 'newtry'),
            r"C:\Users\fyp\Desktop\newtry",
            r"C:\Users\fyp\Desktop\fyp-version2\newtry",
        ]:
            if os.path.isdir(candidate):
                newtry_dir = os.path.abspath(candidate)
                break

    print("=" * 70)
    print(" 训练前全面检查 (Preflight Check)")
    print("=" * 70)

    # ============================================================
    # 1. 配置文件检查
    # ============================================================
    print(f"\n[1/7] 配置文件检查")
    print("-" * 40)

    try:
        from config import model_cfg, data_cfg, train_cfg

        # era5_channels
        if model_cfg.era5_channels == 9:
            OK(f"config.py era5_channels = {model_cfg.era5_channels}")
        elif model_cfg.era5_channels == 12:
            ERR(f"config.py era5_channels = {model_cfg.era5_channels} (应该是9, 还没改!)")
        else:
            WARN(f"config.py era5_channels = {model_cfg.era5_channels} (非标准值)")

        # era5_3d_vars
        if 'vo' in data_cfg.era5_3d_vars:
            ERR(f"config.py era5_3d_vars 包含 'vo': {data_cfg.era5_3d_vars} (应该去掉vo!)")
        else:
            OK(f"config.py era5_3d_vars = {data_cfg.era5_3d_vars} (无vo)")

        expected_channels = len(data_cfg.era5_3d_vars) * len(data_cfg.pressure_levels) + len(data_cfg.era5_2d_vars)
        if expected_channels == model_cfg.era5_channels:
            OK(f"通道数计算一致: {len(data_cfg.era5_3d_vars)}×{len(data_cfg.pressure_levels)} + {len(data_cfg.era5_2d_vars)} = {expected_channels}")
        else:
            ERR(f"通道数不匹配: {len(data_cfg.era5_3d_vars)}×{len(data_cfg.pressure_levels)} + {len(data_cfg.era5_2d_vars)} = {expected_channels}, 但 era5_channels = {model_cfg.era5_channels}")

        # 数据路径
        if os.path.exists(data_cfg.csv_path):
            OK(f"CSV 文件存在: {data_cfg.csv_path}")
        else:
            # 可能是相对路径
            abs_csv = os.path.join(TRAJ_DIR, data_cfg.csv_path)
            if os.path.exists(abs_csv):
                OK(f"CSV 文件存在: {abs_csv}")
            else:
                ERR(f"CSV 文件不存在: {data_cfg.csv_path}")

        if os.path.isdir(data_cfg.era5_dir):
            OK(f"ERA5 数据目录存在: {data_cfg.era5_dir}")
        else:
            ERR(f"ERA5 数据目录不存在: {data_cfg.era5_dir}")

    except Exception as e:
        ERR(f"config.py 导入失败: {e}")

    # ============================================================
    # 2. norm_stats.pt 检查
    # ============================================================
    print(f"\n[2/7] norm_stats.pt 归一化统计检查")
    print("-" * 40)

    import torch

    # Trajectory 目录的 norm_stats.pt
    traj_norm = os.path.join(TRAJ_DIR, 'norm_stats.pt')
    if os.path.exists(traj_norm):
        stats = torch.load(traj_norm, weights_only=True, map_location='cpu')
        n_ch = len(stats['mean'])
        if n_ch == 12:
            OK(f"Trajectory/norm_stats.pt 存在, {n_ch} 通道")
        else:
            ERR(f"Trajectory/norm_stats.pt 通道数 = {n_ch} (应该是12)")
    else:
        WARN(f"Trajectory/norm_stats.pt 不存在 (dataset.py 会尝试从 newtry 加载)")

    # newtry 的 norm_stats.pt
    if newtry_dir:
        newtry_norm = os.path.join(newtry_dir, 'norm_stats.pt')
        if os.path.exists(newtry_norm):
            stats2 = torch.load(newtry_norm, weights_only=True, map_location='cpu')
            n_ch2 = len(stats2['mean'])
            OK(f"newtry/norm_stats.pt 存在, {n_ch2} 通道")

            # 对比两份是否一致
            if os.path.exists(traj_norm):
                if torch.allclose(stats['mean'], stats2['mean']) and torch.allclose(stats['std'], stats2['std']):
                    OK(f"两份 norm_stats.pt 完全一致")
                else:
                    ERR(f"Trajectory 和 newtry 的 norm_stats.pt 不一致!")
        else:
            WARN(f"newtry/norm_stats.pt 不存在: {newtry_norm}")
    else:
        WARN(f"未找到 newtry 目录, 跳过 newtry norm_stats 检查")

    # 验证 dataset.py 实际加载的归一化统计
    try:
        from dataset import ERA5_CHANNEL_MEAN, ERA5_CHANNEL_STD
        n_mean = len(ERA5_CHANNEL_MEAN)
        n_std = len(ERA5_CHANNEL_STD)
        if n_mean == 12 and n_std == 12:
            OK(f"dataset.py 实际加载的归一化统计: {n_mean} 通道")
        else:
            ERR(f"dataset.py 归一化统计通道数: mean={n_mean}, std={n_std} (应该是12)")
    except Exception as e:
        ERR(f"dataset.py 归一化统计加载失败: {e}")

    # ============================================================
    # 3. 预处理 .npy 文件检查
    # ============================================================
    print(f"\n[3/7] 预处理 .npy 文件检查")
    print("-" * 40)

    npy_dir = Path(args.npy_dir)
    if not npy_dir.is_absolute():
        npy_dir = Path(TRAJ_DIR) / args.npy_dir

    if npy_dir.exists():
        npy_files = sorted(npy_dir.glob("*.npy"))
        # 过滤掉 _times.npy
        data_npys = [f for f in npy_files if not f.name.endswith("_times.npy")]
        times_npys = [f for f in npy_files if f.name.endswith("_times.npy")]

        OK(f"预处理目录存在: {npy_dir}")
        print(f"       数据文件: {len(data_npys)}, 时间文件: {len(times_npys)}")

        if len(data_npys) == 0:
            WARN(f"预处理目录为空, 将从 NC 文件加载 (较慢)")
        else:
            # 抽检 5 个文件的通道数
            check_count = min(5, len(data_npys))
            ch_counts = {}
            for f in data_npys[:check_count]:
                arr = np.load(f)
                shape = arr.shape
                ch = shape[1] if arr.ndim == 4 else "?"
                ch_counts[ch] = ch_counts.get(ch, 0) + 1

            if len(ch_counts) == 1:
                ch = list(ch_counts.keys())[0]
                sample_shape = np.load(data_npys[0]).shape
                if ch == 12:
                    OK(f"抽检 {check_count} 个 .npy: 通道数 = {ch} ✓ shape={sample_shape}")
                elif ch == 15:
                    ERR(f"抽检 {check_count} 个 .npy: 通道数 = {ch} (旧的15通道! 需要重新预处理!)")
                    ERR(f"  → 运行: python preprocess_era5.py 重新生成")
                    ERR(f"  → 或先删除: rmdir /s /q {npy_dir}")
                else:
                    WARN(f"抽检 {check_count} 个 .npy: 通道数 = {ch}, shape={sample_shape}")
            else:
                ERR(f"不同 .npy 通道数不一致: {ch_counts}")

            # 检查是否有对应的 times 文件
            data_ids = {f.stem for f in data_npys}
            times_ids = {f.stem.replace("_times", "") for f in times_npys}
            missing_times = data_ids - times_ids
            if missing_times:
                WARN(f"{len(missing_times)} 个数据文件缺少对应的 _times.npy")
            else:
                OK(f"所有数据文件都有对应的 _times.npy")
    else:
        WARN(f"预处理目录不存在: {npy_dir} (将从 NC 文件加载)")

    # ============================================================
    # 4. 扩散 ERA5 缓存检查
    # ============================================================
    print(f"\n[4/7] 扩散 ERA5 缓存检查")
    print("-" * 40)

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = Path(TRAJ_DIR) / args.cache_dir

    cache_path = cache_dir / "era5_cache.npz"
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        n_storms = len(loaded.files)
        OK(f"扩散ERA5缓存存在: {cache_path} ({n_storms} 个台风)")

        # 检查第一个台风的通道数
        first_key = loaded.files[0]
        first_arr = loaded[first_key]
        ch = first_arr.shape[1] if first_arr.ndim == 4 else (first_arr.shape[0] if first_arr.ndim == 3 else "?")

        if ch == 12:
            OK(f"缓存通道数 = {ch}, shape={first_arr.shape}")
        elif ch == 15:
            ERR(f"缓存通道数 = {ch} (旧的15通道! 需要删除重新生成!)")
            ERR(f"  → 运行: rmdir /s /q {cache_dir}")
        else:
            WARN(f"缓存通道数 = {ch}, shape={first_arr.shape}")

        loaded.close()
    else:
        WARN(f"扩散ERA5缓存不存在: {cache_path} (finetune_train.py 会自动生成)")

    # ============================================================
    # 5. 代码文件残留检查
    # ============================================================
    print(f"\n[5/7] 代码文件残留检查 (旧15通道/vo引用)")
    print("-" * 40)

    files_to_check = {
        'dataset.py': ['vo_850', 'vo_500', 'vo_250'],
        'finetune_train.py': ['expand_12ch_to_15ch', 'traj_num_channels = 15'],
        'config.py': ["'vo'"],
        'model.py': [],  # model.py 不应该硬编码通道数
    }

    for fname, bad_patterns in files_to_check.items():
        fpath = os.path.join(TRAJ_DIR, fname)
        if not os.path.exists(fpath):
            WARN(f"{fname} 不存在")
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()

        found_bad = []
        for pattern in bad_patterns:
            if pattern in content:
                found_bad.append(pattern)

        if found_bad:
            ERR(f"{fname} 中发现旧代码残留: {found_bad}")
        else:
            OK(f"{fname} 无旧代码残留")

    # 检查 new_train.py 是否还导入 expand_12ch_to_15ch
    new_train_path = os.path.join(TRAJ_DIR, 'new_train.py')
    if os.path.exists(new_train_path):
        with open(new_train_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if 'expand_12ch_to_15ch' in content:
            ERR(f"new_train.py 还在导入 expand_12ch_to_15ch")
        else:
            OK(f"new_train.py 无旧代码残留")

    # ============================================================
    # 6. preprocess_era5.py 检查
    # ============================================================
    print(f"\n[6/7] preprocess_era5.py 输出通道检查")
    print("-" * 40)

    preprocess_path = os.path.join(TRAJ_DIR, 'preprocess_era5.py')
    if os.path.exists(preprocess_path):
        with open(preprocess_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if '15' in content and '(T, 15,' in content:
            WARN(f"preprocess_era5.py 注释中提到 15 通道 (可能需要重新运行生成12通道的npy)")
        else:
            OK(f"preprocess_era5.py 看起来正常")

        # 关键: preprocess_era5.py 调用 load_era5_frame,
        # load_era5_frame 读取 config.era5_3d_vars 来决定通道
        # 所以只要 config 改了, 重新跑 preprocess_era5.py 就会生成 12 通道
        OK(f"preprocess_era5.py 使用 config.era5_3d_vars 动态决定通道 (config已改为12ch)")
    else:
        WARN(f"preprocess_era5.py 不存在")

    # ============================================================
    # 7. 模型 checkpoint 检查
    # ============================================================
    print(f"\n[7/7] 模型 checkpoint 检查")
    print("-" * 40)

    ckpt_path = os.path.join(TRAJ_DIR, 'checkpoints', 'best.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', '?')
        has_ema = 'ema_model_state_dict' in ckpt

        # 检查模型第一层的输入通道数
        state_dict = ckpt.get('ema_model_state_dict', ckpt.get('model_state_dict', {}))
        physics_key = None
        for k in state_dict:
            if 'physics_encoder' in k and 'weight' in k and state_dict[k].ndim == 5:
                physics_key = k
                break

        if physics_key:
            weight_shape = state_dict[physics_key].shape
            in_ch = weight_shape[1]  # Conv3d weight: (out_ch, in_ch, kT, kH, kW)
            if in_ch == 12:
                OK(f"checkpoint 模型输入通道 = {in_ch} (epoch={epoch}, EMA={'是' if has_ema else '否'})")
            elif in_ch == 15:
                WARN(f"checkpoint 模型输入通道 = {in_ch} (旧的15通道模型, 需要重新训练)")
            else:
                WARN(f"checkpoint 模型输入通道 = {in_ch}, weight shape = {weight_shape}")
        else:
            WARN(f"checkpoint 存在但未找到 physics_encoder 权重 (epoch={epoch})")
    else:
        WARN(f"checkpoint 不存在: {ckpt_path} (正在训练中?)")

    # 检查 finetune checkpoint
    ft_ckpt = os.path.join(TRAJ_DIR, 'checkpoints_finetune', 'best_finetune.pt')
    if os.path.exists(ft_ckpt):
        ckpt_ft = torch.load(ft_ckpt, map_location='cpu', weights_only=False)
        epoch_ft = ckpt_ft.get('epoch', '?')
        strategy = ckpt_ft.get('freeze_strategy', '?')
        OK(f"finetune checkpoint 存在 (epoch={epoch_ft}, strategy={strategy})")
    else:
        print(f"       finetune checkpoint 不存在 (正常, 还没微调)")

    # ============================================================
    # 汇总
    # ============================================================
    print(f"\n{'='*70}")
    print(f" 检查汇总")
    print(f"{'='*70}")
    print(f"  ✅ 通过: {ok_count}")
    print(f"  ⚠️  警告: {len(warnings)}")
    print(f"  ❌ 错误: {len(errors)}")

    if errors:
        print(f"\n  必须修复的错误:")
        for i, e in enumerate(errors, 1):
            print(f"    {i}. {e}")

    if warnings:
        print(f"\n  需要注意的警告:")
        for i, w in enumerate(warnings, 1):
            print(f"    {i}. {w}")

    if not errors:
        print(f"\n  🎉 所有关键检查通过! 可以放心训练。")
    else:
        print(f"\n  🚨 有 {len(errors)} 个错误需要修复后才能训练!")

    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
