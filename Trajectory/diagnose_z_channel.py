"""
诊断 preprocessed_era5 中 z 通道的数据质量
扫描所有 {storm_id}.npy, 检测 z_850/z_500/z_250 通道的异常

用法: python diagnose_z_channel.py --data_dir preprocessed_era5 --exclude_top 40
"""
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# 通道顺序: u_850,u_500,u_250, v_850,v_500,v_250, z_850,z_500,z_250
VAR_NAMES = ['u_850','u_500','u_250','v_850','v_500','v_250','z_850','z_500','z_250']
Z_INDICES = [6, 7, 8]  # z_850, z_500, z_250
Z_NAMES = ['z_850', 'z_500', 'z_250']

# z 物理合理范围 (单位: m²/s², 即 J/kg)
# 850hPa: ~1200-1600m 位势高度 -> z ≈ 11772-15696
# 500hPa: ~5000-5900m -> z ≈ 49050-57879
# 250hPa: ~9800-11000m -> z ≈ 96138-107910
Z_REASONABLE_RANGES = {
    6: (8000, 20000),    # z_850
    7: (40000, 65000),   # z_500
    8: (80000, 120000),  # z_250
}


def analyze_storm(npy_path: Path) -> dict:
    """分析单个台风的 z 通道质量"""
    data = np.load(npy_path)  # (T, C, H, W)
    T, C, H, W = data.shape
    storm_id = npy_path.stem

    result = {
        'storm_id': storm_id,
        'T': T,
        'C': C,
        'issues': [],
        'z_stats': {},
        'severity': 0.0,  # 严重程度评分 (越高越差)
    }

    if C < 9:
        result['issues'].append(f'通道数不足: {C} < 9')
        result['severity'] = 100.0
        return result

    for z_idx, z_name in zip(Z_INDICES, Z_NAMES):
        z_data = data[:, z_idx, :, :]  # (T, H, W)
        lo, hi = Z_REASONABLE_RANGES[z_idx]

        stats = {
            'mean': float(np.nanmean(z_data)),
            'std': float(np.nanstd(z_data)),
            'min': float(np.nanmin(z_data)),
            'max': float(np.nanmax(z_data)),
            'nan_ratio': float(np.isnan(z_data).sum() / z_data.size),
            'zero_ratio': float((z_data == 0).sum() / z_data.size),
        }

        # 逐时间步检查方差
        per_step_std = np.array([z_data[t].std() for t in range(T)])
        stats['min_step_std'] = float(per_step_std.min())
        stats['max_step_std'] = float(per_step_std.max())
        stats['mean_step_std'] = float(per_step_std.mean())

        # 逐时间步检查均值
        per_step_mean = np.array([z_data[t].mean() for t in range(T)])
        stats['mean_range'] = float(per_step_mean.max() - per_step_mean.min())

        # 超出合理范围的比例
        out_of_range = ((z_data < lo) | (z_data > hi))
        stats['out_of_range_ratio'] = float(out_of_range.sum() / z_data.size)

        result['z_stats'][z_name] = stats

        # 问题检测
        severity_add = 0.0

        # 1. 全零
        if stats['zero_ratio'] > 0.5:
            result['issues'].append(f'{z_name}: {stats["zero_ratio"]*100:.1f}% 为零')
            severity_add += 50

        # 2. NaN
        if stats['nan_ratio'] > 0.01:
            result['issues'].append(f'{z_name}: {stats["nan_ratio"]*100:.1f}% NaN')
            severity_add += 40

        # 3. 均值超出合理范围
        if stats['mean'] < lo * 0.5 or stats['mean'] > hi * 1.5:
            result['issues'].append(
                f'{z_name}: 均值 {stats["mean"]:.0f} 超出合理范围 [{lo},{hi}]'
            )
            severity_add += 30

        # 4. 空间方差过小 (可能是常数填充)
        if stats['mean_step_std'] < 10:
            result['issues'].append(
                f'{z_name}: 空间方差极小 (std={stats["mean_step_std"]:.2f})，可能是常数填充'
            )
            severity_add += 25

        # 5. 超出合理范围的像素比例
        if stats['out_of_range_ratio'] > 0.3:
            result['issues'].append(
                f'{z_name}: {stats["out_of_range_ratio"]*100:.1f}% 像素超出合理范围'
            )
            severity_add += 20

        # 6. 时间步间均值跳变过大
        if stats['mean_range'] > (hi - lo) * 0.5:
            result['issues'].append(
                f'{z_name}: 时间步均值跳变 {stats["mean_range"]:.0f}，可能有突变帧'
            )
            severity_add += 15

        result['severity'] += severity_add

    return result


def main():
    parser = argparse.ArgumentParser(description='诊断 preprocessed_era5 中 z 通道质量')
    parser.add_argument('--data_dir', type=str, default='preprocessed_era5',
                        help='预处理数据目录')
    parser.add_argument('--exclude_top', type=int, default=40,
                        help='输出最差的 N 个台风到排除列表')
    parser.add_argument('--output', type=str, default='excluded_typhoons.txt',
                        help='排除列表输出文件名')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 目录不存在 {data_dir}")
        return

    # 扫描所有 npy 文件 (排除 _times.npy)
    npy_files = sorted([
        f for f in data_dir.glob('*.npy')
        if not f.name.endswith('_times.npy')
    ])
    print(f"{'='*90}")
    print(f"  Z 通道数据质量诊断")
    print(f"  数据目录: {data_dir.absolute()}")
    print(f"  台风数量: {len(npy_files)}")
    print(f"{'='*90}\n")

    results = []
    for i, npy_path in enumerate(npy_files):
        result = analyze_storm(npy_path)
        results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  已扫描 {i+1}/{len(npy_files)} ...")

    # ---- 统计汇总 ----
    n_total = len(results)
    n_has_issues = sum(1 for r in results if r['issues'])
    n_clean = n_total - n_has_issues

    print(f"\n{'='*90}")
    print(f"  总体统计")
    print(f"{'='*90}")
    print(f"  总台风数:     {n_total}")
    print(f"  z 通道正常:   {n_clean}  ({n_clean/n_total*100:.1f}%)")
    print(f"  z 通道异常:   {n_has_issues}  ({n_has_issues/n_total*100:.1f}%)")

    # ---- 问题分类统计 ----
    issue_types = defaultdict(int)
    for r in results:
        for iss in r['issues']:
            # 提取问题类型关键词
            if '为零' in iss:
                issue_types['全零/半零'] += 1
            elif 'NaN' in iss:
                issue_types['含 NaN'] += 1
            elif '超出合理范围' in iss and '像素' in iss:
                issue_types['像素越界'] += 1
            elif '超出合理范围' in iss:
                issue_types['均值越界'] += 1
            elif '方差极小' in iss:
                issue_types['方差过小(常数填充)'] += 1
            elif '跳变' in iss:
                issue_types['时间步突变'] += 1

    if issue_types:
        print(f"\n  问题分类:")
        for itype, cnt in sorted(issue_types.items(), key=lambda x: -x[1]):
            print(f"    {itype:20s}: {cnt:4d} 个台风")

    # ---- z 通道分布概览 (仅正常台风) ----
    clean_results = [r for r in results if not r['issues']]
    if clean_results:
        print(f"\n{'='*90}")
        print(f"  正常台风 z 通道统计 ({len(clean_results)} 个)")
        print(f"{'='*90}")
        print(f"  {'通道':>8}  {'均值 mean':>12}  {'均值 std':>10}  "
              f"{'空间std mean':>12}  {'range':>12}")
        print(f"  {'-'*70}")
        for z_name in Z_NAMES:
            means = [r['z_stats'][z_name]['mean'] for r in clean_results]
            stds = [r['z_stats'][z_name]['mean_step_std'] for r in clean_results]
            print(f"  {z_name:>8}  {np.mean(means):>12.1f}  {np.std(means):>10.1f}  "
                  f"{np.mean(stds):>12.2f}  "
                  f"[{np.min(means):>8.0f}, {np.max(means):>8.0f}]")

    # ---- 异常台风 z 通道统计 ----
    bad_results = [r for r in results if r['issues']]
    if bad_results:
        print(f"\n{'='*90}")
        print(f"  异常台风 z 通道统计 ({len(bad_results)} 个)")
        print(f"{'='*90}")
        print(f"  {'通道':>8}  {'均值 mean':>12}  {'均值 std':>10}  "
              f"{'空间std mean':>12}  {'range':>12}")
        print(f"  {'-'*70}")
        for z_name in Z_NAMES:
            vals = [r['z_stats'][z_name]['mean'] for r in bad_results
                    if z_name in r['z_stats']]
            stds = [r['z_stats'][z_name]['mean_step_std'] for r in bad_results
                    if z_name in r['z_stats']]
            if vals:
                print(f"  {z_name:>8}  {np.mean(vals):>12.1f}  {np.std(vals):>10.1f}  "
                      f"{np.mean(stds):>12.2f}  "
                      f"[{np.min(vals):>8.0f}, {np.max(vals):>8.0f}]")

    # ---- 按严重程度排序，列出最差的 N 个 ----
    results_sorted = sorted(results, key=lambda r: r['severity'], reverse=True)

    n_show = args.exclude_top
    print(f"\n{'='*90}")
    print(f"  最差的 {n_show} 个台风 (按严重程度排序)")
    print(f"{'='*90}")
    print(f"  {'排名':>4}  {'台风ID':<22}  {'T步':>4}  {'严重度':>6}  {'问题'}")
    print(f"  {'-'*85}")

    exclude_ids = []
    for rank, r in enumerate(results_sorted[:n_show], 1):
        issues_str = '; '.join(r['issues'][:3]) if r['issues'] else '(无明显问题)'
        if len(r['issues']) > 3:
            issues_str += f' ... (+{len(r["issues"])-3})'
        print(f"  {rank:>4}  {r['storm_id']:<22}  {r['T']:>4}  {r['severity']:>6.1f}  {issues_str}")
        exclude_ids.append(r['storm_id'])

    # ---- 保存排除列表 ----
    with open(args.output, 'w') as f:
        for sid in exclude_ids:
            f.write(f"{sid}\n")
    print(f"\n排除列表已保存: {args.output} ({len(exclude_ids)} 个台风)")

    # ---- 对比: 排除后 z 通道变化 ----
    exclude_set = set(exclude_ids)
    kept = [r for r in results if r['storm_id'] not in exclude_set]
    kept_clean = [r for r in kept if not r['issues']]

    print(f"\n{'='*90}")
    print(f"  排除 {n_show} 个后的统计")
    print(f"{'='*90}")
    print(f"  剩余台风: {len(kept)} ({len(kept_clean)} 个完全正常)")
    if kept:
        print(f"\n  {'通道':>8}  {'排除前 mean':>12}  {'排除后 mean':>12}  {'降幅':>8}")
        print(f"  {'-'*50}")
        for z_name in Z_NAMES:
            all_means = [r['z_stats'][z_name]['mean'] for r in results
                         if z_name in r['z_stats']]
            kept_means = [r['z_stats'][z_name]['mean'] for r in kept
                          if z_name in r['z_stats']]
            if all_means and kept_means:
                before_std = np.std(all_means)
                after_std = np.std(kept_means)
                pct = (before_std - after_std) / (before_std + 1e-8) * 100
                print(f"  {z_name:>8}  std={before_std:>8.1f}    std={after_std:>8.1f}    {pct:>6.1f}%")

    print(f"\n完成! 可在评估时使用 --exclude_file {args.output}")


if __name__ == '__main__':
    main()
