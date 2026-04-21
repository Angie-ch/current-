"""
找出 z 场最离谱的台风，输出排除后的 Mean RMSE
支持按年份过滤 (--min_year 1980 排除卫星时代前的低质量 ERA5 数据)

用法:
  python find_worst_typhoons.py --cache eval_results/per_typhoon_rmse.npz --exclude_top 40
  python find_worst_typhoons.py --cache eval_results/per_typhoon_rmse.npz --min_year 1980
  python find_worst_typhoons.py --cache eval_results/per_typhoon_rmse.npz --min_year 1980 --exclude_top 10
"""
import argparse
import numpy as np

VAR_NAMES = ['u_850','u_500','u_250','v_850','v_500','v_250','z_850','z_500','z_250']


def extract_year(tid: str) -> int:
    """从台风 ID 提取年份，如 '1980094N02179' -> 1980"""
    try:
        return int(tid[:4])
    except (ValueError, IndexError):
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--exclude_top", type=int, default=40, help="排除最差的 N 个台风")
    parser.add_argument("--min_year", type=int, default=None,
                        help="只保留 >= 该年份的台风 (如 1980，排除卫星时代前数据)")
    args = parser.parse_args()

    data = np.load(args.cache, allow_pickle=True)
    stacked = data["stacked"]  # (N, T, C)
    tids = list(data["tids"])
    N, T, C = stacked.shape

    # ---- 年份过滤 ----
    if args.min_year is not None:
        keep_mask = [extract_year(tid) >= args.min_year for tid in tids]
        removed_tids = [tid for tid, keep in zip(tids, keep_mask) if not keep]
        tids = [tid for tid, keep in zip(tids, keep_mask) if keep]
        stacked = stacked[keep_mask]
        N = len(tids)

        # 统计被过滤的年份分布
        year_counts = {}
        for tid in removed_tids:
            y = extract_year(tid)
            decade = f"{y // 10 * 10}s"
            year_counts[decade] = year_counts.get(decade, 0) + 1

        print(f"{'='*90}")
        print(f"  [年份过滤] min_year={args.min_year}")
        print(f"  排除 {len(removed_tids)} 个台风，剩余 {N} 个")
        if year_counts:
            dist_str = ', '.join(f"{k}: {v}" for k, v in sorted(year_counts.items()))
            print(f"  排除年代分布: {dist_str}")
        print(f"{'='*90}\n")

    # ---- 全量统计 (过滤后的全集) ----
    full_mean = np.nanmean(stacked, axis=0)
    full_median = np.nanmedian(stacked, axis=0)

    print(f"{'='*90}")
    print(f"  全量 Mean RMSE ({N} typhoons)")
    print(f"{'='*90}")
    print(f"{'时效':>6}", end="")
    for vn in VAR_NAMES[:C]:
        print(f"  {vn:>8}", end="")
    print()
    print(f"  {'-'*(6 + 10*C)}")

    for t in range(T):
        n_valid = np.sum(~np.isnan(stacked[:, t, 0]))
        if n_valid == 0:
            continue
        print(f"  +{(t+1)*3:>3}h", end="")
        for v in range(C):
            print(f"  {full_mean[t, v]:>8.2f}", end="")
        print()

    print(f"\n  {'平均':>4}", end="")
    for v in range(C):
        print(f"  {np.nanmean(full_mean[:, v]):>8.2f}", end="")
    print()

    # ---- Mean vs Median 差距 (全量) ----
    print(f"\n{'='*90}")
    print(f"  [全量] Mean vs Median 差距")
    print(f"{'='*90}")
    print(f"  {'':>20}  {'Mean':>10}  {'Median':>10}  {'比值':>8}")
    for vi, vn in [(6,'z_850'), (7,'z_500'), (8,'z_250')]:
        m = np.nanmean(full_mean[:, vi])
        md = np.nanmean(full_median[:, vi])
        ratio = m / md if md > 0 else float('inf')
        print(f"  {vn:>20}  {m:>10.1f}  {md:>10.1f}  {ratio:>8.2f}x")

    # ---- 按 z_500 排序找最差的 ----
    if args.exclude_top > 0:
        z500_max = []
        for i, tid in enumerate(tids):
            z500 = stacked[i, :, 7]
            valid = z500[~np.isnan(z500)]
            if len(valid) > 0:
                z500_max.append((tid, i, np.max(valid), np.mean(valid), np.median(valid)))
            else:
                z500_max.append((tid, i, 0, 0, 0))

        z500_max.sort(key=lambda x: x[2], reverse=True)

        n_exclude = min(args.exclude_top, N)
        print(f"\n{'='*90}")
        print(f"  z_500 最差的 {n_exclude} 个台风 (按最大 RMSE 排序)")
        print(f"{'='*90}")
        print(f"  {'排名':>4}  {'台风ID':<22}  {'年份':>4}  {'最大z500':>10}  {'平均z500':>10}  {'中位z500':>10}")
        print(f"  {'-'*80}")

        exclude_ids = set()
        exclude_indices = set()
        for rank, (tid, idx, mx, avg, med) in enumerate(z500_max[:n_exclude], 1):
            year = extract_year(tid)
            print(f"  {rank:>4}  {tid:<22}  {year:>4}  {mx:>10.1f}  {avg:>10.1f}  {med:>10.1f}")
            exclude_ids.add(tid)
            exclude_indices.add(idx)

        # 排除后的统计
        keep_indices = [i for i in range(N) if i not in exclude_indices]
        n_keep = len(keep_indices)
        keep_stacked = stacked[keep_indices]
        keep_mean = np.nanmean(keep_stacked, axis=0)
        keep_median = np.nanmedian(keep_stacked, axis=0)

        print(f"\n{'='*90}")
        print(f"  排除 {n_exclude} 个后 Mean RMSE ({n_keep} typhoons)")
        print(f"{'='*90}")
        print(f"{'时效':>6}", end="")
        for vn in VAR_NAMES[:C]:
            print(f"  {vn:>8}", end="")
        print()
        print(f"  {'-'*(6 + 10*C)}")

        for t in range(T):
            n_valid = np.sum(~np.isnan(keep_stacked[:, t, 0]))
            if n_valid == 0:
                continue
            print(f"  +{(t+1)*3:>3}h", end="")
            for v in range(C):
                print(f"  {keep_mean[t, v]:>8.2f}", end="")
            print()

        print(f"\n  {'平均':>4}", end="")
        for v in range(C):
            print(f"  {np.nanmean(keep_mean[:, v]):>8.2f}", end="")
        print()

        # 对比
        print(f"\n{'='*90}")
        print(f"  [对比] z 通道 Mean RMSE 平均")
        print(f"{'='*90}")
        print(f"  {'':>20}  {'全量({N})':>12}  {'排除后({n_keep})':>12}  {'降幅':>8}")
        for vi, vn in [(6,'z_850'), (7,'z_500'), (8,'z_250')]:
            full = np.nanmean(full_mean[:, vi])
            keep = np.nanmean(keep_mean[:, vi])
            pct = (full - keep) / full * 100
            print(f"  {vn:>20}  {full:>12.1f}  {keep:>12.1f}  {pct:>7.1f}%")

        # Mean vs Median 差距 (排除后)
        print(f"\n{'='*90}")
        print(f"  [排除后] Mean vs Median 差距")
        print(f"{'='*90}")
        print(f"  {'':>20}  {'Mean':>10}  {'Median':>10}  {'比值':>8}")
        for vi, vn in [(6,'z_850'), (7,'z_500'), (8,'z_250')]:
            m = np.nanmean(keep_mean[:, vi])
            md = np.nanmean(keep_median[:, vi])
            ratio = m / md if md > 0 else float('inf')
            print(f"  {vn:>20}  {m:>10.1f}  {md:>10.1f}  {ratio:>8.2f}x")

        # 保存排除列表
        with open("excluded_typhoons.txt", "w") as f:
            for tid, idx, mx, avg, med in z500_max[:n_exclude]:
                f.write(f"{tid}\n")
        print(f"\n排除列表已保存: excluded_typhoons.txt ({n_exclude} 个台风)")
        print(f"可在评估时使用此列表过滤数据集")


if __name__ == "__main__":
    main()
