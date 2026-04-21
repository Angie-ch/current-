"""
逐台风 z 场爆炸诊断 (轻量版)
从 evaluate_multi.py 缓存的 .npz 文件读取, 无需重新推理

用法:
  1. 先跑一次 evaluate_multi.py (会自动保存缓存)
  2. python diagnose_z_explosion.py --cache eval_results/per_typhoon_rmse.npz
"""
import argparse
import numpy as np

VAR_NAMES = [
    'u_850', 'u_500', 'u_250',
    'v_850', 'v_500', 'v_250',
    'z_850', 'z_500', 'z_250',
]


def main():
    parser = argparse.ArgumentParser(description="z 场爆炸诊断")
    parser.add_argument("--cache", type=str, required=True,
                        help="evaluate_multi.py 保存的 per_typhoon_rmse.npz")
    parser.add_argument("--threshold", type=float, default=500,
                        help="z RMSE 爆炸阈值 (默认 500)")
    args = parser.parse_args()

    data = np.load(args.cache, allow_pickle=True)
    stacked = data["stacked"]       # (N, T, C)
    all_tids = list(data["tids"])    # N typhoon IDs
    N, ar_steps, C = stacked.shape
    explosion_threshold = args.threshold

    print(f"加载缓存: {N} 个台风, {ar_steps} 步, {C} 通道")

    # ---- 1. 逐台风 z_500 排名 ----
    z500_idx = 7
    typhoon_z500_avg = []
    for i, tid in enumerate(all_tids):
        z500_rmse = stacked[i, :, z500_idx]
        valid_z500 = z500_rmse[~np.isnan(z500_rmse)]
        if len(valid_z500) > 0:
            typhoon_z500_avg.append((
                tid, np.mean(valid_z500), np.max(valid_z500),
                np.argmax(valid_z500), len(valid_z500)
            ))

    typhoon_z500_avg.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*90}")
    print(f"  [排名] z_500 平均 RMSE (最差 -> 最好), 爆炸阈值={explosion_threshold}")
    print(f"{'='*90}")
    print(f"  {'排名':>4}  {'台风ID':<20}  {'平均z500':>10}  {'最大z500':>10}  {'最差时效':>8}  {'有效步':>6}  {'状态'}")
    print(f"  {'-'*80}")
    for rank, (tid, avg, mx, worst_t, n) in enumerate(typhoon_z500_avg, 1):
        status = "** 爆炸 **" if mx > explosion_threshold else "正常"
        worst_h = (worst_t + 1) * 3
        print(f"  {rank:>4}  {tid:<20}  {avg:>10.1f}  {mx:>10.1f}  +{worst_h:>3}h    {n:>6}  {status}")

    # ---- 2. 爆炸台风逐时效明细 ----
    exploding = [(tid, i) for i, tid in enumerate(all_tids)
                 if np.nanmax(stacked[i, :, z500_idx]) > explosion_threshold]

    if exploding:
        print(f"\n{'='*90}")
        print(f"  [详细] {len(exploding)} 个爆炸台风逐时效 RMSE")
        print(f"{'='*90}")

        for tid, idx in exploding:
            print(f"\n  >>> {tid} <<<")
            print(f"  {'时效':>6}  {'z_850':>10}  {'z_500':>10}  {'z_250':>10}  "
                  f"{'u_850':>10}  {'v_850':>10}  {'标记'}")
            print(f"  {'-'*75}")

            first_explosion_step = None
            for t in range(ar_steps):
                if np.isnan(stacked[idx, t, 0]):
                    break
                z850 = stacked[idx, t, 6]
                z500 = stacked[idx, t, 7]
                z250 = stacked[idx, t, 8]
                u850 = stacked[idx, t, 0]
                v850 = stacked[idx, t, 3]

                flag = ""
                if z500 > explosion_threshold:
                    flag = "<-- 爆炸!"
                    if first_explosion_step is None:
                        first_explosion_step = t
                elif z500 > explosion_threshold * 0.5:
                    flag = "<-- 警告"

                print(f"  +{(t+1)*3:>3}h  {z850:>10.1f}  {z500:>10.1f}  {z250:>10.1f}  "
                      f"{u850:>10.2f}  {v850:>10.2f}  {flag}")

            if first_explosion_step is not None:
                print(f"  -> 首次爆炸: +{(first_explosion_step+1)*3}h")

        # ---- 3. 爆炸台风之间的共性分析 ----
        print(f"\n{'='*90}")
        print(f"  [共性分析] 爆炸台风首次爆炸时效分布")
        print(f"{'='*90}")
        for tid, idx in exploding:
            z500_series = stacked[idx, :, z500_idx]
            valid = ~np.isnan(z500_series)
            explode_steps = np.where(valid & (z500_series > explosion_threshold))[0]
            if len(explode_steps) > 0:
                first_h = (explode_steps[0] + 1) * 3
                count = len(explode_steps)
                print(f"  {tid:<20}  首次爆炸: +{first_h:>3}h, "
                      f"爆炸步数: {count}/{int(valid.sum())}")

    else:
        print(f"\n  没有爆炸台风 (所有 z_500 最大 RMSE < {explosion_threshold})")

    # ---- 4. 排除爆炸台风后的 Mean RMSE ----
    exploding_indices = {idx for _, idx in exploding} if exploding else set()
    normal_indices = [i for i in range(N) if i not in exploding_indices]

    if normal_indices and exploding:
        normal_mean = np.nanmean(stacked[normal_indices], axis=0)  # (T, C)
        full_mean = np.nanmean(stacked, axis=0)

        print(f"\n{'='*90}")
        print(f"  [对比] 排除 {len(exploding)} 个爆炸台风后 vs 全部 (Mean RMSE)")
        print(f"{'='*90}")
        print(f"  {'':>8}  {'--- z_850 ---':>25}  {'--- z_500 ---':>25}  {'--- z_250 ---':>25}")
        print(f"  {'时效':>6}  {'全部':>10}  {'排除后':>10}  {'全部':>10}  {'排除后':>10}  {'全部':>10}  {'排除后':>10}")
        print(f"  {'-'*80}")

        for t in range(ar_steps):
            n_valid = np.sum(~np.isnan(stacked[normal_indices, t, 0]))
            if n_valid == 0:
                continue
            print(f"  +{(t+1)*3:>3}h", end="")
            for vi in [6, 7, 8]:
                print(f"  {full_mean[t, vi]:>10.1f}  {normal_mean[t, vi]:>10.1f}", end="")
            print()

        print(f"\n  {'平均':>6}", end="")
        for vi in [6, 7, 8]:
            print(f"  {np.nanmean(full_mean[:, vi]):>10.1f}  {np.nanmean(normal_mean[:, vi]):>10.1f}", end="")
        print()

    print(f"\n诊断完成!")


if __name__ == "__main__":
    main()
