"""
诊断脚本: 逐变量分析推理结果，找出 RMSE 的真正分布
自动从推理结果推断变量数和时间步数
"""
import torch
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/newtry/outputs")
    parser.add_argument("--sample_id", type=int, default=0)
    # 支持自定义变量配置
    parser.add_argument("--pl_vars", nargs="+", default=["u", "v"])
    parser.add_argument("--sfc_vars", nargs="+", default=["u10m", "v10m"])
    parser.add_argument("--levels", nargs="+", type=int, default=[850, 500])
    args = parser.parse_args()

    # 构建变量名
    var_names = []
    for var in args.pl_vars:
        for lev in args.levels:
            var_names.append(f"{var}_{lev}")
    for var in args.sfc_vars:
        var_names.append(var)
    n_vars = len(var_names)

    # 加载推理结果
    path = os.path.join(args.output_dir, f"single_pred_{args.sample_id}.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)

    pred = data["prediction"]      # (1, C*T, 40, 40)
    gt = data["ground_truth"]      # (1, C*T, 40, 40)
    tid = data["typhoon_id"]

    total_channels = pred.shape[1]
    n_steps = total_channels // n_vars

    print(f"台风 ID: {tid}")
    print(f"预测 shape: {pred.shape}, 真值 shape: {gt.shape}")
    print(f"变量数: {n_vars}, 预测步数: {n_steps}")
    print()

    print(f"{'变量':<12} {'真值均值':>12} {'真值std':>12} {'RMSE':>10} {'相对误差%':>10}")
    print("=" * 58)

    total_rmse = []
    for t_step in range(n_steps):
        print(f"\n--- 预测第 {t_step+1} 步 (+{(t_step+1)*3}h) ---")
        for v_idx in range(n_vars):
            ch = t_step * n_vars + v_idx
            p = pred[0, ch]   # (40, 40)
            g = gt[0, ch]     # (40, 40)

            gt_mean = g.mean().item()
            gt_std = g.std().item()
            rmse = torch.sqrt(((p - g) ** 2).mean()).item()
            rel_err = (rmse / (abs(gt_mean) + 1e-8)) * 100

            total_rmse.append(rmse)
            print(f"{var_names[v_idx]:<12} {gt_mean:>12.2f} {gt_std:>12.2f} {rmse:>10.2f} {rel_err:>9.2f}%")

    print(f"\n所有通道 RMSE 均值: {np.mean(total_rmse):.2f}")
    print(f"去掉 z 变量后 RMSE 均值: {np.mean([r for i, r in enumerate(total_rmse) if not var_names[i % n_vars].startswith('z_')]):.2f}")


if __name__ == "__main__":
    main()
