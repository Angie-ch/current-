#!/usr/bin/env python3
"""
测试适配器 — 验证能否加载 newtry 的 best_eps.pt
"""
import sys
import torch
from pathlib import Path

# 添加父目录到路径，以便使用绝对导入
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

from compare_fm_dm.configs.config import DataConfig, ModelConfig, TrainConfig, get_config
from compare_fm_dm.models.adapter import load_newtry_checkpoint, AdaptedDiffusionModel

def main():
    print("=" * 60)
    print("测试 newtry checkpoint 适配器")
    print("=" * 60)

    # 配置
    data_cfg, model_cfg, train_cfg, infer_cfg = get_config()
    # 确保与 newtry 一致
    data_cfg.data_root = "/root/autodl-tmp/fyp_final/preprocessed_9ch_40x40"
    data_cfg.norm_stats_path = "/root/autodl-tmp/fyp_final/preprocessed_9ch_40x40/norm_stats.pt"

    print(f"\n配置检查:")
    print(f"  in_channels: {model_cfg.in_channels} (期望: 9)")
    print(f"  cond_channels: {model_cfg.cond_channels} (期望: 45)")
    print(f"  d_model: {model_cfg.d_model}")
    print(f"  n_dit_layers: {model_cfg.n_dit_layers}")
    print(f"  prediction_type: {model_cfg.prediction_type}")

    # 加载 checkpoint
    ckpt_path = "/root/autodl-tmp/fyp_final/Ver4/newtry/checkpoints/best_eps.pt"
    print(f"\n加载 checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_newtry_checkpoint(ckpt_path, data_cfg, model_cfg, train_cfg, device)

    print(f"\n✅ 模型创建成功!")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   设备: {device}")

    # 测试前向传播
    print(f"\n测试前向传播:")
    B = 2
    cond = torch.randn(B, model_cfg.cond_channels, 40, 40).to(device)
    target = torch.randn(B, model_cfg.in_channels, 40, 40).to(device)

    with torch.no_grad():
        output = model(cond, target)
        print(f"   loss_mse: {output['loss_mse'].item():.4f}")
        print(f"   loss_div: {output['loss_div'].item():.4f}")
        print(f"   loss_curl: {output['loss_curl'].item():.4f}")
        print(f"   eps_pred shape: {output['eps_pred'].shape}")

    # 测试采样
    print(f"\n测试 DDIM 采样:")
    with torch.no_grad():
        pred = model.sample(cond, device, ddim_steps=10, clamp_range=(-5.0, 5.0))
        print(f"   预测 shape: {pred.shape}")
        print(f"   预测范围: [{pred.min().item():.3f}, {pred.max().item():.3f}]")

    print(f"\n✅ 所有测试通过! 适配器工作正常")
    print(f"\n现在可以使用以下命令运行对比实验:")
    print(f"  cd /root/autodl-tmp/fyp_final/Ver4/compare_fm_dm")
    print(f"  python run_comparison.py \\")
    print(f"      --data_root /path/to/era5_data \\")
    print(f"      --work_dir ./results \\")
    print(f"      --external_dm_ckpt /path/to/newtry/best_eps.pt \\")
    print(f"      --skip_train")

if __name__ == "__main__":
    main()
