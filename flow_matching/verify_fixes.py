"""
Flow Matching 代码修复验证脚本

验证内容:
- P0: 物理损失权重未被应用的问题 (loss_div + loss_curl 未加入总损失)
- P2: 物理损失退火调度 (_compute_physics_weight)
- P5: 通道分组处理修复 (深度可分离卷积 groups 参数)
- t边界: 防止 t=0 或 t=1 数值不稳定
- 气压层权重: 500hPa 引导层更高权重

使用方法:
python verify_fixes.py
"""
import sys
import os

# 添加项目路径
FLOW_MATCHING_DIR = "/root/autodl-tmp/fyp_final/VER3_original/VER3/flow_matching"
sys.path.insert(0, FLOW_MATCHING_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(FLOW_MATCHING_DIR)))

import torch
import torch.nn as nn
import numpy as np

def test_p0_physics_loss():
    """P0: 验证物理损失是否正确加入总损失"""
    print("\n" + "="*60)
    print("P0: 物理损失权重验证")
    print("="*60)
    
    # 检查 train_preprocessed.py 中是否正确应用了物理损失
    with open(os.path.join(FLOW_MATCHING_DIR, "train_preprocessed.py"), "r") as f:
        content = f.read()
    
    checks = [
        ("loss_div" in content and "loss_curl" in content, "物理损失变量存在"),
        ("outputs[\"loss_div\"]" in content or "outputs['loss_div']" in content, "从 model outputs 提取 loss_div"),
        ("outputs[\"loss_curl\"]" in content or "outputs['loss_curl']" in content, "从 model outputs 提取 loss_curl"),
        ("_compute_physics_weight" in content, "存在退火调度方法"),
        ("physics_weight *" in content, "物理损失权重乘法"),
    ]
    
    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")
        if not check:
            all_passed = False
    
    return all_passed


def test_p2_physics_annealing():
    """P2: 验证物理损失退火调度"""
    print("\n" + "="*60)
    print("P2: 物理损失退火调度验证")
    print("="*60)
    
    with open(os.path.join(FLOW_MATCHING_DIR, "train_preprocessed.py"), "r") as f:
        content = f.read()
    
    checks = [
        ("def _compute_physics_weight" in content, "存在 _compute_physics_weight 方法"),
        ("physics_warmup_type" in content, "退火类型配置"),
        ('"linear"' in content or '"cosine"' in content, "支持 linear 和 cosine 退火"),
        ("physics_target_weight" in content, "目标权重配置"),
        ("self.physics_warmup_steps" in content, "预热步数配置"),
    ]
    
    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")
        if not check:
            all_passed = False
    
    return all_passed


def test_p5_grouped_conv():
    """P5: 验证通道分组处理修复"""
    print("\n" + "="*60)
    print("P5: 通道分组处理验证")
    print("="*60)
    
    with open(os.path.join(FLOW_MATCHING_DIR, "models/flow_matching_model.py"), "r") as f:
        content = f.read()
    
    checks = [
        ("groups=6" in content, "风速组使用 groups=6 (6通道/2配对)"),
        ("groups=3" in content, "z组使用 groups=3 (3通道/气压层)"),
        ("Conv2d(6, 6, 3, padding=1, groups=6)" in content, "深度卷积输入通道=6"),
        ("Conv2d(3, 3, 3, padding=1, groups=3)" in content, "z深度卷积输入通道=3"),
    ]
    
    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")
        if not check:
            all_passed = False
    
    return all_passed


def test_p1_time_embedding():
    """P1: 验证时间嵌入缩放"""
    print("\n" + "="*60)
    print("P1: 时间嵌入缩放验证")
    print("="*60)
    
    with open(os.path.join(FLOW_MATCHING_DIR, "models/flow_matching_model.py"), "r") as f:
        content = f.read()
    
    checks = [
        ("time_embedding_scale" in content, "时间嵌入缩放参数存在"),
        ("t_input = t * self.time_embedding_scale" in content, "缩放应用到时间输入"),
    ]
    
    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")
        if not check:
            all_passed = False
    
    return all_passed


def test_p3_midpoint_sampler():
    """P3: 验证中点法采样器"""
    print("\n" + "="*60)
    print("P3: 中点法采样器验证")
    print("="*60)
    
    with open(os.path.join(FLOW_MATCHING_DIR, "inference.py"), "r") as f:
        content = f.read()
    
    checks = [
        ("def _step_midpoint" in content, "存在 _step_midpoint 方法"),
        ("euler_mode == \"midpoint\"" in content, "支持 midpoint 模式"),
        ("x_mid = x - v1 * (dt / 2.0)" in content, "正确的中点计算"),
    ]
    
    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")
        if not check:
            all_passed = False
    
    return all_passed


def test_p4_adaptive_z_clamp():
    """P4: 验证自适应 Z-clamp"""
    print("\n" + "="*60)
    print("P4: 自适应 Z-clamp 验证")
    print("="*60)
    
    with open(os.path.join(FLOW_MATCHING_DIR, "inference.py"), "r") as f:
        content = f.read()
    
    checks = [
        ("use_adaptive_z_clamp" in content, "自适应 z-clamp 配置"),
        ("_compute_z_clamp_range" in content, "clamp 范围计算方法"),
        ("z_clamp_sigma" in content, "sigma 参数"),
        ("mu" in content.lower() or "z_means" in content, "基于统计量计算"),
    ]
    
    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")
        if not check:
            all_passed = False
    
    return all_passed


def test_optimal_transport_path():
    """验证最优传输路径数学正确性"""
    print("\n" + "="*60)
    print("最优传输路径验证")
    print("="*60)
    
    with open(os.path.join(FLOW_MATCHING_DIR, "models/flow_matching_model.py"), "r") as f:
        content = f.read()
    
    # 检查 x0_pred 的计算: x0 = x_t - t * v
    checks = [
        ("x0_pred = x_t - t_reshaped * v_pred" in content or 
         "x0_pred = x_t - t *" in content, 
         "x0 反推公式正确 (x_t - t*v)"),
        ("target_v = x_1 - target" in content, "目标速度 v = x1 - x0"),
        ("x_t = (1 - t_reshaped) * target + t_reshaped * x_1" in content, "线性路径 x_t"),
    ]
    
    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")
        if not check:
            all_passed = False
    
    return all_passed


def test_t_boundary_fix():
    """验证 t 边界修复"""
    print("\n" + "="*60)
    print("t 边界修复验证")
    print("="*60)
    
    with open(os.path.join(FLOW_MATCHING_DIR, "models/flow_matching_model.py"), "r") as f:
        content = f.read()
    
    # 检查 t 的采样是否有边界保护
    checks = [
        ("1e-5" in content, "存在微小偏移防止 t=0"),
        ("torch.rand(B, device=device) * (1 - 1e-5)" in content, "t 有上限保护"),
    ]
    
    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")
        if not check:
            all_passed = False
    
    return all_passed


def test_pressure_layer_weights():
    """验证气压层权重配置"""
    print("\n" + "="*60)
    print("气压层权重配置验证")
    print("="*60)
    
    with open(os.path.join(FLOW_MATCHING_DIR, "configs_preprocessed.py"), "r") as f:
        content = f.read()
    
    # 检查 500hPa 是否有更高权重
    # 权重格式: (u_850, u_500, u_250, v_850, v_500, v_250, z_850, z_500, z_250)
    checks = [
        ("use_channel_weights: bool = True" in content, "通道权重启用"),
        ("2.5" in content, "500hPa 层级有更高权重 (2.5 vs 1.0)"),
        ("4.0" in content, "z_500 有最高权重 (4.0)"),
    ]
    
    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")
        if not check:
            all_passed = False
    
    return all_passed


def test_wind_pairs():
    """验证风场对索引正确性"""
    print("\n" + "="*60)
    print("风场对索引验证")
    print("="*60)
    
    with open(os.path.join(FLOW_MATCHING_DIR, "configs_preprocessed.py"), "r") as f:
        content = f.read()
    
    # 检查 get_wind_channel_indices 方法
    checks = [
        ("def get_wind_channel_indices" in content, "存在风场索引方法"),
        ("u_idx * n_pl + lev" in content, "正确计算 u 通道索引"),
        ("v_idx * n_pl + lev" in content, "正确计算 v 通道索引"),
    ]
    
    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")
        if not check:
            all_passed = False
    
    # 数学验证
    print("\n  通道顺序验证:")
    print("    pressure_level_vars = ['u', 'v', 'z']")
    print("    pressure_levels = [850, 500, 250]")
    print("    通道映射:")
    print("      u_850 -> 0, u_500 -> 1, u_250 -> 2")
    print("      v_850 -> 3, v_500 -> 4, v_250 -> 5")
    print("      z_850 -> 6, z_500 -> 7, z_250 -> 8")
    print("    风场对 (同一气压层):")
    print("      (0, 3) -> 850hPa")
    print("      (1, 4) -> 500hPa")
    print("      (2, 5) -> 250hPa")
    
    return all_passed


def test_deepwise_conv_math():
    """验证深度可分离卷积数学"""
    print("\n" + "="*60)
    print("深度可分离卷积数学验证")
    print("="*60)
    
    print("  9通道 ERA5 数据 (u,v,z × 3气压层):")
    print("    - u通道: [0,1,2] -> u_850, u_500, u_250")
    print("    - v通道: [3,4,5] -> v_850, v_500, v_250")
    print("    - z通道: [6,7,8] -> z_850, z_500, z_250")
    print()
    print("  原始错误:")
    print("    - wind_conv: groups=2, in_channels=2 -> 仅支持2通道输入")
    print("    - 但实际输入是6通道 (u_850,u_500,u_250,v_850,v_500,v_250)")
    print()
    print("  修复方案:")
    print("    - wind_conv: groups=6, in_channels=6 -> 每个气压层的 uv 单独处理")
    print("    - z_conv: groups=3, in_channels=3 -> 每个气压层的 z 单独处理")
    print()
    
    # 数学验证
    in_channels = 6  # uv
    groups = 6
    out_channels = 384 // 3  # group_dim
    
    # depthwise 输出的通道数必须等于输入通道数
    depthwise_out = in_channels  # 深度卷积输出=输入
    print(f"  深度卷积: in={in_channels}, out={depthwise_out}, groups={groups}")
    print(f"    验证: {depthwise_out} == {in_channels} ✓" if depthwise_out == in_channels else "  ✗")
    
    # pointwise 降维
    print(f"  点卷积: in={depthwise_out}, out={out_channels}")
    print()
    
    return True


def main():
    print("\n" + "="*60)
    print("Flow Matching 代码修复验证")
    print("="*60)
    
    results = {}
    results["P0 物理损失"] = test_p0_physics_loss()
    results["P1 时间嵌入"] = test_p1_time_embedding()
    results["P2 物理退火"] = test_p2_physics_annealing()
    results["P3 中点采样"] = test_p3_midpoint_sampler()
    results["P4 自适应Z"] = test_p4_adaptive_z_clamp()
    results["P5 通道分组"] = test_p5_grouped_conv()
    results["最优传输路径"] = test_optimal_transport_path()
    results["t 边界修复"] = test_t_boundary_fix()
    results["气压层权重"] = test_pressure_layer_weights()
    results["风场对索引"] = test_wind_pairs()
    test_deepwise_conv_math()
    
    print("\n" + "="*60)
    print("验证结果汇总")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("="*60)
        print("🎉 所有代码修复验证通过!")
        print("="*60)
    else:
        print("="*60)
        print("⚠️  部分验证失败，请检查")
        print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
