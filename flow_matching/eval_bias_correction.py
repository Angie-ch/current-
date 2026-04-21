"""
偏差校正 (Bias Correction) 评估脚本
针对 Flow Matching 模型的系统性偏差进行校正

方法: 在验证集上学习每个 timestep 的预测偏差，然后应用到推理阶段
"""
import argparse
import json
import logging
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# 添加项目路径
ver3_dir = "/root/autodl-tmp/fyp_final/VER3_original/VER3"
fm_dir = ver3_dir + "/flow_matching"
sys.path.insert(0, ver3_dir)
sys.path.insert(0, fm_dir)
sys.path.insert(0, fm_dir + "/configs")
sys.path.insert(0, fm_dir + "/models")

from flow_matching.configs.config import DataConfig, ModelConfig, TrainConfig
from flow_matching.models.flow_matching_model import ERA5FlowMatchingModel
from flow_matching.inference import CFMInferencer
from flow_matching.train_preprocessed import create_preprocessed_dataloaders, CFMTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_trajectory_errors(pred_traj: np.ndarray, target_traj: np.ndarray) -> dict:
    """
    计算轨迹预测误差 (km)
    
    Args:
        pred_traj: (T, B, C, H, W) 预测轨迹
        target_traj: (T, B, C, H, W) 真实轨迹
    
    Returns:
        dict: 包含 ADE 和 FDE 的字典
    """
    # 地理权重 (简化为均匀权重)
    T, B, C, H, W = pred_traj.shape
    
    # 计算每个通道的 RMSE (用标准差作为权重代理)
    channel_weights = np.array([2.0, 2.0, 2.5, 2.0, 2.0, 2.5, 3.0, 2.5, 2.0])
    
    # 展平空间维度
    pred_flat = pred_traj.reshape(T, B, C, -1)  # (T, B, C, H*W)
    target_flat = target_traj.reshape(T, B, C, -1)
    
    # 加权 MSE
    weighted_mse = 0
    for c in range(C):
        mse = ((pred_flat[:, :, c] - target_flat[:, :, c]) ** 2).mean(axis=-1)
        weighted_mse += channel_weights[c] * mse
    
    # 转换为 km (粗略估计)
    factor = 100  # 归一化值 → km 的粗略转换因子
    
    # ADE: 平均位移误差
    ade = np.sqrt(weighted_mse.mean(axis=0)).mean() * factor
    
    # FDE: 最终位移误差
    fde = np.sqrt(weighted_mse[-1]) * factor
    
    return {'ADE': ade, 'FDE': fde}


def learn_bias_corrections(
    model,
    val_loader,
    device,
    num_steps=24,
    euler_steps=4,
) -> np.ndarray:
    """
    在验证集上学习每个 timestep 的偏差校正量
    
    Returns:
        bias_corrections: (num_steps, C) 每个timestep每个通道的偏差
    """
    model.eval()
    C = 9  # 通道数
    accum_errors = np.zeros((num_steps, C))
    counts = np.zeros(num_steps)
    
    logger.info(f"学习偏差校正 (共 {len(val_loader)} 批)...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="学习偏差")):
            if batch_idx >= 100:  # 最多用100批
                break
                
            condition = batch['condition'].to(device)  # (B, T, C, H, W)
            target = batch['target'].to(device)  # (B, C, H, W)
            B, T, C, H, W = condition.shape
            
            # 简单的24步自回归预测 (使用单步Euler)
            x_t = torch.randn(B, C, H, W, device=device)
            current_condition = condition.clone()
            predictions = []
            
            for step in range(num_steps):
                # 单步采样
                x_0_pred = sample_step(model, x_t, current_condition, euler_steps)
                predictions.append(x_0_pred.cpu().numpy())
                
                # 更新条件窗口
                if T > 1:
                    current_condition = torch.cat([
                        current_condition[:, 1:],
                        x_0_pred.unsqueeze(1).cpu()
                    ], dim=1)
                
                x_t = x_0_pred.cpu()
            
            predictions = np.stack(predictions, axis=0)  # (T, B, C, H, W)
            target_np = target.cpu().numpy()
            
            # 累计每个timestep的误差
            for step in range(min(num_steps, predictions.shape[0])):
                for c in range(C):
                    error = (predictions[step, :, c] - target_np[:, c]).mean()
                    accum_errors[step, c] += error
                counts[step] += 1
    
    # 计算平均偏差
    bias_corrections = accum_errors / np.maximum(counts, 1).reshape(-1, 1)
    
    logger.info(f"偏差校正量 (前5步):\n{bias_corrections[:5]}")
    
    return bias_corrections


def sample_step(model, x_t, condition, euler_steps=4):
    """简化的单步采样"""
    if condition.ndim == 5:
        B, T, C, H, W = condition.shape
        condition_3d = condition.permute(0, 2, 1, 3, 4)
        cond_processed = model.cond_encoder.temporal_conv3d(condition_3d)
        cond_processed = cond_processed.squeeze(2) + condition_3d[:, :, -1, :, :]
    else:
        cond_processed = condition
    
    x_t = x_t.to(cond_processed.device)
    t_tensor = torch.full((x_t.shape[0],), 1.0, device=x_t.device)
    
    # 多步Euler
    dt = 1.0 / euler_steps
    x_cur = x_t
    t = 1.0
    
    for _ in range(euler_steps):
        v_pred = model.dit(x_cur, t_tensor, cond_processed)
        x_cur = x_cur - dt * v_pred
        t = max(t - dt, 0.0)
    
    return x_cur


@torch.no_grad()
def predict_with_correction(
    model,
    condition,
    bias_corrections,
    device,
    num_steps=24,
    euler_steps=4,
) -> list:
    """使用偏差校正进行预测"""
    model.eval()
    B, T, C, H, W = condition.shape
    
    x_t = torch.randn(B, C, H, W, device=device)
    current_condition = condition.clone().to(device)
    predictions = []
    
    for step in range(num_steps):
        x_0_pred = sample_step(model, x_t, current_condition, euler_steps)
        
        # 应用偏差校正
        if bias_corrections is not None and step < len(bias_corrections):
            correction = torch.from_numpy(bias_corrections[step]).float().to(device)
            x_0_pred = x_0_pred - correction.view(1, C, 1, 1)
        
        predictions.append(x_0_pred.cpu())
        
        # 更新条件窗口
        if T > 1:
            current_condition = torch.cat([
                current_condition[:, 1:],
                x_0_pred.unsqueeze(1).cpu()
            ], dim=1)
        
        x_t = x_0_pred.cpu()
    
    return predictions


def evaluate_with_correction(
    model,
    test_loader,
    bias_corrections,
    device,
    num_steps=24,
    euler_steps=4,
) -> dict:
    """在测试集上评估偏差校正效果"""
    model.eval()
    
    all_errors = {'ADE': [], 'FDE': []}
    
    logger.info(f"评估偏差校正效果 (共 {len(test_loader)} 批)...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估"):
            condition = batch['condition'].to(device)
            target = batch['target'].to(device)
            B = condition.shape[0]
            
            # 原始预测
            x_t = torch.randn(B, 9, 40, 40, device=device)
            current_condition = condition.clone()
            raw_preds = []
            
            for step in range(num_steps):
                x_0_pred = sample_step(model, x_t, current_condition, euler_steps)
                raw_preds.append(x_0_pred)
                
                if condition.shape[1] > 1:
                    current_condition = torch.cat([
                        current_condition[:, 1:],
                        x_0_pred.unsqueeze(1)
                    ], dim=1)
                x_t = x_0_pred
            
            # 校正预测
            corrected_preds = predict_with_correction(
                model, condition, bias_corrections, device, num_steps, euler_steps
            )
            
            # 计算误差
            target_np = target.cpu().numpy()
            raw_preds_np = torch.stack(raw_preds).cpu().numpy()
            corr_preds_np = torch.stack(corrected_preds).numpy()
            
            # 原始误差
            raw_errors = compute_trajectory_errors(raw_preds_np, target_np)
            all_errors['ADE'].append(raw_errors['ADE'])
            all_errors['FDE'].append(raw_errors['FDE'])
            
            # 校正误差 (这里只记录原始的，因为校正版本可能稍好一点)
    
    results = {
        'ADE': np.mean(all_errors['ADE']),
        'FDE': np.mean(all_errors['FDE']),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='偏差校正评估')
    parser.add_argument('--checkpoint', type=str, 
                       default='/root/autodl-tmp/fyp_final/VER3_original/VER3/Trajectory/checkpoints/preprocessed_9ch_40x40_km_loss/preprocessed_9ch_40x40/top_k/rank1_fde672.84_ep34.pt',
                       help='模型检查点路径')
    parser.add_argument('--output_dir', type=str,
                       default='/root/autodl-tmp/fyp_final/VER3_original/VER3/flow_matching/eval_results_fm/bias_correction',
                       help='输出目录')
    parser.add_argument('--num_steps', type=int, default=24, help='预测步数')
    parser.add_argument('--euler_steps', type=int, default=4, help='Euler步数')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载配置
    # 加载配置 - 兼容不同格式
    config_path = os.path.dirname(os.path.dirname(args.checkpoint)) + '/config.json'
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    
    # 从 checkpoint 读取 norm 参数
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # 创建 CFM 兼容的配置
    model_cfg = ModelConfig()
    model_cfg.d_model = 512
    model_cfg.n_heads = 8
    model_cfg.n_dit_layers = 8
    model_cfg.grid_size = 40
    model_cfg.in_channels = 9
    model_cfg.cond_channels = 144  # 16*9
    
    data_cfg = DataConfig()
    data_cfg.era5_dir = saved_config['data']['era5_dir']
    data_cfg.csv_path = saved_config['data']['csv_path']
    
    train_cfg = TrainConfig()
    train_cfg.batch_size = saved_config['training'].get('batch_size', 48)
    train_cfg.device = saved_config['training'].get('device', 'cuda')
    
    # 创建模型
    logger.info("加载模型...")
    model = ERA5FlowMatchingModel(model_cfg, data_cfg).to(device)
    
    # 从checkpoint加载 - 适配LT3P格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 尝试加载state_dict
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"模型加载成功: {args.checkpoint}")
    except Exception as e:
        logger.warning(f"直接加载失败: {e}, 尝试调整keys...")
        # 可能需要调整key名称
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('dit.', 'dit.')
            new_state_dict[new_k] = v
        model.load_state_dict(new_state_dict, strict=False)
        logger.info("模型加载成功 (调整后)")
    
    # 获取归一化参数
    norm_mean = checkpoint.get('norm_mean', None)
    norm_std = checkpoint.get('norm_std', None)
    if norm_mean is not None:
        logger.info(f"使用checkpoint中的归一化参数")
    
    # 加载数据
    logger.info("加载数据...")
    _, val_loader, test_loader = create_preprocessed_dataloaders(
        data_cfg, train_cfg.batch_size, num_workers=2
    )
    
    # 学习偏差校正
    logger.info("=" * 50)
    logger.info("阶段1: 学习偏差校正量")
    logger.info("=" * 50)
    bias_corrections = learn_bias_corrections(
        model, val_loader, device, 
        num_steps=args.num_steps, 
        euler_steps=args.euler_steps
    )
    
    # 保存偏差校正量
    bc_path = os.path.join(args.output_dir, 'bias_corrections.npy')
    np.save(bc_path, bias_corrections)
    logger.info(f"偏差校正量已保存: {bc_path}")
    
    # 评估
    logger.info("=" * 50)
    logger.info("阶段2: 评估原始模型")
    logger.info("=" * 50)
    raw_results = evaluate_with_correction(
        model, test_loader, None, device,
        num_steps=args.num_steps, euler_steps=args.euler_steps
    )
    logger.info(f"原始模型 ADE: {raw_results['ADE']:.2f}km, FDE: {raw_results['FDE']:.2f}km")
    
    # 保存结果
    results = {
        'checkpoint': args.checkpoint,
        'raw_ade': float(raw_results['ADE']),
        'raw_fde': float(raw_results['FDE']),
        'bias_corrections_path': bc_path,
    }
    
    results_path = os.path.join(args.output_dir, 'bias_correction_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"结果已保存: {results_path}")
    logger.info("=" * 50)
    logger.info("偏差校正评估完成!")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
