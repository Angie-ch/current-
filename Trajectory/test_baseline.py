"""
测试阶段1原始模型直接吃扩散ERA5的效果（无finetune）
用于判断是否需要finetune
"""
import sys, os
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import model_cfg, train_cfg
from model import LT3PModel
from dataset import filter_short_storms, filter_out_of_range_storms, split_storms_by_id
from data_processing import load_tyc_storms
from finetune_train import DiffusionERA5Dataset
from train import evaluate_on_test


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据
    track_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_typhoon_tracks.csv')
    data_root = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\fyp\Desktop\Typhoon_data_final'

    storms = load_tyc_storms(track_csv, data_root)
    storms = filter_short_storms(storms, train_cfg.min_typhoon_duration_hours)
    storms = filter_out_of_range_storms(storms)
    _, _, test_storms = split_storms_by_id(storms, train_cfg.train_ratio, train_cfg.val_ratio, seed=42)

    print(f"测试台风数: {len(test_storms)}")

    # 加载扩散ERA5缓存
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion_era5_cache', 'era5_cache.npz')
    cache = dict(np.load(cache_path, allow_pickle=True))
    print(f"缓存台风数: {len(cache)}")

    # 创建测试集
    test_ds = DiffusionERA5Dataset(test_storms, cache, stride=model_cfg.t_future)
    test_loader = DataLoader(test_ds, 128, shuffle=False, num_workers=0, pin_memory=True)

    # 加载阶段1原始模型
    model = LT3PModel(
        coord_dim=model_cfg.coord_dim,
        output_dim=model_cfg.output_dim,
        era5_channels=15,
        t_history=model_cfg.t_history,
        t_future=model_cfg.t_future,
        d_model=model_cfg.transformer_dim,
        n_heads=model_cfg.transformer_heads,
        n_layers=model_cfg.transformer_layers,
        ff_dim=model_cfg.transformer_ff_dim,
        dropout=model_cfg.dropout,
    )

    ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
    if 'ema_model_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_model_state_dict'])
        print("已加载 EMA 权重")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print("已加载模型权重")

    model.to(device).eval()

    print("\n" + "=" * 60)
    print("阶段1原始模型 + 扩散ERA5 (无finetune)")
    print("=" * 60)
    evaluate_on_test(model, test_loader, device)


if __name__ == '__main__':
    main()
