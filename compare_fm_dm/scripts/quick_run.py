"""
快速入口脚本 — 训练 + 评估 + 可视化一键运行

使用方法:
    # 完整训练和评估
    python scripts/quick_run.py --mode full --data_root /path/to/data

    # 仅训练
    python scripts/quick_run.py --mode train --data_root /path/to/data

    # 仅评估 (需要已有checkpoint)
    python scripts/quick_run.py --mode eval --data_root /path/to/data

    # 仅可视化 (需要已有results)
    python scripts/quick_run.py --mode visualize
"""
import os
import sys
import json
import argparse
import logging

import numpy as np
import torch

from configs import DataConfig, ModelConfig, TrainConfig, InferenceConfig, ComparisonConfig, get_config, get_comparison_config
from data.dataset import build_dataloaders
from models.unified_model import UnifiedModel, create_model
from models.trainer import UnifiedTrainer, EMA
from evaluation.metrics import ComparisonEvaluator
from run_comparison import ComparisonExperiment, autoregressive_inference, load_checkpoint, find_ar_sequences
from visualization.plots import visualize_comparison_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_single_model(
    method: str,
    data_cfg, model_cfg, train_cfg, infer_cfg,
    train_loader, val_loader,
    work_dir: str,
    device: torch.device,
) -> UnifiedModel:
    """训练单个模型"""
    logger.info(f"=" * 60)
    logger.info(f"训练模型: {method.upper()}")
    logger.info(f"=" * 60)

    model = create_model(model_cfg, data_cfg, train_cfg, method=method)
    model = model.to(device)

    trainer = UnifiedTrainer(
        model, train_loader, val_loader,
        train_cfg, data_cfg,
        work_dir=work_dir, method=method,
    )
    trainer.train()

    # 加载best checkpoint
    best_ckpt = os.path.join(work_dir, f"checkpoints_{method}", "best.pt")
    if os.path.exists(best_ckpt):
        model, _ = load_checkpoint(model, best_ckpt, device, use_ema=True)

    model.eval()
    return model


def quick_evaluate(
    model: UnifiedModel,
    test_dataset,
    data_cfg,
    infer_cfg,
    norm_mean, norm_std,
    device: torch.device,
    method: str,
    num_samples: int = 100,
    ar_steps: int = 24,
) -> dict:
    """快速评估"""
    logger.info(f"评估 {method.upper()} 模型...")

    start_indices = find_ar_sequences(test_dataset, ar_steps, num_samples)
    if len(start_indices) == 0:
        start_indices = list(range(min(num_samples, len(test_dataset))))

    logger.info("评估样本数: {}, AR步数: {}".format(len(start_indices), ar_steps))

    preds, gts = autoregressive_inference(
        model, test_dataset, start_indices, ar_steps,
        device, method,
        clamp_range=infer_cfg.clamp_range,
        z_clamp_range=infer_cfg.z_clamp_range,
        euler_steps=infer_cfg.euler_steps,
        ddim_steps=infer_cfg.ddim_steps,
        noise_sigma=infer_cfg.autoregressive_noise_sigma,
        infer_cfg=infer_cfg,
    )

    evaluator = ComparisonEvaluator(
        data_cfg=data_cfg, device=device,
        norm_mean=norm_mean, norm_std=norm_std,
    )

    results = evaluator.evaluate_single(preds, gts, method_name=method.upper())
    return results


def main():
    parser = argparse.ArgumentParser(description="快速训练+评估")

    # 模式
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "train", "eval", "visualize"])

    # 数据
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/fyp_final/preprocessed_9ch_40x40")
    parser.add_argument("--preprocess_dir", type=str, default=None)
    parser.add_argument("--norm_stats", type=str, default=None)

    # 实验
    parser.add_argument("--run_fm", action="store_true", default=True)
    parser.add_argument("--run_dm", action="store_true", default=True)
    parser.add_argument("--skip_train", action="store_true", default=False)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)

    # 评估参数
    parser.add_argument("--ar_steps", type=int, default=24)
    parser.add_argument("--euler_steps", type=int, default=4)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eval_samples", type=int, default=100)

    # 输出
    parser.add_argument("--work_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="comparison_results")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 配置
    data_cfg, model_cfg, train_cfg, infer_cfg = get_config()
    data_cfg.data_root = args.data_root
    data_cfg.era5_dir = args.data_root
    if args.preprocess_dir:
        data_cfg.preprocessed_dir = args.preprocess_dir
    if args.norm_stats:
        data_cfg.norm_stats_path = args.norm_stats

    train_cfg.max_epochs = args.epochs
    train_cfg.batch_size = args.batch_size
    train_cfg.learning_rate = args.lr
    train_cfg.seed = args.seed

    infer_cfg.autoregressive_steps = args.ar_steps
    infer_cfg.euler_steps = args.euler_steps
    infer_cfg.ddim_steps = args.ddim_steps

    comp_cfg = get_comparison_config()
    comp_cfg.run_fm = args.run_fm
    comp_cfg.run_dm = args.run_dm
    comp_cfg.num_eval_samples = args.eval_samples
    comp_cfg.output_dir = args.output_dir
    comp_cfg.device = args.device

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    logger.info("加载数据...")
    train_loader, val_loader, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)
    test_dataset = test_loader.dataset
    logger.info(f"训练: {len(train_loader.dataset)}, 验证: {len(val_loader.dataset)}, 测试: {len(test_dataset)}")

    all_results = {}

    # 训练
    if args.mode in ["full", "train"]:
        fm_model = None
        dm_model = None

        if args.run_fm:
            fm_ckpt = os.path.join(args.work_dir, "checkpoints_fm", "best.pt")
            if args.skip_train and os.path.exists(fm_ckpt):
                logger.info("跳过FM训练，加载已有checkpoint")
                fm_model = create_model(model_cfg, data_cfg, train_cfg, method="fm")
                fm_model = fm_model.to(device)
                fm_model, _ = load_checkpoint(fm_model, fm_ckpt, device)
            else:
                fm_model = train_single_model(
                    "fm", data_cfg, model_cfg, train_cfg, infer_cfg,
                    train_loader, val_loader, args.work_dir, device,
                )

        if args.run_dm:
            dm_ckpt = os.path.join(args.work_dir, "checkpoints_dm", "best.pt")
            if args.skip_train and os.path.exists(dm_ckpt):
                logger.info("跳过DM训练，加载已有checkpoint")
                dm_model = create_model(model_cfg, data_cfg, train_cfg, method="dm")
                dm_model = dm_model.to(device)
                dm_model, _ = load_checkpoint(dm_model, dm_ckpt, device)
            else:
                dm_model = train_single_model(
                    "dm", data_cfg, model_cfg, train_cfg, infer_cfg,
                    train_loader, val_loader, args.work_dir, device,
                )

        # 保存模型引用供评估使用
        if args.mode == "train":
            logger.info("训练完成!")
            return

        # 继续评估
        if fm_model:
            all_results["FM"] = quick_evaluate(
                fm_model, test_dataset, data_cfg, infer_cfg,
                norm_mean, norm_std, device, "fm",
                num_samples=args.eval_samples, ar_steps=args.ar_steps,
            )
        if dm_model:
            all_results["DM"] = quick_evaluate(
                dm_model, test_dataset, data_cfg, infer_cfg,
                norm_mean, norm_std, device, "dm",
                num_samples=args.eval_samples, ar_steps=args.ar_steps,
            )

    # 仅评估模式
    elif args.mode == "eval":
        fm_model = None
        dm_model = None

        fm_ckpt = os.path.join(args.work_dir, "checkpoints_fm", "best.pt")
        dm_ckpt = os.path.join(args.work_dir, "checkpoints_dm", "best.pt")

        if args.run_fm and os.path.exists(fm_ckpt):
            fm_model = create_model(model_cfg, data_cfg, train_cfg, method="fm")
            fm_model = fm_model.to(device)
            fm_model, _ = load_checkpoint(fm_model, fm_ckpt, device)
            all_results["FM"] = quick_evaluate(
                fm_model, test_dataset, data_cfg, infer_cfg,
                norm_mean, norm_std, device, "fm",
                num_samples=args.eval_samples, ar_steps=args.ar_steps,
            )

        if args.run_dm and os.path.exists(dm_ckpt):
            dm_model = create_model(model_cfg, data_cfg, train_cfg, method="dm")
            dm_model = dm_model.to(device)
            dm_model, _ = load_checkpoint(dm_model, dm_ckpt, device)
            all_results["DM"] = quick_evaluate(
                dm_model, test_dataset, data_cfg, infer_cfg,
                norm_mean, norm_std, device, "dm",
                num_samples=args.eval_samples, ar_steps=args.ar_steps,
            )

    # 仅可视化模式
    elif args.mode == "visualize":
        results_path = os.path.join(args.output_dir, "comparison_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                all_results = json.load(f)
        else:
            logger.error(f"找不到结果文件: {results_path}")
            return

    # 保存结果
    if all_results:
        results_path = os.path.join(args.output_dir, "comparison_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"结果已保存: {results_path}")

        # 可视化
        visualize_comparison_results(all_results, args.output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
