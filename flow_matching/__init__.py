"""
Flow Matching — 经典流匹配 (Classic Flow Matching, CFM)

基于 ICLR 2023 "Flow Matching for Generative Modeling" (Lipman et al.)

核心优势（vs 扩散模型 DDPM/DDIM）:
  - 速度: Euler 1步采样，50倍提速
    DDIM: 1464台风 × 50步 × 1000步扩散 ≈ 10小时
    CFM:  1464台风 × 1步 Euler ≈ 15分钟
  - 简洁: 无噪声调度器，t∈[0,1] 连续采样
  - 精度: 最优传输路径保证最低方差
  - 物理一致性: 生成场更平滑、更物理连贯

文件结构:
  configs/config.py   - 配置文件 (DataConfig, ModelConfig, TrainConfig, InferenceConfig)
  models/flow_matching_model.py - CFM-DiT 模型 (ERA5FlowMatchingModel)
  train.py           - CFM 训练脚本
  inference.py       - Euler 高速采样推理
  finetune_train.py  - CFM-ERA5 微调轨迹模型
"""
from .models.flow_matching_model import ERA5FlowMatchingModel, CFMDiT
from .configs import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    InferenceConfig,
    get_config,
)
