#!/bin/bash
# Flow Matching Z通道优化训练启动脚本
# 方案F+G+I组合: Z通道权重重分配 + 物理损失提前 + Z通道数据增强

set -e

PROJECT_DIR="/root/autodl-tmp/fyp_final"
cd "$PROJECT_DIR"

# 环境变量
export PYTHONPATH="$PROJECT_DIR/VER3_original/VER3:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2

# 训练参数
ERA5_DIR="$PROJECT_DIR/preprocessed_9ch_40x40"
CSV_PATH="$PROJECT_DIR/VER3_original/VER3/Trajectory/processed_typhoon_tracks.csv"
WORK_DIR="$PROJECT_DIR/VER3_original/VER3/flow_matching"
TRAIN_SCRIPT="$WORK_DIR/train_z_optimized.py"

# 检查数据
if [ ! -d "$ERA5_DIR" ]; then
    echo "错误: 数据目录不存在: $ERA5_DIR"
    exit 1
fi

if [ ! -f "$CSV_PATH" ]; then
    echo "错误: CSV文件不存在: $CSV_PATH"
    exit 1
fi

# 创建checkpoint目录
mkdir -p "$WORK_DIR/checkpoints/z_optimized"

# 训练命令
echo "=========================================="
echo "FM Z通道优化训练启动 (方案F+G+I)"
echo "=========================================="
echo "数据目录: $ERA5_DIR"
echo "工作目录: $WORK_DIR"
echo "训练脚本: $TRAIN_SCRIPT"
echo "=========================================="

python3 "$TRAIN_SCRIPT" \
    --era5_dir "$ERA5_DIR" \
    --csv_path "$CSV_PATH" \
    --work_dir "$WORK_DIR" \
    --batch_size 64 \
    --max_epochs 200 \
    --lr 2e-4 \
    --seed 42 \
    2>&1 | tee "$WORK_DIR/train_z_optimized_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "训练完成!"
echo "Checkpoint 保存于: $WORK_DIR/checkpoints/z_optimized/"
echo "=========================================="
