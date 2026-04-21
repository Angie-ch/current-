# Flow Matching 训练改进建议

## 当前问题分析

### 1. Z通道变化小
- **原因**: 推理时Z-delta clamp过于保守(400/350/250 m²/s²)
- **原因**: 训练时Z通道损失权重偏低
- **原因**: Z和UV没有联合约束

### 2. RMSE大
- **原因**: x0_loss_weight=0.1过低
- **原因**: physics_loss_weight=0.05过低
- **原因**: channel_weights中Z权重过高(主导梯度)

### 3. 风场移动方向错误
- **原因**: 地转平衡约束未启用(use_geostrophic_physics=False)
- **原因**: UV和Z独立优化，没有物理一致性

---

## 改进方案

### 方案A: 配置优化 (不需要改代码)

编辑 `configs/config.py` 中的 `TrainConfig`:

```python
# 提高x0_loss权重 (最关键)
x0_loss_weight: float = 0.5  # 从0.1改为0.5

# 提高物理损失权重
physics_target_weight: float = 0.3  # 从0.05改为0.3

# 启用地转平衡
use_geostrophic_physics: bool = True
geostrophic_weight: float = 0.15  # 新增

# 调整通道权重 (降低Z权重)
channel_weights: Tuple[float, ...] = (
    2.0, 2.0, 3.0,   # u_850, u_500, u_250 (提高)
    2.0, 2.0, 3.0,   # v_850, v_500, v_250 (提高)
    1.0, 1.2, 1.5,   # z_850, z_500, z_250 (降低)
)

# 提前启动物理损失
physics_warmup_start_epoch: int = 50  # 从100改为50
physics_warmup_end_epoch: int = 150  # 从160改为150
```

### 方案B: 推理优化 (不需要重新训练)

修改 `inference.py` 中的推理配置:

```python
# 放松Z-clamp限制
z_delta_max = [600.0, 500.0, 400.0]  # 从[400,350,250]改为更大值

# 使用Heun求解器替代Midpoint
euler_mode = 'heun'  # 精度更高
euler_steps = 20  # 从8改为20

# 增加集成次数
ensemble_k = 7  # 从3改为7
```

### 方案C: 新损失函数 (需要改代码)

使用改进的损失函数:

1. `UVZSeparatedLoss` - 分离UV和Z通道的损失
2. `GeostrophicBalanceLoss` - 地转平衡约束
3. `TemporalConsistencyLoss` - 时间一致性

---

## 快速修复清单

### 1. 推理时调整 (立竿见影)

在 `visualize_uv_comparison.py` 中:

```python
# 增大Z-delta clamp
z_delta_max_phys = [600.0, 500.0, 400.0]  # 原来[400,350,250]

# 减小clamp范围
clamp_range = (-3.5, 3.5)  # 原来(-5,5)
```

### 2. 模型改进

#### 模型结构问题
- `temporal_conv3d` 的stride=(2,1,1)会压缩时间维度
- 建议改为stride=(1,1,1)保持时间分辨率

#### 条件编码问题
- 当前只使用最后一帧作为条件
- 建议使用所有16帧历史(condition.ndim==5)

---

## 推荐改进优先级

| 优先级 | 改进 | 预期效果 | 难度 |
|--------|------|----------|------|
| ⭐⭐⭐ | 提高x0_loss_weight到0.5 | RMSE降低15-20% | 低 |
| ⭐⭐⭐ | 提高physics_loss_weight到0.3 | 风场方向改善 | 低 |
| ⭐⭐ | 增加euler_steps到20 | 预测精度提升 | 低 |
| ⭐⭐ | 放松Z-clamp限制 | Z变化更自然 | 低 |
| ⭐ | 启用地转平衡约束 | 物理一致性 | 中 |
| ⭐ | 分离UV/Z损失 | 各通道优化 | 中 |

---

## 测试建议

1. 先在推理阶段测试(不需要重新训练)
2. 观察Z通道变化是否更自然
3. 观察RMSE是否降低
4. 如果效果好，再考虑重新训练
