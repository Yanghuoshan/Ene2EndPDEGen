# Ground Truth 对比可视化 - 使用指南

## 快速开始

### 运行推理
```bash
cd /root/workspace/Ene2EndPDEGen
python scripts/inference_sd_non_connection_shallow.py config_shallow_Gabor.yml
```

### 输出验证

脚本运行完成后，检查保存路径（通常为 `SAVE_PATH`配置的目录，默认为 `saved_models`）：

```bash
ls -lh saved_models/
# 应该包含：
# - comparison_trajectory.gif (或 generated_trajectory.gif 如果无GT)
# - frequency_energy_spectrum.png
# - error_metrics.txt (如果有GT)
```

## 输出文件详解

### 1. comparison_trajectory.gif

**格式**: 3列动画 (GT | Prediction | Error)

**使用场景**：
- 快速可视化检查模型质量
- 对比生成特征与真实数据的相似性
- 识别哪些区域存在较大误差

**查看方式**:
```bash
# Linux
eog saved_models/comparison_trajectory.gif  # Eye of GNOME

# 或在notebook中:
from IPython.display import Image
Image('saved_models/comparison_trajectory.gif')
```

### 2. frequency_energy_spectrum.png

**布局**: 2×2 图标 (当有GT时)

上行:
- 左: 时间轴能量谱对比 (Generated vs GT)
- 右: 时间轴能量误差

下行:
- 左: 初始时刻空间能量谱对比  
- 右: 初始时刻空间能量误差

**解读**:
- **对数坐标**: 便于观察多个数量级的能量分布
- **实线 ○**: 模型生成的结果
- **虚线 □**: Ground Truth (GT)
- **能量误差**: 量化GT和Pred在各频率的差异

**应用**:
- 评估模型是否捕捉到正确的时间和空间尺度
- 识别哪些频率范围模型表现不好
- 检查能量级联是否正确

### 3. error_metrics.txt

**内容**: 详细的数值误差统计

**示例输出结构**:
```
ERROR METRICS COMPARISON
============================================================

Global Error Metrics:
  MSE (L2^2):  X.XXXXE-0X
  RMSE (L2):   X.XXXXE-0X
  MAE (L1):    X.XXXXE-0X
  Max Error:   X.XXXXE-0X

Vorticity:
  RMSE:  X.XXXXE-0X
  MAE:   X.XXXXE-0X
  MSE:   X.XXXXE-0X

Height:
  RMSE:  X.XXXXE-0X
  MAE:   X.XXXXE-0X
  MSE:   X.XXXXE-0X

Temporal Error Evolution:
  t=0: RMSE = X.XXXXE-0X
  t=1: RMSE = X.XXXXE-0X
  ...
```

**指标说明**:

| 指标 | 含义 | 特点 |
|------|------|------|
| MSE | 均方误差 | 对大误差更敏感 |
| RMSE | 均方根误差 | 与原始数据量纲一致 |
| MAE | 平均绝对误差 | 更稳健，不受离群值影响 |
| Max Error | 最大误差 | 最坏情况下的性能 |

## 故障排除

### 问题1: 没有生成 error_metrics.txt

**可能原因**: 数据集无法加载ground truth

**解决方案**:
1. 检查数据集路径是否正确
2. 验证数据集格式是否为有效的h5文件
3. 查看console输出中的警告信息 "Warning: Could not load ground truth sample from dataset"

### 问题2: 频域图表不完整 (只有1×2 而不是 2×2)

**原因**: 这是正常的，当GT数据不可用时，只显示生成的结果

**验证方法**:
```bash
grep "Spatial Energy Spectrum - Generated" frequency_energy_spectrum.png
```

### 问题3: 对比gif尺寸很大

**性能优化**:
```bash
# 转换为更小的格式 (降低帧率)
ffmpeg -i comparison_trajectory.gif -vf fps=5 -loop 0 comparison_trajectory_light.gif
```

### 问题4: 图表中的颜色奇怪

**检查事项**:
- 是否通过了数据归一化步骤 (查看是否加载了 `normalizer_params.pt`)
- 颜色范围是否使用了combined min/max (应该是)

## 配置调优

### 降低推理时间
```yaml
sampling_mode: one-step  # 更快的生成
num_sampling_steps: 1    # 最少步数
```

### 生成更高质量的可视化
```yaml
chunk_size: 32           # 更长的轨迹
# 这会生成更多的时间步数据，虽然不影响频域分析，但能看更多帧的动画
```

## 高级用法

### 自定义对比逻辑

如需修改，编辑 `inference_sd_non_connection_shallow.py`:

1. **修改颜色映射** (第335行):
```python
cmap='RdBu_r'  # 改为其他色图如 'viridis', 'plasma'等
```

2. **修改误差计算** (第576行):
```python
error_l2 = np.sqrt(np.mean((trajectory_pred_np - trajectory_gt_np)**2))
# 可改为相对误差等
```

3. **修改频谱计算** (第410行):  
```python
# 当前为1D和2D FFT, 可改为小波变换等其他方法
```

## 验证结果质量

### 检查指标的合理性

```python
# 在Jupyter中验证
import numpy as np

metrics = {}
# 读取 error_metrics.txt 并解析
# 检查 RMSE 是否小于gt的标准差
# 检查 MAE 是否与 RMSE 呈合理比例
```

### 可视化对比
```python
from PIL import Image
import matplotlib.pyplot as plt

gif = Image.open('saved_models/comparison_trajectory.gif')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 提取几帧并显示
# ... (代码省略)
```

## 批量处理多个配置

```bash
#!/bin/bash
for config in config_shallow_*.yml; do
    echo "Processing $config..."
    python scripts/inference_sd_non_connection_shallow.py $config
    
    # 将结果重命名以区分
    SAVE_PATH=$(grep "save_path:" $config | cut -d' ' -f2)
    mv $SAVE_PATH/error_metrics.txt ${SAVE_PATH}/error_metrics_${config%.yml}.txt
done
```

## 性能基准

在典型硬件上的预期运行时间（对于 T_CHUNK=16, N=1024 points）:

- 生成轨迹: ~0.5-2 秒 (取决于采样步数)
- 生成动画: ~10-30 秒 (取决于帧数)
- 频域分析: ~5-15 秒
- 总计: ~15-50 秒

## 与其他推理脚本的差异

本脚本是基于 `inference_sd_non_connection_shallow.py` 的增强版本。
其他推理文件（mesh, MHD等）可以类似地进行增强。

要在其他脚本中添加相同功能，参考以下核心部分：
1. GT加载逻辑 (第57-68行)
2. 对比动画更新函数 (第336-365行)  
3. 频能对比 (第423-565行)
4. 误差指标 (第568-633行)
