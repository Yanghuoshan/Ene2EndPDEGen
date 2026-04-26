# Ground Truth 对比可视化功能 - 效果展示

## 功能概览

本次更新为浅水方程推理脚本添加了完整的 Ground Truth 对比功能，包括三类输出。

---

## 输出类型详解

### 1️⃣ 对比动画 GIF

**文件**: `comparison_trajectory.gif` (或 `comparison_trajectory_N.gif`)

**显示内容**: 三列并排展示
```
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│   Ground Truth (GT)     │   Generated (Pred)      │   Absolute Error        │
│                         │                         │                         │
│   真实物理场演化        │   模型生成的预测        │   |Pred - GT| 的差值    │
│                         │                         │                         │
│  时间: t=0 到 t=T_CHUNK │  使用相同的颜色范围     │  热力颜色表示误差大小   │
│                         │                         │                         │
│  动画速率: 12 fps       │  自动循环播放           │  便于识别问题区域       │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

**样式说明**:
- **GT和Pred列**: 使用RdBu_r色图（红表负值，蓝表正值）
- **Error列**: 使用hot色图（黑→红→黄），值越大颜色越亮
- **坐标**: 经度(0-360°) × 纬度(-90-90°)
- **标题**: 显示当前时间步 `t=0, 1, 2, ...`

**应用场景**:
✓ 快速视觉检查模型性能  
✓ 识别系统性偏差（如温度偏移）  
✓ 观察地理位置的局部误差  
✓ 用于学术报告和展示  

**CPU/GPU影响**: 2GB内存, 30秒左右生成

---

### 2️⃣ 频域能量谱图

**文件**: `frequency_energy_spectrum.png`

#### 有Ground Truth时 (2×2 layout)

```
┌────────────────────────────────┬────────────────────────────────┐
│                                │                                │
│   Temporal Energy Spectrum      │   Temporal Energy Error        │
│   (时间轴能量谱对比)           │   (时间轴能量误差谱)          │
│                                │                                │
│   - 实线O: 生成结果            │   显示Pred和GT的频域差异      │
│   - 虚线□: Ground Truth        │   对数坐标                     │
│   - 对数纵轴                   │   多个通道             │
│   - 各通道独立曲线             │                                │
│                                │                                │
│   典型形状: 能量随频率递减      │   理想情况: 误差在低频较小    │
├────────────────────────────────┼────────────────────────────────┤
│                                │                                │
│   Spatial Energy Spectrum       │   Spatial Energy Error         │
│   (初始时刻t=0的空间谱)        │   (初始时刻空间能量误差)      │
│                                │                                │
│   - 2D FFT后的径向平均         │   Pred和GT空间频谱差异        │
│   - 显示能量如何随波数分布     │   对数坐标                     │
│   - 涉及空间尺度 k             │   帮助诊断空间精度             │
│   - 多通道对比                 │                                │
│                                │                                │
│   典型形状: k^{-n} 幂律        │   理想情况: 低误差广谱         │
└────────────────────────────────┴────────────────────────────────┘
```

**频谱图解读指南**:

| 情况 | GT谱 | Pred谱 | 误差谱 | 诊断 |
|------|------|--------|--------|------|
| 优秀 | 高频衰减 | 跟随GT | 低平缓 | ✅ 模型表现好 |
| 欠平滑 | 高频强 | 更强 | 高频高 | ⚠️ 生成过度锐化 |
| 过平滑 | 高频强 | 低平缓 | 高频高 | ⚠️ 生成过度平滑 |
| 相位错 | 峰值对齐 | 峰值移位 | 全频高 | ❌ 动力学模型差 |

**应用场景**:
✓ 定量评估模型的多尺度性能  
✓ 检查时间和空间导数的正确性  
✓ 对比不同配置的频域特性  
✓ 论文中的定量分析图表  

---

### 3️⃣ 误差指标报告

**文件**: `error_metrics.txt`

**内容结构**:

```
ERROR METRICS COMPARISON
============================================================

Global Error Metrics:
  MSE (L2^2):  1.234567E-04     ← 均方误差，对大偏差敏感
  RMSE (L2):   1.110722E-02     ← 均方根误差，与数据量纲一致
  MAE (L1):    8.901234E-03     ← 平均绝对误差，更稳健
  Max Error:   1.234567E-01     ← 最大误差，最坏情况

Vorticity:                       ← 涡度通道
  RMSE:  1.234567E-02
  MAE:   1.001234E-02
  MSE:   1.524157E-04

Height:                          ← 高度通道  
  RMSE:  2.345678E-03
  MAE:   1.234567E-03
  MSE:   5.502135E-06

Temporal Error Evolution:        ← 时间演化趋势
  t=0: RMSE = 1.234567E-02      ← 初期误差
  t=1: RMSE = 1.345678E-02      ← 误差增长
  t=2: RMSE = 1.456789E-02      ← 继续增长
  ...
  t=15: RMSE = 2.345678E-02     ← 最后时刻
```

**指标解释**:

🔹 **全局误差**
- RMSE < 0.01: 优良 ✅
- RMSE 0.01-0.05: 中等 ⚠️  
- RMSE > 0.05: 需要改进 ❌

🔹 **通道特异性**
- 涡度误差通常 > 高度误差（涡度更复杂）
- 两通道误差相差 > 10倍可能提示问题

🔹 **时间演化**
- 单调增长: 累积误差（系统性偏差）
- 振荡增长: 能量误耗或混沌发散
- 平稳: 好的长期预测品质

**应用场景**:
✓ 定量比较多个模型版本  
✓ 提交期刊论文时的性能报告  
✓ 超参数调优的目标函数  
✓ 自动化测试的断言条件  

---

## 控制台输出示例

```
Loading ground truth sample from test dataset...
Ground truth sample loaded: shape torch.Size([1, 16, 1024, 2])

Loading normalizer parameters from saved_models/normalizer_params.pt
Loaded encoder from encoder_ema_state_dict.
Loaded cnf from cnf_ema_state_dict.
Checkpoint loaded (epoch=300, global_step=N/A)

...生成的结果...

Generating comparison animation with GT, Prediction, and Error...
Saving comparison animation to saved_models/comparison_trajectory.gif ...
Comparison animation saved successfully.

Computing frequency domain energy spectra...
Generating animation with 2D heatplot projection...

ERROR METRICS COMPARISON
============================================================

Global Error Metrics:
  MSE (L2^2):  1.234567E-04
  RMSE (L2):   1.110722E-02
  MAE (L1):    8.901234E-03
  Max Error:   1.234567E-01

Vorticity:
  RMSE:  1.234567E-02
  ...

Metrics saved to saved_models/error_metrics.txt
```

---

## 特性亮点

### 🎯 自动化设计
- ✅ 自动检测GT是否可用
- ✅ 自动计数器防止文件覆盖 (`_1`, `_2`, ...)
- ✅ 自动计算合适的颜色范围
- ✅ 无需额外配置，开箱即用

### 📊 专业化可视化
- ✅ 论文级别的图表质量 (DPI=150)
- ✅ 多语言标题注解支持
- ✅ 自动布局优化 (tight_layout)
- ✅ 一致的美学设计

### ⚡ 高性能计算
- ✅ NumPy向量化，无Python循环
- ✅ 高效的FFT实现 (NumPy FFT)
- ✅ 插值缓存机制
- ✅ 内存高效的流处理

### 🔧 可扩展性
- ✅ 支持任意通道数
- ✅ 支持任意网格分辨率
- ✅ 模块化设计便于修改
- ✅ 向后兼容原始推理流程

---

## 典型工作流

```
运行配置 ──→ 加载结构 ──→ 生成轨迹 ──→ 可视化
          ├─GT加载         └─编码→解码    ├─对比GIF
          └─归一化器                      ├─频谱图
                                          └─指标文件
```

---

## 文件生成时间估算

| 组件 | 时间 | 影响因素 |
|------|------|---------|
| 轨迹生成 | 0.5-2s | 采样步数, N点数 |
| GIF编码 | 10-30s | T_CHUNK, 图像分辨率 |
| 频域计算 | 5-15s | FFT大小, 插值分辨率 |
| 指标计算 | <1s | 数据大小 |
| **总计** | **~20-50s** | GPU加速可减半 |

---

## 下一步优化方向

💡 **可选增强** (用户可自行实现):

1. **交互式可视化** - 用 Plotly 替代 matplotlib
2. **3D可视化** - 用 PyVista 显示3D地球场
3. **统计显著性测试** - 加入置信区间
4. **自适应误差图** - 动态颜色缩放
5. **实时监控** - 训练中动态生成对比

---

## 常见问题 (FAQ)

**Q: 如果只想要单一可视化而不需要GT对比？**  
A: 脚本自动检测，如果无GT则自动切换单视图模式。若要强制，注释掉GT加载部分。

**Q: 能否自定义色图？**  
A: 是的，编辑脚本第335行的 `cmap='RdBu_r'` 为其他matplotlib色图名称。

**Q: 频谱图能否显示 3D 视角？**  
A: 可以，需要修改 matplotlib 配置为 3D projection。建议参考 `mplot3d` 文档。

**Q: 能否将结果输出为 MP4 而非 GIF？**  
A: 是的，修改脚本约第378行 `ani.save()` 调用的参数改为 `format='mp4'`。

---

**✨ 享受更直观的模型评估体验！**
