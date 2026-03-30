# E-GAN Radio Anomaly Detection

基于 E-GAN (Encoder-GAN) 的无线电信号异常检测系统。

> **声明**: 这是遵循论文《A Radio Anomaly Detection Algorithm Based on Modified Generative Adversarial Network》核心思想的**工程化实现**，不是作者官方逐层复现版本。模型结构和部分细节根据工程需求做了适配。

## 1. 项目简介

本项目实现了一个针对 IQ 原始数据的无监督异常检测系统：
- 输入：`.IQ` 格式的原始二进制信号文件
- 处理：IQ → 切窗 → STFT → Spectrogram → E-GAN 重构
- 输出：异常分数、异常/正常判断、异常定位热力图

## 2. 方法概述

### 2.1 核心流程

```
IQ 文件 → 切窗 → STFT → 2D Spectrogram → E-GAN → 异常检测
```

### 2.2 E-GAN 结构

- **Encoder (E)**: 将 spectrogram 编码为 latent vector z
- **Generator (G)**: 从 z 重构 spectrogram
- **Discriminator (D)**: 区分真实 spectrogram 和重构 spectrogram

### 2.3 训练思路

- 训练集**仅包含正常样本**
- E 和 G 学习正常 spectrogram 的重构
- D 学习区分真实 vs 重构
- 损失函数: Adversarial Loss + Reconstruction Loss (L1)

### 2.4 检测思路

对任意 spectrogram y:
```
y_hat = G(E(y))
AR(y) = mean(|y - y_hat|)           # 重构误差分数
AD(y) = |sigmoid(D(y)) - sigmoid(D(y_hat))|  # 判别器分数
A(y) = (1-λ) * AR(y) + λ * AD(y)    # 综合异常分数
```

### 2.5 阈值确定

通过正常样本的 anomaly score 分布，结合目标误报率 (PFA) 计算阈值:
```
eta = quantile(normal_scores, 1 - target_pfa)
```

## 3. 数据目录准备

### 3.1 目录结构

```
dataset/
├── train/
│   └── normal/          # 训练集 - 仅正常样本
│       ├── file1.IQ
│       └── file2.IQ
├── val/
│   ├── normal/          # 验证集 - 正常样本
│   └── anomaly/         # 验证集 - 异常样本 (可选)
└── test/
    ├── normal/          # 测试集 - 正常样本
    └── anomaly/         # 测试集 - 异常样本 (可选)
```

### 3.2 IQ 文件格式

- 格式: int16, interleaved I,Q,I,Q,...
- 字节序: little-endian
- 文件名示例: `161.975MHz_30kHz_64000_16_20250729082303.IQ`
  - 中心频率: 161.975MHz
  - 带宽: 30kHz
  - 采样率: 64000 Hz
  - 位深: 16 bit
  - 时间戳: YYYYMMDDhhmmss

## 4. 快速开始

### 4.1 安装依赖

```bash
pip install -r requirements.txt
```

### 4.2 准备数据

将你的 IQ 文件按上述目录结构组织。

### 4.3 修改配置

编辑 `configs/default.yaml`，设置数据路径:

```yaml
data:
  root_dir: "path/to/dataset"
  train_normal_dir: "train/normal"
  ...
```

## 5. 训练模型

```bash
python -m src.train --config configs/default.yaml
```

训练过程中会自动:
- 保存 checkpoint 到 `outputs/checkpoints/`
- 记录训练日志到 `outputs/logs/`
- 保存训练曲线到 `outputs/figures/`

### 5.1 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 100 | 训练轮数 |
| batch_size | 32 | 批大小 |
| learning_rate | 0.0002 | 学习率 |
| adv_weight | 1.0 | 对抗损失权重 |
| recon_weight | 50.0 | 重构损失权重 |

## 6. 校准阈值

阈值校准使用正常样本的 anomaly score 分布:

```bash
python -m src.eval --config configs/default.yaml --calibrate
```

这将:
1. 加载训练好的模型
2. 在正常验证集上计算所有 score
3. 根据 `target_pfa` 计算阈值 eta
4. 保存到 `outputs/threshold.json`

## 7. 评估模型

```bash
python -m src.eval --config configs/default.yaml
```

评估将输出:
- Window 级指标: ROC-AUC, PR-AUC, Accuracy, F1 等
- File 级指标: 聚合后的分类指标
- CSV 结果文件: `predictions_window.csv`, `predictions_file.csv`
- 可视化图表: ROC 曲线, PR 曲线, Score 分布直方图

## 8. 推理单个文件

```bash
python -m src.infer --config configs/default.yaml --input path/to/file.IQ
```

输出:
- 每个窗口的 anomaly score
- 文件级聚合分数
- 正常/异常判断
- 异常热力图保存到 `outputs/figures/`

### 8.1 查看异常定位热力图

推理时会自动生成:
- `localization_xxx.png`: |y - y_hat| 热力图，显示异常在时频图上的位置
- `recon_xxx.png`: 原图 vs 重构图对比

## 9. Sanity Check (数据检查)

验证 IQ 文件是否正确读取:

```bash
python -m src.preprocess --config configs/default.yaml --preview path/to/file.IQ
```

这将显示:
- I/Q 波形
- 幅度直方图
- PSD 功率谱密度
- Spectrogram 预览

## 10. 如果 IQ 读取不对

### 10.1 检查配置

编辑 `configs/default.yaml`:

```yaml
iq:
  dtype: "int16"           # 数据类型
  interleaved: true        # I,Q 交织存储
  little_endian: true      # 字节序
  bit_depth: 16            # 位深
  remove_dc: true          # 是否去除直流分量
  normalize_mode: "rms"    # 归一化方式: "rms" 或 "max"
```

### 10.2 常见问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 数据全是噪声 | 字节序错误 | 尝试 `little_endian: false` |
| 数据范围异常 | dtype 错误 | 检查是 int16/float32/int8 |
| 只有 I 或 Q | 非交织存储 | 设置 `interleaved: false` |
| spectrogram 全黑/全白 | 归一化问题 | 调整 `normalize_mode` |

### 10.3 修改 IO 函数

核心读取函数在 `src/io_utils.py`:

```python
def read_iq_file(path, dtype="int16", interleaved=True, little_endian=True):
    ...
```

## 11. 已知局限

1. **单一采样率**: 当前每次训练/评估假设同一采样率。混合采样率需要分组处理
2. **窗口大小固定**: 不同带宽信号可能需要不同窗口大小
3. **阈值敏感**: 阈值依赖正常样本分布，需要足够多正常样本
4. **无监督学习**: 无法针对特定异常类型优化

## 12. 后续可扩展方向

1. **多采样率支持**: 按 metadata 分组训练多个模型
2. **自适应窗口**: 根据带宽自动调整窗口大小
3. **更复杂模型**: VAE-GAN, Transformer 等
4. **增量学习**: 支持在线更新模型
5. **异常分类**: 区分不同类型的异常
6. **集成部署**: ONNX 导出, 边缘设备部署

## 13. 项目结构

```
radioModel/
├── configs/
│   └── default.yaml      # 配置文件
├── src/
│   ├── io_utils.py       # IQ 文件读写
│   ├── preprocess.py     # 预处理和 STFT
│   ├── dataset.py        # PyTorch Dataset
│   ├── model.py          # E-GAN 模型
│   ├── losses.py         # 损失函数
│   ├── metrics.py        # 评估指标
│   ├── visualize.py      # 可视化
│   ├── utils.py          # 工具函数
│   ├── thresholds.py     # 阈值计算
│   ├── aggregation.py    # 文件级聚合
│   ├── train.py          # 训练脚本
│   ├── eval.py           # 评估脚本
│   └── infer.py          # 推理脚本
├── outputs/
│   ├── checkpoints/      # 模型权重
│   ├── logs/             # 训练日志
│   └── figures/          # 可视化图表
├── requirements.txt
└── README.md
```

## 14. 引用

如果本项目对你有帮助，请引用原论文:

```
@article{radio_anomaly_egan,
  title={A Radio Anomaly Detection Algorithm Based on Modified Generative Adversarial Network},
  ...
}
```

## 15. License

MIT License
