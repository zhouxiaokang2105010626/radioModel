# src/__init__.py
# E-GAN Radio Anomaly Detection Package

"""
E-GAN 无线电信号异常检测包

基于论文《A Radio Anomaly Detection Algorithm Based on Modified Generative Adversarial Network》
的工程化实现。

主要模块:
- io_utils: IQ 文件读写
- preprocess: 预处理和 STFT
- dataset: PyTorch Dataset
- model: E-GAN 模型
- losses: 损失函数
- thresholds: 阈值计算
- aggregation: 文件级聚合
- metrics: 评估指标
- visualize: 可视化
- utils: 工具函数
- train: 训练脚本
- eval: 评估脚本
- infer: 推理脚本
"""

__version__ = "1.0.0"
__author__ = "E-GAN Radio Project"
