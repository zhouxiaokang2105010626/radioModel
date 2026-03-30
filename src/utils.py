"""
utils.py - 通用工具函数

包含:
- 配置加载
- 随机种子设置
- Checkpoint 保存/加载
- 日志工具
- 其他辅助函数
"""

import os
import random
import json
import yaml
import numpy as np
import torch
from typing import Dict, Any, Optional
from datetime import datetime
import logging


def setup_logging(log_dir: str, name: str = 'egan') -> logging.Logger:
    """
    设置日志
    
    Args:
        log_dir: 日志目录
        name: 日志名称
        
    Returns:
        Logger 对象
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers
    logger.handlers = []
    
    # 文件 handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    保存配置到 YAML 文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def set_seed(seed: int):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = 'cuda') -> torch.device:
    """
    获取设备
    
    Args:
        device_str: 设备字符串 ('cuda', 'cpu', 'cuda:0', etc.)
        
    Returns:
        torch.device 对象
    """
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        return torch.device('cpu')
    
    return torch.device(device_str)


def save_checkpoint(
    state: Dict[str, Any],
    save_path: str,
    is_best: bool = False
):
    """
    保存 checkpoint
    
    Args:
        state: 状态字典
        save_path: 保存路径
        is_best: 是否是最优模型
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(state, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(state, best_path)
        print(f"保存最优模型: {best_path}")


def load_checkpoint(
    load_path: str,
    model,
    optimizer_d=None,
    optimizer_eg=None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    加载 checkpoint
    
    Args:
        load_path: checkpoint 路径
        model: 模型
        optimizer_d: D 优化器
        optimizer_eg: EG 优化器
        device: 设备
        
    Returns:
        checkpoint 信息字典
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer_d is not None and 'optimizer_d_state_dict' in checkpoint:
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    if optimizer_eg is not None and 'optimizer_eg_state_dict' in checkpoint:
        optimizer_eg.load_state_dict(checkpoint['optimizer_eg_state_dict'])
    
    print(f"加载 checkpoint: {load_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型参数量
    
    Args:
        model: 模型
        
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_json(data: Dict[str, Any], save_path: str):
    """
    保存 JSON 文件
    
    Args:
        data: 数据字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(load_path: str) -> Dict[str, Any]:
    """
    加载 JSON 文件
    
    Args:
        load_path: 文件路径
        
    Returns:
        数据字典
    """
    with open(load_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class EarlyStopping:
    """
    早停机制
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: 耐心值
            min_delta: 最小改进量
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前分数
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class AverageMeter:
    """
    平均值计量器
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def get_output_paths(config: dict) -> Dict[str, str]:
    """
    获取输出路径
    
    Args:
        config: 配置字典
        
    Returns:
        路径字典
    """
    output_dir = config.get('output', {}).get('output_dir', 'outputs')
    
    paths = {
        'output_dir': output_dir,
        'checkpoints_dir': os.path.join(output_dir, 'checkpoints'),
        'logs_dir': os.path.join(output_dir, 'logs'),
        'figures_dir': os.path.join(output_dir, 'figures'),
        'threshold_file': os.path.join(output_dir, 'threshold.json'),
        'metrics_file': os.path.join(output_dir, 'metrics.json'),
        'predictions_window_file': os.path.join(output_dir, 'predictions_window.csv'),
        'predictions_file_file': os.path.join(output_dir, 'predictions_file.csv'),
    }
    
    # 创建目录
    for key, path in paths.items():
        if key.endswith('_dir'):
            ensure_dir(path)
    
    return paths


if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    
    # 测试随机种子
    set_seed(42)
    print(f"随机数测试: {random.random():.4f}")
    
    # 测试设备
    device = get_device('cuda')
    print(f"使用设备: {device}")
    
    # 测试早停
    early_stop = EarlyStopping(patience=3, mode='min')
    scores = [1.0, 0.9, 0.8, 0.81, 0.82, 0.83, 0.84]
    for i, score in enumerate(scores):
        should_stop = early_stop(score)
        print(f"Epoch {i+1}: score={score}, should_stop={should_stop}")
        if should_stop:
            break
    
    print("测试完成!")
