"""
dataset.py - PyTorch Dataset 实现

主要功能:
1. Window-level dataset 用于训练
2. File-level dataset 用于评估
3. 支持 normal-only 训练模式
4. 支持标签缺失的推理模式
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import warnings
from tqdm import tqdm


class SpectrogramDataset(Dataset):
    """
    Spectrogram 数据集
    
    用于加载预处理后的 spectrogram 数据
    支持 window-level 训练和评估
    """
    
    def __init__(
        self,
        file_paths: List[str],
        config: dict,
        label: Optional[int] = None,  # 0=normal, 1=anomaly, None=unknown
        preload: bool = True,
        transform=None
    ):
        """
        Args:
            file_paths: IQ 文件路径列表
            config: 配置字典
            label: 标签 (0=正常, 1=异常, None=未知)
            preload: 是否预加载所有数据到内存
            transform: 可选的数据增强
        """
        self.file_paths = file_paths
        self.config = config
        self.label = label
        self.preload = preload
        self.transform = transform
        
        # 存储所有窗口信息
        self.windows: List[Dict[str, Any]] = []
        self.spectrograms: List[np.ndarray] = []
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载所有文件并提取窗口"""
        from .preprocess import process_file_to_spectrograms
        
        iq_config = self.config.get('iq', {})
        
        for file_path in tqdm(self.file_paths, desc="Loading files"):
            try:
                specs, win_infos = process_file_to_spectrograms(
                    file_path, self.config, iq_config
                )
                
                for spec, win_info in zip(specs, win_infos):
                    win_info['label'] = self.label
                    self.windows.append(win_info)
                    
                    if self.preload:
                        self.spectrograms.append(spec)
                        
            except Exception as e:
                warnings.warn(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        print(f"加载完成: {len(self.windows)} 个窗口，来自 {len(self.file_paths)} 个文件")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        Returns:
            字典包含:
            - spectrogram: (1, H, W) tensor
            - label: 标签 (可能为 -1 表示未知)
            - window_info: 窗口元数据
        """
        win_info = self.windows[idx]
        
        if self.preload:
            spec = self.spectrograms[idx]
        else:
            # 动态加载
            from .preprocess import process_file_to_spectrograms
            specs, _ = process_file_to_spectrograms(
                win_info['file_path'],
                self.config,
                self.config.get('iq', {})
            )
            spec = specs[win_info['window_index']]
        
        # 转为 tensor
        spec_tensor = torch.from_numpy(spec).float()
        
        # 应用变换
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
        
        # 标签处理
        label = win_info['label'] if win_info['label'] is not None else -1
        
        return {
            'spectrogram': spec_tensor,
            'label': label,
            'window_info': win_info
        }


class FileDataset(Dataset):
    """
    文件级数据集
    
    用于文件级评估，每次返回一个文件的所有窗口
    """
    
    def __init__(
        self,
        file_paths: List[str],
        config: dict,
        label: Optional[int] = None
    ):
        """
        Args:
            file_paths: IQ 文件路径列表
            config: 配置字典
            label: 标签 (0=正常, 1=异常, None=未知)
        """
        self.file_paths = file_paths
        self.config = config
        self.label = label
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个文件的所有窗口
        
        Returns:
            字典包含:
            - spectrograms: (N, 1, H, W) tensor
            - label: 标签
            - file_path: 文件路径
            - window_infos: 窗口元数据列表
        """
        from .preprocess import process_file_to_spectrograms
        
        file_path = self.file_paths[idx]
        iq_config = self.config.get('iq', {})
        
        specs, win_infos = process_file_to_spectrograms(
            file_path, self.config, iq_config
        )
        
        # 转为 tensor
        spec_tensor = torch.from_numpy(np.stack(specs, axis=0)).float()
        
        label = self.label if self.label is not None else -1
        
        return {
            'spectrograms': spec_tensor,
            'label': label,
            'file_path': file_path,
            'window_infos': win_infos
        }


def create_datasets(config: dict) -> Dict[str, Dataset]:
    """
    根据配置创建所有数据集
    
    Args:
        config: 配置字典
        
    Returns:
        字典包含各数据集:
        - train: 训练集 (仅正常样本)
        - val_normal: 验证集正常样本
        - val_anomaly: 验证集异常样本 (可能不存在)
        - test_normal: 测试集正常样本
        - test_anomaly: 测试集异常样本 (可能不存在)
    """
    from .io_utils import get_iq_files, filter_files_by_metadata
    
    data_config = config.get('data', {})
    root_dir = data_config.get('root_dir', 'data')
    
    # 构建目录路径
    dirs = {
        'train_normal': os.path.join(root_dir, data_config.get('train_normal_dir', 'train/normal')),
        'val_normal': os.path.join(root_dir, data_config.get('val_normal_dir', 'val/normal')),
        'val_anomaly': os.path.join(root_dir, data_config.get('val_anomaly_dir', 'val/anomaly')),
        'test_normal': os.path.join(root_dir, data_config.get('test_normal_dir', 'test/normal')),
        'test_anomaly': os.path.join(root_dir, data_config.get('test_anomaly_dir', 'test/anomaly')),
    }
    
    # 过滤参数
    filter_params = {
        'filter_center_freq': data_config.get('filter_center_freq'),
        'filter_sample_rate': data_config.get('filter_sample_rate'),
        'filter_bandwidth': data_config.get('filter_bandwidth'),
    }
    
    datasets = {}
    
    # 训练集
    train_files = get_iq_files(dirs['train_normal'])
    train_files = filter_files_by_metadata(train_files, **filter_params)
    if train_files:
        datasets['train'] = SpectrogramDataset(train_files, config, label=0)
    else:
        warnings.warn(f"训练集目录为空或不存在: {dirs['train_normal']}")
        datasets['train'] = None
    
    # 验证集 - 正常样本
    val_normal_files = get_iq_files(dirs['val_normal'])
    val_normal_files = filter_files_by_metadata(val_normal_files, **filter_params)
    if val_normal_files:
        datasets['val_normal'] = SpectrogramDataset(val_normal_files, config, label=0)
    else:
        datasets['val_normal'] = None
    
    # 验证集 - 异常样本
    val_anomaly_files = get_iq_files(dirs['val_anomaly'])
    val_anomaly_files = filter_files_by_metadata(val_anomaly_files, **filter_params)
    if val_anomaly_files:
        datasets['val_anomaly'] = SpectrogramDataset(val_anomaly_files, config, label=1)
    else:
        datasets['val_anomaly'] = None
    
    # 测试集 - 正常样本
    test_normal_files = get_iq_files(dirs['test_normal'])
    test_normal_files = filter_files_by_metadata(test_normal_files, **filter_params)
    if test_normal_files:
        datasets['test_normal'] = SpectrogramDataset(test_normal_files, config, label=0)
    else:
        datasets['test_normal'] = None
    
    # 测试集 - 异常样本
    test_anomaly_files = get_iq_files(dirs['test_anomaly'])
    test_anomaly_files = filter_files_by_metadata(test_anomaly_files, **filter_params)
    if test_anomaly_files:
        datasets['test_anomaly'] = SpectrogramDataset(test_anomaly_files, config, label=1)
    else:
        datasets['test_anomaly'] = None
    
    return datasets


def create_dataloaders(
    datasets: Dict[str, Dataset],
    config: dict
) -> Dict[str, DataLoader]:
    """
    创建 DataLoader
    
    Args:
        datasets: 数据集字典
        config: 配置字典
        
    Returns:
        DataLoader 字典
    """
    train_config = config.get('train', {})
    batch_size = train_config.get('batch_size', 32)
    num_workers = train_config.get('num_workers', 4)
    
    dataloaders = {}
    
    for name, dataset in datasets.items():
        if dataset is None:
            dataloaders[name] = None
            continue
        
        shuffle = 'train' in name
        
        # Window dataset 需要自定义 collate_fn
        dataloaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_window_batch
        )
    
    return dataloaders


def collate_window_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义批次整理函数
    
    Args:
        batch: 样本列表
        
    Returns:
        整理后的批次
    """
    spectrograms = torch.stack([item['spectrogram'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    window_infos = [item['window_info'] for item in batch]
    
    return {
        'spectrogram': spectrograms,
        'label': labels,
        'window_info': window_infos
    }


def get_combined_val_dataset(datasets: Dict[str, Dataset]) -> Optional[Dataset]:
    """
    合并验证集的正常和异常样本
    
    Args:
        datasets: 数据集字典
        
    Returns:
        合并后的数据集
    """
    val_normal = datasets.get('val_normal')
    val_anomaly = datasets.get('val_anomaly')
    
    if val_normal is None and val_anomaly is None:
        return None
    
    if val_anomaly is None:
        return val_normal
    
    if val_normal is None:
        return val_anomaly
    
    # 合并两个数据集
    from torch.utils.data import ConcatDataset
    return ConcatDataset([val_normal, val_anomaly])


def get_combined_test_dataset(datasets: Dict[str, Dataset]) -> Optional[Dataset]:
    """
    合并测试集的正常和异常样本
    
    Args:
        datasets: 数据集字典
        
    Returns:
        合并后的数据集
    """
    test_normal = datasets.get('test_normal')
    test_anomaly = datasets.get('test_anomaly')
    
    if test_normal is None and test_anomaly is None:
        return None
    
    if test_anomaly is None:
        return test_normal
    
    if test_normal is None:
        return test_anomaly
    
    from torch.utils.data import ConcatDataset
    return ConcatDataset([test_normal, test_anomaly])


if __name__ == "__main__":
    import yaml
    import sys
    
    # 测试数据集加载
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("创建数据集...")
        datasets = create_datasets(config)
        
        for name, ds in datasets.items():
            if ds is not None:
                print(f"  {name}: {len(ds)} 样本")
            else:
                print(f"  {name}: 空")
        
        # 测试加载一个批次
        if datasets['train'] is not None:
            dataloaders = create_dataloaders(datasets, config)
            for batch in dataloaders['train']:
                print(f"批次形状: {batch['spectrogram'].shape}")
                break
    else:
        print("用法: python -m src.dataset configs/default.yaml")
