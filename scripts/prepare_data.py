"""
prepare_data.py - 数据目录准备脚本

用于:
1. 创建标准数据目录结构
2. 将 IQ 文件分配到 train/val/test
3. 预览数据分布
"""

import os
import sys
import argparse
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_utils import (
    get_iq_files, parse_iq_filename_metadata, 
    read_iq_file, sanity_check_iq
)


def create_dataset_structure(root_dir: str):
    """
    创建标准数据目录结构
    
    dataset/
    ├── train/
    │   └── normal/
    ├── val/
    │   ├── normal/
    │   └── anomaly/
    └── test/
        ├── normal/
        └── anomaly/
    """
    dirs = [
        'train/normal',
        'val/normal',
        'val/anomaly',
        'test/normal',
        'test/anomaly',
    ]
    
    for d in dirs:
        path = os.path.join(root_dir, d)
        os.makedirs(path, exist_ok=True)
        print(f"创建目录: {path}")
    
    print(f"\n数据目录结构已创建: {root_dir}")


def list_data_files(root_dir: str) -> Dict[str, List[str]]:
    """
    列出数据目录中的文件
    
    Returns:
        各目录的文件列表字典
    """
    result = {}
    
    for split in ['train', 'val', 'test']:
        for label in ['normal', 'anomaly']:
            dir_path = os.path.join(root_dir, split, label)
            if os.path.exists(dir_path):
                files = get_iq_files(dir_path)
                key = f"{split}/{label}"
                result[key] = files
    
    return result


def print_data_summary(file_dict: Dict[str, List[str]]):
    """打印数据摘要"""
    print("\n=== 数据摘要 ===")
    
    total = 0
    for key, files in sorted(file_dict.items()):
        count = len(files)
        total += count
        print(f"  {key}: {count} 个文件")
    
    print(f"  总计: {total} 个文件")


def split_files(
    source_dir: str,
    dest_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    copy: bool = True,
    seed: int = 42
):
    """
    将源目录中的文件分割到 train/val/test
    
    Args:
        source_dir: 源目录
        dest_dir: 目标目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        copy: True=复制, False=移动
        seed: 随机种子
    """
    random.seed(seed)
    
    # 获取所有文件
    files = get_iq_files(source_dir)
    if not files:
        print(f"源目录为空: {source_dir}")
        return
    
    print(f"找到 {len(files)} 个文件")
    
    # 随机打乱
    random.shuffle(files)
    
    # 计算分割点
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    print(f"分割: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    
    # 创建目录结构
    create_dataset_structure(dest_dir)
    
    # 移动/复制文件
    def process_files(files, dest_subdir):
        for f in files:
            src = f
            dst = os.path.join(dest_dir, dest_subdir, os.path.basename(f))
            if copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
    
    process_files(train_files, 'train/normal')
    process_files(val_files, 'val/normal')
    process_files(test_files, 'test/normal')
    
    print(f"\n文件已{'复制' if copy else '移动'}到 {dest_dir}")


def preview_file(file_path: str, save_dir: str = None):
    """
    预览单个 IQ 文件
    
    Args:
        file_path: 文件路径
        save_dir: 图片保存目录
    """
    print(f"\n=== 预览文件: {file_path} ===")
    
    # 解析 metadata
    metadata = parse_iq_filename_metadata(file_path)
    print("\n文件元数据:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    
    # 读取数据
    data = read_iq_file(file_path, dtype='int16', interleaved=True, little_endian=True)
    print(f"\n数据信息:")
    print(f"  样本数: {len(data)}")
    print(f"  时长: {len(data) / metadata.get('sample_rate_hz', 1):.2f} 秒")
    print(f"  数据类型: {data.dtype}")
    
    # Sanity check
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{Path(file_path).stem}_sanity.png")
    
    sanity_check_iq(data, metadata, save_path=save_path, show=save_dir is None)


def analyze_metadata(files: List[str]):
    """
    分析文件元数据分布
    
    Args:
        files: 文件路径列表
    """
    from collections import Counter
    
    center_freqs = Counter()
    bandwidths = Counter()
    sample_rates = Counter()
    
    for f in files:
        metadata = parse_iq_filename_metadata(f)
        center_freqs[metadata.get('center_freq_hz')] += 1
        bandwidths[metadata.get('bandwidth_hz')] += 1
        sample_rates[metadata.get('sample_rate_hz')] += 1
    
    print("\n=== 元数据分布 ===")
    
    print("\n中心频率:")
    for freq, count in sorted(center_freqs.items()):
        print(f"  {freq} Hz: {count} 个文件")
    
    print("\n带宽:")
    for bw, count in sorted(bandwidths.items()):
        print(f"  {bw} Hz: {count} 个文件")
    
    print("\n采样率:")
    for sr, count in sorted(sample_rates.items()):
        print(f"  {sr} Hz: {count} 个文件")


def main():
    parser = argparse.ArgumentParser(description='数据准备工具')
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 创建目录结构
    create_parser = subparsers.add_parser('create', help='创建数据目录结构')
    create_parser.add_argument('--root', type=str, default='dataset', help='数据根目录')
    
    # 分割数据
    split_parser = subparsers.add_parser('split', help='分割数据到 train/val/test')
    split_parser.add_argument('--source', type=str, required=True, help='源目录')
    split_parser.add_argument('--dest', type=str, default='dataset', help='目标目录')
    split_parser.add_argument('--train-ratio', type=float, default=0.7)
    split_parser.add_argument('--val-ratio', type=float, default=0.15)
    split_parser.add_argument('--test-ratio', type=float, default=0.15)
    split_parser.add_argument('--copy', action='store_true', help='复制而非移动')
    
    # 列出文件
    list_parser = subparsers.add_parser('list', help='列出数据目录中的文件')
    list_parser.add_argument('--root', type=str, default='dataset', help='数据根目录')
    
    # 预览文件
    preview_parser = subparsers.add_parser('preview', help='预览 IQ 文件')
    preview_parser.add_argument('--file', type=str, required=True, help='文件路径')
    preview_parser.add_argument('--save-dir', type=str, default=None, help='图片保存目录')
    
    # 分析元数据
    analyze_parser = subparsers.add_parser('analyze', help='分析文件元数据')
    analyze_parser.add_argument('--dir', type=str, required=True, help='目录路径')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        create_dataset_structure(args.root)
    
    elif args.command == 'split':
        split_files(
            args.source, args.dest,
            args.train_ratio, args.val_ratio, args.test_ratio,
            copy=args.copy
        )
    
    elif args.command == 'list':
        file_dict = list_data_files(args.root)
        print_data_summary(file_dict)
    
    elif args.command == 'preview':
        preview_file(args.file, args.save_dir)
    
    elif args.command == 'analyze':
        files = get_iq_files(args.dir)
        analyze_metadata(files)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
