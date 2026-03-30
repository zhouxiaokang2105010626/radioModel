"""
aggregation.py - 文件级聚合策略

将窗口级分数聚合为文件级分数
支持多种聚合策略
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def aggregate_max(scores: np.ndarray) -> float:
    """
    最大值聚合
    
    取所有窗口的最大分数作为文件级分数
    适合检测局部异常
    """
    return float(np.max(scores))


def aggregate_mean(scores: np.ndarray) -> float:
    """
    均值聚合
    
    取所有窗口的平均分数
    适合检测全局性异常
    """
    return float(np.mean(scores))


def aggregate_topk_mean(
    scores: np.ndarray,
    topk_ratio: float = 0.1
) -> float:
    """
    Top-K 均值聚合
    
    取分数最高的 top K% 窗口的平均值
    平衡了 max 和 mean 的特点
    
    Args:
        scores: 窗口分数数组
        topk_ratio: Top-K 比例
        
    Returns:
        聚合分数
    """
    k = max(1, int(len(scores) * topk_ratio))
    topk_scores = np.sort(scores)[-k:]
    return float(np.mean(topk_scores))


def aggregate_abnormal_ratio(
    scores: np.ndarray,
    threshold: float
) -> float:
    """
    异常窗口比例聚合
    
    计算分数超过阈值的窗口比例
    
    Args:
        scores: 窗口分数数组
        threshold: 阈值
        
    Returns:
        异常窗口比例
    """
    n_abnormal = np.sum(scores > threshold)
    return float(n_abnormal / len(scores))


def aggregate_file_scores(
    window_results: List[Dict[str, Any]],
    method: str = 'topk_mean',
    threshold: float = None,
    topk_ratio: float = 0.1
) -> List[Dict[str, Any]]:
    """
    将窗口级结果聚合为文件级结果
    
    Args:
        window_results: 窗口级结果列表，每个元素包含:
            - file_path
            - window_index
            - score_total
            - pred_label (可选)
            - true_label (可选)
            - 其他 metadata
        method: 聚合方法
        threshold: 阈值 (abnormal_ratio 方法需要)
        topk_ratio: Top-K 比例
        
    Returns:
        文件级结果列表
    """
    # 按文件分组
    file_windows = defaultdict(list)
    for win in window_results:
        file_path = win['file_path']
        file_windows[file_path].append(win)
    
    file_results = []
    
    for file_path, windows in file_windows.items():
        # 提取分数
        scores = np.array([w['score_total'] for w in windows])
        
        # 聚合
        if method == 'max':
            file_score = aggregate_max(scores)
        elif method == 'mean':
            file_score = aggregate_mean(scores)
        elif method == 'topk_mean':
            file_score = aggregate_topk_mean(scores, topk_ratio)
        elif method == 'abnormal_ratio':
            if threshold is None:
                raise ValueError("abnormal_ratio 方法需要提供 threshold")
            file_score = aggregate_abnormal_ratio(scores, threshold)
        else:
            raise ValueError(f"未知的聚合方法: {method}")
        
        # 获取 metadata (从第一个窗口)
        first_window = windows[0]
        metadata = first_window.get('metadata', {})
        
        # 获取高风险窗口
        high_risk_indices = np.argsort(scores)[-max(1, int(len(scores) * topk_ratio)):][::-1]
        high_risk_windows = [
            {
                'window_index': windows[i]['window_index'],
                'score': float(scores[i])
            }
            for i in high_risk_indices
        ]
        
        # 构建文件结果
        file_result = {
            'file_path': file_path,
            'center_freq_hz': metadata.get('center_freq_hz'),
            'bandwidth_hz': metadata.get('bandwidth_hz'),
            'sample_rate_hz': metadata.get('sample_rate_hz'),
            'n_windows': len(windows),
            'aggregate_method': method,
            'file_score': file_score,
            'score_max': float(np.max(scores)),
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'high_risk_windows': high_risk_windows,
        }
        
        # 如果有真实标签
        if 'true_label' in first_window and first_window['true_label'] is not None:
            file_result['true_label'] = first_window['true_label']
        
        file_results.append(file_result)
    
    return file_results


def add_file_predictions(
    file_results: List[Dict[str, Any]],
    threshold: float
) -> List[Dict[str, Any]]:
    """
    为文件级结果添加预测标签
    
    Args:
        file_results: 文件级结果列表
        threshold: 阈值
        
    Returns:
        添加了预测标签的结果列表
    """
    for result in file_results:
        result['threshold'] = threshold
        result['pred_label'] = 1 if result['file_score'] > threshold else 0
    
    return file_results


def get_hardest_samples(
    file_results: List[Dict[str, Any]],
    n_hardest: int = 10,
    label: int = None
) -> List[Dict[str, Any]]:
    """
    获取最难的样本
    
    Args:
        file_results: 文件级结果列表
        n_hardest: 返回数量
        label: 筛选特定标签 (None=全部, 0=正常, 1=异常)
        
    Returns:
        最难样本列表
    """
    filtered = file_results
    
    if label is not None:
        filtered = [r for r in filtered if r.get('true_label') == label]
    
    if label == 0:
        # 正常样本中分数最高的 (最容易误报)
        sorted_results = sorted(filtered, key=lambda x: x['file_score'], reverse=True)
    else:
        # 异常样本中分数最低的 (最容易漏检)
        sorted_results = sorted(filtered, key=lambda x: x['file_score'])
    
    return sorted_results[:n_hardest]


def group_results_by_metadata(
    file_results: List[Dict[str, Any]],
    group_key: str = 'sample_rate_hz'
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    按 metadata 分组结果
    
    Args:
        file_results: 文件级结果列表
        group_key: 分组键
        
    Returns:
        分组后的结果字典
    """
    groups = defaultdict(list)
    
    for result in file_results:
        key_value = result.get(group_key)
        groups[key_value].append(result)
    
    return dict(groups)


if __name__ == "__main__":
    # 测试聚合函数
    print("测试聚合函数...")
    
    # 模拟窗口结果
    window_results = []
    for i in range(20):
        window_results.append({
            'file_path': 'file_a.IQ' if i < 10 else 'file_b.IQ',
            'window_index': i % 10,
            'score_total': np.random.exponential(0.1),
            'true_label': 0 if i < 10 else 1,
            'metadata': {
                'sample_rate_hz': 64000,
                'center_freq_hz': 161.975e6
            }
        })
    
    # 聚合
    file_results = aggregate_file_scores(
        window_results,
        method='topk_mean',
        topk_ratio=0.2
    )
    
    print(f"文件级结果: {len(file_results)}")
    for r in file_results:
        print(f"  {r['file_path']}: score={r['file_score']:.4f}")
