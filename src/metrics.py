"""
metrics.py - 评估指标计算

包含:
- 分类指标 (AUC, F1, Precision, Recall 等)
- 阈值相关指标 (FPR, TPR)
- 混淆矩阵
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
import warnings


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签 (0=正常, 1=异常)
        y_pred: 预测标签
        y_score: 预测分数 (用于 AUC)
        
    Returns:
        指标字典
    """
    metrics = {}
    
    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 处理只有一类的情况
    unique_labels = np.unique(y_true)
    
    if len(unique_labels) > 1:
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC 指标
        if y_score is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_score)
                metrics['pr_auc'] = average_precision_score(y_true, y_score)
            except Exception as e:
                warnings.warn(f"计算 AUC 失败: {e}")
                metrics['roc_auc'] = np.nan
                metrics['pr_auc'] = np.nan
    else:
        warnings.warn("测试集只有一类样本，部分指标无法计算")
        metrics['precision'] = np.nan
        metrics['recall'] = np.nan
        metrics['f1'] = np.nan
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics['confusion_matrix'] = cm.tolist()
    
    # TN, FP, FN, TP
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        
        # FPR 和 TPR
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return metrics


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    计算指定阈值下的指标
    
    Args:
        y_true: 真实标签
        y_score: 预测分数
        threshold: 阈值
        
    Returns:
        指标字典
    """
    y_pred = (y_score > threshold).astype(int)
    
    metrics = compute_classification_metrics(y_true, y_pred, y_score)
    metrics['threshold'] = threshold
    
    return metrics


def compute_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 ROC 曲线
    
    Args:
        y_true: 真实标签
        y_score: 预测分数
        
    Returns:
        (fpr, tpr, thresholds)
    """
    return roc_curve(y_true, y_score)


def compute_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 PR 曲线
    
    Args:
        y_true: 真实标签
        y_score: 预测分数
        
    Returns:
        (precision, recall, thresholds)
    """
    return precision_recall_curve(y_true, y_score)


def find_threshold_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float
) -> Tuple[float, float]:
    """
    找到达到目标 FPR 的阈值
    
    Args:
        y_true: 真实标签
        y_score: 预测分数
        target_fpr: 目标 FPR
        
    Returns:
        (threshold, actual_fpr)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # 找到最接近目标 FPR 的点
    idx = np.argmin(np.abs(fpr - target_fpr))
    
    return float(thresholds[idx]), float(fpr[idx])


def find_threshold_at_tpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_tpr: float
) -> Tuple[float, float]:
    """
    找到达到目标 TPR (Recall) 的阈值
    
    Args:
        y_true: 真实标签
        y_score: 预测分数
        target_tpr: 目标 TPR
        
    Returns:
        (threshold, actual_tpr)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # 找到最接近目标 TPR 的点
    idx = np.argmin(np.abs(tpr - target_tpr))
    
    return float(thresholds[idx]), float(tpr[idx])


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = 'youden'
) -> Tuple[float, Dict[str, float]]:
    """
    计算最优阈值
    
    Args:
        y_true: 真实标签
        y_score: 预测分数
        method: 方法
            - 'youden': 最大化 Youden's J (TPR - FPR)
            - 'f1': 最大化 F1 分数
            
    Returns:
        (optimal_threshold, metrics_at_threshold)
    """
    if method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        idx = np.argmax(j_scores)
        optimal_threshold = float(thresholds[idx])
        
    elif method == 'f1':
        # 搜索不同阈值
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        idx = np.argmax(f1_scores[:-1])  # 最后一个点是 (1, 0)
        optimal_threshold = float(thresholds[idx])
    else:
        raise ValueError(f"未知的方法: {method}")
    
    metrics = compute_threshold_metrics(y_true, y_score, optimal_threshold)
    
    return optimal_threshold, metrics


def summarize_score_distribution(
    scores_normal: np.ndarray,
    scores_anomaly: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    总结分数分布统计
    
    Args:
        scores_normal: 正常样本分数
        scores_anomaly: 异常样本分数 (可选)
        
    Returns:
        分布统计字典
    """
    summary = {
        'normal': {
            'count': len(scores_normal),
            'mean': float(np.mean(scores_normal)),
            'std': float(np.std(scores_normal)),
            'min': float(np.min(scores_normal)),
            'max': float(np.max(scores_normal)),
            'median': float(np.median(scores_normal)),
            'q25': float(np.percentile(scores_normal, 25)),
            'q75': float(np.percentile(scores_normal, 75)),
            'q95': float(np.percentile(scores_normal, 95)),
            'q99': float(np.percentile(scores_normal, 99)),
        }
    }
    
    if scores_anomaly is not None and len(scores_anomaly) > 0:
        summary['anomaly'] = {
            'count': len(scores_anomaly),
            'mean': float(np.mean(scores_anomaly)),
            'std': float(np.std(scores_anomaly)),
            'min': float(np.min(scores_anomaly)),
            'max': float(np.max(scores_anomaly)),
            'median': float(np.median(scores_anomaly)),
            'q25': float(np.percentile(scores_anomaly, 25)),
            'q75': float(np.percentile(scores_anomaly, 75)),
        }
        
        # 可分性指标
        d_prime = (summary['anomaly']['mean'] - summary['normal']['mean']) / \
                  np.sqrt(0.5 * (summary['normal']['std']**2 + summary['anomaly']['std']**2 + 1e-10))
        summary['separability'] = {
            'd_prime': float(d_prime)
        }
    
    return summary


if __name__ == "__main__":
    # 测试指标计算
    print("测试指标计算...")
    
    np.random.seed(42)
    
    # 模拟数据
    n_normal = 100
    n_anomaly = 50
    
    # 正常样本分数低，异常样本分数高
    scores_normal = np.random.exponential(scale=0.1, size=n_normal)
    scores_anomaly = np.random.exponential(scale=0.2, size=n_anomaly) + 0.1
    
    y_true = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    y_score = np.concatenate([scores_normal, scores_anomaly])
    
    # 计算最优阈值
    threshold, metrics = compute_optimal_threshold(y_true, y_score, method='youden')
    
    print(f"最优阈值: {threshold:.4f}")
    print(f"指标:")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"  {key}: {value}")
    
    # 分布统计
    summary = summarize_score_distribution(scores_normal, scores_anomaly)
    print(f"\n分布统计:")
    print(f"  正常样本: mean={summary['normal']['mean']:.4f}, std={summary['normal']['std']:.4f}")
    print(f"  异常样本: mean={summary['anomaly']['mean']:.4f}, std={summary['anomaly']['std']:.4f}")
    print(f"  D-prime: {summary['separability']['d_prime']:.4f}")
