"""
thresholds.py - 异常分数计算与阈值校准

核心功能:
1. 计算异常分数 (AR + AD)
2. 基于正常样本校准阈值
3. 预测正常/异常
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import json
import os


def compute_reconstruction_score(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    计算重构分数 AR
    
    AR(y) = mean(|y - y_hat|)
    
    Args:
        y: 原始 spectrogram
        y_hat: 重构 spectrogram
        reduction: 'mean' 或 'none'
        
    Returns:
        AR 分数
    """
    diff = torch.abs(y - y_hat)
    
    if reduction == 'mean':
        # 每个样本的平均重构误差
        return diff.view(diff.size(0), -1).mean(dim=1)
    elif reduction == 'none':
        # 返回完整的差异图 (用于 localization)
        return diff
    else:
        return diff.mean()


def compute_discriminator_score(
    d_y: torch.Tensor,
    d_y_hat: torch.Tensor
) -> torch.Tensor:
    """
    计算判别器分数 AD
    
    AD(y) = |sigmoid(D(y)) - sigmoid(D(y_hat))|
    
    Args:
        d_y: D(y) logits
        d_y_hat: D(y_hat) logits
        
    Returns:
        AD 分数, shape = (B,)
    """
    p_y = torch.sigmoid(d_y)
    p_y_hat = torch.sigmoid(d_y_hat)
    
    ad = torch.abs(p_y - p_y_hat).squeeze(-1)
    
    return ad


def compute_anomaly_score(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    d_y: torch.Tensor,
    d_y_hat: torch.Tensor,
    lambda_score: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算综合异常分数
    
    A(y) = (1 - lambda) * AR(y) + lambda * AD(y)
    
    Args:
        y: 原始 spectrogram
        y_hat: 重构 spectrogram
        d_y: D(y) logits
        d_y_hat: D(y_hat) logits
        lambda_score: AD 分数权重
        
    Returns:
        (ar_score, ad_score, total_score)
    """
    ar_score = compute_reconstruction_score(y, y_hat, reduction='mean')
    ad_score = compute_discriminator_score(d_y, d_y_hat)
    
    # 归一化 AR 和 AD 到相似量级
    # AR 通常在 [0, 1] 左右 (取决于 spectrogram 范围)
    # AD 在 [0, 1] 范围
    
    # 综合分数
    total_score = (1 - lambda_score) * ar_score + lambda_score * ad_score
    
    return ar_score, ad_score, total_score


def compute_localization_map(
    y: torch.Tensor,
    y_hat: torch.Tensor
) -> torch.Tensor:
    """
    计算异常定位热力图
    
    Args:
        y: 原始 spectrogram, shape = (B, 1, H, W)
        y_hat: 重构 spectrogram
        
    Returns:
        定位热力图, shape = (B, H, W)
    """
    diff = torch.abs(y - y_hat)
    return diff.squeeze(1)  # 去除通道维度


def fit_threshold(
    scores_normal: np.ndarray,
    target_pfa: float = 0.05
) -> float:
    """
    基于正常样本分数拟合阈值
    
    使用分位数方法: eta = quantile(scores, 1 - target_pfa)
    
    Args:
        scores_normal: 正常样本的异常分数数组
        target_pfa: 目标误报率 (False Positive Rate)
        
    Returns:
        阈值 eta
    """
    # 计算 1 - target_pfa 分位数
    # 例如 target_pfa=0.05 时，取 95% 分位数
    quantile = 1.0 - target_pfa
    eta = np.quantile(scores_normal, quantile)
    
    return float(eta)


def fit_threshold_with_margin(
    scores_normal: np.ndarray,
    target_pfa: float = 0.05,
    margin_factor: float = 1.1
) -> float:
    """
    带安全边际的阈值拟合
    
    Args:
        scores_normal: 正常样本分数
        target_pfa: 目标误报率
        margin_factor: 边际因子
        
    Returns:
        阈值
    """
    base_threshold = fit_threshold(scores_normal, target_pfa)
    return base_threshold * margin_factor


def predict_from_score(
    score: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    根据分数和阈值预测标签
    
    score > threshold -> anomaly (1)
    score <= threshold -> normal (0)
    
    Args:
        score: 异常分数数组
        threshold: 阈值
        
    Returns:
        预测标签数组
    """
    return (score > threshold).astype(np.int32)


def calibrate_threshold(
    model,
    dataloader,
    config: dict,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    在正常样本上校准阈值
    
    Args:
        model: E-GAN 模型
        dataloader: 正常样本 DataLoader
        config: 配置
        device: 设备
        
    Returns:
        阈值信息字典
    """
    model.eval()
    
    lambda_score = config.get('score', {}).get('lambda_score', 0.5)
    target_pfa = config.get('score', {}).get('target_pfa', 0.05)
    
    all_ar_scores = []
    all_ad_scores = []
    all_total_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['spectrogram'].to(device)
            
            # 前向传播
            z, x_hat, d_real, d_fake = model(x)
            
            # 计算分数
            ar, ad, total = compute_anomaly_score(
                x, x_hat, d_real, d_fake, lambda_score
            )
            
            all_ar_scores.extend(ar.cpu().numpy().tolist())
            all_ad_scores.extend(ad.cpu().numpy().tolist())
            all_total_scores.extend(total.cpu().numpy().tolist())
    
    all_ar_scores = np.array(all_ar_scores)
    all_ad_scores = np.array(all_ad_scores)
    all_total_scores = np.array(all_total_scores)
    
    # 计算阈值
    threshold_ar = fit_threshold(all_ar_scores, target_pfa)
    threshold_ad = fit_threshold(all_ad_scores, target_pfa)
    threshold_total = fit_threshold(all_total_scores, target_pfa)
    
    result = {
        'threshold_ar': threshold_ar,
        'threshold_ad': threshold_ad,
        'threshold_total': threshold_total,
        'target_pfa': target_pfa,
        'lambda_score': lambda_score,
        'n_samples': len(all_total_scores),
        'score_stats': {
            'ar_mean': float(all_ar_scores.mean()),
            'ar_std': float(all_ar_scores.std()),
            'ar_min': float(all_ar_scores.min()),
            'ar_max': float(all_ar_scores.max()),
            'ad_mean': float(all_ad_scores.mean()),
            'ad_std': float(all_ad_scores.std()),
            'total_mean': float(all_total_scores.mean()),
            'total_std': float(all_total_scores.std()),
        }
    }
    
    return result


def save_threshold(
    threshold_info: Dict[str, float],
    save_path: str
):
    """保存阈值信息到 JSON"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(threshold_info, f, indent=2)
    print(f"阈值已保存: {save_path}")


def load_threshold(load_path: str) -> Dict[str, float]:
    """从 JSON 加载阈值信息"""
    with open(load_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # 测试
    print("测试阈值计算...")
    
    # 模拟正常样本分数
    np.random.seed(42)
    scores_normal = np.random.exponential(scale=0.1, size=1000)
    
    # 拟合阈值
    eta = fit_threshold(scores_normal, target_pfa=0.05)
    print(f"正常样本分数统计:")
    print(f"  Mean: {scores_normal.mean():.4f}")
    print(f"  Std: {scores_normal.std():.4f}")
    print(f"  95% 分位数阈值: {eta:.4f}")
    
    # 验证误报率
    predictions = predict_from_score(scores_normal, eta)
    actual_fpr = predictions.sum() / len(predictions)
    print(f"  实际误报率: {actual_fpr:.4f}")
