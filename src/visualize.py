"""
visualize.py - 可视化工具

包含:
- 训练曲线
- ROC/PR 曲线
- Score 分布直方图
- Spectrogram 对比
- 异常定位热力图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from typing import List, Dict, Optional, Tuple
import warnings


def plot_training_curves(
    train_history: Dict[str, List[float]],
    save_path: str,
    show: bool = False
):
    """
    绘制训练曲线
    
    Args:
        train_history: 训练历史字典，包含各种损失
        save_path: 保存路径
        show: 是否显示
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(train_history.get('loss_d', [])) + 1)
    
    # D 损失
    ax = axes[0, 0]
    if 'loss_d' in train_history:
        ax.plot(epochs, train_history['loss_d'], 'b-', label='D Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Discriminator Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # EG 损失
    ax = axes[0, 1]
    if 'loss_eg' in train_history:
        ax.plot(epochs, train_history['loss_eg'], 'r-', label='EG Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Encoder-Generator Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 重构损失
    ax = axes[1, 0]
    if 'loss_recon' in train_history:
        ax.plot(epochs, train_history['loss_recon'], 'g-', label='Recon Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 对抗损失
    ax = axes[1, 1]
    if 'loss_adv' in train_history:
        ax.plot(epochs, train_history['loss_adv'], 'm-', label='Adv Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Adversarial Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"训练曲线已保存: {save_path}")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    save_path: str,
    title: str = 'ROC Curve',
    show: bool = False
):
    """
    绘制 ROC 曲线
    
    Args:
        fpr: False Positive Rate
        tpr: True Positive Rate
        roc_auc: AUC 值
        save_path: 保存路径
        title: 标题
        show: 是否显示
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"ROC 曲线已保存: {save_path}")


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    save_path: str,
    title: str = 'Precision-Recall Curve',
    show: bool = False
):
    """
    绘制 PR 曲线
    
    Args:
        precision: Precision
        recall: Recall
        pr_auc: AUC 值
        save_path: 保存路径
        title: 标题
        show: 是否显示
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AUC = {pr_auc:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"PR 曲线已保存: {save_path}")


def plot_score_histogram(
    scores_normal: np.ndarray,
    scores_anomaly: Optional[np.ndarray],
    threshold: Optional[float],
    save_path: str,
    title: str = 'Anomaly Score Distribution',
    show: bool = False
):
    """
    绘制分数分布直方图
    
    Args:
        scores_normal: 正常样本分数
        scores_anomaly: 异常样本分数 (可选)
        threshold: 阈值 (可选)
        save_path: 保存路径
        title: 标题
        show: 是否显示
    """
    plt.figure(figsize=(10, 6))
    
    # 确定 bins 范围
    all_scores = scores_normal.copy()
    if scores_anomaly is not None and len(scores_anomaly) > 0:
        all_scores = np.concatenate([scores_normal, scores_anomaly])
    
    bins = np.linspace(all_scores.min(), all_scores.max(), 50)
    
    # 正常样本
    plt.hist(scores_normal, bins=bins, alpha=0.6, color='green', 
             label=f'Normal (n={len(scores_normal)})', density=True)
    
    # 异常样本
    if scores_anomaly is not None and len(scores_anomaly) > 0:
        plt.hist(scores_anomaly, bins=bins, alpha=0.6, color='red',
                 label=f'Anomaly (n={len(scores_anomaly)})', density=True)
    
    # 阈值线
    if threshold is not None:
        plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.4f}')
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"分数分布图已保存: {save_path}")


def plot_score_boxplot(
    scores_by_group: Dict[str, np.ndarray],
    save_path: str,
    title: str = 'Score Distribution by Group',
    show: bool = False
):
    """
    绘制分数箱线图
    
    Args:
        scores_by_group: 分组分数字典
        save_path: 保存路径
        title: 标题
        show: 是否显示
    """
    plt.figure(figsize=(10, 6))
    
    labels = list(scores_by_group.keys())
    data = [scores_by_group[label] for label in labels]
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # 着色
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xlabel('Group')
    plt.ylabel('Anomaly Score')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"箱线图已保存: {save_path}")


def plot_spectrogram_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: str,
    title: str = 'Spectrogram Comparison',
    show: bool = False
):
    """
    绘制原始 vs 重构 spectrogram 对比
    
    Args:
        original: 原始 spectrogram (H, W) 或 (1, H, W)
        reconstructed: 重构 spectrogram
        save_path: 保存路径
        title: 标题
        show: 是否显示
    """
    # 处理维度
    if original.ndim == 3:
        original = original[0]
    if reconstructed.ndim == 3:
        reconstructed = reconstructed[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 原始
    ax = axes[0]
    im = ax.imshow(original, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title('Original')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax)
    
    # 重构
    ax = axes[1]
    im = ax.imshow(reconstructed, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title('Reconstructed')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax)
    
    # 差异
    ax = axes[2]
    diff = np.abs(original - reconstructed)
    im = ax.imshow(diff, aspect='auto', origin='lower', cmap='hot')
    ax.set_title(f'|Difference| (mean={diff.mean():.4f})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"Spectrogram 对比图已保存: {save_path}")


def plot_localization_heatmap(
    original: np.ndarray,
    heatmap: np.ndarray,
    save_path: str,
    window_info: Optional[Dict] = None,
    score: Optional[float] = None,
    show: bool = False
):
    """
    绘制异常定位热力图
    
    Args:
        original: 原始 spectrogram
        heatmap: 定位热力图 (差异图)
        save_path: 保存路径
        window_info: 窗口信息
        score: 异常分数
        show: 是否显示
    """
    # 处理维度
    if original.ndim == 3:
        original = original[0]
    if heatmap.ndim == 3:
        heatmap = heatmap[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始
    ax = axes[0]
    im = ax.imshow(original, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title('Original Spectrogram')
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Frequency Bin')
    plt.colorbar(im, ax=ax, label='Magnitude')
    
    # 热力图
    ax = axes[1]
    im = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='hot')
    ax.set_title('Anomaly Localization Heatmap')
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Frequency Bin')
    plt.colorbar(im, ax=ax, label='Anomaly Intensity')
    
    # 添加信息
    info_text = ""
    if window_info:
        info_text += f"File: {os.path.basename(window_info.get('file_path', 'Unknown'))}\n"
        info_text += f"Window: {window_info.get('window_index', 'N/A')}\n"
    if score is not None:
        info_text += f"Anomaly Score: {score:.4f}"
    
    if info_text:
        fig.suptitle(info_text, fontsize=10, y=1.02)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_window_scores(
    scores: np.ndarray,
    threshold: Optional[float],
    save_path: str,
    file_path: Optional[str] = None,
    show: bool = False
):
    """
    绘制单个文件内窗口分数曲线
    
    Args:
        scores: 窗口分数数组
        threshold: 阈值
        save_path: 保存路径
        file_path: 文件路径
        show: 是否显示
    """
    plt.figure(figsize=(12, 4))
    
    window_indices = np.arange(len(scores))
    
    plt.plot(window_indices, scores, 'b-', linewidth=1.5, marker='o', 
             markersize=3, label='Window Score')
    
    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.4f}')
    
    plt.xlabel('Window Index')
    plt.ylabel('Anomaly Score')
    
    title = 'Window Anomaly Scores'
    if file_path:
        title += f'\n{os.path.basename(file_path)}'
    plt.title(title)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    title: str = 'Confusion Matrix',
    labels: List[str] = ['Normal', 'Anomaly'],
    show: bool = False
):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        save_path: 保存路径
        title: 标题
        labels: 标签名
        show: 是否显示
    """
    plt.figure(figsize=(8, 6))
    
    # 归一化版本
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    im = plt.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    
    # 添加数值标注
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2%})',
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > thresh else 'black',
                    fontsize=12)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"混淆矩阵已保存: {save_path}")


if __name__ == "__main__":
    # 测试可视化函数
    print("测试可视化函数...")
    
    # 测试训练曲线
    history = {
        'loss_d': np.random.rand(100) * 0.5 + np.linspace(1, 0.5, 100),
        'loss_eg': np.random.rand(100) * 0.5 + np.linspace(2, 0.8, 100),
        'loss_recon': np.random.rand(100) * 0.1 + np.linspace(0.5, 0.1, 100),
        'loss_adv': np.random.rand(100) * 0.1 + np.linspace(0.7, 0.4, 100),
    }
    
    plot_training_curves(history, 'test_output/training_curves.png')
    
    # 测试分数直方图
    scores_normal = np.random.exponential(0.1, 500)
    scores_anomaly = np.random.exponential(0.2, 200) + 0.1
    
    plot_score_histogram(
        scores_normal, scores_anomaly, 
        threshold=0.2,
        save_path='test_output/score_histogram.png'
    )
    
    print("测试完成!")
