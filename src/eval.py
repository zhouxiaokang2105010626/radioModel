"""
eval.py - E-GAN 评估脚本

评估流程:
1. 加载模型和阈值
2. 在测试集上计算 anomaly score
3. 计算 window-level 和 file-level 指标
4. 生成可视化和报告
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config, get_device, load_checkpoint, 
    setup_logging, get_output_paths, save_json
)
from src.dataset import create_datasets, create_dataloaders, collate_window_batch
from src.model import create_model
from src.thresholds import (
    compute_anomaly_score, calibrate_threshold, 
    save_threshold, load_threshold, predict_from_score
)
from src.aggregation import aggregate_file_scores, add_file_predictions
from src.metrics import (
    compute_classification_metrics, compute_roc_curve, 
    compute_pr_curve, summarize_score_distribution
)
from src.visualize import (
    plot_roc_curve, plot_pr_curve, plot_score_histogram,
    plot_confusion_matrix, plot_score_boxplot
)


def evaluate_windows(
    model,
    dataloader,
    config,
    device
):
    """
    Window-level 评估
    
    Returns:
        window_results: 窗口级结果列表
    """
    model.eval()
    
    lambda_score = config.get('score', {}).get('lambda_score', 0.5)
    
    window_results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            x = batch['spectrogram'].to(device)
            labels = batch['label'].numpy()
            window_infos = batch['window_info']
            
            # 前向传播
            z, x_hat, d_real, d_fake = model(x)
            
            # 计算分数
            ar, ad, total = compute_anomaly_score(
                x, x_hat, d_real, d_fake, lambda_score
            )
            
            ar = ar.cpu().numpy()
            ad = ad.cpu().numpy()
            total = total.cpu().numpy()
            
            # 保存结果
            for i in range(len(labels)):
                win_info = window_infos[i]
                metadata = win_info.get('metadata', {})
                
                result = {
                    'file_path': win_info.get('file_path', ''),
                    'window_index': win_info.get('window_index', 0),
                    'start_sample': win_info.get('start_sample', 0),
                    'end_sample': win_info.get('end_sample', 0),
                    'center_freq_hz': metadata.get('center_freq_hz'),
                    'bandwidth_hz': metadata.get('bandwidth_hz'),
                    'sample_rate_hz': metadata.get('sample_rate_hz'),
                    'score_ar': float(ar[i]),
                    'score_ad': float(ad[i]),
                    'score_total': float(total[i]),
                    'true_label': int(labels[i]) if labels[i] >= 0 else None,
                    'metadata': metadata,
                }
                window_results.append(result)
    
    return window_results


def add_window_predictions(window_results, threshold):
    """为窗口结果添加预测标签"""
    for result in window_results:
        result['threshold'] = threshold
        result['pred_label'] = 1 if result['score_total'] > threshold else 0
    return window_results


def save_window_results(window_results, save_path):
    """保存窗口级结果到 CSV"""
    # 移除 metadata 字段
    results_clean = []
    for r in window_results:
        r_clean = {k: v for k, v in r.items() if k != 'metadata'}
        results_clean.append(r_clean)
    
    df = pd.DataFrame(results_clean)
    df.to_csv(save_path, index=False)
    print(f"窗口级结果已保存: {save_path}")


def save_file_results(file_results, save_path):
    """保存文件级结果到 CSV"""
    # 移除 high_risk_windows 字段
    results_clean = []
    for r in file_results:
        r_clean = {k: v for k, v in r.items() if k != 'high_risk_windows'}
        results_clean.append(r_clean)
    
    df = pd.DataFrame(results_clean)
    df.to_csv(save_path, index=False)
    print(f"文件级结果已保存: {save_path}")


def main(args):
    """主评估函数"""
    
    # 加载配置
    config = load_config(args.config)
    
    # 获取输出路径
    paths = get_output_paths(config)
    
    # 设置日志
    logger = setup_logging(paths['logs_dir'], 'eval')
    logger.info(f"配置文件: {args.config}")
    
    # 获取设备
    device = get_device(config.get('train', {}).get('device', 'cuda'))
    logger.info(f"使用设备: {device}")
    
    # ==================
    # 创建模型并加载权重
    # ==================
    logger.info("加载模型...")
    model = create_model(config, device)
    
    # 查找最佳 checkpoint
    checkpoint_dir = paths['checkpoints_dir']
    best_checkpoint = None
    
    # 优先加载 best
    for f in os.listdir(checkpoint_dir):
        if 'best' in f and f.endswith('.pth'):
            best_checkpoint = os.path.join(checkpoint_dir, f)
            break
    
    # 否则加载最新的
    if best_checkpoint is None:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            # 按 epoch 排序
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)
            best_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    
    if best_checkpoint is None:
        logger.error("未找到 checkpoint!")
        return
    
    load_checkpoint(best_checkpoint, model, device=str(device))
    logger.info(f"加载 checkpoint: {best_checkpoint}")
    
    # ==================
    # 加载阈值
    # ==================
    threshold_info = None
    if os.path.exists(paths['threshold_file']):
        threshold_info = load_threshold(paths['threshold_file'])
        threshold = threshold_info['threshold_total']
        logger.info(f"加载阈值: {threshold:.4f}")
    else:
        logger.warning("未找到阈值文件，将进行校准...")
        args.calibrate = True
    
    # ==================
    # 创建数据集
    # ==================
    logger.info("创建数据集...")
    datasets = create_datasets(config)
    dataloaders = create_dataloaders(datasets, config)
    
    # ==================
    # 阈值校准 (如果需要)
    # ==================
    if args.calibrate:
        logger.info("校准阈值...")
        calibration_loader = dataloaders.get('val_normal') or dataloaders['train']
        
        if calibration_loader is not None:
            threshold_info = calibrate_threshold(
                model, calibration_loader, config, str(device)
            )
            save_threshold(threshold_info, paths['threshold_file'])
            threshold = threshold_info['threshold_total']
            logger.info(f"阈值校准完成: {threshold:.4f}")
        else:
            logger.error("无法校准阈值，没有可用的正常样本数据集")
            return
    
    # ==================
    # 评估测试集
    # ==================
    logger.info("评估测试集...")
    
    all_window_results = []
    
    # 正常样本
    if dataloaders.get('test_normal') is not None:
        normal_results = evaluate_windows(
            model, dataloaders['test_normal'], config, device
        )
        logger.info(f"正常样本: {len(normal_results)} 窗口")
        all_window_results.extend(normal_results)
    
    # 异常样本
    if dataloaders.get('test_anomaly') is not None:
        anomaly_results = evaluate_windows(
            model, dataloaders['test_anomaly'], config, device
        )
        logger.info(f"异常样本: {len(anomaly_results)} 窗口")
        all_window_results.extend(anomaly_results)
    
    if not all_window_results:
        logger.error("测试集为空!")
        return
    
    # 添加预测标签
    all_window_results = add_window_predictions(all_window_results, threshold)
    
    # 保存窗口结果
    save_window_results(all_window_results, paths['predictions_window_file'])
    
    # ==================
    # 计算窗口级指标
    # ==================
    logger.info("计算窗口级指标...")
    
    # 分离有标签和无标签的样本
    labeled_results = [r for r in all_window_results if r['true_label'] is not None]
    
    window_metrics = {}
    
    if labeled_results:
        y_true = np.array([r['true_label'] for r in labeled_results])
        y_score = np.array([r['score_total'] for r in labeled_results])
        y_pred = np.array([r['pred_label'] for r in labeled_results])
        
        window_metrics = compute_classification_metrics(y_true, y_pred, y_score)
        
        logger.info("Window-level 指标:")
        logger.info(f"  ROC-AUC: {window_metrics.get('roc_auc', 'N/A'):.4f}")
        logger.info(f"  PR-AUC: {window_metrics.get('pr_auc', 'N/A'):.4f}")
        logger.info(f"  Accuracy: {window_metrics.get('accuracy', 'N/A'):.4f}")
        logger.info(f"  F1: {window_metrics.get('f1', 'N/A'):.4f}")
        logger.info(f"  Precision: {window_metrics.get('precision', 'N/A'):.4f}")
        logger.info(f"  Recall: {window_metrics.get('recall', 'N/A'):.4f}")
        logger.info(f"  FPR: {window_metrics.get('fpr', 'N/A'):.4f}")
        
        # 绘制 ROC 曲线
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = compute_roc_curve(y_true, y_score)
            plot_roc_curve(
                fpr, tpr, window_metrics.get('roc_auc', 0),
                os.path.join(paths['figures_dir'], 'roc_curve_window.png'),
                title='ROC Curve (Window-level)'
            )
            
            # 绘制 PR 曲线
            precision, recall, _ = compute_pr_curve(y_true, y_score)
            plot_pr_curve(
                precision, recall, window_metrics.get('pr_auc', 0),
                os.path.join(paths['figures_dir'], 'pr_curve_window.png'),
                title='PR Curve (Window-level)'
            )
        
        # 分数分布直方图
        scores_normal = np.array([r['score_total'] for r in labeled_results if r['true_label'] == 0])
        scores_anomaly = np.array([r['score_total'] for r in labeled_results if r['true_label'] == 1])
        
        if len(scores_normal) > 0:
            plot_score_histogram(
                scores_normal, 
                scores_anomaly if len(scores_anomaly) > 0 else None,
                threshold,
                os.path.join(paths['figures_dir'], 'score_histogram_window.png'),
                title='Anomaly Score Distribution (Window-level)'
            )
        
        # 混淆矩阵
        if 'confusion_matrix' in window_metrics:
            plot_confusion_matrix(
                np.array(window_metrics['confusion_matrix']),
                os.path.join(paths['figures_dir'], 'confusion_matrix_window.png'),
                title='Confusion Matrix (Window-level)'
            )
    
    # ==================
    # 文件级聚合
    # ==================
    logger.info("计算文件级指标...")
    
    score_config = config.get('score', {})
    aggregation_method = score_config.get('aggregation_method', 'topk_mean')
    topk_ratio = score_config.get('topk_ratio', 0.1)
    
    file_results = aggregate_file_scores(
        all_window_results,
        method=aggregation_method,
        threshold=threshold,
        topk_ratio=topk_ratio
    )
    
    file_results = add_file_predictions(file_results, threshold)
    
    # 保存文件结果
    save_file_results(file_results, paths['predictions_file_file'])
    
    # 计算文件级指标
    labeled_files = [r for r in file_results if r.get('true_label') is not None]
    
    file_metrics = {}
    
    if labeled_files:
        y_true_file = np.array([r['true_label'] for r in labeled_files])
        y_score_file = np.array([r['file_score'] for r in labeled_files])
        y_pred_file = np.array([r['pred_label'] for r in labeled_files])
        
        file_metrics = compute_classification_metrics(y_true_file, y_pred_file, y_score_file)
        
        logger.info("File-level 指标:")
        logger.info(f"  ROC-AUC: {file_metrics.get('roc_auc', 'N/A'):.4f}")
        logger.info(f"  PR-AUC: {file_metrics.get('pr_auc', 'N/A'):.4f}")
        logger.info(f"  Accuracy: {file_metrics.get('accuracy', 'N/A'):.4f}")
        logger.info(f"  F1: {file_metrics.get('f1', 'N/A'):.4f}")
        
        # 绘制文件级曲线
        if len(np.unique(y_true_file)) > 1:
            fpr_f, tpr_f, _ = compute_roc_curve(y_true_file, y_score_file)
            plot_roc_curve(
                fpr_f, tpr_f, file_metrics.get('roc_auc', 0),
                os.path.join(paths['figures_dir'], 'roc_curve_file.png'),
                title='ROC Curve (File-level)'
            )
            
            precision_f, recall_f, _ = compute_pr_curve(y_true_file, y_score_file)
            plot_pr_curve(
                precision_f, recall_f, file_metrics.get('pr_auc', 0),
                os.path.join(paths['figures_dir'], 'pr_curve_file.png'),
                title='PR Curve (File-level)'
            )
        
        # 文件级分数直方图
        scores_normal_file = np.array([r['file_score'] for r in labeled_files if r['true_label'] == 0])
        scores_anomaly_file = np.array([r['file_score'] for r in labeled_files if r['true_label'] == 1])
        
        if len(scores_normal_file) > 0:
            plot_score_histogram(
                scores_normal_file,
                scores_anomaly_file if len(scores_anomaly_file) > 0 else None,
                threshold,
                os.path.join(paths['figures_dir'], 'score_histogram_file.png'),
                title='Anomaly Score Distribution (File-level)'
            )
    
    # ==================
    # 保存完整指标
    # ==================
    all_metrics = {
        'threshold': threshold,
        'threshold_info': threshold_info,
        'window_level': window_metrics,
        'file_level': file_metrics,
        'n_windows_total': len(all_window_results),
        'n_files_total': len(file_results),
        'aggregation_method': aggregation_method,
    }
    
    # 添加分数分布统计
    if labeled_results:
        scores_normal = np.array([r['score_total'] for r in labeled_results if r['true_label'] == 0])
        scores_anomaly = np.array([r['score_total'] for r in labeled_results if r['true_label'] == 1])
        all_metrics['score_distribution'] = summarize_score_distribution(
            scores_normal, scores_anomaly if len(scores_anomaly) > 0 else None
        )
    
    save_json(all_metrics, paths['metrics_file'])
    logger.info(f"完整指标已保存: {paths['metrics_file']}")
    
    logger.info("评估完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='E-GAN Evaluation')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='重新校准阈值'
    )
    
    args = parser.parse_args()
    
    main(args)
