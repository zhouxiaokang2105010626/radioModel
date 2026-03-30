"""
infer.py - E-GAN 单文件推理脚本

推理流程:
1. 加载模型和阈值
2. 处理单个 IQ 文件
3. 计算每个窗口的 anomaly score
4. 聚合为文件级分数
5. 输出判断结果和热力图
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config, get_device, load_checkpoint, 
    get_output_paths, ensure_dir
)
from src.io_utils import read_iq_auto, parse_iq_filename_metadata
from src.preprocess import (
    process_file_to_spectrograms, iq_to_spectrogram,
    create_windows_with_metadata, preprocess_iq
)
from src.model import create_model
from src.thresholds import (
    compute_anomaly_score, compute_localization_map,
    load_threshold, predict_from_score
)
from src.aggregation import aggregate_max, aggregate_mean, aggregate_topk_mean
from src.visualize import (
    plot_localization_heatmap, plot_spectrogram_comparison,
    plot_window_scores
)


def infer_file(
    file_path: str,
    model,
    config: dict,
    threshold: float,
    device: str,
    save_dir: str = None,
    save_visualizations: bool = True
):
    """
    对单个 IQ 文件进行推理
    
    Args:
        file_path: IQ 文件路径
        model: E-GAN 模型
        config: 配置字典
        threshold: 阈值
        device: 设备
        save_dir: 可视化保存目录
        save_visualizations: 是否保存可视化
        
    Returns:
        推理结果字典
    """
    model.eval()
    
    iq_config = config.get('iq', {})
    score_config = config.get('score', {})
    
    lambda_score = score_config.get('lambda_score', 0.5)
    aggregation_method = score_config.get('aggregation_method', 'topk_mean')
    topk_ratio = score_config.get('topk_ratio', 0.1)
    
    print(f"\n处理文件: {file_path}")
    
    # 读取文件
    data, metadata = read_iq_auto(
        file_path,
        iq_config,
        use_wav_fallback=config.get('data', {}).get('use_wav_fallback', False)
    )
    
    print(f"  文件信息:")
    print(f"    采样点数: {len(data)}")
    print(f"    中心频率: {metadata.get('center_freq_hz')} Hz")
    print(f"    带宽: {metadata.get('bandwidth_hz')} Hz")
    print(f"    采样率: {metadata.get('sample_rate_hz')} Hz")
    
    # 处理为 spectrogram
    spectrograms, window_infos = process_file_to_spectrograms(
        file_path, config, iq_config
    )
    
    print(f"    窗口数: {len(spectrograms)}")
    
    if len(spectrograms) == 0:
        print("  警告: 没有生成任何窗口!")
        return None
    
    # 转为 tensor
    x = torch.from_numpy(np.stack(spectrograms, axis=0)).float().to(device)
    
    # 推理
    window_results = []
    localization_maps = []
    
    with torch.no_grad():
        # 批量处理
        batch_size = config.get('train', {}).get('batch_size', 32)
        
        for start_idx in range(0, len(x), batch_size):
            end_idx = min(start_idx + batch_size, len(x))
            x_batch = x[start_idx:end_idx]
            
            # 前向传播
            z, x_hat, d_real, d_fake = model(x_batch)
            
            # 计算分数
            ar, ad, total = compute_anomaly_score(
                x_batch, x_hat, d_real, d_fake, lambda_score
            )
            
            # 计算定位图
            loc_maps = compute_localization_map(x_batch, x_hat)
            
            # 保存结果
            for i in range(end_idx - start_idx):
                global_idx = start_idx + i
                win_info = window_infos[global_idx]
                
                result = {
                    'window_index': global_idx,
                    'start_sample': win_info.get('start_sample', 0),
                    'end_sample': win_info.get('end_sample', 0),
                    'score_ar': float(ar[i].cpu().numpy()),
                    'score_ad': float(ad[i].cpu().numpy()),
                    'score_total': float(total[i].cpu().numpy()),
                    'pred_label': 1 if total[i].cpu().numpy() > threshold else 0,
                    'original': x_batch[i].cpu().numpy(),
                    'reconstructed': x_hat[i].cpu().numpy(),
                    'localization': loc_maps[i].cpu().numpy(),
                }
                window_results.append(result)
    
    # 聚合
    scores = np.array([r['score_total'] for r in window_results])
    
    if aggregation_method == 'max':
        file_score = aggregate_max(scores)
    elif aggregation_method == 'mean':
        file_score = aggregate_mean(scores)
    elif aggregation_method == 'topk_mean':
        file_score = aggregate_topk_mean(scores, topk_ratio)
    else:
        file_score = aggregate_topk_mean(scores, topk_ratio)
    
    file_pred = 1 if file_score > threshold else 0
    
    # 找出高风险窗口
    high_risk_indices = np.argsort(scores)[-max(1, int(len(scores) * topk_ratio)):][::-1]
    
    # 输出结果
    print(f"\n  === 检测结果 ===")
    print(f"  文件级分数: {file_score:.4f}")
    print(f"  阈值: {threshold:.4f}")
    print(f"  判断结果: {'异常 (ANOMALY)' if file_pred == 1 else '正常 (NORMAL)'}")
    print(f"\n  窗口统计:")
    print(f"    总窗口数: {len(window_results)}")
    print(f"    异常窗口数: {sum(r['pred_label'] for r in window_results)}")
    print(f"    分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"    分数均值: {scores.mean():.4f}")
    
    print(f"\n  高风险窗口 (Top {len(high_risk_indices)}):")
    for idx in high_risk_indices[:5]:  # 只显示前5个
        r = window_results[idx]
        print(f"    窗口 {idx}: score={r['score_total']:.4f}, "
              f"samples [{r['start_sample']}-{r['end_sample']}]")
    
    # 保存可视化
    if save_visualizations and save_dir:
        ensure_dir(save_dir)
        file_basename = Path(file_path).stem
        
        # 1. 窗口分数曲线
        plot_window_scores(
            scores, threshold,
            os.path.join(save_dir, f'{file_basename}_window_scores.png'),
            file_path=file_path
        )
        print(f"\n  窗口分数曲线已保存")
        
        # 2. 高风险窗口的热力图
        num_to_save = min(5, len(high_risk_indices))
        for i, idx in enumerate(high_risk_indices[:num_to_save]):
            r = window_results[idx]
            
            # 热力图
            plot_localization_heatmap(
                r['original'], r['localization'],
                os.path.join(save_dir, f'{file_basename}_localization_win{idx}.png'),
                window_info={'file_path': file_path, 'window_index': idx},
                score=r['score_total']
            )
            
            # 重构对比
            plot_spectrogram_comparison(
                r['original'], r['reconstructed'],
                os.path.join(save_dir, f'{file_basename}_recon_win{idx}.png'),
                title=f'Window {idx}, Score={r["score_total"]:.4f}'
            )
        
        print(f"  热力图已保存 ({num_to_save} 个窗口)")
    
    # 构建返回结果
    result = {
        'file_path': file_path,
        'metadata': metadata,
        'n_windows': len(window_results),
        'file_score': file_score,
        'threshold': threshold,
        'pred_label': file_pred,
        'prediction': 'anomaly' if file_pred == 1 else 'normal',
        'aggregation_method': aggregation_method,
        'window_scores': {
            'min': float(scores.min()),
            'max': float(scores.max()),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
        },
        'high_risk_windows': [
            {
                'window_index': int(idx),
                'score': float(window_results[idx]['score_total']),
                'start_sample': window_results[idx]['start_sample'],
                'end_sample': window_results[idx]['end_sample'],
            }
            for idx in high_risk_indices
        ],
        'window_results': [
            {
                'window_index': r['window_index'],
                'score_total': r['score_total'],
                'pred_label': r['pred_label'],
            }
            for r in window_results
        ]
    }
    
    return result


def main(args):
    """主推理函数"""
    
    # 加载配置
    config = load_config(args.config)
    
    # 获取输出路径
    paths = get_output_paths(config)
    
    # 获取设备
    device = get_device(config.get('train', {}).get('device', 'cuda'))
    print(f"使用设备: {device}")
    
    # ==================
    # 加载模型
    # ==================
    print("加载模型...")
    model = create_model(config, device)
    
    # 查找 checkpoint
    checkpoint_dir = paths['checkpoints_dir']
    best_checkpoint = None
    
    for f in os.listdir(checkpoint_dir):
        if 'best' in f and f.endswith('.pth'):
            best_checkpoint = os.path.join(checkpoint_dir, f)
            break
    
    if best_checkpoint is None:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            checkpoints = sorted(checkpoints)
            best_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    
    if best_checkpoint is None:
        print("错误: 未找到 checkpoint!")
        return
    
    load_checkpoint(best_checkpoint, model, device=str(device))
    print(f"加载 checkpoint: {best_checkpoint}")
    
    # ==================
    # 加载阈值
    # ==================
    if os.path.exists(paths['threshold_file']):
        threshold_info = load_threshold(paths['threshold_file'])
        threshold = threshold_info['threshold_total']
        print(f"加载阈值: {threshold:.4f}")
    else:
        print("警告: 未找到阈值文件，使用默认阈值 0.1")
        threshold = 0.1
    
    # ==================
    # 推理
    # ==================
    # 创建推理输出目录
    infer_output_dir = os.path.join(paths['figures_dir'], 'inference')
    
    result = infer_file(
        args.input,
        model,
        config,
        threshold,
        str(device),
        save_dir=infer_output_dir,
        save_visualizations=not args.no_viz
    )
    
    if result is None:
        return
    
    # 保存结果到 JSON
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            infer_output_dir,
            f'{Path(args.input).stem}_result.json'
        )
    
    ensure_dir(os.path.dirname(output_path))
    
    # 移除大型数据以便保存
    result_to_save = {k: v for k, v in result.items()}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"\n推理结果已保存: {output_path}")
    
    # 打印最终结论
    print("\n" + "=" * 50)
    if result['pred_label'] == 1:
        print("最终判断: [异常] ANOMALY DETECTED")
    else:
        print("最终判断: [正常] NORMAL")
    print("=" * 50)


def batch_infer(args):
    """批量推理多个文件"""
    from src.io_utils import get_iq_files
    
    # 加载配置
    config = load_config(args.config)
    paths = get_output_paths(config)
    device = get_device(config.get('train', {}).get('device', 'cuda'))
    
    # 加载模型
    model = create_model(config, device)
    
    checkpoint_dir = paths['checkpoints_dir']
    best_checkpoint = None
    for f in os.listdir(checkpoint_dir):
        if 'best' in f and f.endswith('.pth'):
            best_checkpoint = os.path.join(checkpoint_dir, f)
            break
    
    if best_checkpoint:
        load_checkpoint(best_checkpoint, model, device=str(device))
    
    # 加载阈值
    threshold = 0.1
    if os.path.exists(paths['threshold_file']):
        threshold_info = load_threshold(paths['threshold_file'])
        threshold = threshold_info['threshold_total']
    
    # 获取文件列表
    files = get_iq_files(args.input_dir)
    print(f"找到 {len(files)} 个 IQ 文件")
    
    infer_output_dir = os.path.join(paths['figures_dir'], 'batch_inference')
    
    results = []
    for file_path in files:
        result = infer_file(
            file_path, model, config, threshold, str(device),
            save_dir=infer_output_dir,
            save_visualizations=not args.no_viz
        )
        if result:
            results.append({
                'file_path': result['file_path'],
                'file_score': result['file_score'],
                'prediction': result['prediction'],
            })
    
    # 保存汇总
    summary_path = os.path.join(infer_output_dir, 'batch_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n批量推理完成，汇总已保存: {summary_path}")
    
    # 打印统计
    n_normal = sum(1 for r in results if r['prediction'] == 'normal')
    n_anomaly = sum(1 for r in results if r['prediction'] == 'anomaly')
    print(f"统计: 正常 {n_normal}, 异常 {n_anomaly}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='E-GAN Inference')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入 IQ 文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出 JSON 路径 (可选)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='不保存可视化'
    )
    
    args = parser.parse_args()
    
    main(args)
