"""
train.py - E-GAN 训练脚本

训练流程:
1. 加载配置
2. 创建数据集和模型
3. 训练 E-GAN (normal-only)
4. 保存 checkpoint 和训练曲线
"""

import os
import sys
import argparse
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config, save_config, set_seed, get_device, 
    save_checkpoint, load_checkpoint, count_parameters,
    format_time, setup_logging, get_output_paths,
    EarlyStopping, AverageMeter
)
from src.dataset import create_datasets, create_dataloaders
from src.model import create_model
from src.losses import EGANLoss
from src.visualize import plot_training_curves, plot_spectrogram_comparison
from src.thresholds import calibrate_threshold, save_threshold


def train_epoch(
    model,
    dataloader,
    optimizer_d,
    optimizer_eg,
    loss_fn,
    device,
    config,
    epoch,
    logger
):
    """
    训练一个 epoch
    
    Returns:
        epoch 平均损失字典
    """
    model.train()
    
    train_config = config.get('train', {})
    gradient_clip_norm = train_config.get('gradient_clip_norm', 1.0)
    
    # 损失记录
    meters = {
        'loss_d': AverageMeter(),
        'loss_eg': AverageMeter(),
        'loss_recon': AverageMeter(),
        'loss_adv': AverageMeter(),
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    
    for batch in pbar:
        x = batch['spectrogram'].to(device)
        batch_size = x.size(0)
        
        # ==================
        # 1. 前向传播
        # ==================
        z, x_hat, d_real, d_fake = model(x)
        
        # ==================
        # 2. 更新判别器 D
        # ==================
        optimizer_d.zero_grad()
        
        # 重新计算 D 输出 (用于正确的梯度)
        d_real = model.discriminate(x)
        d_fake = model.discriminate(x_hat.detach())
        
        loss_d, d_dict = loss_fn.compute_d_loss(d_real, d_fake)
        loss_d.backward()
        
        if gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.discriminator.parameters(), gradient_clip_norm)
        
        optimizer_d.step()
        
        # ==================
        # 3. 更新编码器和生成器 EG
        # ==================
        optimizer_eg.zero_grad()
        
        # 重新前向传播
        z = model.encode(x)
        x_hat = model.decode(z)
        d_fake = model.discriminate(x_hat)
        
        loss_eg, eg_dict = loss_fn.compute_eg_loss(x, x_hat, d_fake)
        loss_eg.backward()
        
        if gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.encoder.parameters(), gradient_clip_norm)
            nn.utils.clip_grad_norm_(model.generator.parameters(), gradient_clip_norm)
        
        optimizer_eg.step()
        
        # ==================
        # 4. 记录损失
        # ==================
        meters['loss_d'].update(d_dict['loss_d'], batch_size)
        meters['loss_eg'].update(eg_dict['loss_eg'], batch_size)
        meters['loss_recon'].update(eg_dict['loss_recon'], batch_size)
        meters['loss_adv'].update(eg_dict['loss_adv'], batch_size)
        
        # 更新进度条
        pbar.set_postfix({
            'D': f"{meters['loss_d'].avg:.4f}",
            'EG': f"{meters['loss_eg'].avg:.4f}",
            'Recon': f"{meters['loss_recon'].avg:.4f}",
        })
    
    # 返回平均损失
    return {k: v.avg for k, v in meters.items()}


def validate(model, dataloader, loss_fn, device):
    """
    验证
    
    Returns:
        验证损失字典
    """
    model.eval()
    
    meters = {
        'loss_d': AverageMeter(),
        'loss_eg': AverageMeter(),
        'loss_recon': AverageMeter(),
    }
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['spectrogram'].to(device)
            batch_size = x.size(0)
            
            z, x_hat, d_real, d_fake = model(x)
            
            loss_d, d_dict = loss_fn.compute_d_loss(d_real, d_fake)
            loss_eg, eg_dict = loss_fn.compute_eg_loss(x, x_hat, d_fake)
            
            meters['loss_d'].update(d_dict['loss_d'], batch_size)
            meters['loss_eg'].update(eg_dict['loss_eg'], batch_size)
            meters['loss_recon'].update(eg_dict['loss_recon'], batch_size)
    
    return {k: v.avg for k, v in meters.items()}


def save_reconstruction_examples(
    model, dataloader, device, save_dir, num_examples=5
):
    """保存重构示例"""
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch['spectrogram'].to(device)
            
            z = model.encode(x)
            x_hat = model.decode(z)
            
            for i in range(min(x.size(0), num_examples - count)):
                orig = x[i].cpu().numpy()
                recon = x_hat[i].cpu().numpy()
                
                save_path = os.path.join(save_dir, f'recon_example_{count}.png')
                plot_spectrogram_comparison(
                    orig, recon, save_path,
                    title=f'Reconstruction Example {count}',
                    show=False
                )
                count += 1
                
                if count >= num_examples:
                    return


def main(args):
    """主训练函数"""
    
    # 加载配置
    config = load_config(args.config)
    
    # 获取输出路径
    paths = get_output_paths(config)
    
    # 设置日志
    logger = setup_logging(paths['logs_dir'], 'train')
    logger.info(f"配置文件: {args.config}")
    
    # 保存配置副本
    save_config(config, os.path.join(paths['output_dir'], 'config.yaml'))
    
    # 设置随机种子
    train_config = config.get('train', {})
    set_seed(train_config.get('seed', 42))
    
    # 获取设备
    device = get_device(train_config.get('device', 'cuda'))
    logger.info(f"使用设备: {device}")
    
    # ==================
    # 创建数据集
    # ==================
    logger.info("创建数据集...")
    datasets = create_datasets(config)
    
    if datasets['train'] is None:
        logger.error("训练集为空!")
        return
    
    logger.info(f"训练集: {len(datasets['train'])} 样本")
    
    dataloaders = create_dataloaders(datasets, config)
    
    # ==================
    # 创建模型
    # ==================
    logger.info("创建模型...")
    model = create_model(config, device)
    
    logger.info(f"模型参数量: {count_parameters(model):,}")
    logger.info(f"  Encoder: {count_parameters(model.encoder):,}")
    logger.info(f"  Generator: {count_parameters(model.generator):,}")
    logger.info(f"  Discriminator: {count_parameters(model.discriminator):,}")
    
    # ==================
    # 创建优化器
    # ==================
    lr = train_config.get('learning_rate', 0.0002)
    betas = tuple(train_config.get('betas', [0.5, 0.999]))
    
    # D 单独优化器
    optimizer_d = Adam(
        model.discriminator.parameters(),
        lr=lr,
        betas=betas
    )
    
    # E 和 G 联合优化器
    optimizer_eg = Adam(
        list(model.encoder.parameters()) + list(model.generator.parameters()),
        lr=lr,
        betas=betas
    )
    
    # 学习率调度器
    epochs = train_config.get('epochs', 100)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=epochs)
    scheduler_eg = CosineAnnealingLR(optimizer_eg, T_max=epochs)
    
    # ==================
    # 创建损失函数
    # ==================
    loss_fn = EGANLoss(
        adv_weight=train_config.get('adv_weight', 1.0),
        recon_weight=train_config.get('recon_weight', 50.0),
        label_smoothing=train_config.get('label_smoothing', 0.1),
        device=str(device)
    )
    
    # ==================
    # 早停
    # ==================
    patience = train_config.get('early_stopping_patience')
    early_stopping = None
    if patience is not None and patience > 0:
        early_stopping = EarlyStopping(patience=patience, mode='min')
        logger.info(f"启用早停机制, patience={patience}")
    
    # ==================
    # 训练循环
    # ==================
    logger.info("开始训练...")
    start_time = time.time()
    
    history = {
        'loss_d': [],
        'loss_eg': [],
        'loss_recon': [],
        'loss_adv': [],
        'val_loss_recon': [],
    }
    
    best_recon_loss = float('inf')
    save_interval = train_config.get('save_interval', 10)
    val_interval = train_config.get('val_interval', 5)
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_losses = train_epoch(
            model, dataloaders['train'],
            optimizer_d, optimizer_eg,
            loss_fn, device, config,
            epoch, logger
        )
        
        # 更新学习率
        scheduler_d.step()
        scheduler_eg.step()
        
        # 记录历史
        for key, value in train_losses.items():
            if key in history:
                history[key].append(value)
        
        # 验证
        val_losses = None
        if val_interval > 0 and epoch % val_interval == 0:
            if dataloaders.get('val_normal') is not None:
                val_losses = validate(model, dataloaders['val_normal'], loss_fn, device)
                history['val_loss_recon'].append(val_losses['loss_recon'])
        
        # 日志
        epoch_time = time.time() - epoch_start
        log_msg = (
            f"Epoch {epoch}/{epochs} | "
            f"D: {train_losses['loss_d']:.4f} | "
            f"EG: {train_losses['loss_eg']:.4f} | "
            f"Recon: {train_losses['loss_recon']:.4f} | "
            f"Time: {format_time(epoch_time)}"
        )
        if val_losses:
            log_msg += f" | Val Recon: {val_losses['loss_recon']:.4f}"
        
        logger.info(log_msg)
        
        # 保存 checkpoint
        is_best = train_losses['loss_recon'] < best_recon_loss
        if is_best:
            best_recon_loss = train_losses['loss_recon']
        
        if epoch % save_interval == 0 or is_best:
            checkpoint_path = os.path.join(
                paths['checkpoints_dir'], 
                f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                    'optimizer_eg_state_dict': optimizer_eg.state_dict(),
                    'train_losses': train_losses,
                    'best_recon_loss': best_recon_loss,
                },
                checkpoint_path,
                is_best=is_best
            )
        
        # 早停检查
        if early_stopping is not None:
            if early_stopping(train_losses['loss_recon']):
                logger.info(f"早停触发，在 epoch {epoch}")
                break
    
    total_time = time.time() - start_time
    logger.info(f"训练完成! 总用时: {format_time(total_time)}")
    
    # ==================
    # 保存训练曲线
    # ==================
    plot_training_curves(
        history,
        os.path.join(paths['figures_dir'], 'training_curves.png'),
        show=False
    )
    
    # ==================
    # 保存重构示例
    # ==================
    if config.get('output', {}).get('save_recon_examples', True):
        logger.info("保存重构示例...")
        save_reconstruction_examples(
            model, dataloaders['train'], device,
            os.path.join(paths['figures_dir'], 'recon_examples'),
            num_examples=config.get('output', {}).get('num_examples_to_save', 10)
        )
    
    # ==================
    # 校准阈值
    # ==================
    logger.info("校准阈值...")
    
    # 优先使用验证集，否则使用训练集
    calibration_loader = dataloaders.get('val_normal') or dataloaders['train']
    
    threshold_info = calibrate_threshold(
        model, calibration_loader, config, str(device)
    )
    
    save_threshold(threshold_info, paths['threshold_file'])
    
    logger.info(f"阈值校准完成:")
    logger.info(f"  Threshold (total): {threshold_info['threshold_total']:.4f}")
    logger.info(f"  Target PFA: {threshold_info['target_pfa']}")
    logger.info(f"  基于 {threshold_info['n_samples']} 个正常样本")
    
    logger.info("训练流程完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='E-GAN Training')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从 checkpoint 恢复训练'
    )
    
    args = parser.parse_args()
    
    main(args)
