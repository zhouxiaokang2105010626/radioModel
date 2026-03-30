"""
run_demo.py - 一键运行演示脚本

直接运行: python run_demo.py
"""

# 必须在导入 torch 之前设置这些环境变量
# 如果直接运行此脚本仍有问题，请使用 run_final.cmd 或 final_run.py
if __name__ == "__main__":
    import os
    import sys
    
    # 检查环境变量是否已设置
    if os.environ.get('TORCH_DISABLE_DYNAMO') != '1':
        # 重新启动脚本，但带上环境变量
        import subprocess
        env = os.environ.copy()
        env['TORCH_DISABLE_DYNAMO'] = '1'
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run([sys.executable, __file__], env=env)
        sys.exit(result.returncode)

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TORCH_DISABLE_DYNAMO'] = '1'  # Disable torch._dynamo to fix ONNX import issue

# 设置项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

def main():
    import yaml
    import numpy as np
    import torch
    from tqdm import tqdm
    
    print("=" * 60)
    print("E-GAN 无线电信号异常检测 - 完整演示")
    print("=" * 60)
    
    # ==========================================
    # 1. 加载配置
    # ==========================================
    print("\n[1/6] 加载配置...")
    
    config_path = 'configs/demo.yaml'
    if not os.path.exists(config_path):
        config_path = 'configs/default.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 调整配置
    config['data']['root_dir'] = 'dataset'
    config['train']['epochs'] = 5  # 快速演示
    config['train']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['train']['batch_size'] = 16
    config['train']['num_workers'] = 0
    config['train']['save_interval'] = 5
    config['train']['val_interval'] = 2
    
    print(f"    训练轮数: {config['train']['epochs']}")
    print(f"    批量大小: {config['train']['batch_size']}")
    print(f"    设备: {config['train']['device']}")
    
    # ==========================================
    # 2. 准备数据
    # ==========================================
    print("\n[2/6] 准备数据...")
    
    from src.io_utils import get_iq_files
    from src.utils import set_seed, get_device, get_output_paths, save_checkpoint
    from src.dataset import create_datasets, create_dataloaders
    
    # 检查数据目录
    if not os.path.exists('dataset/train/normal'):
        print("    数据目录不存在，从 data/ 复制...")
        import shutil
        os.makedirs('dataset/train/normal', exist_ok=True)
        os.makedirs('dataset/val/normal', exist_ok=True)
        os.makedirs('dataset/test/normal', exist_ok=True)
        
        files = get_iq_files('data')
        n = len(files)
        n_train = max(1, n - 2)
        
        for f in files[:n_train]:
            shutil.copy2(f, 'dataset/train/normal/')
        for f in files[n_train:n_train+1]:
            shutil.copy2(f, 'dataset/val/normal/')
        for f in files[n_train+1:]:
            shutil.copy2(f, 'dataset/test/normal/')
    
    set_seed(42)
    device = get_device(config['train']['device'])
    paths = get_output_paths(config)
    
    datasets = create_datasets(config)
    dataloaders = create_dataloaders(datasets, config)
    
    print(f"    训练集: {len(datasets['train'])} 个窗口")
    if datasets.get('val_normal'):
        print(f"    验证集: {len(datasets['val_normal'])} 个窗口")
    
    # ==========================================
    # 3. 创建模型
    # ==========================================
    print("\n[3/6] 创建模型...")
    
    from src.model import create_model, count_parameters
    from src.losses import EGANLoss
    
    model = create_model(config, device)
    print(f"    总参数: {count_parameters(model):,}")
    
    # 优化器
    lr = config['train']['learning_rate']
    betas = tuple(config['train']['betas'])
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=betas)
    optimizer_eg = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.generator.parameters()),
        lr=lr, betas=betas
    )
    
    loss_fn = EGANLoss(
        adv_weight=config['train']['adv_weight'],
        recon_weight=config['train']['recon_weight'],
        label_smoothing=config['train']['label_smoothing'],
        device=str(device)
    )
    
    # ==========================================
    # 4. 训练
    # ==========================================
    print("\n[4/6] 开始训练...")
    start_time = time.time()
    
    epochs = config['train']['epochs']
    history = {'loss_d': [], 'loss_eg': [], 'loss_recon': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {'d': [], 'eg': [], 'recon': []}
        
        pbar = tqdm(dataloaders['train'], desc=f'Epoch {epoch}/{epochs}', leave=True)
        for batch in pbar:
            x = batch['spectrogram'].to(device)
            
            # Forward
            z, x_hat, d_real, d_fake = model(x)
            
            # Update D
            optimizer_d.zero_grad()
            d_real = model.discriminate(x)
            d_fake_det = model.discriminate(x_hat.detach())
            loss_d, _ = loss_fn.compute_d_loss(d_real, d_fake_det)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
            optimizer_d.step()
            
            # Update EG
            optimizer_eg.zero_grad()
            z = model.encode(x)
            x_hat = model.decode(z)
            d_fake = model.discriminate(x_hat)
            loss_eg, eg_dict = loss_fn.compute_eg_loss(x, x_hat, d_fake)
            loss_eg.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            optimizer_eg.step()
            
            epoch_losses['d'].append(loss_d.item())
            epoch_losses['eg'].append(loss_eg.item())
            epoch_losses['recon'].append(eg_dict['loss_recon'])
            
            pbar.set_postfix({
                'D': f"{loss_d.item():.4f}",
                'Recon': f"{eg_dict['loss_recon']:.4f}"
            })
        
        avg_d = np.mean(epoch_losses['d'])
        avg_eg = np.mean(epoch_losses['eg'])
        avg_recon = np.mean(epoch_losses['recon'])
        
        history['loss_d'].append(avg_d)
        history['loss_eg'].append(avg_eg)
        history['loss_recon'].append(avg_recon)
        
        print(f"    Epoch {epoch}: D={avg_d:.4f}, EG={avg_eg:.4f}, Recon={avg_recon:.4f}")
    
    train_time = time.time() - start_time
    print(f"\n    训练完成! 用时: {train_time:.1f}s ({train_time/60:.1f}min)")
    
    # 保存模型
    checkpoint_path = os.path.join(paths['checkpoints_dir'], 'checkpoint_final.pth')
    save_checkpoint({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path, is_best=True)
    print(f"    模型已保存: {checkpoint_path}")
    
    # ==========================================
    # 5. 校准阈值
    # ==========================================
    print("\n[5/6] 校准阈值...")
    
    from src.thresholds import calibrate_threshold, save_threshold
    
    cal_loader = dataloaders.get('val_normal') or dataloaders['train']
    threshold_info = calibrate_threshold(model, cal_loader, config, str(device))
    save_threshold(threshold_info, paths['threshold_file'])
    
    print(f"    阈值: {threshold_info['threshold_total']:.4f}")
    print(f"    目标PFA: {threshold_info['target_pfa']}")
    
    # ==========================================
    # 6. 快速评估
    # ==========================================
    print("\n[6/6] 快速评估...")
    
    from src.thresholds import compute_anomaly_score
    
    model.eval()
    all_scores = []
    
    test_loader = dataloaders.get('test_normal') or dataloaders.get('val_normal') or dataloaders['train']
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['spectrogram'].to(device)
            z, x_hat, d_real, d_fake = model(x)
            _, _, total = compute_anomaly_score(
                x, x_hat, d_real, d_fake,
                config['score']['lambda_score']
            )
            all_scores.extend(total.cpu().numpy().tolist())
    
    scores = np.array(all_scores)
    threshold = threshold_info['threshold_total']
    
    print(f"    测试样本数: {len(scores)}")
    print(f"    分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"    分数均值: {scores.mean():.4f}")
    print(f"    分数标准差: {scores.std():.4f}")
    
    n_anomaly = np.sum(scores > threshold)
    print(f"    检测为异常: {n_anomaly}/{len(scores)} ({100*n_anomaly/len(scores):.1f}%)")
    
    # ==========================================
    # 完成
    # ==========================================
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    print(f"""
输出文件:
  - 模型: {checkpoint_path}
  - 阈值: {paths['threshold_file']}

后续操作:
  1. 评估: python -m src.eval --config configs/demo.yaml
  2. 推理: python -m src.infer --config configs/demo.yaml --input data/xxx.IQ
""")


if __name__ == "__main__":
    main()
