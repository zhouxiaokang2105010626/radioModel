"""
quick_start.py - 快速开始脚本

演示如何使用现有数据文件快速跑通整个流程
"""

import os
import sys
import shutil

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def setup_demo_data():
    """
    使用现有的 data 目录文件设置演示数据
    """
    data_dir = os.path.join(project_root, 'data')
    dataset_dir = os.path.join(project_root, 'dataset')
    
    # 检查原始数据
    from src.io_utils import get_iq_files
    files = get_iq_files(data_dir)
    
    if not files:
        print("错误: data 目录中没有 IQ 文件!")
        print(f"请将 IQ 文件放入: {data_dir}")
        return False
    
    print(f"找到 {len(files)} 个 IQ 文件")
    
    # 创建数据集目录
    dirs = [
        'train/normal',
        'val/normal', 
        'test/normal',
    ]
    
    for d in dirs:
        path = os.path.join(dataset_dir, d)
        os.makedirs(path, exist_ok=True)
    
    # 简单分割: 70% train, 15% val, 15% test
    n = len(files)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:] if n_train + n_val < n else [files[-1]]
    
    # 确保每个集合至少有一个文件
    if not val_files and len(train_files) > 1:
        val_files = [train_files[-1]]
        train_files = train_files[:-1]
    if not test_files:
        test_files = val_files[-1:] if val_files else train_files[-1:]
    
    print(f"\n分割结果:")
    print(f"  训练集: {len(train_files)} 个文件")
    print(f"  验证集: {len(val_files)} 个文件")
    print(f"  测试集: {len(test_files)} 个文件")
    
    # 复制文件 (使用符号链接以节省空间，如果失败则复制)
    def copy_files(src_files, dest_dir):
        for src in src_files:
            dst = os.path.join(dest_dir, os.path.basename(src))
            if not os.path.exists(dst):
                try:
                    # 尝试创建符号链接
                    os.symlink(src, dst)
                except (OSError, NotImplementedError):
                    # 符号链接失败，使用复制
                    shutil.copy2(src, dst)
    
    copy_files(train_files, os.path.join(dataset_dir, 'train/normal'))
    copy_files(val_files, os.path.join(dataset_dir, 'val/normal'))
    copy_files(test_files, os.path.join(dataset_dir, 'test/normal'))
    
    print(f"\n数据集已准备: {dataset_dir}")
    return True


def update_config_for_demo():
    """
    更新配置文件以使用 dataset 目录
    """
    import yaml
    
    config_path = os.path.join(project_root, 'configs', 'default.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新数据路径
    config['data']['root_dir'] = 'dataset'
    
    # 减少训练轮数以便快速测试
    config['train']['epochs'] = 20
    config['train']['save_interval'] = 5
    config['train']['val_interval'] = 5
    
    # 如果没有 GPU，使用 CPU
    import torch
    if not torch.cuda.is_available():
        print("注意: CUDA 不可用，将使用 CPU 训练 (会很慢)")
        config['train']['device'] = 'cpu'
        config['train']['batch_size'] = 8  # 减小批量大小
    
    # 保存更新后的配置
    demo_config_path = os.path.join(project_root, 'configs', 'demo.yaml')
    with open(demo_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"演示配置已保存: {demo_config_path}")
    return demo_config_path


def print_next_steps(config_path):
    """打印下一步操作"""
    print("\n" + "=" * 60)
    print("数据准备完成! 请按以下步骤继续:")
    print("=" * 60)
    
    print(f"""
1. 安装依赖:
   pip install -r requirements.txt

2. 训练模型:
   python -m src.train --config {config_path}

3. 评估模型:
   python -m src.eval --config {config_path}

4. 推理单个文件:
   python -m src.infer --config {config_path} --input data/xxx.IQ

5. 查看结果:
   - 训练曲线: outputs/figures/training_curves.png
   - 重构示例: outputs/figures/recon_examples/
   - 评估指标: outputs/metrics.json
   - 预测结果: outputs/predictions_window.csv
""")


def main():
    print("=" * 60)
    print("E-GAN Radio Anomaly Detection - 快速开始")
    print("=" * 60)
    
    # 1. 设置演示数据
    print("\n[1/3] 准备演示数据...")
    if not setup_demo_data():
        return
    
    # 2. 检查依赖
    print("\n[2/3] 检查依赖...")
    try:
        import torch
        import numpy
        import scipy
        import sklearn
        import PIL
        print(f"  PyTorch: {torch.__version__}")
        print(f"  NumPy: {numpy.__version__}")
        print(f"  SciPy: {scipy.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  警告: 缺少依赖 - {e}")
        print("  请运行: pip install -r requirements.txt")
    
    # 3. 更新配置
    print("\n[3/3] 生成演示配置...")
    try:
        config_path = update_config_for_demo()
    except Exception as e:
        print(f"  警告: 配置更新失败 - {e}")
        config_path = "configs/default.yaml"
    
    # 打印下一步
    print_next_steps(config_path)


if __name__ == "__main__":
    main()
