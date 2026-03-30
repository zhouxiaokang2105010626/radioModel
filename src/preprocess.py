"""
preprocess.py - 信号预处理与 STFT 转换

主要功能:
1. IQ 数据预处理 (去直流、归一化)
2. 长序列切窗
3. STFT 计算
4. Spectrogram 生成与调整
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from scipy import signal
from PIL import Image
import warnings


# ============================================================
# IQ 预处理
# ============================================================

def remove_dc(data: np.ndarray) -> np.ndarray:
    """
    去除直流分量
    
    Args:
        data: complex64 IQ 数据
        
    Returns:
        去除直流后的数据
    """
    return data - np.mean(data)


def normalize_iq(
    data: np.ndarray, 
    mode: str = "rms"
) -> np.ndarray:
    """
    归一化 IQ 数据
    
    Args:
        data: complex64 IQ 数据
        mode: 归一化方式
            - "rms": RMS 归一化
            - "max": 最大幅度归一化
            - "none": 不归一化
            
    Returns:
        归一化后的数据
    """
    if mode == "none" or mode is None:
        return data
    
    if mode == "rms":
        rms = np.sqrt(np.mean(np.abs(data) ** 2))
        if rms > 0:
            return data / rms
        return data
    
    elif mode == "max":
        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data / max_val
        return data
    
    else:
        warnings.warn(f"未知的归一化方式: {mode}，不进行归一化")
        return data


def preprocess_iq(
    data: np.ndarray,
    remove_dc_flag: bool = True,
    normalize_mode: str = "rms"
) -> np.ndarray:
    """
    完整的 IQ 预处理流程
    
    Args:
        data: complex64 IQ 数据
        remove_dc_flag: 是否去除直流
        normalize_mode: 归一化方式
        
    Returns:
        预处理后的数据
    """
    if remove_dc_flag:
        data = remove_dc(data)
    
    data = normalize_iq(data, normalize_mode)
    
    return data


# ============================================================
# 切窗处理
# ============================================================

def sliding_window(
    data: np.ndarray,
    window_size: int,
    hop_size: int
) -> List[np.ndarray]:
    """
    对长序列进行滑动窗口切分
    
    Args:
        data: 一维数据数组
        window_size: 窗口大小
        hop_size: 滑动步长
        
    Returns:
        窗口列表
    """
    windows = []
    n_samples = len(data)
    
    start = 0
    while start + window_size <= n_samples:
        windows.append(data[start:start + window_size])
        start += hop_size
    
    return windows


def create_windows_with_metadata(
    data: np.ndarray,
    file_path: str,
    metadata: Dict[str, Any],
    window_size: int,
    hop_size: int
) -> List[Dict[str, Any]]:
    """
    创建带元数据的窗口列表
    
    Args:
        data: IQ 数据
        file_path: 文件路径
        metadata: 文件元数据
        window_size: 窗口大小
        hop_size: 滑动步长
        
    Returns:
        窗口信息列表，每个元素包含:
        - data: 窗口数据
        - file_path: 来源文件
        - window_index: 窗口索引
        - start_sample: 起始采样点
        - end_sample: 结束采样点
        - metadata: 文件元数据
    """
    windows = []
    n_samples = len(data)
    
    window_idx = 0
    start = 0
    
    while start + window_size <= n_samples:
        window_info = {
            'data': data[start:start + window_size].copy(),
            'file_path': file_path,
            'window_index': window_idx,
            'start_sample': start,
            'end_sample': start + window_size,
            'metadata': metadata.copy()
        }
        windows.append(window_info)
        
        start += hop_size
        window_idx += 1
    
    return windows


# ============================================================
# STFT 和 Spectrogram
# ============================================================

def compute_stft(
    data: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 64,
    win_length: Optional[int] = None,
    window_type: str = "hann"
) -> np.ndarray:
    """
    计算短时傅里叶变换
    
    Args:
        data: complex64 IQ 数据
        n_fft: FFT 点数
        hop_length: STFT hop length
        win_length: 窗口长度
        window_type: 窗函数类型
        
    Returns:
        STFT 结果，shape = (n_fft, n_frames)
    """
    if win_length is None:
        win_length = n_fft
    
    # 获取窗函数
    window = signal.get_window(window_type, win_length)
    
    # 计算 STFT
    f, t, Zxx = signal.stft(
        data,
        fs=1.0,  # 归一化频率
        window=window,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        return_onesided=False  # 复数信号需要双边谱
    )
    
    # 重新排列频率轴，将 DC 放在中心
    Zxx = np.fft.fftshift(Zxx, axes=0)
    
    return Zxx


def stft_to_spectrogram(
    stft_result: np.ndarray,
    power: float = 1.0,
    use_log: bool = True,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    将 STFT 结果转换为 spectrogram
    
    Args:
        stft_result: STFT 结果
        power: 功率指数 (1=幅度, 2=功率)
        use_log: 是否使用 log 压缩
        epsilon: log 计算时的小量，防止 log(0)
        
    Returns:
        Spectrogram，shape = (n_freq, n_time)
    """
    # 计算幅度
    spec = np.abs(stft_result) ** power
    
    # Log 压缩
    if use_log:
        spec = np.log1p(spec)
    
    return spec.astype(np.float32)


def normalize_spectrogram(
    spec: np.ndarray,
    mode: str = "minmax"
) -> np.ndarray:
    """
    归一化 spectrogram
    
    Args:
        spec: Spectrogram
        mode: 归一化方式
            - "minmax": [0, 1] 归一化
            - "standard": 标准化 (mean=0, std=1)
            - "none": 不归一化
            
    Returns:
        归一化后的 spectrogram
    """
    if mode == "none" or mode is None:
        return spec
    
    if mode == "minmax":
        min_val = spec.min()
        max_val = spec.max()
        if max_val - min_val > 1e-10:
            return (spec - min_val) / (max_val - min_val)
        return spec - min_val
    
    elif mode == "standard":
        mean_val = spec.mean()
        std_val = spec.std()
        if std_val > 1e-10:
            return (spec - mean_val) / std_val
        return spec - mean_val
    
    else:
        warnings.warn(f"未知的归一化方式: {mode}")
        return spec


def resize_spectrogram(
    spec: np.ndarray,
    target_h: int,
    target_w: int
) -> np.ndarray:
    """
    调整 spectrogram 尺寸
    
    Args:
        spec: Spectrogram, shape = (H, W)
        target_h: 目标高度
        target_w: 目标宽度
        
    Returns:
        调整后的 spectrogram
    """
    if spec.shape[0] == target_h and spec.shape[1] == target_w:
        return spec
    
    # 使用 PIL 进行双线性插值
    img = Image.fromarray(spec)
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    
    return np.array(img_resized, dtype=np.float32)


# ============================================================
# 完整预处理流水线
# ============================================================

def iq_to_spectrogram(
    iq_data: np.ndarray,
    config: dict
) -> np.ndarray:
    """
    完整的 IQ -> Spectrogram 转换流水线
    
    Args:
        iq_data: complex64 IQ 数据 (单个窗口)
        config: 配置字典，包含 iq, spectrogram 等配置
        
    Returns:
        Spectrogram, shape = (1, H, W)，单通道
    """
    iq_cfg = config.get('iq', {})
    spec_cfg = config.get('spectrogram', {})
    
    # 1. IQ 预处理
    data = preprocess_iq(
        iq_data,
        remove_dc_flag=iq_cfg.get('remove_dc', True),
        normalize_mode=iq_cfg.get('normalize_mode', 'rms')
    )
    
    # 2. 计算 STFT
    stft_result = compute_stft(
        data,
        n_fft=spec_cfg.get('n_fft', 256),
        hop_length=spec_cfg.get('hop_length', 64),
        win_length=spec_cfg.get('win_length', None),
        window_type=spec_cfg.get('window_type', 'hann')
    )
    
    # 3. 转换为 spectrogram
    spec = stft_to_spectrogram(
        stft_result,
        power=spec_cfg.get('power', 1.0),
        use_log=spec_cfg.get('use_log', True)
    )
    
    # 4. 调整尺寸
    spec = resize_spectrogram(
        spec,
        target_h=spec_cfg.get('resize_h', 64),
        target_w=spec_cfg.get('resize_w', 64)
    )
    
    # 5. 归一化
    spec = normalize_spectrogram(
        spec,
        mode=spec_cfg.get('normalize', 'minmax')
    )
    
    # 6. 添加通道维度
    spec = spec[np.newaxis, :, :]  # (1, H, W)
    
    return spec


def process_file_to_spectrograms(
    file_path: str,
    config: dict,
    iq_config: dict
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    处理单个 IQ 文件，生成所有窗口的 spectrogram
    
    Args:
        file_path: IQ 文件路径
        config: 完整配置
        iq_config: IQ 读取配置
        
    Returns:
        (spectrogram 列表, 窗口元数据列表)
    """
    from .io_utils import read_iq_auto
    
    windowing_cfg = config.get('windowing', {})
    
    # 读取 IQ 文件
    data, metadata = read_iq_auto(
        file_path,
        iq_config,
        use_wav_fallback=config.get('data', {}).get('use_wav_fallback', False)
    )
    
    # 切窗
    windows = create_windows_with_metadata(
        data,
        file_path,
        metadata,
        window_size=windowing_cfg.get('window_size_samples', 8192),
        hop_size=windowing_cfg.get('hop_size_samples', 4096)
    )
    
    # 处理每个窗口
    spectrograms = []
    window_infos = []
    
    for win_info in windows:
        spec = iq_to_spectrogram(win_info['data'], config)
        spectrograms.append(spec)
        
        # 去除原始数据以节省内存
        win_info_copy = {k: v for k, v in win_info.items() if k != 'data'}
        window_infos.append(win_info_copy)
    
    return spectrograms, window_infos


# ============================================================
# 命令行预览工具
# ============================================================

def preview_spectrogram(
    spec: np.ndarray,
    title: str = "Spectrogram",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    预览 spectrogram
    
    Args:
        spec: Spectrogram, shape = (H, W) 或 (1, H, W)
        title: 图标题
        save_path: 保存路径
        show: 是否显示
    """
    import matplotlib.pyplot as plt
    
    if spec.ndim == 3:
        spec = spec[0]
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bin')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    import sys
    import yaml
    from .io_utils import read_iq_auto, parse_iq_filename_metadata, sanity_check_iq
    
    # 默认配置
    default_config = {
        'iq': {
            'dtype': 'int16',
            'interleaved': True,
            'little_endian': True,
            'remove_dc': True,
            'normalize_mode': 'rms'
        },
        'windowing': {
            'window_size_samples': 8192,
            'hop_size_samples': 4096
        },
        'spectrogram': {
            'n_fft': 256,
            'hop_length': 64,
            'window_type': 'hann',
            'power': 1.0,
            'use_log': True,
            'resize_h': 64,
            'resize_w': 64,
            'normalize': 'minmax'
        }
    }
    
    if len(sys.argv) > 1:
        # 加载配置 (如果提供)
        if '--config' in sys.argv:
            config_idx = sys.argv.index('--config')
            config_path = sys.argv[config_idx + 1]
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = default_config
        
        # 获取输入文件
        if '--preview' in sys.argv:
            preview_idx = sys.argv.index('--preview')
            input_path = sys.argv[preview_idx + 1]
        else:
            input_path = sys.argv[-1]
        
        print(f"处理文件: {input_path}")
        
        # 读取数据
        data, metadata = read_iq_auto(
            input_path,
            config.get('iq', default_config['iq']),
            use_wav_fallback=False
        )
        
        print(f"元数据: {metadata}")
        print(f"数据长度: {len(data)} samples")
        
        # Sanity check
        sanity_check_iq(data, metadata, show=True)
        
        # 生成 spectrogram
        spectrograms, window_infos = process_file_to_spectrograms(
            input_path, config, config.get('iq', default_config['iq'])
        )
        
        print(f"生成 {len(spectrograms)} 个窗口")
        
        # 显示第一个 spectrogram
        if spectrograms:
            preview_spectrogram(spectrograms[0], title="First Window Spectrogram")
    else:
        print("用法: python -m src.preprocess --config configs/default.yaml --preview path/to/file.IQ")
