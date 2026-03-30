"""
io_utils.py - IQ 文件读取与元数据解析

主要功能:
1. 从文件名解析 metadata (中心频率、带宽、采样率等)
2. 读取二进制 IQ 文件为 complex64 数组
3. 可选读取 .IQ.wav 文件
4. Sanity check 可视化
"""

import os
import re
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# ============================================================
# 元数据解析
# ============================================================

def parse_freq_string(freq_str: str) -> Optional[float]:
    """
    解析频率字符串，返回 Hz
    
    支持格式:
    - "161.975MHz" -> 161975000.0
    - "30kHz" -> 30000.0
    - "1.5GHz" -> 1500000000.0
    
    Args:
        freq_str: 频率字符串
        
    Returns:
        频率值 (Hz)，解析失败返回 None
    """
    freq_str = freq_str.strip().lower()
    
    # 匹配数字和单位
    match = re.match(r'^([\d.]+)\s*(hz|khz|mhz|ghz)?$', freq_str)
    if not match:
        return None
    
    value = float(match.group(1))
    unit = match.group(2) or 'hz'
    
    multipliers = {
        'hz': 1.0,
        'khz': 1e3,
        'mhz': 1e6,
        'ghz': 1e9
    }
    
    return value * multipliers.get(unit, 1.0)


def parse_iq_filename_metadata(path: str) -> Dict[str, Any]:
    """
    从 IQ 文件名解析元数据
    
    文件名格式: 中心频率_带宽_采样率_位深_时间戳.IQ
    示例: 161.975MHz_30kHz_64000_16_20250729082303.IQ
    
    Args:
        path: IQ 文件路径
        
    Returns:
        字典包含:
        - center_freq_hz: 中心频率 (Hz)
        - bandwidth_hz: 带宽 (Hz)
        - sample_rate_hz: 采样率 (Hz)
        - bit_depth: 位深度
        - timestamp: 时间戳字符串
        
        如果解析失败，对应字段为 None
    """
    metadata = {
        'center_freq_hz': None,
        'bandwidth_hz': None,
        'sample_rate_hz': None,
        'bit_depth': None,
        'timestamp': None,
        'filepath': str(path)
    }
    
    # 获取文件名 (去除 .IQ 和可能的 .wav 后缀)
    filename = os.path.basename(path)
    # 去除所有后缀
    base_name = filename
    for ext in ['.wav', '.IQ', '.iq']:
        if base_name.lower().endswith(ext.lower()):
            base_name = base_name[:-len(ext)]
    
    # 尝试按下划线分割
    parts = base_name.split('_')
    
    if len(parts) >= 5:
        try:
            # 中心频率
            metadata['center_freq_hz'] = parse_freq_string(parts[0])
            
            # 带宽
            metadata['bandwidth_hz'] = parse_freq_string(parts[1])
            
            # 采样率 (可能是纯数字或带单位)
            sr_str = parts[2]
            if sr_str.isdigit():
                metadata['sample_rate_hz'] = float(sr_str)
            else:
                metadata['sample_rate_hz'] = parse_freq_string(sr_str)
            
            # 位深度
            metadata['bit_depth'] = int(parts[3])
            
            # 时间戳
            metadata['timestamp'] = parts[4]
            
        except (ValueError, IndexError) as e:
            warnings.warn(f"解析文件名 '{filename}' 时部分字段失败: {e}")
    else:
        warnings.warn(
            f"文件名 '{filename}' 格式不符合预期 "
            f"(期望: 中心频率_带宽_采样率_位深_时间戳.IQ)，元数据将为空"
        )
    
    return metadata


# ============================================================
# IQ 文件读取
# ============================================================

def read_iq_file(
    path: str,
    dtype: str = "int16",
    interleaved: bool = True,
    little_endian: bool = True,
    max_samples: Optional[int] = None
) -> np.ndarray:
    """
    读取二进制 IQ 文件，返回 complex64 数组
    
    Args:
        path: IQ 文件路径
        dtype: 数据类型 ("int16", "int8", "float32")
        interleaved: 是否交织存储 (I,Q,I,Q,...)
        little_endian: 是否小端序
        max_samples: 最大读取复数样本数 (None 表示全部读取)
        
    Returns:
        complex64 数组，shape = (n_samples,)
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 数据格式错误
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"IQ 文件不存在: {path}")
    
    # 确定 numpy dtype
    dtype_map = {
        'int16': np.dtype('<i2') if little_endian else np.dtype('>i2'),
        'int8': np.dtype('i1'),
        'float32': np.dtype('<f4') if little_endian else np.dtype('>f4'),
        'float64': np.dtype('<f8') if little_endian else np.dtype('>f8'),
    }
    
    if dtype not in dtype_map:
        raise ValueError(f"不支持的数据类型: {dtype}，支持: {list(dtype_map.keys())}")
    
    np_dtype = dtype_map[dtype]
    
    # 读取原始数据
    try:
        raw_data = np.fromfile(path, dtype=np_dtype)
    except Exception as e:
        raise ValueError(f"读取文件 {path} 失败: {e}")
    
    if len(raw_data) == 0:
        raise ValueError(f"文件 {path} 为空或读取失败")
    
    # 转换为复数
    if interleaved:
        # I,Q,I,Q,... 格式
        if len(raw_data) % 2 != 0:
            warnings.warn(f"文件 {path} 样本数为奇数，截断最后一个样本")
            raw_data = raw_data[:-1]
        
        # 重塑为 (n_samples, 2)
        iq_pairs = raw_data.reshape(-1, 2)
        complex_data = iq_pairs[:, 0].astype(np.float32) + 1j * iq_pairs[:, 1].astype(np.float32)
    else:
        # 前半部分是 I，后半部分是 Q
        n_samples = len(raw_data) // 2
        i_data = raw_data[:n_samples].astype(np.float32)
        q_data = raw_data[n_samples:2*n_samples].astype(np.float32)
        complex_data = i_data + 1j * q_data
    
    # 限制样本数
    if max_samples is not None and len(complex_data) > max_samples:
        complex_data = complex_data[:max_samples]
    
    return complex_data.astype(np.complex64)


def read_iq_wav_file(path: str) -> Tuple[np.ndarray, int]:
    """
    读取 .IQ.wav 文件 (可选备选方案)
    
    假设 wav 文件是立体声，左声道为 I，右声道为 Q
    
    Args:
        path: wav 文件路径
        
    Returns:
        (complex64 数组, 采样率)
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("读取 wav 文件需要安装 soundfile: pip install soundfile")
    
    data, sample_rate = sf.read(path)
    
    if data.ndim == 1:
        # 单声道，无法获取 Q 分量
        warnings.warn(f"wav 文件 {path} 是单声道，Q 分量设为 0")
        complex_data = data.astype(np.float32) + 0j
    elif data.ndim == 2:
        # 立体声: 左=I, 右=Q
        i_data = data[:, 0].astype(np.float32)
        q_data = data[:, 1].astype(np.float32)
        complex_data = i_data + 1j * q_data
    else:
        raise ValueError(f"wav 文件 {path} 维度异常: {data.shape}")
    
    return complex_data.astype(np.complex64), sample_rate


def read_iq_auto(
    path: str,
    iq_config: dict,
    use_wav_fallback: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    自动读取 IQ 文件，返回复数数据和元数据
    
    Args:
        path: 文件路径
        iq_config: IQ 读取配置字典
        use_wav_fallback: 是否尝试 wav 备选
        
    Returns:
        (complex64 数组, metadata 字典)
    """
    path = Path(path)
    
    # 解析元数据
    metadata = parse_iq_filename_metadata(str(path))
    
    # 尝试读取 .IQ 文件
    if path.suffix.lower() == '.iq' or str(path).lower().endswith('.iq'):
        try:
            data = read_iq_file(
                path,
                dtype=iq_config.get('dtype', 'int16'),
                interleaved=iq_config.get('interleaved', True),
                little_endian=iq_config.get('little_endian', True)
            )
            return data, metadata
        except Exception as e:
            if use_wav_fallback:
                warnings.warn(f"读取 IQ 文件失败: {e}，尝试 wav 备选")
            else:
                raise
    
    # 尝试 wav 备选
    wav_path = Path(str(path) + '.wav') if not str(path).endswith('.wav') else path
    if use_wav_fallback and wav_path.exists():
        data, sr = read_iq_wav_file(str(wav_path))
        if metadata['sample_rate_hz'] is None:
            metadata['sample_rate_hz'] = sr
        return data, metadata
    
    # 如果是 .wav 文件直接读取
    if path.suffix.lower() == '.wav':
        data, sr = read_iq_wav_file(str(path))
        if metadata['sample_rate_hz'] is None:
            metadata['sample_rate_hz'] = sr
        return data, metadata
    
    raise FileNotFoundError(f"无法读取文件: {path}")


# ============================================================
# 文件列表获取
# ============================================================

def get_iq_files(
    directory: str,
    extensions: Tuple[str, ...] = ('.IQ', '.iq'),
    recursive: bool = False
) -> list:
    """
    获取目录下所有 IQ 文件
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名
        recursive: 是否递归搜索
        
    Returns:
        文件路径列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    files = []
    
    if recursive:
        for ext in extensions:
            files.extend(directory.rglob(f'*{ext}'))
    else:
        for ext in extensions:
            files.extend(directory.glob(f'*{ext}'))
    
    # 过滤掉 .IQ.wav 文件，只保留 .IQ
    files = [f for f in files if not str(f).lower().endswith('.iq.wav')]
    
    return sorted([str(f) for f in files])


def filter_files_by_metadata(
    files: list,
    filter_center_freq: Optional[float] = None,
    filter_sample_rate: Optional[float] = None,
    filter_bandwidth: Optional[float] = None,
    tolerance: float = 0.01  # 1% 容差
) -> list:
    """
    根据元数据过滤文件列表
    
    Args:
        files: 文件路径列表
        filter_center_freq: 中心频率过滤值 (Hz)
        filter_sample_rate: 采样率过滤值 (Hz)
        filter_bandwidth: 带宽过滤值 (Hz)
        tolerance: 数值比较容差
        
    Returns:
        过滤后的文件列表
    """
    if filter_center_freq is None and filter_sample_rate is None and filter_bandwidth is None:
        return files
    
    filtered = []
    
    for path in files:
        metadata = parse_iq_filename_metadata(path)
        
        # 检查中心频率
        if filter_center_freq is not None:
            if metadata['center_freq_hz'] is None:
                continue
            if abs(metadata['center_freq_hz'] - filter_center_freq) / filter_center_freq > tolerance:
                continue
        
        # 检查采样率
        if filter_sample_rate is not None:
            if metadata['sample_rate_hz'] is None:
                continue
            if abs(metadata['sample_rate_hz'] - filter_sample_rate) / filter_sample_rate > tolerance:
                continue
        
        # 检查带宽
        if filter_bandwidth is not None:
            if metadata['bandwidth_hz'] is None:
                continue
            if abs(metadata['bandwidth_hz'] - filter_bandwidth) / filter_bandwidth > tolerance:
                continue
        
        filtered.append(path)
    
    return filtered


# ============================================================
# Sanity Check 可视化
# ============================================================

def sanity_check_iq(
    data: np.ndarray,
    metadata: Dict[str, Any],
    num_samples_plot: int = 1000,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    IQ 数据健全性检查，生成可视化图表
    
    Args:
        data: complex64 IQ 数据
        metadata: 元数据字典
        num_samples_plot: 绘制的样本点数
        save_path: 图片保存路径
        show: 是否显示图片
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. I/Q 波形
    ax = axes[0, 0]
    t = np.arange(min(num_samples_plot, len(data)))
    ax.plot(t, data[:len(t)].real, 'b-', alpha=0.7, label='I (In-phase)')
    ax.plot(t, data[:len(t)].imag, 'r-', alpha=0.7, label='Q (Quadrature)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.set_title('I/Q Waveform (First {} samples)'.format(len(t)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 幅度直方图
    ax = axes[0, 1]
    magnitude = np.abs(data)
    ax.hist(magnitude, bins=100, density=True, alpha=0.7, color='green')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Magnitude Histogram')
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'Mean: {magnitude.mean():.2f}\nStd: {magnitude.std():.2f}\nMax: {magnitude.max():.2f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. IQ 星座图
    ax = axes[1, 0]
    sample_indices = np.random.choice(len(data), min(5000, len(data)), replace=False)
    ax.scatter(data[sample_indices].real, data[sample_indices].imag, 
               alpha=0.3, s=1, c='blue')
    ax.set_xlabel('I (In-phase)')
    ax.set_ylabel('Q (Quadrature)')
    ax.set_title('IQ Constellation (5000 samples)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 4. PSD (功率谱密度)
    ax = axes[1, 1]
    sample_rate = metadata.get('sample_rate_hz', 1.0)
    
    # 计算 PSD
    from scipy import signal
    f, psd = signal.welch(data, fs=sample_rate, nperseg=min(1024, len(data)))
    
    ax.semilogy(f / 1e3, psd)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('PSD')
    ax.set_title('Power Spectral Density')
    ax.grid(True, alpha=0.3)
    
    # 添加元数据信息
    meta_str = (
        f"File: {os.path.basename(metadata.get('filepath', 'Unknown'))}\n"
        f"Center Freq: {metadata.get('center_freq_hz', 'N/A')} Hz\n"
        f"Bandwidth: {metadata.get('bandwidth_hz', 'N/A')} Hz\n"
        f"Sample Rate: {metadata.get('sample_rate_hz', 'N/A')} Hz\n"
        f"Total Samples: {len(data)}"
    )
    fig.suptitle(meta_str, fontsize=10, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sanity check 图片已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # 简单测试
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"测试读取: {test_path}")
        
        metadata = parse_iq_filename_metadata(test_path)
        print(f"元数据: {metadata}")
        
        data = read_iq_file(test_path)
        print(f"数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"前10个样本: {data[:10]}")
        
        sanity_check_iq(data, metadata)
    else:
        print("用法: python io_utils.py path/to/file.IQ")
