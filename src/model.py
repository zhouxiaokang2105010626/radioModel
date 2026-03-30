"""
model.py - E-GAN 模型定义

E-GAN 结构:
- Encoder (E): spectrogram -> latent vector z
- Generator (G): z -> reconstructed spectrogram
- Discriminator (D): 区分真实 vs 重构 spectrogram

设计原则:
- 轻量级 CNN 结构
- 稳定易训练
- 支持 spectral normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Tuple


def get_norm_layer(use_spectral_norm: bool = True):
    """获取归一化包装器"""
    if use_spectral_norm:
        return spectral_norm
    return lambda x: x


class Encoder(nn.Module):
    """
    Encoder 网络
    
    将 spectrogram 编码为 latent vector
    
    结构: Conv2d -> LeakyReLU -> Conv2d -> ... -> Flatten -> Linear
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        base_channels: int = 32,
        input_size: int = 64
    ):
        """
        Args:
            in_channels: 输入通道数
            latent_dim: 隐空间维度
            base_channels: 基础通道数
            input_size: 输入图像尺寸 (假设正方形)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # 编码器主干
        # 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 -> 16
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 8
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 4
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 计算 flatten 后的维度
        # 对于 64x64 输入: 4x4 x (base_channels*8)
        self.flatten_dim = (input_size // 16) ** 2 * base_channels * 8
        
        # 映射到 latent space
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim * 2, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 spectrogram, shape = (B, 1, H, W)
            
        Returns:
            latent vector z, shape = (B, latent_dim)
        """
        features = self.encoder(x)
        z = self.fc(features)
        return z


class Generator(nn.Module):
    """
    Generator 网络
    
    从 latent vector 生成 spectrogram
    
    结构: Linear -> Reshape -> ConvTranspose2d -> ... -> Tanh
    """
    
    def __init__(
        self,
        out_channels: int = 1,
        latent_dim: int = 128,
        base_channels: int = 32,
        output_size: int = 64
    ):
        """
        Args:
            out_channels: 输出通道数
            latent_dim: 隐空间维度
            base_channels: 基础通道数
            output_size: 输出图像尺寸
        """
        super().__init__()
        
        self.output_size = output_size
        self.base_channels = base_channels
        
        # 初始尺寸 4x4
        self.init_size = output_size // 16
        self.init_channels = base_channels * 8
        
        # 映射层
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim * 2, self.init_channels * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 解码器主干
        # 4 -> 8 -> 16 -> 32 -> 64
        self.decoder = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose2d(self.init_channels, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # 8 -> 16
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # 16 -> 32
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 32 -> 64
            nn.ConvTranspose2d(base_channels, out_channels, 4, 2, 1),
            nn.Sigmoid(),  # 输出归一化到 [0, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: latent vector, shape = (B, latent_dim)
            
        Returns:
            重构的 spectrogram, shape = (B, 1, H, W)
        """
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator 网络
    
    区分真实 spectrogram 和重构 spectrogram
    
    结构: Conv2d -> LeakyReLU -> Conv2d -> ... -> Linear -> scalar
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        input_size: int = 64,
        use_spectral_norm: bool = True
    ):
        """
        Args:
            in_channels: 输入通道数
            base_channels: 基础通道数
            input_size: 输入图像尺寸
            use_spectral_norm: 是否使用 spectral normalization
        """
        super().__init__()
        
        norm = get_norm_layer(use_spectral_norm)
        
        # 判别器主干
        # 64 -> 32 -> 16 -> 8 -> 4
        self.features = nn.Sequential(
            # 64 -> 32
            norm(nn.Conv2d(in_channels, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 -> 16
            norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 8
            norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 4
            norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 计算 flatten 后的维度
        self.flatten_dim = (input_size // 16) ** 2 * base_channels * 8
        
        # 输出层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            norm(nn.Linear(self.flatten_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  # 输出 logit
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 spectrogram, shape = (B, 1, H, W)
            
        Returns:
            判别器输出 logit, shape = (B, 1)
        """
        features = self.features(x)
        logit = self.classifier(features)
        return logit


class EGAN(nn.Module):
    """
    E-GAN 完整模型
    
    包含 Encoder, Generator, Discriminator
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        base_channels: int = 32,
        input_size: int = 64,
        use_spectral_norm: bool = True
    ):
        """
        Args:
            in_channels: 输入通道数
            latent_dim: 隐空间维度
            base_channels: 基础通道数
            input_size: 输入/输出图像尺寸
            use_spectral_norm: 判别器是否使用 spectral norm
        """
        super().__init__()
        
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            input_size=input_size
        )
        
        self.generator = Generator(
            out_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            output_size=input_size
        )
        
        self.discriminator = Discriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            input_size=input_size,
            use_spectral_norm=use_spectral_norm
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码"""
        return self.generator(z)
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """重构"""
        z = self.encode(x)
        return self.decode(z)
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """判别"""
        return self.discriminator(x)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整前向传播
        
        Args:
            x: 输入 spectrogram
            
        Returns:
            (z, x_hat, d_real, d_fake)
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        d_real = self.discriminate(x)
        d_fake = self.discriminate(x_hat)
        
        return z, x_hat, d_real, d_fake


def create_model(config: dict, device: str = 'cuda') -> EGAN:
    """
    根据配置创建模型
    
    Args:
        config: 配置字典
        device: 设备
        
    Returns:
        EGAN 模型
    """
    model_config = config.get('model', {})
    spec_config = config.get('spectrogram', {})
    
    input_size = spec_config.get('resize_h', 64)  # 假设正方形
    
    model = EGAN(
        in_channels=1,
        latent_dim=model_config.get('latent_dim', 128),
        base_channels=model_config.get('base_channels', 32),
        input_size=input_size,
        use_spectral_norm=model_config.get('use_spectral_norm', True)
    )
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    print("测试 E-GAN 模型...")
    
    model = EGAN(
        in_channels=1,
        latent_dim=128,
        base_channels=32,
        input_size=64,
        use_spectral_norm=True
    )
    
    print(f"Encoder 参数: {count_parameters(model.encoder):,}")
    print(f"Generator 参数: {count_parameters(model.generator):,}")
    print(f"Discriminator 参数: {count_parameters(model.discriminator):,}")
    print(f"总参数: {count_parameters(model):,}")
    
    # 测试前向传播
    x = torch.randn(4, 1, 64, 64)
    z, x_hat, d_real, d_fake = model(x)
    
    print(f"\n输入形状: {x.shape}")
    print(f"Latent 形状: {z.shape}")
    print(f"重构形状: {x_hat.shape}")
    print(f"D(real) 形状: {d_real.shape}")
    print(f"D(fake) 形状: {d_fake.shape}")
