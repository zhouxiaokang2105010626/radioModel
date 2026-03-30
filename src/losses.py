"""
losses.py - 损失函数定义

包含:
- 判别器损失 (D loss)
- 编码器-生成器损失 (EG loss)
- 重构损失 (Reconstruction loss)
- 对抗损失 (Adversarial loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class GANLoss:
    """
    GAN 损失函数管理器
    
    支持:
    - BCEWithLogits loss
    - 标签平滑
    """
    
    def __init__(
        self,
        label_smoothing: float = 0.0,
        device: str = 'cuda'
    ):
        """
        Args:
            label_smoothing: 标签平滑值 (0-0.3 推荐)
            device: 设备
        """
        self.label_smoothing = label_smoothing
        self.device = device
        
        # 真/假标签
        self.real_label = 1.0 - label_smoothing
        self.fake_label = label_smoothing
        
        # BCE 损失
        self.criterion = nn.BCEWithLogitsLoss()
    
    def get_target_tensor(
        self, 
        prediction: torch.Tensor, 
        target_is_real: bool
    ) -> torch.Tensor:
        """
        创建目标标签张量
        
        Args:
            prediction: 判别器输出
            target_is_real: 是否为真实标签
            
        Returns:
            目标标签张量
        """
        if target_is_real:
            target_val = self.real_label
        else:
            target_val = self.fake_label
        
        return torch.full_like(prediction, target_val, device=self.device)
    
    def __call__(
        self, 
        prediction: torch.Tensor, 
        target_is_real: bool
    ) -> torch.Tensor:
        """
        计算 GAN 损失
        
        Args:
            prediction: 判别器输出 (logits)
            target_is_real: 是否为真实标签
            
        Returns:
            损失值
        """
        target = self.get_target_tensor(prediction, target_is_real)
        loss = self.criterion(prediction, target)
        return loss


def compute_discriminator_loss(
    d_real: torch.Tensor,
    d_fake: torch.Tensor,
    gan_loss: GANLoss
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算判别器损失
    
    D 的目标: 
    - 对真实样本输出高分 (接近 1)
    - 对假样本输出低分 (接近 0)
    
    Args:
        d_real: D(x) 输出
        d_fake: D(G(E(x))) 输出 (已 detach)
        gan_loss: GAN 损失函数
        
    Returns:
        (总损失, 损失详情字典)
    """
    # 真实样本损失
    loss_real = gan_loss(d_real, target_is_real=True)
    
    # 假样本损失
    loss_fake = gan_loss(d_fake, target_is_real=False)
    
    # 总损失
    loss_d = (loss_real + loss_fake) * 0.5
    
    loss_dict = {
        'loss_d_real': loss_real.item(),
        'loss_d_fake': loss_fake.item(),
        'loss_d': loss_d.item()
    }
    
    return loss_d, loss_dict


def compute_reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mode: str = 'l1'
) -> torch.Tensor:
    """
    计算重构损失
    
    Args:
        x: 原始 spectrogram
        x_hat: 重构 spectrogram
        mode: 损失类型 ('l1', 'l2', 'smooth_l1')
        
    Returns:
        重构损失
    """
    if mode == 'l1':
        return F.l1_loss(x_hat, x)
    elif mode == 'l2':
        return F.mse_loss(x_hat, x)
    elif mode == 'smooth_l1':
        return F.smooth_l1_loss(x_hat, x)
    else:
        raise ValueError(f"未知的重构损失类型: {mode}")


def compute_eg_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    d_fake: torch.Tensor,
    gan_loss: GANLoss,
    adv_weight: float = 1.0,
    recon_weight: float = 50.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算编码器-生成器损失
    
    EG 的目标:
    - 重构误差小
    - 欺骗 D (让 D 认为重构图是真的)
    
    Args:
        x: 原始 spectrogram
        x_hat: 重构 spectrogram
        d_fake: D(x_hat) 输出
        gan_loss: GAN 损失函数
        adv_weight: 对抗损失权重
        recon_weight: 重构损失权重
        
    Returns:
        (总损失, 损失详情字典)
    """
    # 对抗损失 (希望 D 认为 fake 是 real)
    loss_adv = gan_loss(d_fake, target_is_real=True)
    
    # 重构损失
    loss_recon = compute_reconstruction_loss(x, x_hat, mode='l1')
    
    # 总损失
    loss_eg = adv_weight * loss_adv + recon_weight * loss_recon
    
    loss_dict = {
        'loss_adv': loss_adv.item(),
        'loss_recon': loss_recon.item(),
        'loss_eg': loss_eg.item()
    }
    
    return loss_eg, loss_dict


class EGANLoss:
    """
    E-GAN 完整损失函数
    
    封装所有损失计算
    """
    
    def __init__(
        self,
        adv_weight: float = 1.0,
        recon_weight: float = 50.0,
        label_smoothing: float = 0.1,
        device: str = 'cuda'
    ):
        """
        Args:
            adv_weight: 对抗损失权重
            recon_weight: 重构损失权重
            label_smoothing: 标签平滑
            device: 设备
        """
        self.adv_weight = adv_weight
        self.recon_weight = recon_weight
        self.gan_loss = GANLoss(label_smoothing, device)
    
    def compute_d_loss(
        self,
        d_real: torch.Tensor,
        d_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算判别器损失"""
        return compute_discriminator_loss(d_real, d_fake, self.gan_loss)
    
    def compute_eg_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        d_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算 EG 损失"""
        return compute_eg_loss(
            x, x_hat, d_fake, 
            self.gan_loss,
            self.adv_weight,
            self.recon_weight
        )


if __name__ == "__main__":
    # 测试损失函数
    print("测试损失函数...")
    
    device = 'cpu'
    
    # 模拟数据
    batch_size = 4
    x = torch.rand(batch_size, 1, 64, 64)
    x_hat = torch.rand(batch_size, 1, 64, 64)
    d_real = torch.randn(batch_size, 1)
    d_fake = torch.randn(batch_size, 1)
    
    # 创建损失函数
    loss_fn = EGANLoss(
        adv_weight=1.0,
        recon_weight=50.0,
        label_smoothing=0.1,
        device=device
    )
    
    # 计算损失
    loss_d, d_dict = loss_fn.compute_d_loss(d_real, d_fake.detach())
    loss_eg, eg_dict = loss_fn.compute_eg_loss(x, x_hat, d_fake)
    
    print(f"D 损失: {d_dict}")
    print(f"EG 损失: {eg_dict}")
