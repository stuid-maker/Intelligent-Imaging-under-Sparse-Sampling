#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雷达仿真数据生成器
生成包含5个点目标的雷达回波数据
支持稀疏采样模拟
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class RadarDataset(Dataset):
    """雷达仿真数据集（支持稀疏采样）"""
    
    def __init__(self, num_samples=1000, image_size=64, num_targets=5, 
                 noise_level=0.1, sampling_rate=0.2, use_sparse_sampling=True):
        """
        初始化雷达数据集
        
        参数:
            num_samples: 样本数量
            image_size: 图像尺寸（距离-多普勒平面大小），建议使用64x64或更大
            num_targets: 点目标数量
            noise_level: 噪声水平
            sampling_rate: 稀疏采样率（0-1之间，如0.2表示只保留20%的数据）
            use_sparse_sampling: 是否使用稀疏采样（True: 返回稀疏采样图像和ground truth对）
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_targets = num_targets
        self.noise_level = noise_level
        self.sampling_rate = sampling_rate
        self.use_sparse_sampling = use_sparse_sampling
        
        # 生成数据
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            if use_sparse_sampling:
                # 生成稀疏采样图像对 (sparse_image, ground_truth)
                sparse_img, ground_truth = self._generate_sparse_sampled_data()
                self.data.append((sparse_img, ground_truth))
                self.labels.append(0)  # 标签暂时保留，可用于其他任务
            else:
                # 原始模式：只生成完整图像
                radar_data = self._generate_radar_data()
                self.data.append(radar_data)
                self.labels.append(0)
    
    def _generate_radar_data(self):
        """
        生成单个雷达回波数据（距离-多普勒平面）
        返回复数形式的2D数组（高清ground truth）
        """
        # 初始化距离-多普勒平面（复数形式）
        range_doppler = np.zeros((self.image_size, self.image_size), dtype=complex)
        
        # 生成5个点目标
        for _ in range(self.num_targets):
            # 随机生成目标位置（距离和多普勒）
            range_bin = np.random.randint(5, self.image_size - 5)
            doppler_bin = np.random.randint(5, self.image_size - 5)
            
            # 随机生成目标幅度和相位
            amplitude = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            
            # 在目标位置添加复数信号
            # 使用sinc函数模拟点目标的响应（更真实的雷达回波）
            for dr in range(-2, 3):
                for dd in range(-2, 3):
                    r_idx = range_bin + dr
                    d_idx = doppler_bin + dd
                    if 0 <= r_idx < self.image_size and 0 <= d_idx < self.image_size:
                        # sinc函数模拟点扩散函数
                        sinc_val = np.sinc(dr / 2.0) * np.sinc(dd / 2.0)
                        range_doppler[r_idx, d_idx] += amplitude * sinc_val * np.exp(1j * phase)
        
        # 添加噪声（复数噪声）
        noise_real = np.random.normal(0, self.noise_level, (self.image_size, self.image_size))
        noise_imag = np.random.normal(0, self.noise_level, (self.image_size, self.image_size))
        noise = noise_real + 1j * noise_imag
        
        range_doppler = range_doppler + noise
        
        # 转换为torch tensor
        return torch.from_numpy(range_doppler).type(torch.complex64)
    
    def _generate_sparse_sampled_data(self):
        """
        生成稀疏采样数据对
        返回: (sparse_image, ground_truth)
        - sparse_image: 稀疏采样后的"坏图"（作为网络输入）
        - ground_truth: 原始高清图（作为训练目标）
        """
        # 1. 生成高清ground truth图像
        x_ground_truth = self._generate_radar_data()  # (H, W) complex tensor
        
        # 2. 变换到频率域 (k-space)
        k_space_data = torch.fft.fft2(x_ground_truth)
        k_space_data = torch.fft.fftshift(k_space_data)  # 把低频移到中间
        
        # 3. 生成采样掩模 (Mask)
        # 随机采样掩模
        mask = torch.rand(self.image_size, self.image_size) < self.sampling_rate
        mask = mask.float()  # 变成 0 和 1
        
        # 4. 应用掩模进行稀疏采样
        y_sparse_k_space = k_space_data * mask
        
        # 5. 直接转换回来得到稀疏采样后的"坏图" (Zero-filled Reconstruction)
        k_space_filled = torch.fft.ifftshift(y_sparse_k_space)
        x_sparse = torch.fft.ifft2(k_space_filled)
        
        # 6. 添加channel维度: (1, H, W)
        x_sparse = x_sparse.unsqueeze(0)
        x_ground_truth = x_ground_truth.unsqueeze(0)
        
        return x_sparse, x_ground_truth
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.use_sparse_sampling:
            sparse_img, ground_truth = self.data[idx]
            return sparse_img, ground_truth
        else:
            return self.data[idx], self.labels[idx]


def generate_radar_data(num_samples=1000, image_size=64, num_targets=5, 
                        noise_level=0.1, sampling_rate=0.2, use_sparse_sampling=True):
    """
    便捷函数：生成雷达仿真数据
    
    参数:
        num_samples: 样本数量
        image_size: 图像尺寸（建议64或更大）
        num_targets: 点目标数量
        noise_level: 噪声水平
        sampling_rate: 稀疏采样率
        use_sparse_sampling: 是否使用稀疏采样
    
    返回:
        RadarDataset对象
    """
    return RadarDataset(num_samples, image_size, num_targets, noise_level, 
                       sampling_rate, use_sparse_sampling)


def visualize_sparse_sampling_comparison(dataset, idx=0, save_path=None):
    """
    可视化稀疏采样效果对比
    """
    import matplotlib.pyplot as plt
    
    if dataset.use_sparse_sampling:
        sparse_img, ground_truth = dataset[idx]
        sparse_img = sparse_img.squeeze().cpu()
        ground_truth = ground_truth.squeeze().cpu()
        
        # 计算zero-filled重建
        k_space_sparse = torch.fft.fftshift(torch.fft.fft2(sparse_img))
        mask = (torch.abs(k_space_sparse) > 1e-6).float()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 第一行：幅度图
        im1 = axes[0, 0].imshow(torch.abs(ground_truth).numpy(), cmap='hot', aspect='auto')
        axes[0, 0].set_title('Ground Truth - Magnitude')
        axes[0, 0].set_xlabel('Doppler')
        axes[0, 0].set_ylabel('Range')
        plt.colorbar(im1, ax=axes[0, 0], label='Magnitude')
        
        im2 = axes[0, 1].imshow(mask.numpy(), cmap='gray', aspect='auto')
        axes[0, 1].set_title(f'Sampling Mask ({int(dataset.sampling_rate*100)}%)')
        axes[0, 1].set_xlabel('Doppler')
        axes[0, 1].set_ylabel('Range')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(torch.abs(sparse_img).numpy(), cmap='hot', aspect='auto')
        axes[0, 2].set_title('Sparse Sampled (Zero-filled) - Magnitude')
        axes[0, 2].set_xlabel('Doppler')
        axes[0, 2].set_ylabel('Range')
        plt.colorbar(im3, ax=axes[0, 2], label='Magnitude')
        
        # 第二行：相位图
        im4 = axes[1, 0].imshow(torch.angle(ground_truth).numpy(), cmap='hsv', aspect='auto')
        axes[1, 0].set_title('Ground Truth - Phase')
        axes[1, 0].set_xlabel('Doppler')
        axes[1, 0].set_ylabel('Range')
        plt.colorbar(im4, ax=axes[1, 0], label='Phase (rad)')
        
        axes[1, 1].axis('off')  # 空白
        
        im5 = axes[1, 2].imshow(torch.angle(sparse_img).numpy(), cmap='hsv', aspect='auto')
        axes[1, 2].set_title('Sparse Sampled - Phase')
        axes[1, 2].set_xlabel('Doppler')
        axes[1, 2].set_ylabel('Range')
        plt.colorbar(im5, ax=axes[1, 2], label='Phase (rad)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'已保存可视化结果到 {save_path}')
        else:
            plt.show()
        
        plt.close()
    else:
        print("数据集未启用稀疏采样模式")


if __name__ == "__main__":
    # 测试代码
    print("测试稀疏采样数据生成...")
    dataset = RadarDataset(
        num_samples=10, 
        image_size=64,  # 使用更大的图像尺寸
        num_targets=5, 
        sampling_rate=0.2,  # 20%采样率
        use_sparse_sampling=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    sparse_img, ground_truth = dataset[0]
    print(f"稀疏图像形状: {sparse_img.shape}")
    print(f"Ground truth形状: {ground_truth.shape}")
    print(f"数据类型: {sparse_img.dtype}")
    print(f"稀疏图像幅度范围: [{torch.abs(sparse_img).min():.4f}, {torch.abs(sparse_img).max():.4f}]")
    print(f"Ground truth幅度范围: [{torch.abs(ground_truth).min():.4f}, {torch.abs(ground_truth).max():.4f}]")
    
    # 可视化对比
    print("\n生成可视化对比图...")
    visualize_sparse_sampling_comparison(dataset, idx=0, save_path='sparse_sampling_comparison.png')
