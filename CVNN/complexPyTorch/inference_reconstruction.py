#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的模型进行图像重建推理
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from radar_data_generator import RadarDataset, visualize_sparse_sampling_comparison
from complexPyTorch.complexLayers import (
    ComplexBatchNorm2d, ComplexConv2d,
    ComplexDropout2d
)
from complexPyTorch.complexFunctions import complex_relu
import numpy as np
import matplotlib.pyplot as plt


class SimpleComplexReconstructionNet(nn.Module):
    """简化的复数图像重建网络（需要与训练时保持一致）"""
    
    def __init__(self):
        super(SimpleComplexReconstructionNet, self).__init__()
        
        self.conv1 = ComplexConv2d(1, 64, 3, padding=1)
        self.bn1 = ComplexBatchNorm2d(64, track_running_stats=False)
        
        self.conv2 = ComplexConv2d(64, 128, 3, padding=1)
        self.bn2 = ComplexBatchNorm2d(128, track_running_stats=False)
        
        self.conv3 = ComplexConv2d(128, 256, 3, padding=1)
        self.bn3 = ComplexBatchNorm2d(256, track_running_stats=False)
        
        self.conv4 = ComplexConv2d(256, 128, 3, padding=1)
        self.bn4 = ComplexBatchNorm2d(128, track_running_stats=False)
        
        self.conv5 = ComplexConv2d(128, 64, 3, padding=1)
        self.bn5 = ComplexBatchNorm2d(64, track_running_stats=False)
        
        self.conv6 = ComplexConv2d(64, 1, 3, padding=1)
        
        self.dropout = ComplexDropout2d(p=0.2)
    
    def forward(self, x):
        x = complex_relu(self.bn1(self.conv1(x)))
        x = complex_relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = complex_relu(self.bn3(self.conv3(x)))
        x = complex_relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = complex_relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return x


def load_model(model_path, device):
    """加载训练好的模型"""
    model = SimpleComplexReconstructionNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'已加载模型: {model_path}')
    if 'test_loss' in checkpoint:
        print(f'模型测试损失: {checkpoint["test_loss"]:.6f}')
    if 'epoch' in checkpoint:
        print(f'训练轮数: {checkpoint["epoch"]}')
    
    return model


def reconstruct_single(model, device, sparse_img):
    """对单个稀疏采样图像进行重建"""
    model.eval()
    with torch.no_grad():
        sparse_img = sparse_img.unsqueeze(0).to(device)  # 添加batch维度
        reconstructed = model(sparse_img)
        return reconstructed.squeeze(0).cpu()


def visualize_reconstruction(sparse_img, ground_truth, reconstructed, save_path=None):
    """可视化重建结果对比"""
    sparse_img = sparse_img.squeeze().cpu()
    ground_truth = ground_truth.squeeze().cpu()
    reconstructed = reconstructed.squeeze().cpu()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：幅度图
    im1 = axes[0, 0].imshow(torch.abs(ground_truth).numpy(), cmap='hot', aspect='auto')
    axes[0, 0].set_title('Ground Truth - Magnitude')
    axes[0, 0].set_xlabel('Doppler')
    axes[0, 0].set_ylabel('Range')
    plt.colorbar(im1, ax=axes[0, 0], label='Magnitude')
    
    im2 = axes[0, 1].imshow(torch.abs(sparse_img).numpy(), cmap='hot', aspect='auto')
    axes[0, 1].set_title('Sparse Sampled (Input) - Magnitude')
    axes[0, 1].set_xlabel('Doppler')
    axes[0, 1].set_ylabel('Range')
    plt.colorbar(im2, ax=axes[0, 1], label='Magnitude')
    
    im3 = axes[0, 2].imshow(torch.abs(reconstructed).numpy(), cmap='hot', aspect='auto')
    axes[0, 2].set_title('Reconstructed (Output) - Magnitude')
    axes[0, 2].set_xlabel('Doppler')
    axes[0, 2].set_ylabel('Range')
    plt.colorbar(im3, ax=axes[0, 2], label='Magnitude')
    
    # 第二行：相位图
    im4 = axes[1, 0].imshow(torch.angle(ground_truth).numpy(), cmap='hsv', aspect='auto')
    axes[1, 0].set_title('Ground Truth - Phase')
    axes[1, 0].set_xlabel('Doppler')
    axes[1, 0].set_ylabel('Range')
    plt.colorbar(im4, ax=axes[1, 0], label='Phase (rad)')
    
    im5 = axes[1, 1].imshow(torch.angle(sparse_img).numpy(), cmap='hsv', aspect='auto')
    axes[1, 1].set_title('Sparse Sampled - Phase')
    axes[1, 1].set_xlabel('Doppler')
    axes[1, 1].set_ylabel('Range')
    plt.colorbar(im5, ax=axes[1, 1], label='Phase (rad)')
    
    im6 = axes[1, 2].imshow(torch.angle(reconstructed).numpy(), cmap='hsv', aspect='auto')
    axes[1, 2].set_title('Reconstructed - Phase')
    axes[1, 2].set_xlabel('Doppler')
    axes[1, 2].set_ylabel('Range')
    plt.colorbar(im6, ax=axes[1, 2], label='Phase (rad)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'已保存可视化结果到 {save_path}')
    else:
        plt.show()
    
    plt.close()


def calculate_metrics(reconstructed, ground_truth):
    """计算重建质量指标"""
    reconstructed = reconstructed.squeeze().cpu()
    ground_truth = ground_truth.squeeze().cpu()
    
    # MSE (实部和虚部分别计算)
    mse_real = torch.mean((reconstructed.real - ground_truth.real) ** 2).item()
    mse_imag = torch.mean((reconstructed.imag - ground_truth.imag) ** 2).item()
    mse_total = mse_real + mse_imag
    
    # PSNR (基于幅度)
    max_val = torch.abs(ground_truth).max().item()
    psnr = 20 * np.log10(max_val / (np.sqrt(mse_total) + 1e-10))
    
    # 幅度误差
    magnitude_error = torch.mean(torch.abs(torch.abs(reconstructed) - torch.abs(ground_truth))).item()
    
    return {
        'MSE': mse_total,
        'MSE_real': mse_real,
        'MSE_imag': mse_imag,
        'PSNR': psnr,
        'Magnitude_Error': magnitude_error
    }


def main():
    """主推理函数"""
    # 设置参数
    model_path = 'checkpoints/best_reconstruction_model.pth'  # 或 'checkpoints/final_reconstruction_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = 5  # 要测试的样本数量
    image_size = 64
    sampling_rate = 0.2
    
    print(f'使用设备: {device}')
    
    # 加载模型
    try:
        model = load_model(model_path, device)
    except FileNotFoundError:
        print(f'错误: 找不到模型文件 {model_path}')
        print('请先运行 train_reconstruction.py 训练模型')
        return
    
    # 生成测试数据
    print('\n正在生成测试数据...')
    test_set = RadarDataset(
        num_samples=num_samples, 
        image_size=image_size, 
        num_targets=5, 
        noise_level=0.1,
        sampling_rate=sampling_rate,
        use_sparse_sampling=True
    )
    
    # 进行重建
    print('\n开始重建...\n')
    all_metrics = []
    
    for i in range(num_samples):
        sparse_img, ground_truth = test_set[i]
        reconstructed = reconstruct_single(model, device, sparse_img)
        
        # 计算指标
        metrics = calculate_metrics(reconstructed, ground_truth)
        all_metrics.append(metrics)
        
        print(f'样本 {i+1}:')
        print(f'  MSE: {metrics["MSE"]:.6f}')
        print(f'  PSNR: {metrics["PSNR"]:.2f} dB')
        print(f'  幅度误差: {metrics["Magnitude_Error"]:.6f}')
        print()
        
        # 可视化第一个样本
        if i == 0:
            visualize_reconstruction(
                sparse_img, 
                ground_truth, 
                reconstructed,
                save_path='reconstruction_result.png'
            )
    
    # 计算平均指标
    avg_metrics = {
        'MSE': np.mean([m['MSE'] for m in all_metrics]),
        'PSNR': np.mean([m['PSNR'] for m in all_metrics]),
        'Magnitude_Error': np.mean([m['Magnitude_Error'] for m in all_metrics])
    }
    
    print('\n平均指标:')
    print(f'  平均 MSE: {avg_metrics["MSE"]:.6f}')
    print(f'  平均 PSNR: {avg_metrics["PSNR"]:.2f} dB')
    print(f'  平均幅度误差: {avg_metrics["Magnitude_Error"]:.6f}')
    
    print('\n推理完成！')


if __name__ == '__main__':
    main()
