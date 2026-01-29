#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的模型进行推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from radar_data_generator import RadarDataset
from complexPyTorch.complexLayers import (
    ComplexBatchNorm2d, ComplexConv2d, ComplexLinear,
    ComplexDropout2d, ComplexBatchNorm1d
)
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
import numpy as np
import matplotlib.pyplot as plt


class ComplexNet(nn.Module):
    """复数神经网络模型（需要与训练时保持一致）"""
    
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 10, 5, 1)
        self.bn2d = ComplexBatchNorm2d(10, track_running_stats=False)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.fc1 = ComplexLinear(4*4*20, 500)
        self.dropout = ComplexDropout2d(p=0.3)
        self.bn1d = ComplexBatchNorm1d(500, track_running_stats=False)
        self.fc2 = ComplexLinear(500, 10)
             
    def forward(self, x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = self.bn2d(x)
        x = self.conv2(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*20)
        x = self.fc1(x)
        x = self.dropout(x)
        x = complex_relu(x)
        x = self.bn1d(x)
        x = self.fc2(x)
        x = x.abs()
        x = F.log_softmax(x, dim=1)
        return x


def load_model(model_path, device):
    """加载训练好的模型"""
    model = ComplexNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'已加载模型: {model_path}')
    if 'accuracy' in checkpoint:
        print(f'模型准确率: {checkpoint["accuracy"]:.2f}%')
    if 'epoch' in checkpoint:
        print(f'训练轮数: {checkpoint["epoch"]}')
    
    return model


def predict_single(model, device, data):
    """对单个样本进行预测"""
    model.eval()
    with torch.no_grad():
        data = data.unsqueeze(0).to(device)  # 添加batch维度
        output = model(data)
        prob = torch.exp(output)  # log_softmax转回概率
        pred = output.argmax(dim=1).item()
        confidence = prob[0][pred].item()
    
    return pred, confidence, prob[0].cpu().numpy()


def visualize_radar_data(data, title="Radar Data"):
    """可视化雷达数据（幅度和相位）"""
    data_np = data.squeeze().cpu().numpy()
    magnitude = np.abs(data_np)
    phase = np.angle(data_np)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 幅度图
    im1 = axes[0].imshow(magnitude, cmap='hot', aspect='auto')
    axes[0].set_title(f'{title} - Magnitude')
    axes[0].set_xlabel('Doppler')
    axes[0].set_ylabel('Range')
    plt.colorbar(im1, ax=axes[0], label='Magnitude')
    
    # 相位图
    im2 = axes[1].imshow(phase, cmap='hsv', aspect='auto')
    axes[1].set_title(f'{title} - Phase')
    axes[1].set_xlabel('Doppler')
    axes[1].set_ylabel('Range')
    plt.colorbar(im2, ax=axes[1], label='Phase (rad)')
    
    plt.tight_layout()
    return fig


def main():
    """主推理函数"""
    # 设置参数
    model_path = 'checkpoints/best_model.pth'  # 或 'checkpoints/final_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = 5  # 要测试的样本数量
    
    print(f'使用设备: {device}')
    
    # 加载模型
    try:
        model = load_model(model_path, device)
    except FileNotFoundError:
        print(f'错误: 找不到模型文件 {model_path}')
        print('请先运行 train.py 训练模型')
        return
    
    # 生成测试数据
    print('\n正在生成测试数据...')
    test_set = RadarDataset(num_samples=num_samples, image_size=28, num_targets=5, noise_level=0.1)
    
    # 进行预测
    print('\n开始预测...\n')
    for i in range(num_samples):
        data, label = test_set[i]
        pred, confidence, probs = predict_single(model, device, data)
        
        print(f'样本 {i+1}:')
        print(f'  真实标签: {label}')
        print(f'  预测类别: {pred}')
        print(f'  置信度: {confidence:.4f}')
        print(f'  各类别概率: {probs}')
        print()
        
        # 可视化第一个样本
        if i == 0:
            fig = visualize_radar_data(data, f"Sample {i+1} - Predicted Class: {pred}")
            plt.savefig('radar_sample_visualization.png', dpi=150, bbox_inches='tight')
            print('已保存可视化结果到 radar_sample_visualization.png')
            plt.close()
    
    print('推理完成！')


if __name__ == '__main__':
    main()
