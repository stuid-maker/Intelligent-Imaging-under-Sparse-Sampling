#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练复数神经网络模型 - 图像重建任务
从稀疏采样图像恢复完整图像
"""
import resource_limits  # 放在 import torch 之前，用于限制 CPU/显存占用（见 resource_limits.py）

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from radar_data_generator import RadarDataset
from complexPyTorch.complexLayers import (
    ComplexBatchNorm2d, ComplexConv2d, ComplexConvTranspose2d,
    ComplexDropout2d
)
from complexPyTorch.complexFunctions import complex_relu, complex_avg_pool2d, complex_upsample


class ComplexReconstructionNet(nn.Module):
    """复数图像重建网络（U-Net风格）"""
    
    def __init__(self):
        super(ComplexReconstructionNet, self).__init__()
        
        # 编码器（下采样）
        self.enc1 = ComplexConv2d(1, 32, 3, padding=1)
        self.enc1_bn = ComplexBatchNorm2d(32, track_running_stats=False)
        
        self.enc2 = ComplexConv2d(32, 64, 3, padding=1)
        self.enc2_bn = ComplexBatchNorm2d(64, track_running_stats=False)
        
        self.enc3 = ComplexConv2d(64, 128, 3, padding=1)
        self.enc3_bn = ComplexBatchNorm2d(128, track_running_stats=False)
        
        # 瓶颈层
        self.bottleneck = ComplexConv2d(128, 256, 3, padding=1)
        self.bottleneck_bn = ComplexBatchNorm2d(256, track_running_stats=False)
        
        # 解码器（上采样）
        self.dec3 = ComplexConvTranspose2d(256, 128, 2, stride=2)
        self.dec3_conv = ComplexConv2d(256, 128, 3, padding=1)  # 256 = 128(skip) + 128(dec)
        self.dec3_bn = ComplexBatchNorm2d(128, track_running_stats=False)
        
        self.dec2 = ComplexConvTranspose2d(128, 64, 2, stride=2)
        self.dec2_conv = ComplexConv2d(128, 64, 3, padding=1)  # 128 = 64(skip) + 64(dec)
        self.dec2_bn = ComplexBatchNorm2d(64, track_running_stats=False)
        
        self.dec1 = ComplexConvTranspose2d(64, 32, 2, stride=2)
        self.dec1_conv = ComplexConv2d(64, 32, 3, padding=1)  # 64 = 32(skip) + 32(dec)
        self.dec1_bn = ComplexBatchNorm2d(32, track_running_stats=False)
        
        # 输出层
        self.output = ComplexConv2d(32, 1, 3, padding=1)
        
        self.dropout = ComplexDropout2d(p=0.2)
    
    def forward(self, x):
        # 编码器
        e1 = complex_relu(self.enc1_bn(self.enc1(x)))
        e1_pool = complex_avg_pool2d(e1, 2, 2)
        
        e2 = complex_relu(self.enc2_bn(self.enc2(e1_pool)))
        e2_pool = complex_avg_pool2d(e2, 2, 2)
        
        e3 = complex_relu(self.enc3_bn(self.enc3(e2_pool)))
        e3_pool = complex_avg_pool2d(e3, 2, 2)
        
        # 瓶颈层
        bottleneck = complex_relu(self.bottleneck_bn(self.bottleneck(e3_pool)))
        
        # 解码器（带跳跃连接）
        d3 = self.dec3(bottleneck)
        # 调整尺寸以匹配e3
        if d3.shape[2:] != e3.shape[2:]:
            d3 = complex_upsample(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3_cat = torch.cat([d3, e3], dim=1)
        d3 = complex_relu(self.dec3_bn(self.dec3_conv(d3_cat)))
        
        d2 = self.dec2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = complex_upsample(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2_cat = torch.cat([d2, e2], dim=1)
        d2 = complex_relu(self.dec2_bn(self.dec2_conv(d2_cat)))
        
        d1 = self.dec1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = complex_upsample(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1_cat = torch.cat([d1, e1], dim=1)
        d1 = complex_relu(self.dec1_bn(self.dec1_conv(d1_cat)))
        
        # 输出
        output = self.output(d1)
        
        return output


class SimpleComplexReconstructionNet(nn.Module):
    """简化的复数图像重建网络（更简单，适合快速测试）"""
    
    def __init__(self):
        super(SimpleComplexReconstructionNet, self).__init__()
        
        # 简单的编码-解码结构
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


def complex_mse_loss(pred, target):
    """复数MSE损失"""
    # 分别计算实部和虚部的MSE
    real_loss = F.mse_loss(pred.real, target.real)
    imag_loss = F.mse_loss(pred.imag, target.imag)
    return real_loss + imag_loss


def complex_l1_loss(pred, target):
    """复数L1损失"""
    real_loss = F.l1_loss(pred.real, target.real)
    imag_loss = F.l1_loss(pred.imag, target.imag)
    return real_loss + imag_loss


def train(model, device, train_loader, optimizer, epoch):
    """训练函数"""
    model.train()
    total_loss = 0
    for batch_idx, (sparse_img, ground_truth) in enumerate(train_loader):
        sparse_img = sparse_img.to(device)
        ground_truth = ground_truth.to(device)
        
        optimizer.zero_grad()
        output = model(sparse_img)
        
        # 使用复数MSE损失
        loss = complex_mse_loss(output, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(sparse_img), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item())
            )
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def test(model, device, test_loader):
    """测试函数"""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for sparse_img, ground_truth in test_loader:
            sparse_img = sparse_img.to(device)
            ground_truth = ground_truth.to(device)
            output = model(sparse_img)
            test_loss += complex_mse_loss(output, ground_truth).item()
    
    test_loss /= len(test_loader)
    
    print('\nTest set: Average loss: {:.6f}\n'.format(test_loss))
    
    return test_loss


def main():
    """主训练函数"""
    # 设置参数
    batch_size = getattr(resource_limits, 'BATCH_SIZE_RECONSTRUCTION', 32)
    num_workers = getattr(resource_limits, 'NUM_WORKERS', 0)
    n_train = 1000
    n_test = 100
    image_size = 64  # 使用更大的图像尺寸
    num_targets = 5
    noise_level = 0.1
    sampling_rate = 0.2  # 20%采样率
    num_epochs = 20
    learning_rate = 1e-3
    save_dir = 'checkpoints'
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备与资源限制（限制配置在 resource_limits.py）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resource_limits.apply_limits(device)
    print(f'使用设备: {device}')
    
    # 生成雷达仿真数据（稀疏采样模式）
    print('正在生成训练数据（稀疏采样模式）...')
    train_set = RadarDataset(
        num_samples=n_train, 
        image_size=image_size, 
        num_targets=num_targets, 
        noise_level=noise_level,
        sampling_rate=sampling_rate,
        use_sparse_sampling=True
    )
    test_set = RadarDataset(
        num_samples=n_test, 
        image_size=image_size, 
        num_targets=num_targets, 
        noise_level=noise_level,
        sampling_rate=sampling_rate,
        use_sparse_sampling=True
    )
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    # 创建模型（使用简化版本）
    print('正在创建模型...')
    model = SimpleComplexReconstructionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    print('开始训练...\n')
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        
        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            model_path = os.path.join(save_dir, 'best_reconstruction_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'train_loss': train_loss,
            }, model_path)
            print(f'已保存最佳模型 (测试损失: {test_loss:.6f}) 到 {model_path}\n')
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_reconstruction_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': test_loss,
        'train_loss': train_loss,
    }, final_model_path)
    print(f'训练完成！最终模型已保存到 {final_model_path}')
    print(f'最佳测试损失: {best_loss:.6f}')


if __name__ == '__main__':
    main()
