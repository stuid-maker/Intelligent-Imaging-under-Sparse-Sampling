#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练复数神经网络模型
用于雷达信号处理
"""
import resource_limits  # 放在 import torch 之前，用于限制 CPU/显存占用（见 resource_limits.py）

import os
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


class ComplexNet(nn.Module):
    """复数神经网络模型"""
    
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
        x = x.abs()  # 取幅度作为最终输出
        x = F.log_softmax(x, dim=1)
        return x


def train(model, device, train_loader, optimizer, epoch):
    """训练函数"""
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data), 
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
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy


def main():
    """主训练函数"""
    # 设置参数（batch/workers 可从 resource_limits.py 读入以降低占用）
    batch_size = getattr(resource_limits, 'BATCH_SIZE_CLASSIFICATION', 64)
    num_workers = getattr(resource_limits, 'NUM_WORKERS', 0)
    n_train = 1000
    n_test = 100
    image_size = 28
    num_targets = 5
    noise_level = 0.1
    num_epochs = 10
    learning_rate = 5e-3
    save_dir = 'checkpoints'
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备与资源限制（限制配置在 resource_limits.py）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resource_limits.apply_limits(device)
    print(f'使用设备: {device}')
    
    # 生成雷达仿真数据
    print('正在生成训练数据...')
    train_set = RadarDataset(
        num_samples=n_train, 
        image_size=image_size, 
        num_targets=num_targets, 
        noise_level=noise_level
    )
    test_set = RadarDataset(
        num_samples=n_test, 
        image_size=image_size, 
        num_targets=num_targets, 
        noise_level=noise_level
    )
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    # 创建模型
    print('正在创建模型...')
    model = ComplexNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # 训练循环
    print('开始训练...\n')
    best_accuracy = 0
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'test_loss': test_loss,
            }, model_path)
            print(f'已保存最佳模型 (准确率: {accuracy:.2f}%) 到 {model_path}\n')
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'test_loss': test_loss,
    }, final_model_path)
    print(f'训练完成！最终模型已保存到 {final_model_path}')
    print(f'最佳准确率: {best_accuracy:.2f}%')


if __name__ == '__main__':
    main()
