# complexPyTorch

一个用于在 PyTorch 中构建和使用复数神经网络（Complex Valued Neural Networks, CVNN）的高级工具箱。

## 算法简介

**复数神经网络（CVNN）**是一种特殊的神经网络架构，其权重、激活值和输入输出都是复数形式。与传统的实数神经网络相比，CVNN 在处理复数数据时具有以下优势：

- **直接处理复数数据**：无需将复数数据拆分为实部和虚部，可以直接在复数域中进行运算
- **保留相位信息**：复数表示天然包含幅度和相位信息，这对于信号处理至关重要
- **更好的物理意义**：许多物理现象（如波传播、电磁场）在复数域中具有更简单的线性特性
- **参数效率**：在某些情况下，复数网络可以用更少的参数达到相同的表达能力

本项目基于论文 [C. Trabelsi et al., "Deep Complex Networks" (ICLR 2018)](https://openreview.net/forum?id=H1T2hmZAb) 实现，提供了完整的复数神经网络层和函数支持。

## 原理

### 复数神经网络的基本原理

#### 1. 复数卷积层（Complex Convolution）

复数卷积通过分别对实部和虚部进行卷积，然后按照复数乘法规则组合：

```
z = (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
```

在实现中，使用两个实数卷积层分别处理实部和虚部：
- `conv_r`: 处理实部
- `conv_i`: 处理虚部

然后通过 `apply_complex` 函数组合结果：
```python
output = (conv_r(real) - conv_i(imag)) + 1j * (conv_r(imag) + conv_i(real))
```

#### 2. 复数激活函数

复数激活函数通常对实部和虚部分别应用激活函数：
- **复数 ReLU**: `complex_relu(z) = ReLU(Re(z)) + 1j * ReLU(Im(z))`
- **复数 Tanh**: `complex_tanh(z) = tanh(Re(z)) + 1j * tanh(Im(z))`
- **复数 Sigmoid**: `complex_sigmoid(z) = sigmoid(Re(z)) + 1j * sigmoid(Im(z))`

#### 3. 复数批归一化（Complex BatchNorm）

复数批归一化有两种实现方式：

**朴素方法（Naive Approach）**：
- 分别对实部和虚部进行批归一化
- 实现简单，计算效率高
- 但忽略了实部和虚部之间的相关性

**协方差方法（Covariance Approach）**：
- 考虑复数数据的协方差矩阵
- 计算协方差矩阵的逆平方根
- 更符合复数数据的统计特性，但计算复杂度更高

协方差矩阵形式：
```
C = [Crr  Cri]
    [Cri  Cii]
```

其中：
- `Crr`: 实部的方差
- `Cii`: 虚部的方差  
- `Cri`: 实部和虚部的协方差

#### 4. 复数线性层

复数线性层同样使用两个实数线性层，通过复数乘法规则组合输出。

## 使用场景

### 1. 雷达信号处理 

**雷达回波数据**天然是复数形式（I/Q 数据），包含：
- **幅度信息**：目标反射强度
- **相位信息**：目标距离和速度信息

**应用示例**：
- 目标检测与识别
- 距离-多普勒（Range-Doppler）图像处理
- 合成孔径雷达（SAR）图像分析
- 多普勒雷达信号处理

本项目提供了 `radar_data_generator.py` 用于生成雷达仿真数据，包含多个点目标的距离-多普勒平面数据。

### 2. 通信信号处理

- 调制解调
- 信道估计
- 信号检测与分类
- 正交频分复用（OFDM）系统

### 3. 物理光学

- 光波传播模拟
- 全息成像
- 干涉测量
- 光学系统设计

### 4. 音频信号处理

- 频谱分析
- 语音识别
- 音乐信息检索
- 声学信号处理

## 安装

```bash
pip install complexPyTorch
```

**要求**：PyTorch >= 1.7（支持 `torch.complex64` 类型）

## 支持的层和函数

本项目实现了以下复数神经网络组件：

### 层（Layers）
- `ComplexConv2d` - 2D 复数卷积层
- `ComplexConvTranspose2d` - 2D 复数转置卷积层
- `ComplexLinear` - 复数全连接层
- `ComplexBatchNorm1d` - 1D 复数批归一化（协方差方法）
- `ComplexBatchNorm2d` - 2D 复数批归一化（协方差方法）
- `NaiveComplexBatchNorm1d` - 1D 复数批归一化（朴素方法）
- `NaiveComplexBatchNorm2d` - 2D 复数批归一化（朴素方法）
- `ComplexDropout` - 复数 Dropout
- `ComplexDropout2d` - 2D 复数 Dropout
- `ComplexMaxPool2d` - 2D 复数最大池化
- `ComplexAvgPool2d` - 2D 复数平均池化
- `ComplexGRU` - 复数 GRU
- `ComplexLSTM` - 复数 LSTM

### 激活函数（Functions）
- `complex_relu` - 复数 ReLU
- `complex_tanh` - 复数 Tanh
- `complex_sigmoid` - 复数 Sigmoid

## 语法和使用

语法设计遵循 PyTorch 的标准 API：
- **模块（Modules）**：以 `Complex` 开头，如 `ComplexConv2d`、`ComplexBatchNorm2d`
- **函数（Functions）**：以 `complex_` 开头，如 `complex_relu`、`complex_max_pool2d`

主要区别是输入和输出都是复数张量（`torch.complex64` 类型）。

## 示例：雷达数据处理

以下示例展示了如何使用复数神经网络处理雷达仿真数据：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from radar_data_generator import RadarDataset
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

# 生成雷达仿真数据
batch_size = 64
train_set = RadarDataset(num_samples=1000, image_size=28, num_targets=5, noise_level=0.1)
test_set = RadarDataset(num_samples=100, image_size=28, num_targets=5, noise_level=0.1)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# 定义复数神经网络
class ComplexNet(nn.Module):
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

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ComplexNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)  # 数据已经是复数类型
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# 训练循环
for epoch in range(4):
    train(model, device, train_loader, optimizer, epoch)
```

## 快速开始

### 1. 训练模型

使用提供的训练脚本直接开始训练：

```bash
python train.py
```

训练脚本会：
- 自动生成雷达仿真数据（1000个训练样本，100个测试样本）
- 创建并训练复数神经网络模型
- 自动保存最佳模型和最终模型到 `checkpoints/` 目录
- 显示训练进度和测试准确率

**训练参数说明**（可在 `train.py` 中修改）：
- `batch_size = 64`: 批次大小
- `n_train = 1000`: 训练样本数量
- `n_test = 100`: 测试样本数量
- `num_epochs = 10`: 训练轮数
- `learning_rate = 5e-3`: 学习率

**训练输出**：
- `checkpoints/best_model.pth`: 验证集上表现最好的模型
- `checkpoints/final_model.pth`: 最后一轮的模型

### 2. 使用训练好的模型进行推理

训练完成后，使用推理脚本进行预测：

```bash
python inference.py
```

推理脚本会：
- 加载训练好的模型
- 生成新的测试样本
- 进行预测并显示结果
- 可视化雷达数据（幅度和相位图）

**注意**：确保 `checkpoints/best_model.pth` 或 `checkpoints/final_model.pth` 存在。

### 3. 使用 Jupyter Notebook

也可以使用 `Example.ipynb` 进行交互式训练和实验：

1. 打开 Jupyter Notebook
2. 运行 `Example.ipynb` 中的代码单元
3. 可以实时查看训练过程和结果

### 4. 使用自己的数据

如果要使用自己的雷达数据：

1. **修改数据加载**：替换 `radar_data_generator.py` 中的数据生成部分
2. **创建自定义 Dataset**：继承 `torch.utils.data.Dataset`，确保返回复数张量
3. **调整模型输入尺寸**：根据你的数据尺寸修改网络结构

示例：
```python
from torch.utils.data import Dataset
import torch

class MyRadarDataset(Dataset):
    def __init__(self, data_path):
        # 加载你的数据
        self.data = ...  # 复数数据
        self.labels = ...
    
    def __getitem__(self, idx):
        # 返回复数张量，形状: (channels, height, width)
        return torch.complex64(self.data[idx]), self.labels[idx]
```

## 关于批归一化

对于大多数层，可以直接使用 PyTorch 的标准函数和模块。例如，`complex_relu` 函数简单地对实部和虚部分别应用 ReLU 激活函数。

复数批归一化有两种实现方式：

1. **协方差方法**（`ComplexBatchNorm1d/2d`）：
   - 计算协方差矩阵的逆平方根
   - 更符合复数数据的统计特性
   - 计算复杂度较高

2. **朴素方法**（`NaiveComplexBatchNorm1d/2d`）：
   - 分别对实部和虚部进行批归一化
   - 计算效率高
   - 实验表明性能差异可能不大

建议先尝试朴素方法，如果性能不满足需求再使用协方差方法。

## 版本历史

- **v0.4.1**: 当前版本，使用 `torch.complex64` 类型（需要 PyTorch >= 1.7）
- **早期版本**: 使用两个实数张量表示复数（实部和虚部分开）

## 引用

如果本项目对您的研究有帮助，请考虑引用：

[![DOI](https://img.shields.io/badge/DOI-10.1103%2FPhysRevX.11.021060-blue)](https://doi.org/10.1103/PhysRevX.11.021060)

**原始论文**：
- C. Trabelsi et al., "Deep Complex Networks", International Conference on Learning Representations (ICLR), 2018. [链接](https://openreview.net/forum?id=H1T2hmZAb)

## 致谢

感谢 Piotr Bialecki 在 PyTorch 论坛上提供的宝贵帮助。

## 许可证

MIT License
