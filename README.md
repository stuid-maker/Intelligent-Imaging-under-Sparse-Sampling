# 稀疏假设下的合成孔径雷达智能成像（大创项目）

北京理工大学大创项目：基于稀疏表示与复数神经网络的 SAR 成像与重建。

## 项目结构

- **omp_omp_demo.py** — Orthogonal Matching Pursuit (OMP) 稀疏重构 Demo（NumPy + sklearn）
- **CVNN/complexPyTorch/** — 复数神经网络 (CVNN) 雷达图像重建
  - `train_reconstruction.py` — 重建模型训练
  - `train.py` — 分类模型训练
  - `inference_reconstruction.py` — 重建推理与可视化
  - `resource_limits.py` — 训练时 CPU/显存占用限制（可选）
  - `radar_data_generator.py` — 雷达仿真数据生成

## 环境要求

- Python 3.x
- PyTorch：`pip install torch`
- 可选：`pip install numpy scikit-learn`

## 快速开始

```bash
# 进入复数网络目录
cd CVNN/complexPyTorch

# 训练重建模型（会自动应用 resource_limits 若存在）
python train_reconstruction.py

# 运行重建推理与可视化
python inference_reconstruction.py
```

## 解读实验结果

### A. 数据指标分析

- **PSNR: 22.49 dB**  
  网络已经学到了核心特征，但还有很大的提升空间（通常优化好能达到 28–32 dB）。

- **MSE: 0.018**  
  均方误差较低，说明整体像素值的偏差不大。

### B. 视觉效果分析

网络成功抑制了绝大部分底噪（De-noising 能力验证），目标的亮度被拉回了接近 Ground Truth 的水平。

## 下一步：如何让 PSNR 从 22 dB 冲向 30 dB

1. **换个 Loss 函数**  
   当前点目标略有晕染。可尝试把损失从 MSELoss (L2) 换成 L1Loss。  
   L1 的梯度恒定，更易得到稀疏解，有利于黑背景更黑、亮点边缘更锐利。

2. **增加难度，逼迫网络“进化”**  
   - 目标数量随机：`num_targets = random.randint(5, 20)`  
   - 不只生成单像素点，可改为生成**小方块**或** MNIST 数字**等作为雷达目标。

3. **引入物理约束**  
   采用 Residual Learning（残差学习）或在网络中加入 Skip Connection（如 ResNet、U-Net 的跨层连接），有助于保留高频细节（点目标中心与边缘）。

## 总结

在 20% 稀疏采样率下，模型实现了从低信噪比观测数据中恢复目标幅度的功能，PSNR 达到 22.49 dB，有效抑制了采样带来的旁瓣干扰。当前在高频细节（点目标边缘锐度）上仍存在模糊，后续拟通过引入 L1 损失与更复杂的拓扑结构（残差/跳跃连接）进行优化。
