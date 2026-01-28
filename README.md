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

## 上传到 GitHub

在项目根目录执行：

```bash
git add .
git commit -m "Initial commit: OMP demo + CVNN SAR reconstruction"
git branch -M main
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

先到 [GitHub](https://github.com/new) 新建一个空仓库（不要勾选 README），再执行上述 `remote` 和 `push`。
