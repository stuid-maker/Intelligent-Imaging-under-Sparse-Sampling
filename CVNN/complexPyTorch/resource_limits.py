# -*- coding: utf-8 -*-
"""
训练时的 CPU/GPU 资源占用限制。
在运行 train.py 或 train_reconstruction.py 之前先 import 本模块即可生效。

用法（在训练脚本最开头）:
    import resource_limits   # 必须放在 import torch 之前
    import torch
    ...
    # 在 main() 里获取 device 之后调用:
    resource_limits.apply_limits(device)
"""

import os

# 必须在对 numpy/torch 做 import 之前设置，否则无效
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "4")

# 下面这些在 apply_limits 时使用，可按机器配置修改
MAX_CPU_THREADS = 4
CUDA_MEMORY_FRACTION = 0.5
BATCH_SIZE_RECONSTRUCTION = 8   # train_reconstruction 建议 batch（省显存可改 4）
BATCH_SIZE_CLASSIFICATION = 16  # train 建议 batch
NUM_WORKERS = 0


def apply_limits(device, max_cpu_threads=None, cuda_memory_fraction=None):
    """
    对当前进程施加 CPU 线程数和（可选）GPU 显存上限。
    在训练脚本中拿到 device 之后调用一次即可。

    device : torch.device
    max_cpu_threads : int, 默认用本文件里的 MAX_CPU_THREADS
    cuda_memory_fraction : float in (0, 1], 默认用 CUDA_MEMORY_FRACTION
    """
    import torch
    n = max_cpu_threads if max_cpu_threads is not None else MAX_CPU_THREADS
    torch.set_num_threads(n)
    if device.type == "cuda" and hasattr(torch.cuda, "set_per_process_memory_fraction"):
        frac = cuda_memory_fraction if cuda_memory_fraction is not None else CUDA_MEMORY_FRACTION
        torch.cuda.set_per_process_memory_fraction(frac)
