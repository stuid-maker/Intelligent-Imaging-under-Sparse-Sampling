import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit


def omp(D: np.ndarray, y: np.ndarray, K: int | None = None, tol: float = 1e-6):
    """
    Orthogonal Matching Pursuit (OMP) 基于 NumPy 的简单实现。

    参数
    ----
    D : np.ndarray, shape (m, n)
        字典矩阵，每一列是一个原子（建议列向量单位化）
    y : np.ndarray, shape (m,) 或 (m, 1)
        目标信号
    K : int or None
        稀疏度上限（最多选多少个原子）。如果为 None，则只用 tol 控制停止
    tol : float
        残差的 L2 范数小于该值时停止

    返回
    ----
    x : np.ndarray, shape (n,)
        稀疏系数向量
    support : list[int]
        被选中的原子索引列表
    """
    # 确保 y 为一维向量
    y = np.asarray(y).reshape(-1)
    m, n = D.shape

    # 初始化
    r = y.copy()          # 残差
    support: list[int] = []          # 支持集
    x = np.zeros(n)       # 系数

    # 最大迭代次数：不超过字典列数
    max_iter = n if K is None else min(K, n)

    for _ in range(max_iter):
        # 1. 计算所有原子与残差的相关性
        correlations = D.T @ r     # shape (n,)
        # 选取绝对值最大的位置
        j = int(np.argmax(np.abs(correlations)))

        # 如果该原子已经被选过，提前结束（防止死循环）
        if j in support:
            break
        support.append(j)

        # 2. 构造子字典 D_S
        Ds = D[:, support]         # shape (m, |S|)

        # 3. 在子空间上做最小二乘拟合：min ||y - Ds * z||
        # np.linalg.lstsq 返回 (z, residuals, rank, s)
        z, _, _, _ = np.linalg.lstsq(Ds, y, rcond=None)

        # 4. 更新残差
        r = y - Ds @ z

        # 5. 检查残差是否足够小
        if np.linalg.norm(r) < tol:
            break

    # 将子空间解 z 写回到完整系数向量 x
    x[support] = z

    return x, support


def omp_sklearn(D: np.ndarray, y: np.ndarray, use_ratio: float = 1 / 8, tol: float | None = None):
    """
    使用 scikit-learn 的 OrthogonalMatchingPursuit 来做 OMP。

    D : (m, n)  字典矩阵
    y : (m,)    目标信号
    use_ratio : K 与 m 的比例，例如 1/8
    tol :       残差阈值（可选），与 n_nonzero_coefs 二选一使用
    """
    y = np.asarray(y).reshape(-1)
    m, n = D.shape

    # K = m/8，但不能超过特征数 n，且至少为 1
    K = max(1, min(int(m * use_ratio), n))

    model = OrthogonalMatchingPursuit(
        n_nonzero_coefs=K if tol is None else None,
        tol=tol,
        fit_intercept=False,
        normalize=False,  # D 已经单位化，可以关闭归一化
    )
    model.fit(D, y)

    x = model.coef_  # 形状 (n,)
    support = list(np.flatnonzero(x))

    return x, support


def _demo():
    """简单的自测示例：python omp_omp_demo.py 直接运行即可。"""
    # 1. 构造一个字典 D（m=50, n=100）
    m, n = 50, 100
    np.random.seed(0)
    D = np.random.randn(m, n)

    # 一般会把每个原子单位化，效果更稳定
    D = D / np.linalg.norm(D, axis=0, keepdims=True)

    # 2. 构造一个稀疏真实系数 x_true（只有 3 个非零）
    x_true = np.zeros(n)
    true_support = [3, 20, 77]
    x_true[true_support] = [1.5, -2.0, 0.7]

    # 3. 生成观测 y
    y = D @ x_true

    # 4. 手写 OMP 重构，K = m/8（向下取整，至少为 1）
    K = max(1, m // 8)
    x_hat_manual, support_hat_manual = omp(D, y, K=K, tol=1e-8)

    # 5. sklearn OMP 重构，使用同样的 K 比例
    x_hat_sklearn, support_hat_sklearn = omp_sklearn(D, y, use_ratio=1 / 8)

    print("m =", m, "K = m/8 =", K)
    print("真实支持集:", true_support)
    print("手写 OMP 支持集:", support_hat_manual)
    print("sklearn OMP 支持集:", support_hat_sklearn)
    print("手写 OMP 重构误差:", np.linalg.norm(x_hat_manual - x_true))
    print("sklearn OMP 重构误差:", np.linalg.norm(x_hat_sklearn - x_true))


if __name__ == "__main__":
    _demo()

