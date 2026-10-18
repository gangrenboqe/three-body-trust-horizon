"""动力系统定义：等质量三体（Newton 引力，Plummer 软化）。

软化（softening）ε² 加在 r² 上防止近距奇点导致积分器崩溃——
这是数值做法，物理上等于换成 Plummer 软化势，README 诚实边界里必须注明。

约定：q 形状 (N, D)——N 个质点、D 维；这里 N=3、D=2。
"""
from __future__ import annotations

import numpy as np

G = 1.0
SOFTENING2 = 1e-6 ** 2  # 施工单指定软化 1e-6


def three_body_accel(q: np.ndarray, masses: np.ndarray | None = None,
                     g: float = G, soft2: float = SOFTENING2) -> np.ndarray:
    """N 体加速度：a_i = Σ_{j≠i} G m_j (q_j - q_i) / (|q_j - q_i|² + soft²)^{3/2}"""
    n = q.shape[0]
    m = np.ones(n) if masses is None else masses
    accel = np.zeros_like(q)
    for i in range(n):
        diff = q - q[i]                      # (N, D)
        diff[i] = 0.0                        # 不对自己求
        r2 = np.sum(diff * diff, axis=1) + soft2
        accel[i] = g * np.sum((m / r2 ** 1.5)[:, None] * diff, axis=0)
    return accel


def three_body_deriv(accel_fn=None):
    """RK4 用的导数包装。"""
    af = accel_fn or three_body_accel
    def deriv(state: np.ndarray) -> np.ndarray:
        return np.stack([state[1], af(state[0])])
    return deriv


def total_energy(q: np.ndarray, v: np.ndarray, masses: np.ndarray | None = None,
                 g: float = G, soft2: float = SOFTENING2) -> float:
    """总机械能 = 动能 + 成对势能（Plummer 软化）。守恒量，用于漂移监控。"""
    n = q.shape[0]
    m = np.ones(n) if masses is None else masses
    kinetic = 0.5 * float(np.sum(m[:, None] * v * v))
    potential = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r2 = float(np.sum((q[j] - q[i]) ** 2)) + soft2
            potential -= g * m[i] * m[j] / np.sqrt(r2)
    return kinetic + potential
