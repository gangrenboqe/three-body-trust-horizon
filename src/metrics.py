"""监控量：能量相对漂移 + 轨道分歧判据（阶段 2 用）。"""
from __future__ import annotations

import numpy as np

from systems import total_energy


def energy_drift(state0: np.ndarray, state_t: np.ndarray, masses=None,
                 g: float = 1.0, soft2: float = 1e-12) -> float:
    """|E(t) - E(0)| / |E(0)| —— 辛积分器的健康指标，应长期贴在容差内不系统性增长。"""
    q0, v0 = state0[0], state0[1]
    qt, vt = state_t[0], state_t[1]
    e0 = total_energy(q0, v0, masses, g, soft2)
    et = total_energy(qt, vt, masses, g, soft2)
    return abs(et - e0) / max(abs(e0), 1e-300)


def separation(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """两条同构轨道的状态分歧度：所有质点位置差的 Frobenius 范数。"""
    return float(np.linalg.norm(q_a - q_b))


def trust_horizon(separations: np.ndarray, times: np.ndarray, tolerance: float) -> float | None:
    """首次超过容差的时刻（分歧判据）。永不超则返回 None。

    separations[i] = 第 i 个采样时刻 t 上，参照轨道与受扰轨道的位置差。
    """
    idx = np.nonzero(separations > tolerance)[0]
    return float(times[idx[0]]) if idx.size else None
