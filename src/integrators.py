"""积分器：Forest-Ruth 4 阶辛积分器 + RK4 对照。

为什么必须辛积分器（施工单 4.1）：
辛积分器保相空间体积，能量长期不系统性漂移；
RK4 短期精度高，但能量误差随时间累积，混沌系统跑久了结果不可信。
本模块同时实现两者，供对照实测（这个对照本身就是项目要展示的证据之一）。

参考：Forest & Ruth 1990, Physica D 43, 105；Yoshida 1990, Phys. Lett. A 150, 262。
零科学计算栈依赖之外只用 numpy。
"""
from __future__ import annotations

import numpy as np

# Forest-Ruth 组合系数（drift-kick 序列，sum(c)=sum(d)=1 保证 4 阶）
_THETA1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_THETA0 = -2.0 ** (1.0 / 3.0) * _THETA1
FR_C = (_THETA1 / 2.0, (_THETA0 + _THETA1) / 2.0, (_THETA0 + _THETA1) / 2.0, _THETA1 / 2.0)
FR_D = (_THETA1, _THETA0, _THETA1, 0.0)  # 最后一次 kick 为 0，可省一次力计算


def forest_ruth_step(state: np.ndarray, accel_fn, dt: float) -> np.ndarray:
    """走一步 Forest-Ruth 4 阶。state 形状任意，前半是坐标解释、由 accel_fn 约定。

    约定：state = [q, v] 沿第 0 轴拼接（q、v 同形状）。
    accel_fn(q) 返回 dq²/dt²（即加速度），形状同 q。
    """
    q, v = state[0], state[1]
    for c, d in zip(FR_C, FR_D):
        if c != 0.0:
            q = q + c * dt * v
        if d != 0.0:
            v = v + d * dt * accel_fn(q)
    return np.stack([q, v])


def forest_ruth_integrate(state0: np.ndarray, accel_fn, dt: float, n_steps: int,
                          checkpoint_every: int | None = None, checkpoint_cb=None) -> np.ndarray:
    """跑 n_steps 步。可选每 checkpoint_every 步回调一次（阶段 2 断续框架的挂点）。"""
    state = state0.copy()
    for i in range(n_steps):
        state = forest_ruth_step(state, accel_fn, dt)
        if checkpoint_every and (i + 1) % checkpoint_every == 0 and checkpoint_cb:
            checkpoint_cb(i + 1, state)
    return state


def rk4_step(state: np.ndarray, deriv_fn, dt: float) -> np.ndarray:
    """经典 RK4 一步（对照组）。state=[q,v]，deriv_fn(state)→d(state)/dt。"""
    k1 = deriv_fn(state)
    k2 = deriv_fn(state + 0.5 * dt * k1)
    k3 = deriv_fn(state + 0.5 * dt * k2)
    k4 = deriv_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def make_rk4_deriv(accel_fn):
    """把 accel_fn 包装成 RK4 需要的 deriv_fn。"""
    def deriv(state: np.ndarray) -> np.ndarray:
        q, v = state[0], state[1]
        return np.stack([v, accel_fn(q)])
    return deriv
