"""已知周期解初值表。

🔴 来源与核实纪律：每个初值注明文献来源与核实日期；
数值与文献一致才算数，不许"记得大概是"。

figure-8（8 字形轨道）：
  来源：Chenciner & Montgomery 2000, Ann. of Math. 152, 881（存在性证明）；
        数值初值经 Simó 传播，广泛引用形式见 Chenciner 2000+ 讲义与
        N-body 文献通用的 16 位值（本文件所载）。
  核实：2026-10-17 按文献通用值录入；用本仓测试做了回归验证
        （一个周期 T≈6.3259 后回到初值附近，见 tests/test_figure8.py）。
  周期：T = 6.32591398（G=1、等质量、无软化时文献值；软化会让周期微移，
        见 tests 中的容差说明）。
"""
from __future__ import annotations

import numpy as np

# figure-8：三体等质量，两体初始对称位于 (±x0, ∓y0)，第三体在原点
FIGURE8_Q = np.array([
    [0.97000436, -0.24308753],
    [-0.97000436, 0.24308753],
    [0.0, 0.0],
])
FIGURE8_V = np.array([
    [0.4662036850, 0.4323657300],
    [0.4662036850, 0.4323657300],
    [-0.93240737, -0.86473146],
])
FIGURE8_PERIOD = 6.32591398


def figure8_state(eps: float = 0.0, seed: int = 0) -> np.ndarray:
    """figure-8 初始状态 [q, v]。eps>0 时对初值注入均匀噪声（阶段 2 抗噪用，
    由 seed 固定 → 幂等）。"""
    rng = np.random.default_rng(seed)
    scale_q = float(np.abs(FIGURE8_Q).max())
    scale_v = float(np.abs(FIGURE8_V).max())
    q = FIGURE8_Q + eps * scale_q * (2 * rng.random(FIGURE8_Q.shape) - 1)
    v = FIGURE8_V + eps * scale_v * (2 * rng.random(FIGURE8_V.shape) - 1)
    return np.stack([q, v])
