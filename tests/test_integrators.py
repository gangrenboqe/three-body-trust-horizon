"""积分器测试：谐振子精确性、能量守恒、辛 vs 非辛对照。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from integrators import forest_ruth_integrate, forest_ruth_step, make_rk4_deriv, rk4_step
from systems import three_body_accel, three_body_deriv, total_energy


def harmonic_state(t_period: float = 1.0, dt: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """一维谐振子对（q=[1], v=[0], ω=1），解析解可对照。"""
    state = np.stack([np.array([[1.0]]), np.array([[0.0]])])
    n = int(round(t_period / dt))
    return state, n


class HarmonicOscillatorTest(unittest.TestCase):
    """谐振子 ω=1：能量严格守恒的已知系统，积分器正确性的第一道关。"""

    def _run(self, stepper, n: int, dt: float):
        state = np.stack([np.array([[1.0]]), np.array([[0.0]])])
        accel = lambda q: -q  # ω=1
        for _ in range(n):
            state = stepper(state, accel, dt)
        return state

    def test_forest_ruth_harmonic_energy_conserved(self):
        dt, n = 1e-3, 2000  # t=2.0
        end = self._run(forest_ruth_step, n, dt)
        q, v = float(end[0, 0, 0]), float(end[1, 0, 0])
        e = 0.5 * (q * q + v * v)
        self.assertLess(abs(e - 0.5), 1e-9, f"谐振子能量漂移 {abs(e-0.5):.2e} 超容差")

    def test_forest_ruth_harmonic_phase_accuracy(self):
        dt, n = 1e-3, 2000
        end = self._run(forest_ruth_step, n, dt)
        q_exact = float(np.cos(2.0))  # t=2.0 的解析解
        self.assertLess(abs(float(end[0, 0, 0]) - q_exact), 1e-6)

    def test_rk4_harmonic_also_accurate_short_term(self):
        """RK4 短期也准——对照的意义在长期，这里只验证实现正确。"""
        dt, n = 1e-3, 2000
        state = np.stack([np.array([[1.0]]), np.array([[0.0]])])
        deriv = lambda s: np.stack([s[1], -s[0]])
        for _ in range(n):
            state = rk4_step(state, deriv, dt)
        self.assertLess(abs(float(state[0, 0, 0]) - float(np.cos(2.0))), 1e-9)


class ThreeBodyEnergyTest(unittest.TestCase):
    def test_figure8_energy_drift_small_over_one_period(self):
        """figure-8 跑一个周期（T≈6.326）：FR4 相对漂移应远小于 RK4。"""
        from initial_conditions import FIGURE8_PERIOD, FIGURE8_Q, FIGURE8_V

        dt = 2e-3
        n = int(round(FIGURE8_PERIOD / dt))
        state0 = np.stack([FIGURE8_Q.copy(), FIGURE8_V.copy()])
        e0 = total_energy(state0[0], state0[1])

        end_fr = forest_ruth_integrate(state0, three_body_accel, dt, n)
        drift_fr = abs(total_energy(end_fr[0], end_fr[1]) - e0) / abs(e0)

        state = state0.copy()
        deriv = three_body_deriv()
        for _ in range(n):
            state = rk4_step(state, deriv, dt)
        drift_rk4 = abs(total_energy(state[0], state[1]) - e0) / abs(e0)

        self.assertLess(drift_fr, 1e-7, f"FR4 一个周期能量漂移 {drift_fr:.2e} 超容差")
        self.assertLess(drift_fr, drift_rk4,
                        f"FR4 漂移 {drift_fr:.2e} 应小于 RK4 {drift_rk4:.2e}（辛优势未体现）")

    def test_figure8_returns_near_start_after_period(self):
        """周期解回归：一个周期后状态应回到初值附近（周期性数值验证）。"""
        from initial_conditions import FIGURE8_PERIOD, FIGURE8_Q, FIGURE8_V
        dt = 2e-3
        n = int(round(FIGURE8_PERIOD / dt))
        state0 = np.stack([FIGURE8_Q.copy(), FIGURE8_V.copy()])
        end = forest_ruth_integrate(state0, three_body_accel, dt, n)
        pos_err = float(np.linalg.norm(end[0] - state0[0]))
        # 软化 1e-6 + 离散化会让周期微移；容差放宽到 0.02 并在 README 注明
        self.assertLess(pos_err, 0.02, f"一个周期后位置偏差 {pos_err:.4f} 过大")


if __name__ == "__main__":
    unittest.main()
