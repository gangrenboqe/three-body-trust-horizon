"""metrics 与初值注入的测试：trust_horizon 判据、噪声注入幂等。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from initial_conditions import figure8_state
from metrics import separation, trust_horizon


class TrustHorizonTest(unittest.TestCase):
    def test_detects_first_crossing(self):
        seps = np.array([0.1, 0.2, 0.5, 0.9])
        times = np.array([0.0, 1.0, 2.0, 3.0])
        self.assertEqual(trust_horizon(seps, times, 0.3), 2.0)

    def test_returns_none_when_never_crosses(self):
        self.assertIsNone(trust_horizon(np.array([0.1, 0.2]), np.array([0.0, 1.0]), 0.5))

    def test_separation_symmetry(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.1, 0.0], [0.0, 0.9]])
        # 位置差 = (0.1, 0.0) 与 (0.0, 0.1)，Frobenius 范数 = sqrt(0.01+0.01)
        self.assertAlmostEqual(separation(a, b), separation(b, a))
        self.assertAlmostEqual(separation(a, b), np.sqrt(0.02))


class NoiseIdempotencyTest(unittest.TestCase):
    def test_same_seed_same_noise(self):
        """幂等根基：同 seed 注入的噪声逐位一致（阶段 2 断点续跑的前提）。"""
        a = figure8_state(eps=1e-3, seed=42)
        b = figure8_state(eps=1e-3, seed=42)
        c = figure8_state(eps=1e-3, seed=43)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_zero_eps_is_exact(self):
        self.assertTrue(np.array_equal(figure8_state(eps=0.0, seed=7),
                                       figure8_state(eps=0.0, seed=12345)))


if __name__ == "__main__":
    unittest.main()
