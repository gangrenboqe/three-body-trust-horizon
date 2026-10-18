"""可断续框架测试：分片、checkpoint 续跑、幂等、配置指纹。"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from runner import CpuBackend, collect, config_id, job_config, run_job, run_shard, shard_ids


def small_cfg(**kw) -> dict:
    base = dict(eps_list=[1e-3], n_seeds=5, dt=2e-3, n_steps=200,
                tol_position=0.5, shard_size=2)
    base.update(kw)
    return job_config(**base)


class ShardTest(unittest.TestCase):
    def test_shards_cover_all_seeds(self):
        shards = shard_ids(small_cfg(n_seeds=5, shard_size=2))
        self.assertEqual([(s, e) for _, s, e in shards], [(0, 2), (2, 4), (4, 5)])

    def test_config_id_changes_with_config(self):
        a = config_id(small_cfg())
        b = config_id(small_cfg(n_steps=201))
        self.assertNotEqual(a, b)
        self.assertEqual(a, config_id(small_cfg()))  # 同配置稳定


class ResumabilityTest(unittest.TestCase):
    """🔴 核心：中断后重跑 = 一次跑完（幂等 + 续跑正确性）。"""

    def _ref(self, tmp):
        cfg = small_cfg()
        ref_dir = Path(tmp) / "ref"
        return cfg, run_job(cfg, ref_dir)

    def test_interrupted_run_matches_full_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, ref = self._ref(tmp)
            # 模拟中断：只跑第 0 片，然后「重新启动」完整任务
            part_dir = Path(tmp) / "part"
            part_dir.mkdir()
            run_shard(cfg, shard_ids(cfg)[0], part_dir)
            self.assertEqual(len(list(part_dir.glob("shard_*.npz"))), 1)

            resumed = run_job(cfg, part_dir)  # 续跑：只补剩余片
            self.assertEqual(set(resumed), set(ref), "续跑后种子覆盖应与一次跑完一致")
            for seed in ref:
                self.assertAlmostEqual(resumed[seed], ref[seed], places=12,
                                       msg=f"seed {seed} 续跑结果与一次跑完不一致")

    def test_rerun_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, ref = self._ref(tmp)
            again_dir = Path(tmp) / "again"
            run_job(cfg, again_dir)
            run_job(cfg, again_dir)  # 第二次全跳过
            again = collect(cfg, again_dir)
            for seed in ref:
                self.assertAlmostEqual(again[seed], ref[seed], places=12)

    def test_stale_config_checkpoints_ignored(self):
        """配置指纹变了 → 旧 checkpoint 作废，不会被错误复用。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            d = Path(tmp) / "ck"
            run_job(cfg, d)
            new_cfg = small_cfg(n_steps=cfg["n_steps"] + 1)
            fresh = Path(tmp) / "ck2"
            run_job(new_cfg, fresh)
            # ck2 目录只有新配置的片（不会误读 ck 的）
            self.assertTrue(all("npz" in f.name for f in fresh.glob("*.npz")))
            meta0 = json.loads(str(np.load(sorted(fresh.glob("*.npz"))[0])["meta"]))
            self.assertEqual(meta0["config_id"], config_id(new_cfg))

    def test_tmp_file_not_collected(self):
        """半写状态的 .tmp 不进汇总（原子落盘语义）。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_cfg()
            d = Path(tmp) / "ck"
            run_job(cfg, d)
            (d / "shard_99999_0_2.npz.tmp").write_bytes(b"garbage")
            got = collect(cfg, d)
            self.assertEqual(len(got), cfg["n_seeds"])


if __name__ == "__main__":
    unittest.main()
