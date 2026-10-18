"""可断续框架：把「万级轨道系综 × 噪声档」切成独立分片，支持中断后续跑。

设计（施工单 4.2 三个硬要求 + 幂等）：
- 分片：任务 = seed 列表；每 shard_size 个 seed 一片，片间零依赖，可单独跑。
- checkpoint：每片跑完立刻存 npz（seeds + 最终状态 + 配置）；半片中断不留脏数据
  （文件名带 .tmp，写完原子改名）。
- 断点续跑：启动时扫 checkpoint 目录，已完成的片直接跳过。
- 幂等：同 seed + 同配置 → 同结果（seed 决定噪声注入；FR4 单步只依赖当前状态，
  所以「存最终状态」即可精确续跑，不需要历史）。

后端协议（阶段 2 GPU 接入点）：integrate_batch(seeds, eps, dt, n_steps) → states (B, 2, N, D)
CPU 实现逐条调 forest_ruth；GPU 实现等 torch cu128 就位后在 gpu_kernel.py 提供，接口不变。

容差口径：分歧判据 = 受扰轨道与名义轨道位置差 > tol_position（默认 0.5，
即初值尺度的约一半——明显不再贴近周期解）；解体时间 = 首次超差时刻。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from initial_conditions import figure8_state
from integrators import forest_ruth_integrate
from metrics import separation
from systems import three_body_accel

CONFIG_VERSION = 1


def job_config(eps_list: list[float], n_seeds: int, dt: float, n_steps: int,
               tol_position: float = 0.5, shard_size: int = 250) -> dict:
    return {
        "version": CONFIG_VERSION,
        "family": "figure8",
        "eps_list": eps_list,
        "n_seeds": n_seeds,
        "dt": dt,
        "n_steps": n_steps,
        "tol_position": tol_position,
        "shard_size": shard_size,
    }


def config_id(cfg: dict) -> str:
    """配置指纹：配置变了的旧 checkpoint 自动作废，不会被错误复用。"""
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def shard_ids(cfg: dict) -> list[tuple[int, int, int]]:
    """返回 (shard_index, seed_start, seed_end)。seed 区间 [start, end)。"""
    out, s = [], 0
    idx = 0
    while s < cfg["n_seeds"]:
        e = min(s + cfg["shard_size"], cfg["n_seeds"])
        out.append((idx, s, e))
        idx += 1
        s = e
    return out


class CpuBackend:
    """逐条轨道积分（对照实现；GPU 版接口相同）。"""

    name = "cpu"

    def integrate_batch(self, seeds: list[int], eps: float, dt: float, n_steps: int) -> np.ndarray:
        states = np.empty((len(seeds), 2, 3, 2), dtype=np.float64)
        for k, seed in enumerate(seeds):
            s0 = figure8_state(eps=eps, seed=seed)
            states[k] = forest_ruth_integrate(s0, three_body_accel, dt, n_steps)
        return states


def nominal_state() -> np.ndarray:
    """名义轨道（eps=0），所有受扰轨道与它比分歧。"""
    return figure8_state(eps=0.0, seed=0)


def run_shard(cfg: dict, shard: tuple[int, int, int], ckpt_dir: Path,
              backend: CpuBackend | None = None, force: bool = False) -> Path:
    """跑一个分片并存 checkpoint。已完成且非 force → 直接返回（续跑的核心）。"""
    backend = backend or CpuBackend()
    idx, s0, s1 = shard
    seeds = list(range(s0, s1))
    out = ckpt_dir / f"shard_{idx:05d}_{s0}_{s1}.npz"
    if out.exists() and not force:
        return out

    jid = config_id(cfg)
    states = backend.integrate_batch(seeds, eps=cfg["eps_list"][0], dt=cfg["dt"], n_steps=cfg["n_steps"])
    # 分歧：与名义轨道比（简化口径：阶段 2 完整版按解族多名义轨道取中位）
    nom = nominal_state()
    seps = np.array([separation(states[k][0], nom[0]) for k in range(len(seeds))])
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:  # 传句柄：savez 不追加后缀，名字完全可控
        np.savez_compressed(
            fh, seeds=np.array(seeds), states=states, separations=seps,
            final=False,
            meta=json.dumps({"config_id": jid, "backend": backend.name, "eps": cfg["eps_list"][0]},
                            ensure_ascii=False),
        )
    tmp.rename(out)  # 原子落盘：中途断电不会留半片脏数据
    return out


def collect(cfg: dict, ckpt_dir: Path) -> dict[int, float]:
    """汇总全部已完成分片 → {seed: 解体时刻（当前口径：终态分歧是否超容差）}。

    阶段 1 骨架先记「终态分歧值」；精确解体时刻需要逐步采样（阶段 2 开 sample_every）。
    返回 {seed: final_separation}，缺片不报错（断续语义：有多少算多少）。
    """
    result: dict[int, float] = {}
    jid = config_id(cfg)
    for f in sorted(ckpt_dir.glob("shard_*.npz")):
        data = np.load(f, allow_pickle=False)
        if json.loads(str(data["meta"]))["config_id"] != jid:
            continue  # 旧配置的 checkpoint，作废跳过
        for seed, sep in zip(data["seeds"], data["separations"]):
            result[int(seed)] = float(sep)
    return result


def run_job(cfg: dict, ckpt_dir: Path, backend: CpuBackend | None = None) -> dict[int, float]:
    """跑整个任务：自动跳过已完成分片（断点续跑）；被中断后重新调用本函数即可。"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for shard in shard_ids(cfg):
        run_shard(cfg, shard, ckpt_dir, backend)
    return collect(cfg, ckpt_dir)
