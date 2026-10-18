"""v1 预览实验：figure-8 抗噪寿命谱（CPU 小规模）。

对每条受扰轨道（seed 固定 → 幂等）：
  与名义轨道（eps=0）同步 FR4 积分，每 sample_every 步比一次位置分离度，
  首次超过 tol_position 的时刻记为解体时间 T；跑到上限仍未超 → 记「右删失」
  （censored：不是没解体，是这段预算内没解体——统计上必须区分）。

输出（benchmarks/v1_preview/）：
  lifetime.csv        每 seed 一行：eps, seed, T, censored, final_sep
  summary.json        每档：解体数/删失数/解体样本的中位数与最小值
  epsilon_vs_T.png    ε↔T 散点（对数 x 轴；删失点用 ▲ 画在预算上限处）

诚实口径：这是**小样本预览**（默认 30/档、CPU、5 周期预算、单一名义轨道），
不是最终寿命谱；最终谱要 GPU 万级系综 + 多名义轨道（阶段 2/3）。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from initial_conditions import FIGURE8_PERIOD, figure8_state
from integrators import forest_ruth_step
from metrics import separation
from systems import three_body_accel

DEFAULT_EPS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def nominal_trajectory(dt: float, n_steps: int, sample_every: int) -> list[np.ndarray]:
    """名义轨道（eps=0）：每 sample_every 步存一次位置。"""
    state = figure8_state(eps=0.0)
    traj = [state[0].copy()]
    for i in range(n_steps):
        state = forest_ruth_step(state, three_body_accel, dt)
        if (i + 1) % sample_every == 0:
            traj.append(state[0].copy())
    return traj


def measure_lifetime(seed: int, eps: float, dt: float, n_steps: int,
                     sample_every: int, nominal: list[np.ndarray],
                     tol: float) -> tuple[float | None, float]:
    """返回 (T 或 None=删失, 终态分离度)。"""
    state = figure8_state(eps=eps, seed=seed)
    k = 0
    for i in range(n_steps):
        state = forest_ruth_step(state, three_body_accel, dt)
        if (i + 1) % sample_every == 0:
            k += 1
            if separation(state[0], nominal[min(k, len(nominal) - 1)]) > tol:
                return (i + 1) * dt, separation(state[0], nominal[min(k, len(nominal) - 1)])
    return None, separation(state[0], nominal[-1])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="figure-8 抗噪寿命预览（CPU）")
    p.add_argument("--seeds-per-eps", type=int, default=30)
    p.add_argument("--max-periods", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=2e-3)
    p.add_argument("--sample-every", type=int, default=20)
    p.add_argument("--tol", type=float, default=0.5)
    p.add_argument("--eps-list", type=str, default=",".join(str(e) for e in DEFAULT_EPS))
    p.add_argument("--seed-base", type=int, default=1000, help="seed = base + 0..n-1")
    p.add_argument("--out", type=str, default=None, help="输出目录（默认 benchmarks/v1_preview）")
    args = p.parse_args(argv)

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "benchmarks" / "v1_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    eps_list = [float(e) for e in args.eps_list.split(",")]
    dt, n_steps = args.dt, int(round(args.max_periods * FIGURE8_PERIOD / args.dt))

    print(f"配置：dt={dt}, n_steps={n_steps}（≈{args.max_periods} 周期）, "
          f"{len(eps_list)} 档 × {args.seeds_per_eps} 条, tol={args.tol}, 采样间隔={args.sample_every} 步")

    print("积分名义轨道（eps=0）……")
    nominal = nominal_trajectory(dt, n_steps, args.sample_every)

    rows = []
    for eps in eps_list:
        ts, censored = [], 0
        for j in range(args.seeds_per_eps):
            seed = args.seed_base + j
            t, final_sep = measure_lifetime(seed, eps, dt, n_steps, args.sample_every, nominal, args.tol)
            if t is None:
                censored += 1
            else:
                ts.append(t)
            rows.append({"eps": eps, "seed": seed, "T": t if t is not None else "",
                         "censored": int(t is None), "final_sep": final_sep})
        med = statistics.median(ts) if ts else None
        print(f"ε={eps:g}: 解体 {len(ts)}/{args.seeds_per_eps}（删失 {censored}）"
              + (f"，解体样本中位 T={med:.3f}，最小 T={min(ts):.3f}" if ts else "，本预算内无一解体"))

    # 数据文件
    csv_path = out_dir / "lifetime.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "seed", "T", "censored", "final_sep"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "config": {"dt": dt, "n_steps": n_steps, "max_periods": args.max_periods,
                    "sample_every": args.sample_every, "tol_position": args.tol,
                    "seeds_per_eps": args.seeds_per_eps, "seed_base": args.seed_base,
                    "integrator": "Forest-Ruth 4 阶", "softening": 1e-6},
        "per_eps": {},
    }
    for eps in eps_list:
        sub = [r for r in rows if r["eps"] == eps]
        ts = [r["T"] for r in sub if not r["censored"]]
        summary["per_eps"][f"{eps:g}"] = {
            "n": len(sub), "dissolved": len(ts), "censored": len(sub) - len(ts),
            "T_median": statistics.median(ts) if ts else None,
            "T_min": min(ts) if ts else None,
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图：对数 x 轴（ε），y=T；删失点画在预算上限（▲）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # Windows 中文字体
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.5, 5))
    budget = args.max_periods * FIGURE8_PERIOD
    for eps in eps_list:
        sub = [r for r in rows if r["eps"] == eps]
        xs_d = [eps for r in sub if not r["censored"]]
        ys_d = [r["T"] for r in sub if not r["censored"]]
        xs_c = [eps for r in sub if r["censored"]]
        ax.scatter(xs_d, ys_d, s=18, alpha=0.65, marker="o", label="_")
        if xs_c:
            ax.scatter(xs_c, [budget] * len(xs_c), s=26, alpha=0.8, marker="^", label="_")
        ts = [r["T"] for r in sub if not r["censored"]]
        if ts:
            ax.scatter([eps], [statistics.median(ts)], s=140, marker="_", linewidths=3, color="black", label="_")
    ax.set_xscale("log")
    ax.set_xlabel("初值噪声 ε（对数轴）")
    ax.set_ylabel("解体时间 T（时间单位；一个 figure-8 周期 ≈ 6.33）")
    ax.set_title(f"figure-8 抗噪寿命预览（CPU，每档 {args.seeds_per_eps} 条，"
                 f"预算 {args.max_periods:g} 周期）\n● 解体　▲ 到预算仍存活（画在上限）　— 中位数")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, budget * 1.06)
    fig.tight_layout()
    png_path = out_dir / "epsilon_vs_T.png"
    fig.savefig(png_path, dpi=150)

    print(f"\n产物：\n  {csv_path}\n  {out_dir / 'summary.json'}\n  {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
