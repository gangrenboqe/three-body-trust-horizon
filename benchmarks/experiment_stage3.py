"""阶段 3：figure-8 抗噪可信时界（trust horizon）寿命谱 + 数据集导出。

定义（本文件写死，避免口径漂移）：
  对每条受扰轨道（初值噪声 ε，seed 固定），与名义轨道（ε=0）同步积分，
  每隔 sample_every 步比较三体位置差的 Frobenius 范数 sep(t)；
  **首次 sep(t) > tol 的时刻记为可信时界 T**；到预算上限仍未超过 → 记「右删失」。
  ⚠️ 这不是 v1 的「解体时间」：v1 口径已被 v2/v3 推翻（figure-8 线性稳定，低 ε 不会解体）。
  本文件测的是「数值轨道偏离名义轨道到不可忽略（tol）所需的时间」，即 trust horizon。

诚实口径（写进 summary.json 与 README）：
  - CPU 小样本（默认 5 档 ε × 30 条），**不是 GPU 万级系综**；
  - 预算默认 20 个 figure-8 周期（≈126.6 时间单位），到上限未超 = 右删失，不是「永不」；
  - 单一名义轨道（ε=0），软化 1e-6；结论不外推到其他周期解。
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
    state = figure8_state(eps=0.0)
    traj = [state[0].copy()]
    for i in range(n_steps):
        state = forest_ruth_step(state, three_body_accel, dt)
        if (i + 1) % sample_every == 0:
            traj.append(state[0].copy())
    return traj


def measure_horizon(seed: int, eps: float, dt: float, n_steps: int,
                    sample_every: int, nominal: list[np.ndarray],
                    tol: float) -> tuple[float | None, float]:
    state = figure8_state(eps=eps, seed=seed)
    k = 0
    # ⚠️ final_sep 必须取「最后一个对齐采样点」的分离度。
    # 2026-10-27 修 bug：原实现循环结束后用 state[0]（第 n_steps 步）去比 nominal[-1]
    # （最后一个采样点，n_steps 不一定整除 sample_every，默认差 19 步），
    # 凭空多出 |v|·Δt≈0.059 的位移，把所有右删失行抬到同一个假「地板」。
    last_sep = separation(state[0], nominal[0])
    for i in range(n_steps):
        state = forest_ruth_step(state, three_body_accel, dt)
        if (i + 1) % sample_every == 0:
            k += 1
            last_sep = separation(state[0], nominal[min(k, len(nominal) - 1)])
            if last_sep > tol:
                return (i + 1) * dt, last_sep
    return None, last_sep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="figure-8 可信时界寿命谱（阶段 3，CPU 小样本）")
    p.add_argument("--seeds-per-eps", type=int, default=30)
    p.add_argument("--max-periods", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=2e-3)
    p.add_argument("--sample-every", type=int, default=20)
    p.add_argument("--tol", type=float, default=0.1,
                   help="可信时界判据：位置分离度首次超过该值（默认 0.1，约为轨道尺度 10%%）")
    p.add_argument("--eps-list", type=str, default=",".join(str(e) for e in DEFAULT_EPS))
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "benchmarks" / "stage3_lifetime")
    out_dir.mkdir(parents=True, exist_ok=True)
    eps_list = [float(e) for e in args.eps_list.split(",")]
    dt = args.dt
    n_steps = int(round(args.max_periods * FIGURE8_PERIOD / dt))
    budget = args.max_periods * FIGURE8_PERIOD

    print(f"配置：dt={dt}, n_steps={n_steps}（≈{args.max_periods:g} 周期，预算 T_budget={budget:.2f}），"
          f"{len(eps_list)} 档 × {args.seeds_per_eps} 条，tol={args.tol}, sample_every={args.sample_every}",
          flush=True)
    print("积分名义轨道（ε=0）……", flush=True)
    nominal = nominal_trajectory(dt, n_steps, args.sample_every)
    print(f"名义轨道完成（{len(nominal)} 个采样点）", flush=True)

    rows = []
    for eps in eps_list:
        ts, censored = [], 0
        for j in range(args.seeds_per_eps):
            seed = args.seed_base + j
            t, final_sep = measure_horizon(seed, eps, dt, n_steps, args.sample_every, nominal, args.tol)
            if t is None:
                censored += 1
            else:
                ts.append(t)
            rows.append({"eps": eps, "seed": seed, "T": t if t is not None else "",
                         "censored": int(t is None), "final_sep": final_sep})
            if (j + 1) % 5 == 0:
                print(f"  [ε={eps:g}] {j+1}/{args.seeds_per_eps} 条完成"
                      f"（已越界 {len(ts)}，删失 {censored}）", flush=True)
        med = statistics.median(ts) if ts else None
        print(f"ε={eps:g}: 越界 {len(ts)}/{args.seeds_per_eps}（删失 {censored}）"
              + (f"，中位 T={med:.3f}，最小 T={min(ts):.3f}" if ts else "，本预算内无一越界"),
              flush=True)

    csv_path = out_dir / "lifetime.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "seed", "T", "censored", "final_sep"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "definition": "T = 首次 |q_perturbed(t) - q_nominal(t)|_F > tol 的时刻；到预算未超 = 右删失",
        "config": {"dt": dt, "n_steps": n_steps, "max_periods": args.max_periods,
                   "T_budget": budget, "sample_every": args.sample_every,
                   "tol_position": args.tol, "seeds_per_eps": args.seeds_per_eps,
                   "seed_base": args.seed_base, "integrator": "Forest-Ruth 4 阶",
                   "softening": 1e-6, "nominal_trajectory": "figure-8, eps=0"},
        "per_eps": {},
    }
    for eps in eps_list:
        sub = [r for r in rows if r["eps"] == eps]
        ts = [r["T"] for r in sub if not r["censored"]]
        summary["per_eps"][f"{eps:g}"] = {
            "n": len(sub), "crossed": len(ts), "censored": len(sub) - len(ts),
            "T_median": statistics.median(ts) if ts else None,
            "T_min": min(ts) if ts else None,
            "T_max": max(ts) if ts else None,
            "censored_fraction": (len(sub) - len(ts)) / len(sub),
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                  gridspec_kw={"width_ratios": [2.1, 1]})
    for eps in eps_list:
        sub = [r for r in rows if r["eps"] == eps]
        xs_d = [eps for r in sub if not r["censored"]]
        ys_d = [r["T"] for r in sub if not r["censored"]]
        xs_c = [eps for r in sub if r["censored"]]
        ax.scatter(xs_d, ys_d, s=18, alpha=0.65, marker="o", color="#1f77b4")
        if xs_c:
            ax.scatter(xs_c, [budget] * len(xs_c), s=26, alpha=0.8, marker="^", color="#d62728")
        ts = [r["T"] for r in sub if not r["censored"]]
        if ts:
            ax.scatter([eps], [statistics.median(ts)], s=150, marker="_", linewidths=3, color="black")
    ax.set_xscale("log")
    ax.set_xlabel("初值噪声 ε（对数轴）")
    ax.set_ylabel("可信时界 T（时间单位；1 周期 ≈ 6.33）")
    ax.set_title(f"figure-8 抗噪可信时界寿命谱（每档 {args.seeds_per_eps} 条，预算 {args.max_periods:g} 周期）\n"
                 "● 已越界　▲ 到预算仍可信（右删失）　— 中位数")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, budget * 1.08)

    fracs = [summary["per_eps"][f"{e:g}"]["censored_fraction"] for e in eps_list]
    ax2.bar(range(len(eps_list)), [1 - f for f in fracs], color="#1f77b4", alpha=0.85)
    ax2.set_xticks(range(len(eps_list)))
    ax2.set_xticklabels([f"{e:g}" for e in eps_list], rotation=45)
    ax2.set_xlabel("ε"); ax2.set_ylabel("越界比例（crossed / n）")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("各档越界比例\n（其余为右删失）")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png = out_dir / "epsilon_vs_T_spectrum.png"
    fig.savefig(png, dpi=150)

    print(f"\n产物：\n  {csv_path}\n  {out_dir/'summary.json'}\n  {png}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
