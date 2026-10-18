"""v2 实验：figure-8 分离度增长曲线 + 最大李雅普诺夫指数 μ 测定。

🔴 口径修正（WorkBuddy 2026-10-17 复核 v1 后指出）：figure-8 是**线性稳定**周期解——
Simó (2002) 给出 elliptical stability（特征值在单位圆上），线性稳定意味着小扰动**不指数增长**，
轨道会长期在附近振荡。v1 测「解体时间」时低 ε 档全删失是**物理**，不是预算不够。

因此 v2 改测：
  1) 逐步记录 sep(t) = ‖q_受扰 − q_名义‖（每 sample_every 步采样，sep > 0.1 早停）；
  2) **只取标度区** 1e-8 < sep < 0.1 的采样点做线性回归 ln(sep) = μ·t + c
     （下界避开浮点噪声地板，上界避开轨道散架后的非线性段——这一步不许省）；
  3) 斜率 μ 即最大李雅普诺夫指数估计，同时报 R² 与拟合点数；
  4) 自洽性检验：μ 是轨道性质、与扰动大小无关——不同 ε 测出的 μ 应接近。

诚实口径：
  - 若 μ ≈ 0（或 R² 很低、无指数标度区）→ **实测与 Simó 线性稳定结论一致，是成功结果**；
  - 若 μ > 0 且显著 → 是真发现，必须写清复现方法，不许含糊。

本轮仍为 CPU（不装 torch）；不动 integrators/systems/runner/tests。
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

FIT_LO, FIT_HI = 1e-8, 0.1  # 标度区（施工口径：下界避浮点噪声，上界避散架）


def nominal_trajectory(dt: float, n_steps: int, sample_every: int) -> list[np.ndarray]:
    state = figure8_state(eps=0.0)
    traj = [state[0].copy()]
    for i in range(n_steps):
        state = forest_ruth_step(state, three_body_accel, dt)
        if (i + 1) % sample_every == 0:
            traj.append(state[0].copy())
    return traj


def sep_series(seed: int, eps: float, dt: float, n_steps: int, sample_every: int,
               nominal: list[np.ndarray], stop_above: float = 0.1) -> tuple[np.ndarray, np.ndarray, bool]:
    """一条受扰轨道的 sep(t) 采样序列。返回 (times, seps, stopped_early)。"""
    state = figure8_state(eps=eps, seed=seed)
    times, seps = [0.0], [separation(state[0], nominal[0])]
    k = 0
    for i in range(n_steps):
        state = forest_ruth_step(state, three_body_accel, dt)
        if (i + 1) % sample_every == 0:
            k += 1
            t = (i + 1) * dt
            sep = separation(state[0], nominal[min(k, len(nominal) - 1)])
            times.append(t)
            seps.append(sep)
            if sep > stop_above:
                return np.array(times), np.array(seps), True
    return np.array(times), np.array(seps), False


def fit_mu(times: np.ndarray, seps: np.ndarray) -> dict:
    """只取标度区 (FIT_LO, FIT_HI) 内的点做 ln(sep)=μ·t+c 线性回归。"""
    mask = (seps > FIT_LO) & (seps < FIT_HI)
    t = times[mask]
    y = np.log(seps[mask])
    n = int(mask.sum())
    out = {"n_points": n, "mu": None, "intercept": None, "r2": None}
    if n < 10:
        out["note"] = "标度区内点数不足 10，拟合不可靠"
        return out
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out.update({
        "mu": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "t_min": float(t.min()), "t_max": float(t.max()),
        "sep_min": float(seps[mask].min()), "sep_max": float(seps[mask].max()),
    })
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="figure-8 分离度增长曲线 + Lyapunov 指数 μ 测定（CPU）")
    p.add_argument("--seeds-per-eps", type=int, default=30)
    p.add_argument("--max-periods", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=2e-3)
    p.add_argument("--sample-every", type=int, default=20)
    p.add_argument("--eps-list", type=str, default="1e-2,1e-3,1e-4,1e-5,1e-6")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "benchmarks" / "v2_lyapunov"
    out_dir.mkdir(parents=True, exist_ok=True)
    eps_list = [float(e) for e in args.eps_list.split(",")]
    dt = args.dt
    n_steps = int(round(args.max_periods * FIGURE8_PERIOD / dt))

    print(f"配置：dt={dt}, n_steps={n_steps}（≈{args.max_periods} 周期）, "
          f"{len(eps_list)} 档 × {args.seeds_per_eps} 条, 标度区 [{FIT_LO:g}, {FIT_HI:g}]")
    print("积分名义轨道（eps=0）……")
    nominal = nominal_trajectory(dt, n_steps, args.sample_every)

    series_rows, mu_rows = [], []
    for eps in eps_list:
        mus, r2s = [], []
        for j in range(args.seeds_per_eps):
            seed = args.seed_base + j
            times, seps, early = sep_series(seed, eps, dt, n_steps, args.sample_every, nominal)
            for t, s in zip(times, seps):
                series_rows.append({"eps": eps, "seed": seed, "t": t, "sep": s})
            fit = fit_mu(times, seps)
            mu_rows.append({"eps": eps, "seed": seed, "early_stop": int(early), **fit})
            if fit["mu"] is not None:
                mus.append(fit["mu"])
                if fit["r2"] is not None:
                    r2s.append(fit["r2"])
        line = f"ε={eps:g}: "
        if mus:
            line += (f"μ 中位 {statistics.median(mus):.4g}，均值 {statistics.mean(mus):.4g}±"
                     f"{statistics.pstdev(mus):.2g}，n_fit={len(mus)}"
                     + (f"，R² 中位 {statistics.median(r2s):.3f}" if r2s else ""))
        else:
            line += "无一条轨道有可用标度区拟合"
        print(line)

    # 产物 1：sep 序列（长表）
    with open(out_dir / "sep_series.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "seed", "t", "sep"])
        w.writeheader()
        w.writerows(series_rows)

    # 产物 2：μ 表
    with open(out_dir / "mu_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "seed", "early_stop", "n_points",
                                          "mu", "intercept", "r2", "t_min", "t_max",
                                          "sep_min", "sep_max", "note"])
        w.writeheader()
        for r in mu_rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in w.fieldnames})

    # 产物 3：汇总
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "hypothesis": "figure-8 线性稳定（Simó 2002）→ 预期 μ≈0 或无显著指数标度区",
        "config": {"dt": dt, "n_steps": n_steps, "max_periods": args.max_periods,
                    "sample_every": args.sample_every, "fit_window": [FIT_LO, FIT_HI],
                    "seeds_per_eps": args.seeds_per_eps, "seed_base": args.seed_base,
                    "integrator": "Forest-Ruth 4 阶", "softening": 1e-6},
        "per_eps": {},
    }
    for eps in eps_list:
        sub = [r for r in mu_rows if r["eps"] == eps and r["mu"] is not None]
        summary["per_eps"][f"{eps:g}"] = {
            "n_orbits": args.seeds_per_eps,
            "n_fittable": len(sub),
            "mu_median": statistics.median([r["mu"] for r in sub]) if sub else None,
            "mu_mean": statistics.mean([r["mu"] for r in sub]) if sub else None,
            "mu_pstdev": statistics.pstdev([r["mu"] for r in sub]) if sub else None,
            "r2_median": statistics.median([r["r2"] for r in sub if r["r2"] is not None]) if sub else None,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图 1：t vs sep（log y），每档取 3 条代表轨道 + 每档标度区带
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(eps_list)))
    for ci, eps in enumerate(eps_list):
        sub_seeds = sorted({r["seed"] for r in series_rows if r["eps"] == eps})
        for seed in sub_seeds[:3]:
            pts = [(r["t"], r["sep"]) for r in series_rows if r["eps"] == eps and r["seed"] == seed]
            if pts:
                t, s = zip(*pts)
                ax.plot(t, s, color=colors[ci], alpha=0.55, linewidth=1.1,
                        label=f"ε={eps:g}" if seed == sub_seeds[0] else "_")
    ax.axhspan(FIT_LO, FIT_HI, color="gray", alpha=0.15, label="拟合标度区")
    ax.set_yscale("log")
    ax.set_xlabel("t（时间单位；一个 figure-8 周期 ≈ 6.33）")
    ax.set_ylabel("sep(t) = ‖q_受扰 − q_名义‖（对数轴）")
    ax.set_title(f"figure-8 分离度增长曲线（CPU，每档 {args.seeds_per_eps} 条，每档画 3 条）")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "sep_curves.png", dpi=150)

    # 图 2：μ 估计跨 ε 对比（每轨道一点 + 中位线）
    fig2, ax2 = plt.subplots(figsize=(7, 4.8))
    for ci, eps in enumerate(eps_list):
        mus = [r["mu"] for r in mu_rows if r["eps"] == eps and r["mu"] is not None]
        if mus:
            ax2.scatter([eps] * len(mus), mus, s=20, alpha=0.6, color=colors[ci])
            ax2.scatter([eps], [statistics.median(mus)], s=150, marker="_",
                        linewidths=3, color="black")
    ax2.set_xscale("log")
    ax2.set_xlabel("初值噪声 ε（对数轴）")
    ax2.set_ylabel("拟合斜率 μ（= 最大李雅普诺夫指数估计）")
    ax2.set_title("μ 的自洽性检验：μ 是轨道性质，不同 ε 应测出接近的值\n（黑线=该档中位数；若 μ≈0 即与 Simó 线性稳定结论一致）")
    ax2.axhline(0.0, color="red", linestyle="--", alpha=0.6, label="μ=0（线性稳定）")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(out_dir / "mu_vs_eps.png", dpi=150)

    print(f"\n产物：\n  {out_dir / 'sep_series.csv'}\n  {out_dir / 'mu_table.csv'}\n  "
          f"{out_dir / 'summary.json'}\n  {out_dir / 'sep_curves.png'}\n  {out_dir / 'mu_vs_eps.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
