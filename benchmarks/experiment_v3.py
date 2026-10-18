"""v3 实验：分段拟合 + 20 周期预算（判别「μ 是否只是被平均掉的表象」）。

🔴 本轮要修正的解读（WorkBuddy 2026-10-17 复核 v2 后指出，且已由数据证实）：
v2 报的 μ≈0.09（全窗口拟合）是**平均量，不是真实速率**。同一条轨道分段看：
前半段 ≈0.25、后半段 ≈0.04——真实情况是「先快后慢、速率在衰减」。因此
「μ>0 ⇒ 不稳定」的解读是**错的**，必须用分段拟合来看速率的走向。

本脚本做什么：
 1) 只跑一档（默认 ε=1e-5），30 条轨道，预算 20 周期（探索性，非最终定论）；
 2) **按全局时间四等分**把时间轴切成 S1/S2/S3/S4 四段（边界固定，与轨道无关，
    保证跨轨道、跨 ε 可比；段定义 = 总积分时长的 0-25%/25-50%/50-75%/75-100%）；
 3) 每段**只取标度区内点**（1e-8 < sep < 0.1）独立回归 ln(sep)=μ_seg·t+c，
    点数 <10 的段标出但不报 μ（不许硬报）；
 4) 两图：主图（代表轨道 sep(t) + 4 条分段拟合线，斜率/R²/点数标在图上）+
    衰减图（各段 μ_seg 分布 + 中位数连线，一眼看斜率是否在变小）；
 5) f(t) 不变性再验：同 seed 跨 ε 的 sep 增长倍数应相同（sep(t)=ε·f(t)），
    并画 sep(t)/ε 折叠图（不同 ε 曲线应重合）。

判据（两种都是好结果，四段全报，不许只报好看的）：
   A. S3、S4 的 μ_seg 继续明显下降（如 0.25→0.04→0.01→0.003）→ 线性稳定确认，
      与 Simó (2002) 一致；
   B. S4 稳在 ~0.04 不再降 → 真有正 Lyapunov 指数，与 Simó 不一致，须写清
      复现方法 + 排除过的干扰（软化 / 数值扩散）；
   C. 段间跳动、R² 普遍低 → 写「没测出干净的指数标度区」，不硬报数。

边界：本轮仍 CPU（不装 torch）；integrators/systems/runner/tests 零改动；
v1/v2 产物保留不删；20 周期为探索性预算，README 必须写明「非最终定论」。
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

# 复用 v2 的采样与拟合基础设施（只 import，不改动 experiment_v2.py 一个字符）
from experiment_v2 import FIT_HI, FIT_LO, nominal_trajectory, sep_series  # noqa: E402

from initial_conditions import FIGURE8_PERIOD  # noqa: E402

SEG_LABELS = ["S1", "S2", "S3", "S4"]


def seg_edges(total_time: float, n_segments: int = 4) -> np.ndarray:
    """全局时间四等分边界（与轨道无关，固定）。"""
    return np.linspace(0.0, total_time, n_segments + 1)


def fit_segment(t: np.ndarray, s: np.ndarray) -> dict:
    """一段内的标度区直线拟合。点数 <10 → 不报 μ。"""
    if t.size < 10:
        return {"n_points": int(t.size), "mu_seg": None, "intercept": None, "r2": None,
                "note": "标度区内点数不足 10，不报 μ"}
    y = np.log(s)
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"n_points": int(t.size), "mu_seg": float(slope), "intercept": float(intercept),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None, "note": ""}


def collect_series(seed: int, eps: float, dt: float, n_steps: int, sample_every: int,
                   nominal, samples: int) -> list[tuple[float, float]]:
    """取一条轨道的 (t, sep) 采样序列（截断到 samples 点，保证跨 ε 用同一时间网格）。"""
    times, seps, _ = sep_series(seed, eps, dt, n_steps, sample_every, nominal, stop_above=FIT_HI)
    pts = list(zip(times.tolist(), seps.tolist()))
    return pts[:samples]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="figure-8 分段 Lyapunov 拟合（CPU，探索性预算）")
    p.add_argument("--eps-seg", type=float, default=1e-5, help="主分析档（默认 1e-5）")
    p.add_argument("--eps-check", type=float, default=1e-4, help="f(t) 不变性对照档")
    p.add_argument("--seeds-per-eps", type=int, default=30)
    p.add_argument("--check-orbits", type=int, default=30, help="对照档轨道数（可少于主档）")
    p.add_argument("--max-periods", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=2e-3)
    p.add_argument("--sample-every", type=int, default=20)
    p.add_argument("--n-segments", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "benchmarks" / "v3_segmented"
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = args.dt
    n_steps = int(round(args.max_periods * FIGURE8_PERIOD / dt))
    total_time = n_steps * dt
    edges = seg_edges(total_time, args.n_segments)
    n_samples = n_steps // args.sample_every  # 统一采样点数（预估上界）

    print(f"配置：dt={dt}, n_steps={n_steps}（≈{args.max_periods:g} 周期, 总时长 {total_time:.2f}）, "
          f"标度区 [{FIT_LO:g}, {FIT_HI:g}], 分段=全局时间四等分")
    seg_desc = ", ".join(
        f"{SEG_LABELS[i]}[{edges[i]:.1f}, {edges[i + 1]:.1f})" for i in range(args.n_segments)
    )
    print(f"段边界：{seg_desc}")

    print("积分名义轨道（eps=0）……")
    nominal = nominal_trajectory(dt, n_steps, args.sample_every)

    series_rows, seg_rows = [], []
    growth_rows = []

    for eps, n_orbits, tag in ((args.eps_seg, args.seeds_per_eps, "seg"),
                               (args.eps_check, args.check_orbits, "check")):
        print(f"\n=== 档 {tag}: ε={eps:g}, {n_orbits} 条 ===")
        per_orbit_mu_full = {}
        for j in range(n_orbits):
            seed = args.seed_base + j
            pts = collect_series(seed, eps, dt, n_steps, args.sample_every, nominal, n_samples)
            # 逐轨道进度（flush=True）：任务是分钟级，必须让运行者能实时看到进展，
            # 不搞「后台黑箱等半天」（2026-10-18 用户反馈后加的）。
            print(f"  [{tag} ε={eps:g}] {j + 1}/{n_orbits} seed={seed}", flush=True)
            for t, s in pts:
                series_rows.append({"eps": eps, "seed": seed, "t": t, "sep": s})

            ts = np.array([t for t, _ in pts])
            ss = np.array([s for _, s in pts])
            band = (ss > FIT_LO) & (ss < FIT_HI)
            full = fit_segment(ts[band], ss[band])
            per_orbit_mu_full[seed] = full["mu_seg"]

            for i in range(args.n_segments):
                in_seg = (ts >= edges[i]) & (ts < edges[i + 1]) & band
                fit = fit_segment(ts[in_seg], ss[in_seg])
                seg_rows.append({"eps": eps, "seed": seed, "seg": SEG_LABELS[i],
                                 "t_lo": float(edges[i]), "t_hi": float(edges[i + 1]), **fit})

            if pts:
                first_sep, last_sep = pts[0][1], pts[-1][1]
                growth_rows.append({"eps": eps, "seed": seed,
                                    "sep_first": first_sep, "sep_last": last_sep,
                                    "growth": (last_sep / first_sep) if first_sep > 0 else None,
                                    "last_in_band": int(last_sep < FIT_HI)})

        # 逐段汇总
        for i, lab in enumerate(SEG_LABELS):
            mus = [r["mu_seg"] for r in seg_rows if r["eps"] == eps and r["seg"] == lab
                   and r["mu_seg"] is not None]
            r2s = [r["r2"] for r in seg_rows if r["eps"] == eps and r["seg"] == lab
                   and r["r2"] is not None]
            msg = f"  {lab} [{edges[i]:.1f}, {edges[i+1]:.1f}): 可拟合 {len(mus)}/{n_orbits}"
            if mus:
                msg += (f"，μ_seg 中位 {statistics.median(mus):.4f}"
                        f"（min {min(mus):.4f} / max {max(mus):.4f}）"
                        + (f"，R² 中位 {statistics.median(r2s):.3f}" if r2s else ""))
            print(msg)

    # ---------- 产物 ----------
    with open(out_dir / "sep_series_v3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "seed", "t", "sep"])
        w.writeheader(); w.writerows(series_rows)

    seg_fields = ["eps", "seed", "seg", "t_lo", "t_hi", "n_points", "mu_seg", "intercept", "r2", "note"]
    with open(out_dir / "seg_fit_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=seg_fields)
        w.writeheader()
        for r in seg_rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in seg_fields})

    with open(out_dir / "growth_factors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "seed", "sep_first", "sep_last", "growth", "last_in_band"])
        w.writeheader(); w.writerows(growth_rows)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "note": "20 周期为探索性预算，结论非最终定论",
        "correction": "v2 的全窗口 μ 是平均量；真实速率需看分段",
        "config": {"dt": dt, "n_steps": n_steps, "max_periods": args.max_periods,
                    "sample_every": args.sample_every, "fit_window": [FIT_LO, FIT_HI],
                    "seg_edges": edges.tolist(), "n_segments": args.n_segments,
                    "eps_seg": args.eps_seg, "eps_check": args.eps_check,
                    "seeds_per_eps": args.seeds_per_eps, "check_orbits": args.check_orbits,
                    "seed_base": args.seed_base},
        "per_eps": {},
        "growth_invariance": {},
    }
    for eps, n_orbits in ((args.eps_seg, args.seeds_per_eps), (args.eps_check, args.check_orbits)):
        ekey = f"{eps:g}"
        summary["per_eps"][ekey] = {}
        for lab in SEG_LABELS:
            mus = [r["mu_seg"] for r in seg_rows if r["eps"] == eps and r["seg"] == lab and r["mu_seg"] is not None]
            r2s = [r["r2"] for r in seg_rows if r["eps"] == eps and r["seg"] == lab and r["r2"] is not None]
            summary["per_eps"][ekey][lab] = {
                "n_fittable": len(mus), "n_orbits": n_orbits,
                "mu_median": statistics.median(mus) if mus else None,
                "mu_min": min(mus) if mus else None, "mu_max": max(mus) if mus else None,
                "r2_median": statistics.median(r2s) if r2s else None,
                "r2_min": min(r2s) if r2s else None,
            }
        g = {r["seed"]: r["growth"] for r in growth_rows if r["eps"] == eps}
        summary["growth_invariance"][ekey] = {
            "n": len(g),
            "growth_median": statistics.median([v for v in g.values() if v]) if g else None,
            "growth_min": min([v for v in g.values() if v]) if g else None,
            "growth_max": max([v for v in g.values() if v]) if g else None,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    eps_seg = args.eps_seg
    # 代表轨道：全窗口 μ 最接近中位数的那条（避免挑好看的）
    per_seed_full = {}
    for seed in sorted({r["seed"] for r in series_rows if r["eps"] == eps_seg}):
        pts = [(r["t"], r["sep"]) for r in series_rows if r["eps"] == eps_seg and r["seed"] == seed]
        ts = np.array([t for t, _ in pts]); ss = np.array([s for _, s in pts])
        band = (ss > FIT_LO) & (ss < FIT_HI)
        per_seed_full[seed] = fit_segment(ts[band], ss[band])["mu_seg"]
    valid = {k: v for k, v in per_seed_full.items() if v is not None}
    rep_seed = min(valid, key=lambda k: abs(valid[k] - statistics.median(valid.values()))) if valid else args.seed_base

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    seg_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for seed in sorted({r["seed"] for r in series_rows if r["eps"] == eps_seg})[:6]:
        if seed == rep_seed:
            continue
        pts = sorted([(r["t"], r["sep"]) for r in series_rows if r["eps"] == eps_seg and r["seed"] == seed])
        ax.plot([t for t, _ in pts], [s for _, s in pts], color="gray", alpha=0.25, linewidth=0.9)
    rep_pts = sorted([(r["t"], r["sep"]) for r in series_rows if r["eps"] == eps_seg and r["seed"] == rep_seed])
    ax.plot([t for t, _ in rep_pts], [s for _, s in rep_pts], color="black", linewidth=1.6,
            label=f"代表轨道 seed={rep_seed}（ε={eps_seg:g}）")
    ax.axhspan(FIT_LO, FIT_HI, color="gray", alpha=0.12, label="拟合标度区")
    for i, lab in enumerate(SEG_LABELS):
        row = next((r for r in seg_rows if r["eps"] == eps_seg and r["seed"] == rep_seed and r["seg"] == lab), None)
        if not row or row["mu_seg"] is None:
            continue
        t_lo, t_hi = row["t_lo"], row["t_hi"]
        tt = np.linspace(t_lo, t_hi, 50)
        yy = np.exp(row["intercept"] + row["mu_seg"] * tt)
        ax.plot(tt, yy, color=seg_colors[i], linewidth=2.4,
                label=f"{lab} μ={row['mu_seg']:.3f} R²={row['r2']:.2f} n={row['n_points']}")
    ax.set_yscale("log")
    ax.set_ylim(FIT_LO, FIT_HI * 1.4)
    ax.set_xlabel("t（时间单位；一个 figure-8 周期 ≈ 6.33）")
    ax.set_ylabel("sep(t)（对数轴）")
    ax.set_title(f"分段拟合：代表轨道 sep(t) + 四段独立回归（ε={eps_seg:g}，{args.max_periods:g} 周期探索性预算）")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out_dir / "seg_fit_main.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7.6, 4.8))
    for i, lab in enumerate(SEG_LABELS):
        mus = [r["mu_seg"] for r in seg_rows if r["eps"] == eps_seg and r["seg"] == lab and r["mu_seg"] is not None]
        if not mus:
            continue
        ax2.scatter([i] * len(mus), mus, s=22, alpha=0.5, color=seg_colors[i])
        ax2.scatter([i], [statistics.median(mus)], s=170, marker="_", linewidths=3, color="black")
    med = [statistics.median([r["mu_seg"] for r in seg_rows if r["eps"] == eps_seg and r["seg"] == lab
                              and r["mu_seg"] is not None])
           if any(r["mu_seg"] is not None for r in seg_rows if r["eps"] == eps_seg and r["seg"] == lab)
           else None for lab in SEG_LABELS]
    xs = [i for i, m in enumerate(med) if m is not None]
    ax2.plot(xs, [med[i] for i in xs], color="black", alpha=0.5, linewidth=1.4, zorder=0)
    ax2.set_xticks(range(len(SEG_LABELS)))
    ax2.set_xticklabels([f"{lab}\n({int(edges[i])}-{int(edges[i+1])})" for i, lab in enumerate(SEG_LABELS)])
    ax2.set_xlabel("时间分段（全局时间四等分）")
    ax2.set_ylabel("μ_seg（该段 ln(sep) 斜率）")
    ax2.set_title(f"速率是否在衰减？（ε={eps_seg:g}，{args.seeds_per_eps} 条轨道；黑线=各段中位数）")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout(); fig2.savefig(out_dir / "mu_seg_decay.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(8.2, 5.0))
    for eps, color in ((args.eps_seg, "#1f77b4"), (args.eps_check, "#d62728")):
        for seed in (args.seed_base, args.seed_base + 1):
            pts = sorted([(r["t"], r["sep"]) for r in series_rows if r["eps"] == eps and r["seed"] == seed])
            if pts:
                ax3.plot([t for t, _ in pts], [s / eps for _, s in pts], color=color,
                         alpha=0.85, linewidth=1.2, label=f"ε={eps:g}（shown / ε）")
    ax3.set_yscale("log")
    ax3.set_xlabel("t（时间单位）")
    ax3.set_ylabel("sep(t) / ε（对数轴）")
    ax3.set_title("f(t) 不变性再验（20 周期）：不同 ε 的 sep/ε 曲线应重合 → sep(t)=ε·f(t)")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(fontsize=8)
    fig3.tight_layout(); fig3.savefig(out_dir / "ft_invariance.png", dpi=150)

    print(f"\n产物：\n  {out_dir / 'sep_series_v3.csv'}\n  {out_dir / 'seg_fit_table.csv'}\n  "
          f"{out_dir / 'growth_factors.csv'}\n  {out_dir / 'summary.json'}\n  "
          f"{out_dir / 'seg_fit_main.png'}\n  {out_dir / 'mu_seg_decay.png'}\n  {out_dir / 'ft_invariance.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
