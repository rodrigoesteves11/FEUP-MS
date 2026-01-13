"""
run_mixes.py

Experiência: Policy × Mix (composição de agentes) × Seeds
- Mantém N=300 constante
- Corre vários mixes (balanced / F_high / C_high / N_high)
- Calcula KPIs globais por run (com burn-in)
- Gera:
  1) CSV com resultados por run (kpi_results_by_mix.csv)
  2) Tabela robusta (mediana) em Policy|Mix × KPIs (matrix_policy_mix_median_raw.csv)
  3) Heatmap simples e legível: Δ vs balanced (mediana), em percentagem
     (heatmap_policy_mix_delta_vs_balanced.png)

Notas:
- Para evitar resultados “estranhos” por outliers/trajectórias explosivas:
  - usa MEDIANA em vez de média
  - converte drawdown para magnitude positiva (drawdown_mag = -max_drawdown)
  - (opcional) usa log10(bubble_peak) em vez do bubble_peak bruto
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import MarketModel, POLICY_PRESETS


CONFIG = {

    "initial_price": 20.0,
    "initial_dividend": 1.0,
    "Q": 300.0,
    "r": 0.05,


    "d_bar": 1.0,
    "rho": 0.95,
    "sigma_d": 0.15,


    "initial_wealth": 1000.0,
    "gamma_range": (0.5, 1.5),
    "sigma2_range": (1.0, 6.0),


    "kappa_f_range": (0.05, 0.20),
    "kappa_c_range": (0.5, 2.0),
    "chartist_L_choices": (5, 20, 60),
    "sigma_n_range": (0.01, 0.05),


    "beta": 1.0,
    "phi": 0.2,


    "tatonnement_K": 50,
    "tatonnement_eta": 0.2,
}

EXPERIMENT = {
    "steps": 50,
    "burn_in": 5,
    "seeds": list(range(1, 31)),
    "policies": ["none", "moderate", "excessive"],


    "agent_mixes": [
        {"mix": "balanced", "n_fundamentalists": 100, "n_chartists": 100, "n_noise": 100},
        {"mix": "F_high",   "n_fundamentalists": 150, "n_chartists": 75,  "n_noise": 75},
        {"mix": "C_high",   "n_fundamentalists": 75,  "n_chartists": 150, "n_noise": 75},
        {"mix": "N_high",   "n_fundamentalists": 75,  "n_chartists": 75,  "n_noise": 150},
    ],

    "out_dir": "results_mixes",
}

EVENTS = {
    "vol_window": 20,
    "crash_k": 2.0,
    "drawdown_crash": -0.25,
    "bubble_threshold": 1.5,
    "bubble_min_len": 5,
}


def rolling_std(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=max(2, window // 2)).std()


def count_runs(bool_series: pd.Series, min_len: int = 1) -> Tuple[int, float]:
    min_len = max(1, int(min_len))
    in_run = False
    lengths: List[int] = []
    current = 0

    for v in bool_series.fillna(False).astype(bool).values:
        if v:
            if not in_run:
                in_run = True
                current = 1
            else:
                current += 1
        else:
            if in_run:
                if current >= min_len:
                    lengths.append(current)
                in_run = False
                current = 0

    if in_run and current >= min_len:
        lengths.append(current)

    if len(lengths) == 0:
        return 0, 0.0
    return len(lengths), float(np.mean(lengths))


def compute_kpis(model_df: pd.DataFrame, burn_in: int) -> Dict[str, float]:
    df = model_df.copy()
    df = df.iloc[burn_in:].reset_index(drop=True)

    price = pd.to_numeric(df["Price"], errors="coerce")
    fundamental = pd.to_numeric(df["FundamentalPrice"], errors="coerce")
    mispricing = pd.to_numeric(df["Mispricing"], errors="coerce")
    bubble_ratio = pd.to_numeric(df["BubbleRatio"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    turnover = pd.to_numeric(df["Turnover"], errors="coerce")
    gini = pd.to_numeric(df["GiniWealthDisc"], errors="coerce")
    drawdown = pd.to_numeric(df["Drawdown"], errors="coerce")

    eps = 1e-12
    logret = np.log(np.maximum(price, eps) / np.maximum(price.shift(1), eps))
    vol = rolling_std(logret, EVENTS["vol_window"])

    crash_by_ret = (logret < (-EVENTS["crash_k"] * vol)) & vol.notna()
    crash_steps_ret = int(crash_by_ret.sum())
    crash_eps_ret, crash_dur_mean_ret = count_runs(crash_by_ret, min_len=1)

    crash_by_dd = (drawdown < EVENTS["drawdown_crash"])
    crash_steps_dd = int(crash_by_dd.sum())
    crash_eps_dd, crash_dur_mean_dd = count_runs(crash_by_dd, min_len=1)

    bubble_flag = (bubble_ratio > EVENTS["bubble_threshold"])
    n_bubbles, bubble_dur_mean = count_runs(bubble_flag, min_len=EVENTS["bubble_min_len"])

    rel_mispricing = (np.maximum(price, eps) / np.maximum(fundamental, eps)) - 1.0

    return {
        "vol_mean": float(vol.mean(skipna=True)),
        "vol_max": float(vol.max(skipna=True)),
        "mean_abs_mispricing": float(np.nanmean(np.abs(mispricing))),
        "mean_abs_rel_mispricing": float(np.nanmean(np.abs(rel_mispricing))),
        "volume_mean": float(np.nanmean(volume)),
        "turnover_mean": float(np.nanmean(turnover)),
        "gini_final": float(gini.iloc[-1]) if len(gini) else float("nan"),
        "max_drawdown": float(np.nanmin(drawdown)),
        "crash_steps_ret": float(crash_steps_ret),
        "crash_episodes_ret": float(crash_eps_ret),
        "crash_dur_mean_ret": float(crash_dur_mean_ret),
        "crash_steps_dd": float(crash_steps_dd),
        "crash_episodes_dd": float(crash_eps_dd),
        "crash_dur_mean_dd": float(crash_dur_mean_dd),
        "n_bubbles": float(n_bubbles),
        "bubble_peak": float(np.nanmax(bubble_ratio)),
        "bubble_dur_mean": float(bubble_dur_mean),
    }


def run_one(policy: str, seed: int, steps: int, mix_cfg: Dict) -> pd.DataFrame:
    cfg = CONFIG.copy()
    cfg["n_fundamentalists"] = int(mix_cfg["n_fundamentalists"])
    cfg["n_chartists"] = int(mix_cfg["n_chartists"])
    cfg["n_noise"] = int(mix_cfg["n_noise"])

    N = cfg["n_fundamentalists"] + cfg["n_chartists"] + cfg["n_noise"]
    if N != 300:
        raise ValueError(f"Mix '{mix_cfg['mix']}' não mantém N=300 (N={N}).")

    model = MarketModel(**cfg, policy_name=policy, seed=seed, max_steps=steps)
    model_df = model.run(steps)
    return model_df


def make_policy_mix_matrix_median(results_df: pd.DataFrame, kpis: List[str],
                                  policies_order: List[str], mixes_order: List[str]) -> pd.DataFrame:
    """
    Matriz (policy,mix) x KPIs usando MEDIANA (robusta a outliers).
    """
    mat = results_df.groupby(["policy", "mix"])[kpis].median()
    mat = mat.reindex(pd.MultiIndex.from_product([policies_order, mixes_order], names=["policy", "mix"]))
    return mat


def plot_delta_heatmap(delta_df: pd.DataFrame, title: str, out_path: str):
    """
    Heatmap legível: cores = Δ vs balanced (mediana), anotação = percentagem.
    Cores clipped para ±100% (só para visualização).
    """
    row_labels = [f"{p} | {m}" for p, m in delta_df.index.to_list()]
    clipped = delta_df.clip(lower=-1.0, upper=1.0)

    fig, ax = plt.subplots(
        figsize=(1.25 * len(delta_df.columns) + 6, 0.45 * len(delta_df.index) + 2),
        constrained_layout=True,
    )
    im = ax.imshow(clipped.values, aspect="auto", vmin=-1.0, vmax=1.0)

    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(delta_df.shape[1]))
    ax.set_xticklabels(delta_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(delta_df.shape[0]))
    ax.set_yticklabels(row_labels)

    for i in range(delta_df.shape[0]):
        for j in range(delta_df.shape[1]):
            v = delta_df.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.0%}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Δ vs balanced (mediana), cores clipped ±100%")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    out_dir = EXPERIMENT["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    steps = EXPERIMENT["steps"]
    burn_in = EXPERIMENT["burn_in"]
    seeds = EXPERIMENT["seeds"]
    policies = EXPERIMENT["policies"]
    mixes = EXPERIMENT["agent_mixes"]
    mixes_order = [m["mix"] for m in mixes]

    print("=== Experiment (Policy × Mix) ===")
    print(f"steps={steps} burn_in={burn_in} seeds={len(seeds)} policies={policies} mixes={mixes_order}")
    print("Policy presets:")
    for p in policies:
        print(f"  {p}: {POLICY_PRESETS[p]}")
    print()


    rows = []
    for policy in policies:
        for mix_cfg in mixes:
            mix_name = mix_cfg["mix"]
            for seed in seeds:
                print(f"Running policy={policy} mix={mix_name} seed={seed} ...")
                model_df = run_one(policy, seed, steps, mix_cfg)

                kpis = compute_kpis(model_df, burn_in=burn_in)
                kpis.update({
                    "policy": policy,
                    "mix": mix_name,
                    "seed": seed,
                    "n_fundamentalists": mix_cfg["n_fundamentalists"],
                    "n_chartists": mix_cfg["n_chartists"],
                    "n_noise": mix_cfg["n_noise"],
                })
                rows.append(kpis)

    results_df = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "kpi_results_by_mix.csv")
    results_df.to_csv(csv_path, index=False)
    print("\nSaved:", csv_path)

    df = results_df.copy()
    df["drawdown_mag"] = -pd.to_numeric(df["max_drawdown"], errors="coerce")

    bp = pd.to_numeric(df["bubble_peak"], errors="coerce").clip(lower=1e-12)
    df["log_bubble_peak"] = np.log10(bp)


    KPI_SIMPLE = ["vol_mean", "drawdown_mag", "volume_mean", "log_bubble_peak"]


    mat_median = make_policy_mix_matrix_median(df, KPI_SIMPLE, policies, mixes_order)
    median_csv = os.path.join(out_dir, "matrix_policy_mix_median_raw.csv")
    mat_median.to_csv(median_csv)
    print("Saved:", median_csv)
    print("\nMatrix (median raw):")
    print(mat_median)


    def robust_norm_by_column(x: pd.DataFrame, qlo: float = 0.10, qhi: float = 0.90) -> pd.DataFrame:
        out = x.copy()
        for c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce")
            lo = float(s.quantile(qlo))
            hi = float(s.quantile(qhi))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.nanmin(s.values))
                hi = float(np.nanmax(s.values))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                out[c] = 0.0
            else:
                out[c] = ((s - lo) / (hi - lo)).clip(0.0, 1.0)
        return out

    def plot_median_heatmap(mat_norm: pd.DataFrame, mat_raw: pd.DataFrame, title: str, out_path: str):
        row_labels = [f"{p} | {m}" for p, m in mat_norm.index.to_list()]

        fig, ax = plt.subplots(
            figsize=(1.15 * len(mat_norm.columns) + 6, 0.45 * len(mat_norm.index) + 2),
            constrained_layout=True,
        )

        im = ax.imshow(mat_norm.values, aspect="auto", vmin=0.0, vmax=1.0)

        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(mat_norm.shape[1]))
        ax.set_xticklabels(mat_norm.columns, rotation=30, ha="right")
        ax.set_yticks(range(mat_norm.shape[0]))
        ax.set_yticklabels(row_labels)



        n_mix = len(mixes_order)
        for y in [n_mix - 0.5, 2 * n_mix - 0.5]:
            ax.axhline(y, color="black", linewidth=1)


        fmt_map = {
            "vol_mean": "{:.3f}",
            "drawdown_mag": "{:.2f}",
            "crash_episodes_ret": "{:.2f}",
            "n_bubbles": "{:.2f}",
            "volume_mean": "{:.0f}",
            "log_bubble_peak": "{:.2f}",
        }

        for i in range(mat_raw.shape[0]):
            for j in range(mat_raw.shape[1]):
                v = mat_raw.values[i, j]
                if np.isfinite(v):
                    col = mat_raw.columns[j]
                    fmt = fmt_map.get(col, "{:.2g}")
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("normalizado por KPI (quantis 10–90%)")

        fig.savefig(out_path, dpi=220)
        plt.close(fig)

    mat_norm = robust_norm_by_column(mat_median, qlo=0.10, qhi=0.90)
    out_png = os.path.join(out_dir, "heatmap_policy_mix_median_kpis.png")
    plot_median_heatmap(
        mat_norm,
        mat_median,
        title="Policy × Mix — Medians (colors normalized by KPI; numbers = medians)",
        out_path=out_png,
    )
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
