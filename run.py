"""
run.py

Runner de experiências:
- corre vários cenários (none/moderate/excessive)
- corre várias seeds por cenário
- calcula KPIs por run (com burn-in)
- gera gráficos por run (8 painéis)
- gera gráficos comparativos por cenário (boxplots de KPIs)

Dependências: mesa, pandas, matplotlib
"""

from __future__ import annotations

import os
import math
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import MarketModel, POLICY_PRESETS


# ----------------------------
# Configuração central
# ----------------------------

CONFIG = {
    # agentes
    "n_fundamentalists": 100,
    "n_chartists": 100,
    "n_noise": 100,

    # mercado
    "initial_price": 20.0,
    "initial_dividend": 1.0,
    "Q": 300.0,
    "r": 0.05,

    # dividendos
    "d_bar": 1.0,
    "rho": 0.95,
    "sigma_d": 0.15,

    # riqueza/risco
    "initial_wealth": 1000.0,
    "gamma_range": (0.5, 1.5),
    "sigma2_range": (1.0, 6.0),

    # comportamentos
    "kappa_f_range": (0.05, 0.20),
    "kappa_c_range": (0.5, 2.0),
    "chartist_L_choices": (5, 20, 60),
    "sigma_n_range": (0.01, 0.05),

    # trading
    "beta": 1.0,
    "phi": 0.2,

    # tatonnement
    "tatonnement_K": 50,
    "tatonnement_eta": 0.2,
}

EXPERIMENT = {
    "steps": 50,
    "burn_in": 5,
    "seeds": list(range(1, 31)),  # 30 seeds
    "policies": ["none", "moderate", "excessive"],
    "out_dir": "results_new_model",
}

# thresholds para eventos
EVENTS = {
    "vol_window": 20,
    "crash_k": 2.0,               # crash: logret < -k * vol_rolling
    "drawdown_crash": -0.25,      # crash alternativo por dd
    "bubble_threshold": 1.5,      # bolha: ratio > 1.5
    "bubble_min_len": 10,         # por >= 10 passos
}


# ----------------------------
# KPIs
# ----------------------------

def rolling_std(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=max(2, window // 2)).std()


def count_runs(bool_series: pd.Series) -> Tuple[int, float]:
    """
    Conta episódios consecutivos True.
    Retorna: (n_episodios, duracao_media)
    """
    in_run = False
    lengths = []
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
                lengths.append(current)
                in_run = False
                current = 0
    if in_run:
        lengths.append(current)

    if len(lengths) == 0:
        return 0, 0.0
    return len(lengths), float(np.mean(lengths))


def compute_kpis(model_df: pd.DataFrame, burn_in: int) -> Dict[str, float]:
    df = model_df.copy()
    df = df.iloc[burn_in:].reset_index(drop=True)

    # base series
    price = df["Price"]
    fundamental = df["FundamentalPrice"]
    mispricing = df["Mispricing"]
    bubble_ratio = df["BubbleRatio"]
    volume = df["Volume"]
    turnover = df["Turnover"]
    gini = df["GiniWealthDisc"]  # recomendado (remove efeito mecânico do bond)
    drawdown = df["Drawdown"]

    # returns e vol rolling
    logret = np.log(price / price.shift(1))
    vol = rolling_std(logret, EVENTS["vol_window"])

    # crashes
    crash_by_ret = (logret < (-EVENTS["crash_k"] * vol)) & vol.notna()
    n_crashes_ret = int(crash_by_ret.sum())

    crash_by_dd = (drawdown < EVENTS["drawdown_crash"])
    n_crashes_dd = int(crash_by_dd.sum())

    # bolhas
    bubble_flag = (bubble_ratio > EVENTS["bubble_threshold"])
    n_bubbles, bubble_dur_mean = count_runs(bubble_flag)

    out = {
        "vol_mean": float(vol.mean(skipna=True)),
        "vol_max": float(vol.max(skipna=True)),
        "mean_abs_mispricing": float(np.mean(np.abs(mispricing))),
        "mean_abs_rel_mispricing": float(np.mean(np.abs((price / fundamental) - 1.0))),
        "volume_mean": float(volume.mean()),
        "turnover_mean": float(turnover.mean()),
        "gini_mean": float(gini.mean()),
        "gini_final": float(gini.iloc[-1]) if len(gini) else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "n_crashes_ret": float(n_crashes_ret),
        "n_crashes_dd": float(n_crashes_dd),
        "n_bubbles": float(n_bubbles),
        "bubble_peak": float(bubble_ratio.max()),
        "bubble_dur_mean": float(bubble_dur_mean),
    }
    return out


# ----------------------------
# Gráficos (por run)
# ----------------------------

def plot_single_run(model_df: pd.DataFrame, policy: str, seed: int, out_path: str, burn_in: int = 0):
    df = model_df.copy()
    if burn_in > 0:
        df = df.iloc[burn_in:].reset_index(drop=True)

    price = df["Price"]
    fundamental = df["FundamentalPrice"]
    mispricing = df["Mispricing"]
    bubble_ratio = df["BubbleRatio"]
    volume = df["Volume"]
    gini = df["GiniWealthDisc"]
    drawdown = df["Drawdown"]

    logret = np.log(price / price.shift(1))
    vol = rolling_std(logret, EVENTS["vol_window"])
    crash_flag = (logret < (-EVENTS["crash_k"] * vol)) & vol.notna()

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f"Policy={policy} | Seed={seed} | burn_in={burn_in}", fontsize=14)

    # 1) Price vs fundamental
    ax = axes[0, 0]
    ax.plot(price, label="Price")
    ax.plot(fundamental, linestyle="--", label="Fundamental")
    ax.set_title("Price vs Fundamental")
    ax.set_xlabel("t")
    ax.grid(alpha=0.3)
    ax.legend()

    # 2) Bubble ratio
    ax = axes[0, 1]
    ax.plot(bubble_ratio, label="P/P*")
    ax.axhline(1.0, linewidth=1)
    ax.axhline(EVENTS["bubble_threshold"], linestyle="--", linewidth=1, label="Bubble threshold")
    ax.set_title("Bubble Ratio")
    ax.set_xlabel("t")
    ax.grid(alpha=0.3)
    ax.legend()

    # 3) Log returns
    ax = axes[1, 0]
    ax.plot(logret, linewidth=0.8)
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Log Returns")
    ax.set_xlabel("t")
    ax.grid(alpha=0.3)

    # 4) Rolling volatility
    ax = axes[1, 1]
    ax.plot(vol, linewidth=1.0)
    ax.set_title(f"Rolling Volatility (window={EVENTS['vol_window']})")
    ax.set_xlabel("t")
    ax.grid(alpha=0.3)

    # 5) Mispricing
    ax = axes[2, 0]
    ax.plot(mispricing, linewidth=1.0)
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Mispricing (P - P*)")
    ax.set_xlabel("t")
    ax.grid(alpha=0.3)

    # 6) Volume + rolling mean
    ax = axes[2, 1]
    ax.plot(volume, linewidth=0.8, alpha=0.7, label="Volume")
    ax.plot(volume.rolling(20, min_periods=10).mean(), linewidth=1.5, label="Vol (roll mean 20)")
    ax.set_title("Trading Volume")
    ax.set_xlabel("t")
    ax.grid(alpha=0.3)
    ax.legend()

    # 7) Gini (zoom se muito baixo)
    ax = axes[3, 0]
    ax.plot(gini, linewidth=1.0)
    gmax = float(gini.max()) if len(gini) else 0.0
    ax.set_ylim(0, max(0.01, gmax * 1.2))
    ax.set_title("Gini (discounted wealth)")
    ax.set_xlabel("t")
    ax.grid(alpha=0.3)

    # 8) Drawdown + crash markers
    ax = axes[3, 1]
    ax.plot(drawdown, linewidth=1.0, label="Drawdown")
    crash_idx = crash_flag[crash_flag == True].index
    if len(crash_idx) > 0:
        ax.scatter(crash_idx, drawdown.loc[crash_idx], s=30, marker="v", label="Crash (ret criterion)")
    ax.axhline(EVENTS["drawdown_crash"], linestyle="--", linewidth=1, label="DD crash threshold")
    ax.set_title("Drawdown + Crashes")
    ax.set_xlabel("t")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------
# Gráficos comparativos (por cenário)
# ----------------------------

def plot_kpi_boxplots(results_df: pd.DataFrame, out_path: str):
    policies = EXPERIMENT["policies"]

    kpis = [
        "vol_mean", "vol_max",
        "mean_abs_mispricing",
        "volume_mean",
        "gini_final",
        "max_drawdown",
        "n_crashes_ret",
        "n_bubbles",
        "bubble_peak",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    for ax, kpi in zip(axes, kpis):
        data = [results_df.loc[results_df["policy"] == p, kpi].dropna().values for p in policies]
        ax.boxplot(data, tick_labels=policies, showfliers=False)
        ax.set_title(kpi)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------
# Runner principal
# ----------------------------

def run_one(policy: str, seed: int, steps: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = MarketModel(
        **CONFIG,
        policy_name=policy,
        seed=seed,
    )
    model_df = model.run(steps)
    agent_df = model.datacollector.get_agent_vars_dataframe()
    return model_df, agent_df


def main():
    out_dir = EXPERIMENT["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    steps = EXPERIMENT["steps"]
    burn_in = EXPERIMENT["burn_in"]
    seeds = EXPERIMENT["seeds"]
    policies = EXPERIMENT["policies"]

    print("=== Experiment configuration ===")
    print(f"steps={steps} | burn_in={burn_in} | seeds={len(seeds)} | policies={policies}")
    print("Policy presets:")
    for p in policies:
        print(f"  {p}: {POLICY_PRESETS[p]}")
    print()

    rows = []
    # Guarda 1 run “representativa” por cenário (primeira seed)
    representative_seed = seeds[0] if seeds else 1

    for policy in policies:
        for seed in seeds:
            print(f"Running policy={policy} seed={seed} ...")
            model_df, agent_df = run_one(policy, seed, steps)

            kpis = compute_kpis(model_df, burn_in=burn_in)
            kpis.update({"policy": policy, "seed": seed})
            rows.append(kpis)

            # guardar gráficos só para seed representativa (para não gerar 90 imagens)
            if seed == representative_seed:
                fig_path = os.path.join(out_dir, f"run_policy={policy}_seed={seed}.png")
                plot_single_run(model_df, policy, seed, fig_path, burn_in=burn_in)

    results_df = pd.DataFrame(rows)
    results_csv = os.path.join(out_dir, "kpi_results.csv")
    results_df.to_csv(results_csv, index=False)

    # boxplots comparativos
    boxplot_path = os.path.join(out_dir, "kpi_boxplots_by_policy.png")
    plot_kpi_boxplots(results_df, boxplot_path)

    print("\n=== Done ===")
    print(f"Saved: {results_csv}")
    print(f"Saved: {boxplot_path}")
    print(f"Saved: run_policy=*_seed={representative_seed}.png (1 por cenário)")

    # print resumo por cenário
    print("\n=== KPI summary (mean ± std) ===")
    for policy in policies:
        sub = results_df[results_df["policy"] == policy]
        print(f"\nPolicy: {policy}")
        for col in ["vol_mean", "mean_abs_mispricing", "volume_mean", "gini_final", "max_drawdown", "n_crashes_ret", "n_bubbles", "bubble_peak"]:
            mu = sub[col].mean()
            sd = sub[col].std()
            print(f"  {col}: {mu:.4f} ± {sd:.4f}")


if __name__ == "__main__":
    main()
