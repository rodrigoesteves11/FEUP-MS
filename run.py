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
    gini = df["GiniWealthDisc"]  # recommended (removes mechanical bond effect)
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
# Main runner
# ----------------------------

def run_one(policy: str, seed: int, steps: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = MarketModel(
        **CONFIG,
        policy_name=policy,
        seed=seed,
        max_steps=steps
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
    
    for policy in policies:
        for seed in seeds:
            print(f"Running policy={policy} seed={seed} ...")
            model_df, agent_df = run_one(policy, seed, steps)

            kpis = compute_kpis(model_df, burn_in=burn_in)
            kpis.update({"policy": policy, "seed": seed})
            rows.append(kpis)

    results_df = pd.DataFrame(rows)
    results_csv = os.path.join(out_dir, "kpi_results.csv")
    results_df.to_csv(results_csv, index=False)

    print("\n=== Done ===")
    print(f"Saved: {results_csv}")
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
