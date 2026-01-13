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

CONFIG = {

    "n_fundamentalists": 100,
    "n_chartists": 100,
    "n_noise": 100,


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
    "out_dir": "results_new_model",
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
    """
    Conta episódios consecutivos True com duração >= min_len.
    Retorna: (n_episodios, duracao_media)
    """
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

    out = {
        "vol_mean": float(vol.mean(skipna=True)),
        "vol_max": float(vol.max(skipna=True)),

        "mean_abs_mispricing": float(np.nanmean(np.abs(mispricing))),
        "mean_abs_rel_mispricing": float(np.nanmean(np.abs(rel_mispricing))),

        "volume_mean": float(np.nanmean(volume)),
        "turnover_mean": float(np.nanmean(turnover)),

        "gini_mean": float(np.nanmean(gini)),
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
    return out


def compute_agenttype_kpis(agent_df: pd.DataFrame, model_df: pd.DataFrame, burn_in: int) -> Dict[str, float]:
    """
    KPIs por tipo de agente (apenas versões "share" dentro de cada política):
      A1) volume_share_{Type}        = quota do volume (proxy impacto), soma abs(dShares) por tipo / total
      A2) abs_exposure_share_{Type}  = quota da exposição média, mean(|Shares|*Price) por tipo / soma dos tipos

    Ambas ficam em [0,1] e cada linha (política) é comparável.
    """
    out: Dict[str, float] = {}
    if agent_df is None or len(agent_df) == 0:
        return out

    adf = agent_df.reset_index()
    if "AgentType" not in adf.columns or "Shares" not in adf.columns:
        return out


    step_col = "Step" if "Step" in adf.columns else adf.columns[0]
    id_col = "AgentID" if "AgentID" in adf.columns else adf.columns[1]


    adf = adf[adf[step_col] >= burn_in].copy()
    if len(adf) == 0:
        return out


    mdf = model_df.copy()
    if "Step" not in mdf.columns:
        mdf = mdf.reset_index()
    price_by_step = mdf[["Step", "Price"]].copy()
    adf = adf.merge(price_by_step, left_on=step_col, right_on="Step", how="left")


    adf.sort_values([id_col, step_col], inplace=True)
    adf["dShares"] = adf.groupby(id_col)["Shares"].diff()
    adf["abs_dShares"] = adf["dShares"].abs()


    price_num = pd.to_numeric(adf["Price"], errors="coerce")
    adf["abs_exposure_eur"] = adf["Shares"].abs() * price_num


    vol_by_type = adf.groupby("AgentType")["abs_dShares"].sum(min_count=1)
    vol_total = float(vol_by_type.sum()) if pd.notna(vol_by_type.sum()) else 0.0


    exp_by_type = adf.groupby("AgentType")["abs_exposure_eur"].mean()
    exp_total = float(exp_by_type.sum()) if pd.notna(exp_by_type.sum()) else 0.0

    for t in sorted(set(vol_by_type.index).union(set(exp_by_type.index))):
        v = float(vol_by_type.get(t, 0.0)) if pd.notna(vol_by_type.get(t, 0.0)) else 0.0
        e = float(exp_by_type.get(t, 0.0)) if pd.notna(exp_by_type.get(t, 0.0)) else 0.0

        out[f"volume_share_{t}"] = (v / vol_total) if vol_total > 0 else 0.0
        out[f"abs_exposure_share_{t}"] = (e / exp_total) if exp_total > 0 else 0.0

    return out

def _short_agent_label(col_suffix: str) -> str:

    s = col_suffix.replace("Agent", "")
    s = s.replace("Fundamentalist", "Fund")
    s = s.replace("Chartist", "Chart")
    s = s.replace("Noise", "Noise")
    return s

def make_policy_agent_matrix(results_df: pd.DataFrame, prefix: str, policies_order=None) -> pd.DataFrame:
    """
    Constrói matriz Policy × AgentType (média por política) para colunas que começam com `prefix`.
    Ex: prefix="volume_share_"
    """
    if policies_order is None:
        policies_order = ["none", "moderate", "excessive"]

    cols = [c for c in results_df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"Não encontrei colunas com prefix='{prefix}' no results_df.")

    mat = results_df.groupby("policy")[cols].mean()
    mat = mat.reindex(policies_order)


    new_cols = {}
    for c in mat.columns:
        suffix = c[len(prefix):]
        new_cols[c] = _short_agent_label(suffix)
    mat = mat.rename(columns=new_cols)


    desired = ["Fund", "Chart", "Noise"]
    existing = [c for c in desired if c in mat.columns]
    remaining = [c for c in mat.columns if c not in existing]
    mat = mat[existing + remaining]

    return mat

def plot_two_heatmaps_side_by_side(
    mat_left: pd.DataFrame,
    mat_right: pd.DataFrame,
    title_left: str,
    title_right: str,
    suptitle: str,
    out_path: str,
    fmt: str = "{:.0%}",
):
    """
    Desenha 2 heatmaps lado a lado (1 PNG) com a mesma escala [0,1].
    Ideal para quotas/percentagens.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    mats = [mat_left, mat_right]
    titles = [title_left, title_right]

    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat.values, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns)
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(mat.index)


        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=11)


    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("share (mean across seeds)")

    fig.suptitle(suptitle, fontweight="bold", fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


    





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

            agent_kpis = compute_agenttype_kpis(agent_df, model_df, burn_in=burn_in)
            kpis.update(agent_kpis)

            kpis.update({"policy": policy, "seed": seed})
            rows.append(kpis)

    results_df = pd.DataFrame(rows)
    results_csv = os.path.join(out_dir, "kpi_results.csv")
    results_df.to_csv(results_csv, index=False)




    mat_vol = make_policy_agent_matrix(results_df, prefix="volume_share_", policies_order=policies)
    mat_exp = make_policy_agent_matrix(results_df, prefix="abs_exposure_share_", policies_order=policies)

    out_path = os.path.join(out_dir, "heatmap_agent_impact_and_exposure.png")
    plot_two_heatmaps_side_by_side(
        mat_left=mat_vol,
        mat_right=mat_exp,
        title_left="Volume share (impact proxy)",
        title_right="Exposure share (exposure quota)",
        suptitle="Policy × Agent type — shares (mean of 30 seeds)",
        out_path=out_path,
        fmt="{:.0%}",
    )

    print(f"Saved combined heatmap figure: {out_path}")
    print("\nMatrix — volume share")
    print(mat_vol)
    print("\nMatrix — exposure share")
    print(mat_exp)
  
    
    print("\n=== Done ===")
    print(f"Saved: {results_csv}")

    print("\n=== KPI summary (mean ± std) ===")
    for policy in policies:
        sub = results_df[results_df["policy"] == policy]
        print(f"\nPolicy: {policy}")
        for col in [
            "vol_mean",
            "mean_abs_mispricing",
            "volume_mean",
            "gini_final",
            "max_drawdown",
            "crash_episodes_ret",
            "n_bubbles",
            "bubble_peak",
            "crash_steps_ret",
            "crash_episodes_ret",
            "bubble_dur_mean"
        ]:
            mu = sub[col].mean()
            sd = sub[col].std()
            print(f"  {col}: {mu:.4f} ± {sd:.4f}")



if __name__ == "__main__":
    main()
