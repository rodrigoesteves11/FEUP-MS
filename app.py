from __future__ import annotations

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import solara

from mesa.visualization import SolaraViz
from mesa.visualization.utils import update_counter


import mesa.visualization.solara_viz as _mesa_solara_viz

_ORIG_MAKE_INITIAL_GRID_LAYOUT = _mesa_solara_viz.make_initial_grid_layout


def _make_initial_grid_layout_full_width(num_components: int):
    if num_components == 1:
        return [{"i": "0", "x": 0, "y": 0, "w": 12, "h": 12}]
    return _ORIG_MAKE_INITIAL_GRID_LAYOUT(num_components)


_mesa_solara_viz.make_initial_grid_layout = _make_initial_grid_layout_full_width

from model import MarketModel, POLICY_PRESETS

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

POLICY_COLORS = {"none": "#e74c3c", "moderate": "#f1c40f", "excessive": "#2ecc71"}

RESULTS_DIR = "results_new_model"
CSV_PATH = os.path.join(RESULTS_DIR, "kpi_results.csv")


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


def _safe_series(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return a
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def radar_factory(num_vars: int):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    theta += np.pi / 2

    def draw_poly(ax, values, color, label):
        values = np.asarray(values, dtype=float)
        values = np.clip(values, 0.0, 1.0)

        v = np.concatenate((values, [values[0]]))
        th = np.concatenate((theta, [theta[0]]))

        ax.plot(th, v, color=color, linewidth=2, marker="o", markersize=3, label=label, zorder=3)
        ax.fill(th, v, color=color, alpha=0.10, zorder=2)

    return theta, draw_poly


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _rolling_volatility_from_price(price: np.ndarray, window: int = 20) -> np.ndarray:
    price = _safe_series(price)
    if price.size < 2:
        return np.zeros_like(price, dtype=float)

    logret = np.zeros_like(price, dtype=float)
    logret[1:] = np.log(np.maximum(price[1:], 1e-12) / np.maximum(price[:-1], 1e-12))

    w = min(window, logret.size)
    vol = pd.Series(logret).rolling(window=w, min_periods=max(2, w // 2)).std().to_numpy()
    return _safe_series(vol)


def _robust_bounds(s: pd.Series, qlo: float = 0.10, qhi: float = 0.90) -> tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return 0.0, 1.0
    lo = float(s.quantile(qlo))
    hi = float(s.quantile(qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(s.min())
        hi = float(s.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0, 1.0
    return lo, hi


def _norm_with_bounds(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def _make_live_dashboard_figure(model) -> plt.Figure:
    df = model.datacollector.get_model_vars_dataframe()
    if df is None or len(df) == 0:
        fig = plt.figure(figsize=(16, 10))
        fig.text(0.5, 0.5, "Waiting... click Play to start", ha="center", va="center", fontsize=16)
        return fig

    t = _safe_series(df["Step"].to_numpy())
    price = _safe_series(df["Price"].to_numpy())
    fundamental = _safe_series(df["FundamentalPrice"].to_numpy())
    mispricing = _safe_series(df["Mispricing"].to_numpy())
    volume = _safe_series(df["Volume"].to_numpy())
    gini = _safe_series(df["GiniWealthDisc"].to_numpy())
    drawdown = _safe_series(df["Drawdown"].to_numpy())
    bubble_ratio = _safe_series(df["BubbleRatio"].to_numpy())

    last = df.iloc[-1]

    policy_name = getattr(model, "policy_name", "none")
    seed = getattr(model, "seed", None)

    fig = plt.figure(figsize=(16, 10))


    gs = GridSpec(
        3,
        4,
        figure=fig,
        hspace=0.40,
        wspace=0.30,
        left=0.06,
        right=0.72,
        top=0.92,
        bottom=0.08,
    )

    fig.suptitle(
        f"Live Monitor — Policy: {str(policy_name).upper()}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )


    ax_price = fig.add_subplot(gs[0, :])
    ax_price.plot(t, price, color="black", lw=1.5, label="Market Price")
    ax_price.plot(t, fundamental, color="blue", ls="--", alpha=0.6, label="Fundamental")

    ax_price.fill_between(
        t, price, fundamental,
        where=(price > fundamental),
        color="red", alpha=0.20, interpolate=True, label="Bubble (risk)"
    )
    ax_price.fill_between(
        t, price, fundamental,
        where=(price <= fundamental),
        color="green", alpha=0.20, interpolate=True, label="Discount"
    )

    ax_price.set_title("1. Price Dynamics: Speculative Gap", fontweight="bold")
    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.grid(True, alpha=0.2)


    ax_misp = fig.add_subplot(gs[1, :2])
    ax_misp.plot(t, mispricing, color="#c0392b", lw=1.5)
    ax_misp.axhline(0, color="black", lw=1)
    ax_misp.fill_between(t, mispricing, 0, color="#c0392b", alpha=0.10)
    ax_misp.set_title("3. Mispricing (Deviation from Fair Value)", fontweight="bold", size=10)
    ax_misp.grid(True, alpha=0.2)


    ax_vol = fig.add_subplot(gs[1, 2:])
    ax_vol.bar(t, volume, color="#8e44ad", alpha=0.60, width=1.0)
    ax_vol.set_title("4. Liquidity (Trading Volume)", fontweight="bold", size=10)
    ax_vol.grid(True, axis="y", alpha=0.2)


    ax_dd = fig.add_subplot(gs[2, :2])
    ax_dd.plot(t, drawdown, color="#d35400", lw=1.5)
    ax_dd.fill_between(t, drawdown, 0, color="#d35400", alpha=0.20)
    ax_dd.axhline(-0.25, ls=":", color="black", lw=1)
    ax_dd.set_title("5. Underwater Plot (Drawdown Depth)", fontweight="bold", size=10)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(True, alpha=0.2)


    ax_gini = fig.add_subplot(gs[2, 2:])
    ax_gini.plot(t, gini, color="#27ae60", lw=1.5)
    ax_gini.set_title("6. Inequality (Gini)", fontweight="bold", size=10)
    ax_gini.set_ylabel("Gini")
    ax_gini.grid(True, alpha=0.2)



    ax_radar = fig.add_axes([0.74, 0.72, 0.16, 0.16], projection="polar")

    theta, draw_poly = radar_factory(4)
    radar_labels = ["Vol", "Crash", "Illiq", "Gini"]



    vol_series = _rolling_volatility_from_price(price, window=20)
    volat_now = float(vol_series[-1]) if vol_series.size else 0.0
    vol_score = _clip01(volat_now / 0.10)


    worst_dd = float(np.nanmin(drawdown))
    crash_score = _clip01((-worst_dd) / 0.25)


    Q = float(getattr(model, "Q", 300.0))
    vol_now = float(last["Volume"])
    liquidity_score = _clip01(vol_now / (0.50 * Q))
    illiq_score = 1.0 - liquidity_score


    gini_now = float(last["GiniWealthDisc"])
    gini_score = _clip01((gini_now - 0.0) / (0.50 - 0.0))

    radar_vals = [vol_score, crash_score, illiq_score, gini_score]


    vals = np.asarray(radar_vals, dtype=float)
    vals = np.clip(vals, 0.0, 1.0)
    v = np.concatenate((vals, [vals[0]]))
    th = np.concatenate((theta, [theta[0]]))
    ax_radar.plot(th, v, color="#8e44ad", linewidth=2, marker="o", markersize=4, zorder=3)
    ax_radar.fill(th, v, color="#9b59b6", alpha=0.25, zorder=2)

    ax_radar.set_ylim(0.0, 1.0)
    ax_radar.set_xticks(theta)
    ax_radar.set_xticklabels(radar_labels, size=8, color="#2c3e50")
    ax_radar.set_yticks([])
    ax_radar.grid(True, alpha=0.4, color="#bdc3c7")
    ax_radar.set_title("Risk", fontweight="bold", size=9, y=1.02, color="#2c3e50")


    ax_info = fig.add_axes([0.76, 0.07, 0.24, 0.56])
    ax_info.axis("off")

    policy_info = POLICY_PRESETS.get(policy_name, POLICY_PRESETS["none"])
    misp_pct = (
        (float(last["Mispricing"]) / float(last["FundamentalPrice"]) * 100.0)
        if float(last["FundamentalPrice"]) != 0.0
        else 0.0
    )

    info_text = (
        f"CURRENT STATE (Step {int(last['Step'])})\n\n"
        f"Price: {float(last['Price']):.2f}\n"
        f"Fundamental: {float(last['FundamentalPrice']):.2f}\n"
        f"Dividend: {float(last['Dividend']):.3f}\n\n"
        f"Mispricing: {float(last['Mispricing']):+.2f} ({misp_pct:+.1f}%)\n"
        f"Bubble ratio: {float(last['BubbleRatio']):.2f}\n"
        f"Volume: {float(last['Volume']):.0f}\n"
        f"Gini: {float(last['GiniWealthDisc']):.3f}\n\n"
        f"POLICY\n"
        f"Tax: {policy_info.tau*100:.1f}%\n"
        f"Leverage: {policy_info.L_max:.2f}x\n"
        f"Short ban: {policy_info.short_ban}\n"
    )

    ax_info.text(
        0.0,
        1.0,
        info_text,
        fontsize=9,
        family="monospace",
        va="top",
        bbox=dict(facecolor="#f5deb3", alpha=0.45, boxstyle="round,pad=0.5", edgecolor="#888"),
    )

    return fig


@solara.component
def LiveDashboard(model):
    update_counter.get()
    fig = _make_live_dashboard_figure(model)
    
    def _export_live_png():
        os.makedirs("screenshots", exist_ok=True)
        seed = getattr(model, "seed", "unknown")
        policy = getattr(model, "policy_name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("screenshots", f"live_dashboard_{policy}_seed{seed}_{timestamp}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved PNG: {path}")
    
    def _export_live_pdf():
        os.makedirs("screenshots", exist_ok=True)
        seed = getattr(model, "seed", "unknown")
        policy = getattr(model, "policy_name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("screenshots", f"live_dashboard_{policy}_seed{seed}_{timestamp}.pdf")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved PDF: {path}")
    
    with solara.Column(style={"width": "100%"}):
        with solara.Row():
            solara.Button("📥 Export as PNG", on_click=_export_live_png, small=True)
            solara.Button("📥 Export as PDF", on_click=_export_live_pdf, small=True)
        solara.FigureMatplotlib(fig, format="png")



def _load_kpis(csv_path: str) -> pd.DataFrame | None:
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def _make_aggregate_figure(results_df: pd.DataFrame) -> plt.Figure:
    df = results_df.copy()
    policies = ["none", "moderate", "excessive"]

    needed = ["policy", "vol_mean", "max_drawdown", "volume_mean", "gini_final"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")

    fig = plt.figure(figsize=(16, 10))


    gs = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[1.0, 0.85],
        hspace=0.35,
        wspace=0.35,
        left=0.06,
        right=0.98,
        top=0.90,
        bottom=0.08,
    )

    fig.suptitle(
        "Aggregate Analysis (30 Seeds per Policy)",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )


    ax_radar = fig.add_subplot(gs[0, 1:3], projection="polar")
    theta, draw_poly = radar_factory(4)
    radar_labels = ["Volatility", "Crash risk", "Illiquidity", "Gini"]


    vol_lo, vol_hi = _robust_bounds(df["vol_mean"], 0.10, 0.90)
    crash_mag = -pd.to_numeric(df["max_drawdown"], errors="coerce")
    crash_lo, crash_hi = _robust_bounds(crash_mag, 0.10, 0.90)
    volm_lo, volm_hi = _robust_bounds(df["volume_mean"], 0.10, 0.90)
    g_lo, g_hi = _robust_bounds(df["gini_final"], 0.10, 0.90)


    means = df.groupby("policy")[["vol_mean", "max_drawdown", "volume_mean", "gini_final"]].mean()

    for pol in policies:
        mean_vol = float(means.loc[pol, "vol_mean"])
        mean_crash = float(-means.loc[pol, "max_drawdown"])
        mean_liq = float(means.loc[pol, "volume_mean"])
        mean_g = float(means.loc[pol, "gini_final"])

        vol_score = _norm_with_bounds(mean_vol, vol_lo, vol_hi)
        crash_score = _norm_with_bounds(mean_crash, crash_lo, crash_hi)

        liq_score = _norm_with_bounds(mean_liq, volm_lo, volm_hi)
        illiq_score = 1.0 - liq_score

        g_score = _norm_with_bounds(mean_g, g_lo, g_hi)

        vals = [vol_score, crash_score, illiq_score, g_score]
        draw_poly(ax_radar, vals, POLICY_COLORS[pol], pol.upper())

    ax_radar.set_ylim(0.0, 1.0)
    ax_radar.set_xticks(theta)
    ax_radar.set_xticklabels(radar_labels, size=11, fontweight="bold")
    ax_radar.set_yticks([])
    ax_radar.grid(True, alpha=0.2)


    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.30, 1.15), fontsize=9, frameon=True)


    def _styled_box(ax, series_list, title, ylabel=None):
        bp = ax.boxplot(series_list, patch_artist=True, labels=["None", "Mod", "Exc"], showfliers=False)
        for patch, color in zip(
            bp["boxes"],
            [POLICY_COLORS["none"], POLICY_COLORS["moderate"], POLICY_COLORS["excessive"]],
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.60)
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)


    ax_box1 = fig.add_subplot(gs[1, 0])
    _styled_box(
        ax_box1,
        [df[df["policy"] == p]["vol_mean"] for p in policies],
        "Volatility (dispersion)",
    )

    ax_box2 = fig.add_subplot(gs[1, 1])
    _styled_box(
        ax_box2,
        [df[df["policy"] == p]["max_drawdown"] for p in policies],
        "Crash severity",
        ylabel="Max drawdown",
    )

    ax_box3 = fig.add_subplot(gs[1, 2])
    _styled_box(
        ax_box3,
        [df[df["policy"] == p]["volume_mean"] for p in policies],
        "Mean liquidity (volume)",
    )

    ax_box4 = fig.add_subplot(gs[1, 3])
    _styled_box(
        ax_box4,
        [df[df["policy"] == p]["gini_final"] for p in policies],
        "Final inequality (Gini)",
    )


    fig.text(
        0.02,
        0.56,
        "INTERPRETATION:\n\n"
        "1) The radar summarizes mean trade-offs across policies.\n"
        "2) The boxplots show dispersion across seeds.\n\n"
        "Larger radar area indicates higher costs/risks on average.",
        fontsize=11,
        bbox=dict(facecolor="#f0f0f0", alpha=0.8, boxstyle="round"),
    )

    return fig


@solara.component
def AggregatePage(model):
    df = solara.use_memo(lambda: _load_kpis(CSV_PATH), [])
    with solara.Column(style={"width": "100%"}):
        if df is None:
            solara.Warning("CSV not found. Run `python run.py` first to generate results.", dense=True)
            return
        fig = solara.use_memo(lambda: _make_aggregate_figure(df), dependencies=[len(df)])
        
        def _export_aggregate_png():
            os.makedirs("screenshots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("screenshots", f"aggregate_results_{timestamp}.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved PNG: {path}")
        
        def _export_aggregate_pdf():
            os.makedirs("screenshots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("screenshots", f"aggregate_results_{timestamp}.pdf")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved PDF: {path}")
        
        with solara.Row():
            solara.Button("📥 Export as PNG", on_click=_export_aggregate_png, small=True)
            solara.Button("📥 Export as PDF", on_click=_export_aggregate_pdf, small=True)
        solara.FigureMatplotlib(fig, format="png")


model_params = {
    "policy_name": {
        "type": "Select",
        "label": "Policy",
        "value": "none",
        "values": ["none", "moderate", "excessive"],
    },
    "seed": {
        "type": "SliderInt",
        "label": "Seed",
        "value": 1,
        "min": 1,
        "max": 30,
        "step": 1,
    },
    **CONFIG,
}

components = [
    (LiveDashboard, 0),
    (AggregatePage, 1),
]

model = MarketModel(**CONFIG, policy_name="none", seed=1)

page = SolaraViz(
    model,
    components=components,
    model_params=model_params,
    name="Market ABM Dashboard",
    play_interval=150,
    render_interval=1,
)
