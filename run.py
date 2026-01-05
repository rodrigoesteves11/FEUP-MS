"""
Script principal para executar a simulação.
"""

import matplotlib.pyplot as plt
from model import MarketModel


# ===== CONFIGURAÇÃO CENTRAL =====
CONFIG = {
    # Agentes
    "n_fundamentalists": 10,
    "n_chartists": 10,
    "n_noise": 10,
    
    # Mercado
    "initial_price": 20.0,
    "r": 0.05,
    "d_bar": 1.0,
    "Q": 30.0,  # Total de ações (N_agents para ~1 share/agente)
    
    # Risco/Variância
    "perceived_variance": 2.5,
    
    # Ruído
    "sigma_n": 2.0,
    
    # Simulação
    "steps": 1000,
    "seed": 42,
}


def run_simulation(config=None):
    """Executa a simulação e retorna os dados."""
    if config is None:
        config = CONFIG
    
    # Separar steps do resto (não é parâmetro do model)
    steps = config.pop("steps", 1000)
    
    print("Model Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  Fundamental Price (d_bar/r): {config['d_bar']/config['r']:.2f}")
    print(f"  Steps: {steps}")
    print()
    
    model = MarketModel(**config)
    
    # Restaurar steps no config para referência
    config["steps"] = steps
    
    model_data = model.run(steps)
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    return model, model_data, agent_data


def plot_results(model, model_data, agent_data):
    """Visualiza os resultados da simulação com 7 gráficos ASM-like."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    
    # 1. Preço vs Preço Fundamental
    ax1 = axes[0, 0]
    ax1.plot(model_data["Price"], label="Market Price", color="blue", linewidth=1)
    ax1.plot(model_data["FundamentalPrice"], label="Fundamental Price", 
             color="red", linestyle="--", linewidth=1)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Price (€)")
    ax1.set_title("Price vs Fundamental Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Log Returns
    ax2 = axes[0, 1]
    log_returns = model_data["LogReturn"].dropna()
    ax2.plot(log_returns, color="darkgreen", linewidth=0.8, alpha=0.8)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.fill_between(log_returns.index, 0, log_returns.values, 
                     where=log_returns > 0, alpha=0.3, color="green")
    ax2.fill_between(log_returns.index, 0, log_returns.values, 
                     where=log_returns < 0, alpha=0.3, color="red")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Log Return")
    ax2.set_title("Log Returns: ln(P_t / P_{t-1})")
    ax2.grid(True, alpha=0.3)
    
    # 3. Volatilidade Rolling
    ax3 = axes[1, 0]
    volatility = model_data["VolatilityRolling"].dropna()
    ax3.plot(volatility, color="orange", linewidth=1)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Volatility (std)")
    ax3.set_title("Rolling Volatility (window=20)")
    ax3.grid(True, alpha=0.3)
    
    # 4. Mispricing
    ax4 = axes[1, 1]
    mispricing = model_data["Mispricing"]
    ax4.plot(mispricing, color="purple", linewidth=1)
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax4.fill_between(mispricing.index, 0, mispricing.values, 
                     where=mispricing > 0, alpha=0.3, color="red", label="Overvalued")
    ax4.fill_between(mispricing.index, 0, mispricing.values, 
                     where=mispricing < 0, alpha=0.3, color="blue", label="Undervalued")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("P - P*")
    ax4.set_title("Mispricing (Market - Fundamental)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Volume
    ax5 = axes[2, 0]
    volume = model_data["Volume"]
    ax5.bar(volume.index, volume.values, color="steelblue", alpha=0.7, width=1)
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Volume (Σ|Δq|)")
    ax5.set_title("Trading Volume")
    ax5.grid(True, alpha=0.3)
    
    # 6. Gini Coefficient (escala ajustada para valores baixos)
    ax6 = axes[2, 1]
    gini = model_data["GiniWealth"]
    ax6.plot(gini, color="brown", linewidth=1)
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Gini Coefficient")
    ax6.set_title("Wealth Inequality (Gini)")
    # Ajustar ylim dinamicamente (0.01 para dar zoom quando Gini é baixo)
    gini_max = max(0.01, gini.max() * 1.2)
    ax6.set_ylim(0, gini_max)
    ax6.grid(True, alpha=0.3)
    
    # 7. Drawdown e Crashes
    ax7 = axes[3, 0]
    drawdown = model_data["Drawdown"]
    is_crash = model_data["IsCrash"]
    ax7.fill_between(drawdown.index, 0, drawdown.values, 
                     color="red", alpha=0.5, label="Drawdown")
    ax7.plot(drawdown, color="darkred", linewidth=1)
    # Marcar crashes
    crash_steps = is_crash[is_crash == True].index
    if len(crash_steps) > 0:
        ax7.scatter(crash_steps, drawdown.loc[crash_steps], 
                    color="black", s=50, marker="v", label="Crash", zorder=5)
    ax7.set_xlabel("Step")
    ax7.set_ylabel("Drawdown")
    ax7.set_title("Drawdown from Peak (crashes marked)")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Riqueza por Tipo de Agente (descontada)
    ax8 = axes[3, 1]
    agent_data_reset = agent_data.reset_index()
    for agent_type in ["FundamentalistAgent", "ChartistAgent", "NoiseAgent"]:
        type_data = agent_data_reset[agent_data_reset["AgentType"] == agent_type]
        wealth_by_step = type_data.groupby("Step")["WealthDisc"].mean()
        ax8.plot(wealth_by_step, label=agent_type.replace("Agent", ""), linewidth=1)
    ax8.set_xlabel("Step")
    ax8.set_ylabel("Mean Discounted Wealth")
    ax8.set_title("Discounted Wealth by Agent Type")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("simulation_results.png", dpi=150)
    plt.show()


def print_summary_stats(model, model_data, agent_data):
    """Imprime estatísticas resumidas."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Preço
    print(f"\nPRICE DYNAMICS:")
    print(f"  Final Price: {model_data['Price'].iloc[-1]:.2f}")
    print(f"  Fundamental Price: {model_data['FundamentalPrice'].iloc[-1]:.2f}")
    print(f"  Mean Mispricing: {model_data['Mispricing'].mean():.4f}")
    print(f"  Mispricing Std: {model_data['Mispricing'].std():.4f}")
    
    # Volatilidade
    print(f"\nVOLATILITY:")
    vol = model_data["VolatilityRolling"].dropna()
    print(f"  Mean Volatility: {vol.mean():.6f}")
    print(f"  Max Volatility: {vol.max():.6f}")
    
    # Crashes
    n_crashes = model_data["IsCrash"].sum()
    print(f"\nEXTREME EVENTS:")
    print(f"  Number of Crashes (r < -2σ): {n_crashes}")
    print(f"  Max Drawdown: {model_data['Drawdown'].min():.4f}")
    
    # Riqueza (só descontada - TotalWealth explode com r>0)
    print(f"\nWEALTH (discounted):")
    print(f"  Discounted Total Wealth: {model_data['TotalWealthDisc'].iloc[-1]:.2f}")
    print(f"  Final Gini: {model_data['GiniWealth'].iloc[-1]:.4f}")
    print(f"  Initial Total Wealth: {model_data['TotalWealthDisc'].iloc[0]:.2f}")
    
    # Market clearing (usa Q do model)
    print(f"\nMARKET CLEARING:")
    print(f"  Total Shares: {model_data['TotalShares'].iloc[-1]:.2f} (Q={model.Q})")
    print(f"  Mean Shares Error: {model_data['SharesError'].mean():.8f}")
    
    # Volume
    print(f"\nVOLUME:")
    print(f"  Mean Volume: {model_data['Volume'].mean():.2f}")
    print(f"  Max Volume: {model_data['Volume'].max():.2f}")
    
    print("="*60)


if __name__ == "__main__":
    print("Running ASM-style simulation...")
    model, model_data, agent_data = run_simulation()
    
    print_summary_stats(model, model_data, agent_data)
    plot_results(model, model_data, agent_data)

