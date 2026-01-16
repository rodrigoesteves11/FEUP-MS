"""
viz_step.py

Versão simplificada - guarda frames como imagens para ver a evolução do que faria o viz_simple.py em tempo real.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import time
import sys

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from model import MarketModel, POLICY_PRESETS


def run_with_visualization(policy="none", steps=50, save_every=10, seed=None):
    """
    Corre a simulação e guarda visualizações periódicas.
    """
    print("=" * 70)
    print(f"MARKET ABM VISUALIZATION - Policy: {policy.upper()}")
    print("=" * 70)
    print(f"Steps: {steps} | Saving every {save_every} steps | Seed: {seed}")
    print()
    

    out_dir = f"viz_output_{policy}"
    if seed is not None:
        out_dir = f"viz_output_{policy}_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    

    model = MarketModel(
        n_fundamentalists=100,
        n_chartists=100,
        n_noise=100,
        initial_price=20.0,
        initial_dividend=1.0,
        Q=300.0,
        r=0.05,
        d_bar=1.0,
        rho=0.95,
        sigma_d=0.15,
        initial_wealth=1000.0,
        gamma_range=(0.5, 1.5),
        sigma2_range=(1.0, 6.0),
        kappa_f_range=(0.05, 0.20),
        kappa_c_range=(0.5, 2.0),
        chartist_L_choices=(5, 20, 60),
        sigma_n_range=(0.01, 0.05),
        beta=1.0,
        phi=0.2,
        tatonnement_K=50,
        tatonnement_eta=0.2,
        policy_name=policy,
        seed=seed,
    )
    
    print(f"Model initialized with policy: {policy}")
    print(f"Policy settings: {POLICY_PRESETS[policy]}")
    print()
    

    for step in range(steps):
        model.step()
        
        if (step + 1) % save_every == 0 or step == 0:
            print(f"Step {step+1}/{steps} - Saving visualization...")
            save_visualization(model, out_dir, step + 1)
    
    print()
    print("=" * 70)
    print(f"DONE! Visualizations saved to: {out_dir}/")
    print("=" * 70)
    print()
    

    print("Creating final summary...")
    create_final_summary(model, out_dir)
    print(f"Final summary saved to: {out_dir}/final_summary.png")
    
    return model


def save_visualization(model, out_dir, step):
    """Salva uma visualização do estado atual."""
    df = model.datacollector.get_model_vars_dataframe()
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Market ABM - Policy: {model.policy_name.upper()} - Step {step}', 
                 fontsize=16, fontweight='bold')
    
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(df['Step'], df['Price'], label='Price', color='#0066cc', linewidth=2)
    ax.plot(df['Step'], df['FundamentalPrice'], label='Fundamental', 
           color='#006600', linestyle='--', linewidth=2)
    ax.set_title('Price vs Fundamental', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(df['Step'], df['BubbleRatio'], color='#ff6600', linewidth=2)
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_title('Bubble Ratio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('P/P*')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(df['Step'], df['Mispricing'], color='#cc0000', linewidth=1.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_title('Mispricing', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('P - P*')
    ax.grid(alpha=0.3)
    
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(df['Step'], df['Volume'], color='#9900cc', linewidth=1.5)
    ax.set_title('Trading Volume', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Volume')
    ax.grid(alpha=0.3)
    
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(df['Step'], df['GiniWealthDisc'], color='#cc6600', linewidth=1.5)
    ax.set_title('Gini (Wealth Inequality)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gini')
    ax.grid(alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(df['Step'], df['LogReturn'], color='#00cc66', linewidth=1.0, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_title('Log Returns', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Log Return')
    ax.grid(alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis('off')
    
    last = df.iloc[-1]
    policy_info = POLICY_PRESETS[model.policy_name]
    
    info_text = f"""
    CURRENT STATE (Step {step})
    
    Price: {last['Price']:.2f}  |  Fundamental: {last['FundamentalPrice']:.2f}  |  Dividend: {last['Dividend']:.3f}
    Mispricing: {last['Mispricing']:+.2f} ({(last['Mispricing']/last['FundamentalPrice']*100):+.1f}%)
    Bubble Ratio: {last['BubbleRatio']:.2f}  |  Volume: {last['Volume']:.0f}  |  Gini: {last['GiniWealthDisc']:.3f}
    
    POLICY: {model.policy_name.upper()}
    Tax: {policy_info.tau*100:.1f}%  |  Leverage: {policy_info.L_max:.1f}x  |  Short Ban: {policy_info.short_ban}
    Q_max: {policy_info.q_max}  |  C_min: {policy_info.C_min}
    """
    
    ax.text(0.05, 0.5, info_text, fontsize=11, verticalalignment='center',
           fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(f"{out_dir}/step_{step:04d}.png", dpi=100, bbox_inches='tight')
    plt.close(fig)


def create_final_summary(model, out_dir):
    """Cria um resumo final com todas as métricas."""
    df = model.datacollector.get_model_vars_dataframe()
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Market ABM - FINAL SUMMARY - Policy: {model.policy_name.upper()}', 
                 fontsize=18, fontweight='bold')
    
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    

    plots = [
        (gs[0, :2], 'Price', ['Price', 'FundamentalPrice'], ['#0066cc', '#006600'], ['Price', 'Fundamental']),
        (gs[0, 2], 'BubbleRatio', ['BubbleRatio'], ['#ff6600'], ['Bubble Ratio']),
        (gs[1, 0], 'Mispricing', ['Mispricing'], ['#cc0000'], ['Mispricing']),
        (gs[1, 1], 'Volume', ['Volume'], ['#9900cc'], ['Volume']),
        (gs[1, 2], 'Turnover', ['Turnover'], ['#cc00cc'], ['Turnover']),
        (gs[2, 0], 'LogReturn', ['LogReturn'], ['#00cc66'], ['Log Return']),
        (gs[2, 1], 'GiniWealthDisc', ['GiniWealthDisc'], ['#cc6600'], ['Gini']),
        (gs[2, 2], 'Drawdown', ['Drawdown'], ['#cc0066'], ['Drawdown']),
    ]
    
    for pos, title, cols, colors, labels in plots:
        ax = fig.add_subplot(pos)
        for col, color, label in zip(cols, colors, labels):
            ax.plot(df['Step'], df[col], label=label, color=color, linewidth=2)
        
        if title == 'BubbleRatio':
            ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
            ax.axhline(1.5, color='red', linestyle='--', alpha=0.5)
        elif title in ['Mispricing', 'LogReturn', 'Drawdown']:
            ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
        ax.grid(alpha=0.3)
        if len(labels) > 1:
            ax.legend()
    

    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')
    

    burn_in = 5
    df_analysis = df.iloc[burn_in:]
    
    stats = f"""
    FINAL STATISTICS (after burn-in of {burn_in} steps)
    
    Price:        Mean={df_analysis['Price'].mean():.2f}  |  Final={df.iloc[-1]['Price']:.2f}  |  Min={df_analysis['Price'].min():.2f}  |  Max={df_analysis['Price'].max():.2f}
    Mispricing:   Mean Abs={df_analysis['Mispricing'].abs().mean():.2f}  |  Final={df.iloc[-1]['Mispricing']:+.2f}
    Bubble Ratio: Mean={df_analysis['BubbleRatio'].mean():.2f}  |  Max={df_analysis['BubbleRatio'].max():.2f}  |  Final={df.iloc[-1]['BubbleRatio']:.2f}
    Volume:       Mean={df_analysis['Volume'].mean():.0f}  |  Total={df_analysis['Volume'].sum():.0f}
    Turnover:     Mean={df_analysis['Turnover'].mean():.2%}
    Gini:         Mean={df_analysis['GiniWealthDisc'].mean():.3f}  |  Final={df.iloc[-1]['GiniWealthDisc']:.3f}
    Drawdown:     Max={df_analysis['Drawdown'].min():.2%}
    
    Policy: {model.policy_name.upper()} - {POLICY_PRESETS[model.policy_name]}
    """
    
    ax.text(0.05, 0.5, stats, fontsize=11, verticalalignment='center',
           fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.savefig(f"{out_dir}/final_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    import sys
    

    if len(sys.argv) < 2:
        print("ERROR: Policy argument is required!")
        print(f"\nUsage: python3 viz_step.py <policy> [steps] [save_every] [seed]")
        print(f"\nArguments:")
        print(f"  policy: {list(POLICY_PRESETS.keys())} (REQUIRED)")
        print(f"  steps: integer (optional, default=50)")
        print(f"  save_every: integer (optional, default=10)")
        print(f"  seed: integer (optional, uses random if omitted)")
        print(f"\nExamples:")
        print(f"  python3 viz_step.py none                (policy=none, 50 steps, save every 10, random seed)")
        print(f"  python3 viz_step.py moderate 100        (policy=moderate, 100 steps, save every 10, random seed)")
        print(f"  python3 viz_step.py excessive 75 5      (policy=excessive, 75 steps, save every 5, random seed)")
        print(f"  python3 viz_step.py none 50 10 42       (policy=none, 50 steps, save every 10, seed 42)")
        sys.exit(1)
    
    policy = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    save_every = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    if policy not in POLICY_PRESETS:
        print(f"ERROR: Invalid policy '{policy}'")
        print(f"Available policies: {list(POLICY_PRESETS.keys())}")
        sys.exit(1)
    
    print("=" * 70)
    print("MARKET ABM - STEP-BY-STEP VISUALIZATION (PNG Snapshots)")
    print("=" * 70)
    seed_info = f"_seed{seed}" if seed is not None else "_random"
    print(f"Policy: {policy.upper()}{seed_info} | Steps: {steps} | Save every: {save_every}")
    print("=" * 70)
    

    model = run_with_visualization(policy=policy, steps=steps, save_every=save_every, seed=seed)
    
    print("\nYou can now:")
    print(f"  1. Open viz_output_{policy}{seed_info}/final_summary.png to see full results")
    print(f"  2. Browse viz_output_{policy}{seed_info}/step_*.png to see evolution")
