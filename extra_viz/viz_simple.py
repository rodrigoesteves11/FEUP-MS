"""
viz_simple.py

Animação em tempo real do modelo de mercado com visualizações dinâmicas
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import numpy as np

from model import MarketModel, POLICY_PRESETS


class LiveMarketViz:
    def __init__(self, policy="none", steps=50, seed=None):
        self.policy = policy
        self.total_steps = steps
        self.seed = seed
        

        self.model = MarketModel(
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
        

        self.history = {
            'step': [],
            'price': [],
            'fundamental': [],
            'mispricing': [],
            'bubble_ratio': [],
            'volume': [],
            'gini': [],
            'log_return': [],
        }
        

        self.fig = plt.figure(figsize=(16, 10))
        seed_text = f" | Seed: {seed}" if seed is not None else ""
        self.fig.suptitle(f'Market ABM - Policy: {policy.upper()}{seed_text}', fontsize=16, fontweight='bold')
        
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        self.ax_price = self.fig.add_subplot(gs[0, :2])
        self.ax_bubble = self.fig.add_subplot(gs[0, 2])
        self.ax_mispricing = self.fig.add_subplot(gs[1, 0])
        self.ax_volume = self.fig.add_subplot(gs[1, 1])
        self.ax_gini = self.fig.add_subplot(gs[1, 2])
        self.ax_returns = self.fig.add_subplot(gs[2, 0])
        self.ax_info = self.fig.add_subplot(gs[2, 1:])
        self.ax_info.axis('off')
        

        self.ax_price.set_title('Price vs Fundamental')
        self.ax_price.set_xlabel('Step')
        self.ax_price.set_ylabel('Price')
        self.ax_price.grid(alpha=0.3)
        
        self.ax_bubble.set_title('Bubble Ratio')
        self.ax_bubble.set_xlabel('Step')
        self.ax_bubble.set_ylabel('P/P*')
        self.ax_bubble.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        self.ax_bubble.axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Bubble threshold')
        self.ax_bubble.grid(alpha=0.3)
        self.ax_bubble.legend(fontsize=8)
        
        self.ax_mispricing.set_title('Mispricing')
        self.ax_mispricing.set_xlabel('Step')
        self.ax_mispricing.set_ylabel('P - P*')
        self.ax_mispricing.axhline(0, color='black', linestyle='-', alpha=0.5)
        self.ax_mispricing.grid(alpha=0.3)
        
        self.ax_volume.set_title('Trading Volume')
        self.ax_volume.set_xlabel('Step')
        self.ax_volume.set_ylabel('Volume')
        self.ax_volume.grid(alpha=0.3)
        
        self.ax_gini.set_title('Gini (Wealth Inequality)')
        self.ax_gini.set_xlabel('Step')
        self.ax_gini.set_ylabel('Gini')
        self.ax_gini.grid(alpha=0.3)
        
        self.ax_returns.set_title('Log Returns')
        self.ax_returns.set_xlabel('Step')
        self.ax_returns.set_ylabel('Log Return')
        self.ax_returns.axhline(0, color='black', linestyle='-', alpha=0.5)
        self.ax_returns.grid(alpha=0.3)
        
    def update(self, frame):

        self.model.step()
        

        self.update_display()
        
        return []
    
    def update_display(self):
        """Atualiza os gráficos com o estado atual do modelo"""
        df = self.model.datacollector.get_model_vars_dataframe()
        last = df.iloc[-1]
        
        self.history['step'].append(last['Step'])
        self.history['price'].append(last['Price'])
        self.history['fundamental'].append(last['FundamentalPrice'])
        self.history['mispricing'].append(last['Mispricing'])
        self.history['bubble_ratio'].append(last['BubbleRatio'])
        self.history['volume'].append(last['Volume'])
        self.history['gini'].append(last['GiniWealthDisc'])
        self.history['log_return'].append(last['LogReturn'])
        
        steps = self.history['step']
        
        self.ax_price.clear()
        self.ax_price.plot(steps, self.history['price'], label='Price', color='#0066cc', linewidth=2)
        self.ax_price.plot(steps, self.history['fundamental'], label='Fundamental', 
                          color='#006600', linestyle='--', linewidth=2)
        self.ax_price.set_title('Price vs Fundamental')
        self.ax_price.set_xlabel('Step')
        self.ax_price.set_ylabel('Price')
        self.ax_price.legend()
        self.ax_price.grid(alpha=0.3)
        
        self.ax_bubble.clear()
        self.ax_bubble.plot(steps, self.history['bubble_ratio'], color='#ff6600', linewidth=2)
        self.ax_bubble.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        self.ax_bubble.axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Bubble threshold')
        self.ax_bubble.set_title('Bubble Ratio')
        self.ax_bubble.set_xlabel('Step')
        self.ax_bubble.set_ylabel('P/P*')
        self.ax_bubble.legend(fontsize=8)
        self.ax_bubble.grid(alpha=0.3)
        
        self.ax_mispricing.clear()
        self.ax_mispricing.plot(steps, self.history['mispricing'], color='#cc0000', linewidth=1.5)
        self.ax_mispricing.axhline(0, color='black', linestyle='-', alpha=0.5)
        self.ax_mispricing.set_title('Mispricing')
        self.ax_mispricing.set_xlabel('Step')
        self.ax_mispricing.set_ylabel('P - P*')
        self.ax_mispricing.grid(alpha=0.3)
        
        self.ax_volume.clear()
        self.ax_volume.plot(steps, self.history['volume'], color='#9900cc', linewidth=1.5)
        self.ax_volume.set_title('Trading Volume')
        self.ax_volume.set_xlabel('Step')
        self.ax_volume.set_ylabel('Volume')
        self.ax_volume.grid(alpha=0.3)
        
        self.ax_gini.clear()
        self.ax_gini.plot(steps, self.history['gini'], color='#cc6600', linewidth=1.5)
        self.ax_gini.set_title('Gini (Wealth Inequality)')
        self.ax_gini.set_xlabel('Step')
        self.ax_gini.set_ylabel('Gini')
        self.ax_gini.grid(alpha=0.3)
        
        self.ax_returns.clear()
        self.ax_returns.plot(steps, self.history['log_return'], color='#00cc66', linewidth=1.0, alpha=0.7)
        self.ax_returns.axhline(0, color='black', linestyle='-', alpha=0.5)
        self.ax_returns.set_title('Log Returns')
        self.ax_returns.set_xlabel('Step')
        self.ax_returns.set_ylabel('Log Return')
        self.ax_returns.grid(alpha=0.3)
        
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        policy_info = POLICY_PRESETS[self.policy]
        info_text = f"""
        CURRENT STATE (Step {last['Step']})
        
        Price: {last['Price']:.2f}  |  Fundamental: {last['FundamentalPrice']:.2f}  |  Dividend: {last['Dividend']:.3f}
        Mispricing: {last['Mispricing']:+.2f} ({(last['Mispricing']/last['FundamentalPrice']*100):+.1f}%)
        Bubble Ratio: {last['BubbleRatio']:.2f}  |  Volume: {last['Volume']:.0f}  |  Gini: {last['GiniWealthDisc']:.3f}
        
        POLICY: {self.policy.upper()}
        Tax: {policy_info.tau*100:.1f}%  |  Leverage: {policy_info.L_max:.1f}x  |  Short Ban: {policy_info.short_ban}
        """
        
        self.ax_info.text(0.05, 0.5, info_text, fontsize=11, verticalalignment='center',
                         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    def run(self):
        print(f"Starting visualization for policy: {self.policy}")
        print(f"Running for {self.total_steps} steps")
        print("Close the window to stop")
        print()
        
        self.update_display()
        
        anim = FuncAnimation(self.fig, self.update, frames=self.total_steps - 4, 
                           interval=100, repeat=False, blit=True)
        plt.show()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("ERROR: Policy argument is required!")
        print(f"\nUsage: python3 viz_simple.py <policy> [steps] [seed]")
        print(f"\nArguments:")
        print(f"  policy: {list(POLICY_PRESETS.keys())} (REQUIRED)")
        print(f"  steps: integer (optional, default=50)")
        print(f"  seed: integer (optional, uses random if omitted)")
        print(f"\nExamples:")
        print(f"  python3 viz_simple.py none              (policy=none, 50 steps, random seed)")
        print(f"  python3 viz_simple.py moderate 100      (policy=moderate, 100 steps, random seed)")
        print(f"  python3 viz_simple.py excessive 75 42   (policy=excessive, 75 steps, seed 42)")
        sys.exit(1)
    
    policy = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if policy not in POLICY_PRESETS:
        print(f"ERROR: Invalid policy '{policy}'")
        print(f"Available policies: {list(POLICY_PRESETS.keys())}")
        sys.exit(1)
    
    print("=" * 70)
    print("MARKET ABM - LIVE VISUALIZATION")
    print("=" * 70)
    seed_info = f" with seed {seed}" if seed is not None else " (random seed)"
    print(f"Policy: {policy.upper()}{seed_info} | Steps: {steps}")
    print("=" * 70)
    
    viz = LiveMarketViz(policy=policy, steps=steps, seed=seed)
    viz.run()
