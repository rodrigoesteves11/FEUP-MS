"""
Modelo de mercado financeiro com agentes heterogéneos.
"""

import mesa
import numpy as np
from mesa.datacollection import DataCollector
from agents import FundamentalistAgent, ChartistAgent, NoiseAgent


class MarketModel(mesa.Model):
    """Modelo principal - clearing Walrasiano e dinâmica de dividendos."""
    
    def __init__(
        self,
        n_fundamentalists=10,
        n_chartists=10,
        n_noise=10,
        initial_price=100.0,
        initial_dividend=1.0,
        r=0.05,
        Q=1000.0,
        d_bar=1.0,
        rho=0.9,
        sigma_d=0.1,
        initial_wealth=1000.0,
        risk_aversion=1.0,
        perceived_variance=0.04,
        L=5,
        chi=0.5,
        lambda_=0.5,
        sigma_n=1.0,
        seed=None
    ):
        super().__init__(seed=seed)
        
        # ===== PARÂMETROS DO MERCADO =====
        self.r = r                  # Taxa de juro sem risco
        self.Q = Q                  # Quantidade total de ações (oferta fixa)
        self.d_bar = d_bar          # Média histórica dos dividendos (d̄)
        self.rho = rho              # Persistência dos dividendos (ρ)
        self.sigma_d = sigma_d      # Volatilidade dos dividendos (σ_d)
        
        # ===== ESTADO DO MERCADO =====
        self.price = initial_price          # P_t (preço atual)
        self.dividend = initial_dividend    # D_t (dividendo atual)
        self.price_history = [initial_price] * (L + 1)  # Histórico para chartistas
        
        # ===== CRIAÇÃO DOS AGENTES =====
        for _ in range(n_fundamentalists):
            FundamentalistAgent(
                model=self,
                initial_wealth=initial_wealth,
                risk_aversion=risk_aversion,
                perceived_variance=perceived_variance,
                rho=rho
            )
        
        for _ in range(n_chartists):
            ChartistAgent(
                model=self,
                initial_wealth=initial_wealth,
                risk_aversion=risk_aversion,
                perceived_variance=perceived_variance,
                L=L,
                chi=chi,
                lambda_=lambda_
            )
        
        for _ in range(n_noise):
            NoiseAgent(
                model=self,
                initial_wealth=initial_wealth,
                risk_aversion=risk_aversion,
                perceived_variance=perceived_variance,
                sigma_n=sigma_n
            )
        
        # ===== DATA COLLECTOR =====
        self.datacollector = DataCollector(
            model_reporters={
                "Price": lambda m: m.price,
                "Dividend": lambda m: m.dividend,
            },
            agent_reporters={
                "Wealth": lambda a: a.wealth,
                "Allocation": lambda a: a.allocation,
            }
        )
    
    def compute_clearing_price(self):
        """
        Clearing Walrasiano: P_{t+1} = Σ(x_{i,t} * W_{i,t}) / Q
        """
        total_demand = sum(a.allocation * a.wealth for a in self.agents)
        return total_demand / self.Q
    
    def compute_next_dividend(self):
        """
        Dividendo estocástico: D_{t+1} = d̄ + ρ(D_t - d̄) + ε_t
        onde ε_t ~ N(0, σ²_d)
        """
        epsilon = np.random.normal(0, self.sigma_d)
        return self.d_bar + self.rho * (self.dividend - self.d_bar) + epsilon
    
    def step(self):
        """
        Executa um passo da simulação:
        1. Agentes calculam expectativas e demanda (x_i,t)
        2. Clearing Walrasiano determina P_{t+1}
        3. Dividendo D_{t+1} é gerado
        4. Agentes atualizam riqueza W_{i,t+1}
        """
        # Fase 1: Agentes calculam demanda
        self.agents.shuffle_do("step")
        
        # Fase 2: Calcular novo preço via clearing
        new_price = self.compute_clearing_price()
        self.price_history.append(new_price)
        self.price = new_price
        
        # Fase 3: Gerar novo dividendo
        self.dividend = self.compute_next_dividend()
        
        # Fase 4: Atualizar riqueza dos agentes
        self.agents.do("update_wealth")
        
        # Recolher dados
        self.datacollector.collect(self)
    
    def run(self, steps):
        """Executa a simulação por N passos."""
        for _ in range(steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()


