"""
Modelo de mercado financeiro com agentes heterogéneos.
"""

import math
import mesa
from mesa.datacollection import DataCollector
from agents import FundamentalistAgent, ChartistAgent, NoiseAgent


def compute_gini(values):
    """Calcula o coeficiente de Gini para uma lista de valores."""
    if len(values) == 0:
        return 0.0
    values = sorted(values)
    n = len(values)
    total = sum(values)
    if total == 0:
        return 0.0
    cumsum = 0
    gini_sum = 0
    for i, v in enumerate(values):
        cumsum += v
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total)


class MarketModel(mesa.Model):
    """Modelo principal - clearing Σq(P)=Q e dinâmica de dividendos."""
    
    def __init__(
        self,
        n_fundamentalists=10,
        n_chartists=10,
        n_noise=10,
        initial_price=20.0,
        initial_dividend=1.0,
        r=0.05,
        Q=15.0,
        d_bar=1.0,
        rho=0.95,
        sigma_d=0.12,
        initial_wealth=500.0,
        risk_aversion=1.0,
        perceived_variance=2.5,   # Var(P+D) em euros² (não % de retorno)
        heterogeneity_delta=0.7,  # δ para heterogeneidade (0.2-0.5)
        L=10,
        chi=0.45,
        lambda_=0.3,
        sigma_n=1.5,              # Ruído em euros (compatível com preços)
        seed=None
    ):
        super().__init__(seed=seed)
        
        # ===== VALIDATIONS (GUARDRAILS) =====
        if Q <= 0:
            raise ValueError("Q (stock supply) must be positive")
        if r <= 0:
            raise ValueError("r (risk-free rate) must be positive")
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        if perceived_variance <= 0:
            raise ValueError("perceived_variance must be positive")
        if L < 1:
            raise ValueError("L (moving average window) must be >= 1")
        if not (0 < rho < 1):
            raise ValueError("rho must be between 0 and 1 (exclusive)")
        if sigma_d < 0:
            raise ValueError("sigma_d must be non-negative")
        if sigma_n < 0:
            raise ValueError("sigma_n must be non-negative")
        if initial_price <= 0:
            raise ValueError("initial_price must be positive")
        if initial_dividend < 0:
            raise ValueError("initial_dividend must be non-negative")
        if initial_wealth <= 0:
            raise ValueError("initial_wealth must be positive")
        if d_bar <= 0:
            raise ValueError("d_bar (mean dividend) must be positive")
        if not (0 <= heterogeneity_delta < 1):
            raise ValueError("heterogeneity_delta must be in [0, 1)")
        
        # ===== PARÂMETROS DO MERCADO =====
        self.r = r                  # Taxa de juro sem risco
        self.Q = Q                  # Quantidade total de ações (oferta fixa)
        self.d_bar = d_bar          # Média histórica dos dividendos (d̄)
        self.rho = rho              # Persistência dos dividendos (ρ)
        self.sigma_d = sigma_d      # Volatilidade dos dividendos (σ_d)
        self.L = L                  # Janela da média móvel (para chartistas)
        
        # ===== ESTADO DO MERCADO =====
        self.price = initial_price          # P_t (preço atual)
        self.prev_price = initial_price     # P_{t-1} para log returns
        self.dividend = initial_dividend    # D_t (dividendo atual)
        self.price_history = [initial_price] * (L + 1)  # Histórico para chartistas
        
        # ===== MÉTRICAS AVANÇADAS =====
        self.step_count = 0                 # Contador de steps
        self.peak_price = initial_price     # Preço máximo (para drawdown)
        self.volume = 0.0                   # Volume de transações
        self.log_returns = []               # Histórico de log returns
        self.initial_total_wealth = initial_wealth * (n_fundamentalists + n_chartists + n_noise)
        
        # ===== CRIAÇÃO DOS AGENTES COM HETEROGENEIDADE =====
        delta = heterogeneity_delta
        
        for _ in range(n_fundamentalists):
            # α_i ~ U(α(1-δ), α(1+δ))
            alpha_i = self.random.uniform(
                risk_aversion * (1 - delta),
                risk_aversion * (1 + delta)
            )
            # σ²_i ~ U(σ²(1-δ), σ²(1+δ))
            sigma2_i = self.random.uniform(
                perceived_variance * (1 - delta),
                perceived_variance * (1 + delta)
            )
            FundamentalistAgent(
                model=self,
                initial_wealth=initial_wealth,
                risk_aversion=alpha_i,
                perceived_variance=sigma2_i,
                rho=rho
            )
        
        for _ in range(n_chartists):
            alpha_i = self.random.uniform(
                risk_aversion * (1 - delta),
                risk_aversion * (1 + delta)
            )
            sigma2_i = self.random.uniform(
                perceived_variance * (1 - delta),
                perceived_variance * (1 + delta)
            )
            ChartistAgent(
                model=self,
                initial_wealth=initial_wealth,
                risk_aversion=alpha_i,
                perceived_variance=sigma2_i,
                L=L,
                chi=chi,
                lambda_=lambda_
            )
        
        for _ in range(n_noise):
            alpha_i = self.random.uniform(
                risk_aversion * (1 - delta),
                risk_aversion * (1 + delta)
            )
            sigma2_i = self.random.uniform(
                perceived_variance * (1 - delta),
                perceived_variance * (1 + delta)
            )
            NoiseAgent(
                model=self,
                initial_wealth=initial_wealth,
                risk_aversion=alpha_i,
                perceived_variance=sigma2_i,
                sigma_n=sigma_n
            )
        
        # ===== DATA COLLECTOR =====
        self.datacollector = DataCollector(
            model_reporters={
                # Preços
                "Price": lambda m: m.price,
                "FundamentalPrice": lambda m: (m.d_bar/m.r) + (m.rho/(1+m.r-m.rho))*(m.dividend - m.d_bar),
                "Dividend": lambda m: m.dividend,
                
                # Mispricing
                "Mispricing": lambda m: m.price - (m.d_bar + m.rho*(m.dividend - m.d_bar)) / m.r,
                
                # Returns e Volatilidade
                "LogReturn": lambda m: m.log_returns[-1] if m.log_returns else 0.0,
                "VolatilityRolling": lambda m: m._compute_rolling_volatility(20),
                
                # Volume (proxy de liquidez)
                "Volume": lambda m: m.volume,
                
                # Riqueza descontada
                "TotalWealthDisc": lambda m: sum(a.wealth for a in m.agents) / ((1 + m.r) ** m.step_count) if m.step_count > 0 else sum(a.wealth for a in m.agents),
                
                # Gini (desigualdade)
                "GiniWealth": lambda m: compute_gini([a.wealth / ((1 + m.r) ** m.step_count) if m.step_count > 0 else a.wealth for a in m.agents]),
                
                # Drawdown (negativo = queda desde pico)
                "Drawdown": lambda m: (m.price - m.peak_price) / m.peak_price if m.peak_price > 0 else 0.0,
                
                # Crashes e Bolhas (k=2 é menos exigente que k=3)
                "IsCrash": lambda m: m._is_crash(k=2),
                "BubbleRatio": lambda m: m.price / ((m.d_bar + m.rho*(m.dividend - m.d_bar)) / m.r) if m.r > 0 else 1.0,
                
                # Controlo
                "TotalWealth": lambda m: sum(a.wealth for a in m.agents),
                "TotalShares": lambda m: sum(a.shares for a in m.agents),
                "MeanShares": lambda m: sum(a.shares for a in m.agents) / len(m.agents) if len(m.agents) > 0 else 0,
                "SharesError": lambda m: abs(sum(a.shares for a in m.agents) - m.Q),
                "Step": lambda m: m.step_count,
            },
            agent_reporters={
                "Wealth": lambda a: a.wealth,
                "WealthDisc": lambda a: a.wealth / ((1 + a.model.r) ** a.model.step_count) if a.model.step_count > 0 else a.wealth,
                "WealthShare": lambda a: a.wealth / sum(ag.wealth for ag in a.model.agents) if sum(ag.wealth for ag in a.model.agents) > 0 else 0,
                "Shares": lambda a: a.shares,
                "AgentType": lambda a: type(a).__name__,
            }
        )
    
    def _compute_rolling_volatility(self, window=20):
        """Calcula volatilidade rolling dos log returns."""
        if len(self.log_returns) < 2:
            return 0.0
        recent = self.log_returns[-window:]
        if len(recent) < 2:
            return 0.0
        mean = sum(recent) / len(recent)
        variance = sum((r - mean) ** 2 for r in recent) / (len(recent) - 1)
        return math.sqrt(variance)
    
    def _is_crash(self, k=3):
        """Verifica se houve crash (log_return < -k * volatility)."""
        if len(self.log_returns) < 2:
            return False
        vol = self._compute_rolling_volatility(20)
        if vol <= 0:
            return False
        last_return = self.log_returns[-1]
        return last_return < -k * vol
    
    def compute_clearing_price(self):
        """
        Clearing: resolve Σ q_i(P) = Q analiticamente.
        
        Com demanda escalada por wealth relativa:
        q_i = scale_i * (E[P+D] - (1+r)P) / (α_i * σ²_i)
        onde scale_i = W_i / W̄ (wealth relativa à média)
        
        A = Σ (scale_i / (α_i * σ²_i))
        B = Σ (scale_i * E_i[P_{t+1} + D_{t+1}] / (α_i * σ²_i))
        
        P_t = (B - Q) / ((1+r) * A)
        """
        A = 0.0
        B = 0.0
        
        # Calcular média da wealth
        mean_wealth = sum(a.wealth for a in self.agents) / len(self.agents)
        
        for agent in self.agents:
            wealth_scale = agent.wealth / mean_wealth if mean_wealth > 0 else 1.0
            inv_alpha_sigma = 1.0 / (agent.risk_aversion * agent.perceived_variance)
            expected_payoff = agent.compute_expected_payoff()
            
            A += wealth_scale * inv_alpha_sigma
            B += wealth_scale * expected_payoff * inv_alpha_sigma
        
        if A <= 0:
            return self.price  # Fallback
        
        new_price = (B - self.Q) / ((1 + self.r) * A)
        
        return new_price
    
    def compute_next_dividend(self):
        """
        Dividendo estocástico: D_{t+1} = d̄ + ρ(D_t - d̄) + ε_t
        onde ε_t ~ N(0, σ²_d)
        
        Garante dividendo não-negativo.
        """
        epsilon = self.random.gauss(0, self.sigma_d)
        new_dividend = self.d_bar + self.rho * (self.dividend - self.d_bar) + epsilon
        return max(0.0, new_dividend)
    
    def step(self):
        """
        Executa um passo da simulação (ordem temporal correta):
        1. Temos P_t, D_t e shares antigas q_{i,t} do período anterior
        2. Agentes calculam expectativas E[P_{t+1}], E[D_{t+1}]
        3. Clearing determina P_{t+1} que satisfaz Σq(P) = Q
        4. Gera-se D_{t+1}
        5. Atualiza-se riqueza W_{i,t+1} usando shares ANTIGAS q_{i,t}
        6. Calcula-se novas shares q_{i,t+1} para o próximo período
        """
        # Guardar preço e shares antigos
        self.prev_price = self.price
        old_shares = {agent.unique_id: agent.shares for agent in self.agents}
        
        # Fase 1: Agentes calculam expectativas
        self.agents.shuffle_do("step")
        
        # Fase 2: Calcular novo preço via clearing Σq(P) = Q
        new_price = self.compute_clearing_price()
        self.price_history.append(new_price)
        self.price_history = self.price_history[-(self.L + 1):]
        self.price = new_price
        
        # Calcular log return
        if self.prev_price > 0 and new_price > 0:
            log_ret = math.log(new_price / self.prev_price)
            self.log_returns.append(log_ret)
        
        # Atualizar peak price para drawdown
        if new_price > self.peak_price:
            self.peak_price = new_price
        
        # Fase 3: Gerar novo dividendo
        self.dividend = self.compute_next_dividend()
        
        # Fase 4: Atualizar riqueza usando shares ANTIGAS (do período anterior)
        for agent in self.agents:
            agent.update_wealth(self.prev_price)
        
        # Fase 5: Calcular novas shares para o próximo período
        for agent in self.agents:
            agent.shares = agent.compute_demand(new_price)
        
        # Calcular volume (soma das mudanças absolutas de shares)
        self.volume = sum(abs(agent.shares - old_shares[agent.unique_id]) 
                         for agent in self.agents)
        
        # Incrementar contador de passos
        self.step_count += 1
        
        # Recolher dados
        self.datacollector.collect(self)
    
    def run(self, steps):
        """Executa a simulação por N passos."""
        for _ in range(steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()


