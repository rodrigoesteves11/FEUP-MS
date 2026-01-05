"""
agents.py

Agentes para simulação ABM de mercado com:
- Fundamentalistas (mean reversion ao fundamental)
- Chartistas (momentum em log)
- Noise traders (expectativa ruidosa)

Cada agente produz uma expectativa de payoff de 1 passo:
    mu_i,t = E_t[P_{t+1} + D_{t+1}]
"""

from __future__ import annotations

import math
import mesa


class TraderAgent(mesa.Agent):
    """
    Base class para agentes.

    Estado:
      - cash: C_{i,t}
      - shares: q_{i,t}
      - wealth_base: W_{i,t} calculada no início do step após liquidação (fixa dentro do step)

    Expectativas:
      - expected_price: E_t[P_{t+1}]
      - expected_dividend: E_t[D_{t+1}]
      - mu: E_t[P_{t+1} + D_{t+1}]
    """

    def __init__(self, model, cash: float, shares: float, gamma: float, sigma2: float):
        super().__init__(model)
        self.cash = float(cash)
        self.shares = float(shares)

        # Preferências/risco
        self.gamma = float(gamma)     # aversão ao risco
        self.sigma2 = float(sigma2)   # var(payoff) em euros^2

        # Expectativas (atualizadas em cada step)
        self.expected_price = 0.0
        self.expected_dividend = 0.0
        self.mu = 0.0

        # Wealth base fixada no início do step (após liquidação)
        self.wealth_base = 0.0

    def compute_expected_dividend(self) -> float:
        raise NotImplementedError

    def compute_expected_price(self) -> float:
        raise NotImplementedError

    def step(self):
        """
        Fase de expectativas: chamada pelo modelo ANTES do tâtonnement.
        """
        self.expected_dividend = float(self.compute_expected_dividend())
        self.expected_price = float(self.compute_expected_price())
        self.mu = self.expected_price + self.expected_dividend

    # Utilitário: wealth mark-to-market no preço atual do modelo
    def wealth(self) -> float:
        return self.cash + self.shares * self.model.price


class FundamentalistAgent(TraderAgent):
    """
    Fundamentalista:
      E_t[P_{t+1}] = P_t + kappa_f (P*_t - P_t)
      E_t[D_{t+1}] = d_bar + rho (D_t - d_bar)
    """

    def __init__(self, model, cash, shares, gamma, sigma2, kappa_f: float):
        super().__init__(model, cash, shares, gamma, sigma2)
        self.kappa_f = float(kappa_f)

    def compute_expected_dividend(self) -> float:
        d_bar = self.model.d_bar
        rho = self.model.rho
        D_t = self.model.dividend
        return d_bar + rho * (D_t - d_bar)

    def compute_expected_price(self) -> float:
        P_t = self.model.price
        P_star = self.model.fundamental_price(self.model.dividend)
        return P_t + self.kappa_f * (P_star - P_t)


class ChartistAgent(TraderAgent):
    """
    Chartista (momentum em log):
      m_t = ln(P_t / P_{t-L})
      E_t[P_{t+1}] = P_t * exp(kappa_c * m_t)
      E_t[D_{t+1}] = d_bar
    """

    def __init__(self, model, cash, shares, gamma, sigma2, L: int, kappa_c: float):
        super().__init__(model, cash, shares, gamma, sigma2)
        self.L = int(L)
        self.kappa_c = float(kappa_c)

    def compute_expected_dividend(self) -> float:
        return self.model.d_bar

    def compute_expected_price(self) -> float:
        P_t = self.model.price
        hist = self.model.price_history

        if self.L <= 0 or len(hist) <= self.L:
            # Sem histórico suficiente: sem sinal
            return P_t

        P_t_L = hist[-(self.L + 1)]
        if P_t_L <= 0:
            return P_t

        m_t = math.log(P_t / P_t_L)
        return P_t * math.exp(self.kappa_c * m_t)


class NoiseAgent(TraderAgent):
    """
    Noise trader (expectativa ruidosa multiplicativa):
      E_t[P_{t+1}] = P_t * exp(sigma_n * xi),  xi ~ N(0,1)
      E_t[D_{t+1}] = d_bar
    """

    def __init__(self, model, cash, shares, gamma, sigma2, sigma_n: float):
        super().__init__(model, cash, shares, gamma, sigma2)
        self.sigma_n = float(sigma_n)

    def compute_expected_dividend(self) -> float:
        return self.model.d_bar

    def compute_expected_price(self) -> float:
        P_t = self.model.price
        xi = self.model.random.gauss(0.0, 1.0)
        return P_t * math.exp(self.sigma_n * xi)
