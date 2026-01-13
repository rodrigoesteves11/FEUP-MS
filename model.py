"""
model.py

Mercado ABM sem livro de ordens:
- Specialist ajusta preço por tâtonnement:
    P <- P * exp( eta * Z(P) / Q )
  onde Z(P) = sum_i delta_q_eff_i(P)

- Cada agente calcula mu_i,t = E_t[P_{t+1} + D_{t+1}] no início do step.
- A ordem desejada é baseada numa posição-alvo (exposição em euros) limitada por tanh.

Regulação fixa via policy:
- none / moderate / excessive
- aplica taxa, short ban, leverage/margem, q_max e C_min.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import mesa
from mesa.datacollection import DataCollector

from agents import FundamentalistAgent, ChartistAgent, NoiseAgent


# ----------------------------
# Utilitários
# ----------------------------

def compute_gini_nonnegative(values: List[float]) -> float:
    """
    Gini standard requer valores não-negativos.
    Aqui assumimos que `values` já foram clipados para >=0.
    """
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0

    values_sorted = sorted(values)
    cum = 0.0
    gsum = 0.0
    for i, v in enumerate(values_sorted, start=1):
        cum += v
        gsum += (2 * i - n - 1) * v
    return gsum / (n * total)


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------
# Políticas
# ----------------------------

@dataclass(frozen=True)
class PolicyParams:
    tau: float              # taxa transação (proporcional)
    L_max: float            # limite de alavancagem (exposição <= L_max * W_base)
    short_ban: bool         # se True, impõe q >= 0
    q_max: float            # limite superior de posição (shares); use math.inf para "sem"
    C_min: float            # cash mínimo permitido; use -math.inf para "sem"


POLICY_PRESETS: Dict[str, PolicyParams] = {
    "none": PolicyParams(
        tau=0.0,
        L_max=math.inf,
        short_ban=False,
        q_max=math.inf,
        C_min=-math.inf,
    ),
    "moderate": PolicyParams(
        tau=0.003,      # 0.3% transaction tax
        L_max=1.3,      # 1.3x leverage limit
        short_ban=False,
        q_max=math.inf,
        C_min=-math.inf,
    ),
    "excessive": PolicyParams(
        tau=0.01,       # 1% transaction tax
        L_max=1.3,      # 1.3x leverage limit
        short_ban=True, # Proíbe short selling
        q_max=20.0,     # máximo 20 shares
        C_min=-200.0,   # margem limitada
    ),
}


class MarketModel(mesa.Model):
    """
    Modelo principal do mercado com:
    - 3 tipos fixos de agentes
    - price discovery por tâtonnement
    - políticas de regulação fixa aplicadas às ordens
    """

    def __init__(
        self,
        # Agentes
        n_fundamentalists: int = 100,
        n_chartists: int = 100,
        n_noise: int = 100,

        # Mercado
        initial_price: float = 20.0,
        initial_dividend: float = 1.0,
        Q: float = 300.0,
        r: float = 0.05,

        # Dividendo
        d_bar: float = 1.0,
        rho: float = 0.95,
        sigma_d: float = 0.15,

        # Inicialização de riqueza/posições
        initial_wealth: float = 1000.0,  # riqueza total inicial por agente (aprox.)
        # preferências e risco heterogéneos (intervalos)
        gamma_range: Tuple[float, float] = (0.5, 1.5),
        sigma2_range: Tuple[float, float] = (1.0, 6.0),

        # Parâmetros comportamentais (intervalos)
        kappa_f_range: Tuple[float, float] = (0.05, 0.20),   # fundamentalistas
        kappa_c_range: Tuple[float, float] = (0.5, 2.0),     # chartistas
        chartist_L_choices: Tuple[int, ...] = (5, 20, 60),   # chartistas
        sigma_n_range: Tuple[float, float] = (0.01, 0.05),   # noise (em log)

        # Trading rule
        beta: float = 1.0,
        phi: float = 0.2,

        # Tâtonnement
        tatonnement_K: int = 20,
        tatonnement_eta: float = 0.05,

        # Política
        policy_name: str = "none",

        seed: Optional[int] = None,
        
        max_steps: Optional[int] = 50,
    ):
        super().__init__(seed=seed)

        # ----- Guardrails básicos -----
        if initial_price <= 0:
            raise ValueError("initial_price must be > 0")
        if initial_dividend < 0:
            raise ValueError("initial_dividend must be >= 0")
        if Q <= 0:
            raise ValueError("Q must be > 0")
        if r <= 0:
            raise ValueError("r must be > 0")
        if d_bar <= 0:
            raise ValueError("d_bar must be > 0")
        if not (0 < rho < 1):
            raise ValueError("rho must be in (0,1)")
        if sigma_d < 0:
            raise ValueError("sigma_d must be >= 0")
        if beta <= 0:
            raise ValueError("beta must be > 0")
        if not (0 < phi <= 1):
            raise ValueError("phi must be in (0,1]")
        if tatonnement_K <= 0:
            raise ValueError("tatonnement_K must be > 0")
        if tatonnement_eta <= 0:
            raise ValueError("tatonnement_eta must be > 0")
        if policy_name not in POLICY_PRESETS:
            raise ValueError(f"Unknown policy_name: {policy_name}")

        self.max_steps = max_steps
        
        self.policy_name = policy_name
        self.policy = POLICY_PRESETS[policy_name]

        # ----- Parâmetros do mercado -----
        self.Q = float(Q)
        self.r = float(r)
        self.d_bar = float(d_bar)
        self.rho = float(rho)
        self.sigma_d = float(sigma_d)

        self.beta = float(beta)
        self.phi = float(phi)

        self.tatonnement_K = int(tatonnement_K)
        self.tatonnement_eta = float(tatonnement_eta)

        # ----- Estado do mercado -----
        self.price = float(initial_price)        # P_t
        self.prev_price = float(initial_price)   # P_{t-1}
        self.dividend = float(initial_dividend)  # D_t (dividendo "corrente", pago no início do step)
        self.step_count = 0

        # Histórico para momentum
        self.max_L = max(chartist_L_choices) if chartist_L_choices else 1
        self.price_history: List[float] = [self.price] * (self.max_L + 1)

        # Specialist inventory (para conservar total de shares)
        self.specialist_inventory = 0.0

        # Métricas intra-step
        self.volume = 0.0
        self.peak_price = self.price

        # ----- Criar agentes e distribuir shares iniciais -----
        N = int(n_fundamentalists + n_chartists + n_noise)
        if N <= 0:
            raise ValueError("Total number of agents must be > 0")

        # Distribuição inicial de shares: soma(q_i) = Q e specialist começa com 0
        q0 = self.Q / N
        # Cash inicial para que wealth ~ initial_wealth:
        # C0 = W0 - q0 * P0
        C0 = float(initial_wealth) - q0 * self.price
        if C0 < 0:
            raise ValueError(
                "initial_wealth too low for initial share endowment. "
                "Increase initial_wealth or reduce Q or initial_price."
            )

        # Funções de amostragem heterogénea
        def sample_uniform(lo_hi: Tuple[float, float]) -> float:
            lo, hi = lo_hi
            if lo <= 0 or hi <= 0 or hi < lo:
                raise ValueError(f"Invalid range {lo_hi}")
            return self.random.uniform(lo, hi)

        # Criar fundamentalistas
        for _ in range(int(n_fundamentalists)):
            gamma_i = sample_uniform(gamma_range)
            sigma2_i = sample_uniform(sigma2_range)
            kappa_f = self.random.uniform(*kappa_f_range)
            FundamentalistAgent(
                model=self,
                cash=C0,
                shares=q0,
                gamma=gamma_i,
                sigma2=sigma2_i,
                kappa_f=kappa_f,
            )

        # Criar chartistas
        for _ in range(int(n_chartists)):
            gamma_i = sample_uniform(gamma_range)
            sigma2_i = sample_uniform(sigma2_range)
            L_i = int(self.random.choice(chartist_L_choices))
            kappa_c = self.random.uniform(*kappa_c_range)
            ChartistAgent(
                model=self,
                cash=C0,
                shares=q0,
                gamma=gamma_i,
                sigma2=sigma2_i,
                L=L_i,
                kappa_c=kappa_c,
            )

        # Criar noise traders
        for _ in range(int(n_noise)):
            gamma_i = sample_uniform(gamma_range)
            sigma2_i = sample_uniform(sigma2_range)
            sigma_n = self.random.uniform(*sigma_n_range)
            NoiseAgent(
                model=self,
                cash=C0,
                shares=q0,
                gamma=gamma_i,
                sigma2=sigma2_i,
                sigma_n=sigma_n,
            )

        # ----- DataCollector -----
        self.datacollector = DataCollector(
            model_reporters={
                "Step": lambda m: m.step_count,
                "Policy": lambda m: m.policy_name,
                "Price": lambda m: m.price,
                "Dividend": lambda m: m.dividend,
                "FundamentalPrice": lambda m: m.fundamental_price(m.dividend),
                "Mispricing": lambda m: m.price - m.fundamental_price(m.dividend),
                "BubbleRatio": lambda m: (m.price / m.fundamental_price(m.dividend)) if m.fundamental_price(m.dividend) > 0 else 1.0,
                "LogReturn": lambda m: math.log(m.price / m.prev_price) if (m.prev_price > 0 and m.price > 0) else 0.0,
                "Volume": lambda m: m.volume,
                "Turnover": lambda m: (m.volume / m.Q) if m.Q > 0 else 0.0,
                "PeakPrice": lambda m: m.peak_price,
                "Drawdown": lambda m: (m.price - m.peak_price) / m.peak_price if m.peak_price > 0 else 0.0,
                "TotalWealth": lambda m: sum(a.cash + a.shares * m.price for a in m.agents),
                "TotalWealthDisc": lambda m: (sum(a.cash + a.shares * m.price for a in m.agents) / ((1 + m.r) ** m.step_count)) if m.step_count > 0 else sum(a.cash + a.shares * m.price for a in m.agents),
                "GiniWealth": lambda m: compute_gini_nonnegative([max(0.0, a.cash + a.shares * m.price) for a in m.agents]),
                "GiniWealthDisc": lambda m: compute_gini_nonnegative([max(0.0, (a.cash + a.shares * m.price) / ((1 + m.r) ** m.step_count) if m.step_count > 0 else (a.cash + a.shares * m.price)) for a in m.agents]),
                "NegWealthShare": lambda m: (sum(1 for a in m.agents if (a.cash + a.shares * m.price) < 0) / len(m.agents)) if len(m.agents) > 0 else 0.0,
                "SharesAgents": lambda m: sum(a.shares for a in m.agents),
                "SpecialistInv": lambda m: m.specialist_inventory,
            },
            agent_reporters={
                "Cash": lambda a: a.cash,
                "Shares": lambda a: a.shares,
                "Wealth": lambda a: a.cash + a.shares * a.model.price,
                "WealthDisc": lambda a: ((a.cash + a.shares * a.model.price) / ((1 + a.model.r) ** a.model.step_count)) if a.model.step_count > 0 else (a.cash + a.shares * a.model.price),
                "AgentType": lambda a: type(a).__name__,
            }
        )

        # Coletar estado inicial (Step=0)
        self.datacollector.collect(self)

    # ----------------------------
    # Fundamental e dividendo
    # ----------------------------

    def fundamental_price(self, D_t: float) -> float:
        """
        PV do AR(1):
            P*_t = d_bar/r + rho/(1+r-rho) * (D_t - d_bar)
        """
        return (self.d_bar / self.r) + (self.rho / (1 + self.r - self.rho)) * (D_t - self.d_bar)

    def next_dividend(self) -> float:
        """
        AR(1) com choque normal:
            D_{t+1} = d_bar + rho(D_t - d_bar) + sigma_d * eps
        com truncagem a >=0.
        """
        eps = self.random.gauss(0.0, 1.0)
        D_next = self.d_bar + self.rho * (self.dividend - self.d_bar) + self.sigma_d * eps
        return max(0.0, D_next)

    # ----------------------------
    # Liquidação e ordens
    # ----------------------------

    def _settle_cash_and_dividends(self):
        """
        Liquidação no início do step:
          C <- C(1+r) + q*D_t
        e fixa wealth_base (W_base) para o step.
        """
        for a in self.agents:
            a.cash *= (1.0 + self.r)
            a.cash += a.shares * self.dividend

        # wealth_base fixado com P_t (preço antes do clearing)
        for a in self.agents:
            a.wealth_base = a.cash + a.shares * self.price

    def _desired_delta_q(self, a, P_candidate: float) -> float:
        """
        Ordem desejada (antes de regulação), conforme:
          Delta mu(P) = mu_i - (1+r)P
          x* = W_base * tanh( beta * Delta mu / (gamma*sigma2) )
          q* = x* / P
          delta_q = phi * (q* - q)
        """
        if P_candidate <= 0:
            return 0.0

        W = a.wealth_base
        q = a.shares

        # se W <= 0: tenta reduzir risco -> alvo 0
        if W <= 0:
            q_star = 0.0
            return self.phi * (q_star - q)

        delta_mu = a.mu - (1.0 + self.r) * P_candidate
        denom = a.gamma * a.sigma2
        if denom <= 0:
            return 0.0

        signal = self.beta * (delta_mu / denom)
        x_star = W * math.tanh(signal)           # exposição alvo em euros
        q_star = x_star / P_candidate           # ações alvo

        return self.phi * (q_star - q)

    def _apply_policy(self, a, P_candidate: float, delta_q: float) -> float:
        """
        Aplica regras de regulação FIXAS na ordem, por esta ordem:

        1) short ban (se ativo): q' >= 0
        2) q_max (se definido): q' <= q_max
        3) margem/alavancagem: |q'|P <= L_max * W_base
        4) cash floor C_min: C' >= C_min

        Retorna delta_q efetivo.
        """
        pol = self.policy
        tau = pol.tau
        L_max = pol.L_max
        short_ban = pol.short_ban
        q_max = pol.q_max
        C_min = pol.C_min

        if P_candidate <= 0:
            return 0.0

        q = a.shares
        C = a.cash
        W = a.wealth_base

        # ---- Regra 1: short ban ----
        q_prime = q + delta_q
        if short_ban:
            if q_prime < 0:
                q_prime = 0.0
                delta_q = q_prime - q

        # ---- Regra 2: q_max ----
        if q_max != math.inf:
            if short_ban:
                # long-only
                q_prime = clip(q_prime, 0.0, q_max)
            else:
                # cap simétrico (permite short, mas limitado)
                q_prime = clip(q_prime, -q_max, q_max)

            delta_q = q_prime - q

        # ---- Regra 3: margem/leverage ----
        if L_max != math.inf:
            # |q'| <= (L_max * W) / P
            if W <= 0:
                # se wealth_base <= 0, força a reduzir exposição
                q_bound = 0.0
            else:
                q_bound = (L_max * W) / P_candidate

            q_prime = clip(q_prime, -q_bound, q_bound)
            delta_q = q_prime - q

        # ---- Regra 4: cash floor ----
        if C_min != -math.inf:
            # C' = C - delta_q*P - tau*|delta_q|*P >= C_min
            # Para buys (delta_q>0) pode restringir. Para sells, geralmente não.
            if delta_q > 0:
                max_buy = (C - C_min) / (P_candidate * (1.0 + tau)) if (P_candidate > 0) else 0.0
                if max_buy < 0:
                    max_buy = 0.0
                delta_q = min(delta_q, max_buy)

        return delta_q

    def _effective_delta_q(self, a, P_candidate: float) -> float:
        """
        Delta q desejado + regulação.
        """
        dq = self._desired_delta_q(a, P_candidate)
        return self._apply_policy(a, P_candidate, dq)

    # ----------------------------
    # Tâtonnement e execução
    # ----------------------------

    def _tatonnement_price(self) -> float:
        """
        Descoberta de preço por tâtonnement com passo limitado (numérico robusto):
        P^{k+1} = P^k * exp( eta * tanh( Z(P^k) / Q ) )

        - Mantém P>0 por construção
        - Impede overflow porque tanh(.) ∈ (-1, 1)
        - Continua a reagir ao excesso de procura, mas sem saltos absurdos
        """
        P = self.price
        Q = self.Q if self.Q > 0 else 1.0

        for _ in range(self.tatonnement_K):
            Z = 0.0
            for a in self.agents:
                Z += self._effective_delta_q(a, P)

            # passo limitado em log-preço
            delta_logP = self.tatonnement_eta * math.tanh(Z / Q)

            P = P * math.exp(delta_logP)

            if not math.isfinite(P) or P <= 0:
                return self.price  # fallback conservador

        return P

    def _execute_trades(self, P_new: float):
        """
        Executa trades ao preço final P_new.
        Atualiza (cash, shares) dos agentes, volume e inventory do specialist.
        """
        pol = self.policy
        tau = pol.tau

        total_delta = 0.0
        volume = 0.0

        # calcular e executar para cada agente
        for a in self.agents:
            dq_eff = self._effective_delta_q(a, P_new)

            fee = tau * abs(dq_eff) * P_new
            # atualizar posição
            a.shares += dq_eff
            # atualizar cash
            a.cash -= dq_eff * P_new + fee

            total_delta += dq_eff
            volume += abs(dq_eff)

        # Specialist absorve o desequilíbrio líquido
        # Se total_delta > 0, agentes compraram líquido -> specialist vendeu -> inventory diminui
        self.specialist_inventory -= total_delta

        self.volume = volume

    # ----------------------------
    # Step principal
    # ----------------------------

    def step(self):
        """
        Um step completo:

        A) Liquidação: cash*=1+r e recebe dividendos D_t
        B) Expectativas: agentes calculam mu_i,t
        C) Tâtonnement: encontra P_{t+1}
        D) Execução: trades ao preço P_{t+1} com política ativa
        E) Atualiza dividendos: gera D_{t+1}
        F) Atualiza métricas e recolhe dados
        """
        # guardar preço anterior para log return
        self.prev_price = self.price

        # A) liquidação
        self._settle_cash_and_dividends()

        # B) expectativas
        self.agents.shuffle_do("step")

        # C) descobre preço
        P_new = self._tatonnement_price()

        # D) executa trades
        self._execute_trades(P_new)

        # atualiza preço e histórico
        self.price = P_new
        self.price_history.append(self.price)
        if len(self.price_history) > (self.max_L + 1):
            self.price_history = self.price_history[-(self.max_L + 1):]

        # peak/drawdown
        if self.price > self.peak_price:
            self.peak_price = self.price

        # E) novo dividendo para o próximo período
        self.dividend = self.next_dividend()

        # step count e recolha
        self.step_count += 1
        self.datacollector.collect(self)

        if self.max_steps is not None and self.step_count >= int(self.max_steps):
            self.running = False
        
    def run(self, steps: int):
        for _ in range(int(steps)):
            if not getattr(self, "running", True):
                break
            self.step()
        return self.datacollector.get_model_vars_dataframe()
