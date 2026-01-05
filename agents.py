"""
Agentes do modelo de mercado financeiro.
"""

import mesa


class TraderAgent(mesa.Agent):
    """
    Classe base para todos os traders.
    
    Atributos:
        wealth (float): Riqueza total W_{i,t}
        shares (float): Quantidade de ações detidas q_{i,t}
        desired_shares (float): Quantidade de ações desejada (antes do clearing)
        risk_aversion (float): Coeficiente de aversão ao risco α_i
        perceived_variance (float): Variância percebida σ²_i
    """
    
    def __init__(self, model, initial_wealth, risk_aversion, perceived_variance):
        super().__init__(model)
        self.wealth = initial_wealth
        self.initial_wealth = initial_wealth  # Para escalar demanda
        self.shares = 0.0
        self.risk_aversion = risk_aversion
        self.perceived_variance = perceived_variance
        
        # Expectativas (calculadas a cada step)
        self.expected_price = 0.0
        self.expected_dividend = 0.0
        self.expected_payoff = 0.0  # Cache para evitar recálculo
    
    def compute_expected_price(self):
        """Calcula E_t[P_{t+1}] - implementado nas subclasses."""
        raise NotImplementedError
    
    def compute_expected_dividend(self):
        """Calcula E_t[D_{t+1}] - implementado nas subclasses."""
        raise NotImplementedError
    
    def compute_expected_payoff(self):
        """
        Retorna o payoff esperado (cached): E_t[P_{t+1} + D_{t+1}]
        """
        return self.expected_payoff
    
    def compute_demand(self, price):
        """
        Procura de ações (média-variância):
        q_{i,t} = (E_t[P_{t+1} + D_{t+1}] - (1+r) * P_t) / (γ_i * σ²_i)
        
        onde σ²_i = é a variância do payoff em níveis P_{t+1} + D_{t+1} (não em %), ou seja, na mesma unidade do preço.
        
        Args:
            price: Preço atual P_t para calcular a demanda
            
        Returns:
            Quantidade de ações desejada (pode ser negativa = short)
        """
        r = self.model.r
        expected_payoff = self.compute_expected_payoff()
        
        numerator = expected_payoff - (1 + r) * price
        denominator = self.risk_aversion * self.perceived_variance
        
        if denominator <= 0:
            return 0.0
        
        raw_demand = numerator / denominator
        
        # Escalar demanda com riqueza relativa à média (evita mudança de regime com r>0)
        mean_wealth = sum(a.wealth for a in self.model.agents) / len(self.model.agents)
        wealth_scale = self.wealth / mean_wealth if mean_wealth > 0 else 1.0
        
        return raw_demand * wealth_scale
    
    def step(self):
        """
        Passo do agente:
        1. Calcular expectativas de preço e dividendo
        2. Cachear payoff esperado
        """
        self.expected_price = self.compute_expected_price()
        self.expected_dividend = self.compute_expected_dividend()
        self.expected_payoff = self.expected_price + self.expected_dividend
    
    def update_wealth(self, old_price):
        """
        Atualiza a riqueza após o clearing:
        
        W_{i,t+1} = (1+r) * (W_{i,t} - q_{i,t} * P_t) + q_{i,t} * (P_{t+1} + D_{t+1})
        
        onde:
        - W_{i,t} - q_{i,t} * P_t = cash investido no ativo sem risco
        - q_{i,t} * (P_{t+1} + D_{t+1}) = valor das ações + dividendos
        
        Args:
            old_price: Preço P_t usado na compra das ações
        """
        P_new = self.model.price
        D_new = self.model.dividend
        r = self.model.r
        
        # Cash que foi para o ativo sem risco (pode ser negativo se short)
        cash_invested = self.wealth - self.shares * old_price
        
        # Riqueza do ativo sem risco (sempre capitaliza)
        wealth_from_bonds = cash_invested * (1 + r)
        
        # Riqueza das ações (preço novo + dividendo)
        wealth_from_stocks = self.shares * (P_new + D_new)
        
        # Nova riqueza total
        self.wealth = wealth_from_bonds + wealth_from_stocks
        
        # Warning se riqueza negativa
        if self.wealth < 0:
            print(f"WARNING: Agent {self.unique_id} has negative wealth: {self.wealth:.2f}")


class FundamentalistAgent(TraderAgent):
    """
    Agente Fundamentalista.
    
    Calcula o valor fundamental da ação usando dividendos descontados.
    
    Parâmetros:
        rho (float): Persistência dos dividendos ρ
    """
    
    def __init__(self, model, initial_wealth, risk_aversion, perceived_variance, rho):
        super().__init__(model, initial_wealth, risk_aversion, perceived_variance)
        self.rho = rho

    def compute_expected_dividend(self):
        d_bar = self.model.d_bar
        D_t = self.model.dividend
        return d_bar + self.rho * (D_t - d_bar)

    def fundamental_price_from_dividend(self, D):
        r = self.model.r
        d_bar = self.model.d_bar
        rho = self.rho
        B = rho / (1 + r - rho)
        return (d_bar / r) + B * (D - d_bar)

    def compute_expected_price(self):
        # Isto é E_t[P_{t+1}] consistente com AR(1)
        E_D_next = self.compute_expected_dividend()
        return self.fundamental_price_from_dividend(E_D_next)


class ChartistAgent(TraderAgent):
    """
    Agente Chartista (Análise Técnica).
    
    Seleciona entre 3 regras com base no erro histórico:
    - Trend Following: extrapola a tendência
    - Momentum: projeta a última variação
    - Contrarian: aposta na reversão à média
    
    Parâmetros:
        L (int): Janela da média móvel
        chi (float): Parâmetro de reação χ
        lambda_ (float): Fator de memória para erros λ
    """
    
    def __init__(self, model, initial_wealth, risk_aversion, perceived_variance, L, chi, lambda_):
        super().__init__(model, initial_wealth, risk_aversion, perceived_variance)
        self.L = L
        self.chi = chi
        self.lambda_ = lambda_
        
        # Erros acumulados U_{j,t} para cada regra
        self.U_errors = {
            'trend': 0.0,
            'momentum': 0.0,
            'contrarian': 0.0
        }
        
        # Previsões anteriores E_{t-1,j}[P_t] para calcular erros
        self.previous_predictions = {
            'trend': None,
            'momentum': None,
            'contrarian': None
        }
        
        # Regra inicial aleatória (evita enviesamento no arranque)
        self.selected_rule = self.model.random.choice(['trend', 'momentum', 'contrarian'])
    
    def compute_moving_average(self):
        """
        Passo A: MA_L = (1/L) * Σ P_{t-k} para k=0..L-1
        """
        history = self.model.price_history
        if len(history) < self.L:
            return self.model.price
        
        recent_prices = history[-self.L:]
        return sum(recent_prices) / self.L
    
    def compute_all_predictions(self):
        """
        Calcula as previsões das 3 regras para o próximo período.
        
        Returns:
            dict com previsões para cada regra
        """
        P_t = self.model.price
        MA_L = self.compute_moving_average()
        
        # Preço anterior (P_{t-1})
        history = self.model.price_history
        P_prev = history[-2] if len(history) >= 2 else P_t
        
        return {
            'trend': P_t + self.chi * (P_t - MA_L),       # Trend Following
            'momentum': P_t + self.chi * (P_t - P_prev),  # Momentum
            'contrarian': P_t - self.chi * (P_t - MA_L)   # Contrarian
        }
    
    def update_rule_errors(self):
        """
        Passo B: Atualiza os erros acumulados de cada regra.
        U_{j,t} = (1-λ) * U_{j,t-1} + λ * (P_t - E_{t-1,j}[P_t])²
        """
        P_t = self.model.price
        
        for rule in self.U_errors.keys():
            prev_pred = self.previous_predictions[rule]
            
            if prev_pred is not None:
                prediction_error = (P_t - prev_pred) ** 2
                self.U_errors[rule] = (
                    (1 - self.lambda_) * self.U_errors[rule] + 
                    self.lambda_ * prediction_error
                )
    
    def select_best_rule(self):
        """
        Seleciona a regra com menor erro acumulado: min(U_{j,t})
        """
        return min(self.U_errors, key=self.U_errors.get)
    
    def compute_expected_price(self):
        """
        Passo C: Calcula E_t[P_{t+1}] usando a melhor regra.
        """
        # Atualizar erros com base no preço atual
        self.update_rule_errors()
        
        # Calcular previsões de todas as regras
        predictions = self.compute_all_predictions()
        
        # Guardar previsões para o próximo período
        self.previous_predictions = predictions.copy()
        
        # Selecionar a melhor regra
        self.selected_rule = self.select_best_rule()
        
        return predictions[self.selected_rule]
    
    def compute_expected_dividend(self):
        """
        Chartistas assumem E_t[D_{t+1}] = d̄ (média histórica).
        """
        return self.model.d_bar


class NoiseAgent(TraderAgent):
    """
    Agente Noise Trader.
    
    Faz previsões com ruído aleatório, simulando
    trading emocional ou necessidades de liquidez.
    
    Parâmetros:
        sigma_n (float): Desvio-padrão do ruído σ_N
    """
    
    def __init__(self, model, initial_wealth, risk_aversion, perceived_variance, sigma_n):
        super().__init__(model, initial_wealth, risk_aversion, perceived_variance)
        self.sigma_n = sigma_n
    
    def compute_expected_price(self):
        """
        E_t[P_{t+1}] = P_t + ε_t
        onde ε_t ~ N(0, σ²_N)
        """
        P_t = self.model.price
        epsilon = self.model.random.gauss(0, self.sigma_n)
        
        return P_t + epsilon
    
    def compute_expected_dividend(self):
        """
        Noise traders assumem E_t[D_{t+1}] = d̄ (média histórica).
        """
        return self.model.d_bar


