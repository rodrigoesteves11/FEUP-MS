# FEUP-MS: Simulação de Mercado de Ações com Agentes Baseados em Modelos (ABM)

**Modelação e Simulação** — FEUP

Este projeto implementa um **modelo de simulação de mercado de ações baseado em agentes (ABM)** usando o framework **Mesa**. O modelo simula a dinâmica de preços num mercado com três tipos de agentes heterogéneos sob diferentes cenários de regulação.

---

**none (sem regulação)**
```
Volume alto, volatilidade “normal”, mispricing moderado; crashes e bolhas aparecem ocasionalmente.
Conclusão: baseline ativo e relativamente eficiente, mas com mais extremos.
```
**moderate** (τ=0.003, L_max=1.3)
```
↓ mispricing e ↓ bolhas (menos episódios e menor pico); volatilidade/crashes ~ iguais; volume ~ igual.
Conclusão: melhora ligeiramente eficiência e reduz extremos sem matar liquidez.
```
**excessive** (τ=0.01, L_max=1.0, short ban, q_max=2, C_min=0)
```
↓↓ volume (mercado quase parado), ↑↑ mispricing e bolhas grandes/persistentes; ↑ crashes e drawdowns mais profundos.
Conclusão: regulação excessiva destrói liquidez e piora a formação de preços.
```

---

## Índice

1. [Estrutura do Projeto](#estrutura-do-projeto)
2. [Arquitetura do Modelo](#arquitetura-do-modelo)
3. [Tipos de Agentes](#tipos-de-agentes)
4. [Processo de Descoberta de Preços (Tâtonnement)](#processo-de-descoberta-de-preços-tâtonnement)
5. [Processo de Dividendos](#processo-de-dividendos)
6. [Regras de Trading](#regras-de-trading)
7. [Políticas de Regulação](#políticas-de-regulação)
8. [Ciclo de Simulação (Step)](#ciclo-de-simulação-step)
9. [Métricas e KPIs](#métricas-e-kpis)
10. [Configuração e Execução](#configuração-e-execução)
11. [Visualização e Dashboards](#visualização-e-dashboards)
12. [Fórmulas Matemáticas Completas](#fórmulas-matemáticas-completas)

---

## Estrutura do Projeto

```
FEUP-MS/
├── agents.py           # Definição dos 3 tipos de agentes
├── model.py            # Modelo principal do mercado (Mesa)
├── run.py              # Runner de experiências (batch)
├── app.py              # Dashboard interativo (SolaraViz)
├── requirements.txt    # Dependências Python
└── README.md           # Este ficheiro
```

---

## Arquitetura do Modelo

O modelo usa o framework **Mesa** para simulação baseada em agentes. A arquitetura principal inclui:

### Classes Principais

| Ficheiro | Classe | Descrição |
|----------|--------|-----------|
| `agents.py` | `TraderAgent` | Classe base abstrata para todos os agentes |
| `agents.py` | `FundamentalistAgent` | Agente que faz mean-reversion ao preço fundamental |
| `agents.py` | `ChartistAgent` | Agente que segue tendências (momentum) |
| `agents.py` | `NoiseAgent` | Agente com expectativas ruidosas |
| `model.py` | `MarketModel` | Modelo principal do mercado |
| `model.py` | `PolicyParams` | Dataclass com parâmetros de regulação |

### Fluxo de Dados

```
                    ┌─────────────────┐
                    │   MarketModel   │
                    │                 │
                    │  • price (P_t)  │
                    │  • dividend     │
                    │  • policy       │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │Fundamentalist│   │   Chartist   │   │    Noise     │
   │    Agent     │   │    Agent     │   │    Agent     │
   │              │   │              │   │              │
   │  E[P_{t+1}]  │   │  E[P_{t+1}]  │   │  E[P_{t+1}]  │
   │  E[D_{t+1}]  │   │  E[D_{t+1}]  │   │  E[D_{t+1}]  │
   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Tâtonnement   │
                    │                 │
                    │   Z(P) → P_new  │
                    └─────────────────┘
```

---

## Tipos de Agentes

O modelo inclui **três tipos de agentes heterogéneos**, cada um com uma estratégia de formação de expectativas diferente.

### Estado de Cada Agente

Cada agente mantém o seguinte estado:

| Variável | Símbolo | Descrição |
|----------|---------|-----------|
| `cash` | $C_{i,t}$ | Dinheiro disponível |
| `shares` | $q_{i,t}$ | Número de ações detidas |
| `wealth_base` | $W_{i,t}$ | Riqueza base (fixada no início do step) |
| `gamma` | $\gamma_i$ | Coeficiente de aversão ao risco |
| `sigma2` | $\sigma^2_i$ | Variância percebida do payoff |
| `expected_price` | $E_t[P_{t+1}]$ | Preço esperado para t+1 |
| `expected_dividend` | $E_t[D_{t+1}]$ | Dividendo esperado para t+1 |
| `mu` | $\mu_i$ | Payoff esperado total |

### 1. Agente Fundamentalista (`FundamentalistAgent`)

O fundamentalista acredita que o preço converge para o **valor fundamental** e forma expectativas baseadas em mean-reversion.

**Expectativa de Preço:**
$$E_t[P_{t+1}] = P_t + \kappa_f \cdot (P^*_t - P_t)$$

**Expectativa de Dividendo:**
$$E_t[D_{t+1}] = \bar{d} + \rho \cdot (D_t - \bar{d})$$

Onde:
- $P_t$ = preço corrente
- $P^*_t$ = preço fundamental (valor presente dos dividendos)
- $\kappa_f \in [0.05, 0.20]$ = velocidade de convergência (heterogéneo)
- $\bar{d}$ = dividendo médio de longo prazo
- $\rho$ = persistência do processo AR(1) de dividendos

### 2. Agente Chartista (`ChartistAgent`)

O chartista segue **tendências de momentum** e extrapola movimentos passados de preço.

**Cálculo do Momentum (em log):**
$$m_t = \ln\left(\frac{P_t}{P_{t-L}}\right)$$

**Expectativa de Preço:**
$$E_t[P_{t+1}] = P_t \cdot \exp(\kappa_c \cdot m_t)$$

**Expectativa de Dividendo:**
$$E_t[D_{t+1}] = \bar{d}$$

Onde:
- $L \in \{5, 20, 60\}$ = janela de lookback (heterogéneo)
- $\kappa_c \in [0.5, 2.0]$ = intensidade do momentum (heterogéneo)
- Se $L > $ histórico disponível, retorna $P_t$ (sem sinal)

### 3. Agente Noise Trader (`NoiseAgent`)

O noise trader tem **expectativas ruidosas** que seguem um processo estocástico multiplicativo.

**Expectativa de Preço:**
$$E_t[P_{t+1}] = P_t \cdot \exp(\sigma_n \cdot \xi_t)$$

Onde:
- $\xi_t \sim \mathcal{N}(0, 1)$ = choque normal iid
- $\sigma_n \in [0.01, 0.05]$ = volatilidade do ruído (heterogéneo)

**Expectativa de Dividendo:**
$$E_t[D_{t+1}] = \bar{d}$$

---

## Processo de Descoberta de Preços (Tâtonnement)

O modelo usa um mecanismo de **tâtonnement Walrasiano** para descoberta de preços, sem livro de ordens explícito.

### Algoritmo

O specialist ajusta o preço iterativamente até encontrar um preço de equilíbrio:

```python
for k in range(K):
    Z = sum(delta_q_effective(agent, P) for agent in agents)
    delta_logP = eta * tanh(Z / Q)
    P = P * exp(delta_logP)
```

### Fórmula de Ajuste de Preço

$$P^{(k+1)} = P^{(k)} \cdot \exp\left(\eta \cdot \tanh\left(\frac{Z(P^{(k)})}{Q}\right)\right)$$

Onde:
- $P^{(k)}$ = preço na iteração $k$
- $Z(P)$ = excesso de procura agregado ao preço $P$
- $Q$ = número total de ações em circulação
- $\eta$ = tamanho do passo (default: 0.2)
- $\tanh(\cdot)$ = função tangente hiperbólica (limita o passo a $[-1, 1]$)

### Propriedades do Tâtonnement

1. **Estabilidade numérica**: O uso de $\tanh$ garante que o ajuste está limitado
2. **Preservação de positividade**: $P > 0$ sempre (por construção exponencial)
3. **Convergência**: Tende para $Z(P) \approx 0$ em equilíbrio

---

## Processo de Dividendos

Os dividendos seguem um processo **AR(1) estacionário**.

### Processo AR(1)

$$D_{t+1} = \bar{d} + \rho \cdot (D_t - \bar{d}) + \sigma_d \cdot \varepsilon_t$$

Onde:
- $\bar{d} = 1.0$ = dividendo médio de longo prazo
- $\rho = 0.95$ = persistência (autocorrelação)
- $\sigma_d = 0.15$ = volatilidade dos choques
- $\varepsilon_t \sim \mathcal{N}(0, 1)$ = ruído branco
- Truncagem: $D_{t+1} = \max(0, D_{t+1})$

### Preço Fundamental

O preço fundamental é o **valor presente** do fluxo de dividendos AR(1):

$$P^*_t = \frac{\bar{d}}{r} + \frac{\rho}{1 + r - \rho} \cdot (D_t - \bar{d})$$

Onde:
- $r = 0.05$ = taxa de juro sem risco

Esta fórmula deriva da solução analítica do valor presente de um AR(1) infinito.

---

## Regras de Trading

### Payoff Esperado

O payoff esperado de deter uma ação é:

$$\mu_i = E_t[P_{t+1}] + E_t[D_{t+1}]$$

### Ordem Desejada

A ordem desejada baseia-se numa **posição-alvo** derivada de utilidade esperada com aversão ao risco:

**1. Diferença de Payoff:**
$$\Delta\mu_i(P) = \mu_i - (1 + r) \cdot P$$

**2. Sinal Normalizado:**
$$s_i = \frac{\beta \cdot \Delta\mu_i}{\gamma_i \cdot \sigma^2_i}$$

**3. Exposição Alvo (em euros):**
$$x^*_i = W_{i,t} \cdot \tanh(s_i)$$

**4. Posição Alvo (em ações):**
$$q^*_i = \frac{x^*_i}{P}$$

**5. Ordem Desejada:**
$$\Delta q_i = \phi \cdot (q^*_i - q_i)$$

Onde:
- $\beta = 1.0$ = sensibilidade ao sinal
- $\phi = 0.2$ = velocidade de ajuste (partial adjustment)
- $\tanh(\cdot)$ = limita exposição a $[-W, +W]$

### Interpretação

- Se $\mu_i > (1+r)P$: ação está subvalorizada → comprar
- Se $\mu_i < (1+r)P$: ação está sobrevalorizada → vender
- Agentes com maior $\gamma_i$ (mais avessos ao risco) → ordens menores
- Agentes com maior $\sigma^2_i$ (mais incerteza) → ordens menores

---

## Políticas de Regulação

O modelo implementa **três cenários de regulação fixa**:

### Parâmetros de Política

| Parâmetro | Símbolo | Descrição |
|-----------|---------|-----------|
| `tau` | $\tau$ | Taxa de transação (proporcional) |
| `L_max` | $L_{max}$ | Limite de alavancagem |
| `short_ban` | - | Proibição de vendas a descoberto |
| `q_max` | $q_{max}$ | Posição máxima (ações) |
| `C_min` | $C_{min}$ | Cash mínimo obrigatório |

### Cenários Definidos

| Cenário | $\tau$ | $L_{max}$ | Short Ban | $q_{max}$ | $C_{min}$ |
|---------|--------|-----------|-----------|-----------|-----------|
| **none** | 0.0 | $\infty$ | Não | $\infty$ | $-\infty$ |
| **moderate** | 0.003 (0.3%) | 1.3 | Não | $\infty$ | $-\infty$ |
| **excessive** | 0.01 (1%) | 1.0 | Sim | 2.0 | 0.0 |

### Aplicação das Regras (por ordem)

As regras são aplicadas **sequencialmente** à ordem desejada $\Delta q$:

**Regra 1: Short Ban**
```
Se short_ban ativo:
    q' = max(0, q + Δq)
    Δq = q' - q
```

**Regra 2: Posição Máxima**
$$q' = \text{clip}(q + \Delta q, -q_{max}, q_{max})$$

**Regra 3: Limite de Alavancagem**
$$|q'| \leq \frac{L_{max} \cdot W_{base}}{P}$$

**Regra 4: Cash Floor**
$$C' = C - \Delta q \cdot P - \tau \cdot |\Delta q| \cdot P \geq C_{min}$$

Para compras ($\Delta q > 0$):
$$\Delta q \leq \frac{C - C_{min}}{P \cdot (1 + \tau)}$$

---

## Ciclo de Simulação (Step)

Cada step da simulação segue esta sequência:

```
┌──────────────────────────────────────────────────────────────┐
│                        STEP t → t+1                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  A) LIQUIDAÇÃO                                               │
│     • C_i ← C_i × (1 + r)           [juros sobre cash]       │
│     • C_i ← C_i + q_i × D_t         [recebe dividendos]      │
│     • W_base ← C_i + q_i × P_t      [fixa wealth base]       │
│                                                              │
│  B) EXPECTATIVAS                                             │
│     • Cada agente calcula:                                   │
│       - E_t[P_{t+1}]                                         │
│       - E_t[D_{t+1}]                                         │
│       - μ_i = E_t[P_{t+1}] + E_t[D_{t+1}]                    │
│                                                              │
│  C) TÂTONNEMENT                                              │
│     • Para k = 1, ..., K:                                    │
│       - Z(P) = Σ Δq_effective(agent, P)                      │
│       - P ← P × exp(η × tanh(Z/Q))                           │
│                                                              │
│  D) EXECUÇÃO DE TRADES                                       │
│     • Para cada agente:                                      │
│       - Δq_eff = aplicar_política(Δq_desejado)               │
│       - q_i ← q_i + Δq_eff                                   │
│       - C_i ← C_i - Δq_eff × P - τ × |Δq_eff| × P            │
│     • Specialist absorve desequilíbrio líquido               │
│     • Volume = Σ |Δq_eff|                                    │
│                                                              │
│  E) ATUALIZAÇÃO DE DIVIDENDOS                                │
│     • D_{t+1} = d̄ + ρ(D_t - d̄) + σ_d × ε                    │
│                                                              │
│  F) ATUALIZAÇÃO DE MÉTRICAS                                  │
│     • Histórico de preços                                    │
│     • Peak price / Drawdown                                  │
│     • DataCollector.collect()                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Métricas e KPIs

### Métricas Recolhidas por Step (Model Reporters)

| Métrica | Fórmula | Descrição |
|---------|---------|-----------|
| `Price` | $P_t$ | Preço corrente |
| `Dividend` | $D_t$ | Dividendo corrente |
| `FundamentalPrice` | $P^*_t$ | Preço fundamental |
| `Mispricing` | $P_t - P^*_t$ | Desvio do fundamental |
| `BubbleRatio` | $P_t / P^*_t$ | Rácio de bolha |
| `LogReturn` | $\ln(P_t / P_{t-1})$ | Retorno logarítmico |
| `Volume` | $\sum_i |\Delta q_i|$ | Volume de trading |
| `Turnover` | $\text{Volume} / Q$ | Volume normalizado |
| `Drawdown` | $(P_t - P_{peak}) / P_{peak}$ | Queda desde o pico |
| `GiniWealth` | Coef. Gini | Desigualdade de riqueza |
| `GiniWealthDisc` | Gini descontado | Gini com $W/(1+r)^t$ |
| `TotalWealth` | $\sum_i (C_i + q_i \cdot P_t)$ | Riqueza total |

### KPIs Calculados por Run (após burn-in)

| KPI | Descrição |
|-----|-----------|
| `vol_mean` | Volatilidade média (rolling std de log returns) |
| `vol_max` | Volatilidade máxima |
| `mean_abs_mispricing` | Mispricing absoluto médio |
| `mean_abs_rel_mispricing` | Mispricing relativo médio $|P/P^* - 1|$ |
| `volume_mean` | Volume médio |
| `turnover_mean` | Turnover médio |
| `gini_mean` | Gini médio |
| `gini_final` | Gini no final da simulação |
| `max_drawdown` | Drawdown máximo |
| `n_crashes_ret` | Nº de crashes (retorno < -k × vol) |
| `n_crashes_dd` | Nº de crashes (drawdown < threshold) |
| `n_bubbles` | Nº de episódios de bolha |
| `bubble_peak` | Rácio máximo de bolha |
| `bubble_dur_mean` | Duração média de bolhas |

### Definição de Eventos

**Crash (por retorno):**
$$\text{crash}_t = \mathbb{1}\left[\text{logret}_t < -k \cdot \sigma_{rolling}\right]$$

Com $k = 2.0$ e janela de 20 períodos.

**Crash (por drawdown):**
$$\text{crash}_t = \mathbb{1}\left[\text{drawdown}_t < -0.25\right]$$

**Bolha:**
$$\text{bolha}_t = \mathbb{1}\left[\frac{P_t}{P^*_t} > 1.5\right]$$

Contam-se episódios consecutivos de pelo menos 10 períodos.

---

## Configuração e Execução

### Instalação

```bash
python -m venv .venv

source .venv/Scripts/activate

source .venv/bin/activate

pip install -r requirements.txt
```

### Dependências

```
mesa[rec]
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
```

### Configuração Padrão

```python
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
```

### Execução

```bash
python run.py
```

A experiência corre:
- 3 cenários de política (none, moderate, excessive)
- 30 seeds por cenário
- 50 steps por simulação (5 de burn-in)

### Outputs

```
results_new_model/
├── kpi_results.csv                    # Tabela de KPIs
├── kpi_boxplots_by_policy.png         # Boxplots comparativos
├── run_policy=none_seed=1.png         # Gráficos detalhados
├── run_policy=moderate_seed=1.png
└── run_policy=excessive_seed=1.png
```

---

## Visualização e Dashboards

### Dashboard Interativo (SolaraViz)

```bash
python run.py

solara run app.py
```

O dashboard tem **duas páginas**:

| Página | Descrição |
|--------|-----------|
| **Live Monitor** | Visualização em tempo real da simulação com gráficos de preço, mispricing, volume, drawdown, Gini e radar de risco |
| **Aggregate** | Análise comparativa das 3 políticas com radar multi-policy e boxplots dos KPIs |

**Funcionalidades do Live Monitor:**
- Gráfico de preço vs fundamental com gap especulativo
- Radar de risco compacto (Volatilidade, Crash, Iliquidez, Gini)
- Painel de informação com estado atual e parâmetros da política
- Controlo de play/pause e seleção de política/seed

---

## Fórmulas Matemáticas Completas

### Resumo de Notação

| Símbolo | Descrição |
|---------|-----------|
| $P_t$ | Preço no período $t$ |
| $P^*_t$ | Preço fundamental |
| $D_t$ | Dividendo no período $t$ |
| $C_{i,t}$ | Cash do agente $i$ |
| $q_{i,t}$ | Shares do agente $i$ |
| $W_{i,t}$ | Riqueza do agente $i$ |
| $\mu_i$ | Payoff esperado |
| $\gamma_i$ | Aversão ao risco |
| $\sigma^2_i$ | Variância percebida |
| $r$ | Taxa de juro sem risco |
| $\tau$ | Taxa de transação |
| $Q$ | Total de ações |
| $Z(P)$ | Excesso de procura |

### Processo Completo

**1. Dividendo AR(1):**
$$D_{t+1} = \bar{d} + \rho(D_t - \bar{d}) + \sigma_d \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)$$

**2. Preço Fundamental:**
$$P^*_t = \frac{\bar{d}}{r} + \frac{\rho}{1+r-\rho}(D_t - \bar{d})$$

**3. Expectativas (por tipo):**

| Tipo | $E_t[P_{t+1}]$ | $E_t[D_{t+1}]$ |
|------|----------------|----------------|
| Fundamentalista | $P_t + \kappa_f(P^*_t - P_t)$ | $\bar{d} + \rho(D_t - \bar{d})$ |
| Chartista | $P_t \exp(\kappa_c \ln(P_t/P_{t-L}))$ | $\bar{d}$ |
| Noise | $P_t \exp(\sigma_n \xi_t)$ | $\bar{d}$ |

**4. Ordem Desejada:**
$$\Delta q_i = \phi \cdot \left(\frac{W_i \tanh\left(\frac{\beta(\mu_i - (1+r)P)}{\gamma_i \sigma^2_i}\right)}{P} - q_i\right)$$

**5. Tâtonnement:**
$$P^{(k+1)} = P^{(k)} \exp\left(\eta \tanh\left(\frac{Z(P^{(k)})}{Q}\right)\right)$$

**6. Execução:**
$$q_i \leftarrow q_i + \Delta q^{eff}_i$$
$$C_i \leftarrow C_i - \Delta q^{eff}_i \cdot P - \tau |\Delta q^{eff}_i| P$$

**7. Liquidação (início do step seguinte):**
$$C_i \leftarrow C_i(1+r) + q_i D_t$$

---

## Referências

- **Mesa Framework**: https://mesa.readthedocs.io/
- **Artificial Stock Markets**: LeBaron, B. (2006). "Agent-based Computational Finance"
- **Tâtonnement**: Walras, L. (1874). "Elements of Pure Economics"

---

## Licença

Este projeto foi desenvolvido para fins académicos no âmbito da unidade curricular de Modelação e Simulação da FEUP.