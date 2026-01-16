# FEUP-MS: Stock Market Simulation (ABM)

Agent-based model (ABM) for stock market simulation using the **Mesa** framework. Simulates price dynamics with three heterogeneous agent types under different regulatory policies.

## Structure

```
├── agents.py       # 3 agent types
├── model.py        # Main market model (Mesa)
├── app.py          # Interactive dashboard (SolaraViz)
├── run.py          # Batch experiments
└── run_mixes.py    # Agent composition analysis
```

## Installation

```bash
pip install -r requirements.txt
python run.py
solara run app.py
```

## Architecture

```
                    ┌─────────────────┐
                    │   MarketModel   │
                    │                 │
                    │  • price        │
                    │  • dividend     │
                    │  • policy       │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │Fundamentalist│   │   Chartist   │   │    Noise     │
   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Tâtonnement   │
                    │  (equilibrium)  │
                    └─────────────────┘
```

## Agents

| Type | Strategy |
|------|----------|
| **Fundamentalist** | Forms price expectations based on mean-reversion towards the fundamental value, derived from discounted future dividends |
| **Chartist** | Employs technical analysis by extrapolating historical price momentum to forecast future price movements |
| **Noise Trader** | Generates stochastic price expectations through random perturbations, representing uninformed or irrational market participants |

## Configuration

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