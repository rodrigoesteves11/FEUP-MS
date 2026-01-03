"""
Agentes do modelo de mercado financeiro.
"""

import mesa
import numpy as np


class TraderAgent(mesa.Agent):
    """Classe base para todos os traders."""
    pass


class FundamentalistAgent(TraderAgent):
    """Agente que decide com base no valor fundamental (dividendos descontados)."""
    pass


class ChartistAgent(TraderAgent):
    """Agente que usa análise técnica (médias móveis, momentum, contrarian)."""
    pass


class NoiseAgent(TraderAgent):
    """Agente com comportamento aleatório."""
    pass

