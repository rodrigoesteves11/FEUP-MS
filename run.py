"""
Script principal para executar a simulação.
"""

from model import MarketModel


if __name__ == "__main__":
    model = MarketModel()
    model.run(100)

