"""Métricas de avaliação de uma carteira: retorno, risco, Sharpe e gap.

Note a diferença entre essas métricas e os componentes f1/f2 de
`objectives.py`: f1/f2 estão em forma de minimização e normalizados
(z-score), pensados para a escalarização g(w). As métricas aqui são em
unidades originais e de fácil leitura (ex. retorno como fração anual),
pensadas para relatório e tabelas comparativas.

A métrica de score ESG (`src/esg_cp2.py::score_esg_carteira`) não é
calculada aqui — o Checkpoint 1 é mono-objetivo (risco-retorno).
"""

from __future__ import annotations

import numpy as np

from src.data_loader import DadosMercado


def retorno_esperado(w: np.ndarray, mu: np.ndarray) -> float:
    """Retorno esperado anualizado da carteira: mu^T w."""
    return float(mu @ w)


def risco_volatilidade(w: np.ndarray, sigma: np.ndarray) -> float:
    """Volatilidade (desvio-padrão) anualizada da carteira: sqrt(w^T Sigma w)."""
    return float(np.sqrt(w @ sigma @ w))


def indice_sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, taxa_livre_risco: float = 0.0) -> float:
    """Índice de Sharpe anualizado: (retorno - taxa_livre_risco) / risco."""
    risco = risco_volatilidade(w, sigma)
    if risco < 1e-12:
        return 0.0
    return (retorno_esperado(w, mu) - taxa_livre_risco) / risco


def gap_relativo(g_metodo: float, g_otimo: float, eps: float = 1e-8) -> float:
    """Gap relativo de g_metodo frente ao ótimo exato (cvxpy): (g - g*) / |g*|.

    Como g_otimo é o mínimo global do QP convexo, g_metodo >= g_otimo
    sempre (a menos de ruído numérico), logo gap_relativo >= 0: quanto
    maior, pior o método em relação ao ótimo exato.
    """
    denom = abs(g_otimo) if abs(g_otimo) > eps else eps
    return (g_metodo - g_otimo) / denom


def calcular_metricas_carteira(
    w: np.ndarray,
    dados: DadosMercado,
    g: float,
    g_otimo: float,
    taxa_livre_risco: float = 0.0,
) -> dict[str, float]:
    """Calcula o dicionário de métricas de uma carteira para uma linha da tabela de resultados."""
    return {
        "g": g,
        "retorno": retorno_esperado(w, dados.mu),
        "risco": risco_volatilidade(w, dados.sigma),
        "sharpe": indice_sharpe(w, dados.mu, dados.sigma, taxa_livre_risco),
        "gap_relativo": gap_relativo(g, g_otimo),
    }
