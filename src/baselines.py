"""Baselines de comparação para o Differential Evolution (DE).

Três baselines, conforme especificado no Checkpoint 1:

1. `otimo_exato_cvxpy`: g(w) escalarizado é uma função quadrática convexa em
   w (f1 é linear, f2 é uma forma quadrática com Sigma semidefinida
   positiva, e a normalização z-score é uma transformação afim) sob
   restrições lineares (capped-simplex). Logo o problema é um QP de
   média-variância convexo e pode ser resolvido EXATAMENTE com cvxpy, dando
   o ótimo global verdadeiro — a referência para medir o "gap" do DE e do
   random search.

2. `carteira_uniforme`: a carteira 1/N (todos os pesos iguais), baseline
   clássico e surpreendentemente difícil de superar na literatura de
   alocação de portfólio.

3. `rodar_random_search`: amostra carteiras factíveis aleatórias (mesmo
   mecanismo de projeção capped-simplex usado pelo DE) com o MESMO
   orçamento de avaliações de função do DE, para isolar o efeito da busca
   guiada (diferencial) do DE frente a uma busca às cegas com o mesmo custo
   computacional.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from src.data_loader import DadosMercado
from src.objectives import ConfigObjetivo, amostrar_carteiras_aleatorias, g_escalarizado


@dataclass
class ResultadoBaseline:
    """Carteira encontrada por um baseline e o valor de g associado."""

    w: np.ndarray
    g: float


def otimo_exato_cvxpy(dados: DadosMercado, cfg_obj: ConfigObjetivo) -> ResultadoBaseline:
    """Resolve exatamente o QP de média-variância min g(w) s.a. capped-simplex, via cvxpy.

    Reescreve g(w) = lambda . normalizar((f1, f2)) diretamente em termos de
    expressões cvxpy (não reaproveita `g_escalarizado` porque essa função
    opera sobre arrays numpy concretos; aqui precisamos de uma expressão
    SIMBÓLICA para o solver). `cp.psd_wrap` é usado em Sigma para evitar que
    pequenas assimetrias numéricas de ponto flutuante façam o cvxpy rejeitar
    a forma quadrática como não-convexa (DCP).
    """
    n = len(dados.tickers)
    w = cp.Variable(n)

    media, desvio = cfg_obj.stats.media, cfg_obj.stats.desvio

    f1 = -dados.mu @ w
    f2 = cp.quad_form(w, cp.psd_wrap(dados.sigma))

    f1_norm = (f1 - media[0]) / desvio[0]
    f2_norm = (f2 - media[1]) / desvio[1]

    lam1, lam2 = cfg_obj.lambdas
    g = lam1 * f1_norm + lam2 * f2_norm

    restricoes = [
        cp.sum(w) == 1,
        w >= cfg_obj.w_min,
        w <= cfg_obj.w_max,
    ]

    problema = cp.Problem(cp.Minimize(g), restricoes)
    problema.solve()

    if problema.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"cvxpy não encontrou ótimo: status={problema.status}")

    return ResultadoBaseline(w=np.asarray(w.value).flatten(), g=float(problema.value))


def carteira_uniforme(dados: DadosMercado, cfg_obj: ConfigObjetivo) -> ResultadoBaseline:
    """Baseline 1/N: pesos iguais entre todos os ativos."""
    n = len(dados.tickers)
    w = np.full(n, 1.0 / n)
    return ResultadoBaseline(w=w, g=g_escalarizado(w, dados, cfg_obj))


def rodar_random_search(
    dados: DadosMercado,
    cfg_obj: ConfigObjetivo,
    n_avaliacoes: int,
    semente: int,
) -> ResultadoBaseline:
    """Random search: amostra `n_avaliacoes` carteiras factíveis e devolve a melhor.

    Usa o mesmo orçamento de avaliações de g que o DE consumiu (ver
    `ResultadoDE.n_avaliacoes` em optimizer_de.py) para que a comparação
    DE vs random search isole o efeito da busca guiada, não do orçamento.
    """
    n = len(dados.tickers)
    candidatos = amostrar_carteiras_aleatorias(n_avaliacoes, n, cfg_obj.w_min, cfg_obj.w_max, semente)
    valores = np.array([g_escalarizado(w, dados, cfg_obj) for w in candidatos])
    idx = int(valores.argmin())
    return ResultadoBaseline(w=candidatos[idx], g=float(valores[idx]))
