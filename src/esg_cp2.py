"""# CP2 — não usado no Checkpoint 1.

Este módulo isola TUDO que é específico do terceiro objetivo (ESG) do
projeto, para que o Checkpoint 1 (mono-objetivo, risco-retorno) fique
completamente livre de ESG — formulação, função objetivo, baselines,
métricas e figuras do CP1 não importam nada deste arquivo.

Ele existe para que o trabalho já feito em torno do ESG (a função f3, o
gerador sintético do vetor `e` e a métrica de score ESG) não seja perdido:
é exatamente o que o Checkpoint 2 (extensão multiobjetivo, NSGA-II) vai
reativar, junto com `objectives.py::avaliar_componentes`, que hoje
devolve só `(f1, f2)` mas pode voltar a devolver `(f1, f2, f3)`
incorporando as funções abaixo.

Esboço de como isso se encaixaria no CP2 (não executado aqui):
    from src.objectives import f1_retorno, f2_risco
    from src.esg_cp2 import f3_esg, gerar_esg_sintetico

    esg = gerar_esg_sintetico(dados.tickers, semente_esg=2024)
    vetor_objetivos = np.array([f1_retorno(w, dados.mu), f2_risco(w, dados.sigma), f3_esg(w, esg)])
    # -> entra no NSGA-II como os 3 objetivos a minimizar, sem escalarização.
"""

from __future__ import annotations

import numpy as np


def f3_esg(w: np.ndarray, esg: np.ndarray) -> float:
    """f3(w) = -e^T w (score ESG da carteira, negado para minimização).

    # CP2 — não usado no Checkpoint 1 (que é mono-objetivo risco-retorno).
    """
    return float(-esg @ w)


def gerar_esg_sintetico(
    tickers: list[str],
    semente_esg: int = 2024,
    score_min: float = 0.0,
    score_max: float = 100.0,
) -> np.ndarray:
    """Gera um vetor ESG sintético, reprodutível e documentado como placeholder.

    # CP2 — não usado no Checkpoint 1.

    *** ATENÇÃO: PLACEHOLDER ***
    Não há, neste projeto, integração com uma fonte real de dados ESG
    (ex.: Refinitiv, MSCI ESG, Sustainalytics). Para exercitar a futura
    extensão multiobjetivo (Checkpoint 2), geramos scores sintéticos com um
    gerador de números aleatórios SEMEADO (`np.random.default_rng`), de
    forma que o vetor `e` seja sempre o mesmo entre execuções. Ver
    DECISOES.md para a justificativa e o plano de substituição por dados
    reais.

    Args:
        tickers: lista de tickers (define N e a ordem do vetor retornado).
        semente_esg: semente do gerador.
        score_min: limite inferior do score sintético.
        score_max: limite superior do score sintético.

    Returns:
        Vetor `e` de scores ESG sintéticos, shape (N,), em [score_min, score_max].
    """
    rng = np.random.default_rng(semente_esg)
    return rng.uniform(score_min, score_max, size=len(tickers))


def score_esg_carteira(w: np.ndarray, esg: np.ndarray) -> float:
    """Score ESG médio (ponderado pelos pesos) da carteira, em [score_min, score_max].

    # CP2 — não usado no Checkpoint 1. Métrica de relatório (sinal positivo,
    # ao contrário de f3_esg que é negado para minimização).
    """
    return float(esg @ w)
