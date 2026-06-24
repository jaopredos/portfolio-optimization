"""Métricas de avaliação multiobjetivo — Checkpoint 2.

Reusa as métricas genéricas do CP1 (`src.metrics`) que não dependem de
escalarização (retorno, risco, Sharpe) e a métrica de score ESG já isolada
em `src.esg_cp2`. Não existe aqui um equivalente a `g`/`gap_relativo`: em
multiobjetivo não há uma escalarização única, então o "gap ao ótimo" do
CP1 é substituído por métricas de qualidade de fronteira (hipervolume,
contagem de não-dominados) e pela checagem direta de dominância usada para
comparar o ponto single-objective com a fronteira multiobjetivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

from src.data_loader import DadosMercado
from src.esg_cp2 import score_esg_carteira
from src.metrics import indice_sharpe, retorno_esperado, risco_volatilidade


def calcular_metricas_carteira_mo(
    w: np.ndarray,
    dados: DadosMercado,
    esg: np.ndarray,
    taxa_livre_risco: float = 0.0,
) -> dict[str, float]:
    """Métricas em unidades originais de uma carteira: retorno, risco, Sharpe e score ESG."""
    return {
        "retorno": retorno_esperado(w, dados.mu),
        "risco": risco_volatilidade(w, dados.sigma),
        "sharpe": indice_sharpe(w, dados.mu, dados.sigma, taxa_livre_risco),
        "score_esg": score_esg_carteira(w, esg),
    }


def matriz_metricas_mo(
    X: np.ndarray,
    dados: DadosMercado,
    esg: np.ndarray,
    taxa_livre_risco: float = 0.0,
) -> pd.DataFrame:
    """Tabela de métricas em unidades originais, uma linha por carteira de X (shape (n, N)).

    Usada tanto para exportar CSVs quanto para alimentar as figuras de
    `src/plots_mo.py` — em unidades fáceis de ler (retorno %, risco como
    volatilidade, score ESG 0-100), nunca os objetivos normalizados
    internos do pymoo.
    """
    linhas = [calcular_metricas_carteira_mo(w, dados, esg, taxa_livre_risco) for w in X]
    return pd.DataFrame(linhas)


def calcular_hipervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
    """Hipervolume do conjunto de objetivos (normalizados) F, com `ref_point` fixo."""
    return float(HV(ref_point=ref_point)(F))


def contar_nao_dominadas(F: np.ndarray) -> int:
    """Número de soluções não-dominadas em F."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    return len(NonDominatedSorting().do(F, only_non_dominated_front=True))


def is_dominado(ponto: np.ndarray, fronteira: np.ndarray, tol: float = 1e-12) -> bool:
    """True se algum ponto de `fronteira` domina `ponto` (minimização).

    Domina: todo componente <= ao de `ponto` e ao menos um componente
    estritamente menor (com tolerância numérica `tol`).
    """
    menor_ou_igual = np.all(fronteira <= ponto + tol, axis=1)
    estritamente_menor = np.any(fronteira < ponto - tol, axis=1)
    return bool(np.any(menor_ou_igual & estritamente_menor))
