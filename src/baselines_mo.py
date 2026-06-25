"""Baselines de comparação para NSGA-II/SPEA-II — Checkpoint 2 (multiobjetivo).

Dois baselines, no mesmo espírito do CP1 (`src/baselines.py`), adaptados
para um cenário em que não existe um único "melhor" escalar:

1. `carteira_uniforme_mo`: a carteira 1/N (não depende de cfg_obj — os
   objetivos dela são calculados depois, como qualquer outra carteira).

2. `rodar_random_search_mo`: amostra carteiras factíveis aleatórias (mesmo
   mecanismo de projeção capped-simplex do CP1) com o MESMO orçamento de
   avaliações que os MOEAs consumiram, mas devolve TODAS as carteiras
   amostradas — não só a "melhor" (que não existe em multiobjetivo). O
   subconjunto não-dominado é extraído depois com `filtrar_nao_dominadas`,
   para comparação justa com a fronteira do NSGA-II/SPEA-II.

`filtrar_nao_dominadas` também é reusada para agregar as soluções de
várias sementes de um mesmo algoritmo em uma única fronteira final.
"""

from __future__ import annotations

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.data_loader import DadosMercado
from src.objectives import amostrar_carteiras_aleatorias
from src.objectives_mo import ConfigObjetivoMO, f_objetivos_normalizados


def carteira_uniforme_mo(dados: DadosMercado) -> np.ndarray:
    """Baseline 1/N: pesos iguais entre todos os ativos."""
    n = len(dados.tickers)
    return np.full(n, 1.0 / n)


def rodar_random_search_mo(
    dados: DadosMercado,
    cfg_obj_mo: ConfigObjetivoMO,
    n_avaliacoes: int,
    semente: int,
) -> np.ndarray:
    """Amostra `n_avaliacoes` carteiras factíveis e devolve todas (shape (n_avaliacoes, N)).

    Usa o mesmo orçamento de avaliações que os MOEAs consumiram
    (`pop_size * n_gen`) para que a comparação isole o efeito da busca
    guiada por dominância/diversidade frente a uma amostragem às ciegas.
    """
    n = len(dados.tickers)
    return amostrar_carteiras_aleatorias(n_avaliacoes, n, cfg_obj_mo.w_min, cfg_obj_mo.w_max, semente)


def filtrar_nao_dominadas(X: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduz (X, F) ao subconjunto não-dominado de F.

    Usado tanto para extrair a fronteira do random search/1N quanto para
    agregar as soluções de várias sementes de um mesmo MOEA em uma única
    fronteira final (união das populações, depois filtrada).
    """
    indices = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return X[indices], F[indices]


def avaliar_carteiras_mo(X: np.ndarray, dados: DadosMercado, cfg_obj_mo: ConfigObjetivoMO) -> np.ndarray:
    """Objetivos normalizados (f1,f2,f3) de cada linha de X, shape (len(X), 3)."""
    return np.array([f_objetivos_normalizados(w, dados, cfg_obj_mo) for w in X])
