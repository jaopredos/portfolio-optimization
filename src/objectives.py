"""Funções-objetivo do problema de portfólio — Checkpoint 1 (Markowitz mono-objetivo).

Define os dois componentes em forma de MINIMIZAÇÃO:
    f1(w) = -mu^T w     retorno esperado (negado: minimizar f1 = maximizar retorno)
    f2(w) = w^T Sigma w risco (variância do portfólio)

e a escalarização por soma ponderada usada no Checkpoint 1:
    g(w) = lambda1 * f1_norm(w) + lambda2 * f2_norm(w)

Esta é a leitura clássica de Markowitz: minimizar g(w) equivale a maximizar
a utilidade média-variância `mu^T w - delta * w^T Sigma w`, com
`delta = lambda2/lambda1` funcionando como coeficiente de aversão ao risco
(a menos do fator de escala introduzido pela normalização — ver abaixo).

IMPORTANTE — escala dos objetivos: f1 é da ordem de retornos anuais (ex.
~0.05 a 0.40) e f2 é uma variância (tipicamente bem menor, ~0.01 a 0.04).
Somar f1 e f2 diretamente com pesos lambda faria o termo de menor magnitude
(o risco) ser quase irrelevante na prática, independente dos lambdas
escolhidos. Por isso normalizamos cada f_i por z-score ANTES de combinar:
estimamos média e desvio-padrão de cada f_i sobre uma amostra grande de
carteiras factíveis aleatórias (capped-simplex) e usamos essas estatísticas
para colocar f1 e f2 em escalas comparáveis (média ~0, desvio ~1). Os pesos
lambda1/lambda2 do config.yaml então de fato controlam a importância
relativa de cada objetivo, em vez de apenas compensar diferenças de escala.

O terceiro objetivo do projeto final (ESG, `f3(w) = -e^T w`) é reservado
para o Checkpoint 2 (extensão multiobjetivo) e está isolado em
`src/esg_cp2.py` — nada neste módulo depende dele.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.constraints import projetar_capped_simplex
from src.data_loader import DadosMercado


def f1_retorno(w: np.ndarray, mu: np.ndarray) -> float:
    """f1(w) = -mu^T w (retorno esperado, negado para minimização)."""
    return float(-mu @ w)


def f2_risco(w: np.ndarray, sigma: np.ndarray) -> float:
    """f2(w) = w^T Sigma w (variância do portfólio)."""
    return float(w @ sigma @ w)


def avaliar_componentes(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Vetor (f1, f2) bruto (não normalizado) do Checkpoint 1 (risco-retorno)."""
    return np.array([f1_retorno(w, mu), f2_risco(w, sigma)])


@dataclass
class EstatisticasNormalizacao:
    """Média e desvio-padrão de (f1, f2) sobre uma amostra de referência."""

    media: np.ndarray  # shape (2,)
    desvio: np.ndarray  # shape (2,)


def amostrar_carteiras_aleatorias(
    n_amostras: int,
    n_ativos: int,
    w_min: float,
    w_max: float,
    semente: int,
) -> np.ndarray:
    """Amostra `n_amostras` carteiras factíveis i.i.d. para fins de calibração/baseline.

    Gera vetores gaussianos i.i.d. e os projeta no capped-simplex (mesma
    função de reparo usada pelo DE), garantindo que toda amostra devolvida
    já é uma carteira factível. Usada tanto para estimar as estatísticas de
    normalização (`calcular_estatisticas_normalizacao`) quanto pelo baseline
    de random search (`src/baselines.py`).

    Returns:
        Matriz (n_amostras, n_ativos) de carteiras factíveis.
    """
    rng = np.random.default_rng(semente)
    candidatos = rng.normal(size=(n_amostras, n_ativos))
    return np.array([projetar_capped_simplex(c, w_min, w_max) for c in candidatos])


def calcular_estatisticas_normalizacao(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_min: float,
    w_max: float,
    n_amostras: int,
    semente: int,
) -> EstatisticasNormalizacao:
    """Estima média/desvio de f1, f2 sobre uma amostra de carteiras aleatórias factíveis.

    Essas estatísticas são fixas (calculadas uma única vez por execução, com
    semente própria `semente_normalizacao` no config) e usadas para normalizar
    f1, f2 antes da escalarização — ver docstring do módulo.
    """
    n_ativos = len(mu)
    carteiras = amostrar_carteiras_aleatorias(n_amostras, n_ativos, w_min, w_max, semente)
    componentes = np.array([avaliar_componentes(w, mu, sigma) for w in carteiras])

    media = componentes.mean(axis=0)
    desvio = componentes.std(axis=0, ddof=0)
    # Evita divisão por zero no caso degenerado de um componente constante.
    desvio[desvio < 1e-12] = 1.0

    return EstatisticasNormalizacao(media=media, desvio=desvio)


def normalizar(f: np.ndarray, stats: EstatisticasNormalizacao) -> np.ndarray:
    """Normaliza (z-score) o vetor de componentes f usando as estatísticas de referência."""
    return (f - stats.media) / stats.desvio


@dataclass
class ConfigObjetivo:
    """Agrupa tudo que `g_escalarizado` precisa além de w: lambdas, limites e normalização."""

    lambdas: np.ndarray  # (lambda1, lambda2)
    w_min: float
    w_max: float
    stats: EstatisticasNormalizacao


def construir_config_objetivo(dados: DadosMercado, config: dict[str, Any]) -> ConfigObjetivo:
    """Lê config.yaml e monta o ConfigObjetivo (lambdas + estatísticas de normalização).

    Centraliza essa construção para que optimizer_de.py, baselines.py e
    metrics.py usem exatamente a mesma normalização e os mesmos lambdas.
    """
    cfg_obj = config["objetivo"]
    cfg_r = config["restricoes"]

    lambdas = np.array([cfg_obj["lambda1"], cfg_obj["lambda2"]])
    stats = calcular_estatisticas_normalizacao(
        mu=dados.mu,
        sigma=dados.sigma,
        w_min=cfg_r["w_min"],
        w_max=cfg_r["w_max"],
        n_amostras=cfg_obj["n_amostras_normalizacao"],
        semente=cfg_obj["semente_normalizacao"],
    )

    return ConfigObjetivo(lambdas=lambdas, w_min=cfg_r["w_min"], w_max=cfg_r["w_max"], stats=stats)


def g_escalarizado(w: np.ndarray, dados: DadosMercado, cfg_obj: ConfigObjetivo) -> float:
    """g(w) = lambdas . normalizar((f1, f2)). Objetivo mono-objetivo do Checkpoint 1."""
    f = avaliar_componentes(w, dados.mu, dados.sigma)
    f_norm = normalizar(f, cfg_obj.stats)
    return float(cfg_obj.lambdas @ f_norm)
