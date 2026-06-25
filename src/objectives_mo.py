"""Funções-objetivo do problema de portfólio — Checkpoint 2 (multiobjetivo).

Estende o Checkpoint 1 (mono-objetivo, risco-retorno) com o terceiro
componente ESG, reativando o que já estava isolado e pronto em
`src/esg_cp2.py`. Define o vetor bruto de 3 objetivos em forma de
MINIMIZAÇÃO:

    f1(w) = -mu^T w     retorno esperado (negado)
    f2(w) = w^T Sigma w risco (variância do portfólio)
    f3(w) = -e^T w      score ESG (negado)

Este módulo NÃO modifica nem importa de forma circular nada do CP1: reusa
`f1_retorno`/`f2_risco`/`normalizar`/`amostrar_carteiras_aleatorias` de
`src.objectives` (genéricas em número de componentes) e `f3_esg` de
`src.esg_cp2`, sem alterar nenhum dos dois arquivos.

IMPORTANTE — escala dos objetivos: assim como no CP1 (ver docstring de
`src.objectives`), f1 e f2 têm escalas muito diferentes entre si, e agora
f3 (ESG negado, em [-100, 0]) tem uma escala ainda maior. Sem normalizar,
NSGA-II e SPEA-II ficariam dominados pelo objetivo de maior magnitude: o
SPEA-II usa distância euclidiana no espaço de objetivos para estimar
densidade (a vizinhança seria decidida quase só por f3), e o hipervolume
seria quase inteiramente explicado pela dimensão de maior escala,
tornando comparações entre algoritmos/sementes pouco informativas. Por
isso normalizamos (z-score) os 3 componentes ANTES de entrar no
algoritmo — exatamente a mesma estratégia do CP1, só que estendida para 3
dimensões. As métricas e figuras de relatório, no entanto, sempre usam os
valores BRUTOS (recalculados a partir de w), nunca os normalizados
internos do pymoo — mesma separação que o CP1 já faz entre
`g_escalarizado` (normalizado) e `src.metrics` (unidades originais).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data_loader import DadosMercado
from src.esg_cp2 import carregar_esg_real, f3_esg, gerar_esg_sintetico
from src.objectives import amostrar_carteiras_aleatorias, f1_retorno, f2_risco, normalizar


def avaliar_componentes_mo(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, esg: np.ndarray) -> np.ndarray:
    """Vetor (f1, f2, f3) bruto (não normalizado) dos 3 objetivos do Checkpoint 2."""
    return np.array([f1_retorno(w, mu), f2_risco(w, sigma), f3_esg(w, esg)])


@dataclass
class EstatisticasNormalizacaoMO:
    """Média e desvio-padrão de (f1, f2, f3) sobre uma amostra de referência."""

    media: np.ndarray  # shape (3,)
    desvio: np.ndarray  # shape (3,)


def calcular_estatisticas_normalizacao_mo(
    mu: np.ndarray,
    sigma: np.ndarray,
    esg: np.ndarray,
    w_min: float,
    w_max: float,
    n_amostras: int,
    semente: int,
) -> EstatisticasNormalizacaoMO:
    """Estima média/desvio de (f1, f2, f3) sobre uma amostra de carteiras aleatórias factíveis.

    Mesma ideia de `src.objectives.calcular_estatisticas_normalizacao`, mas
    com o vetor de objetivos de 3 componentes — reusa
    `amostrar_carteiras_aleatorias` (genérica em N) em vez de duplicá-la.
    """
    n_ativos = len(mu)
    carteiras = amostrar_carteiras_aleatorias(n_amostras, n_ativos, w_min, w_max, semente)
    componentes = np.array([avaliar_componentes_mo(w, mu, sigma, esg) for w in carteiras])

    media = componentes.mean(axis=0)
    desvio = componentes.std(axis=0, ddof=0)
    desvio[desvio < 1e-12] = 1.0

    return EstatisticasNormalizacaoMO(media=media, desvio=desvio)


@dataclass
class ConfigObjetivoMO:
    """Agrupa tudo que o problema multiobjetivo precisa além de w."""

    w_min: float
    w_max: float
    esg: np.ndarray  # vetor e sintético, shape (N,)
    stats: EstatisticasNormalizacaoMO
    ref_point: np.ndarray  # ponto de referência fixo para o hipervolume, shape (3,)


def construir_config_objetivo_mo(dados: DadosMercado, config: dict[str, Any]) -> ConfigObjetivoMO:
    """Lê config.yaml (seções `esg_cp2`, `objetivo_mo`, `restricoes`, `hipervolume`) e monta o ConfigObjetivoMO.

    Centraliza essa construção para que optimizer_moea.py, baselines_mo.py
    e metrics_mo.py usem exatamente o mesmo vetor ESG, a mesma normalização
    e o mesmo ponto de referência do hipervolume.
    """
    cfg_esg = config["esg_cp2"]
    cfg_obj_mo = config["objetivo_mo"]
    cfg_r = config["restricoes"]
    cfg_hv = config["hipervolume"]

    fonte = cfg_esg.get("fonte_esg", "ise")
    if fonte == "ise":
        esg = carregar_esg_real(
            tickers=dados.tickers,
            score_min=cfg_esg["score_min"],
            score_max=cfg_esg["score_max"],
        )
        print(f"Scores ESG (ISE B3): min={esg.min():.1f}, média={esg.mean():.1f}, max={esg.max():.1f}")
    else:
        esg = gerar_esg_sintetico(
            tickers=dados.tickers,
            semente_esg=cfg_esg["semente_esg"],
            score_min=cfg_esg["score_min"],
            score_max=cfg_esg["score_max"],
        )
        print(f"Scores ESG (sintético): min={esg.min():.1f}, média={esg.mean():.1f}, max={esg.max():.1f}")

    stats = calcular_estatisticas_normalizacao_mo(
        mu=dados.mu,
        sigma=dados.sigma,
        esg=esg,
        w_min=cfg_r["w_min"],
        w_max=cfg_r["w_max"],
        n_amostras=cfg_obj_mo["n_amostras_normalizacao"],
        semente=cfg_obj_mo["semente_normalizacao"],
    )

    # Objetivos normalizados têm média 0 e desvio 1 por construção (z-score),
    # logo um ponto de referência fixo k desvios-padrão acima da média (pior
    # que praticamente toda carteira factível) é o mesmo em qualquer escala
    # original — não depende de mu/Sigma/esg desta execução específica.
    k = cfg_hv["k_desvios_ref_point"]
    ref_point = np.full(3, k)

    return ConfigObjetivoMO(
        w_min=cfg_r["w_min"],
        w_max=cfg_r["w_max"],
        esg=esg,
        stats=stats,
        ref_point=ref_point,
    )


def f_objetivos_normalizados(w: np.ndarray, dados: DadosMercado, cfg_obj_mo: ConfigObjetivoMO) -> np.ndarray:
    """(f1, f2, f3) bruto -> normalizado (z-score). É isto que o pymoo recebe como `out["F"]`."""
    f = avaliar_componentes_mo(w, dados.mu, dados.sigma, cfg_obj_mo.esg)
    return normalizar(f, cfg_obj_mo.stats)
