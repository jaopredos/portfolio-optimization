"""Carregamento de dados de mercado e construção dos parâmetros do modelo.

Este módulo é responsável por:
  1. Baixar (ou ler do cache local) preços históricos via yfinance;
  2. Calcular retornos diários simples e estimar mu (retorno esperado anual)
     e Sigma (matriz de covariância anual), com opção de shrinkage de
     Ledoit-Wolf.

O Checkpoint 1 é mono-objetivo (risco-retorno): este módulo não carrega o
vetor ESG. O gerador sintético de ESG (placeholder para o Checkpoint 2) fica
isolado em `src/esg_cp2.py`, fora deste fluxo.

Tudo que é "mágico" (datas, tickers, flags) vem do config.yaml — este
módulo não tem nenhum valor hard-coded de negócio.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf


@dataclass
class DadosMercado:
    """Agrega tudo que o problema de otimização (CP1, risco-retorno) precisa sobre os ativos."""

    tickers: list[str]
    mu: np.ndarray          # retorno esperado anualizado, shape (N,)
    sigma: np.ndarray       # matriz de covariância anualizada, shape (N, N)
    retornos_diarios: pd.DataFrame  # usado por baselines/diagnóstico, não pelo otimizador


def _resolver_datas(data_fim: str, anos_historico: int) -> tuple[str, str]:
    """Resolve a janela [data_inicio, data_fim] usada para baixar preços.

    `data_fim` pode ser a string especial "today" (resolvida para a data de
    execução) ou uma data explícita "YYYY-MM-DD". Mantemos essa opção
    "today" porque o yfinance reflete o pregão real: fixar uma data de
    término distante no passado/futuro tornaria o experimento incoerente
    com o mercado real. A reprodutibilidade da OTIMIZAÇÃO (DE, baselines)
    não depende dessa escolha — é garantida pelas sementes; apenas os dados
    de entrada (mu, Sigma) variam com a data de execução, o que é esperado
    e documentado no README.
    """
    if data_fim == "today":
        fim = datetime.today()
    else:
        fim = datetime.strptime(data_fim, "%Y-%m-%d")
    inicio = fim - timedelta(days=int(365.25 * anos_historico))
    return inicio.strftime("%Y-%m-%d"), fim.strftime("%Y-%m-%d")


def _caminho_cache(cache_dir: str | Path, tickers: list[str], inicio: str, fim: str) -> Path:
    """Nome de arquivo de cache determinístico a partir dos parâmetros de download."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    chave = "_".join(t.replace(".", "") for t in sorted(tickers))
    nome = f"precos_{chave}_{inicio}_{fim}.csv"
    return cache_dir / nome


def baixar_precos(
    tickers: list[str],
    data_fim: str,
    anos_historico: int,
    cache_dir: str | Path = "data_cache",
    usar_cache: bool = True,
) -> pd.DataFrame:
    """Baixa preços de fechamento ajustado para os tickers informados.

    Usa um cache local em CSV para evitar rebaixar os mesmos dados em toda
    execução (yfinance pode ser lento/instável para listas grandes de
    tickers).

    Args:
        tickers: lista de tickers no formato aceito pelo yfinance (ex. "PETR4.SA").
        data_fim: "today" ou data explícita "YYYY-MM-DD".
        anos_historico: tamanho da janela histórica, em anos.
        cache_dir: diretório de cache.
        usar_cache: se True, lê do cache quando disponível.

    Returns:
        DataFrame (datas x tickers) com preços de fechamento ajustado.
    """
    inicio, fim = _resolver_datas(data_fim, anos_historico)
    caminho = _caminho_cache(cache_dir, tickers, inicio, fim)

    if usar_cache and caminho.exists():
        precos = pd.read_csv(caminho, index_col=0, parse_dates=True)
        # Garante que o cache cobre exatamente os tickers pedidos.
        if set(tickers).issubset(set(precos.columns)):
            return precos[tickers]

    dados_brutos = yf.download(
        tickers,
        start=inicio,
        end=fim,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(dados_brutos.columns, pd.MultiIndex):
        precos = dados_brutos["Close"]
    else:
        # Caso degenerado: um único ticker baixado sem MultiIndex.
        precos = dados_brutos[["Close"]]
        precos.columns = tickers

    precos = precos[tickers].dropna(how="all")
    # Preenche pequenas lacunas (feriados que não coincidem entre ativos),
    # mas não inventa dados onde um ativo nunca negociou.
    precos = precos.ffill().dropna(how="any")

    if usar_cache:
        precos.to_csv(caminho)

    return precos


def calcular_retornos_diarios(precos: pd.DataFrame) -> pd.DataFrame:
    """Retornos diários simples R_t = P_t / P_{t-1} - 1."""
    return precos.pct_change().dropna(how="any")


def estimar_mu_sigma(
    retornos_diarios: pd.DataFrame,
    pregoes_por_ano: int = 252,
    usar_ledoit_wolf: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estima mu (retorno esperado) e Sigma (covariância), anualizados.

    mu é estimado pela média amostral dos retornos diários e anualizado por
    `pregoes_por_ano`. Sigma é estimado pela covariância amostral dos
    retornos diários (ou pelo estimador de shrinkage de Ledoit-Wolf, que
    reduz o ruído de estimação quando N de ativos é grande em relação ao
    número de observações) e anualizado multiplicando por
    `pregoes_por_ano` (escala linear da covariância sob retornos i.i.d.).

    Args:
        retornos_diarios: DataFrame (datas x tickers) de retornos diários.
        pregoes_por_ano: fator de anualização (tipicamente 252).
        usar_ledoit_wolf: se True, usa LedoitWolf().fit(...) em vez da
            covariância amostral simples.

    Returns:
        (mu, sigma) anualizados, com mu.shape == (N,) e sigma.shape == (N, N).
    """
    mu_diario = retornos_diarios.mean().to_numpy()
    mu = mu_diario * pregoes_por_ano

    if usar_ledoit_wolf:
        sigma_diaria = LedoitWolf().fit(retornos_diarios.to_numpy()).covariance_
    else:
        sigma_diaria = np.cov(retornos_diarios.to_numpy(), rowvar=False)

    sigma = sigma_diaria * pregoes_por_ano
    return mu, sigma


def carregar_dados(config: dict[str, Any]) -> DadosMercado:
    """Função de alto nível: lê o config e devolve tudo que o problema (CP1) precisa.

    Args:
        config: dicionário carregado de config.yaml (ver src/utils.carregar_config).

    Returns:
        DadosMercado com tickers, mu e sigma alinhados pelo mesmo índice.
    """
    cfg_dados = config["dados"]

    tickers = list(cfg_dados["tickers"])

    precos = baixar_precos(
        tickers=tickers,
        data_fim=cfg_dados["data_fim"],
        anos_historico=cfg_dados["anos_historico"],
        cache_dir=cfg_dados.get("cache_dir", "data_cache"),
        usar_cache=cfg_dados.get("usar_cache", True),
    )
    retornos = calcular_retornos_diarios(precos)
    mu, sigma = estimar_mu_sigma(
        retornos,
        pregoes_por_ano=cfg_dados.get("pregoes_por_ano", 252),
        usar_ledoit_wolf=cfg_dados.get("usar_ledoit_wolf", True),
    )

    return DadosMercado(tickers=tickers, mu=mu, sigma=sigma, retornos_diarios=retornos)
