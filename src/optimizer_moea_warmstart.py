"""MOEA com Warm-Start: injeta pesos do DE mono-objetivo na população inicial.

Variação do `src/optimizer_moea.py` que aceita uma `initial_population`
customizada via `pymoo.operators.sampling.lhs.LHS` substituído por
`pymoo.core.sampling.Sampling` personalizado. Os pesos carregados de
`results/melhores_pesos_de.csv` (melhor carteira por semente do CP1)
preenchem de 1 a 5 slots da população inicial; o restante é gerado
aleatoriamente (mesmo mecanismo padrão do pymoo).

A ideia é que a solução SO do DE, que já maximiza o Sharpe, forneça um
ponto de partida de alta qualidade no espaço de objetivos, acelerando a
convergência inicial do MOEA na região de interesse (alto retorno, baixo
risco) — verificável pela curva de hipervolume nas primeiras gerações.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.data_loader import DadosMercado
from src.objectives_mo import ConfigObjetivoMO
from src.optimizer_de import RepairCappedSimplex
from src.optimizer_moea import (
    ALGORITMOS_MOEA,
    HistoricoConvergenciaMO,
    ProblemaPortfolioMO,
    ResultadoMOEA,
)

RAIZ_PROJETO = Path(__file__).resolve().parent.parent


class SamplingWarmStart(Sampling):
    """Sampling customizado que injeta pesos pré-definidos nos primeiros slots.

    Os demais indivíduos são amostrados uniformemente na caixa [xl, xu] do
    problema (comportamento padrão do pymoo antes do reparo).
    """

    def __init__(self, pesos_iniciais: np.ndarray) -> None:
        super().__init__()
        self.pesos_iniciais = np.atleast_2d(pesos_iniciais)

    def _do(self, problem: Any, n_samples: int, **kwargs: Any) -> np.ndarray:
        n_injetados = min(len(self.pesos_iniciais), n_samples)
        rng = np.random.default_rng()

        X = rng.uniform(
            low=problem.xl,
            high=problem.xu,
            size=(n_samples, problem.n_var),
        )
        X[:n_injetados] = self.pesos_iniciais[:n_injetados]
        return X


def carregar_pesos_de(
    caminho: str | Path | None = None,
    max_individuos: int = 10,
) -> np.ndarray:
    """Carrega os melhores pesos do DE mono-objetivo de `melhores_pesos_de.csv`.

    Args:
        caminho: caminho para o CSV. Se None, usa o padrão em `results/`.
        max_individuos: número máximo de indivíduos a injetar (1 a 10).

    Returns:
        Matriz (min(n_sementes, max_individuos), n_ativos) com os pesos.

    Raises:
        FileNotFoundError: se o CSV não existir.
        ValueError: se o CSV estiver vazio ou com formato inesperado.
    """
    if caminho is None:
        caminho = RAIZ_PROJETO / "results" / "melhores_pesos_de.csv"
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de pesos do DE não encontrado: {caminho}. "
            "Execute `python -m src.run_experiment` antes."
        )

    df = pd.read_csv(caminho)

    if df.empty:
        raise ValueError(f"CSV de pesos do DE está vazio: {caminho}")

    colunas_pesos = [c for c in df.columns if c != "semente"]
    if not colunas_pesos:
        raise ValueError(f"Nenhuma coluna de peso encontrada em {caminho}")

    pesos = df[colunas_pesos].to_numpy()
    n = min(len(pesos), max_individuos)
    return pesos[:n]


def rodar_moea_warmstart(
    nome_algoritmo: str,
    dados: DadosMercado,
    cfg_obj_mo: ConfigObjetivoMO,
    config: dict[str, Any],
    semente: int,
    pesos_iniciais: np.ndarray,
) -> ResultadoMOEA:
    """Executa MOEA com warm-start: mesma assinatura de `rodar_moea`, mais `pesos_iniciais`.

    Args:
        nome_algoritmo: "NSGA2" ou "SPEA2".
        dados: mu, sigma e tickers.
        cfg_obj_mo: vetor ESG, normalização e ref_point.
        config: dicionário de config.yaml.
        semente: semente desta execução.
        pesos_iniciais: pesos a injetar na população inicial (1 a 5 linhas).
    """
    if nome_algoritmo not in ALGORITMOS_MOEA:
        raise ValueError(
            f"Algoritmo desconhecido: {nome_algoritmo!r}. "
            f"Use um de {list(ALGORITMOS_MOEA)}."
        )

    chave_config = f"otimizador_{nome_algoritmo.lower()}"
    cfg_moea = config[chave_config]

    problema = ProblemaPortfolioMO(dados, cfg_obj_mo)
    reparo = RepairCappedSimplex(w_min=cfg_obj_mo.w_min, w_max=cfg_obj_mo.w_max)
    callback = HistoricoConvergenciaMO(ref_point=cfg_obj_mo.ref_point)

    sampling = SamplingWarmStart(pesos_iniciais)

    AlgoritmoCls = ALGORITMOS_MOEA[nome_algoritmo]
    algoritmo = AlgoritmoCls(
        pop_size=cfg_moea["pop_size"],
        sampling=sampling,
        repair=reparo,
        eliminate_duplicates=cfg_moea.get("eliminate_duplicates", True),
    )

    termination = get_termination("n_gen", cfg_moea["n_gen"])

    res = minimize(
        problema,
        algoritmo,
        termination,
        seed=semente,
        verbose=False,
        callback=callback,
        save_history=False,
    )

    n_avaliacoes = int(res.algorithm.evaluator.n_eval)

    return ResultadoMOEA(
        algoritmo=nome_algoritmo,
        semente=semente,
        X_final=np.asarray(callback.data["X_final"]),
        F_final=np.asarray(callback.data["F_final"]),
        X_inicial=np.asarray(callback.data["X_inicial"]),
        F_inicial=np.asarray(callback.data["F_inicial"]),
        historico_hv=np.array(callback.data["hv"]),
        n_avaliacoes=n_avaliacoes,
    )
