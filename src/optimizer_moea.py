"""Metaheurísticas multiobjetivo do Checkpoint 2: NSGA-II e SPEA-II (pymoo).

Mesmo padrão de três peças do CP1 (`src/optimizer_de.py`), agora para um
problema de 3 objetivos:

1. `ProblemaPortfolioMO` (subclasse de `pymoo.core.problem.Problem`):
   espaço de busca igual ao do CP1 (`n_var` = N ativos, caixa
   `[w_min, w_max]`), mas `n_obj=3` — devolve os objetivos JÁ
   NORMALIZADOS (`f_objetivos_normalizados`, ver `src/objectives_mo.py`),
   nunca os brutos, pelo mesmo motivo de escala explicado naquele módulo.

2. Reparo: reusamos `RepairCappedSimplex` de `src/optimizer_de.py` SEM
   modificação. `NSGA2`/`SPEA2` herdam de
   `pymoo.algorithms.base.genetic.GeneticAlgorithm`, cujo construtor aceita
   `repair=...` e o propaga tanto para `Initialization` (população inicial)
   quanto para `Mating` (toda prole gerada por geração) — exatamente o
   mesmo mecanismo que `DE(repair=...)` usa no CP1. Confirmado lendo o
   código-fonte do pymoo instalado (0.6.1.6).

3. `HistoricoConvergenciaMO` (subclasse de `pymoo.core.callback.Callback`):
   grava, a cada geração, o hipervolume (HV) da população corrente — não
   há um "melhor" escalar único em multiobjetivo, então HV faz o papel que
   o "best-so-far" de `g` fazia no CP1. Também guarda `F`/`X` da PRIMEIRA
   geração notificada (população inicial, já reparada — a "fronteira de
   Pareto inicial" pedida no enunciado) e da ÚLTIMA (fronteira final).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.data_loader import DadosMercado
from src.objectives_mo import ConfigObjetivoMO, f_objetivos_normalizados
from src.optimizer_de import RepairCappedSimplex

ALGORITMOS_MOEA = {"NSGA2": NSGA2, "SPEA2": SPEA2}


class ProblemaPortfolioMO(Problem):
    """Problema do pymoo: minimizar (f1,f2,f3) normalizados na caixa [w_min,w_max]^N.

    Assim como `ProblemaPortfolio` (CP1), a restrição soma(w)=1 não é
    declarada via `n_eq_constr` — é tratada por reparo (`RepairCappedSimplex`),
    garantindo 100% de factibilidade nos indivíduos avaliados.
    """

    def __init__(self, dados: DadosMercado, cfg_obj_mo: ConfigObjetivoMO) -> None:
        self.dados = dados
        self.cfg_obj_mo = cfg_obj_mo
        n_ativos = len(dados.tickers)
        super().__init__(
            n_var=n_ativos,
            n_obj=3,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=cfg_obj_mo.w_min,
            xu=cfg_obj_mo.w_max,
        )

    def _evaluate(self, X: np.ndarray, out: dict[str, Any], *args, **kwargs) -> None:
        valores = np.array([f_objetivos_normalizados(w, self.dados, self.cfg_obj_mo) for w in X])
        out["F"] = valores


class HistoricoConvergenciaMO(Callback):
    """Registra o hipervolume por geração e captura as populações inicial e final.

    O `ref_point` é fixo (vem de `ConfigObjetivoMO.ref_point`, calculado uma
    única vez a partir das estatísticas de normalização) para que o HV seja
    comparável entre gerações, sementes e algoritmos.
    """

    def __init__(self, ref_point: np.ndarray) -> None:
        super().__init__()
        self.hv_indicator = HV(ref_point=ref_point)
        self.data["hv"] = []
        self.data["F_inicial"] = None
        self.data["X_inicial"] = None
        self.data["F_final"] = None
        self.data["X_final"] = None

    def notify(self, algorithm) -> None:
        F = algorithm.pop.get("F")
        X = algorithm.pop.get("X")

        if self.data["F_inicial"] is None:
            self.data["F_inicial"] = F.copy()
            self.data["X_inicial"] = X.copy()

        self.data["hv"].append(float(self.hv_indicator(F)))
        self.data["F_final"] = F.copy()
        self.data["X_final"] = X.copy()


@dataclass
class ResultadoMOEA:
    """Saída de uma execução de um MOEA (NSGA-II ou SPEA-II) para uma única semente."""

    algoritmo: str
    semente: int
    X_final: np.ndarray  # população final (pesos), shape (pop_size, N)
    F_final: np.ndarray  # objetivos normalizados finais, shape (pop_size, 3)
    X_inicial: np.ndarray  # população inicial reparada, shape (pop_size, N)
    F_inicial: np.ndarray  # objetivos normalizados iniciais, shape (pop_size, 3)
    historico_hv: np.ndarray  # hipervolume por geração, shape (n_gen,)
    n_avaliacoes: int


def rodar_moea(
    nome_algoritmo: str,
    dados: DadosMercado,
    cfg_obj_mo: ConfigObjetivoMO,
    config: dict[str, Any],
    semente: int,
) -> ResultadoMOEA:
    """Executa NSGA-II ou SPEA-II (pymoo) para uma semente e devolve o resultado.

    Args:
        nome_algoritmo: "NSGA2" ou "SPEA2".
        dados: mu, sigma e tickers (ver src/data_loader.py).
        cfg_obj_mo: vetor ESG, normalização e ref_point (ver src/objectives_mo.py).
        config: dicionário de config.yaml (lê `otimizador_nsga2` ou `otimizador_spea2`).
        semente: semente desta execução.
    """
    if nome_algoritmo not in ALGORITMOS_MOEA:
        raise ValueError(f"Algoritmo desconhecido: {nome_algoritmo!r}. Use um de {list(ALGORITMOS_MOEA)}.")

    chave_config = f"otimizador_{nome_algoritmo.lower()}"
    cfg_moea = config[chave_config]

    problema = ProblemaPortfolioMO(dados, cfg_obj_mo)
    reparo = RepairCappedSimplex(w_min=cfg_obj_mo.w_min, w_max=cfg_obj_mo.w_max)
    callback = HistoricoConvergenciaMO(ref_point=cfg_obj_mo.ref_point)

    AlgoritmoCls = ALGORITMOS_MOEA[nome_algoritmo]
    algoritmo = AlgoritmoCls(
        pop_size=cfg_moea["pop_size"],
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
