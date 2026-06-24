"""Metaheurística mono-objetivo do Checkpoint 1: Differential Evolution (pymoo).

Três peças se encaixam aqui:

1. `ProblemaPortfolio` (subclasse de `pymoo.core.problem.Problem`): define o
   espaço de busca (n_var = N ativos, caixa [w_min, w_max] por variável) e a
   função de avaliação, que é simplesmente `g_escalarizado` aplicada a cada
   indivíduo da população.

2. `RepairCappedSimplex` (subclasse de `pymoo.core.repair.Repair`): aplica a
   projeção no capped-simplex (src/constraints.py) a cada indivíduo. Isso é
   necessário porque o DE do pymoo só garante limites de caixa
   (w_min <= w_i <= w_max) nativamente — a restrição de igualdade
   soma(w) = 1 não é respeitada pelo operador de mutação/crossover
   diferencial por si só. Passar `repair=RepairCappedSimplex()` para o
   algoritmo `DE(...)` faz o pymoo aplicar essa projeção tanto na população
   inicial quanto em todo indivíduo gerado a cada geração (ver
   `pymoo.algorithms.soo.nonconvex.de.Variant.do`, que chama `self.repair`
   no fim da reprodução). Resultado: todo indivíduo efetivamente avaliado
   pelo DE é uma carteira factível.

3. `HistoricoConvergencia` (subclasse de `pymoo.core.callback.Callback`):
   grava, a cada geração, o melhor valor de g já visto até então
   ("best-so-far"), para as curvas de convergência do relatório.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.constraints import projetar_capped_simplex
from src.data_loader import DadosMercado
from src.objectives import ConfigObjetivo, g_escalarizado


class ProblemaPortfolio(Problem):
    """Problema de otimização do pymoo: minimizar g(w) na caixa [w_min, w_max]^N.

    A restrição soma(w)=1 NÃO é declarada aqui como `n_eq_constr` — ela é
    tratada via reparo (`RepairCappedSimplex`), não via penalização. Isso
    simplifica a função objetivo (não precisa de termos de penalidade) e
    garante 100% de factibilidade nos indivíduos avaliados.
    """

    def __init__(self, dados: DadosMercado, cfg_obj: ConfigObjetivo) -> None:
        self.dados = dados
        self.cfg_obj = cfg_obj
        n_ativos = len(dados.tickers)
        super().__init__(
            n_var=n_ativos,
            n_obj=1,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=cfg_obj.w_min,
            xu=cfg_obj.w_max,
        )

    def _evaluate(self, X: np.ndarray, out: dict[str, Any], *args, **kwargs) -> None:
        # X: shape (pop_size, n_var). Cada linha já chega reparada (capped
        # simplex) pelo RepairCappedSimplex antes de ser avaliada.
        valores = np.array([g_escalarizado(w, self.dados, self.cfg_obj) for w in X])
        out["F"] = valores.reshape(-1, 1)


class RepairCappedSimplex(Repair):
    """Reparo: projeta cada indivíduo da população no capped-simplex.

    Ver docstring do módulo e `src/constraints.projetar_capped_simplex` para
    o algoritmo de projeção (bisseção no multiplicador de Lagrange da
    restrição de igualdade).
    """

    def __init__(self, w_min: float, w_max: float) -> None:
        super().__init__()
        self.w_min = w_min
        self.w_max = w_max

    def _do(self, problem: Problem, X: np.ndarray, **kwargs) -> np.ndarray:
        return np.array([projetar_capped_simplex(x, self.w_min, self.w_max) for x in X])


class HistoricoConvergencia(Callback):
    """Registra, por geração, o melhor g já observado até então (best-so-far)."""

    def __init__(self) -> None:
        super().__init__()
        self.data["best_geracao"] = []
        self.data["best_so_far"] = []

    def notify(self, algorithm) -> None:
        melhor_atual = float(algorithm.pop.get("F").min())
        anterior = self.data["best_so_far"][-1] if self.data["best_so_far"] else np.inf
        self.data["best_geracao"].append(melhor_atual)
        self.data["best_so_far"].append(min(anterior, melhor_atual))


@dataclass
class ResultadoDE:
    """Saída de uma execução do DE para uma única semente."""

    semente: int
    melhor_w: np.ndarray
    melhor_g: float
    historico_best_so_far: np.ndarray  # shape (n_gen,)
    n_avaliacoes: int


def rodar_de(
    dados: DadosMercado,
    cfg_obj: ConfigObjetivo,
    config: dict[str, Any],
    semente: int,
) -> ResultadoDE:
    """Executa o Differential Evolution (pymoo) para uma semente e devolve o resultado.

    A semente controla TODA a aleatoriedade do pymoo nesta execução (via
    `minimize(..., seed=semente)`, que internamente usa um `RandomState`
    próprio do pymoo, independente do estado global do numpy — daí cada
    semente gerar uma trajetória de busca distinta e reprodutível).

    Args:
        dados: mu, sigma e tickers (ver src/data_loader.py).
        cfg_obj: lambdas e estatísticas de normalização (ver src/objectives.py).
        config: dicionário de config.yaml (lê a seção `otimizador_de`).
        semente: semente desta execução.

    Returns:
        ResultadoDE com a melhor carteira, o melhor g e o histórico de convergência.
    """
    cfg_de = config["otimizador_de"]

    problema = ProblemaPortfolio(dados, cfg_obj)
    reparo = RepairCappedSimplex(w_min=cfg_obj.w_min, w_max=cfg_obj.w_max)

    algoritmo = DE(
        pop_size=cfg_de["pop_size"],
        variant=cfg_de["variant"],
        CR=cfg_de["CR"],
        F=cfg_de["F"],
        repair=reparo,
    )

    termination = get_termination("n_gen", cfg_de["n_gen"])
    callback = HistoricoConvergencia()

    res = minimize(
        problema,
        algoritmo,
        termination,
        seed=semente,
        verbose=False,
        callback=callback,
        save_history=False,
    )

    historico = np.array(callback.data["best_so_far"])
    n_avaliacoes = int(res.algorithm.evaluator.n_eval)

    # `res.X` já é factível (passou pelo reparo), mas projetamos de novo por
    # segurança numérica antes de reportar a carteira final.
    melhor_w = projetar_capped_simplex(res.X, cfg_obj.w_min, cfg_obj.w_max)
    melhor_g = float(res.F[0])

    return ResultadoDE(
        semente=semente,
        melhor_w=melhor_w,
        melhor_g=melhor_g,
        historico_best_so_far=historico,
        n_avaliacoes=n_avaliacoes,
    )
