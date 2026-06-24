"""Tratamento das restrições do problema de alocação de portfólio.

O problema impõe, para o vetor de pesos w in R^N:
    soma(w) = 1          (totalmente investido)
    w_min <= w_i <= w_max  para todo i   (long-only com teto de concentração)

A estratégia adotada como padrão é REPARO: em vez de penalizar candidatos
infactíveis na função objetivo (o que exige ajustar coeficientes de
penalidade e pode distorcer o cenário de fitness para a metaheurística),
projetamos cada candidato no conjunto factível — o "capped simplex" — antes
de avaliá-lo. Isso garante que TODO indivíduo avaliado pelo DE é uma
carteira válida, sem hiperparâmetro de penalidade para calibrar.
"""

from __future__ import annotations

import numpy as np


def projetar_capped_simplex(
    v: np.ndarray,
    w_min: float = 0.0,
    w_max: float = 0.20,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> np.ndarray:
    """Projeta um vetor v em R^N no "capped simplex" {w : soma(w)=1, w_min<=w_i<=w_max}.

    Ideia (water-filling / busca por bisseção no multiplicador de Lagrange):
    a solução do problema de projeção (minimizar ||w - v||^2 sujeito às
    restrições) tem a forma fechada
        w_i(tau) = clip(v_i - tau, w_min, w_max)
    para algum escalar tau (o multiplicador associado à restrição de
    igualdade soma(w)=1). A função
        f(tau) = soma_i clip(v_i - tau, w_min, w_max) - 1
    é monótona não-crescente em tau (aumentar tau só pode diminuir ou manter
    cada termo do clip), então existe um único tau* com f(tau*) = 0, que
    encontramos por bisseção. Aplicar o clip nesse tau* dá exatamente a
    projeção euclidiana desejada.

    Pré-condição de factibilidade do próprio domínio: é preciso que
    N * w_min <= 1 <= N * w_max, isto é, que seja possível somar 1 respeitando
    os tetos/pisos por ativo (ver `checar_problema_factivel`).

    Args:
        v: vetor candidato (não necessariamente factível), shape (N,).
        w_min: piso por ativo (tipicamente 0, long-only).
        w_max: teto por ativo (concentração máxima).
        tol: tolerância na busca por tau (em termos de soma(w) - 1).
        max_iter: número máximo de iterações de bisseção.

    Returns:
        Vetor w factível (soma(w)=1, w_min<=w_i<=w_max), shape (N,).
    """
    v = np.asarray(v, dtype=float)
    n = v.shape[0]

    if n * w_min - 1 > tol or 1 - n * w_max > tol:
        raise ValueError(
            f"Capped simplex infactível: precisa N*w_min <= 1 <= N*w_max, "
            f"mas N={n}, w_min={w_min}, w_max={w_max} "
            f"(N*w_min={n * w_min:.4f}, N*w_max={n * w_max:.4f})."
        )

    # Intervalo de busca para tau: fora de [v.min()-w_max, v.max()-w_min] o
    # clip satura todas as coordenadas no mesmo extremo, então a raiz de f
    # certamente está dentro desse intervalo.
    tau_lo = v.min() - w_max
    tau_hi = v.max() - w_min

    def soma_clip(tau: float) -> float:
        return np.clip(v - tau, w_min, w_max).sum()

    for _ in range(max_iter):
        tau_mid = 0.5 * (tau_lo + tau_hi)
        s = soma_clip(tau_mid)
        if abs(s - 1.0) <= tol:
            tau_lo = tau_hi = tau_mid
            break
        # f é não-crescente em tau: soma > 1 significa que tau está baixo
        # demais (precisamos subtrair mais), então sobe o limite inferior.
        if s > 1.0:
            tau_lo = tau_mid
        else:
            tau_hi = tau_mid

    tau = 0.5 * (tau_lo + tau_hi)
    w = np.clip(v - tau, w_min, w_max)

    # Correção numérica final: bisseção pode deixar um resíduo de soma
    # da ordem de `tol`; redistribuímos esse resíduo igualmente entre as
    # coordenadas não saturadas para fechar soma(w)=1 exatamente.
    residuo = 1.0 - w.sum()
    if abs(residuo) > 0:
        livres = (w > w_min + 1e-12) & (w < w_max - 1e-12)
        if livres.any():
            w[livres] += residuo / livres.sum()
        else:
            w += residuo / n
        w = np.clip(w, w_min, w_max)

    return w


def checar_problema_factivel(n_ativos: int, w_min: float, w_max: float) -> None:
    """Verifica se o capped simplex é não-vazio para (n_ativos, w_min, w_max).

    Levanta ValueError se N*w_min > 1 ou N*w_max < 1, casos em que nenhum
    vetor pode simultaneamente somar 1 e respeitar os limites por ativo.
    """
    if n_ativos * w_min > 1.0:
        raise ValueError(
            f"w_min={w_min} é grande demais para N={n_ativos} ativos somarem 1."
        )
    if n_ativos * w_max < 1.0:
        raise ValueError(
            f"w_max={w_max} é pequeno demais para N={n_ativos} ativos somarem 1."
        )


def checar_factibilidade(
    w: np.ndarray,
    w_min: float = 0.0,
    w_max: float = 0.20,
    tol: float = 1e-6,
) -> bool:
    """Confere se w satisfaz soma(w)=1 e w_min<=w_i<=w_max, dentro de `tol`."""
    w = np.asarray(w, dtype=float)
    soma_ok = abs(w.sum() - 1.0) <= tol
    limites_ok = bool(np.all(w >= w_min - tol) and np.all(w <= w_max + tol))
    return soma_ok and limites_ok
