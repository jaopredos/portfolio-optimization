"""Figuras do Checkpoint 2 (multiobjetivo): fronteiras de Pareto, convergência de HV e comparações.

Mesmo padrão estético de `src/plots.py` (CP1): `figsize` fixo, `dpi=150`,
`mkdir(parents=True, exist_ok=True)` antes de salvar. Todas as funções
recebem tabelas em UNIDADES ORIGINAIS (colunas `retorno`, `risco`,
`score_esg`, ver `src.metrics_mo.matriz_metricas_mo`) — nunca os objetivos
normalizados internos do pymoo, para que os eixos sejam interpretáveis
(retorno anualizado, volatilidade anualizada, score ESG em 0-100).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARES_OBJETIVOS = [("retorno", "risco"), ("retorno", "score_esg"), ("risco", "score_esg")]
ROTULOS = {"retorno": "Retorno esperado anual", "risco": "Volatilidade anual", "score_esg": "Score ESG"}
CORES_ALGORITMO = {"NSGA2": "#1f77b4", "SPEA2": "#ff7f0e"}


def _grade_pares_objetivos(titulo: str) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (x, y) in zip(axes, PARES_OBJETIVOS):
        ax.set_xlabel(ROTULOS[x])
        ax.set_ylabel(ROTULOS[y])
        ax.grid(alpha=0.3)
    fig.suptitle(titulo)
    return fig, list(axes)


def _salvar(fig: plt.Figure, caminho_saida: str | Path) -> None:
    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def plot_pareto_pairwise(
    tabelas_por_algoritmo: dict[str, pd.DataFrame],
    caminho_saida: str | Path,
    titulo: str = "Fronteira de Pareto agregada — NSGA-II vs SPEA-II",
) -> None:
    """Grade de scatters pairwise (retorno×risco, retorno×ESG, risco×ESG), uma cor por algoritmo."""
    fig, axes = _grade_pares_objetivos(titulo)
    for nome, tabela in tabelas_por_algoritmo.items():
        cor = CORES_ALGORITMO.get(nome, None)
        for ax, (x, y) in zip(axes, PARES_OBJETIVOS):
            ax.scatter(tabela[x], tabela[y], s=20, alpha=0.7, color=cor, label=nome)
    axes[0].legend()
    _salvar(fig, caminho_saida)


def plot_fronteira_inicial_vs_final(
    tabelas_iniciais: dict[str, pd.DataFrame],
    tabelas_finais: dict[str, pd.DataFrame],
    caminho_saida: str | Path,
    titulo: str = "Fronteira de Pareto: população inicial vs final",
) -> None:
    """Para cada algoritmo, sobrepõe a fronteira inicial (cinza, translúcida) à final (cor sólida).

    Demonstra o progresso evolutivo: o quanto a busca avança da população
    aleatória inicial (já reparada, mas não otimizada) até a convergência.
    """
    algoritmos = list(tabelas_finais.keys())
    fig, axes = plt.subplots(len(algoritmos), 3, figsize=(15, 5 * len(algoritmos)), squeeze=False)
    for linha, nome in enumerate(algoritmos):
        cor = CORES_ALGORITMO.get(nome, None)
        for col, (x, y) in enumerate(PARES_OBJETIVOS):
            ax = axes[linha][col]
            ax.scatter(
                tabelas_iniciais[nome][x], tabelas_iniciais[nome][y],
                s=20, alpha=0.35, color="gray", label="Geração inicial",
            )
            ax.scatter(
                tabelas_finais[nome][x], tabelas_finais[nome][y],
                s=20, alpha=0.8, color=cor, label="Geração final",
            )
            ax.set_xlabel(ROTULOS[x])
            ax.set_ylabel(ROTULOS[y])
            ax.set_title(nome)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.legend()
    fig.suptitle(titulo)
    _salvar(fig, caminho_saida)


def plot_convergencia_hv(
    historicos_hv_por_algoritmo: dict[str, list],
    caminho_saida: str | Path,
    titulo: str = "Convergência do hipervolume (NSGA-II vs SPEA-II)",
) -> None:
    """Hipervolume médio ± desvio-padrão entre sementes, uma linha por algoritmo."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for nome, historicos in historicos_hv_por_algoritmo.items():
        matriz = np.vstack(historicos)  # (n_seeds, n_gen)
        media = matriz.mean(axis=0)
        desvio = matriz.std(axis=0)
        geracoes = np.arange(1, matriz.shape[1] + 1)
        cor = CORES_ALGORITMO.get(nome, None)
        ax.plot(geracoes, media, color=cor, label=f"{nome} — média entre sementes", linewidth=2)
        ax.fill_between(geracoes, media - desvio, media + desvio, color=cor, alpha=0.2)
    ax.set_xlabel("Geração")
    ax.set_ylabel("Hipervolume (objetivos normalizados, ref_point fixo)")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(alpha=0.3)
    _salvar(fig, caminho_saida)


def plot_overlay_referencias(
    tabelas_fronteira_final: dict[str, pd.DataFrame],
    tabela_so_de: pd.DataFrame,
    tabelas_baseline: dict[str, pd.DataFrame],
    caminho_saida: str | Path,
    titulo: str = "Fronteira final com referências (DE mono-objetivo e baselines)",
) -> None:
    """Fronteira final (por algoritmo) + ponto(s) do DE mono-objetivo + baselines (1/N, random search)."""
    fig, axes = _grade_pares_objetivos(titulo)
    for nome, tabela in tabelas_fronteira_final.items():
        cor = CORES_ALGORITMO.get(nome, None)
        for ax, (x, y) in zip(axes, PARES_OBJETIVOS):
            ax.scatter(tabela[x], tabela[y], s=18, alpha=0.5, color=cor, label=f"Fronteira {nome}")

    cores_baseline = {"1/N": "red", "Random Search": "purple"}
    for nome, tabela in tabelas_baseline.items():
        cor = cores_baseline.get(nome, "black")
        for ax, (x, y) in zip(axes, PARES_OBJETIVOS):
            marker = "s" if nome == "1/N" else "x"
            ax.scatter(tabela[x], tabela[y], s=60, color=cor, marker=marker, label=nome)

    for ax, (x, y) in zip(axes, PARES_OBJETIVOS):
        ax.scatter(
            tabela_so_de[x], tabela_so_de[y],
            s=120, color="black", marker="*", edgecolors="gold", linewidths=1.0,
            label="DE mono-objetivo (CP1)",
        )
    axes[0].legend(fontsize=8)
    _salvar(fig, caminho_saida)
