"""Figuras de análise: curva de convergência e boxplot comparativo.

Ambas as figuras seguem o mesmo padrão de "qualidade publicável": eixos
rotulados, título, legenda e `dpi` alto, salvas em `figures/`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_convergencia(
    historicos: list[np.ndarray],
    caminho_saida: str | Path,
    titulo: str = "Convergência do Differential Evolution",
) -> None:
    """Curva de convergência (best-so-far) com média ± desvio-padrão entre sementes.

    Args:
        historicos: lista de arrays (um por semente), cada um com o melhor g
            best-so-far por geração (ver ResultadoDE.historico_best_so_far).
        caminho_saida: caminho do arquivo de imagem a salvar.
        titulo: título do gráfico.
    """
    matriz = np.vstack(historicos)  # (n_seeds, n_gen)
    media = matriz.mean(axis=0)
    desvio = matriz.std(axis=0)
    geracoes = np.arange(1, matriz.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(geracoes, media, color="#1f77b4", label="Média entre sementes", linewidth=2)
    ax.fill_between(
        geracoes,
        media - desvio,
        media + desvio,
        color="#1f77b4",
        alpha=0.25,
        label="± 1 desvio-padrão",
    )
    ax.set_xlabel("Geração")
    ax.set_ylabel("Melhor g(w) encontrado até a geração (best-so-far)")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def plot_boxplot_comparativo(
    valores_de: list[float],
    valores_random_search: list[float],
    g_exato: float,
    g_uniforme: float,
    caminho_saida: str | Path,
    titulo: str = "Comparação do valor final de g(w) entre métodos",
) -> None:
    """Boxplot do g final do DE e do random search (entre sementes), com baselines.

    Os baselines determinísticos (ótimo exato via cvxpy e carteira 1/N) não
    têm distribuição entre sementes — são desenhados como linhas
    horizontais de referência (`axhline`).

    Args:
        valores_de: valor final de g do DE, um por semente.
        valores_random_search: valor final de g do random search, um por semente.
        g_exato: valor de g do ótimo exato (cvxpy), referência.
        g_uniforme: valor de g da carteira 1/N, referência.
        caminho_saida: caminho do arquivo de imagem a salvar.
        titulo: título do gráfico.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [valores_de, valores_random_search],
        tick_labels=["DE", "Random Search"],
        showmeans=True,
    )

    ax.axhline(g_exato, color="green", linestyle="--", linewidth=1.5, label=f"Ótimo exato (cvxpy) = {g_exato:.3f}")
    ax.axhline(g_uniforme, color="red", linestyle=":", linewidth=1.5, label=f"Carteira 1/N = {g_uniforme:.3f}")

    ax.set_ylabel("g(w) final (escalarização normalizada)")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
