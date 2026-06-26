"""Experimento integrado: MOEA normal vs MOEA com Warm-Start (pesos do DE).

Compara a convergência (hipervolume por geração) de ambas as abordagens
sobre as mesmas sementes, usando NSGA-II como algoritmo-alvo. Salva:

  - `results/comparacao_convergencia_warmstart.csv` — HV por geração,
    semente e variante (normal vs warm-start).
  - `figures/convergencia_warmstart.png` — curva de HV médio ± desvio.

Uso: `uv run python -m src.run_experiment_integrated`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import carregar_dados
from src.objectives_mo import construir_config_objetivo_mo
from src.optimizer_moea import ResultadoMOEA, rodar_moea
from src.optimizer_moea_warmstart import carregar_pesos_de, rodar_moea_warmstart
from src.utils import carregar_config

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
ALGORITMO_ALVO = "NSGA2"


def _rodar_variante_normal(
    dados: Any,
    cfg_obj_mo: Any,
    config: dict[str, Any],
    sementes: list[int],
) -> list[ResultadoMOEA]:
    resultados: list[ResultadoMOEA] = []
    for semente in sementes:
        res = rodar_moea(ALGORITMO_ALVO, dados, cfg_obj_mo, config, semente)
        resultados.append(res)
    return resultados


def _rodar_variante_warmstart(
    dados: Any,
    cfg_obj_mo: Any,
    config: dict[str, Any],
    sementes: list[int],
    pesos_de: np.ndarray,
) -> list[ResultadoMOEA]:
    resultados: list[ResultadoMOEA] = []
    for semente in sementes:
        res = rodar_moea_warmstart(
            ALGORITMO_ALVO, dados, cfg_obj_mo, config, semente, pesos_de
        )
        resultados.append(res)
    return resultados


def _construir_df_convergencia(
    resultados: list[ResultadoMOEA],
    variante: str,
) -> pd.DataFrame:
    """Monta DataFrame com colunas: variante, semente, geracao, hv."""
    linhas: list[dict[str, Any]] = []
    for res in resultados:
        for g, hv in enumerate(res.historico_hv, start=1):
            linhas.append(
                {
                    "variante": variante,
                    "semente": res.semente,
                    "geracao": g,
                    "hv": hv,
                }
            )
    return pd.DataFrame(linhas)


def _plot_convergencia_comparativa(
    df: pd.DataFrame,
    caminho_saida: Path,
) -> None:
    """Plota HV médio ± desvio entre sementes para cada variante."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cores = {"MOEA Normal": "#1f77b4", "MOEA Warm-Start": "#e74c3c"}
    estilos = {"MOEA Normal": "--", "MOEA Warm-Start": "-"}

    for variante, grupo in df.groupby("variante"):
        pivot = grupo.pivot(index="semente", columns="geracao", values="hv")
        media = pivot.mean(axis=0)
        desvio = pivot.std(axis=0)
        geracoes = media.index.to_numpy()

        cor = cores.get(str(variante), "gray")
        estilo = estilos.get(str(variante), "-")

        ax.plot(
            geracoes,
            media,
            color=cor,
            linestyle=estilo,
            linewidth=2,
            label=f"{variante} — média",
        )
        ax.fill_between(
            geracoes,
            media - desvio,
            media + desvio,
            color=cor,
            alpha=0.15,
        )

    ax.set_xlabel("Geração")
    ax.set_ylabel("Hipervolume")
    ax.set_title(f"Convergência {ALGORITMO_ALVO}: Normal vs Warm-Start (DE)")
    ax.legend()
    ax.grid(alpha=0.3)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def rodar_experimento_integrado(config: dict[str, Any]) -> pd.DataFrame:
    """Executa o experimento comparativo e retorna o DataFrame de convergência."""
    dados = carregar_dados(config)
    cfg_obj_mo = construir_config_objetivo_mo(dados, config)
    sementes: list[int] = config["experimento"]["sementes"]

    pesos_de = carregar_pesos_de(max_individuos=len(sementes))
    print(f"Warm-Start: {len(pesos_de)} indivíduos injetados do DE.")

    print(f"\n--- {ALGORITMO_ALVO} Normal ---")
    resultados_normal = _rodar_variante_normal(dados, cfg_obj_mo, config, sementes)
    for r in resultados_normal:
        print(f"  semente={r.semente}: HV final = {r.historico_hv[-1]:.2f}")

    print(f"\n--- {ALGORITMO_ALVO} Warm-Start ---")
    resultados_ws = _rodar_variante_warmstart(
        dados,
        cfg_obj_mo,
        config,
        sementes,
        pesos_de,
    )
    for r in resultados_ws:
        print(f"  semente={r.semente}: HV final = {r.historico_hv[-1]:.2f}")

    df_normal = _construir_df_convergencia(resultados_normal, "MOEA Normal")
    df_ws = _construir_df_convergencia(resultados_ws, "MOEA Warm-Start")
    df_completo = pd.concat([df_normal, df_ws], ignore_index=True)

    dir_resultados = RAIZ_PROJETO / config["experimento"]["dir_resultados"]
    dir_resultados.mkdir(parents=True, exist_ok=True)
    caminho_csv = dir_resultados / "comparacao_convergencia_warmstart.csv"
    df_completo.to_csv(caminho_csv, index=False)

    dir_figuras = RAIZ_PROJETO / config["experimento"]["dir_figuras"]
    _plot_convergencia_comparativa(
        df_completo,
        dir_figuras / "convergencia_warmstart.png",
    )

    print(f"\nResultados salvos em {caminho_csv}")
    print(f"Figura salva em {dir_figuras / 'convergencia_warmstart.png'}")

    return df_completo


def main() -> None:
    config = carregar_config()
    rodar_experimento_integrado(config)


if __name__ == "__main__":
    main()
