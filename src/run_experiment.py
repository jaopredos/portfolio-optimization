"""Orquestra o experimento completo do Checkpoint 1.

Checkpoint 1: problema MONO-OBJETIVO de risco-retorno (Markowitz clássico,
sem ESG — ver `src/esg_cp2.py` para o terceiro objetivo, reservado ao CP2).

Passo a passo (tudo dirigido por config/config.yaml):
  1. Carrega dados de mercado (mu, Sigma) — `data_loader.py`.
  2. Constrói a escalarização normalizada (lambdas + estatísticas de
     z-score) — `objectives.py`.
  3. Para cada semente em `experimento.sementes`:
       a. roda o DE (`optimizer_de.py`) e guarda a melhor carteira, o g
          final e o histórico de convergência best-so-far;
       b. roda o random search (`baselines.py`) com o MESMO orçamento de
          avaliações que o DE consumiu naquela semente, usando a mesma
          semente (para reprodutibilidade individual por semente).
  4. Calcula o ótimo exato (cvxpy) e a carteira 1/N uma única vez (são
     deterministicos, não dependem de semente).
  5. Calcula métricas (retorno, risco, Sharpe, gap) de toda carteira
     encontrada — `metrics.py`.
  6. Salva tabelas (CSV) em `results/` e figuras em `figures/` —
     `plots.py`.

Uso: `uv run python -m src.run_experiment` (ou `python src/run_experiment.py`
de dentro do diretório do projeto), com config/config.yaml já preenchido.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import carteira_uniforme, otimo_exato_cvxpy, rodar_random_search
from src.data_loader import DadosMercado, carregar_dados
from src.metrics import calcular_metricas_carteira
from src.objectives import construir_config_objetivo
from src.optimizer_de import rodar_de
from src.plots import plot_boxplot_comparativo, plot_convergencia
from src.utils import carregar_config

RAIZ_PROJETO = Path(__file__).resolve().parent.parent


def rodar_experimento(config: dict) -> dict[str, pd.DataFrame]:
    """Executa o experimento completo e devolve os DataFrames de resultado.

    Returns:
        Dicionário com as chaves "detalhado" (uma linha por semente x método)
        e "agregado" (estatísticas média/desvio/mediana/min/max por método).
    """
    dados: DadosMercado = carregar_dados(config)
    cfg_obj = construir_config_objetivo(dados, config)
    cfg_metricas = config.get("metricas", {})
    rf = cfg_metricas.get("taxa_livre_risco", 0.0)

    print(f"Dados carregados: {len(dados.tickers)} ativos.")

    # --- Baselines determinísticos (não dependem de semente) ---
    exato = otimo_exato_cvxpy(dados, cfg_obj)
    uniforme = carteira_uniforme(dados, cfg_obj)
    print(f"Ótimo exato (cvxpy): g = {exato.g:.4f}")
    print(f"Carteira 1/N:        g = {uniforme.g:.4f}")

    linhas: list[dict] = []
    historicos_de: list[np.ndarray] = []
    pesos_de_por_semente: dict[int, np.ndarray] = {}

    sementes = config["experimento"]["sementes"]
    for semente in sementes:
        print(f"--- Semente {semente} ---")

        resultado_de = rodar_de(dados, cfg_obj, config, semente)
        historicos_de.append(resultado_de.historico_best_so_far)
        pesos_de_por_semente[semente] = resultado_de.melhor_w
        print(f"  DE:            g = {resultado_de.melhor_g:.4f}  ({resultado_de.n_avaliacoes} avaliações)")

        resultado_rs = rodar_random_search(
            dados, cfg_obj, n_avaliacoes=resultado_de.n_avaliacoes, semente=semente
        )
        print(f"  Random search: g = {resultado_rs.g:.4f}  ({resultado_de.n_avaliacoes} avaliações)")

        for metodo, resultado in [("DE", resultado_de), ("Random Search", resultado_rs)]:
            w = resultado.melhor_w if metodo == "DE" else resultado.w
            g = resultado.melhor_g if metodo == "DE" else resultado.g
            metricas = calcular_metricas_carteira(w, dados, g, exato.g, rf)
            linhas.append({"semente": semente, "metodo": metodo, **metricas})

    # Baselines determinísticos entram na tabela com semente=NaN (não se aplica).
    for metodo, resultado in [("Ótimo exato (cvxpy)", exato), ("1/N", uniforme)]:
        metricas = calcular_metricas_carteira(resultado.w, dados, resultado.g, exato.g, rf)
        linhas.append({"semente": np.nan, "metodo": metodo, **metricas})

    df_detalhado = pd.DataFrame(linhas)

    agregados = (
        df_detalhado.groupby("metodo")[["g", "retorno", "risco", "sharpe", "gap_relativo"]]
        .agg(["mean", "std", "median", "min", "max"])
    )

    # --- Figuras ---
    dir_figuras = RAIZ_PROJETO / config["experimento"]["dir_figuras"]
    plot_convergencia(historicos_de, dir_figuras / "convergencia_de.png")

    valores_g_de = df_detalhado.query("metodo == 'DE'")["g"].tolist()
    valores_g_rs = df_detalhado.query("metodo == 'Random Search'")["g"].tolist()
    plot_boxplot_comparativo(valores_g_de, valores_g_rs, exato.g, uniforme.g, dir_figuras / "boxplot_g_final.png")

    # --- Tabelas ---
    dir_resultados = RAIZ_PROJETO / config["experimento"]["dir_resultados"]
    dir_resultados.mkdir(parents=True, exist_ok=True)
    df_detalhado.to_csv(dir_resultados / "resultados_detalhados.csv", index=False)
    agregados.to_csv(dir_resultados / "resultados_agregados.csv")

    df_pesos = pd.DataFrame.from_dict(pesos_de_por_semente, orient="index", columns=dados.tickers)
    df_pesos.index.name = "semente"
    df_pesos.to_csv(dir_resultados / "melhores_pesos_de.csv")

    print(f"\nResultados salvos em {dir_resultados}/")
    print(f"Figuras salvas em {dir_figuras}/")

    return {"detalhado": df_detalhado, "agregado": agregados}


def main() -> None:
    config = carregar_config()
    rodar_experimento(config)


if __name__ == "__main__":
    main()
