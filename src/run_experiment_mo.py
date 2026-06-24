"""Orquestra o experimento completo do Checkpoint 2 (multiobjetivo).

Checkpoint 2 estende o CP1 com o terceiro objetivo (ESG, reativado de
`src/esg_cp2.py`) e resolve o problema com dois MOEAs (NSGA-II e SPEA-II,
via pymoo), cobrindo os 4 pontos pedidos no enunciado:

  1. Versão multiobjetivo rodando — NSGA-II e SPEA-II, 5 sementes cada
     (`src/optimizer_moea.py::rodar_moea`).
  2. Comparação com a versão single-objective — reusa `rodar_de` do CP1
     (sem modificação) sobre os MESMOS dados (mu, Sigma) desta execução, e
     mapeia a carteira encontrada no espaço de 3 objetivos.
  3. Baseline — 1/N e random search (mesmo orçamento dos MOEAs), reduzidos
     ao subconjunto não-dominado.
  4. Fronteira de Pareto inicial — população inicial (já reparada) de cada
     MOEA, comparada à fronteira final.

Tudo é aditivo: nenhum módulo do CP1 é modificado, apenas reusado. Salva
CSVs em `results/*_mo.csv` e figuras em `figures/*_mo.png` — nomes
diferentes dos arquivos do CP1, sem sobrescrever nada.

Uso: `uv run python -m src.run_experiment_mo`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines_mo import (
    avaliar_carteiras_mo,
    carteira_uniforme_mo,
    filtrar_nao_dominadas,
    rodar_random_search_mo,
)
from src.data_loader import DadosMercado, carregar_dados
from src.metrics_mo import calcular_hipervolume, contar_nao_dominadas, is_dominado, matriz_metricas_mo
from src.objectives import construir_config_objetivo
from src.objectives_mo import ConfigObjetivoMO, construir_config_objetivo_mo, f_objetivos_normalizados
from src.optimizer_de import rodar_de
from src.optimizer_moea import ALGORITMOS_MOEA, ResultadoMOEA, rodar_moea
from src.plots_mo import (
    plot_convergencia_hv,
    plot_fronteira_inicial_vs_final,
    plot_overlay_referencias,
    plot_pareto_pairwise,
)
from src.utils import carregar_config

RAIZ_PROJETO = Path(__file__).resolve().parent.parent


def _agregar_fronteira(resultados: list[ResultadoMOEA], campo_X: str, campo_F: str) -> tuple[np.ndarray, np.ndarray]:
    """União das soluções de todas as sementes de um algoritmo, filtrada ao subconjunto não-dominado."""
    X_uniao = np.vstack([getattr(r, campo_X) for r in resultados])
    F_uniao = np.vstack([getattr(r, campo_F) for r in resultados])
    return filtrar_nao_dominadas(X_uniao, F_uniao)


def rodar_experimento_mo(config: dict) -> dict[str, pd.DataFrame]:
    """Executa o experimento multiobjetivo completo e devolve os DataFrames de resultado."""
    dados: DadosMercado = carregar_dados(config)
    cfg_obj_mo: ConfigObjetivoMO = construir_config_objetivo_mo(dados, config)
    cfg_metricas = config.get("metricas", {})
    rf = cfg_metricas.get("taxa_livre_risco", 0.0)

    print(f"Dados carregados: {len(dados.tickers)} ativos.")

    sementes = config["experimento"]["sementes"]

    # --- 1. MOEAs (NSGA-II e SPEA-II), 5 sementes cada ---
    resultados_por_algoritmo: dict[str, list[ResultadoMOEA]] = {nome: [] for nome in ALGORITMOS_MOEA}
    for nome_algoritmo in ALGORITMOS_MOEA:
        for semente in sementes:
            resultado = rodar_moea(nome_algoritmo, dados, cfg_obj_mo, config, semente)
            resultados_por_algoritmo[nome_algoritmo].append(resultado)
            print(
                f"  {nome_algoritmo} (semente={semente}): "
                f"HV final = {resultado.historico_hv[-1]:.2f}  ({resultado.n_avaliacoes} avaliações)"
            )

    # --- Agregação entre sementes: união das populações final/inicial, filtrada a não-dominados ---
    fronteiras_finais: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fronteiras_iniciais: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    tabelas_finais: dict[str, pd.DataFrame] = {}
    tabelas_iniciais: dict[str, pd.DataFrame] = {}
    linhas_metricas_agregadas: list[dict] = []

    for nome_algoritmo, resultados in resultados_por_algoritmo.items():
        X_final_agg, F_final_agg = _agregar_fronteira(resultados, "X_final", "F_final")
        X_inicial_agg, F_inicial_agg = _agregar_fronteira(resultados, "X_inicial", "F_inicial")
        fronteiras_finais[nome_algoritmo] = (X_final_agg, F_final_agg)
        fronteiras_iniciais[nome_algoritmo] = (X_inicial_agg, F_inicial_agg)

        tabelas_finais[nome_algoritmo] = matriz_metricas_mo(X_final_agg, dados, cfg_obj_mo.esg, rf)
        tabelas_iniciais[nome_algoritmo] = matriz_metricas_mo(X_inicial_agg, dados, cfg_obj_mo.esg, rf)

        hv_agregado = calcular_hipervolume(F_final_agg, cfg_obj_mo.ref_point)
        n_nao_dominadas_agregado = contar_nao_dominadas(F_final_agg)
        print(
            f"{nome_algoritmo}: fronteira final agregada (5 sementes) com "
            f"{n_nao_dominadas_agregado} soluções não-dominadas, HV={hv_agregado:.2f}"
        )

        for resultado in resultados:
            linhas_metricas_agregadas.append(
                {
                    "algoritmo": nome_algoritmo,
                    "semente": resultado.semente,
                    "hv_final_semente": calcular_hipervolume(resultado.F_final, cfg_obj_mo.ref_point),
                    "n_nao_dominadas_semente": contar_nao_dominadas(resultado.F_final),
                    "n_avaliacoes": resultado.n_avaliacoes,
                }
            )
        linhas_metricas_agregadas.append(
            {
                "algoritmo": nome_algoritmo,
                "semente": np.nan,
                "hv_final_semente": hv_agregado,
                "n_nao_dominadas_semente": n_nao_dominadas_agregado,
                "n_avaliacoes": resultados[0].n_avaliacoes,
            }
        )

    # --- 2. Comparação com a versão single-objective (DE do CP1, mesmos dados) ---
    cfg_obj_so = construir_config_objetivo(dados, config)
    linhas_so_de: list[dict] = []
    for semente in sementes:
        resultado_de = rodar_de(dados, cfg_obj_so, config, semente)
        w = resultado_de.melhor_w
        metricas = matriz_metricas_mo(np.array([w]), dados, cfg_obj_mo.esg, rf).iloc[0].to_dict()
        ponto_norm = f_objetivos_normalizados(w, dados, cfg_obj_mo)
        linha = {"semente": semente, **metricas}
        for nome_algoritmo, (_, F_final_agg) in fronteiras_finais.items():
            linha[f"dominado_por_{nome_algoritmo}"] = is_dominado(ponto_norm, F_final_agg)
        linhas_so_de.append(linha)
    df_comparacao_so = pd.DataFrame(linhas_so_de)
    tabela_so_de = df_comparacao_so[["retorno", "risco", "sharpe", "score_esg"]]
    print("Comparação com DE mono-objetivo (CP1):")
    print(df_comparacao_so.to_string(index=False))

    # --- 3. Baselines: 1/N e random search (mesmo orçamento total dos MOEAs) ---
    w_uniforme = carteira_uniforme_mo(dados)
    tabela_uniforme = matriz_metricas_mo(np.array([w_uniforme]), dados, cfg_obj_mo.esg, rf)

    n_avaliacoes_budget = max(r.n_avaliacoes for resultados in resultados_por_algoritmo.values() for r in resultados)
    X_rs = rodar_random_search_mo(dados, cfg_obj_mo, n_avaliacoes=n_avaliacoes_budget, semente=sementes[0])
    F_rs = avaliar_carteiras_mo(X_rs, dados, cfg_obj_mo)
    X_rs_nd, _ = filtrar_nao_dominadas(X_rs, F_rs)
    tabela_rs_nd = matriz_metricas_mo(X_rs_nd, dados, cfg_obj_mo.esg, rf)
    print(f"Random search: {len(X_rs)} amostras -> {len(X_rs_nd)} não-dominadas.")

    # --- Figuras ---
    dir_figuras = RAIZ_PROJETO / config["experimento"]["dir_figuras"]
    plot_pareto_pairwise(tabelas_finais, dir_figuras / "pareto_pairwise_mo.png")
    plot_fronteira_inicial_vs_final(tabelas_iniciais, tabelas_finais, dir_figuras / "fronteira_inicial_vs_final_mo.png")
    plot_convergencia_hv(
        {nome: [r.historico_hv for r in resultados] for nome, resultados in resultados_por_algoritmo.items()},
        dir_figuras / "convergencia_hv_mo.png",
    )
    plot_overlay_referencias(
        tabelas_finais,
        tabela_so_de,
        {"1/N": tabela_uniforme, "Random Search": tabela_rs_nd},
        dir_figuras / "overlay_referencias_mo.png",
    )

    # --- Tabelas ---
    dir_resultados = RAIZ_PROJETO / config["experimento"]["dir_resultados"]
    dir_resultados.mkdir(parents=True, exist_ok=True)

    linhas_pareto_final: list[pd.DataFrame] = []
    for nome_algoritmo, (X_final_agg, _) in fronteiras_finais.items():
        df_pesos = pd.DataFrame(X_final_agg, columns=dados.tickers)
        df_metricas = tabelas_finais[nome_algoritmo].reset_index(drop=True)
        df_algoritmo = pd.concat([pd.Series([nome_algoritmo] * len(df_pesos), name="algoritmo"), df_metricas, df_pesos], axis=1)
        linhas_pareto_final.append(df_algoritmo)
    df_pareto_final = pd.concat(linhas_pareto_final, ignore_index=True)
    df_pareto_final.to_csv(dir_resultados / "pareto_final_mo.csv", index=False)

    df_metricas_agregadas = pd.DataFrame(linhas_metricas_agregadas)
    df_metricas_agregadas.to_csv(dir_resultados / "metricas_agregadas_mo.csv", index=False)

    df_comparacao_so.to_csv(dir_resultados / "comparacao_so_mo.csv", index=False)

    df_baselines = pd.concat(
        [
            tabela_uniforme.assign(fonte="1/N"),
            tabela_rs_nd.assign(fonte="Random Search"),
        ],
        ignore_index=True,
    )
    df_baselines.to_csv(dir_resultados / "baselines_mo.csv", index=False)

    print(f"\nResultados salvos em {dir_resultados}/ (sufixo _mo)")
    print(f"Figuras salvas em {dir_figuras}/ (sufixo _mo)")

    return {
        "pareto_final": df_pareto_final,
        "metricas_agregadas": df_metricas_agregadas,
        "comparacao_so": df_comparacao_so,
        "baselines": df_baselines,
    }


def main() -> None:
    config = carregar_config()
    rodar_experimento_mo(config)


if __name__ == "__main__":
    main()
