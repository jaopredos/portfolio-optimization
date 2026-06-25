"""XAI Financeiro: explica quais ativos o MOEA mais manipula para otimizar Risco e ESG.

Post-processing de `results/pareto_final_mo.csv`: treina um
`RandomForestRegressor` onde X = pesos dos ativos na carteira e y = risco
(ou score_esg), e extrai feature importances (MDI) como proxy de
explicação global — identifica quais ações o algoritmo mais zera ou
concentra para derrubar risco ou impulsionar ESG.

Gera dois gráficos em `figures/`:
  - `xai_shap_risco.png`  — importância dos ativos para o objetivo Risco.
  - `xai_shap_esg.png`    — importância dos ativos para o objetivo ESG.

Uso: `uv run python -m src.xai_analysis`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

COLUNAS_METRICAS = {"algoritmo", "retorno", "risco", "sharpe", "score_esg"}


def carregar_pareto(caminho: str | Path | None = None) -> pd.DataFrame:
    """Carrega o CSV da fronteira de Pareto final (NSGA-II + SPEA-II).

    Raises:
        FileNotFoundError: se o CSV não existir.
        ValueError: se o formato for inesperado.
    """
    if caminho is None:
        caminho = RAIZ_PROJETO / "results" / "pareto_final_mo.csv"
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Pareto final não encontrado: {caminho}. "
            "Execute `python -m src.run_experiment_mo` antes."
        )

    df = pd.read_csv(caminho)

    if df.empty:
        raise ValueError(f"CSV da fronteira de Pareto está vazio: {caminho}")

    return df


def _extrair_pesos_e_alvo(
    df: pd.DataFrame,
    alvo: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separa colunas de peso (X) e coluna-alvo (y) do DataFrame.

    Returns:
        (X, y, nomes_ativos)
    """
    colunas_peso = [c for c in df.columns if c not in COLUNAS_METRICAS]
    if not colunas_peso:
        raise ValueError("Nenhuma coluna de peso de ativo encontrada no CSV.")

    if alvo not in df.columns:
        raise ValueError(f"Coluna-alvo '{alvo}' não encontrada no CSV.")

    X = df[colunas_peso]
    y = df[alvo]
    return X, y, colunas_peso


def treinar_modelo(
    X: pd.DataFrame,
    y: pd.Series,
    semente: int = 42,
) -> RandomForestRegressor:
    """Treina um RandomForestRegressor para extrair feature importances.

    O modelo é usado apenas como ferramenta de interpretação (não para
    predição em produção), por isso não fazemos train/test split — queremos
    que o modelo veja TODOS os pontos da fronteira de Pareto para capturar a
    relação global entre pesos e objetivos.
    """
    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=semente,
        n_jobs=-1,
    )
    modelo.fit(X, y)
    return modelo


def _plot_importancias(
    importancias: np.ndarray,
    nomes_ativos: list[str],
    titulo: str,
    caminho_saida: Path,
    cor: str = "#2196F3",
    top_n: int | None = None,
) -> None:
    """Gráfico de barras horizontal com as feature importances."""
    indices_ordenados = np.argsort(importancias)
    if top_n is not None:
        indices_ordenados = indices_ordenados[-top_n:]

    nomes = [nomes_ativos[i] for i in indices_ordenados]
    valores = importancias[indices_ordenados]

    fig, ax = plt.subplots(figsize=(8, max(5, len(nomes) * 0.35)))
    ax.barh(nomes, valores, color=cor, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_title(titulo)
    ax.grid(axis="x", alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def rodar_xai(
    caminho_pareto: str | Path | None = None,
    dir_figuras: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Executa a análise XAI completa e retorna os rankings de importância.

    Returns:
        Dicionário com chaves "risco" e "esg", cada uma contendo um
        DataFrame com colunas (ativo, importancia) ordenado decrescente.
    """
    df = carregar_pareto(caminho_pareto)

    if dir_figuras is None:
        dir_figuras = RAIZ_PROJETO / "figures"
    dir_figuras = Path(dir_figuras)

    resultados: dict[str, pd.DataFrame] = {}

    alvos = {
        "risco": {
            "titulo": "XAI — Importância dos ativos para o Risco do portfólio",
            "arquivo": "xai_shap_risco.png",
            "cor": "#e74c3c",
        },
        "score_esg": {
            "titulo": "XAI — Importância dos ativos para o Score ESG",
            "arquivo": "xai_shap_esg.png",
            "cor": "#27ae60",
        },
    }

    for alvo, meta in alvos.items():
        X, y, nomes_ativos = _extrair_pesos_e_alvo(df, alvo)
        modelo = treinar_modelo(X, y)

        importancias = modelo.feature_importances_

        _plot_importancias(
            importancias=importancias,
            nomes_ativos=nomes_ativos,
            titulo=meta["titulo"],
            caminho_saida=dir_figuras / meta["arquivo"],
            cor=meta["cor"],
        )

        ranking = (
            pd.DataFrame(
                {
                    "ativo": nomes_ativos,
                    "importancia": importancias,
                }
            )
            .sort_values("importancia", ascending=False)
            .reset_index(drop=True)
        )

        resultados[alvo.replace("score_", "")] = ranking

        print(
            f"[{alvo}] Top-5 ativos: "
            + ", ".join(
                f"{r['ativo']} ({r['importancia']:.3f})"
                for _, r in ranking.head(5).iterrows()
            )
        )

    print(f"\nFiguras salvas em {dir_figuras}/")
    return resultados


def main() -> None:
    rodar_xai()


if __name__ == "__main__":
    main()
