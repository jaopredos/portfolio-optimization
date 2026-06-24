"""Utilitários compartilhados entre os módulos do projeto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CAMINHO_CONFIG_PADRAO = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def carregar_config(caminho: str | Path = CAMINHO_CONFIG_PADRAO) -> dict[str, Any]:
    """Carrega o arquivo de configuração YAML do projeto.

    Centralizar essa leitura aqui garante que todos os módulos (data_loader,
    optimizer_de, baselines, run_experiment, etc.) leiam exatamente o mesmo
    config e que mudar um parâmetro no YAML baste para refletir em todo o
    pipeline, sem editar código.

    Args:
        caminho: caminho para o arquivo config.yaml.

    Returns:
        Dicionário com as seções do config (dados, restricoes, objetivo, ...).
    """
    caminho = Path(caminho)
    with open(caminho, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
