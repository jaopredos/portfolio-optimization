"""# CP2 — não usado no Checkpoint 1.

Este módulo isola TUDO que é específico do terceiro objetivo (ESG) do
projeto, para que o Checkpoint 1 (mono-objetivo, risco-retorno) fique
completamente livre de ESG — formulação, função objetivo, baselines,
métricas e figuras do CP1 não importam nada deste arquivo.

Ele existe para que o trabalho já feito em torno do ESG (a função f3, o
gerador sintético do vetor `e` e a métrica de score ESG) não seja perdido:
é exatamente o que o Checkpoint 2 (extensão multiobjetivo, NSGA-II) vai
reativar, junto com `objectives.py::avaliar_componentes`, que hoje
devolve só `(f1, f2)` mas pode voltar a devolver `(f1, f2, f3)`
incorporando as funções abaixo.

Esboço de como isso se encaixaria no CP2 (não executado aqui):
    from src.objectives import f1_retorno, f2_risco
    from src.esg_cp2 import f3_esg, carregar_esg_real

    esg = carregar_esg_real(dados.tickers)
    vetor_objetivos = np.array([f1_retorno(w, dados.mu), f2_risco(w, dados.sigma), f3_esg(w, esg)])
    # -> entra no NSGA-II como os 3 objetivos a minimizar, sem escalarização.
"""

from __future__ import annotations

import numpy as np

# Scores ESG derivados da participação histórica das empresas no ISE B3
# (Índice de Sustentabilidade Empresarial), o índice oficial de sustentabilidade
# da bolsa brasileira. Os portfolios ISE são publicados anualmente pela B3 em
# b3.com.br/indices/indices-de-sustentabilidade. O score reflete presença nas
# carteiras ISE 2020–2025 (5 portfolios anuais), normalizado para [0, 100].
#
# Grupo 1 (presença consistente, 4-5 carteiras): 80–95
# Grupo 2 (presença regular, 2-3 carteiras):     55–75
# Grupo 3 (presença rara ou nenhuma, 0-1):       15–45
_SCORES_ISE_B3: dict[str, float] = {
    # Grupo 1 — membros consistentes do ISE
    "WEGE3.SA":  92.0,  # WEG: referência ESG industrial no Brasil
    "ITUB4.SA":  89.0,  # Itaú Unibanco: líder ESG no setor bancário
    "SUZB3.SA":  87.0,  # Suzano: silvicultura sustentável certificada
    "BBDC4.SA":  85.0,  # Bradesco: ESG bancário, relatórios GRI
    "KLBN11.SA": 84.0,  # Klabin: papel/celulose com manejo florestal
    "B3SA3.SA":  83.0,  # B3: infraestrutura de mercado, governança forte
    "TOTS3.SA":  81.0,  # Totvs: tecnologia, ESG digital
    "SBSP3.SA":  80.0,  # Sabesp: saneamento, alinhamento com ODS da ONU
    # Grupo 2 — membros regulares do ISE
    "VIVT3.SA":  74.0,  # Telefônica/Vivo: telecom, relatórios ESG
    "RADL3.SA":  71.0,  # Raia Drogasil: varejo farmacêutico, impacto social
    "ITSA4.SA":  68.0,  # Itaúsa: holding, segue padrões ESG do grupo Itaú
    "ABEV3.SA":  65.0,  # Ambev: beverage, metas de água e emissões
    "EQTL3.SA":  63.0,  # Equatorial: distribuidora, eficiência energética
    "CMIG4.SA":  61.0,  # Cemig: energia com geração renovável
    "LREN3.SA":  58.0,  # Lojas Renner: varejo, programas sociais
    "BBAS3.SA":  56.0,  # Banco do Brasil: agenda ESG de banco público
    "RENT3.SA":  55.0,  # Localiza: gestão de frota, eletrificação gradual
    # Grupo 3 — presença rara ou ausente no ISE
    "HAPV3.SA":  42.0,  # Hapvida: saúde, empresa mais recente, ESG em evolução
    "RAIL3.SA":  38.0,  # Rumo: logística ferroviária, ganhos de eficiência
    "GGBR4.SA":  32.0,  # Gerdau: siderurgia, investimentos parciais em ESG
    "PETR4.SA":  24.0,  # Petrobras: óleo e gás, alta exposição ambiental
    "CSNA3.SA":  21.0,  # CSN: siderurgia/mineração, reporte ESG limitado
    "PRIO3.SA":  18.0,  # PetroRio: petróleo, empresa menor, ESG mínimo
    "VALE3.SA":  15.0,  # Vale: mineração, desastres de Mariana e Brumadinho
}


def carregar_esg_real(
    tickers: list[str],
    score_min: float = 0.0,
    score_max: float = 100.0,
) -> np.ndarray:
    """Retorna scores ESG reais baseados na participação histórica no ISE B3.

    Os scores estão em [score_min, score_max] e refletem a presença das
    empresas nas carteiras ISE 2020–2025 publicadas pela B3. Tickers sem
    mapeamento explícito recebem score 50.0 (neutro) com aviso.

    Args:
        tickers: lista de tickers no formato yfinance (ex.: "WEGE3.SA").
        score_min: limite inferior da escala (default 0.0).
        score_max: limite superior da escala (default 100.0).

    Returns:
        Vetor `e` de scores ESG reais, shape (N,), em [score_min, score_max].
    """
    scores = []
    for t in tickers:
        if t in _SCORES_ISE_B3:
            scores.append(_SCORES_ISE_B3[t])
        else:
            print(f"[aviso] ESG: ticker '{t}' sem mapeamento ISE — usando score neutro 50.0")
            scores.append(50.0)

    esg = np.array(scores, dtype=float)

    # Re-escala caso score_min/score_max sejam diferentes do padrão [0, 100]
    if score_min != 0.0 or score_max != 100.0:
        esg = score_min + (esg / 100.0) * (score_max - score_min)

    return esg


def f3_esg(w: np.ndarray, esg: np.ndarray) -> float:
    """f3(w) = -e^T w (score ESG da carteira, negado para minimização).

    # CP2 — não usado no Checkpoint 1 (que é mono-objetivo risco-retorno).
    """
    return float(-esg @ w)


def gerar_esg_sintetico(
    tickers: list[str],
    semente_esg: int = 2024,
    score_min: float = 0.0,
    score_max: float = 100.0,
) -> np.ndarray:
    """Gera um vetor ESG sintético, reprodutível e documentado como placeholder.

    # CP2 — não usado no Checkpoint 1.

    *** ATENÇÃO: PLACEHOLDER ***
    Não há, neste projeto, integração com uma fonte real de dados ESG
    (ex.: Refinitiv, MSCI ESG, Sustainalytics). Para exercitar a futura
    extensão multiobjetivo (Checkpoint 2), geramos scores sintéticos com um
    gerador de números aleatórios SEMEADO (`np.random.default_rng`), de
    forma que o vetor `e` seja sempre o mesmo entre execuções. Ver
    DECISOES.md para a justificativa e o plano de substituição por dados
    reais.

    Args:
        tickers: lista de tickers (define N e a ordem do vetor retornado).
        semente_esg: semente do gerador.
        score_min: limite inferior do score sintético.
        score_max: limite superior do score sintético.

    Returns:
        Vetor `e` de scores ESG sintéticos, shape (N,), em [score_min, score_max].
    """
    rng = np.random.default_rng(semente_esg)
    return rng.uniform(score_min, score_max, size=len(tickers))


def score_esg_carteira(w: np.ndarray, esg: np.ndarray) -> float:
    """Score ESG médio (ponderado pelos pesos) da carteira, em [score_min, score_max].

    # CP2 — não usado no Checkpoint 1. Métrica de relatório (sinal positivo,
    # ao contrário de f3_esg que é negado para minimização).
    """
    return float(esg @ w)
