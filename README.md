# Otimização de Portfólio Markowitz (Risco-Retorno-ESG)

Projeto final de **INF0415 — Heurísticas e Modelagem Multiobjetivo** (UFG).

O repositório cobre três entregáveis evolutivos sobre o mesmo problema de
portfólio:

- **Checkpoint 1 (CP1):** modelagem mono-objetivo de Markowitz
  (risco-retorno), otimizado com Differential Evolution e comparado a três
  baselines, com análise estatística sobre 10 sementes.
- **Checkpoint 2 (CP2):** extensão multiobjetivo com ESG real (ISE B3) como
  terceiro objetivo, resolvido com NSGA-II e SPEA-II (pymoo), incluindo
  testes estatísticos Mann-Whitney U + Bonferroni.
- **Pré-Final:** integração CP1 → CP2 via **Warm-Start** (pesos do DE
  injetados na população inicial do NSGA-II) e **XAI Financeiro**
  (RandomForest + feature importances sobre a fronteira de Pareto).

---

## Entregáveis

### Checkpoint 1

**`notebooks/cp1_otimizacao_portfolio.ipynb`**

Notebook autocontido (não importa de `src/`), determinístico e
reproduzível. Contém formulação completa (com LaTeX), código e discussão
dos resultados. Seções:

| Seção | Conteúdo |
|---|---|
| 1–3 | Formulação, setup, dados |
| 4–7 | Objetivos, DE, baselines, métricas |
| 8 | Experimento com 10 sementes |
| 9 | Resultados |
| 10 | Testes Estatísticos (Mann-Whitney U + Bonferroni) |
| 11 | Figuras |
| 12 | Discussão |

### Checkpoint 2

**`notebooks/cp2_otimizacao_portfolio_multiobjetivo.ipynb`**

Notebook autocontido. Reativa ESG como terceiro objetivo com scores reais
(ISE B3) e resolve com NSGA-II e SPEA-II. Seções:

| Seção | Conteúdo |
|---|---|
| 1–3 | Formulação MO, setup, dados |
| 4 | Objetivos MO, ESG real (ISE B3), normalização 3D |
| 5–8 | NSGA-II/SPEA-II, DE mono-obj referência, baselines MO, métricas |
| 9 | Experimento (10 sementes × 2 algoritmos) |
| 10 | Resultados |
| 11 | Testes Estatísticos (Mann-Whitney U + Bonferroni) |
| 12 | Figuras |
| 13 | Discussão de trade-offs |

### Pré-Final

**`notebooks/pre-final_otimizacao_integrada_xai.ipynb`**

Notebook autocontido. Integra DE (CP1) e NSGA-II (CP2) via Warm-Start e
adiciona análise XAI. Usa os mesmos scores ESG reais (ISE B3) do CP2.

| Seção | Conteúdo |
|---|---|
| 1–3 | Formulação Warm-Start/XAI, setup, dados |
| 4 | Objetivos, ESG real (ISE B3), normalização 3D |
| 5 | Metaheurísticas (DE, NSGA-II, SPEA-II) |
| 6 | Warm-Start: `SamplingWarmStart` (injeção de pesos do DE) |
| 7 | Experimento: MOEA Normal vs MOEA Warm-Start (10 sementes) |
| 8 | Resultados e convergência de HV |
| 9 | XAI Financeiro: RandomForest + Feature Importances |
| 10 | Discussão |

---

## Formulação

**Variável de decisão:** pesos `w ∈ R^N` da carteira (N=24 ativos B3), com:

```
soma(w) = 1            (totalmente investido)
0 ≤ w_i ≤ 0.20         (long-only, teto de 20% por ativo)
```

**CP1 — mono-objetivo (Markowitz escalarizado):**

```
min  g(w) = λ₁·f̃₁(w) + λ₂·f̃₂(w)
onde f₁(w) = -μᵀw  (retorno, negado)
     f₂(w) = wᵀΣw  (variância)
     f̃ᵢ   = z-score sobre 5000 carteiras aleatórias (semente 123)
```

**CP2/Pré-Final — multiobjetivo (Pareto):**

```
min  (f₁(w), f₂(w), f₃(w))
onde f₃(w) = -eᵀw  (score ESG negado; e = vetor ISE B3)
```

Os três objetivos são normalizados (z-score) antes de entrar nos MOEAs.

---

## ESG: scores reais ISE B3

Os scores ESG usados no CP2 e no Pré-Final são derivados da **participação
histórica no Índice de Sustentabilidade Empresarial (ISE B3)** nas
carteiras de 2020 a 2025. Fonte: `b3.com.br/indices/indices-de-sustentabilidade`.

| Grupo | Critério | Score |
|---|---|---|
| 1 | Presença consistente (4–5 carteiras ISE) | 80–95 |
| 2 | Presença regular (2–3 carteiras) | 55–75 |
| 3 | Presença rara ou nenhuma (0–1) | 15–45 |

Scores por ticker (ordem decrescente):

| Ticker | Score | Setor |
|---|---|---|
| WEGE3.SA | 92 | Bens industriais |
| ITUB4.SA | 89 | Financeiro |
| SUZB3.SA | 87 | Papel e celulose |
| BBDC4.SA | 85 | Financeiro |
| KLBN11.SA | 84 | Papel e celulose |
| B3SA3.SA | 83 | Financeiro |
| TOTS3.SA | 81 | Tecnologia |
| SBSP3.SA | 80 | Saneamento |
| VIVT3.SA | 74 | Telecom |
| RADL3.SA | 71 | Saúde |
| ITSA4.SA | 68 | Financeiro (holding) |
| ABEV3.SA | 65 | Bebidas |
| EQTL3.SA | 63 | Energia |
| CMIG4.SA | 61 | Energia |
| LREN3.SA | 58 | Varejo |
| BBAS3.SA | 56 | Financeiro |
| RENT3.SA | 55 | Locação |
| HAPV3.SA | 42 | Saúde |
| RAIL3.SA | 38 | Logística |
| GGBR4.SA | 32 | Siderurgia |
| PETR4.SA | 24 | Petróleo |
| CSNA3.SA | 21 | Siderurgia |
| PRIO3.SA | 18 | Petróleo |
| VALE3.SA | 15 | Mineração |

---

## Instalação e reprodução

**Gerenciador de pacotes:** [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

**Reproduzir CP1:**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/cp1_otimizacao_portfolio.ipynb
```

**Reproduzir CP2:**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/cp2_otimizacao_portfolio_multiobjetivo.ipynb
```

**Reproduzir Pré-Final:**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/pre-final_otimizacao_integrada_xai.ipynb
```

**Alternativa interativa (qualquer notebook):**

```bash
uv run jupyter lab
```

Depois: `Kernel > Restart & Run All`.

**Alternativa via linha de comando (pacote modular `src/`):**

```bash
uv run python -m src.run_experiment          # CP1
uv run python -m src.run_experiment_mo       # CP2
uv run python -m src.run_experiment_integrated  # Pré-Final (Warm-Start)
```

---

## Configuração

Nos notebooks, todos os parâmetros ficam na célula `CONFIG` (Seção 2):
tickers, janela de dados, `w_max`, lambdas, parâmetros do DE/NSGA-II/SPEA-II
e sementes. No pacote `src/`, o equivalente é `config/config.yaml`.

---

## Estrutura do repositório

```
notebooks/
  cp1_otimizacao_portfolio.ipynb              Entregável CP1 (mono-objetivo)
  cp2_otimizacao_portfolio_multiobjetivo.ipynb Entregável CP2 (multiobjetivo)
  pre-final_otimizacao_integrada_xai.ipynb    Entregável Pré-Final (Warm-Start + XAI)
  discussao_tradeoffs_cp2.md                  Rascunho de discussão de trade-offs CP2
  _build_notebook.py                          Script utilitário de montagem de notebook
  _inspect.py                                 Script utilitário de inspeção

config/
  config.yaml                Parâmetros dos pipelines modulares (src/)

src/
  data_loader.py             yfinance → mu, Sigma (Ledoit-Wolf opcional)
  objectives.py              f1, f2, normalização z-score, g escalarizada — CP1
  objectives_mo.py           f1, f2, f3 sem escalarização + normalização 3D — CP2/Pré-Final
  constraints.py             Projeção no capped-simplex (reparo de factibilidade)
  optimizer_de.py            Problem + Repair + Callback do pymoo, DE — CP1
  optimizer_moea.py          Problem 3-obj + NSGA-II/SPEA-II (pymoo) + HV — CP2
  optimizer_moea_warmstart.py  Variante com SamplingWarmStart (injeção DE → MOEA)
  baselines.py               Ótimo exato (cvxpy), 1/N, random search — CP1
  baselines_mo.py            1/N e random search no espaço de 3 objetivos — CP2
  metrics.py                 Retorno, risco, Sharpe, gap relativo — CP1
  metrics_mo.py              HV, não-dominância, métricas 3D — CP2/Pré-Final
  esg_cp2.py                 Scores ESG ISE B3 + gerador sintético (legado)
  plots.py                   Convergência e boxplot — CP1
  plots_mo.py                Fronteira inicial/final, HV, overlay — CP2
  xai_analysis.py            RandomForest + MDI feature importances — Pré-Final
  utils.py                   Utilitários compartilhados
  run_experiment.py          Orquestra CP1 → results/ e figures/
  run_experiment_mo.py       Orquestra CP2 → results/ e figures/
  run_experiment_integrated.py  Orquestra Pré-Final → results/ e figures/
  __init__.py

results/
  resultados_agregados.csv          CP1: métricas por método (agregado)
  resultados_detalhados.csv         CP1: uma linha por (semente, método)
  melhores_pesos_de.csv             CP1: pesos ótimos do DE por semente
  metricas_agregadas_mo.csv         CP2: métricas MO por algoritmo (agregado)
  pareto_final_mo.csv               CP2: fronteira de Pareto final agregada
  baselines_mo.csv                  CP2: 1/N e random search no espaço 3D
  comparacao_so_mo.csv              CP2: overlay do DE (CP1) no espaço 3D
  comparacao_convergencia_warmstart.csv  Pré-Final: HV por geração (Normal vs WS)

figures/
  convergencia_de.png               CP1: curva de convergência do DE (10 sementes)
  boxplot_g_final.png               CP1: boxplot de g final por método
  convergencia_hv_mo.png            CP2: HV por geração (NSGA-II e SPEA-II)
  fronteira_inicial_vs_final_mo.png CP2: fronteira inicial vs final
  pareto_pairwise_mo.png            CP2: pares de objetivos da fronteira final
  overlay_referencias_mo.png        CP2: overlay DE/1/N/RS na fronteira MO
  convergencia_warmstart.png        Pré-Final: Normal vs Warm-Start (150 ger.)
  xai_shap_risco.png                Pré-Final: importâncias para risco
  xai_shap_esg.png                  Pré-Final: importâncias para score ESG

data_cache/
  precos_*.csv                      Cache de preços yfinance (evita re-download)
```

---

## Bibliotecas principais

| Biblioteca | Uso |
|---|---|
| [pymoo](https://pymoo.org/) | DE (CP1), NSGA-II, SPEA-II (CP2/Pré-Final), HV, NDS |
| [cvxpy](https://www.cvxpy.org/) | Ótimo exato do QP convexo (baseline CP1) |
| [yfinance](https://github.com/ranaroussi/yfinance) | Download de preços da B3 |
| [scikit-learn](https://scikit-learn.org/) | LedoitWolf (covariância), RandomForestRegressor (XAI) |
| [scipy](https://scipy.org/) | Mann-Whitney U (testes estatísticos CP1 e CP2) |
| matplotlib, pandas, numpy | Análise, visualização, manipulação de dados |
| jupyter / nbconvert | Execução reprodutível dos notebooks |
| pyyaml | Configuração modular (`config/config.yaml`) |

---

## Suposições e defaults

- **Ativos:** 24 ações líquidas da B3 cobrindo setores distintos. Quatro
  tickers cogitados inicialmente (EMBR3, JBSS3, ELET3, CCRO3) retornavam
  "possibly delisted" no yfinance e foram substituídos por ativos de setor
  equivalente — ver comentário na Seção 2 dos notebooks.
- **Janela histórica:** 5 anos até a data de execução (`"today"`). A
  reprodutibilidade da *otimização* é garantida pelas sementes,
  independentemente da janela.
- **w_max = 0.20, w_min = 0.0** (long-only, teto de 20% por ativo).
- **Lambdas CP1:** `(0.5, 0.5)` — aversão ao risco neutra. Configurável.
- **Taxa livre de risco:** 0.0 (Sharpe simplificado).
- **10 sementes** `[1, …, 10]` — mínimo exigido pelo enunciado para análise
  estatística.
- **ESG:** scores reais derivados da participação histórica no ISE B3
  (2020–2025). Detalhes na tabela acima e em `src/esg_cp2.py`.

---

## Resultados de referência

### CP1 (10 sementes)

| Método | g (média) | Sharpe | gap ao ótimo exato |
|---|---|---|---|
| Ótimo exato (cvxpy) | −2,0733 | 1,517 | 0,00% |
| DE | −2,0696 (±0,0015) | 1,512 | ~0,18% (±0,07 p.p.) |
| Random Search (mesmo orçamento) | −1,9507 (±0,0280) | 1,449 | ~5,91% (±1,35 p.p.) |
| 1/N | −0,5523 | 0,443 | ~73,36% |

Testes Mann-Whitney U (DE vs RS, k=3, Bonferroni): p×3=0,0005, r_rb=1,0
(efeito grande) nas três métricas — DE domina RS em todos os 100 pares.

### CP2 (10 sementes × 2 algoritmos)

| Algoritmo | HV médio por semente (±dp) | HV da fronteira agregada |
|---|---|---|
| NSGA-II | 273,61 (±3,81) | 288,15 (409 soluções) |
| SPEA-II | 262,93 (±6,93) | 276,77 (412 soluções) |

Testes Mann-Whitney U (NSGA-II vs SPEA-II, k=3 gerações, Bonferroni):
não-significativo na geração 50 (p×3=1,00); significativo com efeito grande
na geração 100 (p×3=0,0138, r_rb=0,76) e 150 (p×3=0,0030, r_rb=0,88).

### Pré-Final — Warm-Start (10 sementes, ESG real ISE B3)

| Variante | HV médio (±dp) | mín | máx |
|---|---|---|---|
| MOEA Normal | 253,34 (±3,69) | 245,54 | 258,26 |
| MOEA Warm-Start | 256,38 (±1,59) | 253,07 | 259,11 |

O Warm-Start melhora o HV médio em +1,2% e reduz o desvio-padrão entre
sementes em 57% (convergência mais previsível). Fronteiras agregadas:
Normal 399 soluções (HV=264,38), Warm-Start 431 soluções (HV=262,60).

**XAI — Top-5 feature importances (MDI):**
- **Risco:** SUZB3 (60,2%), ABEV3 (14,9%), B3SA3 (6,4%), KLBN11 (4,3%), PETR4 (4,0%)
- **ESG:** PETR4 (74,9%), PRIO3 (15,4%), VALE3 (2,6%), CMIG4 (1,7%), ITUB4 (1,2%)
