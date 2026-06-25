# Otimização de Portfólio Markowitz (Risco-Retorno-ESG)

Projeto final de **INF0415 — Heurísticas e Modelagem Multiobjetivo** (UFG).

O repositório cobre dois checkpoints:

- **Checkpoint 1**: modelagem completa em código do problema clássico de
  Markowitz (risco-retorno, **mono-objetivo**), otimizado com uma
  metaheurística (Differential Evolution) e comparado a três baselines, com
  análise estatística inicial sobre 5 sementes. O ESG **não** faz parte da
  formulação, do experimento, das métricas nem das figuras deste checkpoint:
  fica isolado em `src/esg_cp2.py`, preservado para reativação no CP2 sem
  precisar reescrever o restante do pipeline.
- **Checkpoint 2**: extensão **multiobjetivo** do mesmo problema, reativando
  ESG como terceiro objetivo e resolvendo com dois algoritmos evolutivos
  multiobjetivo (**NSGA-II** e **SPEA-II**, via `pymoo`), comparada à versão
  mono-objetivo do CP1, a baselines mapeados ao espaço de 3 objetivos e à
  fronteira de Pareto inicial (população não-evoluída). Ver
  "Entregável do Checkpoint 2" abaixo.

## Entregável principal: o notebook

**`notebooks/cp1_otimizacao_portfolio.ipynb`** é o entregável do Checkpoint 1.
É um notebook **autocontido** — define inline toda a modelagem, a
metaheurística e as análises, sem importar nada de `src/` — e roda do início
ao fim em um kernel limpo, de forma determinística. Ele contém a formulação
completa (com LaTeX), o código comentado e a discussão dos resultados.

O pacote `src/` (descrito mais abaixo) implementa exatamente a mesma lógica
de forma modular, e é mantido como base para o Checkpoint 2 e como caminho
de linha de comando alternativo (sem precisar de Jupyter). As duas
implementações foram validadas lado a lado e produzem os mesmos resultados.

## Entregável do Checkpoint 2

**`notebooks/cp2_otimizacao_portfolio_multiobjetivo.ipynb`** é o entregável
do Checkpoint 2 — também autocontido (não importa de `src/`), determinístico
e reproduzível do início ao fim em um kernel limpo. Reativa o terceiro
objetivo ESG (`src/esg_cp2.py`) e resolve o problema com **NSGA-II** e
**SPEA-II** (pymoo nativo), cobrindo os quatro pontos pedidos no enunciado:
versão multiobjetivo rodando (5 sementes por algoritmo), comparação com a
versão mono-objetivo (reexecuta o DE do CP1 sobre os mesmos dados e mapeia o
resultado no espaço de 3 objetivos), baselines ($1/N$ e random search com o
mesmo orçamento dos MOEAs) e a fronteira de Pareto inicial (população não-
evoluída) comparada à fronteira final. A discussão de trade-offs (retorno x
risco x ESG, NSGA-II x SPEA-II, comparação com a escalarização do CP1) está
na última seção do próprio notebook.

O pacote `src/` ganhou os módulos `*_mo.py` (descritos na Estrutura abaixo),
implementando a mesma lógica de forma modular — caminho de linha de comando
alternativo via `uv run python -m src.run_experiment_mo`. Reproduzir:

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/cp2_otimizacao_portfolio_multiobjetivo.ipynb
```

## Formulação

Variável de decisão: pesos `w ∈ R^N` de uma carteira de `N` ativos, sujeitos a:

```
soma(w) = 1            (totalmente investido)
0 <= w_i <= w_max       (long-only, com teto de concentração por ativo)
```

Dois componentes, em forma de minimização:

```
f1(w) = -mu^T w     retorno esperado (negado)
f2(w) = w^T Sigma w risco (variância)
```

Objetivo mono-objetivo deste checkpoint — soma ponderada escalarizada e
**normalizada** (ver `DECISOES.md` sobre por que normalizar):

```
g(w) = lambda1 * f1_norm(w) + lambda2 * f2_norm(w)
```

Esta é a leitura clássica de Markowitz: minimizar `g(w)` equivale a maximizar
a utilidade média-variância `mu^T w - delta * w^T Sigma w`, com
`delta = lambda2/lambda1` funcionando como coeficiente de aversão ao risco.

**Extensão multiobjetivo (Checkpoint 2).** Adiciona um terceiro componente,
`f3(w) = -e^T w` (score ESG negado, `e` sintético — ver `src/esg_cp2.py`), e
**não escaleriza**: os três objetivos `(f1,f2,f3)` são otimizados
simultaneamente por dominância de Pareto (NSGA-II/SPEA-II), produzindo uma
fronteira de carteiras em vez de uma única solução. Formulação completa,
discussão de por que normalizar antes de otimizar mesmo sem escalarização, e
a comparação com a leitura mono-objetivo acima: ver Seção 1 do notebook do
CP2.

## Instalação

Gerenciador de pacotes: **[uv](https://docs.astral.sh/uv/)**.

```bash
uv sync
```

Isso cria o ambiente virtual em `.venv/` e instala todas as dependências
fixadas em `uv.lock` (espelhadas, sem hashes, em `requirements.txt` para
quem preferir `pip install -r requirements.txt`).

## Reprodução (no máximo 3 comandos)

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/cp1_otimizacao_portfolio.ipynb
```

Isso baixa os dados (via yfinance, com cache em `data_cache/`), roda o DE e
os baselines para as 5 sementes configuradas, gera as tabelas e figuras e
grava tudo de volta no próprio notebook executado.

Para abrir o notebook interativamente em vez de executá-lo via nbconvert:
`uv run jupyter lab notebooks/cp1_otimizacao_portfolio.ipynb`, depois
`Kernel > Restart & Run All`.

Alternativa via linha de comando (gera CSVs em `results/` e figuras em
`figures/`, usando o pacote modular `src/`):

```bash
uv run python -m src.run_experiment
```

## Configuração

No notebook, tudo é controlado pela célula `CONFIG` (Seção 2): lista de
tickers, janela de dados, `w_max`, os `lambda`, parâmetros do DE, e as
sementes do experimento. Trocar `CONFIG["sementes"]` de 5 para 10 valores,
por exemplo, não exige nenhuma outra mudança no notebook. No pacote `src/`,
o mesmo papel é desempenhado por `config/config.yaml`.

## Estrutura

```
notebooks/cp1_otimizacao_portfolio.ipynb              Entregável CP1: notebook autocontido (mono-objetivo)
notebooks/cp2_otimizacao_portfolio_multiobjetivo.ipynb Entregável CP2: notebook autocontido (multiobjetivo)
config/config.yaml          Configuração dos dois pipelines modulares (src/)
src/
  data_loader.py             yfinance -> mu, Sigma (c/ Ledoit-Wolf opcional)
  objectives.py               f1, f2, normalização (z-score) e g escalarizada — CP1
  constraints.py              Projeção no capped-simplex (reparo de factibilidade)
  optimizer_de.py             Problem + Repair + Callback do pymoo, DE — CP1
  baselines.py                Ótimo exato (cvxpy), 1/N, random search — CP1
  metrics.py                  Retorno, risco, Sharpe, gap relativo — CP1
  plots.py                    Curva de convergência e boxplot comparativo — CP1
  run_experiment.py           Orquestra o CP1, salva CSV e figuras
  esg_cp2.py                  ESG (f3, gerador sintético, métrica) — reativado no CP2
  objectives_mo.py            f1,f2,f3 sem escalarização + normalização 3D — CP2
  optimizer_moea.py           Problem 3-obj + NSGA-II/SPEA-II (pymoo) + hipervolume — CP2
  baselines_mo.py             1/N e random search no espaço de 3 objetivos — CP2
  metrics_mo.py                Métricas + hipervolume + contagem de não-dominados — CP2
  plots_mo.py                  Fronteira inicial/final, convergência de HV, overlay — CP2
  run_experiment_mo.py         Orquestra o CP2, salva CSV (sufixo _mo) e figuras (sufixo _mo)
results/                     CSVs dos dois pipelines modulares (CP1 sem sufixo, CP2 com sufixo _mo)
figures/                     Figuras dos dois pipelines modulares (CP1 sem sufixo, CP2 com sufixo _mo)
```

## Bibliotecas reutilizadas

- **[pymoo](https://pymoo.org/)** — algoritmo Differential Evolution
  (`pymoo.algorithms.soo.nonconvex.de.DE`, CP1), NSGA-II e SPEA-II
  (`pymoo.algorithms.moo.nsga2.NSGA2`, `pymoo.algorithms.moo.spea2.SPEA2`,
  CP2), indicador de hipervolume (`pymoo.indicators.hv.HV`) e ordenação por
  não-dominância (`pymoo.util.nds.non_dominated_sorting`), além da
  infraestrutura de `Problem`, `Repair` e `Callback` reaproveitada nos dois
  checkpoints.
- **[cvxpy](https://www.cvxpy.org/)** — solver do QP convexo para o ótimo
  exato (baseline de referência para o gap do DE).
- **[yfinance](https://github.com/ranaroussi/yfinance)** — download de
  preços históricos da B3.
- **[scikit-learn](https://scikit-learn.org/)** — `LedoitWolf` para
  shrinkage da matriz de covariância.
- **matplotlib**, **pandas**, **numpy**, **pyyaml** — análise, manipulação
  de dados e configuração.
- **jupyter/nbconvert** — execução reprodutível do notebook entregável.

## Suposições e defaults assumidos

- **Ativos**: 24 ações líquidas da B3, cobrindo setores distintos. Quatro
  tickers inicialmente cogitados (EMBR3, JBSS3, ELET3, CCRO3) retornavam
  "possibly delisted" no yfinance no momento da implementação e foram
  substituídos por ativos de setor equivalente — ver comentário no notebook
  (Seção 2) e em `config.yaml`.
- **Janela histórica**: 5 anos terminando na data de execução (`"today"`).
  Isso significa que `mu` e `Sigma` variam ligeiramente entre execuções em
  datas diferentes (mercado real) — a reprodutibilidade da *otimização* em
  si (DE, baselines) é garantida pelas sementes, independentemente disso.
- **w_max = 0.20**, **w_min = 0.0** (long-only).
- **lambdas default**: `(0.5, 0.5)` — peso igual entre retorno e risco
  (coeficiente de aversão ao risco neutro). Configurável.
- **Taxa livre de risco** (para o Sharpe): `0.0` por simplicidade.
- **5 sementes** (`[1, 2, 3, 4, 5]`), trivialmente extensível para 10 ou mais.
- **ESG**: fora de escopo no Checkpoint 1 (reativado no Checkpoint 2 como
  vetor sintético — ver "Resultados do Checkpoint 2" e `src/esg_cp2.py`).

## Resultados do Checkpoint 1 (execução de referência)

Com os defaults (DE: `pop_size=80`, `n_gen=150`), nas 5 sementes:

| Método | g (média) | Sharpe | gap relativo ao ótimo exato |
|---|---|---|---|
| Ótimo exato (cvxpy) | -2.0811 | 1.503 | 0.00% |
| DE | -2.0765 | 1.498 | ~0.22% |
| Random Search (mesmo orçamento) | -1.9497 | 1.435 | ~6.31% |
| 1/N | -0.5523 | 0.431 | ~73.46% |

O DE converge de forma consistente e estável (desvio-padrão de `g` entre
sementes ≈ `0.002`) para muito próximo do ótimo global do QP, superando
claramente o random search com o mesmo orçamento de avaliações. Detalhes,
figuras e discussão completa em `notebooks/cp1_otimizacao_portfolio.ipynb`.

## Resultados do Checkpoint 2 (execução de referência)

Com os defaults (NSGA-II e SPEA-II: `pop_size=100`, `n_gen=150`), nas 5
sementes, reativando ESG (sintético) como terceiro objetivo:

| Algoritmo | HV médio por semente (±desvio) | não-dominadas por semente | HV da fronteira agregada (5 sementes) |
|---|---|---|---|
| NSGA-II | 269,38 (±7,30) | 100/100 | 281,91 (307 soluções) |
| SPEA-II | 266,91 (±5,77) | 100/100 | 277,50 (304 soluções) |

| Método | retorno | risco | Sharpe | score ESG |
|---|---|---|---|---|
| DE mono-objetivo (CP1, mesmos dados) | 25,70% | 17,12% | 1,501 | 39,59 |
| Fronteira NSGA-II (média) | 18,31% | 16,69% | 1,092 | 59,24 |
| Fronteira SPEA-II (média) | 19,06% | 16,64% | 1,145 | 62,36 |

NSGA-II e SPEA-II convergem para fronteiras muito próximas (HV difere por
menos de 1%). O ponto do DE mono-objetivo não é dominado pela fronteira
multiobjetivo em nenhuma das 5 sementes — está sobre a fronteira eficiente
de retorno-risco, mas no canto extremo que ignora ESG por completo, com
score ESG ~20 pontos abaixo da média da fronteira. A fronteira evolui
claramente entre a população inicial e a final: 100% dos pontos
não-dominados da população inicial agregada são dominados pela fronteira
final, e nenhum ponto da fronteira final é dominado pela inicial. Discussão
completa (incluindo a correlação risco-ESG observada na fronteira) em
`notebooks/cp2_otimizacao_portfolio_multiobjetivo.ipynb`, Seção 12.

## Próximos passos

- Testes estatísticos (Mann-Whitney, correção de Bonferroni) entre métodos
  ainda não foram implementados — a tabela detalhada (uma linha por
  semente x método/algoritmo) já está no formato necessário para isso.
- O vetor ESG continua sintético (placeholder, sem fonte real de dados ESG
  integrada) — ver `src/esg_cp2.py` e a Seção 1 do notebook do CP2 para a
  justificativa e o plano de substituição por dados reais.
