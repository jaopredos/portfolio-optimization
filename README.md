# Otimização de Portfólio Markowitz (Risco-Retorno) — Checkpoint 1

Projeto final de **INF0415 — Heurísticas e Modelagem Multiobjetivo** (UFG).

Este é o **Checkpoint 1**: modelagem completa em código do problema clássico
de Markowitz (risco-retorno, **mono-objetivo**), otimizado com uma
metaheurística (Differential Evolution) e comparado a três baselines, com
análise estatística inicial sobre 5 sementes.

O projeto final é uma extensão multiobjetivo do Markowitz incorporando ESG
como terceiro objetivo — mas isso é o Checkpoint 2. Neste checkpoint o ESG
**não** faz parte da formulação, do experimento, das métricas nem das
figuras: ele está isolado em `src/esg_cp2.py` (e em um apêndice isolado do
notebook), preservado para reativação futura sem precisar reescrever o
restante do pipeline. Ver `DECISOES.md`.

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
notebooks/cp1_otimizacao_portfolio.ipynb  Entregável: notebook autocontido (formulação + código + análise)
config/config.yaml          Configuração do pipeline modular (src/)
src/
  data_loader.py             yfinance -> mu, Sigma (c/ Ledoit-Wolf opcional)
  objectives.py               f1, f2, normalização (z-score) e g escalarizada
  constraints.py              Projeção no capped-simplex (reparo de factibilidade)
  optimizer_de.py             Problem + Repair + Callback do pymoo, DE
  baselines.py                Ótimo exato (cvxpy), 1/N, random search
  metrics.py                  Retorno, risco, Sharpe, gap relativo
  plots.py                    Curva de convergência e boxplot comparativo
  run_experiment.py           Orquestra tudo, salva CSV e figuras
  esg_cp2.py                  ESG isolado (f3, gerador sintético, métrica) — não usado no CP1
results/                     CSVs gerados pelo pipeline modular (detalhado, agregado, pesos)
figures/                     Figuras geradas pelo pipeline modular (convergência, boxplot)
```

## Bibliotecas reutilizadas

- **[pymoo](https://pymoo.org/)** — algoritmo Differential Evolution
  (`pymoo.algorithms.soo.nonconvex.de.DE`) e infraestrutura de `Problem`,
  `Repair` e `Callback`.
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
- **ESG**: fora de escopo neste checkpoint (ver seção acima e `DECISOES.md`).

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

## Próximos passos (Checkpoint 2)

- `src/esg_cp2.py` já contém `f3_esg`, o gerador sintético do vetor ESG e a
  métrica de score ESG, isolados e prontos para serem reincorporados.
- `src/objectives.py::avaliar_componentes` devolve hoje só `(f1, f2)`;
  reincorporar `f3` (de `esg_cp2.py`) e um `lambda3` é o ponto de extensão
  direto para o NSGA-II (ou para um CP1 estendido com três objetivos
  escalarizados).
- A normalização e as restrições (capped-simplex) são reaproveitáveis sem
  alteração para a versão multiobjetivo — só muda a dimensão do vetor de
  objetivos.
- Testes estatísticos (Mann-Whitney, correção de Bonferroni) entre métodos
  ainda não foram implementados — a tabela detalhada (uma linha por
  semente x método) já está no formato necessário para isso.
