# Discussão de Trade-offs — CP2: Otimização de Portfólio Multiobjetivo

**Disciplina:** Heurísticas e Modelagem Multiobjetivo — UFG  
**Checkpoint:** CP2 — Versão Multiobjetivo (NSGA-II e SPEA-II)  
**Universo:** 24 ações da B3 | Janela: 5 anos | Sementes: 10  

---

## 1. Introdução e Motivação

A otimização de portfólio clássica de Markowitz escalariza risco e retorno em um único objetivo ponderado, produzindo **uma carteira** por execução. Essa abordagem obriga o analista a definir _a priori_ a importância relativa de cada componente — uma escolha que pode esconder estrutura importante no espaço de soluções.

O Checkpoint 2 estende o CP1 com um terceiro objetivo — o **score de sustentabilidade (ESG)** baseado na participação histórica no ISE B3 (2020–2025) — e abandona a escalarização. Em vez de uma única resposta, os algoritmos evolutivos multiobjetivo (MOEAs) NSGA-II e SPEA-II produzem uma **aproximação da fronteira de Pareto**: um conjunto de carteiras que representam compromissos distintos entre retorno esperado, risco (variância) e responsabilidade socioambiental.

Este documento discute os resultados obtidos e os trade-offs revelados pela fronteira.

---

## 2. Formulação do Problema

**Variável de decisão:** vetor de pesos $w \in \mathbb{R}^{24}$, com restrições:

$$\sum_{i=1}^{24} w_i = 1, \qquad 0 \le w_i \le 0{,}20 \quad \forall i$$

(carteira long-only, totalmente investida, com teto de concentração de 20% por ativo).

**Três objetivos (forma de minimização):**

| Símbolo | Definição | Unidade de relatório |
|---------|-----------|----------------------|
| $f_1(w) = -\mu^\top w$ | Retorno esperado anual, negado | % a.a. |
| $f_2(w) = w^\top \Sigma w$ | Variância do portfólio | (retorno)² |
| $f_3(w) = -e^\top w$ | Score ESG ponderado, negado | pontos 0–100 |

onde $\mu$ é o vetor de retornos esperados anualizados (estimados por média amostral sobre 5 anos) e $\Sigma$ é a matriz de covariância anualizada (estimador de Ledoit-Wolf).

**Scores ESG:** derivados da participação histórica no Índice de Sustentabilidade Empresarial da B3 (carteiras 2020–2025). Quanto mais consistente a presença do ativo nas carteiras do ISE, maior o score.

| Score | Ativos representativos |
|-------|------------------------|
| 80–92 | WEGE3 (92), ITUB4 (89), SUZB3 (87), BBDC4 (85), KLBN11 (84), B3SA3 (83), TOTS3 (81), SBSP3 (80) |
| 55–74 | VIVT3 (74), RADL3 (71), ITSA4 (68), ABEV3 (65), EQTL3 (63), CMIG4 (61), BBAS3 (56), RENT3 (55) |
| 15–42 | LREN3 (58)¹, HAPV3 (42), RAIL3 (38), GGBR4 (32), PETR4 (24), CSNA3 (21), PRIO3 (18), VALE3 (15) |

¹ LREN3 está no grupo médio (58) conforme tabela completa no notebook.

**Dominância de Pareto:** a carteira $w_a$ *domina* $w_b$ se $f_i(w_a) \le f_i(w_b)$ para todo $i$ e estritamente menor em ao menos um $i$. Uma carteira é **Pareto-ótima** se nenhuma outra carteira factível a domina. O conjunto de todas as soluções Pareto-ótimas forma a **fronteira de Pareto** — o objeto de interesse dos MOEAs.

---

## 3. Metodologia

### 3.1 Algoritmos testados

| Algoritmo | Tipo | Parâmetros | Mecanismo de diversidade |
|-----------|------|------------|--------------------------|
| **NSGA-II** | MOEA | pop=100, n_gen=150, CR ativ. | Ordenação por não-dominância + crowding distance |
| **SPEA-II** | MOEA | pop=100, n_gen=150, CR ativ. | Arquivo externo + densidade por k-NN no espaço de objetivos |
| **DE (CP1)** | SO | pop=80, n_gen=150, CR=0,9, F=0,8 | Mutação diferencial (DE/rand/1/bin) |

NSGA-II e SPEA-II operam no espaço de **3 objetivos normalizados** (z-score calculado sobre 5.000 carteiras aleatórias factíveis, para equalizar escalas antes do cálculo de distâncias). O DE opera no espaço de **2 objetivos escalarizados** com $\lambda_1 = \lambda_2 = 0{,}5$, sem conhecimento de $f_3$.

Todos os algoritmos usam o mesmo operador de reparo de factibilidade: **projeção no capped-simplex** por bisseção em $\tau$, garantindo $\sum w_i = 1$ e $0 \le w_i \le 0{,}20$ em toda população gerada.

### 3.2 Baselines

| Baseline | Descrição |
|----------|-----------|
| **1/N** | Pesos iguais (1/24 ≈ 4,17% por ativo) — diversificação ingênua |
| **Random Search** | 15.000 carteiras aleatórias factíveis (mesmo orçamento de avaliações dos MOEAs), reduzidas ao subconjunto não-dominado |

### 3.3 Protocolo experimental

- **10 sementes independentes** por algoritmo (sementes 1–10)
- **15.000 avaliações de $f$** por semente para todos os métodos (pop×gen: NSGA-II/SPEA-II = 100×150; DE = 80×150 ≈ 12.000 + avaliações iniciais)
- **Fronteira agregada:** união das populações finais das 10 sementes, filtrada ao subconjunto não-dominado — equivalente multiobjetivo de "pegar o melhor entre as sementes"
- **Hipervolume (HV):** indicador de qualidade calculado sobre os objetivos normalizados, com ponto de referência fixo $r = (4, 4, 4)$ no espaço normalizado (4 desvios-padrão acima da média de carteiras aleatórias — pior que praticamente toda carteira factível em qualquer dimensão)

---

## 4. Resultados

### 4.1 Hipervolume por algoritmo

O hipervolume mede a "área" (volume, em 3D) dominada pela fronteira no espaço de objetivos normalizados. Valores maiores indicam fronteiras de melhor qualidade — mais próximas do ótimo e mais diversas.

**Por semente (pop_size=100, n_gen=150):**

| Semente | HV NSGA-II | HV SPEA-II |
|---------|------------|------------|
| 1 | 256,39 | 247,38 |
| 2 | 254,04 | 242,53 |
| 3 | 253,16 | 245,02 |
| 4 | 251,67 | 244,85 |
| 5 | 256,13 | 250,99 |
| 6 | 250,92 | 243,97 |
| 7 | 253,07 | 245,46 |
| 8 | 251,57 | 247,95 |
| 9 | 257,42 | 247,47 |
| 10 | 254,90 | 245,62 |
| **Média ± dp** | **253,93 ± 2,24** | **246,12 ± 2,39** |
| **Fronteira agregada (10 sementes)** | **263,70** | **254,44** |
| **Soluções não-dominadas agregadas** | **414** | **445** |

**Observação:** cada semente individual produz exatamente 100 soluções na última geração (pop_size=100). A fronteira agregada é maior porque diferentes sementes exploram regiões diferentes da fronteira de Pareto, e a união das 10 populações finais resulta em mais de 1.000 soluções, das quais ~414 (NSGA-II) ou ~445 (SPEA-II) sobrevivem ao filtro de não-dominância.

### 4.2 Métricas em unidades originais

A tabela abaixo resume as métricas de cada método em grandezas interpretáveis. Para os MOEAs, reporta-se a **média sobre todos os pontos da fronteira agregada** — não uma única carteira, pois a fronteira é um conjunto.

| Método | Retorno esperado | Volatilidade anual | Índice de Sharpe | Score ESG |
|--------|------------------|--------------------|-----------------|-----------|
| **DE mono-objetivo (CP1)** — média 10 sementes | **25,77%** | 17,04% | **1,513** | 59,45 |
| **NSGA-II** — fronteira agregada, média dos pontos | 18,82% | 16,37% | 1,147 | **72,38** |
| **SPEA-II** — fronteira agregada, média dos pontos | 18,84% | **15,80%** | 1,188 | **72,66** |
| **1/N** | 8,00% | 18,08% | 0,443 | 60,08 |
| **Random Search** (90 soluções não-dominadas, média) | ~18% | ~17% | ~1,10 | ~70 |

> **Nota sobre o DE:** o DE escalariza apenas $f_1$ e $f_2$ com $\lambda_1=\lambda_2=0{,}5$, sem visibilidade sobre $f_3$. O score ESG de 59,45 é um *subproduto passivo* da alocação ótima risco-retorno — não uma escolha deliberada.

### 4.3 Fronteira inicial vs fronteira final

Para cada algoritmo, a **fronteira inicial** é o subconjunto não-dominado da população aleatória (reparada para ser factível) após a **primeira geração** — antes de qualquer operação de seleção ou cruzamento. A **fronteira final** corresponde à última geração.

| Momento | NSGA-II | SPEA-II |
|---------|---------|---------|
| Geração inicial (pop aleatória reparada) | 20 soluções não-dominadas | 20 soluções não-dominadas |
| Geração final (150 gerações) | 414 soluções (agregado) | 445 soluções (agregado) |
| Crescimento | ~20,7× | ~22,3× |

Todos os pontos da fronteira inicial são **dominados** pela fronteira final — evidência direta de que a busca evolutiva agrega valor real. O HV cresce aproximadamente 70% entre a geração 1 e a geração 150, com o ganho concentrado nas primeiras 30–50 gerações (conforme curvas de convergência no notebook, Seção 12).

### 4.4 O DE mono-objetivo é dominado pelos MOEAs?

**Não.** Em nenhuma das 10 sementes o ponto do DE é dominado por NSGA-II ou SPEA-II. Isso é **esperado e revela uma propriedade importante**: o DE maximiza o trade-off risco-retorno sem restrição de ESG. No espaço (retorno, risco, ESG), o ponto do DE se situa numa região de **alto retorno e ESG moderado** onde não existem soluções dos MOEAs que o superem simultaneamente nos três objetivos.

Formalmente: nenhuma carteira da fronteira de Pareto consegue retorno maior **e** risco menor **e** ESG maior que o DE ao mesmo tempo. O DE é Pareto-eficiente no espaço bidimensional (retorno, risco) — qualquer melhoria de ESG vem ao custo de retorno ou de aumento de risco.

---

## 5. Discussão de Trade-offs

### 5.1 Trade-off retorno × ESG

Este é o trade-off central do CP2. Os dados mostram uma **correlação negativa robusta** entre retorno esperado e score ESG ao longo da fronteira de Pareto.

**Quantificação:**

- O DE (sem ESG) obtém retorno de **25,77%** e ESG passivo de **59,45**
- A fronteira NSGA-II tem retorno médio de **18,82%** e ESG médio de **72,38**
- Para ganhar ~12,93 pontos ESG, o investidor cede ~6,95 p.p. de retorno esperado

Isso não é arbitrário. Os ativos com maior score ESG no universo são bancos (ITUB4=89, BBDC4=85), indústrias com certificações ambientais (WEGE3=92, SUZB3=87, KLBN11=84) e empresas de tecnologia/saneamento (TOTS3=81, SBSP3=80). Esses setores tendem a ter:

1. **Menor retorno esperado** no período analisado (empresas maduras, menor alavancagem operacional)
2. **Menor risco** (setores menos cíclicos, receitas mais previsíveis)

Por outro lado, os ativos de maior retorno esperado no período são commodities (PETR4, PRIO3, VALE3) e siderurgia (GGBR4, CSNA3) — justamente os de **menor score ESG** (15–32). Assim, a busca por retorno naturalmente converge para ativos ESG-baixos.

**Implicação prática:** um investidor que exige ESG ≥ 75 (acima da média da fronteira) opera numa região da fronteira com retorno médio entre 15–20%, cedendo ~5–10 p.p. de retorno em relação ao DE. Para um portfólio de R$ 1 milhão, isso representa R$ 50.000–100.000 por ano de retorno esperado "não realizado" como custo implícito da restrição ESG.

### 5.2 Trade-off retorno × risco (dimensão clássica de Markowitz)

Dentro da fronteira de Pareto, o trade-off retorno × risco permanece presente mas é modulado pelo ESG:

- Carteiras de **alto retorno** (>22%) na fronteira tendem a ter risco >18% e ESG <55 — concentradas em PETR4, PRIO3 e VALE3
- Carteiras de **baixo risco** (<14%) tendem a ter retorno <15% e ESG >75 — dominadas por bancos e utilities
- A **região intermediária** (retorno 17–21%, risco 15–17%, ESG 65–78) é onde a fronteira é mais densa — o "coração" dos trade-offs

A **média da fronteira SPEA-II** (volatilidade 15,80%) fica sistematicamente abaixo da do NSGA-II (16,37%), sugerindo que o SPEA-II, via seu mecanismo de arquivo externo com densidade estimada por $k$-NN, mantém ligeiramente mais soluções na região de baixo risco.

### 5.3 NSGA-II vs SPEA-II: convergência e qualidade da fronteira

**NSGA-II superou o SPEA-II nesta configuração**, tanto em HV médio quanto em estabilidade entre sementes.

| Critério | NSGA-II | SPEA-II | Favorece |
|----------|---------|---------|----------|
| HV médio por semente | 253,93 | 246,12 | NSGA-II (+7,81 = +3,2%) |
| Desvio-padrão do HV | 2,24 | 2,39 | NSGA-II (mais estável) |
| HV fronteira agregada | 263,70 | 254,44 | NSGA-II (+9,26 = +3,6%) |
| Soluções não-dominadas agg. | 414 | 445 | SPEA-II (mais diverso) |

O SPEA-II produziu *mais* soluções não-dominadas (445 vs 414), mas com HV agregado menor — indicando que suas soluções adicionais exploram regiões de menor qualidade (mais perto do ponto de referência) do que o NSGA-II.

**Por que o NSGA-II é melhor aqui?** O mecanismo de crowding distance do NSGA-II distribui soluções igualmente no espaço de objetivos usando distância euclidiana em coordenadas ordenadas por fronteira. O SPEA-II usa $k$-NN no espaço normalizado de objetivos — uma estimativa de densidade mais sofisticada, mas com custo computacional maior por geração e sensibilidade ao valor de $k$ (fixado pelo pymoo). Com o mesmo orçamento de avaliações (15.000) e pop_size idêntico, o NSGA-II converge mais rápido.

### 5.4 Valor da busca evolutiva versus população aleatória inicial

A **fronteira inicial** (população aleatória reparada, sem nenhuma geração de seleção) tinha apenas 20 soluções não-dominadas por algoritmo. A comparação direta com a fronteira final revela o que os MOEAs efetivamente descobriram:

- **Expansão de ~20× no número de soluções não-dominadas** (de 20 para 414/445)
- **Todos os pontos iniciais são dominados pela fronteira final** — nenhum ponto aleatório sobreviveu ao processo evolutivo de 150 gerações
- O **Random Search** com o mesmo orçamento (15.000 carteiras) produziu apenas 90 soluções não-dominadas, muito inferior às 414/445 dos MOEAs

Isso demonstra que a estrutura evolutiva — seleção por dominância de Pareto, preservação de diversidade via crowding/arquivo, operadores de cruzamento e mutação — é decisiva para aproximar a fronteira de Pareto real. O algoritmo não apenas encontra mais soluções; encontra soluções **qualitativamente superiores** em todo o espaço de objetivos.

### 5.5 O custo do ESG em Sharpe ratio

O índice de Sharpe é a métrica mais usada em gestão de portfólios para comparar eficiência risco-retorno. A tabela abaixo decompõe o "custo" do ESG em Sharpe:

| Método | Sharpe | ESG | Δ Sharpe vs DE | Δ ESG vs DE |
|--------|--------|-----|----------------|-------------|
| DE (CP1) — otimiza Sharpe implícito | **1,513** | 59,45 | — | — |
| NSGA-II fronteira — média | 1,147 | 72,38 | −0,366 | +12,93 |
| SPEA-II fronteira — média | 1,188 | 72,66 | −0,325 | +13,21 |
| 1/N | 0,443 | 60,08 | −1,070 | +0,63 |

Cada ponto adicional de ESG custa aproximadamente **0,025–0,028 de Sharpe** (calculado como Δ Sharpe / Δ ESG entre o DE e a média da fronteira NSGA-II). Esse "preço do ESG" é uma quantidade interpretável que poderia orientar um filtro de carteiras para um investidor com preferência ESG explícita.

Importante: **a fronteira de Pareto não minimiza Sharpe** — ela o distribui ao longo de um espectro. Os pontos da fronteira com ESG < 55 tendem a ter Sharpe > 1,35, próximo ao DE; os pontos com ESG > 80 têm Sharpe < 1,00, refletindo o trade-off.

### 5.6 Interpretação dos casos extremos da fronteira

A fronteira de Pareto não é um ponto médio — ela inclui carteiras "extremas" que são igualmente válidas do ponto de vista de não-dominância:

**Extremo de máximo retorno / mínimo ESG:**
- Concentração elevada em PETR4, PRIO3, VALE3 e GGBR4
- Retorno esperado ~28–30%, volatilidade ~20–22%, ESG ~20–30
- Perfil: carteira agressiva de commodities, máxima exposição a riscos climáticos e regulatórios

**Extremo de máximo ESG / mínimo retorno:**
- Concentração em WEGE3, ITUB4, SUZB3, BBDC4, KLBN11
- Retorno esperado ~10–14%, volatilidade ~13–15%, ESG ~85–90
- Perfil: carteira defensiva/ESG, menor exposição setorial mas menor potencial de ganho

**Região eficiente de Sharpe:**
- Balanceia retorno e risco com ESG moderado (~65–75)
- Sharpe ~1,2–1,4, retorno ~20–24%, volatilidade ~16–18%
- Perfil: próxima ao portfólio de mercado eficiente, com viés ESG moderado

---

## 6. Testes Estatísticos

Para validar que a diferença de HV entre NSGA-II e SPEA-II não é ruído amostral, aplicamos o **teste de Mann-Whitney U bilateral** (não-paramétrico, robusto a não-normalidade) com **correção de Bonferroni** para $k=3$ comparações simultâneas (gerações 50, 100 e 150).

**Hipótese nula:** as distribuições de HV das 10 sementes de NSGA-II e SPEA-II são idênticas.

**Tamanho de efeito:** correlação rank-biserial $r_{rb} = 2U/(n_1 n_2) - 1 \in [-1,+1]$. Interpretação: $|r| < 0{,}10$ negligível, $< 0{,}30$ pequeno, $< 0{,}50$ médio, $\ge 0{,}50$ grande.

| Geração | HV NSGA-II (μ±σ) | HV SPEA-II (μ±σ) | U | p bruto | p×3 (Bonf.) | Sig. (α=0,05) | $r_{rb}$ | Efeito |
|---------|------------------|------------------|---|---------|-------------|---------------|----------|--------|
| 50 | 238,83 ± 3,93 | 233,87 ± 5,44 | 82 | 0,0173 | 0,0518 | **não** | 0,64 | grande |
| 100 | 248,91 ± 2,74 | 242,99 ± 2,69 | 95 | 0,0008 | **0,0023** | **sim** | 0,90 | grande |
| 150 | 253,93 ± 2,24 | 246,12 ± 2,39 | 99 | 0,0002 | **0,0007** | **sim** | 0,98 | grande |

**Interpretação:**

- **Geração 50:** diferença não significativa após correção (p×3=0,0518 > 0,05), mas o tamanho de efeito já é grande ($r_{rb}=0{,}64$). Os algoritmos ainda estão convergindo — o SPEA-II demora mais para separar a fronteira com seu mecanismo de arquivo.
- **Geração 100:** diferença significativa (p×3=0,0023), com efeito muito grande ($r_{rb}=0{,}90$). A vantagem do NSGA-II é robusta e não atribuível ao acaso.
- **Geração 150 (final):** diferença altamente significativa (p×3=0,0007), efeito quase máximo ($r_{rb}=0{,}98$). Com apenas 10 sementes, U=99 de U_max=100 indica que o HV do NSGA-II **supera o SPEA-II em 9 das 10 comparações par a par**.

**Conclusão:** a superioridade do NSGA-II sobre o SPEA-II neste problema, com este orçamento e esta configuração de parâmetros, é **estatisticamente robusta** e de **efeito prático grande**.

---

## 7. Limitações

### 7.1 Scores ESG como proxy

Os scores derivados do ISE B3 capturam a **consistência de presença** em carteiras de sustentabilidade oficiais, mas não são ratings ESG certificados. Empresas pequenas ou com histórico curto podem ter scores baixos por falta de elegibilidade ao ISE, não por má performance ESG. Adicionalmente, os scores são estáticos (baseados em 2020–2025) e não capturam evolução temporal da performance socioambiental.

### 7.2 Dados históricos e estacionaridade

Os parâmetros $\mu$ e $\Sigma$ são estimados sobre retornos históricos de 5 anos. A hipótese implícita de que o passado prediz o futuro é violada por eventos de cauda (choques de commodities, pandemias, mudanças regulatórias). A estimativa de Ledoit-Wolf reduz o risco de estimação da covariância, mas não elimina o risco de não-estacionaridade.

### 7.3 Ausência de custos de transação

A formulação não inclui custos de corretagem, spread bid-ask, liquidez diferenciada entre ativos ou impostos. Carteiras da fronteira que concentram em ativos menos líquidos (KLBN11, TOTS3) podem ter custo de implementação substancialmente maior do que o modelo sugere.

### 7.4 Orçamento de avaliações e convergência

Com 15.000 avaliações e pop_size=100, a fronteira de Pareto **ainda não convergiu completamente** — as curvas de HV mostram crescimento até a geração final (150). Com orçamentos maiores (n_gen=300–500), ambos os algoritmos provavelmente produziriam fronteiras de maior qualidade, possivelmente reduzindo a diferença entre NSGA-II e SPEA-II.

### 7.5 Hipervolume como único indicador de qualidade

O HV é um indicador escalar de qualidade da fronteira, mas não captura todas as dimensões relevantes: distribuição de soluções ao longo da fronteira (spread), proximidade à fronteira de Pareto verdadeira (GD, IGD) ou cobertura de regiões específicas. A análise aqui é suficiente para o CP2 mas seria complementada com GD/IGD em estudos mais aprofundados.

---

## 8. Síntese dos Trade-offs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              MAPA DE TRADE-OFFS: Retorno × Risco × ESG                     │
│                                                                              │
│  ESG ▲                                                                       │
│  90  │ ···• (WEGE3, ITUB4, SUZB3)                                           │
│      │   Fronteira SPEA-II / NSGA-II — extremo ESG                          │
│  75  │     [baixo retorno, baixo risco, alto ESG]                           │
│  ──  │ ─ ─ ─ ─ ─ Média fronteira (72,38 / 72,66) ─ ─ ─ ─ ─                │
│  65  │      ★  DE mono-objetivo (25,77%, 17,04%, 59,45)                     │
│  ──  │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                      │
│  50  │       [alto retorno, alto risco, baixo ESG]                          │
│  20  │            ••• (PETR4, VALE3, PRIO3)                                 │
│      └─────────────────────────────────────────────────────→ Retorno        │
│           8%    14%    18%    22%    26%    30%                              │
└─────────────────────────────────────────────────────────────────────────────┘

★ DE (CP1): máximo Sharpe (1,513) no espaço risco-retorno, ESG passivo
● Fronteira Pareto: ~414–445 soluções cobrindo todo o espectro de trade-offs
```

**Três conclusões centrais:**

1. **A inclusão do ESG como terceiro objetivo revela um espectro de carteiras que o DE não consegue acessar.** O DE encontra o ótimo de Sharpe (1,513) mas não pode navegar o eixo ESG — qualquer combinação retorno-risco-ESG que não maximize Sharpe é invisível para ele.

2. **A busca evolutiva multiobjetivo é indispensável.** Uma população aleatória inicial tem apenas 20 soluções não-dominadas; os MOEAs expandem isso para 414–445 após 150 gerações, com todos os pontos iniciais dominados pela fronteira final.

3. **O NSGA-II supera o SPEA-II com significância estatística e efeito grande nesta configuração,** mas ambos revelam a mesma estrutura de trade-offs: cada ponto ESG adicional custa ~0,025–0,028 de Sharpe, e os setores de commodities/energia são a principal fonte de retorno esperado e de risco ESG.

---

*Todos os números deste documento provêm dos outputs executados do notebook `cp2_otimizacao_portfolio_multiobjetivo.ipynb` e dos CSVs em `results/`. Para reproduzir: executar o notebook do início ao fim com `CONFIG["sementes"] = [1, 2, ..., 10]`.*
