# Metodologia

Este documento registra as decisões técnicas do projeto e a justificativa por trás de cada uma. O objetivo é que qualquer pessoa (incluindo um entrevistador técnico) consiga entender *por que* o pipeline foi construído desta forma, não só *o que* ele faz.

## 1. Dataset e split

- **Dataset:** IEEE-CIS Fraud Detection (Kaggle). 590.540 transações, 434 colunas (transaction + identity via left join por `TransactionID`), taxa de fraude geral de 3.50%.
- **Split:** temporal (não aleatório), usando `TransactionDT`, com proporção **80/20** (`TEST_SIZE_FRACTION = 0.2` em `src/config.py`). Justificativa: split aleatório vaza informação do futuro para o treino, inflando métricas de forma que não se sustenta em produção, onde o modelo só vê o passado — treino sempre cobre o passado, teste sempre cobre o "futuro" em relação ao treino.
- **Implementação:** o corte é feito por **valor** de `TransactionDT` (quantil 0.8), não por posição (`iloc` após `sort_values`). Um corte posicional pode dividir transações com o mesmo timestamp entre treino e teste, já que o sort não é estável e a ordem entre empates não é determinística — coberto por teste (`test_temporal_split_keeps_tied_timestamps_together`).
- **Números do split** (reconferidos após o fix acima; não mudaram, o dataset real não tem colisão no ponto de corte):
  - Treino: 472.432 linhas (~140 dias), 16.599 fraudes, taxa de 3.51%.
  - Teste: 118.108 linhas (~42 dias), 4.064 fraudes, taxa de 3.44%.
  - A taxa **agregada** é comparável entre treino, teste e o dataset geral (3.50%). Isso **não** significa estabilidade intra-bloco: a agregação semanal (ver `01_eda.ipynb`, seção "Estabilidade da taxa de fraude ao longo do tempo") mostra a taxa variando de 1.85% (semana 3) a 5.06% (semana 16) — um regime mais baixo nas primeiras 4 semanas (0–3, entre 1.85% e 2.81%) seguido de um patamar mais alto e ruidoso (semanas 4–26, entre 2.85% e 5.06%) que persiste até o fim, incluindo o bloco de teste (semanas 20–26, entre 2.85% e 4.32%). A separação entre os regimes está no nível, não nos extremos: o mínimo do patamar alto (2.85%, semana 22, já no teste) praticamente encosta no máximo das primeiras semanas (2.81%). A última semana (26) é parcial, com 2.754 transações (13% da mediana semanal de 20.860), e é ela que produz o máximo do bloco de teste; a primeira semana (0) não é parcial em volume (114% da mediana). O split não fica comprometido por isso (o teste em si não sofre a virada abrupta que uma leitura só do agregado poderia esconder), mas a avaliação do modelo deve monitorar performance por semana, não só o agregado.
- **Por que 80/20 e não 70/30:** um corte 70/30 (quantil 0.7 de `TransactionDT`) foi cogitado durante a exploração inicial do EDA, mas descartado. Não há sinal nos dados que justifique abrir mão de 10 p.p. de dados de treino — o teste de 118 mil linhas / 4.064 fraudes já é grande o suficiente para uma avaliação estável, e reduzir o treino reduziria a quantidade de exemplos de fraude disponíveis para o modelo aprender (classe já rara em 3.5%).

## 2. Tratamento de valores ausentes

- **Achado bruto (agregado):** em ~87 colunas (majoritariamente o bloco `id_*` do identity, `DeviceType` e `R_emaildomain`), a taxa de ausência agregada é bem menor entre transações fraudulentas (~45.7%) do que entre legítimas (~77.3%) — diferença de ~31 p.p.
- **Esse número é confundido por `ProductCD` — verificado, não só suspeitado.** `ProductCD == W` responde por ~74% das transações e tem cobertura de `identity` igual a **0%, sempre**, fraude ou não; `W` também é o produto com menor taxa de fraude (~2.0%). No outro extremo, `C` tem a maior taxa de fraude (~11.7%) e ~89% de cobertura de `identity`. Isso sozinho já gera a maior parte do gap de 31 p.p. visto no agregado — sem precisar de nenhum mecanismo de risco.
- **Condicionando por `ProductCD`** (`df.groupby(['ProductCD','isFraud'])['id_02'].apply(lambda s: s.isna().mean())`), o efeito sobrevive mas encolhe muito, de ~31 p.p. para poucos p.p., e **não é consistente na direção**: fraude tem menos ausência em C (+6.5 p.p.), H (+2.2 p.p.) e R (+2.3 p.p.), mas **mais** ausência em S (-1.8 p.p.). Em W não há sinal (ambas as classes 100% ausentes).
- **Interpretação revisada:** o achado agregado superestimava um "sinal de risco forte" que era, em boa parte, composição de produto. Há um resíduo de sinal real em alguns produtos (C, H, R), pequeno e não generalizável — não a diferença dramática do número bruto.
- **Impacto na estratégia de imputação:** não faz sentido aplicar `dropna`/`fillna` ingênuo nessas colunas — mas também não faz sentido usar a ausência bruta como feature isolada, dado o confundidor. A estratégia definida para a Semana 2 é: criar *flags* binárias de "ausência" (`{coluna}_is_missing`) **interagindo com `ProductCD`** (ou treinadas em um modelo que já usa `ProductCD` como feature, deixando a árvore capturar a interação), em vez de tratar ausência como discriminador único e agnóstico de produto.

## 3. Engenharia de features

Implementado em `src/features.py` (orquestrado por `build_features(train_df, test_df)`), testado em `tests/test_features.py`. São 20 colunas novas.

### Decisões transversais

- **Entidade = `card1`.** O dataset não tem ID de cliente. `card1` nunca é ausente (13.553 valores) e é a escolha mais simples. O custo conhecido é que ele mistura clientes distintos que compartilham o mesmo `card1` (o mais frequente tem 14.932 transações), o que enfraquece as features de velocidade. Um uid sintético (`card1` + `addr1` + dia − `D1`) foi considerado e deixado como possível melhoria.
- **Só o passado estrito.** Toda feature de entidade de uma transação em `t` usa só transações com `TransactionDT < t`. A própria linha e os **empates de timestamp** ficam de fora: a ordem entre empates não é determinística (mesmo argumento do split, seção 1), então incluí-los tornaria a feature dependente de um detalhe arbitrário do sort. Há um teste dedicado (`test_no_lookahead_future_changes_do_not_affect_past`) verificando que alterar uma transação futura não muda nenhuma feature do passado.
- **Treino e teste concatenados para velocidade e agregações.** Como cada linha só olha o passado, uma linha de teste usar o histórico do treino não é vazamento; é exatamente o que aconteceria em produção. O que **não** pode ver o teste é qualquer coisa ajustada (o encoder), e ele é ajustado só no treino.
- **Implementação vetorizada.** Uma chave ordenada `entidade * BIG + tempo` e `np.searchsorted` respondem "quantas transações desta entidade antes de t" sem loop por grupo. Médias e desvios usam somas acumuladas. O pipeline completo, com o load, roda em cerca de 40s sobre as 590 mil linhas. Uma conferência contra um cálculo ingênuo, em 300 linhas do `card1` mais frequente, não teve nenhuma divergência.

### Features e hipóteses

| Feature | Hipótese de negócio |
|---|---|
| `card1_cnt_1h`, `_6h`, `_24h` | Fraude vem em rajadas: um cartão comprometido gera muitas transações em pouco tempo. |
| `card1_secs_since_prev` | Mesma hipótese, sem depender de uma janela fixa. |
| `card1_n_prev` | Cartões sem histórico (novos ou pouco usados) são mais arriscados e a agregação neles é menos confiável. |
| `card1_amt_mean_prev`, `_std_prev`, `_max_prev` | Padrão histórico de gasto da entidade. |
| `card1_amt_ratio_mean`, `card1_amt_zscore` | Uma transação muito acima do padrão histórico da entidade é um sinal de risco clássico. |
| `{id_01, id_02, DeviceType, R_emaildomain}_is_missing`, `has_identity` | Sinal residual de ausência (seção 2). Entra junto com `ProductCD` categórica para a árvore aprender a interação. |
| `{P_emaildomain, R_emaildomain, DeviceInfo, id_31, card1}_freq` | Categorias raras (domínio obscuro, dispositivo incomum) são mais suspeitas. |

### Encoding de categóricas

Usamos **frequency encoding** (frequência relativa entre os não-ausentes, ajustada só no treino; categoria nunca vista recebe 0 e ausente continua NaN), mais a coluna original como `category` para o suporte nativo do LightGBM/XGBoost. Target encoding foi descartado nesta etapa porque usa o rótulo: exigiria cálculo out-of-fold cuidadoso para não vazar `isFraud`, e o ganho esperado para modelos de árvore é pequeno.

### Primeira leitura do sinal (treino, taxa de fraude por quintil)

- `card1_secs_since_prev`: 5.16% no quintil mais curto contra 2.45% no mais longo. A hipótese de rajada se confirma.
- `card1_amt_ratio_mean`: sobe de forma monotônica de cerca de 2.8% para 4.94% no quintil mais alto.
- `card1_cnt_24h`: **não monotônico** (2.43%, 3.50%, 4.31%, 4.07%, 3.26%). A contagem alta vem de `card1` muito populares que agregam muitos clientes, e esse é justamente o custo de usar `card1` como entidade. Fica como evidência a favor de testar o uid na iteração seguinte.
- `has_identity`: cerca de 8.6% com identidade contra cerca de 2% sem. Isso é em grande parte composição de `ProductCD` (seção 2), por isso a flag não entra sozinha.

## 4. Modelagem e tratamento de desbalanceamento

- **Por que não SMOTE:** gerar exemplos sintéticos de fraude por interpolação entre pontos minoritários pode criar transações que não existem na distribuição real, especialmente em espaços de features de alta dimensão. Optou-se por `scale_pos_weight` (XGBoost) / `class_weight='balanced'` (regressão logística), que ajusta o peso da função de perda sem inventar dados.
- **Modelos testados:** regressão logística (baseline) → XGBoost.

## 5. Métricas de avaliação

- **Por que não acurácia:** com desbalanceamento severo (~3.5% de fraude), um modelo trivial que sempre prevê "não fraude" atinge >96% de acurácia e é inútil.
- **Métricas usadas:** PR-AUC (Average Precision) e recall a um FPR fixo de 1% — esta última traduzida para a métrica de negócio: "que proporção de fraudes eu detecto, mantendo o bloqueio indevido de clientes legítimos abaixo de 1%?"

## 6. Explicabilidade

*A preencher na Semana 4 — registrar os achados do SHAP: quais features mais pesam nas decisões do modelo, e se isso é consistente com a intuição de negócio.*

## 7. Serving e deploy

*A preencher nas Semanas 5–6 — decisões de arquitetura da API, escolha de plataforma de deploy, e qualquer trade-off relevante.*
