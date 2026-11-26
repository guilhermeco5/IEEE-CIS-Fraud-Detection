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
- **Modelos testados:** regressão logística (baseline) → XGBoost (`src/train.py`, `python -m src.train`).
- **Protocolo de validação, todo temporal.** Dentro do treino, os 20% finais no tempo viram validação de early stopping (parada após 100 rodadas sem melhora de PR-AUC). Depois o modelo é retreinado no treino inteiro com o número de árvores escolhido. O teste é usado uma única vez, só para reportar: nada (nem número de árvores, nem limiar) é escolhido olhando para ele.
- **Baseline:** imputação pela mediana + padronização + regressão logística balanceada, usando só as colunas numéricas com ≤50% de ausência no treino (213 colunas). A logística não aceita `category` nem NaN, e imputar centenas de colunas quase vazias só adiciona ruído e memória.
- **XGBoost:** `tree_method="hist"` com `enable_categorical=True` (usa as colunas `category` nativamente, complementando o frequency encoding da seção 3), `scale_pos_weight` = negativos/positivos do treino (~28). O peso desloca as probabilidades (o modelo não fica calibrado), mas as métricas usadas só dependem da ordenação.
- **Custo:** o pipeline completo levou ~59 min numa máquina com 8 GB de RAM e 12 threads (XGBoost ~25 min por modelo, entre early stopping e refit).

## 5. Métricas de avaliação

- **Por que não acurácia:** com desbalanceamento severo (~3.5% de fraude), um modelo trivial que sempre prevê "não fraude" atinge >96% de acurácia e é inútil.
- **Métricas usadas:** PR-AUC (Average Precision) e recall a um FPR fixo de 1% — esta última traduzida para a métrica de negócio: "que proporção de fraudes eu detecto, mantendo o bloqueio indevido de clientes legítimos abaixo de 1%?". ROC-AUC entra só como referência (é otimista sob desbalanceamento).
- **Ressalva do limiar:** o limiar que dá FPR ≤ 1% é lido na curva do próprio teste. Isso responde "qual o recall neste ponto de operação", mas em produção o limiar teria que ser fixado antes (na validação), e o FPR real poderia passar um pouco de 1%.

### Resultados (Semana 3)

| Modelo | PR-AUC | ROC-AUC | Recall @ FPR 1% | Árvores |
|---|---|---|---|---|
| Regressão logística | 0.138 | 0.814 | 0.6% | — |
| XGBoost sem features da Semana 2 | 0.517 | 0.888 | 43.2% | 1.500 (teto) |
| **XGBoost** | **0.522** | **0.890** | **43.9%** | 1.469 |

Para referência, um classificador aleatório tem PR-AUC ≈ 0.034 (a taxa de fraude do teste).

- **Baseline quebra nos extremos.** A ROC-AUC de 0.81 mostra que a logística ordena razoavelmente bem no geral, mas o recall a FPR 1% é quase zero. O motivo foi verificado, não é bug de medição: no teste, 1.058 transações recebem probabilidade 1.0 (logit acima de 37, com máximo de 100), e só 9 delas são fraude. São linhas com valores fora da faixa vista no treino (colunas de contagem `C*` e de histórico crescem com o tempo), e um modelo linear sobre features padronizadas mas não limitadas extrapola sem freio. Correção conhecida para uma iteração futura: limitar os valores aos percentis do treino ou aplicar `log1p`/`QuantileTransformer`. Como baseline, o número cumpre o papel de mostrar o piso.
- **XGBoost é ~3.8x melhor em PR-AUC** e detecta 43.9% das fraudes bloqueando no máximo 1% dos legítimos.
- **Features da Semana 2: ganho pequeno.** A diferença entre os dois XGBoost é de +0.005 em PR-AUC e +0.7 p.p. em recall. É positiva, mas pequena, e não foi medida a variância (bootstrap/sementes), então não dá para afirmar que é significativa. A ablação também parou no teto de 1.500 árvores sem acionar o early stopping, então ela poderia melhorar um pouco com mais árvores. A comparação está levemente a favor do modelo completo. Leitura honesta: as 431 colunas originais (em especial os blocos anonimizados `V*` e `C*`, que dominam a importância por gain: `V258`, `V70`, `V257`, `V294`, `V91`…) já capturam boa parte do que velocidade e agregação por `card1` trazem. Isso reforça a hipótese da seção 3 de que `card1` é uma entidade grossa demais, e o uid fica como a melhoria mais promissora de features.
- **Degradação ao longo do teste** (`outputs/metrics/xgboost_por_semana.csv`):

| Semana | Taxa de fraude | PR-AUC | Recall @ FPR 1% |
|---|---|---|---|
| 20 | 3.58% | 0.587 | 51.4% |
| 21 | 3.17% | 0.588 | 50.1% |
| 22 | 2.85% | 0.539 | 47.0% |
| 23 | 3.25% | 0.445 | 39.8% |
| 24 | 3.67% | 0.485 | 39.0% |
| 25 | 4.16% | 0.524 | 39.7% |
| 26 (parcial, 2.754 linhas) | 4.32% | 0.475 | 35.3% |

  O recall cai de ~50% nas duas semanas logo após o treino para ~39% a partir da semana 23, sem que a taxa de fraude caia junto. É o padrão esperado de *drift*: os padrões de fraude mudam e o modelo envelhece. Em produção, isso pede retreino periódico (a cada poucas semanas) e monitoramento semanal, exatamente o motivo de a avaliação por semana ter sido pedida na seção 1.

## 6. Explicabilidade

Implementado em `src/explain.py` (`python -m src.explain`), testado em `tests/test_explain.py`. Roda sobre o modelo já salvo, sem retreinar.

### Método

- **TreeSHAP nativo do XGBoost** (`Booster.predict(..., pred_contribs=True)`), não o `shap.TreeExplainer`. O cálculo é exato e aceita as categóricas nativas (`enable_categorical`); a biblioteca `shap` entra só nos gráficos.
- **Escala log-odds** (a margem do modelo). Para cada linha, `base + soma(SHAP) = margem`, e a sigmoide da margem reproduz o `predict_proba` com erro máximo de 1.5e-06 na amostra. Há testes de aditividade com colunas numéricas, categóricas e NaN.
- **Amostra de 20.000 linhas do teste** (semente fixa, taxa de fraude natural de 3.59%) para o SHAP global. A amostra só limita tempo e memória; os casos individuais usam o teste inteiro.
- **O valor base (0.222) e os scores não são probabilidades reais.** O `scale_pos_weight` (seção 4) desloca a margem, e por isso há scores de 0.0000 e 1.0000. A leitura útil é relativa: quem empurra o score para cima ou para baixo.

### O que mais pesa

Top 10 por |SHAP| médio: `C13`, `P_emaildomain`, `C1`, `TransactionAmt`, `C14`, `card6`, `card1_freq`, `card1`, `card1_amt_mean_prev`, `C11` (tabela completa, com o rank no gain, em `outputs/metrics/shap_importancia_global.csv`; gráficos em `outputs/figures/shap_bar_top20.png` e `shap_beeswarm_top20.png`).

- **Consistente com a intuição:** o valor da transação, o e-mail do comprador, o tipo de cartão e o dispositivo estão entre os principais. As colunas `C*` são, segundo a descrição oficial da competição, contagens (por exemplo, de endereços associados ao cartão). No beeswarm, `C1` alto empurra para fraude e `C13`, `C14` e `D1` altos empurram para legítima, o que lê como "conta com histórico é mais confiável".
- **Não interpretável:** a coluna `V*` (features de engenharia da Vesta) é anonimizada. `V70` e `V258` pesam muito, mas não dá para dizer o que medem. É uma limitação do dataset, não do método.

### SHAP e gain discordam

A correlação de Spearman entre os dois rankings é 0.72. As maiores diferenças: `V258` é a 1ª no gain (6.855) e a 36ª no SHAP; `V257` é a 3ª no gain e a 148ª no SHAP. O gain mede o quanto as divisões das árvores reduzem a perda no treino, e favorece features usadas em poucos splits muito úteis, em casos raros. O |SHAP| médio mede o efeito sobre o score em todas as transações. Para explicar o modelo ao negócio, o SHAP é a medida mais honesta; o gain fica como diagnóstico secundário.

### Features da Semana 2

- **Fatia pequena do total:** as 20 colunas respondem por **11.8% do |SHAP| médio**, e 6 delas têm contribuição desprezível (`*_is_missing`, `has_identity` e `R_emaildomain_freq`, todas abaixo de 0.004). É coerente com a ablação (PR-AUC 0.517 contra 0.522): ajudam pouco no agregado.
- **Mas duas entram no top 10:** `card1_freq` (7ª) e `card1_amt_mean_prev` (9ª). Parte do que elas capturam o modelo já aprenderia de `card1` e `TransactionAmt`, o que explica o ganho pequeno apesar do peso individual.
- **Gráficos de dependência** (`shap_dependence_semana2.png`) contra as hipóteses da seção 3:

| Feature | Hipótese | O que o SHAP mostra |
|---|---|---|
| `card1_amt_ratio_mean` | valor acima do padrão é sinal de risco | **Confirmada.** SHAP sobe de -0.20 (quintil mais baixo) para +0.05 (mais alto) |
| `card1_amt_mean_prev` | padrão histórico de gasto | SHAP sobe de -0.41 para +0.08; cartões com gasto histórico baixo são menos arriscados |
| `card1_cnt_24h` | rajadas indicam fraude | **Não confirmada.** Fica em ~0 até 21 transações em 24h e cai para -0.15 acima disso. Bate com a não monotonicidade da seção 3: contagem alta vem de `card1` populares que agregam muitos clientes |
| `card1_secs_since_prev` | rajadas indicam fraude | **Efeito quase nulo** (SHAP entre -0.06 e +0.02 nos quintis), apesar do sinal visto no treino na seção 3 (5.16% contra 2.45%). Uma explicação possível, não testada, é que as colunas `C*` já capturam esse padrão |
| `card1_freq` | categoria rara é mais suspeita | **Direção oposta.** O quintil mais raro tem SHAP -0.33 (menos fraude). Nesse quintil estão os `card1` que não aparecem no treino (frequência 0). Não investigado |
| `card1_n_prev` | cartão sem histórico é mais arriscado | **Não confirmada.** O SHAP é levemente negativo nos extremos (-0.07 no menor, -0.12 no maior), sem tendência clara |

A leitura geral: de todas as hipóteses da Semana 2, só a de valor atípico se confirma com clareza no modelo. As de velocidade, que eram a aposta principal, ficaram fracas. Isso reforça a ideia da seção 3 de que `card1` é uma entidade ruim e de que um uid sintético é a melhoria mais promissora.

### Leitura de três casos (limiar de FPR 1% = 0.7169)

Figuras em `outputs/figures/shap_waterfall_{tp,fp,fn}.png`.

- **Fraude detectada (score 1.0000):** o maior peso é o aparelho (`DeviceInfo` = SM-G920P, +3.12), seguido de `C1` = 86 (+1.77).
- **Alarme falso (score 0.9995):** os mesmos sinais do acerto, `C1` alto (+1.47) e `V258` alto (+1.44). A única feature da Semana 2 relevante, `card1_amt_mean_prev` (-0.37), puxou para legítima mas não bastou. O modelo erra aqui pelos mesmos motivos pelos quais acerta, ou seja, não há como separar esse caso olhando só as features principais.
- **Fraude não detectada (score ~0):** o perfil do dispositivo escondeu a fraude: resolução 1152x864 (-2.01), Firefox 52 (-1.94) e Windows Vista (-0.85), além de `D1`, `D8` e `D13` em torno de 208 (cartão com histórico). É uma configuração rara que o modelo associou a clientes legítimos.

### Limitações

- Features anonimizadas (`V*`, `C*`, `D*`, `M*`) impedem traduzir parte da explicação para o negócio.
- Features correlacionadas (as colunas `V*` vêm em grupos) dividem o crédito entre si, então o |SHAP| de uma isolada subestima o grupo.
- SHAP descreve o modelo, não a causa da fraude: "`C1` alto empurra o score" não significa que `C1` cause fraude.
- Amostra de 20 mil linhas do teste e três casos individuais são ilustrativos, não uma auditoria. Os casos de FP e FN foram escolhidos como os extremos, não como os típicos.
- As explicações valem para o período do teste; com o drift da seção 5, o ranking pode mudar após um retreino.

## 7. Serving e deploy

Implementado em `src/serving.py` (núcleo, sem FastAPI) e `api/` (contrato HTTP), testado em `tests/test_serving.py` e `tests/test_api.py`. O deploy é da Semana 6; esta seção cobre a API e a imagem Docker.

### O problema central: paridade treino/serving

O modelo usa 451 colunas, 20 delas criadas na Semana 2 a partir do histórico do cartão (`card1`) e de um frequency encoder ajustado no treino, além de 31 colunas `category` que precisam ter as categorias do treino. Servir isso calculando as features "de outro jeito" produz *skew* silencioso: a API responde, os números parecem razoáveis e estão errados.

- **Sem segunda implementação.** Para pontuar uma transação, `FraudScorer` monta o histórico do cartão mais a nova linha e chama as mesmas funções do treino (`add_velocity_features`, `add_aggregation_features`, `add_missing_flags`, `apply_frequency_encoder`). Como elas só olham o passado estrito (seção 3), o resultado é o do batch por construção.
- **Detalhes que causariam divergência.** `TransactionAmt` é float32 no treino (o `data_loader` faz downcast), então a linha nova e o histórico também são float32; senão as médias e desvios mudam no arredondamento. As categorias vêm do treino, não do teste. Empates de timestamp ficam de fora do passado, como no batch.
- **Verificação nos dados reais** (`python -m src.serving --check`, 300 linhas aleatórias do teste, `record=False`): **0 das 451 colunas divergem** e a diferença máxima de score contra o `predict_proba` em lote é **0.0**. Um teste com dados sintéticos pontua o teste em ordem de tempo, gravando cada transação, e compara as 451 colunas com o `build_features` em lote. Quebrar a paridade de propósito (histórico vazio) derruba 8 colunas, então o teste detecta o erro que se propõe a detectar.
- **Uma chamada real pela API** (`uvicorn` + `curl`, a fraude detectada da seção 6, TransactionID 3540441) devolveu score 0.999994, igual ao batch, e os mesmos cinco fatores SHAP do waterfall.

### Contrato da API

- **Entrada:** cinco campos obrigatórios (`transaction_dt`, `transaction_amt`, `card1`, `product_cd`, `transaction_id` opcional) mais um dict `features` com qualquer das 434 colunas brutas, pelo nome do dataset. O que faltar é NaN, que o XGBoost trata como ausente. Escolhi esse formato em vez de ~15 campos tipados porque o modelo usa centenas de colunas; só campos tipados o deixariam rodando quase sempre com NaN.
- **Campos inválidos não derrubam a chamada.** Coluna desconhecida, categoria nunca vista no treino e valor não numérico viram NaN e voltam em `ignored_features` com o motivo, para o chamador perceber que o modelo não usou aquilo. Corpo mal formado (valor <= 0, campo obrigatório faltando) retorna 422.
- **Saída:** `fraud_score`, `is_fraud_flag`, `risk_tier`, `threshold` e as cinco `top_factors` (SHAP, em log-odds). A explicação vem na mesma resposta porque o `DMatrix` é montado uma vez e serve ao score e às contribuições.
- **`fraud_score` não é probabilidade calibrada.** O `scale_pos_weight` (seção 4) desloca a margem, o que explica scores de 0.0000 e 1.0000. Por isso o campo não se chama `fraud_probability`: serve para ordenar e comparar com o limiar.
- **Faixas de risco por limiar, não por score fixo.** `alto`: acima do limiar de FPR 1% (0.7169); `médio`: acima do de FPR 5% (0.4032); `baixo`: abaixo. A flag usa o mesmo limiar de `alto`, o ponto de operação das métricas da seção 5.
- **`GET /health`** devolve 503 se o modelo não carregou, e o `/predict` também, com a mensagem dizendo quais comandos rodar. A API sobe mesmo sem artefatos, para o erro aparecer no `/health` e não como container que reinicia em loop.

### Histórico em memória

As features de velocidade e agregação precisam do histórico de cada cartão. O bundle (`models/serving_bundle.joblib`, 3.5 MB) guarda tempo e valor das 590.540 transações de 13.553 cartões, e cada transação pontuada com `record=true` (padrão) entra nele. `record=false` só consulta.

Isso reproduz o treino, mas tem limites que o desenho assume:

- **Estado por processo:** uma réplica só, e reiniciar perde o que foi acrescentado depois do bundle. Em produção seria um feature store (Redis ou similar), com o mesmo cálculo sobre o histórico lido de lá.
- **O histórico já contém o período do teste.** Uma transação com `TransactionDT` dentro desse intervalo "reprisa" o passado; é aceitável para demonstração e é o que permite conferir a paridade com dados reais.
- **Concorrência:** um lock serializa as chamadas; com uma réplica e CPU-bound, isso não é o gargalo.

### Desempenho

Medido numa máquina local, uma requisição por vez: **~145 ms por chamada para um cartão típico e ~225 ms em média numa amostra aleatória do teste**, que tem muitos cartões de histórico longo. O tempo vai para converter as 451 colunas no formato do XGBoost (~46 ms), montar a linha (~42 ms) e calcular o SHAP (~20 ms); a predição em si leva ~3 ms. Dá para acelerar mantendo somas acumuladas por cartão, mas isso reimplementaria a lógica de `features.py` e traria de volta o risco de skew que o desenho evita. Para um scoring online de baixo volume o custo é aceitável; para alto volume é a primeira coisa a otimizar.

### Imagem Docker

`Dockerfile` (Python 3.13-slim, usuário não-root, `HEALTHCHECK` em `/health`, `PORT` configurável para plataformas que injetam a porta) e `requirements-api.txt`.

- **Versões fixas, iguais às do treino.** O modelo e o bundle são pickles; pickles de pandas, scikit-learn e XGBoost não são garantidos entre versões. Por isso a imagem usa o mesmo Python e as mesmas versões do ambiente que gerou os artefatos, e o `requirements-api.txt` deve ser atualizado a cada retreino com outras versões.
- **Dependências enxutas.** Ficam de fora shap, matplotlib, pyarrow, lightgbm, mlflow e jupyter. Isso só é possível porque os imports pesados de `src/explain.py` são tardios (Parte 1 da Semana 5).
- **Só dois artefatos entram na imagem** (`xgboost.joblib` e o bundle de demonstração). Se faltar um, o `COPY` falha no build, em vez de a API subir respondendo 503.
- **`libgomp1` é instalada à parte**, porque o XGBoost usa OpenMP e a imagem slim não a traz.

**Build verificado.** `docker build` e `docker run` foram executados (Docker 29.9.3), e a imagem passou em todas as checagens:

| Verificação | Resultado |
|---|---|
| Build a partir de um clone, sem o dataset | sucesso em 2min30 |
| Tamanho da imagem | 1,31 GB (pandas, numpy, scikit-learn e XGBoost dominam) |
| Subida até `/health` responder | ~9 s |
| `HEALTHCHECK` do Docker | `healthy`, nenhuma falha |
| Usuário do processo | `uid=1000(app)`, não-root |
| Memória em repouso | 154 MB, bem abaixo dos 512 MB do plano gratuito |
| Rotas `/`, `/docs`, `/openapi.json`, `/meta`, `/predict` | 200 |
| Modo demonstração | `record=true` ignorado; histórico intacto em 369 transações |
| Validação de corpo inválido | 422 |
| Logs | nenhum erro ou aviso |

**Paridade confirmada no container:** os cinco exemplos da interface devolveram exatamente os mesmos scores medidos fora dele (0.0225, 0.0719, 0.1556, 0.1606 e 0.9644), com diferença máxima de 0.0000. Ou seja, o empacotamento não alterou o comportamento do modelo.

## 8. Deploy (Semana 6)

### Plataforma

**Render**, plano gratuito, configurado em `render.yaml` (`runtime: docker`, `healthCheckPath: /health`, `autoDeploy`). O deploy sai do próprio repositório: o Render constrói a imagem do `Dockerfile` a cada push na branch principal. A API usa ~220 MB de memória residente depois de carregar o modelo, dentro dos 512 MB do plano.

O que o plano gratuito impõe, e que a demonstração assume:

- **A instância dorme** após cerca de 15 minutos sem tráfego. A primeira chamada depois disso leva alguns segundos, porque o processo sobe e carrega o modelo do zero. A página trata esse caso na mensagem de erro de rede.
- **O histórico em memória reinicia** junto com o processo, voltando ao conteúdo do bundle. Sem efeito prático aqui, porque o modo demonstração não grava nada.
- **Uma réplica, um worker** (`WEB_CONCURRENCY=1`): o histórico é estado por processo (seção 7), então mais de um worker daria respostas diferentes conforme o worker que atendesse.

### O que é publicado, e o que não é

O bundle completo tem `card1`, instante e valor das 590.540 transações do dataset da competição. Publicá-lo num repositório e numa imagem redistribuiria dados derivados do dataset, então a demonstração usa um **bundle reduzido** (`models/serving_bundle_demo.joblib`, 40 KB), gerado por `python -m src.serving --demo`:

- **Histórico de três cartões** (369 transações), escolhidos por terem perfis distintos: rajada recente, ticket alto e ticket baixo estável.
- **O frequency encoder de `card1` também é restrito a eles.** Esse encoder lista cada `card1` do treino com sua frequência; mantê-lo inteiro exporia os identificadores que o histórico reduzido esconde.
- **O resto do bundle são agregados** e vai como está: ordem e tipos das colunas, categorias do treino, frequências de domínio de e-mail, dispositivo e navegador, os limiares e a transação de referência (uma mediana ou moda por coluna).

Consequência honesta: **só esses três cartões têm histórico na demonstração**. Qualquer outro `card1` é tratado como cartão novo (`card1_n_prev` = 0, médias NaN, `card1_freq` = 0), que é o comportamento correto do pipeline para um cartão nunca visto, mas significa que os scores da demonstração não reproduzem as métricas da seção 5. Aquelas valem para o teste completo, com o bundle completo.

### Modo demonstração

Com `FRAUD_DEMO_MODE=1`, o servidor ignora `record=true` e nunca grava transações no histórico. A API é pública e não tem autenticação; sem essa trava, qualquer visitante poderia poluir o histórico dos cartões da demonstração e mudar o resultado para os próximos. A página também envia `record=false` em toda chamada, mas a garantia está no servidor, não no cliente. O `/health` e o `/meta` informam `demo_mode`, e a página mostra o aviso a partir dessa resposta.

### Interface

Um HTML único (`api/static/index.html`) servido pela própria API em `/`: sem framework, sem CDN e sem recurso externo, no mesmo container e na mesma origem. O formulário monta o payload a partir do `/meta` (categorias aceitas, cartões com histórico, limiares e transação de referência), em vez de repetir essas listas no front-end.

Duas decisões que vieram de medir, não de supor:

- **A transação de referência é preenchida por padrão.** Um payload esparso fica fora da distribuição de treino: colunas como `C3`, `C13` e `C14` praticamente nunca faltam nos dados reais, e quando faltam o modelo as lê como sinal forte de transação legítima (SHAP de -2.0, -1.4 e -0.8). Com poucos campos, o perfil de risco da seção 6 pontuava 0.0025; com a referência, 0.9644. A caixa é explicada na própria página.
- **O exemplo de rajada não é vendido como sinal de fraude.** Pontuando a mesma transação 2 minutos e 2 dias depois da última do cartão, o score vai de 0.156 para 0.161, ou seja, praticamente não muda. Isso é coerente com a seção 6, onde as features de velocidade se mostraram fracas, e o texto do exemplo diz exatamente isso em vez de sugerir o contrário.

Os cinco exemplos são sintéticos (valores típicos do treino com alterações pontuais), rotulados como tal, e os números citados foram obtidos rodando o scorer, não estimados.

### Limitações

- Os limiares de FPR 1% e 5% foram calculados no teste, como as métricas da seção 5; fixá-los numa validação separada é uma melhoria anotada. Com o drift (seção 5), eles precisam ser recalculados a cada retreino.
- A demonstração usa histórico parcial (três cartões), então os scores dela não representam o desempenho medido na seção 5.
- A API não tem autenticação nem limite de taxa: o modo demonstração protege o estado, mas não impede abuso de CPU. É uma demonstração, não um serviço de produção.
- O modelo e o bundle são pickles com versões fixas; o `requirements-api.txt` precisa acompanhar qualquer retreino feito em outro ambiente.
- Cada retreino versionado acrescenta ~16 MB ao histórico do Git (`xgboost.joblib`). Git LFS seria a saída se isso crescer.
