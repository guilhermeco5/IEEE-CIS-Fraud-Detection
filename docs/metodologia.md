# Metodologia

Este documento registra as decisões técnicas do projeto e a justificativa por trás de cada uma. O objetivo é que qualquer pessoa (incluindo um entrevistador técnico) consiga entender *por que* o pipeline foi construído desta forma, não só *o que* ele faz.

## 1. Dataset e split

- **Dataset:** IEEE-CIS Fraud Detection (Kaggle). 590.540 transações, 434 colunas (transaction + identity via left join por `TransactionID`), taxa de fraude geral de 3.50%.
- **Split:** temporal (não aleatório), usando `TransactionDT`, com proporção **80/20** (`TEST_SIZE_FRACTION = 0.2` em `src/config.py`). Justificativa: split aleatório vaza informação do futuro para o treino, inflando métricas de forma que não se sustenta em produção, onde o modelo só vê o passado — treino sempre cobre o passado, teste sempre cobre o "futuro" em relação ao treino.
- **Números do split:**
  - Treino: 472.432 linhas (~140 dias), 16.599 fraudes, taxa de 3.51%.
  - Teste: 118.108 linhas (~42 dias), 4.064 fraudes, taxa de 3.44%.
  - A taxa de fraude se mantém estável entre treino, teste e o dataset geral (3.50%) — o corte temporal não distorce o desbalanceamento a ponto de comprometer a avaliação.
- **Por que 80/20 e não 70/30:** um corte 70/30 (quantil 0.7 de `TransactionDT`) foi cogitado durante a exploração inicial do EDA, mas descartado. Não há sinal nos dados que justifique abrir mão de 10 p.p. de dados de treino — o teste de 118 mil linhas / 4.064 fraudes já é grande o suficiente para uma avaliação estável, e reduzir o treino reduziria a quantidade de exemplos de fraude disponíveis para o modelo aprender (classe já rara em 3.5%).

## 2. Tratamento de valores ausentes

- **Achado principal:** a ausência de dados é fortemente informativa, e na direção contra-intuitiva. Em ~87 colunas (majoritariamente o bloco `id_*` do identity, `DeviceType` e `R_emaildomain`), a taxa de ausência é **muito menor entre transações fraudulentas (~45.7%) do que entre legítimas (~77.3%)** — diferença de ~31 p.p.
- **Interpretação:** transações fraudulentas têm muito mais chance de carregar dados de identidade/dispositivo preenchidos do que transações legítimas. Como a tabela `identity` só cobre uma fração das transações (join proposital via `how="left"` em `src/data_loader.py`), a própria presença desses dados parece correlacionar com o fluxo de verificação/risco que a transação passou — não é ruído, é sinal.
- **Impacto na estratégia de imputação:** não faz sentido aplicar `dropna`/`fillna` ingênuo nessas colunas, o que apagaria esse sinal. A estratégia definida para a Semana 2 é: criar *flags* binárias explícitas de "ausência" (`{coluna}_is_missing`) para as colunas com maior diferença entre classes, preservando o padrão de ausência como feature, além de qualquer imputação de valor feita separadamente.
- Nenhuma coluna chega a ter diferença > 50 p.p. entre classes — o sinal é real mas não determinístico sozinho, reforçando que ausência deve virar *feature auxiliar*, não um discriminador único.

## 3. Engenharia de features

*A preencher na Semana 2 — listar cada feature criada e a hipótese de negócio por trás dela.*

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
