# Metodologia

Este documento registra as decisões técnicas do projeto e a justificativa por trás de cada uma. O objetivo é que qualquer pessoa (incluindo um entrevistador técnico) consiga entender *por que* o pipeline foi construído desta forma, não só *o que* ele faz.

## 1. Dataset e split

- **Dataset:** IEEE-CIS Fraud Detection (Kaggle).
- **Split:** temporal (não aleatório), usando `TransactionDT`. Justificativa: split aleatório vaza informação do futuro para o treino, inflando métricas de forma que não se sustenta em produção, onde o modelo só vê o passado.

## 2. Tratamento de valores ausentes

*A preencher na Semana 1 — registrar aqui se a ausência de dados é informativa (diferente entre classes) e como isso influenciou a estratégia de imputação.*

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
