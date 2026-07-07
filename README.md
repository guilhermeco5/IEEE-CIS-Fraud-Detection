# Detecção de Fraude em Transações Financeiras

Pipeline de Machine Learning end-to-end para detecção de fraude em transações financeiras, com engenharia de features, avaliação sob desbalanceamento severo de classes, explicabilidade (SHAP) e serving via API containerizada.

**Status:** em desenvolvimento — projeto pessoal de portfólio.

---

## Problema

Detecção de fraude é um problema de classificação binária com desbalanceamento extremo de classes (tipicamente <5% de transações fraudulentas). O objetivo não é maximizar acurácia — um modelo que sempre prevê "não fraude" já acerta >95% e é inútil — mas maximizar a detecção de fraude (recall) mantendo o custo operacional de falsos positivos (que bloqueiam transações legítimas) em um nível aceitável para o negócio.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle) — ~590 mil transações reais anonimizadas, com colunas categóricas e numéricas, valores ausentes informativos e desbalanceamento de ~3.5% de fraude.

O dataset não está incluído no repositório (ver `data/raw/README.md` para instruções de download).

## Roadmap

- [ ] **Semana 1 — EDA e limpeza.** Investigar padrões de valores ausentes, definir split temporal (não aleatório) usando `TransactionDT`.
- [ ] **Semana 2 — Feature engineering.** Features de velocidade/frequência por entidade, agregações, encoding de categóricas de alta cardinalidade.
- [ ] **Semana 3 — Modelagem e avaliação.** Baseline (regressão logística) → XGBoost/LightGBM. Avaliação com PR-AUC e recall a FPR fixo, não acurácia.
- [ ] **Semana 4 — Explicabilidade.** SHAP para importância de features por predição individual.
- [ ] **Semana 5 — Serving.** API em FastAPI + Dockerfile.
- [ ] **Semana 6 — Deploy.** Deploy em Render/Railway + interface simples de demonstração.

## Estrutura do projeto

```
fraud-detection/
├── data/
│   ├── raw/            ← dataset original (não versionado, ver README local)
│   └── processed/      ← dados após limpeza e split temporal
├── notebooks/
│   └── 01_eda.ipynb    ← exploração inicial
├── src/
│   ├── config.py       ← caminhos e parâmetros centrais
│   ├── data_loader.py  ← carregamento dos dados
│   ├── preprocessing.py← limpeza, split temporal
│   ├── features.py     ← engenharia de features
│   ├── train.py        ← treino dos modelos
│   └── evaluate.py     ← métricas e relatório de avaliação
├── api/
│   ├── app.py           ← API FastAPI para serving do modelo
│   └── schemas.py       ← schemas de request/response (Pydantic)
├── tests/
│   └── test_preprocessing.py
├── models/              ← modelos treinados serializados (não versionado)
├── outputs/
│   ├── figures/         ← gráficos gerados
│   └── metrics/         ← métricas salvas
├── docs/
│   └── metodologia.md   ← decisões técnicas documentadas
├── Dockerfile
├── requirements.txt
└── .gitignore
```

## Como rodar

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Instruções de execução completas serão adicionadas a partir da Semana 1, conforme o pipeline for tomando forma.

## Métricas-alvo (a preencher conforme os resultados)

| Modelo | PR-AUC | Recall @ FPR 1% | Observações |
|---|---|---|---|
| Regressão Logística (baseline) | — | — | — |
| XGBoost | — | — | — |

## Licença

MIT
