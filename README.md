# Detecção de Fraude em Transações Financeiras

Pipeline de Machine Learning end-to-end para detecção de fraude em transações financeiras, com engenharia de features, avaliação sob desbalanceamento severo de classes, explicabilidade (SHAP) e serving via API containerizada.

**Status:** em desenvolvimento (semanas 1 a 4 concluídas: EDA, features, modelagem e explicabilidade), projeto pessoal de portfólio.

---

## Problema

Detecção de fraude é um problema de classificação binária com desbalanceamento extremo de classes (tipicamente <5% de transações fraudulentas). O objetivo não é maximizar acurácia (um modelo que sempre prevê "não fraude" já acerta >95% e é inútil), mas maximizar a detecção de fraude (recall) mantendo o custo operacional de falsos positivos (que bloqueiam transações legítimas) em um nível aceitável para o negócio.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle): ~590 mil transações reais anonimizadas, com colunas categóricas e numéricas, valores ausentes informativos e desbalanceamento de ~3.5% de fraude.

O dataset não está incluído no repositório (ver `data/raw/README.md` para instruções de download).

## Roadmap

- [x] **Semana 1: EDA e limpeza.** Investigar padrões de valores ausentes, definir split temporal (não aleatório) usando `TransactionDT`.
- [x] **Semana 2: Feature engineering.** Features de velocidade/frequência por entidade, agregações, encoding de categóricas de alta cardinalidade.
- [x] **Semana 3: Modelagem e avaliação.** Baseline (regressão logística) → XGBoost/LightGBM. Avaliação com PR-AUC e recall a FPR fixo, não acurácia.
- [x] **Semana 4: Explicabilidade.** SHAP para importância de features por predição individual.
- [ ] **Semana 5: Serving.** API em FastAPI + Dockerfile.
- [ ] **Semana 6: Deploy.** Deploy em Render/Railway + interface simples de demonstração.

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
│   ├── evaluate.py     ← métricas e relatório de avaliação
│   └── explain.py      ← explicabilidade com SHAP (global e por transação)
├── api/
│   ├── app.py           ← API FastAPI para serving do modelo
│   └── schemas.py       ← schemas de request/response (Pydantic)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_evaluate.py
│   └── test_explain.py
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

Com os dados baixados em `data/raw/` (ver `data/raw/README.md`):

```bash
python -m src.train      # features, baseline, XGBoost e ablação; salva modelos e métricas
python -m src.explain    # SHAP global, gráficos de dependência e casos TP/FP/FN
pytest                   # testes
```

`src.explain` usa o modelo salvo por `src.train`; não retreina.

## Métricas-alvo (a preencher conforme os resultados)

Teste temporal (últimos 20%, 118.108 transações, 3.44% de fraude). Detalhes em `docs/metodologia.md`, seção 5.

| Modelo | PR-AUC | ROC-AUC | Recall @ FPR 1% | Observações |
|---|---|---|---|---|
| Regressão Logística (baseline) | 0.138 | 0.814 | 0.6% | 213 colunas numéricas; quebra com outliers fora da faixa do treino |
| XGBoost sem features da Semana 2 | 0.517 | 0.888 | 43.2% | ablação, 431 colunas |
| **XGBoost** | **0.522** | **0.890** | **43.9%** | 451 colunas, 1.469 árvores |

## Licença

MIT
