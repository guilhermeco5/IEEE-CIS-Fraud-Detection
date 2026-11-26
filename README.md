# Detecção de Fraude em Transações Financeiras

Pipeline de Machine Learning end-to-end para detecção de fraude no dataset IEEE-CIS: split temporal, features de velocidade sem vazamento, avaliação sob desbalanceamento severo, explicabilidade com SHAP e serving com paridade treino/serving verificada.

> **Em 30 segundos**
> - **Resultado:** XGBoost detecta **43.9% das fraudes bloqueando no máximo 1% dos clientes legítimos** (PR-AUC 0.522, contra 0.034 de um classificador aleatório).
> - **Avaliação honesta:** split temporal (o teste é o "futuro"), métrica de negócio (recall @ FPR 1%) em vez de acurácia, e a queda de desempenho por semana medida (~50% → ~39% de recall), o que indica *drift*.
> - **Status:** pipeline completo, das semanas 1 a 6: EDA, features, modelagem, SHAP, API e interface de demonstração. Falta só publicar no Render (passo manual). Projeto pessoal de portfólio.

---

## Resultados

Teste temporal: os últimos 20% das transações (118.108 linhas, 3.44% de fraude).

| Modelo | PR-AUC | ROC-AUC | Recall @ FPR 1% | Observações |
|---|---|---|---|---|
| Regressão Logística (baseline) | 0.138 | 0.814 | 0.6% | quebra com valores fora da faixa do treino |
| XGBoost sem features da Semana 2 | 0.517 | 0.888 | 43.2% | ablação, 431 colunas |
| **XGBoost** | **0.522** | **0.890** | **43.9%** | 451 colunas, 1.469 árvores |

Fonte: [`outputs/metrics/comparacao_modelos.csv`](outputs/metrics/comparacao_modelos.csv). Análise completa em [`docs/metodologia.md`, seção 5](docs/metodologia.md#5-métricas-de-avaliação).

## Principais achados

1. **Acurácia engana.** Com ~3.5% de fraude, prever sempre "legítima" acerta mais de 96% e não detecta nada. Por isso as métricas são PR-AUC e recall a FPR fixo de 1%.
2. **O modelo envelhece.** O recall cai de ~51% na primeira semana do teste para ~35–40% a partir da semana 23, sem queda na taxa de fraude. Em produção, isso pede retreino periódico e monitoramento semanal ([`xgboost_por_semana.csv`](outputs/metrics/xgboost_por_semana.csv)).
3. **As features de velocidade ajudam pouco.** O ganho é de +0.005 em PR-AUC sobre a ablação. As colunas originais (`V*`, `C*`) já capturam boa parte do sinal, e `card1` é uma entidade grossa demais. Um uid sintético é a melhoria de features mais promissora.
4. **SHAP e gain discordam** (Spearman 0.72). `V258` é a 1ª no gain e a 36ª no SHAP. Para explicar o modelo ao negócio, o SHAP é a medida mais honesta.
5. **Os erros têm explicação.** Um alarme falso usa os mesmos sinais de um acerto (`C1` e `V258` altos). Uma fraude não detectada se escondeu atrás de um perfil de dispositivo raro associado a clientes legítimos ([seção 6](docs/metodologia.md#6-explicabilidade)).

## Como rodar

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Baixe os dados em `data/raw/` (instruções em [`data/raw/README.md`](data/raw/README.md); exige conta no Kaggle). Depois:

```bash
python -m src.train              # features, baseline, XGBoost e ablação; salva modelos e métricas
python -m src.train --no-ablation  # mesmo, sem o XGBoost de ablação (mais rápido)
python -m src.explain            # SHAP global, dependência e casos TP/FP/FN (usa o modelo salvo)
python -m src.serving            # gera o bundle de serving a partir do modelo salvo
python -m src.serving --check    # confere que o serving reproduz as features do batch
pytest                           # testes
```

### API de scoring e demonstração

O repositório já traz o modelo e um bundle de demonstração, então a API roda a partir de um clone, **sem baixar o dataset**:

```bash
docker build -t fraud-api .
docker run --rm -p 8000:8000 fraud-api     # http://localhost:8000
```

Sem Docker, com as dependências instaladas:

```bash
FRAUD_BUNDLE_FILE=models/serving_bundle_demo.joblib FRAUD_DEMO_MODE=1 uvicorn api.app:app
```

| Rota | O que faz |
|---|---|
| `/` | página de demonstração: formulário, exemplos prontos e os fatores SHAP de cada score |
| `/docs` | documentação interativa (Swagger) |
| `POST /predict` | pontua uma transação |
| `/health`, `/meta` | estado do serviço e opções aceitas pelo formulário |

Para servir o modelo com o histórico completo, gere os artefatos (`python -m src.train` e `python -m src.serving`, exige o dataset) e rode:

```bash
uvicorn api.app:app              # usa models/serving_bundle.joblib por padrão
```

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "transaction_dt": 14000000, "transaction_amt": 59.0, "card1": 13926, "product_cd": "W",
  "features": {"card4": "visa", "card6": "debit", "P_emaildomain": "gmail.com", "C1": 1, "C13": 5}
}'
```

A resposta traz `fraud_score`, `is_fraud_flag`, `risk_tier`, `threshold` e as 5 features que mais pesaram (SHAP). Só 5 campos são obrigatórios; o resto das colunas do dataset vai em `features` e, se faltar, é tratado como ausente. `?record=false` consulta sem gravar a transação no histórico do cartão. O `fraud_score` **não é uma probabilidade calibrada**: serve para ordenar e comparar com o limiar. Detalhes e limitações em [`docs/metodologia.md`, seção 7](docs/metodologia.md#7-serving-e-deploy).

**O que a demonstração não é.** O bundle público tem o histórico de apenas três cartões (os do formulário); qualquer outro `card1` é tratado como cartão novo. Os scores da demonstração, portanto, não reproduzem as métricas da tabela acima, que valem para o teste completo. Ver [`docs/metodologia.md`, seção 8](docs/metodologia.md#8-deploy-semana-6).

### Publicar no Render

A configuração está em [`render.yaml`](render.yaml) (Docker, plano gratuito, health check em `/health`). No painel do Render: **New → Blueprint**, apontando para este repositório. Nada precisa ser enviado à mão, porque os artefatos da demonstração estão versionados.

O plano gratuito suspende a instância após ~15 minutos sem tráfego, então a primeira chamada depois disso leva alguns segundos.

## Em andamento

**Semana 6: Deploy**
- [x] Interface de demonstração (`/`), servida pela própria API
- [x] Artefatos da demonstração versionados e `Dockerfile` apontando para eles
- [x] `render.yaml` com o serviço configurado
- [ ] Publicar no Render e registrar a URL aqui (passo manual, ainda não feito)
- [ ] `docker build` nunca foi executado: o Docker não está instalado na máquina de desenvolvimento e o primeiro build real será o do Render

**Melhorias anotadas**
- uid sintético (`card1` + `addr1` + dia − `D1`) no lugar de `card1` como entidade
- Fixar o limiar de decisão na validação, não no teste
- Medir a variância (bootstrap/sementes) antes de afirmar ganhos pequenos
- Limitar outliers no baseline (`log1p` / `QuantileTransformer`)

## Mapa do repositório

**Código**

| Caminho | Papel |
|---|---|
| `src/config.py` | caminhos e parâmetros centrais |
| `src/data_loader.py` | carregamento e join transaction + identity |
| `src/preprocessing.py` | limpeza e split temporal |
| `src/features.py` | engenharia de features (só passado estrito) |
| `src/train.py` | baseline, XGBoost e ablação |
| `src/evaluate.py` | métricas e avaliação por semana |
| `src/explain.py` | SHAP global e por transação |
| `src/serving.py` | scoring de uma transação com as mesmas features do treino |
| `api/` | API FastAPI (`app.py`) e schemas Pydantic (`schemas.py`) |
| `tests/` | testes de preprocessing, features, avaliação, explicabilidade e serving |
| `api/static/index.html` | página de demonstração servida em `/` |
| `Dockerfile`, `requirements-api.txt`, `render.yaml` | imagem da API, as dependências dela (versões iguais às do treino) e o serviço no Render |

**Referência**

| Caminho | Conteúdo |
|---|---|
| [`docs/metodologia.md`](docs/metodologia.md) | todas as decisões técnicas e o porquê de cada uma |
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | exploração inicial |
| [`data/raw/README.md`](data/raw/README.md) | como baixar o dataset |

**Gerado pelo pipeline** (só as métricas finais são versionadas)

| Caminho | Conteúdo |
|---|---|
| `outputs/metrics/` | comparação de modelos, desempenho por semana, importâncias SHAP e gain |
| `outputs/figures/` | gráficos SHAP e de avaliação (não versionados) |
| `models/` | `xgboost.joblib` e `serving_bundle_demo.joblib` são versionados (a demo precisa deles); o bundle completo, não |
| `data/processed/` | dados após limpeza e split (não versionados) |

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle): 590.540 transações reais anonimizadas, 434 colunas após o join, valores ausentes informativos e 3.50% de fraude. O dataset não está no repositório.

<details>
<summary><strong>Concluído: Semanas 1 a 6</strong></summary>

- **Semana 1: EDA e limpeza.** Padrões de valores ausentes e split temporal (não aleatório) por `TransactionDT`.
- **Semana 2: Feature engineering.** 20 features novas: velocidade e frequência por `card1`, agregações de valor, flags de ausência e frequency encoding de categóricas de alta cardinalidade.
- **Semana 3: Modelagem e avaliação.** Baseline (regressão logística) → XGBoost, com ablação das features da Semana 2. PR-AUC e recall a FPR fixo, por semana do teste.
- **Semana 4: Explicabilidade.** SHAP global, gráficos de dependência e leitura de três casos (acerto, alarme falso e fraude não detectada).
- **Semana 5: Serving.** API FastAPI (`/predict`, `/health`) que calcula as features com as mesmas funções do treino (paridade verificada nos dados reais: 0 colunas divergentes), explica cada score com SHAP e usa um histórico por cartão em memória. `Dockerfile` e `requirements-api.txt` com as versões do treino.
- **Semana 6: Deploy.** Interface de demonstração em `/` (HTML único, sem dependências externas), bundle reduzido a três cartões para não publicar dados do dataset, modo demonstração que impede gravação no histórico e `render.yaml` para o Render.

</details>

## Licença

[MIT](LICENSE)
