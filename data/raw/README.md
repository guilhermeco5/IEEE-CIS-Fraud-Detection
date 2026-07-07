# Dados brutos

Este projeto usa o dataset [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection).

## Como obter

1. Crie uma conta no Kaggle (gratuita) e aceite as regras da competição na página acima.
2. Instale a CLI do Kaggle: `pip install kaggle`
3. Gere seu token de API em https://www.kaggle.com/settings → "Create New Token" e salve em `~/.kaggle/kaggle.json`.
4. Baixe os dados:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw/
cd data/raw && unzip ieee-fraud-detection.zip && rm ieee-fraud-detection.zip
```

Arquivos esperados nesta pasta após o download:
- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

**Nenhum desses arquivos é versionado no Git** (ver `.gitignore` na raiz) — são grandes e não devem entrar no repositório.
