# Imagem da API de scoring. Python 3.13: o mesmo do ambiente que gerou os artefatos.
#
# Antes do build, os artefatos precisam existir em models/ (não são versionados):
#   python -m src.train && python -m src.serving
#
#   docker build -t fraud-api .
#   docker run --rm -p 8000:8000 fraud-api
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# O XGBoost usa OpenMP (libgomp), que a imagem slim não traz.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependências primeiro, para o Docker reaproveitar essa camada quando só o código muda.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY api/ ./api/
# Só o que a API usa (sem o baseline). Se faltar um, o build falha aqui, em vez de a API subir em 503.
COPY models/xgboost.joblib models/serving_bundle.joblib ./models/

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000

# /health responde 503 se o modelo não carregou, o que faz o healthcheck falhar.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'))"]

# Shell form do PORT para plataformas (Render, Railway) que injetam a porta.
CMD ["sh", "-c", "exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
