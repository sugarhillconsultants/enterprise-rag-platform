FROM python:3.11-slim

WORKDIR /webapp

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY ingestion/ ./ingestion/
COPY retrieval/ ./retrieval/
COPY eval/ ./eval/

# The whole point of this deployment is to actually exercise the real
# embedding/reranking path (see docs/architecture.md) — override with
# USE_REAL_MODELS=false at runtime if you want the fast, verified
# TF-IDF/mock stand-ins instead (e.g. for a quick smoke test).
ENV USE_REAL_MODELS=true

EXPOSE 7860

ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "7860"]
