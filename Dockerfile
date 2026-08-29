FROM python:3.11-slim

WORKDIR /webapp

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY ingestion/ ./ingestion/
COPY retrieval/ ./retrieval/
COPY eval/ ./eval/

EXPOSE 7860

ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "7860"]
