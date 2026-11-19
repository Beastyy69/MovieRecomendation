# Ollama and FastAPI share one Render Web Service; only FastAPI is public.
FROM ollama/ollama:0.40.16 AS ollama
FROM python:3.14.7-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_URL=http://127.0.0.1:11434 \
    OLLAMA_MODELS=/opt/ollama/models \
    OLLAMA_MODEL=gemma2:2b \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_CONTEXT_LENGTH=2048 \
    OLLAMA_MAX_QUEUE=2 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama
WORKDIR /app
COPY requirements-render.txt ./
RUN pip install --no-cache-dir -r requirements-render.txt
COPY deploy/runtime.py deploy/runtime.py
# Bake Gemma's blobs and manifest into the image, not Git or ephemeral startup downloads.
RUN python deploy/runtime.py --prepare-model
COPY main.py movies.pkl tfidf_matrix.pkl tfidf_vectorizer.pkl indices.pkl movies_metadata.csv ./
EXPOSE 10000
CMD ["python", "deploy/runtime.py"]
