
FROM python:3.14.7-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-render.txt ./

RUN pip install --no-cache-dir \
    -r requirements-render.txt

COPY main.py movies.pkl tfidf_matrix.pkl \
     tfidf_vectorizer.pkl indices.pkl \
     movies_metadata.csv ./

EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
