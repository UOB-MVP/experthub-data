FROM python:3.13-slim

# ca-certificates: truststore needs the system trust store for Vertex TLS calls.
# build-essential: fallback in case chromadb has no prebuilt wheel for this platform/Python combo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/requirements.txt ./src/requirements.txt
RUN pip install --no-cache-dir -r src/requirements.txt

COPY src/ ./src/
COPY data/ ./data/

WORKDIR /app/src

ENTRYPOINT ["python", "run_pipeline.py"]
