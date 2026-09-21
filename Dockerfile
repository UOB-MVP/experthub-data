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

# Log exactly what was pulled from git into this image, so a bad checkout (missing/extra
# files) is visible in the Cloud Build log instead of surfacing later as a pipeline failure.
RUN set -eu; \
    echo "==== files pulled from git into the image ===="; \
    find src data -type f | sort; \
    echo "==== total file count: $(find src data -type f | wc -l) ===="

WORKDIR /app/src

ENTRYPOINT ["python", "run_pipeline.py"]
