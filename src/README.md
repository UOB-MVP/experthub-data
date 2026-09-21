# Rebuilding the Expert Hub Gemini vector store

This folder contains the standalone code and reference metadata needed to rebuild the Expert Hub Gemini Chroma store
from the PDFs in `../../data`. It creates the governed collection used by the application; the generated vector store
and credentials are deliberately excluded from Git.

## Files in this folder

- `build_vector_store.py` validates, parses, chunks, embeds, and writes the governed Chroma collection.
- `validate_vector_store.py` checks the completed store's collection, counts, metadata contract, and vector dimensions.
- `source_manifest.json` records the expected source paths, hashes, and historical chunk counts for all 544 PDFs.
- `build_catalog.json` preserves the effective document instances, Chroma ID prefixes, and governed metadata needed by
  the consuming Expert Hub application. It contains metadata only, not document text or embeddings.
- `requirements.txt` pins the Python libraries used by the reference build; `.env.example` documents required settings.

## Reference configuration

The build intentionally preserves the settings used for the vector store in `owg_agent_lab`:

| Setting | Value |
|---|---|
| Embedding model | `gemini-embedding-001` |
| Vertex task type | `RETRIEVAL_DOCUMENT` |
| Output dimensions | 768 |
| Chroma collection | `peer_benchmarking_documents_v1` |
| Distance metric | cosine |
| Maximum chunk size | 6,000 characters |
| Chunk overlap | 400 characters |
| Minimum extracted page text | 80 characters |
| Embedding batch size | 16 |
| ChromaDB | 1.5.9 |

Chunking is page-local. Text is extracted with `pypdf`, oversized pages are split on paragraph boundaries where
possible, and each split after the first carries a 400-character overlap.

## 1. Create the environment

Python 3.13 is recommended. From the repository root on Windows PowerShell:

```powershell
cd scripts\Expert-Hub-gemini
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set the client's Vertex express-mode API key. Never commit `.env`:

```dotenv
EXPERT_HUB_EMBEDDING_MODEL=gemini-embedding-001
VERTEX_API_KEY=<client-secret>
```

## 2. Run the credential-free preflight

The preflight hashes all 544 files and reproduces all chunks without sending data to Gemini or creating a vector
store:

```powershell
.\.venv\Scripts\python.exe build_vector_store.py --dry-run --allow-known-uob-substitution
```

Expected result:

```text
preflight complete: 544 source paths, 534 document instances, 15684 chunks, 1 documented substitution(s)
```

Do not proceed if a different source or chunk count is reported.

## 3. Build the vector store

The command below embeds the full corpus and writes output under the ignored
`scripts/Expert-Hub-gemini/output` directory:

```powershell
.\.venv\Scripts\python.exe build_vector_store.py `
  --env-file .env `
  --allow-known-uob-substitution
```

The workflow is resumable. A rerun skips document instances already stored with their complete expected set of chunk
IDs, while a partially stored document is rebuilt. The completed portable output consists of both:

```text
src/output/
|-- chromadb/
|   |-- chroma.sqlite3
|   `-- <UUID-named index directory>/
|-- gemini_manifest.json
`-- result.json
```

Keep the entire `chromadb` directory together. Copying only `chroma.sqlite3` does not produce a complete Chroma
snapshot. `result.json` accumulates one appended node per build/preflight run describing what changed (documents
embedded, resumed, or skipped, and the resulting chunk count) — see [Automated builds on Cloud Run](#automated-builds-on-cloud-run).

## 4. Validate the result

```powershell
.\.venv\Scripts\python.exe validate_vector_store.py
```

The validator expects 15,684 chunks, 531 governed document IDs, 768-dimensional vectors, and the complete retrieval
metadata contract. It does not call Gemini.

To require byte-identical source inputs rather than the accepted UOB substitution, restore the historical
`UOB_EarningsTranscript_Q22026.pdf`, omit `--allow-known-uob-substitution` during the build, and run:

```powershell
.\.venv\Scripts\python.exe validate_vector_store.py --strict-source-match
```

## Known source substitution

The original `UOB_EarningsTranscript_Q22026.pdf` binary is unavailable. The supplied file currently contains the same
PDF as `UOB_2026_Q2.pdf`; both produce 14 chunks. With the explicit allowance above, those 14 chunks retain the
reference transcript IDs and governed metadata but are embedded from the available Q2 PDF content. The substitution is
recorded in `gemini_manifest.json`; the other 15,670 chunks use the intended source PDFs.

The three Standard Chartered BofA `(1)` reports were restored from the manifest archive and now match their reference
hashes.

## Reproducibility notes

- The logical collection, IDs, extracted texts, governed metadata, chunk counts, and embedding dimensions are
  reproduced. Chroma's internal UUIDs and database bytes are not expected to match an existing snapshot.
- Gemini is a managed service. Re-running the same model can produce small numerical differences if Google updates the
  hosted model implementation, even when the model name and inputs remain unchanged.
- Query-time retrieval must use `gemini-embedding-001`, 768 dimensions, and task type `RETRIEVAL_QUERY`.
- Building sends extracted third-party report text to the configured Vertex service. Confirm the client's data handling
  and document-licensing approvals before execution.

## Automated builds on Cloud Run

`run_pipeline.py` and `gcs_sync.py` (in this folder) wrap the three manual steps above into one Cloud Run Job
execution, using Cloud Storage as the persistent state since a Job's container is thrown away after each run.

Layout in the `EXPERT_HUB_GCS_BUCKET` bucket:

```text
gs://<EXPERT_HUB_GCS_BUCKET>/
|-- chromadb/            mirrors this folder's output/ directory (chromadb/ store, gemini_manifest.json, result.json)
`-- result/              mirrors the repo's result/ directory (embedded_documents.json bookkeeping)
```

`result/embedded_documents.json` (at the repo root, parallel to `src/`) records which document instances have already
been embedded, keyed by their source hash and chunk count. On the next run, any document that still matches its
recorded hash/chunk count is skipped entirely — it is not re-parsed or re-embedded. This is in addition to the
resumable upsert behavior already built into `build_vector_store.py`.

Each run appends a node to `output/result.json` describing what changed (documents embedded, resumed, or skipped, and
the resulting chunk count), so the change history survives across runs via the `chromadb/` GCS mirror.

### Pipeline steps (`run_pipeline.py`)

1. Download `gs://<bucket>/chromadb/` and `gs://<bucket>/result/` into the local `output/` and `result/` directories.
2. Run the credential-free preflight (equivalent to `--dry-run --allow-known-uob-substitution`).
3. Run the build (equivalent to `--allow-known-uob-substitution`), skipping documents already recorded in
   `result/embedded_documents.json`.
4. Run `validate_vector_store.py` and print the summary.
5. Upload `output/` and `result/` back to the bucket (this happens even if a step above fails, so partial progress is
   never lost).

Configuration is read entirely from the process environment — no `.env` file is used in the container:

| Env var | Required | Default | Purpose |
|---|---|---|---|
| `EXPERT_HUB_GCS_BUCKET` | yes | — | Bucket holding `chromadb/` and `result/` |
| `EXPERT_HUB_EMBEDDING_MODEL` | yes | — | Must be `gemini-embedding-001` |
| `VERTEX_API_KEY` | yes | — | Vertex express-mode API key (pass as a Cloud Run secret) |
| `GCS_CHROMADB_PREFIX` | no | `chromadb` | GCS prefix for the vector store output |
| `GCS_RESULT_PREFIX` | no | `result` | GCS prefix for the embedded-document bookkeeping |
| `ALLOW_KNOWN_UOB_SUBSTITUTION` | no | `true` | Set `false` to require byte-identical sources |
| `STRICT_SOURCE_MATCH` | no | `false` | Set `true` to fail validation if any substitution was used |

### Build and deploy

```bash
# From the repository root (Dockerfile expects both src/ and data/ in the build context)
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/REPO/expert-hub-vector-build

gcloud run jobs create expert-hub-vector-build \
  --image=REGION-docker.pkg.dev/PROJECT_ID/REPO/expert-hub-vector-build \
  --region=REGION \
  --set-env-vars=EXPERT_HUB_GCS_BUCKET=experthub-files,EXPERT_HUB_EMBEDDING_MODEL=gemini-embedding-001 \
  --set-secrets=VERTEX_API_KEY=VERTEX_API_KEY:latest \
  --max-retries=0 \
  --task-timeout=21600  # 6h headroom for embedding all 534 document instances; tune to your corpus/API throughput

# Trigger a run
gcloud run jobs execute expert-hub-vector-build --region=REGION
```

The Cloud Run Job's service account needs read/write access to the bucket (e.g. `roles/storage.objectAdmin` scoped to
`experthub-files`) and read access to the `VERTEX_API_KEY` secret (`roles/secretmanager.secretAccessor`).
