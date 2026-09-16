# UOB BGE-M3 vector-store builder

This standalone utility builds the document index consumed by the PULSE data
agent. It uses the client-hosted BGE-M3 embeddings API; it does not download or
run the BGE-M3 model locally.

The bundle includes the 1,460 reviewed YAML metadata records. The corresponding
PDFs are distributed separately as `documents.zip`.

This package was aligned to `ss_main_rag_work` commit
`97c23042a67e308def785bc9595ce6446623065b`. The client request shape comes from
`pulse-ai` commit `dc42e953656325b88703a5ec3d32226a82dd2e85`.

## Output contract

- ChromaDB 1.5.9 persistent store under `<output>/chroma`
- Collection `pulse_documents_v2`
- BGE-M3 passage embeddings, 1,024 dimensions and L2-normalised
- Cosine HNSW index
- 2,000-character chunks with 200-character overlap
- PULSE-compatible chunk text, IDs and metadata
- Incremental manifest, API checkpoints and build summary

## 1. Prepare Python

Python 3.13 is recommended.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

## 2. Configure the client embeddings API

Copy `.env.example` to `.env`, then replace only the API key and any endpoint or
model value supplied by the client platform team.

```powershell
Copy-Item .env.example .env
```

The request made for every passage is:

```json
{
  "input": "<document passage>",
  "model": "/data/genai/models/bge-m3",
  "encoding_format": "float",
  "input_type": "passage"
}
```

Authentication is `Authorization: Bearer <UOB_EMBEDDING_API_KEY>`. Standard
`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` and `SSL_CERT_FILE` environment
variables are honoured. The configured system trust store is used by default.

## 3. Extract the separately supplied documents

Extract `documents.zip` into `input`. The expected result is:

```text
input/
  documents/
    cimb/
    dbs/
    hsbc/
    maybank/
    ocbc/
    standard_chartered/
    uob/
```

## 4. Validate inputs without calling the API

```powershell
python .\build_vector_store.py `
  --documents .\input\documents `
  --validate-inputs-only
```

Success reports 1,460 matched PDFs, 1,460 unique document IDs and zero orphan
metadata files. Validation also verifies every PDF SHA-256 against its reviewed
YAML record.

## 5. Build the vector store

For a new output directory:

```powershell
python .\build_vector_store.py `
  --documents .\input\documents `
  --output .\output\rag_index
```

The embeddings API is called once per passage, matching the client API
contract. Vectors are checkpointed every 32 passages. Re-running the command
reuses completed checkpoints and skips documents already represented by the
same PDF and metadata hashes.

Use `--rebuild` only when intentionally replacing an existing collection:

```powershell
python .\build_vector_store.py `
  --documents .\input\documents `
  --output .\output\rag_index `
  --rebuild
```

Do not run two builders against the same output directory simultaneously. Stop
any application that has the same Chroma directory open before rebuilding it.

## Produced files

```text
output/rag_index/
  chroma/
    chroma.sqlite3
    <HNSW segment files>
  checkpoints/
    <restartable API vector batches>
  manifest.json
  last_build.json
```

Copy the complete `rag_index` directory into the application's configured
`RAG_INDEX_DIR`. Keep the PDFs available to the application for document
citations and configure, for example:

```dotenv
RAG_ENABLED=true
RAG_DOCUMENTS_DIR=C:\path\to\input\documents
RAG_INDEX_DIR=C:\path\to\output\rag_index
```

`RAG_CHROMA_DIR` does not need to be set; the application defaults it to the
`chroma` child of `RAG_INDEX_DIR`.

PDF pages with no extractable text are recorded as empty. Image-only PDFs must
be OCR-processed before this builder is run.
