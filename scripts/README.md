# Rebuilding the Expert Hub Gemini vector store

This folder contains the standalone code and reference metadata needed to rebuild the Expert Hub Gemini Chroma store
from the PDFs in `../data`. It creates the governed collection used by the application; the generated vector store and
credentials are deliberately excluded from Git.

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
cd scripts
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

The command below embeds the full corpus and writes output under the ignored `scripts/output` directory:

```powershell
.\.venv\Scripts\python.exe build_vector_store.py `
  --env-file .env `
  --allow-known-uob-substitution
```

The workflow is resumable. A rerun skips document instances already stored with their complete expected set of chunk
IDs, while a partially stored document is rebuilt. The completed portable output consists of both:

```text
scripts/output/
|-- chroma_gemini_embedding/
|   |-- chroma.sqlite3
|   `-- <UUID-named index directory>/
`-- gemini_manifest.json
```

Keep the entire `chroma_gemini_embedding` directory together. Copying only `chroma.sqlite3` does not produce a complete
Chroma snapshot.

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
