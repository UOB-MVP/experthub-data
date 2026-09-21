"""Rebuild the Expert Hub Gemini Chroma vector store from the bundled PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import BinaryIO


EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
COLLECTION_NAME = "peer_benchmarking_documents_v1"
DISTANCE_METRIC = "cosine"
CHROMADB_VERSION = "1.5.9"
MAX_CHUNK_CHARACTERS = 6000
CHUNK_OVERLAP_CHARACTERS = 400
MIN_CHUNK_CHARACTERS = 80
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_MAX_RETRIES = 6

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent.parent / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
SOURCE_MANIFEST_PATH = SCRIPT_DIR / "source_manifest.json"
BUILD_CATALOG_PATH = SCRIPT_DIR / "build_catalog.json"

KNOWN_SOURCE_SUBSTITUTIONS = {
    "reports_data/01 Company disclosures and comms/Earnings call transcripts/UOB/"
    "UOB_EarningsTranscript_Q22026.pdf": {
        "expected_hash": "99f6991a1c76589d8858b97387a6196c",
        "actual_hash": "d162746346ac1dea63ac25d3c898c49f",
        "reason": "Original transcript unavailable; UOB_2026_Q2.pdf is supplied under this filename.",
    }
}


def open_pdf_binary(path: Path) -> BinaryIO:
    """Open a PDF while supporting repository paths longer than 260 characters on Windows."""

    resolved = str(path.resolve())
    platform_path = f"\\\\?\\{resolved}" if os.name == "nt" else resolved
    return open(platform_path, "rb")  # noqa: PTH123 - long-path prefix requires the built-in API


def truncated_sha256(path: Path) -> str:
    """Return the 32-character SHA-256 prefix used by the historical manifest."""

    digest = hashlib.sha256()
    with open_pdf_binary(path) as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:32]


def split_long_page(text: str) -> list[str]:
    """Split one extracted page using the historical paragraph and overlap policy."""

    if len(text) <= MAX_CHUNK_CHARACTERS:
        return [text]
    parts: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        if len(current) + len(paragraph) + 2 > MAX_CHUNK_CHARACTERS and current:
            parts.append(current)
            current = current[-CHUNK_OVERLAP_CHARACTERS:] + "\n\n" + paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current.strip():
        parts.append(current)

    chunks: list[str] = []
    for part in parts:
        while len(part) > MAX_CHUNK_CHARACTERS:
            chunks.append(part[:MAX_CHUNK_CHARACTERS])
            part = part[MAX_CHUNK_CHARACTERS - CHUNK_OVERLAP_CHARACTERS :]
        if part.strip():
            chunks.append(part)
    return chunks


def extract_pdf_chunks(path: Path) -> list[dict[str, object]]:
    """Extract page-aware chunks with the parser and policy used by the reference store."""

    from pypdf import PdfReader

    chunks: list[dict[str, object]] = []
    with open_pdf_binary(path) as source:
        reader = PdfReader(source)
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                continue
            if len(text) < MIN_CHUNK_CHARACTERS:
                continue
            for part_number, part in enumerate(split_long_page(text)):
                chunks.append({"text": part, "page": page_number, "part": part_number})
    return chunks


def source_path(data_dir: Path, manifest_path: str) -> Path:
    """Map a portable reports_data manifest path to this repository's data directory."""

    parts = PurePosixPath(manifest_path).parts
    if not parts or parts[0] != "reports_data" or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"invalid source path in manifest: {manifest_path}")
    candidate = (data_dir / Path(*parts[1:])).resolve()
    data_root = data_dir.resolve()
    if data_root != candidate and data_root not in candidate.parents:
        raise ValueError(f"source path escapes the data directory: {manifest_path}")
    return candidate


def validate_source_corpus(
    data_dir: Path,
    source_manifest: dict[str, object],
    *,
    allow_known_substitution: bool,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Validate every expected path and hash, allowing only the documented UOB exception."""

    files = source_manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source_manifest.json has no files")

    actual_hashes: dict[str, str] = {}
    substitutions: list[dict[str, str]] = []
    failures: list[str] = []
    for relative_path, details in files.items():
        path = source_path(data_dir, relative_path)
        try:
            actual_hash = truncated_sha256(path)
        except FileNotFoundError:
            failures.append(f"missing: {relative_path}")
            continue
        expected_hash = str(details["hash"])
        actual_hashes[relative_path] = actual_hash
        if actual_hash == expected_hash:
            continue
        exception = KNOWN_SOURCE_SUBSTITUTIONS.get(relative_path)
        allowed = (
            allow_known_substitution
            and exception is not None
            and exception["expected_hash"] == expected_hash
            and exception["actual_hash"] == actual_hash
        )
        if allowed:
            substitutions.append({"path": relative_path, **exception})
        else:
            failures.append(f"hash mismatch: {relative_path} (expected {expected_hash}, found {actual_hash})")
    if failures:
        raise RuntimeError("source validation failed:\n- " + "\n- ".join(failures))
    return actual_hashes, substitutions


def validate_build_catalog(
    catalog: dict[str, object], source_manifest: dict[str, object], actual_hashes: dict[str, str]
) -> list[dict[str, object]]:
    """Check that every build record is backed by a validated source and manifest entry."""

    documents = catalog.get("documents")
    manifest_files = source_manifest["files"]
    if not isinstance(documents, list) or not documents:
        raise ValueError("build_catalog.json has no documents")
    for document in documents:
        relative_path = str(document["source_path"])
        if relative_path not in manifest_files or relative_path not in actual_hashes:
            raise ValueError(f"catalog source is unavailable: {relative_path}")
        manifest_hash = str(manifest_files[relative_path]["hash"])
        if str(document["expected_hash"]) != manifest_hash:
            raise ValueError(f"catalog and source manifest disagree for {relative_path}")
        if int(document["chunk_count"]) < 1:
            raise ValueError(f"catalog contains an empty document: {relative_path}")
    return documents


def validate_runtime() -> None:
    """Refuse to build with a Chroma storage version different from the reference runtime."""

    installed = version("chromadb")
    if installed != CHROMADB_VERSION:
        raise RuntimeError(f"ChromaDB {CHROMADB_VERSION} is required; installed version is {installed}")


def load_vertex_client(env_file: Path):
    """Load credentials and create the Vertex express-mode client used for the reference build."""

    import truststore
    from dotenv import load_dotenv
    from google import genai

    truststore.inject_into_ssl()
    load_dotenv(env_file, override=False)
    configured_model = os.environ.get("EXPERT_HUB_EMBEDDING_MODEL", "").strip()
    if configured_model != EMBEDDING_MODEL:
        raise RuntimeError(
            f"EXPERT_HUB_EMBEDDING_MODEL must be {EMBEDDING_MODEL!r}; configured value is {configured_model!r}"
        )
    api_key = os.environ.get("VERTEX_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("VERTEX_API_KEY is required in the selected environment file")
    return genai.Client(vertexai=True, api_key=api_key)


def embed_texts(client, texts: list[str]) -> list[list[float]]:
    """Embed one batch with the retry and exponential-backoff policy of the reference build."""

    from google.genai import types

    delay_seconds = 2.0
    for attempt in range(EMBEDDING_MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIMENSIONS,
                    task_type=DOCUMENT_TASK_TYPE,
                ),
            )
            vectors = [list(embedding.values or []) for embedding in response.embeddings or []]
            if len(vectors) != len(texts) or any(len(vector) != EMBEDDING_DIMENSIONS for vector in vectors):
                raise RuntimeError("Gemini returned an incomplete or incorrectly sized embedding batch")
            return vectors
        except Exception as error:
            message = str(error)
            retryable = any(
                marker in message
                for marker in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE")
            )
            if not retryable or attempt == EMBEDDING_MAX_RETRIES - 1:
                raise
            print(f"    retry {attempt + 1} in {delay_seconds:.0f}s ({message[:100]})", flush=True)
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 60)
    raise RuntimeError("embedding retry loop ended unexpectedly")


def open_collection(output_dir: Path):
    """Create or resume the one expected cosine collection without replacing existing work."""

    import chromadb

    store_dir = output_dir / "chroma_gemini_embedding"
    store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store_dir))
    names = {item.name for item in client.list_collections()}
    unexpected = names - {COLLECTION_NAME}
    if unexpected:
        raise RuntimeError(f"output contains unexpected Chroma collections: {sorted(unexpected)}")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": DISTANCE_METRIC}},
        embedding_function=None,
    )
    return client, collection


def chunk_payload(
    document: dict[str, object], chunks: list[dict[str, object]]
) -> tuple[list[str], list[str], list[dict]]:
    """Apply the reference IDs and governed metadata to one parsed document's chunks."""

    metadata_template = dict(document["metadata"])
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for chunk_index, chunk in enumerate(chunks):
        page = int(chunk["page"])
        part = int(chunk["part"])
        ids.append(f"{document['id_prefix']}:{page}:{part}")
        texts.append(str(chunk["text"]))
        metadatas.append(
            {
                **metadata_template,
                "chunk_index": chunk_index,
                "page": page,
                "part": part,
                "page_start": page,
                "page_end": page,
            }
        )
    return ids, texts, metadatas


def ingest_document(collection, vertex_client, document: dict[str, object], chunks: list[dict[str, object]]) -> str:
    """Resume, replace a partial document, or embed and persist one complete document."""

    ids, texts, metadatas = chunk_payload(document, chunks)
    existing_ids = list(collection.get(ids=ids, include=[])["ids"])
    if len(existing_ids) == len(ids):
        return "already_complete"
    if existing_ids:
        collection.delete(ids=existing_ids)
    try:
        for start in range(0, len(ids), EMBEDDING_BATCH_SIZE):
            end = start + EMBEDDING_BATCH_SIZE
            vectors = embed_texts(vertex_client, texts[start:end])
            collection.upsert(
                ids=ids[start:end],
                embeddings=vectors,
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
    except Exception:
        collection.delete(ids=ids)
        raise
    return "embedded"


def write_output_manifest(
    output_dir: Path,
    catalog: dict[str, object],
    substitutions: list[dict[str, str]],
    embedded_documents: int,
    resumed_documents: int,
) -> None:
    """Write the portable logical-store manifest after a validated build."""

    expected = dict(catalog["expected"])
    manifest = {
        "provider": "gemini",
        "api_backend": "vertex",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "document_task_type": DOCUMENT_TASK_TYPE,
        "query_task_type": QUERY_TASK_TYPE,
        "collection": COLLECTION_NAME,
        "distance_metric": DISTANCE_METRIC,
        "embedding_count": expected["chunks"],
        "document_count": expected["document_ids"],
        "document_instance_count": expected["document_instances"],
        "source_path_count": expected["source_paths"],
        "embedded_document_instances": embedded_documents,
        "resumed_document_instances": resumed_documents,
        "source_substitutions": substitutions,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "gemini_manifest.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def run_build(
    data_dir: Path,
    output_dir: Path,
    env_file: Path | None,
    *,
    dry_run: bool,
    allow_known_substitution: bool,
) -> None:
    """
    Full rebuild workflow:
    1. Validate every source PDF against the historical manifest.
    2. Load the effective document-instance and governed-metadata catalog.
    3. Reproduce page-local chunks and validate their expected counts.
    4. Embed with Gemini and upsert the final governed Chroma collection.
    5. Validate final counts and write a portable build manifest.
    """

    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    catalog = json.loads(BUILD_CATALOG_PATH.read_text(encoding="utf-8"))
    actual_hashes, substitutions = validate_source_corpus(
        data_dir, source_manifest, allow_known_substitution=allow_known_substitution
    )
    documents = validate_build_catalog(catalog, source_manifest, actual_hashes)
    if dry_run:
        total_chunks = 0
        for index, document in enumerate(documents, start=1):
            chunks = extract_pdf_chunks(source_path(data_dir, str(document["source_path"])))
            expected = int(document["chunk_count"])
            if len(chunks) != expected:
                raise RuntimeError(
                    f"chunk count mismatch for {document['source_path']}: expected {expected}, found {len(chunks)}"
                )
            total_chunks += len(chunks)
            if index % 25 == 0 or index == len(documents):
                print(f"preflight {index}/{len(documents)} documents", flush=True)
        print(
            f"preflight complete: {len(source_manifest['files'])} source paths, {len(documents)} document "
            f"instances, {total_chunks} chunks, {len(substitutions)} documented substitution(s)"
        )
        return

    if env_file is None:
        raise ValueError("--env-file is required unless --dry-run is used")
    validate_runtime()
    _, collection = open_collection(output_dir)
    vertex_client = load_vertex_client(env_file)
    embedded_documents = 0
    resumed_documents = 0
    try:
        for index, document in enumerate(documents, start=1):
            chunks = extract_pdf_chunks(source_path(data_dir, str(document["source_path"])))
            expected = int(document["chunk_count"])
            if len(chunks) != expected:
                raise RuntimeError(
                    f"chunk count mismatch for {document['source_path']}: expected {expected}, found {len(chunks)}"
                )
            status = ingest_document(collection, vertex_client, document, chunks)
            embedded_documents += status == "embedded"
            resumed_documents += status == "already_complete"
            print(f"[{index}/{len(documents)}] {status:16s} {document['source_path']}", flush=True)
    finally:
        close = getattr(vertex_client, "close", None)
        if callable(close):
            close()

    expected_chunks = int(catalog["expected"]["chunks"])
    if collection.count() != expected_chunks:
        raise RuntimeError(f"final collection has {collection.count()} chunks; expected {expected_chunks}")
    write_output_manifest(output_dir, catalog, substitutions, embedded_documents, resumed_documents)
    print(f"build complete: {collection.count()} chunks in {output_dir / 'chroma_gemini_embedding'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-known-uob-substitution",
        action="store_true",
        help="accept the documented UOB Q2 transcript substitution and record it in the output manifest",
    )
    args = parser.parse_args()
    run_build(
        args.data_dir,
        args.output_dir,
        args.env_file,
        dry_run=args.dry_run,
        allow_known_substitution=args.allow_known_uob_substitution,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
