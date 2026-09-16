"""Build the PULSE Chroma document index through the client BGE-M3 API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Sequence

import chromadb
import httpx
import numpy as np
import yaml
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from pypdf import PdfReader


COLLECTION_NAME = "pulse_documents_v2"
DISTANCE_METRIC = "cosine"
EMBEDDING_DIMENSION = 1024
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
CHUNK_CHARACTERS = 2_000
CHUNK_OVERLAP_CHARACTERS = 200
CHROMA_WRITE_BATCH_SIZE = 500
EMBEDDING_CHECKPOINT_SIZE = 32
MANIFEST_VERSION = 2
METADATA_SCHEMA_VERSION = 1
CHUNKER_VERSION = 1
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BuildConfig:
    documents_dir: Path
    metadata_dir: Path
    output_dir: Path
    endpoint: str
    api_key: str
    api_model: str
    timeout_seconds: float
    use_system_truststore: bool

    @property
    def chroma_dir(self) -> Path:
        return self.output_dir / "chroma"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"


@dataclass(frozen=True)
class PdfMetadata:
    source: str
    filename: str
    institution: str
    document_id: str
    title: str
    source_type: str
    document_type: str
    published_date: str
    reporting_start: str
    reporting_end: str
    reporting_period_type: str
    fiscal_year: int
    source_url: str
    checksum: str
    covered_bank_ids: tuple[str, ...]
    publisher: str
    geography: str
    source_bundle: str
    metadata_filename: str
    metadata_sha256: str


@dataclass
class PendingDocument:
    metadata: PdfMetadata
    page_count: int
    empty_pages: int
    failed_pages: int
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding process variables."""

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def parse_bool(value: str, *, name: str) -> bool:
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def resolve_config(args: argparse.Namespace, *, require_api: bool) -> BuildConfig:
    load_env_file(SCRIPT_DIR / ".env")
    endpoint = os.environ.get("UOB_EMBEDDING_ENDPOINT", "").strip()
    api_key = os.environ.get("UOB_EMBEDDING_API_KEY", "").strip()
    if require_api and (not endpoint or not api_key or api_key.startswith("replace-with-")):
        raise ValueError(
            "Set UOB_EMBEDDING_ENDPOINT and UOB_EMBEDDING_API_KEY in .env"
        )
    timeout = float(os.environ.get("UOB_EMBEDDING_TIMEOUT_SECONDS", "60"))
    if timeout <= 0:
        raise ValueError("UOB_EMBEDDING_TIMEOUT_SECONDS must be positive")
    return BuildConfig(
        documents_dir=args.documents.resolve(),
        metadata_dir=args.metadata.resolve(),
        output_dir=args.output.resolve(),
        endpoint=endpoint,
        api_key=api_key,
        api_model=os.environ.get(
            "UOB_EMBEDDING_MODEL", "/data/genai/models/bge-m3"
        ).strip(),
        timeout_seconds=timeout,
        use_system_truststore=parse_bool(
            os.environ.get("UOB_EMBEDDING_USE_SYSTEM_TRUSTSTORE", "true"),
            name="UOB_EMBEDDING_USE_SYSTEM_TRUSTSTORE",
        ),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_text(record: dict[str, Any], key: str, metadata_path: Path) -> str:
    value = record.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{metadata_path.name}: missing required field {key!r}")
    return str(value).strip()


def parse_metadata(metadata_path: Path) -> PdfMetadata:
    raw_bytes = metadata_path.read_bytes()
    try:
        record = yaml.safe_load(raw_bytes) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot read {metadata_path.name}: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{metadata_path.name}: YAML root must be a mapping")

    reporting = record.get("reporting_period")
    if not isinstance(reporting, dict):
        raise ValueError(f"{metadata_path.name}: reporting_period must be a mapping")
    fiscal_year = reporting.get("fiscal_year")
    if isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int):
        raise ValueError(
            f"{metadata_path.name}: reporting_period.fiscal_year must be an integer"
        )
    institution = required_text(record, "bank_id", metadata_path).lower()
    source = required_text(record, "source_file", metadata_path).replace("\\", "/")
    checksum = required_text(record, "checksum", metadata_path).lower()
    if not SHA256.fullmatch(checksum):
        raise ValueError(f"{metadata_path.name}: checksum must be a SHA-256 value")
    covered = record.get("covered_bank_ids") or [institution]
    if not isinstance(covered, list) or not all(
        isinstance(value, str) and value.strip() for value in covered
    ):
        raise ValueError(
            f"{metadata_path.name}: covered_bank_ids must be a list of bank IDs"
        )

    return PdfMetadata(
        source=source,
        filename=Path(source).name,
        institution=institution,
        document_id=required_text(record, "document_id", metadata_path),
        title=required_text(record, "title", metadata_path),
        source_type=required_text(record, "source_type", metadata_path),
        document_type=required_text(record, "document_type", metadata_path),
        published_date=required_text(record, "publication_date", metadata_path),
        reporting_start=required_text(reporting, "start", metadata_path),
        reporting_end=required_text(reporting, "end", metadata_path),
        reporting_period_type=required_text(reporting, "period_type", metadata_path),
        fiscal_year=fiscal_year,
        source_url=str(record.get("source_url") or "").strip(),
        checksum=checksum,
        covered_bank_ids=tuple(value.strip().lower() for value in covered),
        publisher=str(record.get("publisher") or "").strip(),
        geography=str(record.get("geography") or "").strip(),
        source_bundle=str(record.get("source_bundle") or "").strip(),
        metadata_filename=metadata_path.name,
        metadata_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def load_metadata_catalog(
    metadata_dir: Path, pdfs: list[Path]
) -> tuple[dict[Path, PdfMetadata], int]:
    """Match every uniquely named PDF to reviewed YAML and verify its checksum."""

    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")
    metadata_paths = sorted([*metadata_dir.rglob("*.yaml"), *metadata_dir.rglob("*.yml")])
    by_filename: dict[str, list[Path]] = {}
    for path in metadata_paths:
        try:
            record = yaml.safe_load(path.read_bytes()) or {}
            source_file = str(record.get("source_file") or "").strip()
        except (OSError, yaml.YAMLError):
            continue
        if source_file:
            by_filename.setdefault(Path(source_file).name.casefold(), []).append(path)
    duplicates = {name for name, paths in by_filename.items() if len(paths) != 1}
    if duplicates:
        raise ValueError(f"duplicate YAML basenames: {', '.join(sorted(duplicates)[:5])}")

    pdf_by_filename: dict[str, Path] = {}
    for path in pdfs:
        filename = path.name.casefold()
        if filename in pdf_by_filename:
            raise ValueError(f"ambiguous PDF filename: {path.name}")
        pdf_by_filename[filename] = path
    missing = sorted(set(pdf_by_filename) - set(by_filename))
    if missing:
        raise ValueError(
            f"{len(missing)} PDFs lack reviewed YAML: {', '.join(missing[:5])}"
        )

    catalog: dict[Path, PdfMetadata] = {}
    document_ids: set[str] = set()
    governed_sources: set[str] = set()
    for filename, pdf_path in pdf_by_filename.items():
        metadata = parse_metadata(by_filename[filename][0])
        if metadata.filename.casefold() != pdf_path.name.casefold():
            raise ValueError(f"{metadata.metadata_filename}: source_file filename mismatch")
        if metadata.document_id in document_ids:
            raise ValueError(f"duplicate document_id: {metadata.document_id}")
        if metadata.source.casefold() in governed_sources:
            raise ValueError(f"duplicate source_file: {metadata.source}")
        if file_sha256(pdf_path) != metadata.checksum:
            raise ValueError(f"{metadata.metadata_filename}: PDF checksum mismatch")
        document_ids.add(metadata.document_id)
        governed_sources.add(metadata.source.casefold())
        catalog[pdf_path.resolve()] = metadata
    return catalog, len(metadata_paths) - len(catalog)


def normalize_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def chunk_text(text: str) -> list[str]:
    """Apply the current PULSE deterministic 2,000/200 character chunking."""

    text = normalize_page_text(text)
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARACTERS, len(text))
        if end < len(text):
            floor = start + CHUNK_CHARACTERS // 2
            boundary = max(
                text.rfind("\n\n", floor, end),
                text.rfind(". ", floor, end),
                text.rfind(" ", floor, end),
            )
            if boundary > start:
                end = boundary + (1 if text[boundary] == "." else 0)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(0, end - CHUNK_OVERLAP_CHARACTERS)
        whitespace = text.find(" ", next_start, end)
        start = whitespace + 1 if whitespace >= 0 else next_start
        if start >= end:
            start = end
    return chunks


def chunk_id(source: str, checksum: str, page: int, position: int) -> str:
    value = f"{source}|{checksum}|{page}|{position}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def extract_pdf(path: Path, metadata: PdfMetadata) -> PendingDocument:
    """Extract page text and produce PULSE-compatible chunks and metadata."""

    reader = PdfReader(str(path))
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    empty_pages = failed_pages = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_chunks = chunk_text(page.extract_text() or "")
        except Exception as exc:
            failed_pages += 1
            print(f"WARNING: skipped page {page_number} in {metadata.source}: {exc}")
            continue
        if not page_chunks:
            empty_pages += 1
            continue
        for position, body in enumerate(page_chunks):
            header = (
                f"Title: {metadata.title}\nInstitution: {metadata.institution}\n"
                f"Document type: {metadata.document_type}\n"
                f"Published date: {metadata.published_date or 'unavailable'}\n"
                f"Source: {metadata.filename}\nPage: {page_number}\n\n"
            )
            ids.append(chunk_id(metadata.source, metadata.checksum, page_number, position))
            documents.append(header + body)
            metadatas.append(
                {
                    "source": metadata.source,
                    "filename": metadata.filename,
                    "institution": metadata.institution,
                    "document_id": metadata.document_id,
                    "title": metadata.title,
                    "source_type": metadata.source_type,
                    "document_type": metadata.document_type,
                    "published_date": metadata.published_date,
                    "reporting_start": metadata.reporting_start,
                    "reporting_end": metadata.reporting_end,
                    "reporting_period_type": metadata.reporting_period_type,
                    "fiscal_year": metadata.fiscal_year,
                    "source_url": metadata.source_url,
                    "covered_bank_ids": ",".join(metadata.covered_bank_ids),
                    "publisher": metadata.publisher,
                    "geography": metadata.geography,
                    "source_bundle": metadata.source_bundle,
                    "page": page_number,
                    "page_start": page_number,
                    "page_end": page_number,
                    "chunk": position,
                    "file_sha256": metadata.checksum,
                    "metadata_file": metadata.metadata_filename,
                    "metadata_sha256": metadata.metadata_sha256,
                }
            )
    if not documents:
        raise ValueError("no extractable text found; OCR may be required")
    return PendingDocument(
        metadata=metadata,
        page_count=len(reader.pages),
        empty_pages=empty_pages,
        failed_pages=failed_pages,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )


class UobEmbeddingClient:
    """Bearer-authenticated adapter for the client OpenAI-compatible endpoint."""

    def __init__(self, config: BuildConfig) -> None:
        if config.use_system_truststore:
            import truststore

            truststore.inject_into_ssl()
        self._model = config.api_model
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout_seconds,
            trust_env=True,
        )
        self._endpoint = config.endpoint

    def close(self) -> None:
        self._client.close()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Call once per passage, preserving the reviewed client API contract."""

        vectors = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("embedding inputs must be non-empty text")
            try:
                response = self._client.post(
                    self._endpoint,
                    json={
                        "input": text,
                        "model": self._model,
                        "encoding_format": "float",
                        "input_type": "passage",
                    },
                )
            except httpx.RequestError as exc:
                raise RuntimeError("The UOB embedding API could not be reached") from exc
            if not response.is_success:
                raise RuntimeError(
                    f"The UOB embedding API returned HTTP {response.status_code}"
                )
            vectors.append(self._validated_vector(response))
        return vectors

    @staticmethod
    def _validated_vector(response: httpx.Response) -> list[float]:
        try:
            data = response.json()["data"]
            if not isinstance(data, list) or len(data) != 1:
                raise ValueError
            item = data[0]
            if not isinstance(item, dict) or item.get("index") != 0:
                raise ValueError
            if item.get("object") != "embedding":
                raise ValueError
            raw_vector = item["embedding"]
            if not isinstance(raw_vector, list) or any(
                isinstance(value, bool) for value in raw_vector
            ):
                raise ValueError
            vector = [float(value) for value in raw_vector]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("The UOB embedding API returned an invalid response") from exc
        if len(vector) != EMBEDDING_DIMENSION or any(
            not math.isfinite(value) for value in vector
        ):
            raise RuntimeError(
                f"The UOB embedding API must return {EMBEDDING_DIMENSION} finite values"
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise RuntimeError("The UOB embedding API returned a zero vector")
        return [value / norm for value in vector]


def collection_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_chars": CHUNK_CHARACTERS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARACTERS,
    }


def open_collection(config: BuildConfig, *, rebuild: bool):
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(config.chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            pass
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        collection = client.create_collection(
            COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": DISTANCE_METRIC,
                    "ef_construction": 200,
                    "ef_search": 200,
                    "max_neighbors": 32,
                }
            },
            metadata=collection_metadata(),
            embedding_function=None,
        )
    actual_metadata = collection.metadata or {}
    mismatches = [
        f"{key}={actual_metadata.get(key)!r}, expected {value!r}"
        for key, value in collection_metadata().items()
        if actual_metadata.get(key) != value
    ]
    actual_metric = (collection.configuration_json.get("hnsw") or {}).get("space")
    if actual_metric != DISTANCE_METRIC:
        mismatches.append(f"hnsw.space={actual_metric!r}, expected 'cosine'")
    if mismatches:
        client.close()
        raise RuntimeError("Incompatible Chroma collection: " + "; ".join(mismatches))
    return client, collection


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (2**attempt))


def manifest_signature(config: BuildConfig) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "parser_version": package_version("pypdf"),
        "chunker_version": CHUNKER_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "embedding_provider": "uob_embedding_api",
        "embedding_api_model": config.api_model,
        "normalized_embeddings": True,
        "chunk_chars": CHUNK_CHARACTERS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARACTERS,
    }


def load_manifest(config: BuildConfig, *, rebuild: bool, collection_count: int) -> dict[str, Any]:
    path = config.output_dir / "manifest.json"
    if rebuild:
        return {**manifest_signature(config), "documents": {}}
    if not path.is_file():
        if collection_count:
            raise RuntimeError("Existing collection has no manifest; rerun with --rebuild")
        return {**manifest_signature(config), "documents": {}}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Existing manifest is unreadable; rerun with --rebuild") from exc
    expected = manifest_signature(config)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Existing manifest is incompatible; rerun with --rebuild")
    if not isinstance(manifest.get("documents"), dict):
        raise RuntimeError("Existing manifest has no document map; rerun with --rebuild")
    return manifest


def checkpoint_signature(metadata: PdfMetadata, config: BuildConfig) -> str:
    values = {
        "checksum": metadata.checksum,
        "metadata_sha256": metadata.metadata_sha256,
        "chunker_version": CHUNKER_VERSION,
        "chunk_chars": CHUNK_CHARACTERS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARACTERS,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "embedding_api_model": config.api_model,
        "input_type": "passage",
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def embed_with_checkpoints(
    document: PendingDocument, config: BuildConfig, embedder: UobEmbeddingClient
) -> list[list[float]]:
    signature = checkpoint_signature(document.metadata, config)
    checkpoint_dir = config.checkpoint_dir / document.metadata.checksum[:20] / signature[:12]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    embeddings: list[list[float]] = []
    for start in range(0, len(document.documents), EMBEDDING_CHECKPOINT_SIZE):
        selected = document.documents[start : start + EMBEDDING_CHECKPOINT_SIZE]
        path = checkpoint_dir / f"{start:06d}.npy"
        if path.is_file():
            array = np.load(path, allow_pickle=False)
            norms = np.linalg.norm(array, axis=1) if array.ndim == 2 else np.array([])
            if (
                array.shape != (len(selected), EMBEDDING_DIMENSION)
                or not np.isfinite(array).all()
                or not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5)
            ):
                raise RuntimeError(f"Invalid embedding checkpoint: {path}")
            batch = array.astype(float).tolist()
        else:
            batch = embedder.embed(selected)
            array = np.asarray(batch, dtype=np.float32)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            temporary.replace(path)
        embeddings.extend(batch)
    return embeddings


def upsert_document(collection, document: PendingDocument, embeddings: list[list[float]]) -> None:
    existing = collection.get(
        where={"source": document.metadata.source}, include=[]
    ).get("ids", [])
    for start in range(0, len(document.ids), CHROMA_WRITE_BATCH_SIZE):
        end = start + CHROMA_WRITE_BATCH_SIZE
        collection.upsert(
            ids=document.ids[start:end],
            documents=document.documents[start:end],
            metadatas=document.metadatas[start:end],
            embeddings=embeddings[start:end],
        )
    obsolete = list(set(existing) - set(document.ids))
    for start in range(0, len(obsolete), CHROMA_WRITE_BATCH_SIZE):
        collection.delete(ids=obsolete[start : start + CHROMA_WRITE_BATCH_SIZE])


def ingest_documents(
    config: BuildConfig,
    catalog: dict[Path, PdfMetadata],
    collection,
    manifest: dict[str, Any],
    *,
    limit: int | None,
) -> dict[str, Any]:
    """Incrementally extract, embed and persist the reviewed document corpus."""

    records: dict[str, Any] = manifest["documents"]
    selected = list(catalog.items())
    if limit is not None:
        selected = selected[: max(limit, 0)]
    indexed = skipped = failed = chunks = empty_pages = failed_pages = 0
    failures: list[dict[str, str]] = []
    current_sources: set[str] = set()
    embedder = UobEmbeddingClient(config)
    try:
        for number, (path, metadata) in enumerate(selected, start=1):
            current_sources.add(metadata.source)
            previous = records.get(metadata.source)
            if (
                previous
                and previous.get("sha256") == metadata.checksum
                and previous.get("metadata_sha256") == metadata.metadata_sha256
                and collection.get(where={"source": metadata.source}, limit=1).get("ids")
            ):
                skipped += 1
                print(f"[{number}/{len(selected)}] skip {metadata.source}", flush=True)
                continue
            try:
                document = extract_pdf(path, metadata)
                print(
                    f"[{number}/{len(selected)}] embed {metadata.source} "
                    f"({len(document.documents)} chunks)",
                    flush=True,
                )
                embeddings = embed_with_checkpoints(document, config, embedder)
                upsert_document(collection, document, embeddings)
                records[metadata.source] = {
                    "sha256": metadata.checksum,
                    "metadata_sha256": metadata.metadata_sha256,
                    "pages": document.page_count,
                    "empty_pages": document.empty_pages,
                    "failed_pages": document.failed_pages,
                    "chunks": len(document.ids),
                }
                manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                write_json_atomic(config.output_dir / "manifest.json", manifest)
                indexed += 1
                chunks += len(document.ids)
                empty_pages += document.empty_pages
                failed_pages += document.failed_pages
            except Exception as exc:
                failed += 1
                failures.append({"source": metadata.source, "error": str(exc)})
                print(f"[{number}/{len(selected)}] FAILED {metadata.source}: {exc}", flush=True)
    finally:
        embedder.close()

    if limit is None:
        stale = set(records) - current_sources
        for source in stale:
            collection.delete(where={"source": source})
            del records[source]
        if stale:
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(config.output_dir / "manifest.json", manifest)
    return {
        "pdfs_selected": len(selected),
        "documents_indexed": indexed,
        "documents_skipped": skipped,
        "documents_failed": failed,
        "chunks_written": chunks,
        "empty_pages": empty_pages,
        "failed_pages": failed_pages,
        "failures": failures,
    }


def run_vector_store_build(
    config: BuildConfig, *, rebuild: bool, limit: int | None, validate_only: bool
) -> dict[str, Any]:
    """
    Vector-store build:
    1. Discover PDFs and validate all reviewed metadata and checksums.
    2. Optionally stop after input validation.
    3. Open the PULSE-compatible Chroma collection.
    4. Incrementally extract, embed, checkpoint and persist documents.
    5. Reopen the collection and record the final persisted count.
    """

    if not config.documents_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {config.documents_dir}")
    pdfs = sorted(config.documents_dir.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found below {config.documents_dir}")
    catalog, orphan_metadata = load_metadata_catalog(config.metadata_dir, pdfs)
    if validate_only:
        return {
            "pdfs_discovered": len(pdfs),
            "metadata_matched": len(catalog),
            "orphan_metadata_ignored": orphan_metadata,
            "unique_document_ids": len({item.document_id for item in catalog.values()}),
        }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    client, collection = open_collection(config, rebuild=rebuild)
    try:
        manifest = load_manifest(
            config, rebuild=rebuild, collection_count=collection.count()
        )
        summary = ingest_documents(
            config, catalog, collection, manifest, limit=limit
        )
        summary.update(
            {
                "pdfs_discovered": len(pdfs),
                "metadata_matched": len(catalog),
                "orphan_metadata_ignored": orphan_metadata,
                "collection_chunks": collection.count(),
            }
        )
    finally:
        client.close()
    reopened_client, reopened = open_collection(config, rebuild=False)
    try:
        summary["persistence_reopen_count"] = reopened.count()
    finally:
        reopened_client.close()
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(config.output_dir / "last_build.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=SCRIPT_DIR / "metadata")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "output" / "rag_index")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-inputs-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = resolve_config(args, require_api=not args.validate_inputs_only)
        summary = run_vector_store_build(
            config,
            rebuild=args.rebuild,
            limit=args.limit,
            validate_only=args.validate_inputs_only,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Vector-store build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 2 if summary.get("documents_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
