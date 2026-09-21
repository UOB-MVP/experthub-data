"""Validate a rebuilt Expert Hub Gemini Chroma store without calling Gemini."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path


CHROMADB_VERSION = "1.5.9"
COLLECTION_NAME = "peer_benchmarking_documents_v1"
EMBEDDING_DIMENSIONS = 768
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
BUILD_CATALOG_PATH = SCRIPT_DIR / "build_catalog.json"
REQUIRED_METADATA = {
    "document_id",
    "bank_id",
    "covered_bank_ids",
    "title",
    "source_type",
    "document_type",
    "publication_date",
    "period_start",
    "period_end",
    "period_type",
    "fiscal_year",
    "source_file",
    "source_url",
    "checksum",
    "chunk_index",
    "page_start",
    "page_end",
    "heading_path",
}


def validate_store(output_dir: Path, *, strict_source_match: bool) -> dict[str, object]:
    """
    Store validation:
    1. Open only the expected Chroma collection.
    2. Check reference chunk/document counts and governed metadata.
    3. Check the stored vector dimensionality.
    4. Surface or reject any source substitutions recorded during the build.
    """

    if version("chromadb") != CHROMADB_VERSION:
        raise RuntimeError(f"ChromaDB {CHROMADB_VERSION} is required")
    import chromadb

    catalog = json.loads(BUILD_CATALOG_PATH.read_text(encoding="utf-8"))
    expected = catalog["expected"]
    store_dir = output_dir / "chromadb"
    if not store_dir.is_dir():
        raise FileNotFoundError(f"vector-store directory is unavailable: {store_dir}")
    client = chromadb.PersistentClient(path=str(store_dir))
    names = [item.name for item in client.list_collections()]
    if names != [COLLECTION_NAME]:
        raise RuntimeError(f"unexpected collections: {names}")
    collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    snapshot = collection.get(include=["metadatas"])
    if collection.count() != int(expected["chunks"]):
        raise RuntimeError(f"expected {expected['chunks']} chunks, found {collection.count()}")
    document_ids = {metadata.get("document_id") for metadata in snapshot["metadatas"]}
    if len(document_ids) != int(expected["document_ids"]):
        raise RuntimeError(f"expected {expected['document_ids']} document IDs, found {len(document_ids)}")
    missing_contract = [metadata for metadata in snapshot["metadatas"] if not REQUIRED_METADATA.issubset(metadata)]
    if missing_contract:
        raise RuntimeError(f"{len(missing_contract)} chunks lack required governed metadata")

    sample = collection.get(ids=[snapshot["ids"][0]], include=["embeddings"])
    dimensions = len(sample["embeddings"][0])
    if dimensions != EMBEDDING_DIMENSIONS:
        raise RuntimeError(f"expected {EMBEDDING_DIMENSIONS} dimensions, found {dimensions}")

    build_manifest_path = output_dir / "gemini_manifest.json"
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    substitutions = list(build_manifest.get("source_substitutions", []))
    if strict_source_match and substitutions:
        raise RuntimeError(f"strict source validation failed: {len(substitutions)} substitution(s) were used")
    return {
        "status": "valid",
        "collection": COLLECTION_NAME,
        "chunks": collection.count(),
        "document_ids": len(document_ids),
        "embedding_dimensions": dimensions,
        "source_substitutions": substitutions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strict-source-match",
        action="store_true",
        help="fail when the build manifest records any source-file substitution",
    )
    args = parser.parse_args()
    summary = validate_store(args.output_dir, strict_source_match=args.strict_source_match)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
