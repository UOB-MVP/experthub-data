"""Cloud Run entrypoint: restore state from GCS, run preflight + build + validate, save state back.

Cloud Run Jobs execute this container to completion and then throw the execution environment away,
so the Chroma store (`chromadb/`) and the embedded-document bookkeeping (`result/`) are mirrored
with Cloud Storage before and after each run. All Vertex/Chroma settings are read directly from the
process environment (Cloud Run env vars/secrets); no .env file is used in the container.
"""

from __future__ import annotations

import json
import os
import sys

import build_vector_store
import gcs_sync
import validate_vector_store

DATA_DIR = build_vector_store.DEFAULT_DATA_DIR
OUTPUT_DIR = build_vector_store.DEFAULT_OUTPUT_DIR
RESULT_DIR = build_vector_store.DEFAULT_RESULT_DIR


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> int:
    bucket = os.environ.get("EXPERT_HUB_GCS_BUCKET", "").strip()
    if not bucket:
        print("EXPERT_HUB_GCS_BUCKET is required", file=sys.stderr)
        return 1

    chromadb_prefix = os.environ.get("GCS_CHROMADB_PREFIX", "chromadb").strip()
    result_prefix = os.environ.get("GCS_RESULT_PREFIX", "result").strip()
    allow_substitution = _env_flag("ALLOW_KNOWN_UOB_SUBSTITUTION", True)
    strict_source_match = _env_flag("STRICT_SOURCE_MATCH", False)

    print(f"restoring state from gs://{bucket}/{chromadb_prefix}/ and gs://{bucket}/{result_prefix}/", flush=True)
    gcs_sync.download_dir(bucket, chromadb_prefix, OUTPUT_DIR)
    gcs_sync.download_dir(bucket, result_prefix, RESULT_DIR)

    exit_code = 0
    try:
        print("running credential-free preflight", flush=True)
        build_vector_store.run_build(
            DATA_DIR, OUTPUT_DIR, None, RESULT_DIR, dry_run=True, allow_known_substitution=allow_substitution
        )

        print("running build", flush=True)
        build_vector_store.run_build(
            DATA_DIR, OUTPUT_DIR, None, RESULT_DIR, dry_run=False, allow_known_substitution=allow_substitution
        )

        print("validating store", flush=True)
        summary = validate_vector_store.validate_store(OUTPUT_DIR, strict_source_match=strict_source_match)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    except Exception as error:  # noqa: BLE001 - report and still save partial state below
        print(f"pipeline failed: {error}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        print(f"saving state to gs://{bucket}/{chromadb_prefix}/ and gs://{bucket}/{result_prefix}/", flush=True)
        gcs_sync.upload_dir(bucket, chromadb_prefix, OUTPUT_DIR)
        gcs_sync.upload_dir(bucket, result_prefix, RESULT_DIR)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
