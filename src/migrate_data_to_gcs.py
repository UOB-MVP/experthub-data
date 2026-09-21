"""Migrate the local data/ source corpus into Cloud Storage, preserving its folder structure.

Uploads every file under DATA_DIR to gs://<bucket>/chromadb/data/<same relative path>. A
companion gs://<bucket>/chromadb/filenames.json record tracks every migrated file's absolute
path from the bucket root (e.g. "chromadb/data/01 Company disclosures and comms/.../report.pdf")
and content hash, so that a rerun:

- skips any file whose bucket path is already recorded (already migrated), and
- skips any file whose content is byte-identical to one already uploaded (a true duplicate),
  even if it lives at a different path.

Filename collisions across category folders (for example, several banks each having a
"ceo-slides-2q-2026.pdf") are NOT treated as duplicates, since each gets its own full bucket
path -- only identical content is ever skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from build_vector_store import truncated_sha256

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_PREFIX = "chromadb/data"
RECORD_BLOB_NAME = "chromadb/filenames.json"


def load_record(bucket) -> dict[str, str]:
    """Return the {absolute_bucket_path: content_hash} record of everything already migrated.

    Keys are the full path from the bucket root (e.g. "chromadb/data/01 Company disclosures and
    comms/Annual reports/BankX/report.pdf"), not bare filenames, so files that share a name but
    live in different category folders never collide.
    """

    blob = bucket.blob(RECORD_BLOB_NAME)
    if not blob.exists():
        return {}
    return json.loads(blob.download_as_text())


def save_record(bucket, record: dict[str, str]) -> None:
    """Persist the updated migration record."""

    blob = bucket.blob(RECORD_BLOB_NAME)
    blob.upload_from_string(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        content_type="application/json",
    )


def migrate(data_dir: Path, bucket_name: str) -> dict[str, object]:
    """Upload every not-yet-migrated, non-duplicate file and return a run summary."""

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    record = load_record(bucket)
    hashes_seen = set(record.values())

    uploaded = 0
    skipped_already_migrated = 0
    skipped_duplicate_content = 0

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(data_dir).as_posix()
        bucket_path = f"{DATA_PREFIX}/{relative_path}"
        if bucket_path in record:
            skipped_already_migrated += 1
            continue

        content_hash = truncated_sha256(path)
        if content_hash in hashes_seen:
            skipped_duplicate_content += 1
            print(f"skipping duplicate content: {bucket_path} (hash {content_hash})", flush=True)
            continue

        bucket.blob(bucket_path).upload_from_filename(str(path))
        record[bucket_path] = content_hash
        hashes_seen.add(content_hash)
        uploaded += 1
        print(f"uploaded {bucket_path}", flush=True)

    save_record(bucket, record)
    return {
        "uploaded": uploaded,
        "skipped_already_migrated": skipped_already_migrated,
        "skipped_duplicate_content": skipped_duplicate_content,
        "total_tracked_files": len(record),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bucket", default=os.environ.get("EXPERT_HUB_GCS_BUCKET", "").strip())
    args = parser.parse_args()

    if not args.bucket:
        print("EXPERT_HUB_GCS_BUCKET is required (env var or --bucket)", file=sys.stderr)
        return 1
    if not args.data_dir.is_dir():
        print(f"data directory not found: {args.data_dir}", file=sys.stderr)
        return 1

    summary = migrate(args.data_dir, args.bucket)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
