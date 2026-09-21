"""Mirror a local directory with a Cloud Storage prefix for the Cloud Run build pipeline.

Cloud Run Jobs are stateless between executions, so the Chroma store and the embedded-document
bookkeeping have to be pulled down before a build and pushed back up afterward. These helpers
keep a GCS prefix and a local directory as an exact mirror (uploads new/changed files, deletes
files that no longer exist locally).
"""

from __future__ import annotations

from pathlib import Path


def _normalized_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else f"{prefix}/"


def download_dir(bucket_name: str, prefix: str, local_dir: Path) -> int:
    """Download every object under gs://bucket_name/prefix/ into local_dir. Returns file count."""

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    normalized = _normalized_prefix(prefix)
    count = 0
    for blob in bucket.list_blobs(prefix=normalized):
        relative = blob.name[len(normalized) :]
        if not relative or blob.name.endswith("/"):
            continue
        target = local_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        count += 1
    return count


def upload_dir(bucket_name: str, prefix: str, local_dir: Path) -> int:
    """Mirror local_dir to gs://bucket_name/prefix/, uploading changes and deleting orphans."""

    from google.cloud import storage

    if not local_dir.is_dir():
        return 0

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    normalized = _normalized_prefix(prefix)

    local_files = {path.relative_to(local_dir).as_posix() for path in local_dir.rglob("*") if path.is_file()}
    for relative in local_files:
        bucket.blob(f"{normalized}{relative}").upload_from_filename(str(local_dir / relative))

    remote_files = {
        blob.name[len(normalized) :]
        for blob in bucket.list_blobs(prefix=normalized)
        if not blob.name.endswith("/")
    }
    for relative in remote_files - local_files:
        bucket.blob(f"{normalized}{relative}").delete()

    return len(local_files)
