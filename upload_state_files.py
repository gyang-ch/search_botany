#!/usr/bin/env python3
"""
Upload books.jsonl, page_log.jsonl, and processed_items.json from each
pipeline's run directory to Azure Blob Storage, under a "state_backups/"
prefix, using the same credentials the pipelines already use.

Usage (from /workspace/search_botany on the runpod machine):

    python upload_state_files.py

Reads credentials from .env / the environment, same precedence as the
pipeline scripts:
    1. AZURE_STORAGE_CONTAINER_SAS_URL
    2. VITE_AZURE_BLOB_BASE + VITE_AZURE_SAS_TOKEN
       (or AZURE_BLOB_BASE + AZURE_SAS_TOKEN)

Optional flags:
    --prefix state_backups   Blob path prefix (default: state_backups)
    --dry-run                List what would be uploaded, upload nothing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# (blob sub-prefix, local run dir). The illustration pipelines nest under
# "illustration_runs/" - keep this in sync with each pipeline's own
# --state-azure-prefix default, since that's the path its in-run periodic
# backup writes to; this script's ad-hoc backups should land in the same
# place, not a sibling path that never gets cleaned up.
RUN_DIRS = [
    ("", "loc_yolo_run"),
    ("", "bodleian_yolo_run"),
    ("", "british_library_yolo_run"),
    ("illustration_runs/", "bodleian_illustration_yolo_run"),
    ("illustration_runs/", "gallica_illustration_yolo_run"),
]
FILENAMES = ["books.jsonl", "page_log.jsonl", "processed_items.json"]

CONTENT_TYPES = {
    ".jsonl": "application/x-ndjson",
    ".json": "application/json",
}


def resolve_container_sas_url() -> str:
    explicit = os.environ.get("AZURE_STORAGE_CONTAINER_SAS_URL")
    if explicit:
        return explicit

    base = os.environ.get("VITE_AZURE_BLOB_BASE") or os.environ.get("AZURE_BLOB_BASE")
    token = os.environ.get("VITE_AZURE_SAS_TOKEN") or os.environ.get("AZURE_SAS_TOKEN")
    if base and token:
        token = token.lstrip("?")
        return f"{base}?{token}"

    raise RuntimeError(
        "Azure credentials missing. Set AZURE_STORAGE_CONTAINER_SAS_URL, or "
        "VITE_AZURE_BLOB_BASE + VITE_AZURE_SAS_TOKEN in .env."
    )


def get_container_client():
    from azure.storage.blob import ContainerClient

    return ContainerClient.from_container_url(resolve_container_sas_url())


def upload_file(container_client, local_path: Path, blob_name: str) -> None:
    from azure.storage.blob import ContentSettings

    content_type = CONTENT_TYPES.get(local_path.suffix, "text/plain")
    with local_path.open("rb") as f:
        container_client.upload_blob(
            name=blob_name,
            data=f,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="state_backups")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    prefix = args.prefix.strip("/")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    to_upload: list[tuple[Path, str]] = []
    for sub_prefix, run_dir in RUN_DIRS:
        dir_path = root / run_dir
        if not dir_path.is_dir():
            print(f"[skip] {run_dir}/ not found")
            continue
        for filename in FILENAMES:
            local_path = dir_path / filename
            if not local_path.is_file():
                print(f"[skip] {run_dir}/{filename} not found")
                continue
            blob_name = f"{prefix}/{sub_prefix}{run_dir}/{filename}"
            to_upload.append((local_path, blob_name))

    if not to_upload:
        print("Nothing to upload.")
        return 0

    if args.dry_run:
        print("Dry run - would upload:")
        for local_path, blob_name in to_upload:
            size_kb = local_path.stat().st_size / 1024
            print(f"  {local_path}  ->  {blob_name}  ({size_kb:.1f} KB)")
        return 0

    container_client = get_container_client()

    for local_path, blob_name in to_upload:
        size_kb = local_path.stat().st_size / 1024
        print(f"[upload] {local_path} -> {blob_name} ({size_kb:.1f} KB)")
        upload_file(container_client, local_path, blob_name)

    # Also write a timestamped copy of processed_items.json so you keep a
    # history of state snapshots over time, not just the latest overwrite.
    for sub_prefix, run_dir in RUN_DIRS:
        local_path = root / run_dir / "processed_items.json"
        if local_path.is_file():
            snap_blob = f"{prefix}/{sub_prefix}{run_dir}/snapshots/processed_items_{timestamp}.json"
            print(f"[snapshot] {local_path} -> {snap_blob}")
            upload_file(container_client, local_path, snap_blob)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
