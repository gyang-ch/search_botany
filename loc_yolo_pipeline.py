#!/usr/bin/env python3
"""
loc_yolo_pipeline.py

Large-scale Library of Congress book-page harvesting with GPU YOLO triage.

Workflow
--------
1. Search the LOC Books endpoint for each keyword in KEYWORDS (botany,
   herbal, materia medica, ...).
2. For each matching book, fetch item metadata and resolve the IIIF
   Presentation manifest (falling back to LOC item "files" resources when a
   manifest can't be found), same approach as loc_vlm_triage.py.
3. Walk every page of the book, ONE PAGE AT A TIME:
     a. download the page image to local disk
     b. run it through the YOLO model on the GPU (CUDA) immediately
     c. if any detected class is plant-related  -> upload to Azure Blob
        Storage, then delete the local file
        if no plant-related class is detected   -> delete the local file
   The pipeline never holds more than one page image on local disk at a
   time, which keeps a RunPod pod's ephemeral storage bounded no matter how
   many/large the books are.
4. Progress is checkpointed after every page to processed_items.json, so a
   killed/restarted run resumes mid-book instead of re-downloading or
   re-uploading pages. A book already fully processed under one keyword is
   skipped when a later keyword matches the same LOC item again.
5. Every page decision (kept/deleted/error) is appended to page_log.jsonl
   for audit purposes.

Default YOLO model
-------------------
Defaults to ./best.pt, a model already trained for this project
(botany_detection_v6) with classes:
    animal, animal_bird, animal_fish, animal_snake, human,
    plant_flower, plant_fruit, plant_grass, plant_herb,
    plant_herb_aquatic, plant_root, plant_tree, plant_wood
A page is treated as "contains a plant" when any detected class name
contains "plant" (see --plant-class-keywords to change this).

Environment
-----------
Credentials are read from environment variables (never hardcode them in this
file, since it goes to GitHub). Put them in a local `.env` (gitignored, see
`.env.example`) or set them as RunPod pod environment variables / secrets.

Either:
    AZURE_STORAGE_CONNECTION_STRING="...."
Or a container SAS URL, either as one variable:
    AZURE_STORAGE_CONTAINER_SAS_URL="https://<account>.blob.core.windows.net/<container>?<sas>"
or split into base + token (matches this project's existing frontend .env
naming, so the same .env can be reused as-is):
    VITE_AZURE_BLOB_BASE="https://<account>.blob.core.windows.net/<container>"
    VITE_AZURE_SAS_TOKEN="sp=...&sig=..."

The SAS token MUST include write permission (sp= must contain "w", typically
"c" too, e.g. sp=racwd) since this pipeline uploads pages. A read-only token
(sp=r) will be rejected at startup with a clear error before any GPU work
happens.

Install (RunPod / GPU machine)
-------------------------------
pip install requests ultralytics azure-storage-blob python-dotenv
# torch/torchvision should already be present in most RunPod CUDA images;
# otherwise install a CUDA build from https://pytorch.org/get-started/locally/

Example
-------
python loc_yolo_pipeline.py \
    --azure-container botany-pages \
    --max-books-per-keyword 100 \
    --output-dir loc_yolo_run

Dry run (no Azure credentials needed, still deletes local files):
python loc_yolo_pipeline.py --dry-run --max-books-per-keyword 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()  # Loads a local .env if present; no-op otherwise.
except ImportError:
    pass

LOC_BASE = "https://www.loc.gov"
SCRIPT_DIR = Path(__file__).resolve().parent

USER_AGENT = (
    "Phyto-Vision-LOC-YOLO-Pipeline/1.0 "
    "(research; contact information should be added by researcher)"
)

DEFAULT_KEYWORDS = [
    "botany",
    "botanical",
    "herbal",
    "herbals",
    "plants",
    "medicinal plants",
    "materia medica",
    "flora",
    "flowers",
    "horticulture",
    "gardening",
    "plant anatomy",
    "natural history",
    "pharmacopoeia",
    "herbarium",
    "agriculture",
    "trees",
    "forestry",
]

DEFAULT_YOLO_MODEL = str(SCRIPT_DIR / "best.pt")
DEFAULT_IMGSZ = 640
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_PLANT_CLASS_KEYWORDS = ["plant"]

DEFAULT_MAX_BOOKS_PER_KEYWORD = 100
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 5


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def default_container_sas_url() -> str | None:
    """
    Resolve a container SAS URL from environment variables without ever
    needing the secret hardcoded in this file.

    Checks, in order:
    1. AZURE_STORAGE_CONTAINER_SAS_URL (one full URL, includes the "?").
    2. VITE_AZURE_BLOB_BASE + VITE_AZURE_SAS_TOKEN (this project's existing
       frontend .env naming, reused as-is) or the unprefixed
       AZURE_BLOB_BASE + AZURE_SAS_TOKEN equivalents.
    """
    explicit = os.environ.get("AZURE_STORAGE_CONTAINER_SAS_URL")
    if explicit:
        return explicit

    base = os.environ.get("VITE_AZURE_BLOB_BASE") or os.environ.get(
        "AZURE_BLOB_BASE"
    )
    token = os.environ.get("VITE_AZURE_SAS_TOKEN") or os.environ.get(
        "AZURE_SAS_TOKEN"
    )

    if base and token:
        return f"{base.rstrip('/')}?{token.lstrip('?')}"

    return None


def validate_sas_write_permission(sas_url: str) -> None:
    """
    Fail fast with a clear message if the SAS token can't write, instead of
    burning GPU time downloading/classifying pages before the first upload
    hits a 403.
    """
    query = urlparse(sas_url).query
    permissions = parse_qs(query).get("sp", [""])[0]

    if "w" not in permissions:
        raise RuntimeError(
            "This Azure SAS token does not grant write permission "
            f"(sp={permissions!r}). Regenerate a container SAS in the "
            "Azure Portal (Storage Account -> Containers -> "
            "<container> -> Shared access tokens) with at least "
            "Read + Write + Create permissions (e.g. sp=racwd), then "
            "update VITE_AZURE_SAS_TOKEN / AZURE_STORAGE_CONTAINER_SAS_URL."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search LOC for botanical-subject books, triage every page with "
            "a GPU YOLO model, and stream plant-positive pages to Azure "
            "Blob Storage."
        )
    )

    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Override the built-in keyword list.",
    )
    parser.add_argument(
        "--collection",
        default="",
        help="Optional LOC collection slug, e.g. selected-digitized-books.",
    )
    parser.add_argument(
        "--max-books-per-keyword",
        type=int,
        default=DEFAULT_MAX_BOOKS_PER_KEYWORD,
        help="Maximum number of LOC search results to fetch per keyword.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="LOC search result page to start from for each keyword.",
    )
    parser.add_argument(
        "--limit-pages-per-book",
        type=int,
        default=0,
        help="Cap on pages processed per book. 0 means process every page.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("loc_yolo_run"),
        help="Directory for the state file, page log, and the one-page-at-a-"
        "time temporary download.",
    )

    # YOLO / GPU
    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL,
        help=f"Path to YOLO weights. Default: {DEFAULT_YOLO_MODEL}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"YOLO inference image size. Default: {DEFAULT_IMGSZ}",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Minimum detection confidence. Default: {DEFAULT_CONF_THRESHOLD}",
    )
    parser.add_argument(
        "--plant-class-keywords",
        nargs="*",
        default=DEFAULT_PLANT_CLASS_KEYWORDS,
        help="A page is kept if any detected class name contains one of "
        "these substrings (case-insensitive). "
        f"Default: {DEFAULT_PLANT_CLASS_KEYWORDS}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for YOLO, e.g. cuda:0. Default: auto-detect CUDA.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running on CPU if no CUDA GPU is available (slow; not "
        "recommended for large-scale harvesting).",
    )

    # Azure
    parser.add_argument(
        "--azure-connection-string",
        default=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
        help="Azure Storage connection string. Defaults to "
        "AZURE_STORAGE_CONNECTION_STRING.",
    )
    parser.add_argument(
        "--azure-container",
        default=os.environ.get("AZURE_STORAGE_CONTAINER"),
        help="Azure Blob container name (used with --azure-connection-string).",
    )
    parser.add_argument(
        "--azure-container-sas-url",
        default=default_container_sas_url(),
        help="Full container SAS URL, used instead of a connection string. "
        "Defaults to AZURE_STORAGE_CONTAINER_SAS_URL, or "
        "VITE_AZURE_BLOB_BASE + VITE_AZURE_SAS_TOKEN combined.",
    )
    parser.add_argument(
        "--azure-prefix",
        default="",
        help="Optional blob name prefix, e.g. 'botany/'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run detection and delete local files as usual, but skip the "
        "Azure upload. Useful for testing without cloud credentials.",
    )

    # HTTP
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds for LOC/IIIF requests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Delay after each successful LOC/image request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Maximum HTTP retries per request.",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------
# LOC / IIIF helpers (same approach as loc_vlm_triage.py)
# --------------------------------------------------------------------------


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json,image/jpeg,image/png,image/*,*/*;q=0.8",
        }
    )
    return session


def safe_id(value: str) -> str:
    value = value.strip().rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180] or "unknown"


def get_json(
    session: requests.Session,
    url: str,
    timeout: int,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    if sleep_seconds:
        time.sleep(sleep_seconds)
    return response.json()


def search_books_paginated(
    session: requests.Session,
    query: str,
    collection: str,
    max_results: int,
    timeout: int,
    sleep_seconds: float,
    start_page: int = 1,
) -> list[dict[str, Any]]:
    endpoint = (
        f"{LOC_BASE}/collections/{collection}/"
        if collection
        else f"{LOC_BASE}/books/"
    )

    results: list[dict[str, Any]] = []
    sp = start_page
    per_page = 100

    while len(results) < max_results:
        params: dict[str, Any] = {"fo": "json", "c": per_page, "sp": sp}
        if query:
            params["q"] = query

        try:
            response = session.get(endpoint, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            print(f"[warn] search failed for query={query!r} sp={sp}: {exc}")
            break

        page_results = data.get("results", [])
        if not isinstance(page_results, list) or not page_results:
            break

        results.extend(page_results)

        if sleep_seconds:
            time.sleep(sleep_seconds)

        pagination = data.get("pagination") or {}
        total_pages = pagination.get("total")
        sp += 1
        if isinstance(total_pages, int) and sp > total_pages:
            break

    return results[:max_results]


def extract_item_id(result: dict[str, Any]) -> str | None:
    value = result.get("id") or result.get("url")
    if not isinstance(value, str):
        return None

    match = re.search(r"/item/([^/?#]+)/?", value)
    if match:
        return match.group(1)

    return safe_id(value)


def item_json_url(result: dict[str, Any]) -> str | None:
    item_id = extract_item_id(result)
    if not item_id:
        return None
    return f"{LOC_BASE}/item/{item_id}/?fo=json"


def recursive_find_manifest_url(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in {
                "iiif_manifest",
                "iiif_manifest_url",
                "manifest",
                "manifest_url",
            }:
                if isinstance(value, str):
                    normalized = value.lower().rstrip("/")
                    if "iiif" in normalized or normalized.endswith(
                        "manifest.json"
                    ):
                        return value

            found = recursive_find_manifest_url(value)
            if found:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = recursive_find_manifest_url(value)
            if found:
                return found

    return None


def extract_image_base_from_url(url: str) -> str | None:
    if not isinstance(url, str):
        return None

    if "/full/" in url:
        return url.split("/full/")[0]

    if url.endswith("/info.json"):
        return url[: -len("/info.json")]

    return None


def fallback_page_images(item_data: dict[str, Any]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    seen: set[str] = set()

    for resource in item_data.get("resources", []) or []:
        for file_group in resource.get("files", []) or []:
            if not isinstance(file_group, list):
                continue

            for file_obj in file_group:
                if not isinstance(file_obj, dict):
                    continue

                url = file_obj.get("url", "")
                base = extract_image_base_from_url(url)

                if base and base not in seen:
                    seen.add(base)
                    pages.append(
                        {
                            "id": len(pages),
                            "label": f"Page {len(pages) + 1}",
                            "image_service": base,
                            "image_url": f"{base}/full/1200,/0/default.jpg",
                            "source_url": url,
                        }
                    )

    return pages


def service_to_base(service: Any) -> str | None:
    if isinstance(service, str):
        return service.rstrip("/")

    if isinstance(service, dict):
        value = service.get("id") or service.get("@id")
        if isinstance(value, str):
            return value.rstrip("/")

    return None


def image_service_from_body(body: Any) -> str | None:
    if isinstance(body, dict):
        service = body.get("service")
        if isinstance(service, list):
            for item in service:
                base = service_to_base(item)
                if base:
                    return base
        else:
            base = service_to_base(service)
            if base:
                return base

        body_id = body.get("id") or body.get("@id")
        if isinstance(body_id, str) and (
            "iiif" in body_id.lower() or "image-services" in body_id.lower()
        ):
            return extract_image_base_from_url(body_id) or body_id.rstrip("/")

    return None


def parse_iiif_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []

    # IIIF Presentation 3
    if isinstance(manifest.get("items"), list):
        for canvas_index, canvas in enumerate(manifest["items"]):
            if not isinstance(canvas, dict):
                continue

            label = canvas.get("label", f"Page {canvas_index + 1}")
            if isinstance(label, dict):
                label = (
                    next(iter(label.get("none", [])), None)
                    or next(iter(label.values()), f"Page {canvas_index + 1}")
                )

            service_base: str | None = None
            for annotation_page in canvas.get("items", []):
                if not isinstance(annotation_page, dict):
                    continue

                for annotation in annotation_page.get("items", []):
                    if not isinstance(annotation, dict):
                        continue

                    body = annotation.get("body", {})
                    service_base = image_service_from_body(body)
                    if service_base:
                        break

                if service_base:
                    break

            if service_base:
                pages.append(
                    {
                        "id": canvas_index,
                        "label": str(label),
                        "image_service": service_base,
                        "image_url": f"{service_base}/full/1200,/0/default.jpg",
                    }
                )

        if pages:
            return pages

    # IIIF Presentation 2
    sequences = manifest.get("sequences", [])
    if isinstance(sequences, list) and sequences:
        canvases = sequences[0].get("canvases", [])

        for canvas_index, canvas in enumerate(canvases):
            if not isinstance(canvas, dict):
                continue

            label = canvas.get("label", f"Page {canvas_index + 1}")

            service_base = None
            for image_annotation in canvas.get("images", []) or []:
                if not isinstance(image_annotation, dict):
                    continue

                resource = image_annotation.get("resource", {})
                service_base = image_service_from_body(resource)

                if service_base:
                    break

            if service_base:
                pages.append(
                    {
                        "id": canvas_index,
                        "label": str(label),
                        "image_service": service_base,
                        "image_url": f"{service_base}/full/1200,/0/default.jpg",
                    }
                )

    return pages


def download_image(
    session: requests.Session,
    image_url: str,
    output_path: Path,
    timeout: int,
    sleep_seconds: float,
) -> None:
    partial_path = output_path.with_name(output_path.name + ".part")

    try:
        response = session.get(image_url, timeout=timeout, stream=True)
        response.raise_for_status()

        with partial_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    handle.write(chunk)

        os.replace(partial_path, output_path)
    finally:
        partial_path.unlink(missing_ok=True)

    if sleep_seconds:
        time.sleep(sleep_seconds)


def download_image_with_retry(
    session: requests.Session,
    image_url: str,
    output_path: Path,
    timeout: int,
    sleep_seconds: float,
    retries: int,
) -> None:
    delay = 2.0

    for attempt in range(retries):
        try:
            download_image(
                session=session,
                image_url=image_url,
                output_path=output_path,
                timeout=timeout,
                sleep_seconds=sleep_seconds,
            )
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    raise RuntimeError("Unreachable image retry state.")


# --------------------------------------------------------------------------
# YOLO (GPU) detection
# --------------------------------------------------------------------------


def resolve_device(requested_device: str | None, allow_cpu: bool) -> str:
    import torch

    if requested_device:
        return requested_device

    if torch.cuda.is_available():
        return "cuda:0"

    if not allow_cpu:
        print(
            "No CUDA GPU detected. This pipeline is meant to run on a "
            "CUDA-enabled machine (e.g. a RunPod GPU pod). Pass --allow-cpu "
            "to run on CPU anyway (much slower).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("[warn] CUDA not available; running YOLO on CPU.")
    return "cpu"


def load_yolo_model(model_path: str, device: str):
    from ultralytics import YOLO

    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO weights not found: {model_path}")

    model = YOLO(model_path)
    model.to(device)
    return model


def detect_plant(
    model,
    image_path: Path,
    device: str,
    imgsz: int,
    conf_threshold: float,
    plant_class_keywords: list[str],
) -> tuple[bool, list[dict[str, Any]], float]:
    results = model.predict(
        source=str(image_path),
        device=device,
        imgsz=imgsz,
        conf=conf_threshold,
        verbose=False,
    )

    detected: list[dict[str, Any]] = []
    max_confidence = 0.0
    has_plant = False
    keywords_lower = [kw.lower() for kw in plant_class_keywords]

    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for cls_idx, confidence in zip(
            boxes.cls.tolist(), boxes.conf.tolist()
        ):
            class_name = names[int(cls_idx)] if isinstance(names, (list, dict)) else str(cls_idx)
            if isinstance(names, dict):
                class_name = names.get(int(cls_idx), str(cls_idx))

            confidence = float(confidence)
            detected.append({"class": class_name, "confidence": confidence})
            max_confidence = max(max_confidence, confidence)

            if any(keyword in class_name.lower() for keyword in keywords_lower):
                has_plant = True

    return has_plant, detected, max_confidence


# --------------------------------------------------------------------------
# Azure Blob Storage
# --------------------------------------------------------------------------


def get_container_client(args: argparse.Namespace):
    from azure.storage.blob import BlobServiceClient, ContainerClient

    if args.azure_container_sas_url:
        validate_sas_write_permission(args.azure_container_sas_url)
        return ContainerClient.from_container_url(args.azure_container_sas_url)

    if not args.azure_connection_string:
        raise RuntimeError(
            "Azure credentials missing. Pass --azure-connection-string, "
            "--azure-container-sas-url, or set "
            "AZURE_STORAGE_CONNECTION_STRING."
        )

    if not args.azure_container:
        raise RuntimeError(
            "--azure-container is required when using a connection string."
        )

    service_client = BlobServiceClient.from_connection_string(
        args.azure_connection_string
    )
    container_client = service_client.get_container_client(args.azure_container)

    try:
        container_client.create_container()
    except Exception:
        pass  # Container already exists.

    return container_client


def upload_image_to_blob(
    container_client, blob_name: str, image_path: Path
) -> None:
    from azure.storage.blob import ContentSettings

    with image_path.open("rb") as handle:
        container_client.upload_blob(
            name=blob_name,
            data=handle,
            overwrite=True,
            content_settings=ContentSettings(content_type="image/jpeg"),
        )


# --------------------------------------------------------------------------
# State / logging
# --------------------------------------------------------------------------


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state_atomic(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def append_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Per-page / per-book pipeline
# --------------------------------------------------------------------------


def process_page(
    *,
    session: requests.Session,
    model,
    container_client,
    args: argparse.Namespace,
    device: str,
    item_id: str,
    keyword: str,
    page: dict[str, Any],
    page_index: int,
    tmp_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    page_id = safe_id(str(page.get("id", page_index)))
    tmp_path = tmp_dir / f"page_{page_index + 1:05d}_{page_id}.jpg"

    record: dict[str, Any] = {
        "loc_item_id": item_id,
        "keyword": keyword,
        "page_index": page_index,
        "page_number": page_index + 1,
        "image_url": page.get("image_url"),
        "action": None,
        "blob_name": None,
        "detected_classes": [],
        "max_confidence": None,
        "error": None,
    }

    try:
        # 1. Download exactly one page image.
        download_image_with_retry(
            session=session,
            image_url=page["image_url"],
            output_path=tmp_path,
            timeout=args.timeout,
            sleep_seconds=args.sleep,
            retries=args.retries,
        )

        # 2. Run it through YOLO on the GPU immediately.
        has_plant, detected, max_confidence = detect_plant(
            model=model,
            image_path=tmp_path,
            device=device,
            imgsz=args.imgsz,
            conf_threshold=args.conf_threshold,
            plant_class_keywords=args.plant_class_keywords,
        )
        record["detected_classes"] = detected
        record["max_confidence"] = max_confidence

        # 3a. Plant detected -> upload, then delete local copy.
        if has_plant:
            prefix = args.azure_prefix.strip("/")
            blob_name = (
                f"{prefix}/{safe_id(item_id)}/page_{page_index + 1:05d}.jpg"
                if prefix
                else f"{safe_id(item_id)}/page_{page_index + 1:05d}.jpg"
            )

            if not args.dry_run:
                upload_image_to_blob(container_client, blob_name, tmp_path)
                record["action"] = "kept_uploaded"
            else:
                record["action"] = "kept_dry_run"

            record["blob_name"] = blob_name

        # 3b. No plant -> delete local copy, nothing uploaded.
        else:
            record["action"] = "deleted_no_plant"

    except Exception as exc:
        record["action"] = "error"
        record["error"] = str(exc)

    finally:
        # Always remove the local copy from the RunPod machine's disk.
        tmp_path.unlink(missing_ok=True)

    append_log(log_path, record)
    return record


def process_book(
    *,
    result: dict[str, Any],
    session: requests.Session,
    model,
    container_client,
    args: argparse.Namespace,
    device: str,
    state: dict[str, Any],
    log_path: Path,
    keyword: str,
) -> None:
    item_id = extract_item_id(result)
    if not item_id:
        print("[skip] could not determine LOC item id from search result")
        return

    item_state = state.get(item_id)

    if item_state and item_state.get("status") == "completed":
        keywords_matched = item_state.setdefault("keywords_matched", [])
        if keyword not in keywords_matched:
            keywords_matched.append(keyword)
            save_state_atomic(args.state_path, state)
        print(f"[skip] {item_id} already completed")
        return

    print(f"\n[book] {item_id} (keyword={keyword!r})")

    try:
        metadata_url = item_json_url(result)
        if not metadata_url:
            raise RuntimeError("Could not construct LOC item JSON URL.")

        item_data = get_json(session, metadata_url, args.timeout, args.sleep)

        manifest_url = recursive_find_manifest_url(item_data)
        if manifest_url and manifest_url.startswith("//"):
            manifest_url = "https:" + manifest_url

        pages: list[dict[str, Any]] = []

        if manifest_url:
            try:
                manifest_data = get_json(
                    session, manifest_url, args.timeout, args.sleep
                )
                pages = parse_iiif_manifest(manifest_data)
            except Exception as exc:
                print(f"[warn] {item_id}: manifest fetch failed: {exc}")

        if not pages:
            pages = fallback_page_images(item_data)

        if not pages:
            print(f"[skip] {item_id}: no page images could be resolved")
            state[item_id] = {
                "status": "failed",
                "error": "no page images found",
                "keywords_matched": [keyword],
            }
            save_state_atomic(args.state_path, state)
            return

        total_pages = len(pages)
        if args.limit_pages_per_book and args.limit_pages_per_book > 0:
            total_pages = min(total_pages, args.limit_pages_per_book)
            pages = pages[:total_pages]

        start_index = item_state.get("next_page_index", 0) if item_state else 0
        pages_kept = item_state.get("pages_kept", 0) if item_state else 0
        pages_deleted = item_state.get("pages_deleted", 0) if item_state else 0
        keywords_matched = (
            list(item_state.get("keywords_matched", [])) if item_state else []
        )
        if keyword not in keywords_matched:
            keywords_matched.append(keyword)

        tmp_dir = args.output_dir / "tmp_page" / safe_id(item_id)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for page_index in range(start_index, total_pages):
            record = process_page(
                session=session,
                model=model,
                container_client=container_client,
                args=args,
                device=device,
                item_id=item_id,
                keyword=keyword,
                page=pages[page_index],
                page_index=page_index,
                tmp_dir=tmp_dir,
                log_path=log_path,
            )

            if record["action"] in ("kept_uploaded", "kept_dry_run"):
                pages_kept += 1
            elif record["action"] == "deleted_no_plant":
                pages_deleted += 1

            state[item_id] = {
                "status": "in_progress",
                "next_page_index": page_index + 1,
                "page_count": total_pages,
                "pages_kept": pages_kept,
                "pages_deleted": pages_deleted,
                "keywords_matched": keywords_matched,
            }
            save_state_atomic(args.state_path, state)

        state[item_id]["status"] = "completed"
        save_state_atomic(args.state_path, state)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        print(
            f"[done] {item_id}: kept={pages_kept} "
            f"deleted={pages_deleted} total={total_pages}"
        )

    except Exception as exc:
        print(f"[error] {item_id}: {exc}")
        previous = state.get(item_id) or {}
        keywords_matched = set(previous.get("keywords_matched", []))
        keywords_matched.add(keyword)
        state[item_id] = {
            **previous,
            "status": "failed",
            "error": str(exc),
            "keywords_matched": sorted(keywords_matched),
        }
        save_state_atomic(args.state_path, state)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.state_path = args.output_dir / "processed_items.json"
    log_path = args.output_dir / "page_log.jsonl"

    device = resolve_device(args.device, args.allow_cpu)
    print(f"Using device: {device}")

    try:
        model = load_yolo_model(args.yolo_model, device)
    except Exception as exc:
        print(f"Failed to load YOLO model: {exc}", file=sys.stderr)
        return 1

    container_client = None
    if not args.dry_run:
        try:
            container_client = get_container_client(args)
        except Exception as exc:
            print(f"Azure setup failed: {exc}", file=sys.stderr)
            return 1
    else:
        print(
            "[dry-run] Azure upload disabled; plant-positive pages are "
            "detected but not uploaded (still deleted locally)."
        )

    state = load_state(args.state_path)
    session = make_session()

    keywords = args.keywords or DEFAULT_KEYWORDS
    print(f"Keywords ({len(keywords)}): {', '.join(keywords)}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Plant class keywords: {args.plant_class_keywords}")
    print(f"Output dir: {args.output_dir}")

    for keyword in keywords:
        print(f"\n=== Keyword: {keyword!r} ===")
        try:
            search_results = search_books_paginated(
                session=session,
                query=keyword,
                collection=args.collection,
                max_results=args.max_books_per_keyword,
                timeout=args.timeout,
                sleep_seconds=args.sleep,
                start_page=args.start_page,
            )
        except Exception as exc:
            print(f"[warn] search failed for {keyword!r}: {exc}")
            continue

        print(f"Found {len(search_results)} candidate item(s) for {keyword!r}")

        for result in search_results:
            process_book(
                result=result,
                session=session,
                model=model,
                container_client=container_client,
                args=args,
                device=device,
                state=state,
                log_path=log_path,
                keyword=keyword,
            )

    completed = sum(1 for v in state.values() if v.get("status") == "completed")
    failed = sum(1 for v in state.values() if v.get("status") == "failed")
    kept_total = sum(v.get("pages_kept", 0) for v in state.values())
    deleted_total = sum(v.get("pages_deleted", 0) for v in state.values())

    print("\n=== Run complete ===")
    print(f"Books completed: {completed}")
    print(f"Books failed: {failed}")
    print(f"Pages kept (uploaded to Azure): {kept_total}")
    print(f"Pages deleted (no plant detected): {deleted_total}")
    print(f"State file: {args.state_path}")
    print(f"Page log: {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
