#!/usr/bin/env python3
"""
bodleian_yolo_pipeline.py

Large-scale Digital Bodleian book-page harvesting with GPU YOLO triage.
Sibling script to loc_yolo_pipeline.py - same one-page-at-a-time GPU triage
and Azure upload strategy, pointed at Digital Bodleian instead of the
Library of Congress (whose transfer speed was too slow for this project).

API reference
-------------
https://digital.bodleian.ox.ac.uk/developer/
https://digital.bodleian.ox.ac.uk/developer/data/

Digital Bodleian search endpoint: GET https://digital.bodleian.ox.ac.uk/search/
    ?q=<query>&page=<n>&rows=<20|40|100>&fq=<lucene filter, repeatable>
Response is JSON-LD: {"member": [...], "view": {"totalPages": n}, ...}.
Each member already carries its own IIIF Presentation manifest URL
(member["manifest"]["id"]), so - unlike LOC - there is no separate item
metadata request: one manifest fetch per book gets both the bibliographic
metadata (manifest["metadata"], a list of {label, value} pairs) and every
page/canvas. Manifests seen in the wild are IIIF Presentation API 2.x
(top-level "sequences"), so parse_iiif_manifest() below (ported from
loc_yolo_pipeline.py/loc_vlm_triage.py) is written IIIF v2/v3 agnostic
and works unchanged.

Workflow
--------
1. Search Digital Bodleian for each keyword in KEYWORDS (botany, herbal,
   materia medica, ...).
2. For each matching object, fetch its IIIF manifest directly (URL is
   already in the search result) and enumerate every page/canvas.
3. Walk every page of the book, ONE PAGE AT A TIME:
     a. download the page image to local disk (all Bodleian HTTP calls use
        the same exponential-backoff retry as image downloads)
     b. run it through the YOLO model on the GPU (CUDA) immediately,
        recording every detection's class, confidence, AND bounding box
        (xyxy) - the GPU cost is already paid, so nothing is discarded
     c. if any detected class is plant-related  -> upload to Azure Blob
        Storage, then delete the local file
        if no plant-related class is detected   -> delete the local file,
        UNLESS it's randomly chosen for the negative QC audit sample
        (--negative-sample-rate), in which case it's uploaded to
        negative_audit/ instead of being discarded outright
   The pipeline never holds more than one page image on local disk at a
   time, which keeps a RunPod pod's ephemeral storage bounded no matter how
   many/large the books are.
4. Progress is checkpointed after every page to processed_items.json, so a
   killed/restarted run resumes mid-book instead of re-downloading or
   re-uploading pages. A book already fully processed under one keyword is
   skipped when a later keyword matches the same Bodleian object again.
   A page-level error (network drop, Bodleian 5xx, Azure failure) does NOT
   advance next_page_index and does NOT get counted as processed - the book
   stops there and the SAME page is retried on the next run, up to
   --max-page-retries attempts (across runs) before the book is marked
   "failed_permanent" and skipped like a completed book. This avoids
   silently creating holes in a book's page sequence.
5. Every page decision (kept/deleted/error) is appended to page_log.jsonl
   for audit purposes.
6. Book-level bibliographic metadata (title/author/date/subjects/language),
   the manifest's rights/attribution statement, page counts, plant-detection
   totals, and an illustration_density (positive_pages / total_pages) ratio
   are written to books.jsonl, one line per book, updated whenever that
   book's processing state changes. Use it to rank newly discovered books
   by how plant-illustration-dense they are without re-running detection.

Default YOLO model
-------------------
Defaults to ./best.pt, a model already trained for this project
(botany_detection_v6) with classes:
    animal, animal_bird, animal_fish, animal_snake, human,
    plant_flower, plant_fruit, plant_grass, plant_herb,
    plant_herb_aquatic, plant_root, plant_tree, plant_wood
A page is treated as "contains a plant" when any detected class name
contains "plant" (see --plant-class-keywords to change this).

Be a good API citizen
----------------------
Digital Bodleian's developer docs ask that large-scale manifest harvesting
be coordinated with their team first (digital-bodleian@bodleian.ox.ac.uk),
and that the User-Agent string include a contact e-mail instead of/along
with e-mailing them - set BODLEIAN_CONTACT_EMAIL (see below) before a large
run. They also note a preference against third parties re-hosting their
images long-term; this pipeline only stores YOLO plant-positive pages (plus
a small negative QC sample) for research/thesis purposes, not a full image
mirror, and each book record in books.jsonl carries the manifest's
rights/attribution statement so provenance and license terms travel with
the data. Default --sleep is a little more conservative than the LOC
pipeline's for the same politeness reason.

Environment
-----------
Credentials are read from environment variables (never hardcode them in this
file, since it goes to GitHub). Put them in a local `.env` (gitignored, see
`.env.example`) or set them as RunPod pod environment variables / secrets.

    BODLEIAN_CONTACT_EMAIL="you@example.org"   # optional but requested by Bodleian

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
python bodleian_yolo_pipeline.py \
    --azure-container botany-pages \
    --max-books-per-keyword 100 \
    --output-dir bodleian_yolo_run

Dry run (no Azure credentials needed, still deletes local files):
python bodleian_yolo_pipeline.py --dry-run --max-books-per-keyword 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()  # Loads a local .env if present; no-op otherwise.
except ImportError:
    pass

BODLEIAN_BASE = "https://digital.bodleian.ox.ac.uk"
BODLEIAN_SEARCH_URL = f"{BODLEIAN_BASE}/search/"
SCRIPT_DIR = Path(__file__).resolve().parent

_contact = os.environ.get("BODLEIAN_CONTACT_EMAIL", "").strip()
USER_AGENT = (
    "Phyto-Vision-Bodleian-YOLO-Pipeline/1.0 "
    f"(research; contact: {_contact})"
    if _contact
    else "Phyto-Vision-Bodleian-YOLO-Pipeline/1.0 "
    "(research; set BODLEIAN_CONTACT_EMAIL for a contact address)"
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
DEFAULT_ROWS_PER_PAGE = 100  # Bodleian only accepts 20, 40, or 100.
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 5
DEFAULT_MAX_PAGE_RETRIES = 5


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
            "Search Digital Bodleian for botanical-subject books, triage "
            "every page with a GPU YOLO model, and stream plant-positive "
            "pages to Azure Blob Storage."
        )
    )

    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Override the built-in keyword list.",
    )
    parser.add_argument(
        "--fq",
        nargs="*",
        default=None,
        help="Optional Digital Bodleian filter queries (Lucene syntax), "
        'repeatable, e.g. --fq \'collections:"Scientific Plate Books"\'. '
        "Multiple values are ANDed, matching the search API.",
    )
    parser.add_argument(
        "--max-books-per-keyword",
        type=int,
        default=DEFAULT_MAX_BOOKS_PER_KEYWORD,
        help="Maximum number of Digital Bodleian search results to fetch "
        "per keyword.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Search result page to start from for each keyword.",
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
        default=Path("bodleian_yolo_run"),
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
        default="botany",
        help="Optional blob name prefix, e.g. 'botany/'.",
    )
    parser.add_argument(
        "--negative-sample-rate",
        type=float,
        default=0.0,
        help="Fraction (0-1) of YOLO-negative pages to keep anyway for "
        "quality-control audit, e.g. 0.01 = 1%%. They are uploaded to "
        "--negative-audit-prefix instead of being deleted outright. "
        "Default: 0 (audit disabled).",
    )
    parser.add_argument(
        "--negative-audit-prefix",
        default="negative_audit",
        help="Blob prefix for the negative audit sample. Default: negative_audit",
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
        help="HTTP timeout in seconds for Digital Bodleian/IIIF requests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Delay after each successful Bodleian/image request. Kept "
        "higher than a courtesy-crawl minimum since Digital Bodleian asks "
        "large harvests to be coordinated with them directly.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Maximum HTTP retries per request (search, manifest, and "
        "image downloads all use the same exponential backoff).",
    )
    parser.add_argument(
        "--max-page-retries",
        type=int,
        default=DEFAULT_MAX_PAGE_RETRIES,
        help="Maximum times a single page is retried across separate runs "
        "before its book is marked failed_permanent and skipped like a "
        f"completed book. Default: {DEFAULT_MAX_PAGE_RETRIES}",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------
# Digital Bodleian / IIIF helpers
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


def strip_html(value: str) -> str:
    """Manifest metadata values are sometimes HTML fragments (e.g. an <a>
    link inside "Homepage"/"Catalogue Description"); collapse to plain text.
    """
    text = re.sub(r"<[^>]+>", "", value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_permanent_http_error(exc: Exception) -> bool:
    """
    True for 4xx responses the server used to deliberately refuse this exact
    request (403 Forbidden, 404 Not Found, ...) - retrying the same URL with
    backoff has no realistic chance of succeeding and just burns time.

    429 Too Many Requests is deliberately excluded even though it's a 4xx:
    it means "you're going too fast," which is the definition of transient -
    it gets the full retry/backoff treatment (honoring Retry-After if the
    server sends one), same as 5xx errors, timeouts, and connection errors.
    """
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int):
        return False
    if status_code == 429:
        return False
    return 400 <= status_code < 500


def retry_delay_seconds(exc: Exception, backoff_delay: float) -> float:
    """
    Use the server's Retry-After header when present (common on 429s) since
    it's a more accurate cooldown than a guessed exponential backoff;
    otherwise fall back to the exponential backoff already in progress.
    """
    response = getattr(exc, "response", None)
    retry_after = None
    if response is not None:
        retry_after = response.headers.get("Retry-After")

    if retry_after:
        try:
            return max(backoff_delay, float(retry_after))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime

                target = parsedate_to_datetime(retry_after)
                now = datetime.now(target.tzinfo) if target.tzinfo else datetime.utcnow()
                return max(backoff_delay, (target - now).total_seconds())
            except Exception:
                pass

    return backoff_delay


def get_json_with_retry(
    session: requests.Session,
    url: str,
    timeout: int,
    sleep_seconds: float,
    retries: int,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delay = 2.0

    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if sleep_seconds:
                time.sleep(sleep_seconds)
            return data
        except Exception as exc:
            if is_permanent_http_error(exc) or attempt == retries - 1:
                raise
            time.sleep(retry_delay_seconds(exc, delay))
            delay = min(delay * 2, 30.0)

    raise RuntimeError("Unreachable JSON retry state.")


def search_books_paginated(
    session: requests.Session,
    query: str,
    fq: list[str] | None,
    max_results: int,
    timeout: int,
    sleep_seconds: float,
    retries: int = DEFAULT_RETRIES,
    start_page: int = 1,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    page = start_page

    while len(results) < max_results:
        params: dict[str, Any] = {
            "q": query,
            "page": page,
            "rows": DEFAULT_ROWS_PER_PAGE,
        }
        if fq:
            params["fq"] = fq

        data: dict[str, Any] | None = None
        delay = 2.0

        for attempt in range(retries):
            try:
                response = session.get(
                    BODLEIAN_SEARCH_URL, params=params, timeout=timeout
                )
                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:
                if is_permanent_http_error(exc) or attempt == retries - 1:
                    print(
                        f"[warn] search failed for query={query!r} page={page}: {exc}"
                    )
                    break
                time.sleep(retry_delay_seconds(exc, delay))
                delay = min(delay * 2, 30.0)

        if data is None:
            break

        page_results = data.get("member", [])
        if not isinstance(page_results, list) or not page_results:
            break

        results.extend(page_results)

        if sleep_seconds:
            time.sleep(sleep_seconds)

        total_pages = (data.get("view") or {}).get("totalPages")
        page += 1
        if isinstance(total_pages, int) and page > total_pages:
            break

    return results[:max_results]


def extract_item_id(result: dict[str, Any]) -> str | None:
    value = result.get("id")
    if not isinstance(value, str):
        return None

    match = re.search(r"/objects/([^/?#]+)/?", value)
    if match:
        return match.group(1)

    return safe_id(value)


def manifest_url_from_result(result: dict[str, Any]) -> str | None:
    manifest = result.get("manifest")
    if isinstance(manifest, dict):
        value = manifest.get("id") or manifest.get("@id")
        if isinstance(value, str):
            return value
    if isinstance(manifest, str):
        return manifest
    return None


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
            "iiif" in body_id.lower() or "image" in body_id.lower()
        ):
            if "/full/" in body_id:
                return body_id.split("/full/")[0]
            return body_id.rstrip("/")

    return None


def parse_iiif_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """
    IIIF-version-agnostic canvas/page walker (Presentation 2 and 3), ported
    from loc_yolo_pipeline.py. Digital Bodleian manifests observed in the
    wild are Presentation 2 (top-level "sequences"), but this also handles
    Presentation 3 ("items") in case a collection has migrated.
    """
    pages: list[dict[str, Any]] = []

    # IIIF Presentation 3
    if isinstance(manifest.get("items"), list) and not manifest.get("sequences"):
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


# Manifest metadata "label" values map loosely onto bibliographic fields;
# Digital Bodleian's own label vocabulary varies across collections
# (a medieval MS says "Attributed Artist", a printed book might say
# "Author"), so each field checks several plausible labels in order.
METADATA_LABELS = {
    "title": ["Title"],
    "author": [
        "Author",
        "Attributed Artist",
        "Artist",
        "Creator",
        "Compiler",
        "Scribe",
        "Illuminated by",
    ],
    "date": ["Date Statement", "Date"],
    "subjects": ["Subject", "Subjects"],
    "language": ["Language"],
    "shelfmark": ["Shelfmark"],
    "collections": ["Collection"],
}


def extract_book_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """
    Best-effort bibliographic metadata extraction from a manifest's
    top-level "metadata" list of {"label": ..., "value": ...} pairs. Values
    are sometimes HTML fragments and labels can repeat (e.g. multiple
    "Collection" entries), so every label is collected into a multimap
    first and looked up by a list of plausible names, left as None/[] when
    absent rather than raising.
    """
    by_label: dict[str, list[str]] = {}
    for entry in manifest.get("metadata", []) or []:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        value = entry.get("value")
        if not isinstance(label, str):
            continue

        values: list[str] = []
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = [v for v in value if isinstance(v, str)]

        cleaned = [strip_html(v) for v in values if strip_html(v)]
        if cleaned:
            by_label.setdefault(label, []).extend(cleaned)

    def first(*keys: str) -> str | None:
        for key in keys:
            values = by_label.get(key)
            if values:
                return values[0]
        return None

    def all_values(*keys: str) -> list[str]:
        found: list[str] = []
        for key in keys:
            found.extend(by_label.get(key, []))
        return found

    return {
        "title": first(*METADATA_LABELS["title"]),
        "author": ", ".join(all_values(*METADATA_LABELS["author"])) or None,
        "date": first(*METADATA_LABELS["date"]),
        "subjects": all_values(*METADATA_LABELS["subjects"]),
        "language": ", ".join(all_values(*METADATA_LABELS["language"])) or None,
        "shelfmark": first(*METADATA_LABELS["shelfmark"]),
        "collections": all_values(*METADATA_LABELS["collections"]),
        "rights_statement": strip_html(
            str(manifest.get("attribution") or manifest.get("rights") or "")
        )
        or None,
    }


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
        except Exception as exc:
            if is_permanent_http_error(exc) or attempt == retries - 1:
                raise
            time.sleep(retry_delay_seconds(exc, delay))
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
) -> tuple[bool, list[dict[str, Any]], float, int]:
    """
    Returns (has_plant, detected, plant_max_confidence, plant_detection_count).

    `detected` includes every detection (not just plant classes) with its
    bounding box, since the GPU cost of computing it is already paid and it
    is useful later for cropping, false-positive audits, and thesis figures.
    `plant_max_confidence`/`plant_detection_count` are scoped to
    plant-related classes only, since a high-confidence non-plant detection
    (e.g. "human") shouldn't count toward a page/book's plant signal.
    """
    results = model.predict(
        source=str(image_path),
        device=device,
        imgsz=imgsz,
        conf=conf_threshold,
        verbose=False,
    )

    detected: list[dict[str, Any]] = []
    plant_max_confidence = 0.0
    plant_detection_count = 0
    has_plant = False
    keywords_lower = [kw.lower() for kw in plant_class_keywords]

    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for cls_idx, confidence, bbox in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            class_name = (
                names.get(int(cls_idx), str(int(cls_idx)))
                if isinstance(names, dict)
                else str(names[int(cls_idx)])
            )

            confidence = float(confidence)
            is_plant = any(
                keyword in class_name.lower() for keyword in keywords_lower
            )

            detected.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": [round(float(v), 2) for v in bbox],
                }
            )

            if is_plant:
                has_plant = True
                plant_detection_count += 1
                plant_max_confidence = max(plant_max_confidence, confidence)

    return has_plant, detected, plant_max_confidence, plant_detection_count


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


def load_books(path: Path) -> dict[str, dict[str, Any]]:
    books: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return books

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            item_id = record.get("bodleian_item_id")
            if item_id:
                books[item_id] = record

    return books


def save_books_atomic(path: Path, books: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in books.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)


# --------------------------------------------------------------------------
# Per-page / per-book pipeline
# --------------------------------------------------------------------------


def build_blob_name(prefix: str, item_id: str, page_index: int) -> str:
    prefix = prefix.strip("/")
    tail = f"{safe_id(item_id)}/page_{page_index + 1:05d}.jpg"
    return f"{prefix}/{tail}" if prefix else tail


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
        "bodleian_item_id": item_id,
        "keyword": keyword,
        "page_index": page_index,
        "page_number": page_index + 1,
        "image_url": page.get("image_url"),
        "action": None,
        "blob_name": None,
        "detected_classes": [],
        "plant_detection_count": 0,
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
        has_plant, detected, plant_max_confidence, plant_detection_count = (
            detect_plant(
                model=model,
                image_path=tmp_path,
                device=device,
                imgsz=args.imgsz,
                conf_threshold=args.conf_threshold,
                plant_class_keywords=args.plant_class_keywords,
            )
        )
        record["detected_classes"] = detected
        record["plant_detection_count"] = plant_detection_count
        record["max_confidence"] = plant_max_confidence if has_plant else None

        # 3a. Plant detected -> upload, then delete local copy.
        if has_plant:
            blob_name = build_blob_name(args.azure_prefix, item_id, page_index)

            if not args.dry_run:
                upload_image_to_blob(container_client, blob_name, tmp_path)
                record["action"] = "kept_uploaded"
            else:
                record["action"] = "kept_dry_run"

            record["blob_name"] = blob_name

        # 3b. No plant -> normally delete local copy, nothing uploaded,
        # except for a small random slice kept as a negative QC audit
        # sample (uploaded, not just left on disk, so review doesn't
        # require re-running this whole pipeline).
        elif args.negative_sample_rate > 0 and random.random() < args.negative_sample_rate:
            blob_name = build_blob_name(
                args.negative_audit_prefix, item_id, page_index
            )

            if not args.dry_run:
                upload_image_to_blob(container_client, blob_name, tmp_path)
                record["action"] = "deleted_no_plant_audited"
            else:
                record["action"] = "deleted_no_plant_audited_dry_run"

            record["blob_name"] = blob_name

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
    books: dict[str, dict[str, Any]],
    log_path: Path,
    keyword: str,
) -> None:
    item_id = extract_item_id(result)
    if not item_id:
        print("[skip] could not determine Bodleian object id from search result")
        return

    item_state = state.get(item_id)
    terminal_statuses = ("completed", "failed_permanent")

    if item_state and item_state.get("status") in terminal_statuses:
        keywords_matched = item_state.setdefault("keywords_matched", [])
        if keyword not in keywords_matched:
            keywords_matched.append(keyword)
            save_state_atomic(args.state_path, state)
            book_record = books.get(item_id)
            if book_record is not None:
                book_record["keywords_matched"] = keywords_matched
                save_books_atomic(args.books_path, books)
        print(f"[skip] {item_id} already {item_state.get('status')}")
        return

    print(f"\n[book] {item_id} (keyword={keyword!r})")

    metadata: dict[str, Any] = {}
    manifest_url: str | None = manifest_url_from_result(result)
    total_pages = item_state.get("page_count", 0) if item_state else 0

    def flush_book_record(status: str) -> None:
        current = state.get(item_id, {})
        pages_kept = current.get("pages_kept", 0)
        books[item_id] = {
            "bodleian_item_id": item_id,
            "bodleian_url": f"{BODLEIAN_BASE}/objects/{item_id}/",
            "manifest_url": manifest_url,
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "date": metadata.get("date"),
            "subjects": metadata.get("subjects", []),
            "language": metadata.get("language"),
            "shelfmark": metadata.get("shelfmark"),
            "collections": metadata.get("collections", []),
            "rights_statement": metadata.get("rights_statement"),
            "status": status,
            "total_pages": total_pages,
            "positive_pages": pages_kept,
            "illustration_density": (
                pages_kept / total_pages if total_pages else None
            ),
            "total_plant_detections": current.get("total_plant_detections", 0),
            "max_confidence": current.get("max_confidence"),
            "pages_negative_audited": current.get("pages_negative_audited", 0),
            "keywords_matched": current.get("keywords_matched", [keyword]),
            "last_updated_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }
        save_books_atomic(args.books_path, books)

    try:
        if not manifest_url:
            raise RuntimeError("Search result had no IIIF manifest URL.")

        manifest_data = get_json_with_retry(
            session, manifest_url, args.timeout, args.sleep, args.retries
        )
        metadata = extract_book_metadata(manifest_data)
        pages = parse_iiif_manifest(manifest_data)

        if not pages:
            print(f"[skip] {item_id}: no page images could be resolved")
            state[item_id] = {
                "status": "failed",
                "error": "no page images found",
                "keywords_matched": [keyword],
            }
            save_state_atomic(args.state_path, state)
            flush_book_record("failed")
            return

        total_pages = len(pages)
        if args.limit_pages_per_book and args.limit_pages_per_book > 0:
            total_pages = min(total_pages, args.limit_pages_per_book)
            pages = pages[:total_pages]

        start_index = item_state.get("next_page_index", 0) if item_state else 0
        pages_kept = item_state.get("pages_kept", 0) if item_state else 0
        pages_deleted = item_state.get("pages_deleted", 0) if item_state else 0
        pages_negative_audited = (
            item_state.get("pages_negative_audited", 0) if item_state else 0
        )
        total_plant_detections = (
            item_state.get("total_plant_detections", 0) if item_state else 0
        )
        book_max_confidence = (
            item_state.get("max_confidence", 0.0) if item_state else 0.0
        )
        failed_pages = list(item_state.get("failed_pages", [])) if item_state else []
        failed_page_retry = (
            dict(item_state.get("failed_page_retry", {})) if item_state else {}
        )
        keywords_matched = (
            list(item_state.get("keywords_matched", [])) if item_state else []
        )
        if keyword not in keywords_matched:
            keywords_matched.append(keyword)

        tmp_dir = args.output_dir / "tmp_page" / safe_id(item_id)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        def save_progress(status: str) -> None:
            state[item_id] = {
                "status": status,
                "next_page_index": next_page_index,
                "page_count": total_pages,
                "pages_kept": pages_kept,
                "pages_deleted": pages_deleted,
                "pages_negative_audited": pages_negative_audited,
                "total_plant_detections": total_plant_detections,
                "max_confidence": book_max_confidence,
                "keywords_matched": keywords_matched,
                "failed_pages": failed_pages,
                "failed_page_retry": failed_page_retry,
            }
            save_state_atomic(args.state_path, state)

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

            # A page-level error must NOT advance next_page_index - the
            # book stops here so the same page is retried (not silently
            # skipped) on the next run.
            if record["action"] == "error":
                retry_count = (
                    failed_page_retry.get("count", 0) + 1
                    if failed_page_retry.get("page_index") == page_index
                    else 1
                )
                failed_page_retry = {"page_index": page_index, "count": retry_count}
                failed_pages.append(
                    {
                        "page_index": page_index,
                        "attempt": retry_count,
                        "error": record["error"],
                        "at_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    }
                )

                next_page_index = page_index  # do not advance past the failure
                if retry_count >= args.max_page_retries:
                    print(
                        f"[error] {item_id}: page {page_index + 1} failed "
                        f"{retry_count} times; giving up on this book "
                        f"(failed_permanent): {record['error']}"
                    )
                    save_progress("failed_permanent")
                else:
                    print(
                        f"[warn] {item_id}: page {page_index + 1} failed "
                        f"(attempt {retry_count}/{args.max_page_retries}); "
                        f"stopping book for later retry: {record['error']}"
                    )
                    save_progress("in_progress")

                flush_book_record(state[item_id]["status"])
                return

            if record["action"] in ("kept_uploaded", "kept_dry_run"):
                pages_kept += 1
                total_plant_detections += record.get("plant_detection_count", 0)
                confidence = record.get("max_confidence")
                if confidence is not None:
                    book_max_confidence = max(book_max_confidence, confidence)
            elif record["action"].startswith("deleted_no_plant"):
                pages_deleted += 1
                if "audited" in record["action"]:
                    pages_negative_audited += 1

            next_page_index = page_index + 1
            save_progress("in_progress")

        state[item_id]["status"] = "completed"
        save_state_atomic(args.state_path, state)
        flush_book_record("completed")

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
        flush_book_record("failed")


def sweep_paused_books(
    *,
    session: requests.Session,
    model,
    container_client,
    args: argparse.Namespace,
    device: str,
    state: dict[str, Any],
    books: dict[str, dict[str, Any]],
    log_path: Path,
) -> None:
    """
    Resume every book currently paused ("in_progress") after a page-level
    error. A book only pauses because download_image_with_retry already
    exhausted its own backoff and still failed - the fix is elapsed
    wall-clock time (transient Bodleian/IIIF-server issues tend to clear
    up), not another immediate retry. Calling this between keyword searches
    (and once more at the end of the run) gives every paused book that gap
    naturally, instead of leaving it stuck until the same item happens to
    resurface under a later keyword's search results.

    A resumed book's manifest URL is reconstructed from its object id
    rather than replayed from the original search result (which isn't kept
    in state), since Digital Bodleian's manifest URL is a deterministic
    function of the object uuid.
    """
    paused_ids = [
        item_id
        for item_id, item_state in state.items()
        if item_state.get("status") == "in_progress"
    ]

    if not paused_ids:
        return

    print(f"\n=== Resuming {len(paused_ids)} paused book(s) ===")

    for item_id in paused_ids:
        keywords_matched = state.get(item_id, {}).get("keywords_matched") or []
        keyword = keywords_matched[-1] if keywords_matched else "resume"
        result = {
            "id": f"{BODLEIAN_BASE}/objects/{item_id}/",
            "manifest": {
                "id": f"https://iiif.bodleian.ox.ac.uk/iiif/manifest/{item_id}.json"
            },
        }

        process_book(
            result=result,
            session=session,
            model=model,
            container_client=container_client,
            args=args,
            device=device,
            state=state,
            books=books,
            log_path=log_path,
            keyword=keyword,
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.state_path = args.output_dir / "processed_items.json"
    args.books_path = args.output_dir / "books.jsonl"
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

    if not _contact:
        print(
            "[warn] BODLEIAN_CONTACT_EMAIL is not set; the User-Agent sent "
            "to Digital Bodleian has no contact address. Their developer "
            "docs ask for one, especially for larger harvests - consider "
            "setting it before a big run."
        )

    state = load_state(args.state_path)
    books = load_books(args.books_path)
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
                fq=args.fq,
                max_results=args.max_books_per_keyword,
                timeout=args.timeout,
                sleep_seconds=args.sleep,
                retries=args.retries,
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
                books=books,
                log_path=log_path,
                keyword=keyword,
            )

        sweep_paused_books(
            session=session,
            model=model,
            container_client=container_client,
            args=args,
            device=device,
            state=state,
            books=books,
            log_path=log_path,
        )

    # One more pass once every keyword has been searched, in case a book
    # paused during the last keyword never got a chance to be swept.
    sweep_paused_books(
        session=session,
        model=model,
        container_client=container_client,
        args=args,
        device=device,
        state=state,
        books=books,
        log_path=log_path,
    )

    completed = sum(1 for v in state.values() if v.get("status") == "completed")
    in_progress = sum(1 for v in state.values() if v.get("status") == "in_progress")
    failed = sum(1 for v in state.values() if v.get("status") == "failed")
    failed_permanent = sum(
        1 for v in state.values() if v.get("status") == "failed_permanent"
    )
    kept_total = sum(v.get("pages_kept", 0) for v in state.values())
    deleted_total = sum(v.get("pages_deleted", 0) for v in state.values())
    audited_total = sum(
        v.get("pages_negative_audited", 0) for v in state.values()
    )
    detections_total = sum(
        v.get("total_plant_detections", 0) for v in state.values()
    )

    print("\n=== Run complete ===")
    print(f"Books completed: {completed}")
    print(f"Books paused for retry (in_progress): {in_progress}")
    print(f"Books failed (will retry next run): {failed}")
    print(f"Books failed_permanent (gave up, page retries exhausted): {failed_permanent}")
    print(f"Pages kept (uploaded to Azure): {kept_total}")
    print(f"Pages deleted (no plant detected): {deleted_total}")
    print(f"Negative pages kept for QC audit: {audited_total}")
    print(f"Total plant detections (boxes, incl. multiple per page): {detections_total}")
    print(f"State file: {args.state_path}")
    print(f"Books metadata: {args.books_path}")
    print(f"Page log: {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
