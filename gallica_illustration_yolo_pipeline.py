#!/usr/bin/env python3
"""
gallica_illustration_yolo_pipeline.py

Large-scale BnF Gallica book-page harvesting with GPU YOLO triage, searching
for pages with RICH ILLUSTRATION. Sibling script to
bodleian_illustration_yolo_pipeline.py - same one-page-at-a-time GPU triage,
Azure upload strategy, and page_layout_best_new.pt model, pointed at BnF
Gallica's SRU search + IIIF APIs instead of Digital Bodleian's.

IMPORTANT - keeping this separate from the Bodleian pipelines' data
--------------------------------------------------------------------------
This is a different library with its own item-id namespace (Gallica ARK
identifiers, e.g. "bpt6k293231"), so there is no risk of a Gallica item
literally overwriting a Bodleian blob by name collision the way the two
Bodleian pipelines could collide with each other. Even so, this script
follows the exact same "<task>/<library>" file management plan laid out in
bodleian_illustration_yolo_pipeline.py, so every library's illustration
search lives in its own clearly labelled corner and nothing has to be
cross-checked by hand later:
  - --azure-prefix defaults to "illustrations/gallica" (siblings:
    "illustrations/bodleian_new", and later "illustrations/loc",
    "illustrations/british_library", ...),
  - --negative-audit-prefix defaults to "illustrations/gallica/negative_audit",
  - --output-dir defaults to "gallica_illustration_yolo_run" (siblings:
    "bodleian_illustration_yolo_run", ...), so local state/temp files never
    collide either,
  - --state-azure-prefix defaults to
    "state_backups/illustration_runs/gallica_illustration_yolo_run",
    alongside the Bodleian pipeline's own
    "state_backups/illustration_runs/bodleian_illustration_yolo_run" under
    the same "illustration_runs/" umbrella.
Because the output directories, Azure prefixes, and local temp/tmux
concerns are all fully separate, this script is safe to run at the same
time as bodleian_illustration_yolo_pipeline.py in a second tmux session on
the same RunPod pod - see "Running both pipelines at once" below.

API reference
-------------
https://api.bnf.fr/fr/api-document-de-gallica
https://api.bnf.fr/fr/api-gallica-de-recherche          (SRU search)
https://api.bnf.fr/fr/api-iiif-de-recuperation-des-images-de-gallica

Gallica SRU search endpoint: GET https://gallica.bnf.fr/SRU
    ?operation=searchRetrieve&version=1.2&query=<CQL>
    &startRecord=<n>&maximumRecords=<0-50>
Response is XML (srw:searchRetrieveResponse), NOT JSON like the Bodleian/LOC
search APIs - see parse_sru_records() below. Each srw:record's
srw:extraRecordData/uri gives the bare ARK identifier (e.g. "bpt6k293231"),
from which both the human-browsable page
(https://gallica.bnf.fr/ark:/12148/<ark>) and the IIIF manifest
(https://gallica.bnf.fr/iiif/ark:/12148/<ark>/manifest.json) are built
deterministically - confirmed against a live query during development.
Query syntax is CQL (Contextual Query Language); the "gallica" index
searches full text + metadata together, and "dc.type" filters by document
type (monographie, manuscrit, carte, image, fascicule, partition, sonore,
objet). This pipeline defaults to (dc.type all "monographie" or dc.type all
"manuscrit") - see --type-filter - to keep results to books/manuscripts
rather than periodicals, maps, or sound recordings.

Gallica's IIIF manifests are IIIF Presentation API 2.x, structurally
identical to Digital Bodleian's (top-level "sequences"/"canvases"), so
parse_iiif_manifest() below is the same walk, just parameterised on image
width (see "IIIF image size and Gallica's rate limit" below). Confirmed
live: a Gallica manifest's own "metadata" is a list of {label, value}
pairs, same shape as Bodleian's, just with a different label vocabulary
(Title/Creator/Date/Language/Shelfmark/Type/Relation observed; no Subject),
so extract_book_metadata() below is a straight port with a different
METADATA_LABELS map, kept as manifest-only (not the richer SRU record) so
metadata extraction behaves identically whether a book is processed fresh
or resumed via sweep_paused_books() - exactly like the Bodleian pipeline.
One consequence: `subjects` in books.jsonl is typically empty for this
pipeline, since Gallica's IIIF manifest doesn't expose dc:subject the way
its SRU search response does.

IIIF image size and Gallica's rate limit
-----------------------------------------
BnF's IIIF Image API documentation states a transitional-phase rate limit
of 5 calls/minute for "full/full" or >1000px requests, with download
bandwidth capped at 832 Ko/s; exceeding it returns 429 Too Many Requests
(confirmed via api.bnf.fr and a third-party client's independent
observation of the same limit). To stay clear of that throttle entirely,
--iiif-width defaults to 1000 (vs. the Bodleian pipeline's 1200) so every
page request is AT the documented threshold, not above it - confirmed live
that Gallica's IIIF Image API (profile: image-api/1.1, quality "native",
also accepts "default") serves /full/1000,/0/default.jpg without issue. If
you raise --iiif-width above 1000, also raise --image-sleep to at least
~12-13s (5/min) to respect the documented limit - the pipeline's generic
429 retry (honours Retry-After) will otherwise absorb the throttle, but
slowly and wastefully. --image-sleep is intentionally separate from
--sleep (used for SRU search + manifest calls, which aren't documented as
rate-limited) so the two can be tuned independently.

Workflow
--------
1. Search Gallica for each keyword in KEYWORDS via SRU (default keywords
   are French - Gallica's catalogue and OCR text are overwhelmingly
   French-language, so English keywords would badly under-match; see the
   English concept -> French keyword table below).
2. For each matching item, fetch its IIIF manifest (URL built
   deterministically from the ARK id) and enumerate every page/canvas.
3. Skip the whole item if it has fewer than --min-pages-per-book pages
   (default 5) - recorded as "skipped_too_short", never rechecked.
4. Otherwise walk every page. If the book has MORE than
   --edge-skip-threshold pages (default 20), the first/last
   --edge-skip-count pages (default 5 each) are logged (IIIF URL + position,
   action "skipped_edge_page") but never downloaded or run through YOLO.
   Every other page is processed ONE PAGE AT A TIME: download -> YOLO on
   GPU immediately (every detection, class+confidence+bbox, is recorded) ->
   illustration-positive pages are uploaded to Azure and the local copy
   deleted; illustration-negative pages are deleted, UNLESS randomly chosen
   for the negative QC audit sample (--negative-sample-rate), in which case
   they're uploaded to --negative-audit-prefix instead. Never more than one
   page image on local disk at a time.
5. Progress is checkpointed after every page to processed_items.json,
   resuming a killed run mid-book. A page-level error pauses the book
   (does NOT advance next_page_index) for retry on the next run, up to
   --max-page-retries attempts, before the book is marked
   "failed_permanent".
6. Every page decision (kept/deleted/skipped/error) is appended to
   page_log.jsonl.
7. Book-level bibliographic metadata, the manifest's attribution/license,
   page counts, illustration-detection totals, and an illustration_density
   ratio are written to books.jsonl, one line per book.
8. processed_items.json/books.jsonl/page_log.jsonl are ALSO pushed to
   Azure under --state-azure-prefix every --state-upload-interval seconds
   (and once more at the end of the run), each upload overwriting the
   previous one - a RunPod pod can die without warning.

English concept -> default French keyword
------------------------------------------
    geometry     -> géométrie          instruments -> instruments
    costume      -> costume            ornament    -> ornement
    heraldry     -> héraldique         medicine    -> médecine
    illuminated  -> enluminure         atlas       -> atlas
    astronomy    -> astronomie         cosmography -> cosmographie
    bestiary     -> bestiaire          book of hours -> livre d'heures
    apocalypse   -> apocalypse
Override with --keywords for a different list (e.g. to search Gallica's
English-language holdings, or add more French terms).

Default YOLO model
-------------------
Defaults to ./page_layout_best_new.pt (the same fine-tuned model as the
current bodleian_illustration_yolo_pipeline.py default), classes:
    illustration, text_block
A page is treated as "contains a rich illustration" when any detected class
name contains "illustration" (see --illustration-class-keywords).

Be a good API citizen
----------------------
No API key is required for Gallica's SRU/IIIF APIs, and BnF's documentation
states open access "except in case of abusive use" - set
GALLICA_CONTACT_EMAIL (see below) so the User-Agent identifies this
research use, and coordinate large harvests with gallica@bnf.fr if running
this at real scale. Non-commercial reuse of Gallica images requires citing
"Source gallica.bnf.fr / Bibliothèque nationale de France" - each book
record in books.jsonl carries the manifest's attribution/license so that
citation travels with the data. Default --sleep/--image-sleep are
conservative for the same reason (see "IIIF image size..." above).

Environment
-----------
Credentials are read from environment variables (never hardcode them in
this file, since it goes to GitHub). Put them in a local `.env` (gitignored)
or set them as RunPod pod environment variables / secrets.

    GALLICA_CONTACT_EMAIL="you@example.org"   # optional but good practice

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
pip install -r requirements.txt
# torch/torchvision should already be present in most RunPod CUDA images;
# otherwise install a CUDA build matching your driver from
# https://pytorch.org/get-started/locally/ (see this project's notes on
# picking a CUDA build that matches `nvidia-smi`'s reported CUDA Version).

Running both pipelines at once
--------------------------------
This script and bodleian_illustration_yolo_pipeline.py write to fully
separate local directories and Azure prefixes (see "IMPORTANT" above), so
it's safe to run them concurrently in two tmux sessions on the same pod:

    tmux new -s bodleian_illustration
    conda activate botany_yolo   # or whatever env has torch/ultralytics
    python bodleian_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux new -s gallica_illustration
    conda activate botany_yolo
    python gallica_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux attach -t bodleian_illustration   # reattach to check on either
    tmux attach -t gallica_illustration
    tmux ls                                 # list all sessions

Both scripts default --device to the same auto-detected cuda:0, so on a
single-GPU pod they will share that one GPU's compute/VRAM. That's fine for
a pod like an RTX A4000 (16GB) running two small YOLO models concurrently -
they'll just interleave GPU time rather than truly run in parallel - but if
you have a multi-GPU pod, pass --device cuda:1 to one of the two scripts to
give each its own GPU.

Example
-------
python gallica_illustration_yolo_pipeline.py \
    --azure-container botany-pages \
    --max-books-per-keyword 100 \
    --output-dir gallica_illustration_yolo_run

Dry run (no Azure credentials needed, still deletes local files):
python gallica_illustration_yolo_pipeline.py --dry-run --max-books-per-keyword 2
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
import xml.etree.ElementTree as ET
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

GALLICA_BASE = "https://gallica.bnf.fr"
GALLICA_SRU_URL = f"{GALLICA_BASE}/SRU"
SCRIPT_DIR = Path(__file__).resolve().parent

SRU_NS = {
    "srw": "http://www.loc.gov/zing/srw/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

_contact = os.environ.get("GALLICA_CONTACT_EMAIL", "").strip()
USER_AGENT = (
    "Phyto-Vision-Gallica-Illustration-YOLO-Pipeline/1.0 "
    f"(research; contact: {_contact})"
    if _contact
    else "Phyto-Vision-Gallica-Illustration-YOLO-Pipeline/1.0 "
    "(research; set GALLICA_CONTACT_EMAIL for a contact address)"
)

# English concept -> default French keyword, see docstring table above.
DEFAULT_KEYWORDS = [
    "géométrie",
    "instruments",
    "costume",
    "ornement",
    "héraldique",
    "médecine",
    "enluminure",
    "atlas",
    "astronomie",
    "cosmographie",
    "bestiaire",
    "livre d'heures",
    "apocalypse",
]

# Restrict results to books/manuscripts by default, not periodicals, maps,
# sound recordings, etc. Pass --type-filter with no values to disable.
DEFAULT_TYPE_FILTER = ["monographie", "manuscrit"]

DEFAULT_YOLO_MODEL = str(SCRIPT_DIR / "page_layout_best_new.pt")
DEFAULT_IMGSZ = 640
DEFAULT_CONF_THRESHOLD = 0.30
DEFAULT_ILLUSTRATION_CLASS_KEYWORDS = ["illustration"]

DEFAULT_MAX_BOOKS_PER_KEYWORD = 100
DEFAULT_ROWS_PER_PAGE = 50  # Gallica SRU caps maximumRecords at 50.
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 5
DEFAULT_MAX_PAGE_RETRIES = 5
DEFAULT_MIN_PAGES_PER_BOOK = 5
DEFAULT_EDGE_SKIP_THRESHOLD = 20
DEFAULT_EDGE_SKIP_COUNT = 5
DEFAULT_STATE_AZURE_PREFIX = "state_backups/illustration_runs/gallica_illustration_yolo_run"
DEFAULT_STATE_UPLOAD_INTERVAL = 3000.0
DEFAULT_IIIF_WIDTH = 1000  # see "IIIF image size and Gallica's rate limit" above.


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
            "Search BnF Gallica for illustration-rich books/manuscripts, "
            "triage every page with a GPU YOLO model, and stream "
            "illustration-positive pages to Azure Blob Storage."
        )
    )

    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Override the built-in (French) keyword list.",
    )
    parser.add_argument(
        "--type-filter",
        nargs="*",
        default=DEFAULT_TYPE_FILTER,
        help="Gallica dc.type values results are restricted to (ORed "
        "together), e.g. monographie manuscrit. Pass --type-filter with no "
        f"values to search all document types. Default: {DEFAULT_TYPE_FILTER}",
    )
    parser.add_argument(
        "--max-books-per-keyword",
        type=int,
        default=DEFAULT_MAX_BOOKS_PER_KEYWORD,
        help="Maximum number of Gallica search results to fetch per keyword.",
    )
    parser.add_argument(
        "--start-record",
        type=int,
        default=1,
        help="SRU startRecord position to start from for each keyword "
        "(1-based).",
    )
    parser.add_argument(
        "--limit-pages-per-book",
        type=int,
        default=0,
        help="Cap on pages processed per book. 0 means process every page.",
    )
    parser.add_argument(
        "--min-pages-per-book",
        type=int,
        default=DEFAULT_MIN_PAGES_PER_BOOK,
        help="Items with fewer than this many pages are skipped entirely "
        "(recorded as status 'skipped_too_short', no pages downloaded or "
        f"run through YOLO). Default: {DEFAULT_MIN_PAGES_PER_BOOK}",
    )
    parser.add_argument(
        "--edge-skip-threshold",
        type=int,
        default=DEFAULT_EDGE_SKIP_THRESHOLD,
        help="Books with more than this many pages have their leading/"
        "trailing pages (see --edge-skip-count) skipped rather than "
        f"downloaded and run through YOLO. Default: {DEFAULT_EDGE_SKIP_THRESHOLD}",
    )
    parser.add_argument(
        "--edge-skip-count",
        type=int,
        default=DEFAULT_EDGE_SKIP_COUNT,
        help="Number of pages to skip at the start AND at the end of a book "
        "whose page count exceeds --edge-skip-threshold. Skipped pages "
        "still get an entry in page_log.jsonl (action 'skipped_edge_page') "
        "recording their IIIF image URL, but are never downloaded or run "
        f"through YOLO. Default: {DEFAULT_EDGE_SKIP_COUNT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gallica_illustration_yolo_run"),
        help="Directory for the state file, page log, and the one-page-at-a-"
        "time temporary download. Kept distinct from the Bodleian "
        "pipelines' output dirs so local state never collides.",
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
        "--illustration-class-keywords",
        nargs="*",
        default=DEFAULT_ILLUSTRATION_CLASS_KEYWORDS,
        help="A page is kept if any detected class name contains one of "
        "these substrings (case-insensitive). "
        f"Default: {DEFAULT_ILLUSTRATION_CLASS_KEYWORDS}",
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
        default="illustrations/gallica",
        help="Blob name prefix for kept page images. Defaults to "
        "'illustrations/gallica' (NOT the container root), matching the "
        "'illustrations/<library>' layout the Bodleian illustration "
        "pipeline uses, so libraries never collide even inside the same "
        "container.",
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
        default="illustrations/gallica/negative_audit",
        help="Blob prefix for the negative audit sample. Default: "
        "illustrations/gallica/negative_audit",
    )
    parser.add_argument(
        "--state-azure-prefix",
        default=DEFAULT_STATE_AZURE_PREFIX,
        help="Blob prefix that processed_items.json/books.jsonl/"
        "page_log.jsonl are periodically pushed to (each upload overwrites "
        "the previous one), so a RunPod pod dying mid-run doesn't lose "
        "progress that only exists on local disk. Default: "
        f"{DEFAULT_STATE_AZURE_PREFIX}",
    )
    parser.add_argument(
        "--state-upload-interval",
        type=float,
        default=DEFAULT_STATE_UPLOAD_INTERVAL,
        help="Minimum seconds between automatic state-file backups to "
        "Azure (a backup is also always attempted once at the very end of "
        f"the run). Default: {DEFAULT_STATE_UPLOAD_INTERVAL:.0f}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run detection and delete local files as usual, but skip the "
        "Azure upload (including state-file backups). Useful for testing "
        "without cloud credentials.",
    )

    # HTTP
    parser.add_argument(
        "--iiif-width",
        type=int,
        default=DEFAULT_IIIF_WIDTH,
        help="Requested IIIF image width in pixels (0 = full resolution). "
        "Kept at or below 1000 by default to stay clear of Gallica's "
        "documented 5-calls/minute throttle on >1000px/full requests - see "
        f"--image-sleep. Default: {DEFAULT_IIIF_WIDTH}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds for Gallica SRU/IIIF requests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Delay after each successful SRU search/manifest request. "
        "Kept conservative since Gallica's general API has no documented "
        "rate limit but a third party independently found ~1 request/3s "
        "the threshold before being treated as abusive.",
    )
    parser.add_argument(
        "--image-sleep",
        type=float,
        default=None,
        help="Delay after each successful page image download. Defaults "
        "to the same value as --sleep. Only needs to be raised above that "
        "if --iiif-width is raised above 1000 (see --iiif-width) to "
        "respect Gallica's documented 5-calls/minute image throttle.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Maximum HTTP retries per request (search, manifest, and "
        "image downloads all use the same exponential backoff, honouring "
        "a Retry-After header on 429s).",
    )
    parser.add_argument(
        "--max-page-retries",
        type=int,
        default=DEFAULT_MAX_PAGE_RETRIES,
        help="Maximum times a single page is retried across separate runs "
        "before its book is marked failed_permanent and skipped like a "
        f"completed book. Default: {DEFAULT_MAX_PAGE_RETRIES}",
    )

    args = parser.parse_args()
    if args.image_sleep is None:
        args.image_sleep = args.sleep
    return args


# --------------------------------------------------------------------------
# Gallica SRU / IIIF helpers
# --------------------------------------------------------------------------


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/xml,application/json,image/jpeg,image/png,image/*,*/*;q=0.8",
        }
    )
    return session


def safe_id(value: str) -> str:
    value = value.strip().rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180] or "unknown"


def strip_html(value: str) -> str:
    """Manifest metadata values are sometimes HTML fragments; collapse to
    plain text."""
    text = re.sub(r"<[^>]+>", "", value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_permanent_http_error(exc: Exception) -> bool:
    """
    True for 4xx responses the server used to deliberately refuse this exact
    request (403 Forbidden, 404 Not Found, ...) - retrying the same URL with
    backoff has no realistic chance of succeeding and just burns time.

    429 Too Many Requests is deliberately excluded even though it's a 4xx:
    Gallica's IIIF API documents exactly this status for its rate limit
    (see docstring), and it means "you're going too fast" - the definition
    of transient. It gets the full retry/backoff treatment (honoring
    Retry-After if the server sends one), same as 5xx errors, timeouts, and
    connection errors.
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


def build_query(keyword: str, type_filter: list[str]) -> str:
    """
    CQL query for the Gallica SRU "gallica" index (full text + metadata),
    optionally ANDed with an ORed dc.type clause to restrict document
    types. `all` requires every word in `keyword` to appear (not
    necessarily adjacent) - good enough recall for a multi-word phrase like
    "livre d'heures" without the stricter phrase-adjacency of `adj`.
    """
    base = f'gallica all "{keyword}"'
    if not type_filter:
        return base
    type_clause = " or ".join(f'dc.type all "{t}"' for t in type_filter)
    return f"{base} and ({type_clause})"


def parse_sru_records(root: ET.Element) -> list[dict[str, Any]]:
    """
    Extract just enough from an SRU searchRetrieveResponse to drive the
    pipeline: the bare Gallica ARK id and the (deterministic) IIIF manifest
    URL. Book-level bibliographic metadata is deliberately NOT sourced here
    - see the docstring's note on why extract_book_metadata() reads the
    IIIF manifest instead, so fresh and resumed books behave identically.
    """
    results: list[dict[str, Any]] = []

    for rec_el in root.findall("srw:records/srw:record", SRU_NS):
        item_id: str | None = None

        extra_el = rec_el.find("srw:extraRecordData", SRU_NS)
        if extra_el is not None:
            uri_el = extra_el.find("uri")
            if uri_el is not None and uri_el.text:
                item_id = uri_el.text.strip()

        if not item_id:
            dc_el = rec_el.find("srw:recordData/oai_dc:dc", SRU_NS)
            if dc_el is not None:
                for ident_el in dc_el.findall("dc:identifier", SRU_NS):
                    match = re.search(r"ark:/12148/([^/\s]+)", ident_el.text or "")
                    if match:
                        item_id = match.group(1)
                        break

        if not item_id:
            continue

        results.append(
            {
                "item_id": item_id,
                "manifest_url": f"{GALLICA_BASE}/iiif/ark:/12148/{item_id}/manifest.json",
            }
        )

    return results


def search_gallica_paginated(
    session: requests.Session,
    keyword: str,
    type_filter: list[str],
    max_results: int,
    timeout: int,
    sleep_seconds: float,
    retries: int = DEFAULT_RETRIES,
    start_record: int = 1,
) -> list[dict[str, Any]]:
    query = build_query(keyword, type_filter)
    results: list[dict[str, Any]] = []
    record_pos = start_record

    while len(results) < max_results:
        params: dict[str, Any] = {
            "operation": "searchRetrieve",
            "version": "1.2",
            "query": query,
            "startRecord": record_pos,
            "maximumRecords": DEFAULT_ROWS_PER_PAGE,
        }

        root: ET.Element | None = None
        delay = 2.0

        for attempt in range(retries):
            try:
                response = session.get(GALLICA_SRU_URL, params=params, timeout=timeout)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                break
            except Exception as exc:
                if is_permanent_http_error(exc) or attempt == retries - 1:
                    print(
                        f"[warn] search failed for keyword={keyword!r} "
                        f"query={query!r}: {exc}"
                    )
                    break
                time.sleep(retry_delay_seconds(exc, delay))
                delay = min(delay * 2, 30.0)

        if root is None:
            break

        page_results = parse_sru_records(root)
        if not page_results:
            break

        results.extend(page_results)

        if sleep_seconds:
            time.sleep(sleep_seconds)

        num_records_text = root.findtext("srw:numberOfRecords", namespaces=SRU_NS)
        try:
            num_records = int(num_records_text) if num_records_text else None
        except ValueError:
            num_records = None

        record_pos += len(page_results)
        if num_records is not None and record_pos > num_records:
            break

    return results[:max_results]


def extract_item_id(result: dict[str, Any]) -> str | None:
    value = result.get("item_id")
    return value if isinstance(value, str) and value else None


def manifest_url_from_result(result: dict[str, Any]) -> str | None:
    value = result.get("manifest_url")
    return value if isinstance(value, str) and value else None


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


def parse_iiif_manifest(manifest: dict[str, Any], width: int) -> list[dict[str, Any]]:
    """
    IIIF-version-agnostic canvas/page walker (Presentation 2 and 3), ported
    from bodleian_illustration_yolo_pipeline.py. Confirmed live that
    Gallica manifests are Presentation 2 (top-level "sequences"), same
    shape as Digital Bodleian's, so this works unchanged apart from the
    `width` parameter - see "IIIF image size and Gallica's rate limit" in
    the module docstring for why it defaults to 1000 rather than Bodleian's
    1200.
    """
    size_segment = f"{width}," if width and width > 0 else "full"
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
                        "image_url": f"{service_base}/full/{size_segment}/0/default.jpg",
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
                        "image_url": f"{service_base}/full/{size_segment}/0/default.jpg",
                    }
                )

    return pages


# Gallica's IIIF manifest "metadata" label vocabulary, confirmed live
# against a real manifest - different from Digital Bodleian's, and notably
# missing a "Subject" label (see docstring note on `subjects`).
METADATA_LABELS = {
    "title": ["Title"],
    "author": ["Creator", "Contributor"],
    "date": ["Date"],
    "subjects": ["Subject"],
    "language": ["Language"],
    "shelfmark": ["Shelfmark"],
    "document_type": ["Type"],
    "catalog_notice": ["Relation"],
}


def extract_book_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """
    Best-effort bibliographic metadata extraction from a manifest's
    top-level "metadata" list of {"label": ..., "value": ...} pairs, ported
    from bodleian_illustration_yolo_pipeline.py. Gallica manifests observed
    in the wild mostly use plain-string values, but at least one field
    ("Format") uses a list of {"@value": ...} dicts instead of plain
    strings, so that shape is handled too (defensively, for fields beyond
    Format that might do the same).
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
            for v in value:
                if isinstance(v, str):
                    values.append(v)
                elif isinstance(v, dict) and isinstance(v.get("@value"), str):
                    values.append(v["@value"])

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

    attribution = strip_html(str(manifest.get("attribution") or ""))
    license_url = str(manifest.get("license") or "").strip()
    rights_statement = " | ".join(p for p in (attribution, license_url) if p) or None

    return {
        "title": first(*METADATA_LABELS["title"]),
        "author": ", ".join(all_values(*METADATA_LABELS["author"])) or None,
        "date": first(*METADATA_LABELS["date"]),
        "subjects": all_values(*METADATA_LABELS["subjects"]),
        "language": ", ".join(all_values(*METADATA_LABELS["language"])) or None,
        "shelfmark": first(*METADATA_LABELS["shelfmark"]),
        "document_type": all_values(*METADATA_LABELS["document_type"]),
        "catalog_notice": first(*METADATA_LABELS["catalog_notice"]),
        "rights_statement": rights_statement,
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


def detect_illustration(
    model,
    image_path: Path,
    device: str,
    imgsz: int,
    conf_threshold: float,
    illustration_class_keywords: list[str],
) -> tuple[bool, list[dict[str, Any]], float, int]:
    """
    Returns (has_illustration, detected, illustration_max_confidence, illustration_detection_count).

    `detected` includes every detection (not just illustration classes) with
    its bounding box, since the GPU cost of computing it is already paid and
    it is useful later for cropping, false-positive audits, and thesis
    figures. `illustration_max_confidence`/`illustration_detection_count`
    are scoped to illustration-related classes only, since a high-confidence
    "text_block" detection shouldn't count toward a page/book's illustration
    signal.
    """
    results = model.predict(
        source=str(image_path),
        device=device,
        imgsz=imgsz,
        conf=conf_threshold,
        verbose=False,
    )

    detected: list[dict[str, Any]] = []
    illustration_max_confidence = 0.0
    illustration_detection_count = 0
    has_illustration = False
    keywords_lower = [kw.lower() for kw in illustration_class_keywords]

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
            is_illustration = any(
                keyword in class_name.lower() for keyword in keywords_lower
            )

            detected.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": [round(float(v), 2) for v in bbox],
                }
            )

            if is_illustration:
                has_illustration = True
                illustration_detection_count += 1
                illustration_max_confidence = max(illustration_max_confidence, confidence)

    return has_illustration, detected, illustration_max_confidence, illustration_detection_count


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


STATE_FILE_CONTENT_TYPES = {".jsonl": "application/x-ndjson", ".json": "application/json"}


def upload_state_file_to_blob(container_client, blob_name: str, local_path: Path) -> None:
    """
    Push one local state file (processed_items.json/books.jsonl/
    page_log.jsonl) to Azure, overwriting whatever was there before.
    """
    from azure.storage.blob import ContentSettings

    content_type = STATE_FILE_CONTENT_TYPES.get(local_path.suffix, "text/plain")
    with local_path.open("rb") as handle:
        container_client.upload_blob(
            name=blob_name,
            data=handle,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )


def backup_state_to_azure(container_client, args: argparse.Namespace) -> None:
    """
    Upload the current processed_items.json/books.jsonl/page_log.jsonl to
    Azure under --state-azure-prefix, each upload overwriting the file
    already there. A RunPod pod's local disk is ephemeral and the pod can
    die without warning, so this is the safety net that keeps a killed
    run's progress recoverable even if the local --output-dir is lost.
    """
    if container_client is None:
        return

    prefix = args.state_azure_prefix.strip("/")
    for local_path in (args.state_path, args.books_path, args.page_log_path):
        if not local_path.exists():
            continue
        blob_name = f"{prefix}/{local_path.name}"
        try:
            upload_state_file_to_blob(container_client, blob_name, local_path)
        except Exception as exc:
            print(f"[warn] state backup upload failed for {local_path.name}: {exc}")


def maybe_backup_state_to_azure(
    container_client, args: argparse.Namespace, force: bool = False
) -> None:
    """
    Rate-limited wrapper around backup_state_to_azure(), called after every
    book so a long run doesn't spend all its time re-uploading the (growing)
    page_log.jsonl. --state-upload-interval controls the minimum gap between
    uploads; `force=True` (used at the very end of main()) bypasses it for a
    final guaranteed backup.
    """
    if container_client is None or args.dry_run:
        return

    now = time.monotonic()
    if not force and (now - args.last_state_upload_time) < args.state_upload_interval:
        return

    backup_state_to_azure(container_client, args)
    args.last_state_upload_time = now


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
            item_id = record.get("gallica_item_id")
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
        "gallica_item_id": item_id,
        "keyword": keyword,
        "page_index": page_index,
        "page_number": page_index + 1,
        "image_url": page.get("image_url"),
        "action": None,
        "blob_name": None,
        "detected_classes": [],
        "illustration_detection_count": 0,
        "max_confidence": None,
        "error": None,
    }

    try:
        # 1. Download exactly one page image. Uses --image-sleep (not
        # --sleep) since Gallica's IIIF image endpoint has a documented,
        # separately-tuned rate limit - see module docstring.
        download_image_with_retry(
            session=session,
            image_url=page["image_url"],
            output_path=tmp_path,
            timeout=args.timeout,
            sleep_seconds=args.image_sleep,
            retries=args.retries,
        )

        # 2. Run it through YOLO on the GPU immediately.
        has_illustration, detected, illustration_max_confidence, illustration_detection_count = (
            detect_illustration(
                model=model,
                image_path=tmp_path,
                device=device,
                imgsz=args.imgsz,
                conf_threshold=args.conf_threshold,
                illustration_class_keywords=args.illustration_class_keywords,
            )
        )
        record["detected_classes"] = detected
        record["illustration_detection_count"] = illustration_detection_count
        record["max_confidence"] = illustration_max_confidence if has_illustration else None

        # 3a. Illustration detected -> upload, then delete local copy.
        if has_illustration:
            blob_name = build_blob_name(args.azure_prefix, item_id, page_index)

            if not args.dry_run:
                upload_image_to_blob(container_client, blob_name, tmp_path)
                record["action"] = "kept_uploaded"
            else:
                record["action"] = "kept_dry_run"

            record["blob_name"] = blob_name

        # 3b. No illustration -> normally delete local copy, nothing uploaded,
        # except for a small random slice kept as a negative QC audit
        # sample (uploaded, not just left on disk, so review doesn't
        # require re-running this whole pipeline).
        elif args.negative_sample_rate > 0 and random.random() < args.negative_sample_rate:
            blob_name = build_blob_name(
                args.negative_audit_prefix, item_id, page_index
            )

            if not args.dry_run:
                upload_image_to_blob(container_client, blob_name, tmp_path)
                record["action"] = "deleted_no_illustration_audited"
            else:
                record["action"] = "deleted_no_illustration_audited_dry_run"

            record["blob_name"] = blob_name

        else:
            record["action"] = "deleted_no_illustration"

    except Exception as exc:
        record["action"] = "error"
        record["error"] = str(exc)

    finally:
        # Always remove the local copy from the RunPod machine's disk.
        tmp_path.unlink(missing_ok=True)

    append_log(log_path, record)
    return record


def log_skipped_edge_page(
    *,
    item_id: str,
    keyword: str,
    page: dict[str, Any],
    page_index: int,
    log_path: Path,
) -> dict[str, Any]:
    """
    Record a leading/trailing page of a long book WITHOUT downloading it or
    running YOLO - see --edge-skip-threshold/--edge-skip-count. The page's
    IIIF image URL and position are still written to page_log.jsonl so its
    existence is never lost, it's just never classified.
    """
    record: dict[str, Any] = {
        "gallica_item_id": item_id,
        "keyword": keyword,
        "page_index": page_index,
        "page_number": page_index + 1,
        "image_url": page.get("image_url"),
        "action": "skipped_edge_page",
        "blob_name": None,
        "detected_classes": [],
        "illustration_detection_count": 0,
        "max_confidence": None,
        "error": None,
    }
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
        print("[skip] could not determine Gallica ARK id from search result")
        return

    item_state = state.get(item_id)
    terminal_statuses = ("completed", "failed_permanent", "skipped_too_short")

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
            "gallica_item_id": item_id,
            "gallica_url": f"{GALLICA_BASE}/ark:/12148/{item_id}",
            "manifest_url": manifest_url,
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "date": metadata.get("date"),
            "subjects": metadata.get("subjects", []),
            "language": metadata.get("language"),
            "shelfmark": metadata.get("shelfmark"),
            "document_type": metadata.get("document_type", []),
            "catalog_notice": metadata.get("catalog_notice"),
            "rights_statement": metadata.get("rights_statement"),
            "status": status,
            "total_pages": total_pages,
            "positive_pages": pages_kept,
            "illustration_density": (
                pages_kept / total_pages if total_pages else None
            ),
            "total_illustration_detections": current.get("total_illustration_detections", 0),
            "max_confidence": current.get("max_confidence"),
            "pages_negative_audited": current.get("pages_negative_audited", 0),
            "pages_skipped_edge": current.get("pages_skipped_edge", 0),
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
        pages = parse_iiif_manifest(manifest_data, width=args.iiif_width)

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

        raw_total_pages = len(pages)

        # Rule: skip items with fewer than --min-pages-per-book pages
        # entirely - too short to be worth any GPU/bandwidth time. Nothing
        # is downloaded; just record it as a terminal state so it isn't
        # re-checked the next time a keyword happens to match it again.
        if raw_total_pages < args.min_pages_per_book:
            print(
                f"[skip] {item_id}: only {raw_total_pages} page(s) "
                f"(< --min-pages-per-book={args.min_pages_per_book}); "
                "skipping item"
            )
            state[item_id] = {
                "status": "skipped_too_short",
                "page_count": raw_total_pages,
                "keywords_matched": [keyword],
            }
            save_state_atomic(args.state_path, state)
            total_pages = raw_total_pages
            flush_book_record("skipped_too_short")
            return

        total_pages = raw_total_pages
        if args.limit_pages_per_book and args.limit_pages_per_book > 0:
            total_pages = min(total_pages, args.limit_pages_per_book)
            pages = pages[:total_pages]

        # Rule: for books with more than --edge-skip-threshold pages, don't
        # spend download/GPU time on the leading/trailing
        # --edge-skip-count pages (title pages, flyleaves, colophons are
        # reliably low-yield for illustration content). Their IIIF URL is
        # still logged (see log_skipped_edge_page) so the page isn't lost,
        # just never fetched or classified. Computed against the book's
        # real length (raw_total_pages), not a --limit-pages-per-book cap.
        edge_skip_indices: set[int] = set()
        if raw_total_pages > args.edge_skip_threshold and args.edge_skip_count > 0:
            edge_skip_indices = set(range(0, args.edge_skip_count)) | set(
                range(raw_total_pages - args.edge_skip_count, raw_total_pages)
            )

        start_index = item_state.get("next_page_index", 0) if item_state else 0
        pages_kept = item_state.get("pages_kept", 0) if item_state else 0
        pages_deleted = item_state.get("pages_deleted", 0) if item_state else 0
        pages_negative_audited = (
            item_state.get("pages_negative_audited", 0) if item_state else 0
        )
        pages_skipped_edge = (
            item_state.get("pages_skipped_edge", 0) if item_state else 0
        )
        total_illustration_detections = (
            item_state.get("total_illustration_detections", 0) if item_state else 0
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
                "pages_skipped_edge": pages_skipped_edge,
                "total_illustration_detections": total_illustration_detections,
                "max_confidence": book_max_confidence,
                "keywords_matched": keywords_matched,
                "failed_pages": failed_pages,
                "failed_page_retry": failed_page_retry,
            }
            save_state_atomic(args.state_path, state)

        for page_index in range(start_index, total_pages):
            if page_index in edge_skip_indices:
                record = log_skipped_edge_page(
                    item_id=item_id,
                    keyword=keyword,
                    page=pages[page_index],
                    page_index=page_index,
                    log_path=log_path,
                )
            else:
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
                total_illustration_detections += record.get("illustration_detection_count", 0)
                confidence = record.get("max_confidence")
                if confidence is not None:
                    book_max_confidence = max(book_max_confidence, confidence)
            elif record["action"].startswith("deleted_no_illustration"):
                pages_deleted += 1
                if "audited" in record["action"]:
                    pages_negative_audited += 1
            elif record["action"] == "skipped_edge_page":
                pages_skipped_edge += 1

            next_page_index = page_index + 1
            save_progress("in_progress")

        state[item_id]["status"] = "completed"
        save_state_atomic(args.state_path, state)
        flush_book_record("completed")

        shutil.rmtree(tmp_dir, ignore_errors=True)

        print(
            f"[done] {item_id}: kept={pages_kept} "
            f"deleted={pages_deleted} skipped_edge={pages_skipped_edge} "
            f"total={total_pages}"
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
    wall-clock time (transient Gallica server issues, or a 429 cooldown,
    tend to clear up), not another immediate retry. Calling this between
    keyword searches (and once more at the end of the run) gives every
    paused book that gap naturally, instead of leaving it stuck until the
    same item happens to resurface under a later keyword's search results.

    A resumed book's manifest URL is reconstructed from its ARK id rather
    than replayed from the original search result (which isn't kept in
    state), since Gallica's manifest URL is a deterministic function of the
    ARK id.
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
            "item_id": item_id,
            "manifest_url": f"{GALLICA_BASE}/iiif/ark:/12148/{item_id}/manifest.json",
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
    args.page_log_path = log_path
    args.last_state_upload_time = 0.0

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
            "[dry-run] Azure upload disabled; illustration-positive pages "
            "are detected but not uploaded (still deleted locally). State "
            "file backups to Azure are also disabled in dry-run mode."
        )

    if not _contact:
        print(
            "[warn] GALLICA_CONTACT_EMAIL is not set; the User-Agent sent "
            "to Gallica has no contact address. Good practice for a large "
            "harvest - consider setting it, and coordinate with "
            "gallica@bnf.fr for real scale."
        )

    state = load_state(args.state_path)
    books = load_books(args.books_path)
    session = make_session()

    keywords = args.keywords or DEFAULT_KEYWORDS
    print(f"Keywords ({len(keywords)}): {', '.join(keywords)}")
    print(f"Type filter: {args.type_filter or '(none - all document types)'}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Illustration class keywords: {args.illustration_class_keywords}")
    print(f"Output dir: {args.output_dir}")
    print(f"IIIF image width: {args.iiif_width or 'full resolution'}")
    print(f"Sleep: {args.sleep}s (search/manifest)  {args.image_sleep}s (images)")
    print(f"Azure image prefix: {args.azure_prefix}")
    print(
        f"Azure state backup prefix: {args.state_azure_prefix} "
        f"(every {args.state_upload_interval:.0f}s)"
    )

    for keyword in keywords:
        print(f"\n=== Keyword: {keyword!r} ===")
        try:
            search_results = search_gallica_paginated(
                session=session,
                keyword=keyword,
                type_filter=args.type_filter,
                max_results=args.max_books_per_keyword,
                timeout=args.timeout,
                sleep_seconds=args.sleep,
                retries=args.retries,
                start_record=args.start_record,
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
            # RunPod pods can die without warning, so push the current
            # state/books/page-log files to Azure every so often (rate
            # limited by --state-upload-interval), overwriting the previous
            # backup each time.
            maybe_backup_state_to_azure(container_client, args)

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
        maybe_backup_state_to_azure(container_client, args)

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

    # Final backup, unconditional, so the run's last state is always in
    # Azure even if --state-upload-interval hasn't elapsed yet.
    maybe_backup_state_to_azure(container_client, args, force=True)

    completed = sum(1 for v in state.values() if v.get("status") == "completed")
    in_progress = sum(1 for v in state.values() if v.get("status") == "in_progress")
    failed = sum(1 for v in state.values() if v.get("status") == "failed")
    failed_permanent = sum(
        1 for v in state.values() if v.get("status") == "failed_permanent"
    )
    skipped_too_short = sum(
        1 for v in state.values() if v.get("status") == "skipped_too_short"
    )
    kept_total = sum(v.get("pages_kept", 0) for v in state.values())
    deleted_total = sum(v.get("pages_deleted", 0) for v in state.values())
    audited_total = sum(
        v.get("pages_negative_audited", 0) for v in state.values()
    )
    skipped_edge_total = sum(
        v.get("pages_skipped_edge", 0) for v in state.values()
    )
    detections_total = sum(
        v.get("total_illustration_detections", 0) for v in state.values()
    )

    print("\n=== Run complete ===")
    print(f"Books completed: {completed}")
    print(f"Books paused for retry (in_progress): {in_progress}")
    print(f"Books failed (will retry next run): {failed}")
    print(f"Books failed_permanent (gave up, page retries exhausted): {failed_permanent}")
    print(f"Books skipped_too_short (< --min-pages-per-book): {skipped_too_short}")
    print(f"Pages kept (uploaded to Azure): {kept_total}")
    print(f"Pages deleted (no illustration detected): {deleted_total}")
    print(f"Negative pages kept for QC audit: {audited_total}")
    print(f"Pages skipped (edge pages, never downloaded/classified): {skipped_edge_total}")
    print(f"Total illustration detections (boxes, incl. multiple per page): {detections_total}")
    print(f"State file: {args.state_path}")
    print(f"Books metadata: {args.books_path}")
    print(f"Page log: {log_path}")
    print(f"Azure state backup: {args.state_azure_prefix} (skipped in dry-run mode)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
