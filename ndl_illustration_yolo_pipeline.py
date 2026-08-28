#!/usr/bin/env python3
"""
ndl_illustration_yolo_pipeline.py

Large-scale National Diet Library (Japan) Digital Collections book-page
harvesting with GPU YOLO triage, searching for pages with RICH
ILLUSTRATION. Sibling script to bodleian_illustration_yolo_pipeline.py,
gallica_illustration_yolo_pipeline.py, mdz_illustration_yolo_pipeline.py,
and wellcome_illustration_yolo_pipeline.py - same one-page-at-a-time GPU
triage, Azure upload strategy, and page_layout_best_new.pt model, pointed
at NDL Search's SRU API + NDL Digital Collections' IIIF API instead.

IMPORTANT - keeping this separate from the other pipelines' data
--------------------------------------------------------------------------
NDL Digital Collections items are identified by a purely numeric
Persistent ID (e.g. "2536504", from info:ndljp/pid/2536504) - a distinct
namespace from every other library this project searches, so there's no
risk of a literal blob-name collision. Even so, this script follows the
exact same "<task>/<library>" file management plan laid out in
bodleian_illustration_yolo_pipeline.py, so every library's illustration
search lives in its own clearly labelled corner and nothing has to be
cross-checked by hand later:
  - --azure-prefix defaults to "illustrations/ndl" (siblings:
    "illustrations/bodleian_new", "illustrations/gallica",
    "illustrations/mdz", "illustrations/wellcome"),
  - --negative-audit-prefix defaults to "illustrations/ndl/negative_audit",
  - --output-dir defaults to "ndl_illustration_yolo_run" (siblings:
    "bodleian_illustration_yolo_run", "gallica_illustration_yolo_run",
    "mdz_illustration_yolo_run", "wellcome_illustration_yolo_run"), so
    local state/temp files never collide either,
  - --state-azure-prefix defaults to
    "state_backups/illustration_runs/ndl_illustration_yolo_run", alongside
    the other pipelines' own prefixes under the same "illustration_runs/"
    umbrella.
Because the output directories, Azure prefixes, and local temp/tmux
concerns are all fully separate, this script is safe to run at the same
time as the other seven illustration pipelines in their own tmux sessions
on the same RunPod pod - see "Running all eight illustration pipelines at
once" below.

API reference
-------------
https://ndlsearch.ndl.go.jp/en/help/api/specifications
https://ndlsearch.ndl.go.jp/file/help/api/specifications/ndlsearch_api_20240105.pdf
    (the full "External Interface Specification" PDF linked from the page
    above - the English help page itself is a client-rendered SPA with
    very little of the actual parameter reference in static HTML, so this
    PDF was the primary source for the details below)
https://ndlsearch.ndl.go.jp/en/news/renkei_20240105
https://dl.ndl.go.jp/static/files/IIIF_interface_En.pdf

Search endpoint (SRU, documented): GET https://ndlsearch.ndl.go.jp/api/sru
    ?operation=searchRetrieve&version=1.2&recordSchema=dcndl
    &recordPacking=xml&query=<CQL>&startRecord=<1-based>
    &maximumRecords=<n>
Response is XML (SRW searchRetrieveResponse). recordSchema=dcndl gives each
matching bibliographic record as DC-NDL RDF/XML (dcndl:BibResource +
dcndl:Item elements), confirmed live during development. A bibliographic
record can bundle several dcndl:Item elements - e.g. one representing the
physical holding at the National Diet Library itself (no online image) and
another representing the SAME work's digitised copy in NDL Digital
Collections (with an rdfs:seeAlso pointing at
https://dl.ndl.go.jp/pid/<numeric>) - or even a copy digitised by a
different, partner institution entirely (a seeAlso host other than
dl.ndl.go.jp, which this pipeline can't do anything with since it isn't
served by NDL's own IIIF API). parse_sru_records() below walks every
dcndl:Item in every record, keeps only the ones with a dl.ndl.go.jp pid
seeAlso, and de-duplicates by that pid - each pid becomes one independent
"book" to process (a multi-volume work naturally yields one pid, and
therefore one book entry, per volume). Because pid->manifest URL is
deterministic (see below), and pid is the only thing this pipeline actually
needs from the search result, no other SRU/dcndl field is parsed - exactly
like the other four pipelines, book-level metadata is read from the IIIF
manifest only (see extract_book_metadata below), never from the search
result, so metadata extraction behaves identically whether a book is
processed fresh or resumed via sweep_paused_books().

mediatype filter and a CQL limitation (confirmed live)
---------------------------------------------------------
Per user instruction, --mediatype-filter defaults to oldmaterials, books,
and manuscripts (NDL's own mediaType vocabulary - "oldmaterials" covers
NDL's famous pre-modern/rare-book collection, the most illustration-dense
by far). The `mediatype` CQL field only supports the "=" operator (no
front/partial match, no "any"/"all" shorthand - confirmed against the
spec's own "SRU field conditions" table). Naively combining a keyword with
an OR-group of mediatype values, e.g.
`anywhere="X" and (mediatype=a or mediatype=b)`, looked like the obvious
CQL - but confirmed LIVE during development that this NDL SRU endpoint
supports NEITHER parenthesised grouping NOR mixing "and" and "or" in the
same query at all (both return "illegal query syntax"; pure "and"-only or
pure "or"-only queries both work fine on their own). There is therefore no
single-request way to ask for "keyword AND (mediatype is one of these
three)". This pipeline works around it by issuing one pure-"and" SRU query
PER (keyword, mediatype value) COMBINATION - e.g. three separate queries
for the three default mediatype values - and merging/de-duplicating the
resulting pids client-side (see search_ndl_paginated()). Pass
--mediatype-filter with no values to search without any mediatype
restriction (a single query per keyword, no "and mediatype=..." clause).

Also confirmed live: maximumRecords is capped at 500, and - regardless of
startRecord/maximumRecords - position 501 onward is simply unreachable
("501件目以降を取得することはできない"). search_ndl_paginated() stops
pagination at that ceiling per (keyword, mediatype) combination.

IIIF manifest URL is deterministic from the pid:
    https://www.dl.ndl.go.jp/api/iiif/{pid}/manifest.json
Confirmed live that NDL manifests are IIIF Presentation API 2.x (top-level
"sequences"/"canvases", canvas-level width/height present), structurally
identical to the other four pipelines', so parse_iiif_manifest() below is
the same version-agnostic walk ported from them, canvas_width/
canvas_height included. The Image API's declared profile is level1 with
only "regionByPct"/"sizeByWh" in its `supports` list (i.e. sizeByW -
requesting a width alone with height auto-scaled - isn't officially
declared) but confirmed live that a plain "/full/1200,/0/default.jpg"
width-only request works anyway, so --iiif-width behaves exactly like the
other pipelines' equivalent flag. NDL's own documented cap is 5000px on the
longer side, comfortably above every default here.

A manifest's "metadata" list uses plain string label/value pairs (like
Bodleian/Gallica/Wellcome, NOT MDZ's multilingual shape), but MULTIPLE
values for one label are "||"-delimited within a single value string
(confirmed against the official IIIF spec PDF), a different convention
again from Wellcome's "; "-delimited packing - see
split_double_pipe_list() in extract_book_metadata() below. The manifest
also carries an "Access Restrictions" metadata value (e.g. "PDM" for
Public Domain Mark on openly viewable items; other values indicate
narrower access, e.g. library-premises-only) - captured into books.jsonl,
but not used to pre-filter search results; an inaccessible item's image
requests will simply 403 and flow through the existing page-error/retry/
failed_permanent handling like any other permanent HTTP error, the same
graceful-degradation posture used throughout this pipeline family.

Default YOLO model
-------------------
Defaults to ./page_layout_best_new.pt (the same fine-tuned model as the
other four illustration pipelines' default), classes:
    illustration, text_block
A page is treated as "contains a rich illustration" when any detected class
name contains "illustration" (see --illustration-class-keywords).

Workflow
--------
1. Search NDL for each keyword in KEYWORDS (Japanese/Sino-Japanese terms;
   see module-level DEFAULT_KEYWORDS), once per --mediatype-filter value
   (see "mediatype filter and a CQL limitation" above), merging/
   de-duplicating pids across those sub-queries.
2. For each pid, fetch its IIIF manifest (URL built deterministically) and
   enumerate every page/canvas.
3. Skip the whole item if it has fewer than --min-pages-per-book pages
   (default 5) - recorded as "skipped_too_short", never rechecked.
4. Otherwise walk every page. If the book has MORE than
   --edge-skip-threshold pages (default 20), the first/last
   --edge-skip-count pages (default 5 each) are logged (IIIF URL + position,
   action "skipped_edge_page") but never downloaded or run through YOLO.
   Every other page is processed ONE PAGE AT A TIME: download -> YOLO on
   GPU immediately (every detection, class+confidence+bbox - both pixel and
   image-normalized, plus its share of the page area - is recorded) ->
   illustration-positive pages are uploaded to Azure and the local copy
   deleted; illustration-negative pages are deleted, UNLESS a SHA-256
   deterministic sampling decision selects them for the negative QC audit
   sample (--negative-sample-rate), in which case they're uploaded to
   --negative-audit-prefix instead. Never more than one page image on local
   disk at a time.
5. Progress is checkpointed after every page to processed_items.json,
   resuming a killed run mid-book. A page-level error pauses the book
   (does NOT advance next_page_index) for retry on the next run, up to
   --max-page-retries attempts, before the book is marked
   "failed_permanent".
6. Every page decision (kept/deleted/skipped/error) is appended to
   page_log.jsonl with full provenance: generic source/source_item_id
   fields alongside the library-specific field kept for backward
   compatibility, the manifest URL, canvas id/label, page index/number, the
   IIIF image service URL, the exact requested image URL, the downloaded
   image's actual pixel dimensions (read from YOLO's own decode - never
   assumed to equal the requested IIIF width), the source canvas's
   width/height from the manifest when available, the blob name when
   uploaded, and - for illustration-negative pages - whether the page was
   selected for the negative audit sample and the deterministic sampling
   value used.
7. Book-level bibliographic metadata, the manifest's attribution/license/
   access-restrictions, page counts, illustration-detection totals, and an
   illustration_density ratio are written to books.jsonl, one line per
   book.
8. processed_items.json/books.jsonl/page_log.jsonl/run_metadata.json are
   ALSO pushed to Azure under --state-azure-prefix every
   --state-upload-interval seconds (and once more at the end of the run),
   each upload overwriting the previous one - a RunPod pod can die without
   warning.
9. run_metadata.json is written once, the first time --output-dir is used,
   and never regenerated on a resumed run (even with different CLI flags) -
   it records which YOLO model (path + SHA-256 of the weights file), class
   names, imgsz, confidence threshold, illustration keywords, and
   Python/PyTorch/Ultralytics/CUDA versions actually produced the corpus in
   that directory. See load_or_create_run_metadata().

Be a good API citizen
----------------------
No API key or registration is required for SRU (registration is only
suggested for continuous/production OpenSearch use, which this pipeline
doesn't use). No numeric rate limit is documented for SRU. Even so, set
NDL_CONTACT_EMAIL (see below) so the User-Agent identifies this research
use, and keep --sleep at a reasonable default - the same posture the other
pipelines' docs ask for. NDL Digital Collections content spans public
domain, various openly-licensed, and access-restricted material; each book
record in books.jsonl carries the manifest's attribution and Access
Restrictions value so that provenance travels with the data.

Environment
-----------
Credentials are read from environment variables (never hardcode them in
this file, since it goes to GitHub). Put them in a local `.env` (gitignored)
or set them as RunPod pod environment variables / secrets.

    NDL_CONTACT_EMAIL="you@example.org"   # optional but good practice

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

Running all eight illustration pipelines at once
----------------------------------------------------
This script and the other seven illustration pipelines write to fully
separate local directories and Azure prefixes (see "IMPORTANT" above), so
it's safe to run all eight concurrently in their own tmux sessions on the
same pod:

    tmux new -s bodleian_illustration
    conda activate botany_yolo   # or whatever env has torch/ultralytics
    python bodleian_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux new -s gallica_illustration
    conda activate botany_yolo
    python gallica_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux new -s mdz_illustration
    conda activate botany_yolo
    python mdz_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux new -s wellcome_illustration
    conda activate botany_yolo
    python wellcome_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux new -s ndl_illustration
    conda activate botany_yolo
    python ndl_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux new -s rmda_illustration
    conda activate botany_yolo
    python rmda_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux new -s loc_illustration
    conda activate botany_yolo
    python loc_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux new -s harvard_yenching_illustration
    conda activate botany_yolo
    python harvard_yenching_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux attach -t bodleian_illustration   # reattach to check on any of them
    tmux attach -t gallica_illustration
    tmux attach -t mdz_illustration
    tmux attach -t wellcome_illustration
    tmux attach -t ndl_illustration
    tmux attach -t rmda_illustration
    tmux attach -t loc_illustration
    tmux attach -t harvard_yenching_illustration
    tmux ls                                 # list all sessions

All eight scripts default --device to the same auto-detected cuda:0, so on
a single-GPU pod they will share that one GPU's compute/VRAM - fine for a
pod like an RTX A4000 (16GB) running eight small YOLO models concurrently
(they'll interleave GPU time rather than truly run in parallel), but if you
have a multi-GPU pod, pass --device cuda:1/cuda:2/cuda:3/cuda:4/cuda:5/cuda:6/cuda:7 to spread
them out.

Example
-------
python ndl_illustration_yolo_pipeline.py \
    --azure-container botany-pages \
    --max-books-per-keyword 100 \
    --output-dir ndl_illustration_yolo_run

Dry run (no Azure credentials needed, still deletes local files):
python ndl_illustration_yolo_pipeline.py --dry-run --max-books-per-keyword 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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

NDL_SRU_URL = "https://ndlsearch.ndl.go.jp/api/sru"
NDL_IIIF_BASE = "https://www.dl.ndl.go.jp/api/iiif"
NDL_VIEWER_BASE = "https://dl.ndl.go.jp"
SCRIPT_DIR = Path(__file__).resolve().parent

SRW_NS = {"srw": "http://www.loc.gov/zing/srw/"}
RDF_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "dcndl": "http://ndl.go.jp/dcndl/terms/",
}
RDF_RESOURCE_ATTR = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
DL_NDL_PID_RE = re.compile(r"https?://dl\.ndl\.go\.jp/pid/(\d+)")

# Generic source identifier used in page_log.jsonl/run_metadata.json so
# records from different library pipelines can eventually be pooled and
# distinguished by a single consistent field name.
SOURCE_NAME = "ndl"
SOURCE_DISPLAY_NAME = "National Diet Library Digital Collections (Japan)"

_contact = os.environ.get("NDL_CONTACT_EMAIL", "").strip()
USER_AGENT = (
    "Phyto-Vision-NDL-Illustration-YOLO-Pipeline/1.0 "
    f"(research; contact: {_contact})"
    if _contact
    else "Phyto-Vision-NDL-Illustration-YOLO-Pipeline/1.0 "
    "(research; set NDL_CONTACT_EMAIL for a contact address)"
)

DEFAULT_KEYWORDS = [
    # Explicit visual material
    "絵入",
    "絵入り",
    "図入",
    "図入り",
    "図譜",
    "図会",
    "図説",
    "画譜",
    "絵本",
    "画本",
    # Natural history / medicine
    "本草",
    "本草図譜",
    "博物",
    "博物学",
    "植物",
    "動物",
    "鳥類",
    "魚類",
    "虫",
    "薬草",
    "医学",
    "解剖",
    # Geography
    "地図",
    "絵図",
    "名所図会",
    "地誌",
    # Technical/scientific
    "天文",
    "暦",
    "機巧",
    "器械",
    "算法",
    # Visual culture
    "浮世絵",
    "妖怪",
    "紋様",
    "図案",
]

# Per user instruction: restrict to NDL's "oldmaterials" (pre-modern/rare
# books - the most illustration-dense collection by far), "books", and
# "manuscripts" mediaType values by default. See module docstring's "CQL
# limitation" section for why each value becomes its own SRU query rather
# than one OR-combined query.
DEFAULT_MEDIATYPE_FILTER = ["oldmaterials", "books", "manuscripts"]

DEFAULT_YOLO_MODEL = str(SCRIPT_DIR / "page_layout_best_new.pt")
DEFAULT_IMGSZ = 640
DEFAULT_CONF_THRESHOLD = 0.30
DEFAULT_ILLUSTRATION_CLASS_KEYWORDS = ["illustration"]

DEFAULT_MAX_BOOKS_PER_KEYWORD = 100
DEFAULT_ROWS_PER_PAGE = 200  # NDL SRU's own default; hard cap is 500.
NDL_SRU_MAX_REACHABLE_POSITION = 500  # confirmed live, see module docstring.
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 5
DEFAULT_MAX_PAGE_RETRIES = 5
DEFAULT_MAX_BOOK_RETRIES = 3
DEFAULT_MIN_PAGES_PER_BOOK = 5
DEFAULT_EDGE_SKIP_THRESHOLD = 20
DEFAULT_EDGE_SKIP_COUNT = 5
DEFAULT_STATE_AZURE_PREFIX = "state_backups/illustration_runs/ndl_illustration_yolo_run"
DEFAULT_STATE_UPLOAD_INTERVAL = 3000.0
DEFAULT_IIIF_WIDTH = 1200
DEFAULT_NEGATIVE_SAMPLE_RATE = 0.02


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
            "Search the National Diet Library (Japan) Digital Collections "
            "for illustration-rich books, triage every page with a GPU "
            "YOLO model, and stream illustration-positive pages to Azure "
            "Blob Storage."
        )
    )

    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Override the built-in (Japanese/Sino-Japanese) keyword list.",
    )
    parser.add_argument(
        "--mediatype-filter",
        nargs="*",
        default=DEFAULT_MEDIATYPE_FILTER,
        help="NDL mediatype values to restrict results to. Each value "
        "becomes its own SRU query per keyword (see module docstring's "
        "CQL-limitation note), merged/de-duplicated client-side. Pass "
        "--mediatype-filter with no values to search without any "
        f"mediatype restriction. Default: {DEFAULT_MEDIATYPE_FILTER}",
    )
    parser.add_argument(
        "--max-books-per-keyword",
        type=int,
        default=DEFAULT_MAX_BOOKS_PER_KEYWORD,
        help="Maximum number of NDL search results (after merging across "
        "--mediatype-filter values) to fetch per keyword.",
    )
    parser.add_argument(
        "--start-record",
        type=int,
        default=1,
        help="SRU startRecord position to start from for each (keyword, "
        "mediatype) query (1-based).",
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
        default=Path("ndl_illustration_yolo_run"),
        help="Directory for the state file, page log, run metadata, and the "
        "one-page-at-a-time temporary download. Kept distinct from the "
        "other pipelines' output dirs so local state never collides.",
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
        default="illustrations/ndl",
        help="Blob name prefix for kept page images. Defaults to "
        "'illustrations/ndl' (NOT the container root), matching the "
        "'illustrations/<library>' layout the other illustration pipelines "
        "use, so libraries never collide even inside the same container.",
    )
    parser.add_argument(
        "--negative-sample-rate",
        type=float,
        default=DEFAULT_NEGATIVE_SAMPLE_RATE,
        help="Fraction (0-1) of YOLO-negative pages to keep anyway for "
        "quality-control audit, e.g. 0.01 = 1%%. They are uploaded to "
        "--negative-audit-prefix instead of being deleted outright. "
        "Selection is deterministic (SHA-256 of source+item id+page id), "
        "not random, so the same page is always chosen or not chosen "
        f"across reruns. Default: {DEFAULT_NEGATIVE_SAMPLE_RATE} "
        f"({DEFAULT_NEGATIVE_SAMPLE_RATE * 100:.0f}%%).",
    )
    parser.add_argument(
        "--negative-audit-prefix",
        default="illustrations/ndl/negative_audit",
        help="Blob prefix for the negative audit sample. Default: "
        "illustrations/ndl/negative_audit",
    )
    parser.add_argument(
        "--state-azure-prefix",
        default=DEFAULT_STATE_AZURE_PREFIX,
        help="Blob prefix that processed_items.json/books.jsonl/"
        "page_log.jsonl/run_metadata.json are periodically pushed to (each "
        "upload overwrites the previous one), so a RunPod pod dying "
        "mid-run doesn't lose progress that only exists on local disk. "
        f"Default: {DEFAULT_STATE_AZURE_PREFIX}",
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
        help="Requested IIIF image width in pixels (0 = full resolution, "
        "capped by NDL at 5000px on the longer side regardless). Default: "
        f"{DEFAULT_IIIF_WIDTH}.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds for NDL SRU/IIIF requests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Delay after each successful search/manifest/image request. "
        "No numeric rate limit is documented for NDL's SRU API, but this "
        "is kept at a reasonable default out of courtesy - see module "
        "docstring. Note that --mediatype-filter multiplies the number of "
        "search requests per keyword (one per value).",
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
    parser.add_argument(
        "--max-book-retries",
        type=int,
        default=DEFAULT_MAX_BOOK_RETRIES,
        help="Maximum times a book-level failure (manifest couldn't be "
        "fetched, no pages could be resolved, etc. - as opposed to a "
        "page-level failure, see --max-page-retries) is retried across "
        "separate runs before the book is marked failed_permanent and "
        f"stops being re-discovered by future searches. Default: "
        f"{DEFAULT_MAX_BOOK_RETRIES}",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------
# NDL SRU / IIIF helpers
# --------------------------------------------------------------------------


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/xml,image/jpeg,image/png,image/*,*/*;q=0.8",
        }
    )
    return session


def safe_id(value: str) -> str:
    value = value.strip().rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180] or "unknown"


def strip_html(value: str) -> str:
    """Manifest metadata values are occasionally HTML fragments; collapse
    to plain text."""
    text = re.sub(r"<[^>]+>", "", value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_double_pipe_list(value: str | None) -> list[str]:
    """
    NDL IIIF manifest metadata packs multiple values for one label into a
    single "||"-delimited string (confirmed against NDL's own IIIF
    interface spec PDF) - a different convention from every other pipeline
    in this project (Bodleian/Gallica repeat the {label, value} entry,
    Wellcome uses "; ", MDZ uses a multilingual list). Splits back into a
    clean list.
    """
    if not value:
        return []
    return [part.strip() for part in value.split("||") if part.strip()]


def is_permanent_http_error(exc: Exception) -> bool:
    """
    True for 4xx responses the server used to deliberately refuse this exact
    request (403 Forbidden, 404 Not Found, ...) - retrying the same URL with
    backoff has no realistic chance of succeeding and just burns time. This
    also covers an access-restricted NDL item's image request (see module
    docstring's "Access Restrictions" note): if that surfaces as a non-429
    4xx, the page stops cleanly instead of retrying pointlessly.

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


def build_cql_query(keyword: str, mediatype: str | None) -> str:
    """
    Builds a pure-"and" CQL query - see module docstring's "CQL limitation"
    section for why this pipeline never mixes "and"/"or" or uses
    parentheses in one query. `anywhere` is NDL Search's simple/full-text
    field (documented as matching the same fields as NDL Search's own
    simple search box).
    """
    keyword = keyword.replace('"', "")
    query = f'anywhere="{keyword}"'
    if mediatype:
        query = f"{query} and mediatype={mediatype}"
    return query


def parse_sru_records(root: ET.Element) -> list[str]:
    """
    Extract every dl.ndl.go.jp Persistent ID (pid) reachable from an SRU
    searchRetrieveResponse's dcndl records - see module docstring for why
    this is the ONLY thing pulled from the search response (book metadata
    comes from the IIIF manifest instead). A single dcndl:BibResource can
    carry several dcndl:Item elements; only ones with a
    rdfs:seeAlso -> https://dl.ndl.go.jp/pid/<n> are usable by this
    pipeline (an item digitised by a partner institution elsewhere, or with
    no online copy at all, is skipped). De-duplicates within this one
    response.
    """
    pids: list[str] = []
    seen: set[str] = set()

    for record_el in root.findall("srw:records/srw:record", SRW_NS):
        record_data = record_el.find("srw:recordData", SRW_NS)
        if record_data is None:
            continue
        rdf_root = record_data.find("rdf:RDF", RDF_NS)
        if rdf_root is None:
            continue

        for item_el in rdf_root.findall("dcndl:Item", RDF_NS):
            for see_also in item_el.findall("rdfs:seeAlso", RDF_NS):
                resource = see_also.get(RDF_RESOURCE_ATTR)
                if not resource:
                    continue
                match = DL_NDL_PID_RE.match(resource)
                if match:
                    pid = match.group(1)
                    if pid not in seen:
                        seen.add(pid)
                        pids.append(pid)
                    break

    return pids


def search_ndl_one_query(
    session: requests.Session,
    keyword: str,
    mediatype: str | None,
    max_results: int,
    timeout: int,
    sleep_seconds: float,
    retries: int,
    start_record: int,
) -> list[str]:
    """
    Pages through ONE (keyword, mediatype) SRU query - see
    search_ndl_paginated() for how results across multiple mediatype
    values get merged.
    """
    pids: list[str] = []
    seen: set[str] = set()
    record_pos = start_record

    while len(pids) < max_results and record_pos <= NDL_SRU_MAX_REACHABLE_POSITION:
        maximum_records = min(
            DEFAULT_ROWS_PER_PAGE,
            NDL_SRU_MAX_REACHABLE_POSITION - record_pos + 1,
        )
        params = {
            "operation": "searchRetrieve",
            "version": "1.2",
            "recordSchema": "dcndl",
            "recordPacking": "xml",
            "query": build_cql_query(keyword, mediatype),
            "startRecord": record_pos,
            "maximumRecords": maximum_records,
        }

        root: ET.Element | None = None
        delay = 2.0

        for attempt in range(retries):
            try:
                response = session.get(NDL_SRU_URL, params=params, timeout=timeout)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                break
            except Exception as exc:
                if is_permanent_http_error(exc) or attempt == retries - 1:
                    print(
                        f"[warn] search failed for keyword={keyword!r} "
                        f"mediatype={mediatype!r} startRecord={record_pos}: {exc}"
                    )
                    break
                time.sleep(retry_delay_seconds(exc, delay))
                delay = min(delay * 2, 30.0)

        if root is None:
            break

        diagnostic = root.find("srw:diagnostics/srw:diagnostic", SRW_NS)
        if diagnostic is not None:
            message = diagnostic.findtext("srw:message", namespaces=SRW_NS)
            print(
                f"[warn] search returned a diagnostic for keyword={keyword!r} "
                f"mediatype={mediatype!r}: {message}"
            )
            break

        for pid in parse_sru_records(root):
            if pid not in seen:
                seen.add(pid)
                pids.append(pid)

        if sleep_seconds:
            time.sleep(sleep_seconds)

        next_position_text = root.findtext("srw:nextRecordPosition", namespaces=SRW_NS)
        try:
            next_position = int(next_position_text) if next_position_text else 0
        except ValueError:
            next_position = 0

        if next_position <= 0:
            break
        record_pos = next_position

    return pids[:max_results]


def search_ndl_paginated(
    session: requests.Session,
    keyword: str,
    mediatype_filter: list[str],
    max_results: int,
    timeout: int,
    sleep_seconds: float,
    retries: int = DEFAULT_RETRIES,
    start_record: int = 1,
) -> list[dict[str, Any]]:
    """
    Runs one SRU query per --mediatype-filter value (or a single
    unrestricted query if the filter is empty - see module docstring's CQL
    limitation note for why this can't be a single OR-combined query),
    merges/de-duplicates the resulting pids, and returns them as result
    dicts in the same shape the other pipelines' search functions use.
    """
    all_pids: list[str] = []
    seen: set[str] = set()

    mediatypes: list[str | None] = list(mediatype_filter) if mediatype_filter else [None]

    for mediatype in mediatypes:
        if len(all_pids) >= max_results:
            break
        pids = search_ndl_one_query(
            session=session,
            keyword=keyword,
            mediatype=mediatype,
            max_results=max_results - len(all_pids),
            timeout=timeout,
            sleep_seconds=sleep_seconds,
            retries=retries,
            start_record=start_record,
        )
        for pid in pids:
            if pid not in seen:
                seen.add(pid)
                all_pids.append(pid)

    return [
        {"item_id": pid, "manifest_url": f"{NDL_IIIF_BASE}/{pid}/manifest.json"}
        for pid in all_pids[:max_results]
    ]


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
    from the other four illustration pipelines. Confirmed live that NDL
    manifests are Presentation 2 (top-level "sequences"), same shape as the
    others', so this works unchanged apart from the `width` parameter. Each
    returned page dict also carries canvas_width/canvas_height (the
    canvas's own declared dimensions, i.e. the SOURCE image size per the
    manifest - not necessarily what gets downloaded, see process_page's
    downloaded_width/height) when the manifest provides them, which NDL's
    do.
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
                        "canvas_width": canvas.get("width"),
                        "canvas_height": canvas.get("height"),
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
                        "canvas_width": canvas.get("width"),
                        "canvas_height": canvas.get("height"),
                    }
                )

    return pages


# NDL manifest metadata label vocabulary, confirmed against NDL's own IIIF
# interface spec PDF and a live manifest - see module docstring. No
# "Subjects" label exists (same gap noted in the Gallica pipeline's
# docstring).
METADATA_LABELS = {
    "date": ["Publication Date"],
    "author": ["Creator"],
    "publisher": ["Publisher"],
    "series_title": ["Series Title"],
    "isbn": ["ISBN"],
    "call_number": ["Call Number"],
    "bibliographic_id": ["Bibliographic ID"],
    "doi": ["DOI"],
    "access_restrictions": ["Access Restrictions"],
    "notes": ["Notes", "Note"],
    "source_url": ["Source (URL)"],
    "viewer_url": ["URL"],
}


def extract_book_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """
    Best-effort bibliographic metadata extraction from a manifest's
    top-level "metadata" list of {"label": ..., "value": ...} pairs, ported
    from the Bodleian/Gallica/Wellcome pipelines (plain-string values), but
    splitting "||"-packed multi-values (see split_double_pipe_list()).
    Kept as manifest-only (not the SRU search response) so metadata
    extraction behaves identically whether a book is processed fresh or
    resumed via sweep_paused_books() - exactly like the other four
    pipelines.
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
            values = split_double_pipe_list(value)
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

    title = first("Title") or strip_html(str(manifest.get("label") or "")) or None
    attribution = strip_html(str(manifest.get("attribution") or "")) or None
    access_restrictions = first(*METADATA_LABELS["access_restrictions"])
    rights_statement = (
        " | ".join(
            p
            for p in (
                attribution,
                f"Access Restrictions: {access_restrictions}" if access_restrictions else None,
            )
            if p
        )
        or None
    )

    return {
        "title": title,
        "author": ", ".join(all_values(*METADATA_LABELS["author"])) or None,
        "date": first(*METADATA_LABELS["date"]),
        "publisher": first(*METADATA_LABELS["publisher"]),
        "series_title": first(*METADATA_LABELS["series_title"]),
        "isbn": first(*METADATA_LABELS["isbn"]),
        "call_number": first(*METADATA_LABELS["call_number"]),
        "bibliographic_id": first(*METADATA_LABELS["bibliographic_id"]),
        "doi": first(*METADATA_LABELS["doi"]),
        "access_restrictions": access_restrictions,
        "notes": " ".join(all_values(*METADATA_LABELS["notes"])) or None,
        "source_url": first(*METADATA_LABELS["source_url"]),
        "viewer_url": first(*METADATA_LABELS["viewer_url"]),
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
) -> tuple[bool, list[dict[str, Any]], float, int, int | None, int | None]:
    """
    Returns (has_illustration, detected, illustration_max_confidence,
    illustration_detection_count, image_width, image_height).

    `detected` includes every detection (not just illustration classes) with
    its bounding box, since the GPU cost of computing it is already paid and
    it is useful later for cropping, false-positive audits, and thesis
    figures. Each detection carries the pixel bbox_xyxy AND
    bbox_normalized_xyxy (each coordinate divided by the actual image
    width/height) plus bbox_area_ratio (box area / total image area).
    image_width/image_height are the ACTUAL decoded pixel dimensions of
    image_path, read from YOLO's own orig_shape rather than assumed to equal
    the requested IIIF width - a source image can legitimately come back a
    different size than requested. `illustration_max_confidence`/
    `illustration_detection_count` are scoped to illustration-related
    classes only, since a high-confidence "text_block" detection shouldn't
    count toward a page/book's illustration signal.
    """
    results = model.predict(
        source=str(image_path),
        device=device,
        imgsz=imgsz,
        conf=conf_threshold,
        verbose=False,
    )

    image_width: int | None = None
    image_height: int | None = None
    if results:
        orig_shape = getattr(results[0], "orig_shape", None)
        if orig_shape:
            image_height, image_width = int(orig_shape[0]), int(orig_shape[1])

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

            bbox_xyxy = [round(float(v), 2) for v in bbox]

            bbox_normalized_xyxy: list[float] | None = None
            bbox_area_ratio: float | None = None
            if image_width and image_height:
                x1, y1, x2, y2 = bbox_xyxy
                bbox_normalized_xyxy = [
                    round(x1 / image_width, 4),
                    round(y1 / image_height, 4),
                    round(x2 / image_width, 4),
                    round(y2 / image_height, 4),
                ]
                box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                bbox_area_ratio = round(
                    box_area / (image_width * image_height), 6
                )

            detected.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_normalized_xyxy": bbox_normalized_xyxy,
                    "bbox_area_ratio": bbox_area_ratio,
                }
            )

            if is_illustration:
                has_illustration = True
                illustration_detection_count += 1
                illustration_max_confidence = max(illustration_max_confidence, confidence)

    return (
        has_illustration,
        detected,
        illustration_max_confidence,
        illustration_detection_count,
        image_width,
        image_height,
    )


# --------------------------------------------------------------------------
# Deterministic negative-audit sampling
# --------------------------------------------------------------------------


def negative_audit_sample_value(source: str, item_id: str, page_identifier: str) -> float:
    """
    Deterministic replacement for random.random() - a page must always land
    on the same side of --negative-sample-rate across reruns, machines, and
    resumed runs, which a process-seeded PRNG can't guarantee. Hashing a
    stable key (source + item id + canvas/page id-or-index) with SHA-256 and
    mapping its first 8 bytes to [0, 1) gives a value that's reproducible
    anywhere yet still spreads pages uniformly across [0, 1) for sampling.
    Deliberately NOT Python's built-in hash(): it's randomized per-process
    (PYTHONHASHSEED) and not stable across runs/machines.
    """
    key = f"{source}:{item_id}:{page_identifier}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


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
    page_log.jsonl/run_metadata.json) to Azure, overwriting whatever was
    there before.
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
    Upload the current processed_items.json/books.jsonl/page_log.jsonl/
    run_metadata.json to Azure under --state-azure-prefix, each upload
    overwriting the file already there. A RunPod pod's local disk is
    ephemeral and the pod can die without warning, so this is the safety
    net that keeps a killed run's progress - and the record of exactly
    which model/config produced it - recoverable even if the local
    --output-dir is lost.
    """
    if container_client is None:
        return

    prefix = args.state_azure_prefix.strip("/")
    for local_path in (
        args.state_path,
        args.books_path,
        args.page_log_path,
        args.run_metadata_path,
    ):
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
# Run metadata (which model/config produced this corpus)
# --------------------------------------------------------------------------


def compute_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_metadata(args: argparse.Namespace, model, device: str) -> dict[str, Any]:
    """
    Snapshot of exactly which model/config produced this corpus - written
    once to run_metadata.json in --output-dir (see
    load_or_create_run_metadata) and mirrored to Azure alongside the other
    state files.
    """
    import torch
    import ultralytics

    names = model.names
    class_names = list(names.values()) if isinstance(names, dict) else list(names)

    device_info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        try:
            index = torch.cuda.current_device()
            device_info["cuda_version"] = torch.version.cuda
            device_info["gpu_name"] = torch.cuda.get_device_name(index)
        except Exception:
            pass

    return {
        "source": SOURCE_NAME,
        "source_name": SOURCE_DISPLAY_NAME,
        "run_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "yolo_model_path": str(args.yolo_model),
        "yolo_model_sha256": compute_sha256(Path(args.yolo_model)),
        "model_class_names": class_names,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf_threshold,
        "illustration_class_keywords": args.illustration_class_keywords,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "device": device,
        **device_info,
    }


def load_or_create_run_metadata(
    path: Path, args: argparse.Namespace, model, device: str
) -> dict[str, Any]:
    """
    Write run_metadata.json exactly once per --output-dir. A resumed run
    must NOT regenerate/overwrite it with a different snapshot - the whole
    point is that it identifies the model/config that produced the corpus
    already sitting in --output-dir and Azure, even if a later resume uses
    different flags. If the weights on disk no longer match the recorded
    hash, warn loudly (likely means --yolo-model was swapped without
    starting a fresh --output-dir) but still leave the file untouched.
    """
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        current_hash = compute_sha256(Path(args.yolo_model))
        recorded_hash = existing.get("yolo_model_sha256")
        if current_hash and recorded_hash and current_hash != recorded_hash:
            print(
                "[warn] run_metadata.json's recorded yolo_model_sha256 does "
                "not match the currently loaded weights - this --output-dir "
                "was previously used with a different model. Leaving "
                "run_metadata.json unchanged; start a fresh --output-dir if "
                "you intend to mix models."
            )
        return existing

    metadata = build_run_metadata(args, model, device)
    save_state_atomic(path, metadata)
    return metadata


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
            item_id = record.get("ndl_item_id")
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
    manifest_url: str | None,
    page: dict[str, Any],
    page_index: int,
    tmp_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    page_identifier = str(page.get("id", page_index))
    page_id = safe_id(page_identifier)
    tmp_path = tmp_dir / f"page_{page_index + 1:05d}_{page_id}.jpg"

    record: dict[str, Any] = {
        # Generic, source-agnostic provenance fields (see module docstring
        # workflow step 6) - the same field names other library pipelines
        # in this project also write, so records can eventually be pooled.
        "source": SOURCE_NAME,
        "source_item_id": item_id,
        "ndl_item_id": item_id,  # kept for backward compatibility
        "keyword": keyword,
        "manifest_url": manifest_url,
        "canvas_id": page.get("id"),
        "page_label": page.get("label"),
        "page_index": page_index,
        "page_number": page_index + 1,
        "image_service_url": page.get("image_service"),
        "image_url": page.get("image_url"),
        "source_width": page.get("canvas_width"),
        "source_height": page.get("canvas_height"),
        "downloaded_width": None,
        "downloaded_height": None,
        "action": None,
        "blob_name": None,
        "detected_classes": [],
        "illustration_detection_count": 0,
        "max_confidence": None,
        "negative_audit_selected": None,
        "negative_audit_sample_value": None,
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
        (
            has_illustration,
            detected,
            illustration_max_confidence,
            illustration_detection_count,
            image_width,
            image_height,
        ) = detect_illustration(
            model=model,
            image_path=tmp_path,
            device=device,
            imgsz=args.imgsz,
            conf_threshold=args.conf_threshold,
            illustration_class_keywords=args.illustration_class_keywords,
        )
        record["detected_classes"] = detected
        record["illustration_detection_count"] = illustration_detection_count
        record["max_confidence"] = illustration_max_confidence if has_illustration else None
        record["downloaded_width"] = image_width
        record["downloaded_height"] = image_height

        # 3a. Illustration detected -> upload, then delete local copy.
        if has_illustration:
            blob_name = build_blob_name(args.azure_prefix, item_id, page_index)

            if not args.dry_run:
                upload_image_to_blob(container_client, blob_name, tmp_path)
                record["action"] = "kept_uploaded"
            else:
                record["action"] = "kept_dry_run"

            record["blob_name"] = blob_name

        # 3b. No illustration -> normally delete local copy, nothing
        # uploaded, except for a deterministically-sampled slice kept as a
        # negative QC audit sample (uploaded, not just left on disk, so
        # review doesn't require re-running this whole pipeline). The
        # sample value is computed and recorded for EVERY negative page,
        # not just ones that end up selected, so the sampling decision is
        # independently verifiable later.
        else:
            sample_value = negative_audit_sample_value(
                SOURCE_NAME, item_id, page_identifier
            )
            is_selected = sample_value < args.negative_sample_rate
            record["negative_audit_sample_value"] = round(sample_value, 8)
            record["negative_audit_selected"] = is_selected

            if is_selected:
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
    manifest_url: str | None,
    page: dict[str, Any],
    page_index: int,
    log_path: Path,
) -> dict[str, Any]:
    """
    Record a leading/trailing page of a long book WITHOUT downloading it or
    running YOLO - see --edge-skip-threshold/--edge-skip-count. The page's
    IIIF image URL and position are still written to page_log.jsonl so its
    existence is never lost, it's just never classified (so
    downloaded_width/height and the negative-audit fields stay null - there
    was no download and no detection to base them on).
    """
    record: dict[str, Any] = {
        "source": SOURCE_NAME,
        "source_item_id": item_id,
        "ndl_item_id": item_id,  # kept for backward compatibility
        "keyword": keyword,
        "manifest_url": manifest_url,
        "canvas_id": page.get("id"),
        "page_label": page.get("label"),
        "page_index": page_index,
        "page_number": page_index + 1,
        "image_service_url": page.get("image_service"),
        "image_url": page.get("image_url"),
        "source_width": page.get("canvas_width"),
        "source_height": page.get("canvas_height"),
        "downloaded_width": None,
        "downloaded_height": None,
        "action": "skipped_edge_page",
        "blob_name": None,
        "detected_classes": [],
        "illustration_detection_count": 0,
        "max_confidence": None,
        "negative_audit_selected": None,
        "negative_audit_sample_value": None,
        "error": None,
    }
    append_log(log_path, record)
    return record


def record_book_failure(
    state: dict[str, Any],
    item_id: str,
    error: str,
    keyword: str,
    max_book_retries: int,
) -> str:
    """
    Shared bookkeeping for a book-level failure - the manifest couldn't be
    fetched, no pages could be resolved from it, or any other exception
    happened before/instead of the per-page loop (as opposed to a
    page-level failure, which pauses the book "in_progress" and is capped
    by a separate mechanism - see --max-page-retries). Tracks
    book_retry_count across separate runs the same way page-level retries
    are tracked: once max_book_retries is reached, the book is marked
    "failed_permanent" (a terminal status, like "completed") so it stops
    being re-discovered and re-attempted on every future run. Without this
    cap, a search result whose manifest is permanently missing or broken
    on the source's end - seen in practice: a library's own catalog can
    reference a "digitized" item whose manifest was never actually
    published - would be retried forever, since a plain "failed" status is
    NOT terminal. Returns the final status assigned, for logging.
    """
    previous = state.get(item_id) or {}
    keywords_matched = set(previous.get("keywords_matched", []))
    keywords_matched.add(keyword)
    retry_count = previous.get("book_retry_count", 0) + 1
    status = "failed_permanent" if retry_count >= max_book_retries else "failed"

    state[item_id] = {
        **previous,
        "status": status,
        "error": error,
        "book_retry_count": retry_count,
        "keywords_matched": sorted(keywords_matched),
    }
    return status


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
        print("[skip] could not determine NDL persistent ID from search result")
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
            "ndl_item_id": item_id,
            "ndl_url": metadata.get("viewer_url")
            or f"{NDL_VIEWER_BASE}/pid/{item_id}",
            "manifest_url": manifest_url,
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "date": metadata.get("date"),
            "publisher": metadata.get("publisher"),
            "series_title": metadata.get("series_title"),
            "isbn": metadata.get("isbn"),
            "call_number": metadata.get("call_number"),
            "bibliographic_id": metadata.get("bibliographic_id"),
            "doi": metadata.get("doi"),
            "access_restrictions": metadata.get("access_restrictions"),
            "notes": metadata.get("notes"),
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
            status = record_book_failure(
                state, item_id, "no page images found", keyword, args.max_book_retries
            )
            save_state_atomic(args.state_path, state)
            flush_book_record(status)
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
                    manifest_url=manifest_url,
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
                    manifest_url=manifest_url,
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
        status = record_book_failure(state, item_id, str(exc), keyword, args.max_book_retries)
        if status == "failed_permanent":
            print(
                f"[error] {item_id}: giving up after repeated book-level "
                "failures (failed_permanent)"
            )
        save_state_atomic(args.state_path, state)
        flush_book_record(status)


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
    wall-clock time (transient NDL server issues tend to clear up), not
    another immediate retry. Calling this between keyword searches (and
    once more at the end of the run) gives every paused book that gap
    naturally, instead of leaving it stuck until the same item happens to
    resurface under a later keyword's search results.

    A resumed book's manifest URL is reconstructed from its pid rather than
    replayed from the original search result (which isn't kept in state),
    since NDL's manifest URL is a deterministic function of the pid - like
    Gallica/MDZ, and unlike Wellcome (which needs a re-fetch since its
    manifest URL isn't derivable from its catalogue id alone).
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
            "manifest_url": f"{NDL_IIIF_BASE}/{item_id}/manifest.json",
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
    args.run_metadata_path = args.output_dir / "run_metadata.json"
    args.last_state_upload_time = 0.0

    device = resolve_device(args.device, args.allow_cpu)
    print(f"Using device: {device}")

    try:
        model = load_yolo_model(args.yolo_model, device)
    except Exception as exc:
        print(f"Failed to load YOLO model: {exc}", file=sys.stderr)
        return 1

    run_metadata = load_or_create_run_metadata(args.run_metadata_path, args, model, device)
    print(
        f"Run metadata: {args.run_metadata_path} "
        f"(yolo_model_sha256={run_metadata.get('yolo_model_sha256')})"
    )

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
            "[warn] NDL_CONTACT_EMAIL is not set; the User-Agent sent to "
            "NDL Search has no contact address. Good practice for any "
            "harvest - consider setting it."
        )

    state = load_state(args.state_path)
    books = load_books(args.books_path)
    session = make_session()

    keywords = args.keywords or DEFAULT_KEYWORDS
    print(f"Keywords ({len(keywords)}): {', '.join(keywords)}")
    print(f"Mediatype filter: {args.mediatype_filter or '(none - all mediatypes)'}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Illustration class keywords: {args.illustration_class_keywords}")
    print(f"Output dir: {args.output_dir}")
    print(f"IIIF image width: {args.iiif_width or 'full resolution'}")
    print(f"Negative audit sample rate: {args.negative_sample_rate}")
    print(f"Azure image prefix: {args.azure_prefix}")
    print(
        f"Azure state backup prefix: {args.state_azure_prefix} "
        f"(every {args.state_upload_interval:.0f}s)"
    )

    for keyword in keywords:
        print(f"\n=== Keyword: {keyword!r} ===")
        try:
            search_results = search_ndl_paginated(
                session=session,
                keyword=keyword,
                mediatype_filter=args.mediatype_filter,
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
            # state/books/page-log/run-metadata files to Azure every so
            # often (rate limited by --state-upload-interval), overwriting
            # the previous backup each time.
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
    print(f"Run metadata: {args.run_metadata_path}")
    print(f"Azure state backup: {args.state_azure_prefix} (skipped in dry-run mode)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
