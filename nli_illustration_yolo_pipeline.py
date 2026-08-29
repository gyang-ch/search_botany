#!/usr/bin/env python3
"""
nli_illustration_yolo_pipeline.py

Large-scale National Library of Israel (NLI) Open Library book/manuscript/
map-page harvesting with GPU YOLO triage, searching for pages with RICH
ILLUSTRATION. Sibling script to bodleian_illustration_yolo_pipeline.py,
gallica_illustration_yolo_pipeline.py, mdz_illustration_yolo_pipeline.py,
wellcome_illustration_yolo_pipeline.py, ndl_illustration_yolo_pipeline.py,
rmda_illustration_yolo_pipeline.py, loc_illustration_yolo_pipeline.py, and
waseda_illustration_yolo_pipeline.py - same one-page-at-a-time GPU triage,
Azure upload strategy, and page_layout_best_new.pt model, pointed at NLI's
IIIF API instead. Uses the user-supplied DEFAULT_KEYWORDS list below
(English/transliterated terms spanning Islamic and Jewish illustrated
manuscript traditions, plus general visual/scientific/cartographic terms),
per user instruction - NOT the Japanese/Sino-Japanese list the NDL/RMDA/
Waseda pipelines share.

IMPORTANT - keeping this separate from the other pipelines' data
--------------------------------------------------------------------------
NLI catalogue records are identified by a DOCID such as
"NNL_MAPS_JER002366843" or "NNL_ALEPH11357291990005171" (an institution-code
prefix + a longer alphanumeric/Aleph-system-number string) - a distinct
namespace from every other library this project searches, so there's no
risk of a literal blob-name collision. Even so, this script follows the
exact same "<task>/<library>" file management plan laid out in
bodleian_illustration_yolo_pipeline.py, so every library's illustration
search lives in its own clearly labelled corner and nothing has to be
cross-checked by hand later:
  - --azure-prefix defaults to "illustrations/nli" (siblings:
    "illustrations/bodleian_new", "illustrations/gallica",
    "illustrations/mdz", "illustrations/wellcome", "illustrations/ndl",
    "illustrations/rmda", "illustrations/loc", "illustrations/waseda"),
  - --negative-audit-prefix defaults to "illustrations/nli/negative_audit",
  - --output-dir defaults to "nli_illustration_yolo_run" (siblings:
    "bodleian_illustration_yolo_run", "gallica_illustration_yolo_run",
    "mdz_illustration_yolo_run", "wellcome_illustration_yolo_run",
    "ndl_illustration_yolo_run", "rmda_illustration_yolo_run",
    "loc_illustration_yolo_run", "waseda_illustration_yolo_run"), so local
    state/temp files never collide either,
  - --state-azure-prefix defaults to
    "state_backups/illustration_runs/nli_illustration_yolo_run", alongside
    the other pipelines' own prefixes under the same "illustration_runs/"
    umbrella.
Because the output directories, Azure prefixes, and local temp/tmux
concerns are all fully separate, this script is safe to run at the same
time as the other illustration pipelines in their own tmux sessions on the
same RunPod pod.

API reference - IMPORTANT CAVEAT: confirmed live that NLI's site and its
legacy search API sit behind an aggressive Cloudflare bot-challenge; only a
REGISTERED, API-KEY-AUTHENTICATED REST endpoint is actually usable from a
plain script
-------------------------------------------------------------------------
https://www.nli.org.il/en/research-and-teach/open-library (the page the
user pointed at) links out to several sub-pages describing NLI's developer
APIs. Investigating live during development (2026-08-29) turned up the
following, confirmed by direct HTTP requests (not just reading docs):

1. www.nli.org.il, merhav.nli.org.il (the modern Primo discovery UI), AND
   primo.nli.org.il (NLI's older, publicly-documented Primo X-Services
   search endpoint, `GET .../PrimoWebServices/xservice/search/brief`) ALL
   return a Cloudflare "managed challenge" (a JavaScript/cookie challenge
   page, HTTP 403, `cf-mitigated: challenge`) to a plain HTTP client - CONFIRMED
   LIVE, not merely suspected. None of these three hosts can be queried by
   this pipeline (or any non-browser script) at all, regardless of
   parameters. This rules out the OLDER, free/keyless Primo X-Services
   search API entirely, even though it is the one still described on a
   stale developer-docs mirror (http://nli-dev.webflow.io/search-api) that
   is itself NOT behind Cloudflare and so remains readable - it is
   documentation for an endpoint this pipeline cannot actually reach.
2. The image/manifest server, iiif.nli.org.il, is DIFFERENT: it answers
   plain HTTP requests directly with real application-level XML errors
   (`<error><parameter>...</parameter><text>...</text></error>`, e.g.
   "Forbidden" for an access-restricted DOCID, "server_error" for an
   unknown/stale DOCID) - confirmed live, no Cloudflare challenge
   encountered against this host. This pipeline's actual page-by-page IIIF
   work (manifest + image fetching) therefore behaves like every other
   pipeline in this project - see point 4 onward.
3. NLI's CURRENT, officially-documented search path (per the current
   www.nli.org.il Search API page, itself unreachable directly - see point
   1 - but described consistently across multiple independent secondary
   sources found during development) is a newer, registration-gated REST
   API, NOT the legacy X-Services endpoint:
       GET https://api.nli.org.il/openlibrary/search
           ?api_key=<your key>&query=<field>,<operator>,<value>[,AND;<field>,<operator>,<value>...]&output_format=xml
   e.g. `query=title,exact,jerusalem,AND;creator,contains,Jerusalem,AND;
   language,exact,eng,AND;start_date,contains,1951`. `api.nli.org.il`
   itself answers a keyless request with a clean AWS API Gateway
   `MissingAuthenticationTokenException` JSON 403 (confirmed live) - i.e.
   it is NOT Cloudflare-challenged, it just requires a real key. A FREE key
   can be self-registered at https://api2.nli.org.il/signup/ - this
   pipeline reads it from the NLI_API_KEY environment variable (see
   "Environment" below) and refuses to start without one, since there is no
   keyless fallback left for search.
4. CONFIRMED-LIVE GAP, READ BEFORE TRUSTING THE SEARCH LAYER BLINDLY: this
   pipeline could NOT be tested end-to-end against a real API key (none was
   available during development) and the endpoint's own documentation page
   could not be fetched directly (see point 1), so the exact XML/JSON
   response schema and pagination parameter names for THIS SPECIFIC REST
   endpoint are a well-informed best effort, not a "confirmed live" fact
   like everything else in this docstring:
     - The query-clause syntax (`field,operator,value`, "any" as the
       all-fields field code, "contains" as the substring operator) is
       reused from NLI's OLDER, closely-related X-Services API (which IS
       fully documented, see point 1) since both are Ex Libris Primo-family
       search layers over the same underlying "PNX" (Primo Normalized XML)
       record schema, and "any"/"contains" are near-universal Primo field/
       operator codes. build_search_query() builds `any,contains,<keyword>`
       for every keyword in DEFAULT_KEYWORDS.
     - Pagination parameter names (`indx`=1-based start index,
       `bulkSize`=page size) are likewise carried over from the
       X-Services convention as the best available guess - see
       NLI_SEARCH_INDEX_PARAM/NLI_SEARCH_PAGE_SIZE_PARAM below, trivial to
       edit if wrong.
     - The response record-identifier path is documented (for X-Services,
       and PNX's "control/recordid" is the same underlying field the
       Presentation API needs - "this identifier is required for the
       Presentation API" is stated directly): parse_search_response_xml()
       does NOT hardcode one fixed nested path, though - it searches the
       ENTIRE returned XML tree for any element whose LOCAL tag name
       (namespace-stripped) is "recordid", so it keeps working even if the
       new API wraps the same PNX record in a different envelope than the
       legacy DOCSET/DOC/PrimoNMBib structure. Total-hit-count extraction is
       similarly a best-effort scan for a "TOTALHITS"-like
       attribute/element, with a robust fallback: pagination ALWAYS stops
       once a page yields zero record ids not already seen (the same safety
       net rmda_illustration_yolo_pipeline.py/
       waseda_illustration_yolo_pipeline.py use for their own
       less-than-fully-documented pagination), so a wrong parameter name
       degrades gracefully into "only the first page of results is usable"
       rather than looping incorrectly.
     - Every raw search response is saved (overwriting the previous one) to
       <output_dir>/debug_last_search_response.xml purely as a diagnostic
       aid - once you have a real NLI_API_KEY and see how the actual
       response looks, check that file first if zero results are ever found
       for a keyword that clearly should have hits, and adjust
       parse_search_response_xml()/build_search_query_params() accordingly.
5. IIIF manifest URL for a known DOCID is deterministic (confirmed live -
   the manifest endpoint answers a real, meaningful error rather than a
   bot-challenge for both invalid and valid-but-restricted DOCIDs):
       http://iiif.nli.org.il/IIIFv21/DOCID/<DOCID>/manifest
   NLI's own documentation states the Presentation API implements IIIF
   Presentation API version 2.1 specifically (http://iiif.io/api/
   presentation/2.1) - so, unlike rmda_illustration_yolo_pipeline.py/
   waseda_illustration_yolo_pipeline.py, this pipeline does not need
   RMDA's language-map metadata handling (a v3-only convention); NLI
   metadata values are plain strings or arrays of plain strings, per the
   v2.1 spec. parse_iiif_manifest() below still keeps both the
   Presentation-2 AND Presentation-3 branches ported unchanged from every
   sibling pipeline anyway, at zero cost, in case NLI ever migrates - the
   same defensive posture this project already uses everywhere.
6. IIIF Image API is version 2.1 too, confirmed by NLI's own documentation,
   with one distinctive, confirmed-live constraint that has NO equivalent
   in any other pipeline in this project: NLI enforces a hard GLOBAL
   MAXIMUM of 576 PIXELS on the requested image size (documented, and
   consistent with this being a smaller/older Image API deployment than the
   other institutions' - NDL/RMDA/Waseda cap in the 2000-5000px range).
   DEFAULT_IIIF_WIDTH is therefore 576 for this pipeline specifically -
   deliberately NOT the 1200px every other sibling pipeline defaults to -
   since requesting more would simply be clamped or rejected by NLI's own
   server. --iiif-width can still be set lower if you want smaller/faster
   downloads, but there is no point setting it higher than 576.
7. Material-type-aware default: --min-pages-per-book defaults to 1 for this
   pipeline (every sibling pipeline defaults to 5). Several of the visual
   traditions this pipeline's own keyword list specifically targets -
   Ketubah (a marriage contract), Megillah (in single-sheet/scroll form),
   and NLI's map collections - are INHERENTLY single-sheet/single-image
   items, not multi-page books; keeping the usual 5-page minimum would
   silently skip almost all of them as "skipped_too_short" before a single
   page was ever looked at, defeating half the point of this specific
   keyword list. --min-pages-per-book can still be raised back up with the
   usual flag if you want to focus purely on multi-page illustrated books.

Default YOLO model
-------------------
Defaults to ./page_layout_best_new.pt (the same fine-tuned model as every
other illustration pipeline's default), classes:
    illustration, text_block
A page is treated as "contains a rich illustration" when any detected class
name contains "illustration" (see --illustration-class-keywords).

Workflow
--------
1. Search NLI for each keyword in KEYWORDS (see module-level
   DEFAULT_KEYWORDS - the user-supplied English/transliterated list, spanning
   Islamic and Jewish illustrated-manuscript terms plus general visual/
   scientific/cartographic terms) via the registered REST Search API - see
   "API reference" above for exactly what is and isn't confirmed about this
   step.
2. For each DOCID, fetch its IIIF manifest (URL built deterministically) and
   enumerate every page/canvas.
3. Skip the whole item if it has fewer than --min-pages-per-book pages
   (default 1 - see point 7 above) - recorded as "skipped_too_short", never
   rechecked.
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
   fields alongside the library-specific nli_docid field, the manifest URL,
   canvas id/label, page index/number, the IIIF image service URL, the
   exact requested image URL, the downloaded image's actual pixel
   dimensions (read from YOLO's own decode - never assumed to equal the
   requested IIIF width), the source canvas's width/height from the
   manifest when available, the blob name when uploaded, and - for
   illustration-negative pages - whether the page was selected for the
   negative audit sample and the deterministic sampling value used.
7. Book-level bibliographic metadata, the manifest's attribution/license,
   page counts, illustration-detection totals, and an illustration_density
   ratio are written to books.jsonl, one line per book.
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
An NLI_API_KEY is mandatory (see above); the legacy X-Services docs this
pipeline's query syntax borrows from say plainly: "The Search API is
protected by overflow control by our Firewall, please avoid continuous
pagination through all results" - kept in mind via a conservative default
--sleep, matching this project's general posture toward APIs with no
numeric rate limit published. Every book record in books.jsonl carries the
manifest's attribution/license fields so provenance travels with the data.

Environment
-----------
Credentials are read from environment variables (never hardcode them in
this file, since it goes to GitHub). Put them in a local `.env` (gitignored)
or set them as RunPod pod environment variables / secrets.

    NLI_API_KEY="..."   # REQUIRED - self-register for free at https://api2.nli.org.il/signup/

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

Running alongside the other illustration pipelines
----------------------------------------------------
This script and the other illustration pipelines write to fully separate
local directories and Azure prefixes (see "IMPORTANT" above), so it's safe
to run this in its own tmux session alongside all the others on the same
pod:

    tmux new -s nli_illustration
    conda activate botany_yolo   # or whatever env has torch/ultralytics
    python nli_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux attach -t nli_illustration   # reattach to check on it
    tmux ls                           # list all sessions

This script defaults --device to the same auto-detected cuda:0 the other
pipelines use, so on a single-GPU pod every concurrently-running
illustration pipeline shares that one GPU's compute/VRAM (they'll interleave
GPU time rather than truly run in parallel), but if you have a multi-GPU
pod, pass --device cuda:1/cuda:2/... to spread them out.

Example
-------
python nli_illustration_yolo_pipeline.py \
    --azure-container botany-pages \
    --max-books-per-keyword 100 \
    --output-dir nli_illustration_yolo_run

Dry run (no Azure credentials needed, still deletes local files; an
NLI_API_KEY is still required since there is no keyless search fallback):
python nli_illustration_yolo_pipeline.py --dry-run --max-books-per-keyword 2
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

NLI_SEARCH_URL = "https://api.nli.org.il/openlibrary/search"
NLI_IIIF_MANIFEST_BASE = "https://iiif.nli.org.il/IIIFv21/DOCID"
SCRIPT_DIR = Path(__file__).resolve().parent

# Best-effort pagination parameter names carried over from NLI's older,
# fully-documented X-Services search API - see module docstring point 4 for
# why these are a well-informed guess rather than a confirmed-live fact for
# THIS specific endpoint, and why the pagination loop degrades gracefully
# (stops after one page) if they turn out to be wrong.
NLI_SEARCH_INDEX_PARAM = "indx"
NLI_SEARCH_PAGE_SIZE_PARAM = "bulkSize"
NLI_SEARCH_QUERY_PARAM = "query"
NLI_SEARCH_OUTPUT_FORMAT_PARAM = "output_format"
NLI_SEARCH_API_KEY_PARAM = "api_key"

# Generic source identifier used in page_log.jsonl/run_metadata.json so
# records from different library pipelines can eventually be pooled and
# distinguished by a single consistent field name.
SOURCE_NAME = "nli"
SOURCE_DISPLAY_NAME = "National Library of Israel Open Library"

_api_key = os.environ.get("NLI_API_KEY", "").strip()

USER_AGENT = "Phyto-Vision-NLI-Illustration-YOLO-Pipeline/1.0 (research)"

# Per user instruction - English/transliterated terms spanning Islamic and
# Jewish illustrated-manuscript traditions plus general visual/scientific/
# cartographic terms. Deliberately NOT the Japanese/Sino-Japanese list the
# NDL/RMDA/Waseda pipelines share.
DEFAULT_KEYWORDS = [
    # Visual characteristics
    "illustrated",
    "illuminated",
    "miniature",
    "pictorial",
    "diagram",
    "map",
    # Scientific / technical
    "astronomy",
    "astrology",
    "medicine",
    "anatomy",
    "botany",
    "zoology",
    "natural history",
    "cosmology",
    "geometry",
    # Islamic illustrated traditions
    "Shahnamah",
    "Khamsah",
    "Ajāʾib al-Makhlūqāt",
    "Dala'il al-Khayrat",
    # Jewish visual traditions
    "Haggadah",
    "Ketubah",
    "Megillah",
    "micrography",
    # Broader geographic/cartographic
    "geography",
    "cartography",
]

DEFAULT_YOLO_MODEL = str(SCRIPT_DIR / "page_layout_best_new.pt")
DEFAULT_IMGSZ = 640
DEFAULT_CONF_THRESHOLD = 0.30
DEFAULT_ILLUSTRATION_CLASS_KEYWORDS = ["illustration"]

DEFAULT_MAX_BOOKS_PER_KEYWORD = 100
NLI_RESULTS_PER_PAGE = 50  # matches the example value in NLI's own X-Services docs.
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 5
DEFAULT_MAX_PAGE_RETRIES = 5
DEFAULT_MAX_BOOK_RETRIES = 3
DEFAULT_MIN_PAGES_PER_BOOK = 1  # see module docstring point 7.
DEFAULT_EDGE_SKIP_THRESHOLD = 20
DEFAULT_EDGE_SKIP_COUNT = 5
DEFAULT_STATE_AZURE_PREFIX = "state_backups/illustration_runs/nli_illustration_yolo_run"
DEFAULT_STATE_UPLOAD_INTERVAL = 3000.0
DEFAULT_IIIF_WIDTH = 576  # NLI's own documented hard global cap - see module docstring point 6.
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
            "Search the National Library of Israel's Open Library for "
            "illustration-rich books/manuscripts/maps, triage every page "
            "with a GPU YOLO model, and stream illustration-positive pages "
            "to Azure Blob Storage."
        )
    )

    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Override the built-in keyword list (Islamic/Jewish "
        "illustrated-manuscript terms plus general visual/scientific/"
        "cartographic terms, per user instruction).",
    )
    parser.add_argument(
        "--api-key",
        default=_api_key or None,
        help="NLI Open Library API key. Defaults to NLI_API_KEY. REQUIRED - "
        "self-register for free at https://api2.nli.org.il/signup/ (see "
        "module docstring - NLI's legacy keyless search API is blocked by "
        "Cloudflare for non-browser clients as of 2026-08-29).",
    )
    parser.add_argument(
        "--max-books-per-keyword",
        type=int,
        default=DEFAULT_MAX_BOOKS_PER_KEYWORD,
        help="Maximum number of NLI search results to fetch per keyword.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help=f"Search result index to start from for each keyword (1-based, "
        f"matches the '{NLI_SEARCH_INDEX_PARAM}' parameter's own "
        f"convention; {NLI_RESULTS_PER_PAGE} results requested per page).",
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
        "run through YOLO). Defaults to 1 for this pipeline specifically - "
        f"see module docstring point 7. Default: {DEFAULT_MIN_PAGES_PER_BOOK}",
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
        default=Path("nli_illustration_yolo_run"),
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
        default="illustrations/nli",
        help="Blob name prefix for kept page images. Defaults to "
        "'illustrations/nli' (NOT the container root), matching the "
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
        default="illustrations/nli/negative_audit",
        help="Blob prefix for the negative audit sample. Default: "
        "illustrations/nli/negative_audit",
    )
    parser.add_argument(
        "--state-azure-prefix",
        default=DEFAULT_STATE_AZURE_PREFIX,
        help="Blob prefix that processed_items.json/books.jsonl/"
        "page_log.jsonl/run_metadata.json are periodically pushed to (each "
        "upload overwriting the previous one), so a RunPod pod dying "
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
        "without cloud credentials. An NLI_API_KEY is still required, since "
        "there is no keyless search fallback (see module docstring).",
    )

    # HTTP
    parser.add_argument(
        "--iiif-width",
        type=int,
        default=DEFAULT_IIIF_WIDTH,
        help="Requested IIIF image width in pixels. NLI enforces a hard "
        f"global maximum of 576px (see module docstring point 6) - there is "
        f"no benefit to setting this above {DEFAULT_IIIF_WIDTH}. Default: "
        f"{DEFAULT_IIIF_WIDTH}.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds for NLI search/manifest/image requests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Delay after each successful search/manifest/image request. "
        "Kept conservative given NLI's own search-API docs explicitly warn "
        "of firewall overflow control against continuous pagination - see "
        "module docstring.",
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
# NLI search / IIIF helpers
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


def is_permanent_http_error(exc: Exception) -> bool:
    """
    True for 4xx responses the server used to deliberately refuse this exact
    request (403 Forbidden, 404 Not Found, ...) - retrying the same URL with
    backoff has no realistic chance of succeeding and just burns time. This
    also covers an access-restricted NLI item's manifest/image request (see
    module docstring point 2): a 403 there stops cleanly instead of
    retrying pointlessly.

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


def build_search_query_params(
    api_key: str, keyword: str, index: int, page_size: int
) -> dict[str, Any]:
    """
    Builds the request params for NLI's registered REST Search API - see
    module docstring point 4 for exactly which parts of this are confirmed
    live vs. a well-informed best effort. `any,contains,<keyword>` is a
    single all-fields substring-match clause, the direct parallel to how
    every sibling pipeline in this project does an "anywhere" keyword
    search (NDL's `anywhere=`, RMDA/Waseda's free-text search box).
    """
    return {
        NLI_SEARCH_API_KEY_PARAM: api_key,
        NLI_SEARCH_QUERY_PARAM: f"any,contains,{keyword}",
        NLI_SEARCH_OUTPUT_FORMAT_PARAM: "xml",
        NLI_SEARCH_INDEX_PARAM: index,
        NLI_SEARCH_PAGE_SIZE_PARAM: page_size,
    }


def find_elements_by_local_tag(root: ET.Element, tag_name: str) -> list[ET.Element]:
    """
    Namespace-agnostic element search - find every element anywhere in the
    tree whose tag matches `tag_name` once any `{namespace}` prefix is
    stripped. Used instead of a fixed XPath because the exact XML envelope
    NLI's newer REST Search API wraps its (almost certainly still
    PNX-shaped) records in has not been confirmed live - see module
    docstring point 4.
    """
    return [
        el
        for el in root.iter()
        if el.tag.rsplit("}", 1)[-1].lower() == tag_name.lower()
    ]


def parse_search_response_xml(xml_text: str) -> tuple[list[str], int | None]:
    """
    Extract every DOCID (via any "recordid"-named element anywhere in the
    response, de-duplicated, first-seen order) and a best-effort total-hit
    count from one NLI search response. Returns (docids, total_count);
    total_count is None if no plausible total-hits attribute/element was
    found (pagination then relies purely on the "a page returned nothing
    new" stopping condition - see search_nli_paginated()). See module
    docstring point 4 for why this is deliberately schema-tolerant rather
    than a single hardcoded XPath.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [], None

    seen: set[str] = set()
    docids: list[str] = []
    for element in find_elements_by_local_tag(root, "recordid"):
        value = (element.text or "").strip()
        if value and value not in seen:
            seen.add(value)
            docids.append(value)

    total_count: int | None = None
    for element in root.iter():
        local_tag = element.tag.rsplit("}", 1)[-1]
        for attr_name, attr_value in element.attrib.items():
            if attr_name.upper() in ("TOTALHITS", "TOTAL_HITS", "TOTAL"):
                try:
                    total_count = int(attr_value)
                except ValueError:
                    pass
        if local_tag.upper() in ("TOTALHITS", "TOTAL_HITS") and element.text:
            try:
                total_count = int(element.text.strip())
            except ValueError:
                pass

    return docids, total_count


def search_nli_paginated(
    session: requests.Session,
    api_key: str,
    keyword: str,
    max_results: int,
    timeout: int,
    sleep_seconds: float,
    retries: int,
    start_index: int,
    debug_response_path: Path | None,
) -> list[dict[str, Any]]:
    """
    Pages through NLI's Search API for one keyword. Stops once max_results
    DOCIDs are collected, a page yields no NEW DOCIDs (the robust fallback
    stopping condition - see module docstring point 4, this is what keeps a
    wrong pagination-parameter-name guess from looping incorrectly), or the
    parsed total hit count (when available) has been reached.
    """
    docids: list[str] = []
    seen: set[str] = set()
    index = start_index

    while len(docids) < max_results:
        params = build_search_query_params(
            api_key, keyword, index, NLI_RESULTS_PER_PAGE
        )

        xml_text: str | None = None
        delay = 2.0
        for attempt in range(retries):
            try:
                response = session.get(NLI_SEARCH_URL, params=params, timeout=timeout)
                response.raise_for_status()
                xml_text = response.text
                break
            except Exception as exc:
                if is_permanent_http_error(exc) or attempt == retries - 1:
                    print(
                        f"[warn] search failed for keyword={keyword!r} "
                        f"index={index}: {exc}"
                    )
                    break
                time.sleep(retry_delay_seconds(exc, delay))
                delay = min(delay * 2, 30.0)

        if xml_text is None:
            break

        if debug_response_path is not None:
            try:
                debug_response_path.write_text(xml_text, encoding="utf-8")
            except Exception:
                pass

        page_docids, total_count = parse_search_response_xml(xml_text)
        new_docids = [d for d in page_docids if d not in seen]

        if not new_docids:
            break

        for docid in new_docids:
            seen.add(docid)
            docids.append(docid)

        if sleep_seconds:
            time.sleep(sleep_seconds)

        index += NLI_RESULTS_PER_PAGE
        already_fetched = index - start_index
        if total_count is not None and already_fetched >= total_count:
            break

    return [
        {"item_id": docid, "manifest_url": f"{NLI_IIIF_MANIFEST_BASE}/{docid}/manifest"}
        for docid in docids[:max_results]
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
    unchanged from every sibling pipeline. NLI's own documentation confirms
    Presentation API 2.1 specifically (top-level "sequences" - see module
    docstring point 5), so only that branch is expected to actually fire in
    practice here, but the Presentation-3 branch is kept anyway at zero
    cost, matching this project's standing defensive posture. Each returned
    page dict also carries canvas_width/canvas_height (the canvas's own
    declared dimensions, i.e. the SOURCE image size per the manifest - not
    necessarily what gets downloaded, see process_page's downloaded_width/
    height) when the manifest provides them.
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
                if isinstance(label, list):
                    label = label[0] if label else f"Page {canvas_index + 1}"

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


def metadata_values(value: Any) -> list[str]:
    """
    IIIF Presentation API 2.x's "metadata" values are plain strings OR
    arrays of plain strings per spec (no language-map shape - that's a v3
    feature RMDA/Waseda need, not needed here - see module docstring point
    5). Normalises either shape into a flat string list.
    """
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


# Best-effort metadata label vocabulary. UNLIKE every field in the rest of
# this module, the exact English label strings NLI's own manifests use for
# creator/date/subject/etc. could not be confirmed live (see module
# docstring point 4 - the same access gap that affects the search layer
# also meant no real manifest could be successfully fetched during
# development to read its metadata block directly). These are the most
# common Dublin-Core-ish label words this project's OTHER confirmed-live
# manifests (Bodleian/Gallica/NDL) use for the same concepts, kept here as a
# reasonable starting point - if books.jsonl comes back with these fields
# mostly empty, open one real manifest by hand
# (http://iiif.nli.org.il/IIIFv21/DOCID/<a real DOCID>/manifest) and adjust
# this dict to match what NLI actually emits.
METADATA_LABELS = {
    "author": ["Creator", "Author"],
    "date": ["Date", "Publication Date"],
    "publisher": ["Publisher"],
    "subject": ["Subject", "Keywords"],
    "notes": ["Notes", "Note", "Description"],
}


def extract_book_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """
    Best-effort bibliographic metadata extraction from a manifest's
    top-level "metadata" list of {"label": ..., "value": ...} pairs. Kept as
    manifest-only (not the search response) so metadata extraction behaves
    identically whether a book is processed fresh or resumed via
    sweep_paused_books() - exactly like every sibling pipeline. See
    METADATA_LABELS above for the caveat on label-name confidence.
    """
    by_label: dict[str, list[str]] = {}
    for entry in manifest.get("metadata", []) or []:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        if not isinstance(label, str):
            continue
        label = label.strip()
        if not label:
            continue

        cleaned = [strip_html(v) for v in metadata_values(entry.get("value")) if strip_html(v)]
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

    title = strip_html(str(manifest.get("label") or "")) or None
    attribution = strip_html(str(manifest.get("attribution") or "")) or None
    license_url = manifest.get("license") if isinstance(manifest.get("license"), str) else None
    rights_statement = " | ".join(p for p in (attribution, license_url) if p) or None

    return {
        "title": title,
        "author": ", ".join(all_values(*METADATA_LABELS["author"])) or None,
        "date": first(*METADATA_LABELS["date"]),
        "publisher": first(*METADATA_LABELS["publisher"]),
        "subject": ", ".join(all_values(*METADATA_LABELS["subject"])) or None,
        "notes": " ".join(all_values(*METADATA_LABELS["notes"])) or None,
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
            item_id = record.get("nli_docid")
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
        "nli_docid": item_id,
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
        "nli_docid": item_id,
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
    being re-discovered and re-attempted on every future run. Returns the
    final status assigned, for logging.
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
        print("[skip] could not determine NLI DOCID from search result")
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
            "source": SOURCE_NAME,
            "source_item_id": item_id,
            "nli_docid": item_id,
            "manifest_url": manifest_url,
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "date": metadata.get("date"),
            "publisher": metadata.get("publisher"),
            "subject": metadata.get("subject"),
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
    wall-clock time (transient server issues tend to clear up), not
    another immediate retry. Calling this between keyword searches (and
    once more at the end of the run) gives every paused book that gap
    naturally, instead of leaving it stuck until the same item happens to
    resurface under a later keyword's search results.

    A resumed book's manifest URL is reconstructed from its DOCID rather
    than replayed from the original search result (which isn't kept in
    state), since NLI's manifest URL is a deterministic function of the
    DOCID - like NDL/RMDA/Waseda.
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
            "manifest_url": f"{NLI_IIIF_MANIFEST_BASE}/{item_id}/manifest",
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

    if not args.api_key:
        print(
            "NLI_API_KEY is not set (and --api-key was not passed). An API "
            "key is REQUIRED - NLI's legacy keyless search API is blocked "
            "by Cloudflare for non-browser clients (confirmed live, see "
            "module docstring). Self-register for a free key at "
            "https://api2.nli.org.il/signup/ and set NLI_API_KEY.",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.state_path = args.output_dir / "processed_items.json"
    args.books_path = args.output_dir / "books.jsonl"
    log_path = args.output_dir / "page_log.jsonl"
    args.page_log_path = log_path
    args.run_metadata_path = args.output_dir / "run_metadata.json"
    args.last_state_upload_time = 0.0
    debug_response_path = args.output_dir / "debug_last_search_response.xml"

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

    state = load_state(args.state_path)
    books = load_books(args.books_path)
    session = make_session()

    keywords = args.keywords or DEFAULT_KEYWORDS
    print(f"Keywords ({len(keywords)}): {', '.join(keywords)}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Illustration class keywords: {args.illustration_class_keywords}")
    print(f"Output dir: {args.output_dir}")
    print(f"IIIF image width: {args.iiif_width} (NLI's own hard cap is 576px)")
    print(f"Negative audit sample rate: {args.negative_sample_rate}")
    print(f"Azure image prefix: {args.azure_prefix}")
    print(
        f"Azure state backup prefix: {args.state_azure_prefix} "
        f"(every {args.state_upload_interval:.0f}s)"
    )
    print(
        "[note] The search layer's exact response schema/pagination params "
        "could not be confirmed live for this pipeline (see module "
        f"docstring point 4) - check {debug_response_path} if a keyword "
        "that should have hits returns none."
    )

    for keyword in keywords:
        print(f"\n=== Keyword: {keyword!r} ===")
        try:
            search_results = search_nli_paginated(
                session=session,
                api_key=args.api_key,
                keyword=keyword,
                max_results=args.max_books_per_keyword,
                timeout=args.timeout,
                sleep_seconds=args.sleep,
                retries=args.retries,
                start_index=args.start_index,
                debug_response_path=debug_response_path,
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
