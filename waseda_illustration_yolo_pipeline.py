#!/usr/bin/env python3
"""
waseda_illustration_yolo_pipeline.py

Large-scale Waseda University Library Kotenseki Sogo Database (古典籍総合
データベース, "Classical Text Database") book-page harvesting with GPU YOLO
triage, searching for pages with RICH ILLUSTRATION. Sibling script to
bodleian_illustration_yolo_pipeline.py, gallica_illustration_yolo_pipeline.py,
mdz_illustration_yolo_pipeline.py, wellcome_illustration_yolo_pipeline.py,
ndl_illustration_yolo_pipeline.py, rmda_illustration_yolo_pipeline.py, and
loc_illustration_yolo_pipeline.py - same one-page-at-a-time GPU triage, Azure
upload strategy, and page_layout_best_new.pt model, pointed at Waseda's IIIF
API instead. Uses the exact same DEFAULT_KEYWORDS list as
ndl_illustration_yolo_pipeline.py/rmda_illustration_yolo_pipeline.py
(Japanese/Sino-Japanese terms), per user instruction.

IMPORTANT - keeping this separate from the other pipelines' data
--------------------------------------------------------------------------
Waseda catalogue item ids look like "bunko30_a0002" or "wa03_06306" (a
call-number-derived collection code + zero-padded number, occasionally with
a further "_NNNN" volume suffix - see "Catalogue items vs. IIIF volumes"
below) - a distinct namespace from every other library this project
searches, so there's no risk of a literal blob-name collision. Even so, this
script follows the exact same "<task>/<library>" file management plan laid
out in bodleian_illustration_yolo_pipeline.py, so every library's
illustration search lives in its own clearly labelled corner and nothing has
to be cross-checked by hand later:
  - --azure-prefix defaults to "illustrations/waseda" (siblings:
    "illustrations/bodleian_new", "illustrations/gallica",
    "illustrations/mdz", "illustrations/wellcome", "illustrations/ndl",
    "illustrations/rmda", "illustrations/loc"),
  - --negative-audit-prefix defaults to
    "illustrations/waseda/negative_audit",
  - --output-dir defaults to "waseda_illustration_yolo_run" (siblings:
    "bodleian_illustration_yolo_run", "gallica_illustration_yolo_run",
    "mdz_illustration_yolo_run", "wellcome_illustration_yolo_run",
    "ndl_illustration_yolo_run", "rmda_illustration_yolo_run",
    "loc_illustration_yolo_run"), so local state/temp files never collide
    either,
  - --state-azure-prefix defaults to
    "state_backups/illustration_runs/waseda_illustration_yolo_run", alongside
    the other pipelines' own prefixes under the same "illustration_runs/"
    umbrella.
Because the output directories, Azure prefixes, and local temp/tmux concerns
are all fully separate, this script is safe to run at the same time as the
other seven illustration pipelines in their own tmux sessions on the same
RunPod pod - see "Running all eight illustration pipelines at once" below.

API reference - IMPORTANT CAVEAT: the referenced page has almost no
technical detail, and had to be reverse-engineered live
-------------------------------------------------------------------------
https://www.wul.waseda.ac.jp/kotenseki/ga_IIIF/about_iiif.html (the page the
user pointed at) is a user-facing page explaining that "some" items support
IIIF viewing via Universal Viewer/Mirador - it documents NO base URL, NO
manifest URL pattern, NO search API, and NO rate-limit/usage policy at all.
Everything below was confirmed LIVE against
https://www.wul.waseda.ac.jp/kotenseki/ during development (2026-08-29):

1. There is no documented search API of any kind (no OAI-PMH/SRU/JSON:API,
   like RMDA - see rmda_illustration_yolo_pipeline.py's own docstring for
   the sibling precedent). The database's own search form
   (https://www.wul.waseda.ac.jp/kotenseki/search.php) POSTs
   application/x-www-form-urlencoded fields directly to itself and returns a
   server-rendered HTML results page:
       POST https://www.wul.waseda.ac.jp/kotenseki/search.php
       cndbn=<query>&cndiiif=<""|"1">&szlmt=<10|30|50|100|500>&page_no=<1-based>
   `cndbn` is the free-text "検索語" field (matches title/author/imprint/
   keyword together, confirmed live: the same wording that appears only in
   a book's "キーワード（主題）" column still matches on cndbn alone) - the
   same style of "search everything" field this project's other pipelines
   use (NDL's `anywhere`, RMDA's `keys`). `cndiiif=1` restricts results to
   IIIF-available items only; deliberately LEFT BLANK by this pipeline (see
   "IIIF availability is scarce" below for why). This pipeline scrapes the
   results HTML for `<A HREF=".../kotenseki/html/<collection>/<item_id>/
   index.html">` links (see parse_search_results_html()) and the
   "ヒット件数：<n>件" hit-count text, exactly like RMDA's approach - equally
   fragile to a front-end template change, for the same reasons documented
   in RMDA's docstring.
2. CONFIRMED LIVE GOTCHA: every page this pipeline scrapes (search.php's
   results AND every item detail page) sends `Content-Type: text/html` with
   NO charset parameter, even though the page's own `<META>` tag declares
   `charset=utf-8` and the content genuinely IS UTF-8. `requests` therefore
   defaults to decoding the response body as ISO-8859-1 (this is standard
   HTTP behaviour, not a `requests` bug) unless told otherwise, producing
   silent mojibake instead of an error - confirmed live: `response.text`
   comes back as `[æºæ°ç©èª]` instead of `[源氏物語]`. Every text (non-JSON,
   non-binary-image) response from wul.waseda.ac.jp is therefore explicitly
   re-decoded as UTF-8 in this pipeline (see fetch_text_with_retry()) rather
   than trusting `requests`' auto-detected encoding.
3. Catalogue items vs. IIIF volumes (the part with no sibling precedent):
   a search hit's item id (e.g. "bunko30_a0002") is a CATALOGUE record, not
   necessarily a single IIIF-viewable unit. Its detail page
   (https://www.wul.waseda.ac.jp/kotenseki/html/<collection>/<item_id>/
   index.html) is scraped for every distinct
   `https://iiif.archive.waseda.jp/iiif/manifest/ktnsk/<volume_id>/
   manifest.json` link on it (see discover_manifest_urls()). Confirmed live
   two shapes exist:
     - Single-volume: the catalogue item id and the volume id are the SAME
       string (e.g. "chi13_04443_0001" - itself already carries a volume
       suffix because that particular work was catalogued one record per
       volume) - exactly one manifest link, matching the item id.
     - Multi-volume: ONE catalogue item (e.g. "bunko30_a0002", Waseda's copy
       of the Tale of Genji, 55 physical volumes) lists MANY manifest links
       on its single detail page, one per volume, each volume id built as
       "<item_id>_<NNNN>" (e.g. "bunko30_a0002_0001" .. "..._0055").
   Each discovered volume id is therefore treated as one independent "book"
   for this pipeline's page-by-page processing - the same "one manifest = one
   book" unit every sibling pipeline uses - while the catalogue item itself
   gets a lightweight bookkeeping entry in processed_items.json (kind
   "catalog_item") recording which volume ids it expanded into, so a
   catalogue item matched by several different keywords across a run is only
   ever scraped for its manifest list ONCE. See process_catalog_item() and
   the "kind" field on state dict entries.
4. IIIF manifest URL for a KNOWN volume id is fully deterministic (confirmed
   live), exactly like NDL/RMDA:
       https://iiif.archive.waseda.jp/iiif/manifest/ktnsk/<volume_id>/manifest.json
   - so once a volume id has been discovered once (via its catalogue item's
   detail page), sweep_paused_books() can reconstruct its manifest URL
   directly on resume without re-scraping the detail page.
5. IIIF Presentation API version is MIXED per volume, confirmed live against
   two real manifests during development - some are Presentation API 2.0
   (top-level "sequences", plain-string metadata label/value pairs, e.g.
   bunko30_a0002_0001), others are Presentation API 3.0 (top-level "items",
   IIIF-native language-map label/value pairs, e.g. chi13_04443_0001) - the
   same "mixed v2/v3 in the wild" situation documented in
   rmda_illustration_yolo_pipeline.py's docstring, for which this pipeline
   reuses the exact same version-agnostic parse_iiif_manifest() walk AND
   RMDA's language_map_values() (transparently passes plain strings through
   unchanged, so it works for both manifest generations without a branch).
   Image API profile is level1 (older/v2 volumes, undeclared-but-working
   sizeByW - like NDL) or level2 (newer/v3 volumes, sizeByW declared) -
   either way a plain width-only "/full/<width>,/0/default.jpg" request
   works, confirmed live, so --iiif-width behaves like every sibling
   pipeline's equivalent flag.
6. Manifest metadata label vocabulary is sparse and was confirmed live to
   differ slightly between the two manifest generations noted above -
   "Publisher"(v2)/"Imprint"(v3) for the imprint statement, "Publication
   Date"(v2)/"Publication"(v3) for the date, and "Descriptopn" - Waseda's own
   typo, confirmed live and left exactly as their system emits it -
   (v2)/"Description"(v3) for the physical-description field; a trailing
   space on the "Notes " label (v2) is stripped before matching. See
   METADATA_LABELS/extract_book_metadata(). No separate publisher-name,
   ISBN, call-number, or subject-heading fields are exposed in the manifest
   itself (a book's call number and keyword/subject terms are only visible
   on its catalogue detail page HTML, not carried into books.jsonl by this
   pipeline, matching the "manifest-only metadata" discipline every sibling
   pipeline follows for resume-consistency). Rights/attribution provenance
   is likewise generation-dependent: v2 volumes carry plain "attribution"/
   "license" strings; v3 volumes carry "rights" (a license URL, e.g.
   Creative Commons BY-NC-SA 4.0 for chi13_04443_0001) and a
   "requiredStatement" pointing at Waseda's own reuse policy
   (https://www.waseda.jp/library/user/using-images/) - all four are
   combined into one rights_statement string per book (see
   extract_book_metadata()) so provenance travels with the data regardless
   of which manifest generation produced it.
7. IIIF AVAILABILITY IS SCARCE - CONFIRMED LIVE, READ BEFORE EXPECTING A
   LARGE HARVEST: searching with `cndiiif=1` and an EMPTY keyword returned
   only 90 volume records in Waseda's ENTIRE kotenseki catalogue (out of
   roughly 300,000 items) as of 2026-08-29 - and most of those 90 are the 55
   Genji Monogatari volumes under one catalogue item plus the 65 volumes of
   one Chinese woodblock-print set, i.e. under a dozen distinct catalogued
   works have any IIIF volumes at all right now. Combining `cndiiif=1` with
   this pipeline's own keyword list live-tested at 0-4 hits per keyword (293
   hits for "図会" drop to 0 the instant `cndiiif=1` is added). Because of
   this, --keywords is searched WITHOUT the cndiiif filter (an "anywhere"
   style search across the whole catalogue, matching the other pipelines'
   search posture) and IIIF availability is instead checked per matched item
   by scraping its detail page for manifest links (see point 3 above); an
   item with none is recorded as a terminal "skipped_no_iiif" catalogue
   entry (see books.jsonl) rather than an error. Expect the overwhelming
   majority of this pipeline's keyword hits to end up skipped_no_iiif given
   Waseda's IIIF rollout is evidently still in an early/pilot stage - this is
   an honest reflection of the live API's current state, not a bug in this
   pipeline, and the pipeline will pick up any future additions to Waseda's
   IIIF-enabled set automatically on a later run with no code changes.

Default YOLO model
-------------------
Defaults to ./page_layout_best_new.pt (the same fine-tuned model as the
other seven illustration pipelines' default), classes:
    illustration, text_block
A page is treated as "contains a rich illustration" when any detected class
name contains "illustration" (see --illustration-class-keywords).

Workflow
--------
1. Search Waseda for each keyword in KEYWORDS (Japanese/Sino-Japanese terms -
   the exact same DEFAULT_KEYWORDS list as ndl_illustration_yolo_pipeline.py/
   rmda_illustration_yolo_pipeline.py, per user instruction) by POSTing to
   search.php and scraping the paginated results HTML (see API reference
   above).
2. For each NEW catalogue item id, fetch its detail page HTML once and
   scrape every distinct IIIF manifest link on it (see "Catalogue items vs.
   IIIF volumes" above). Zero links -> record "skipped_no_iiif" and move on.
   One or more links -> each becomes an independent volume/book, and the
   catalogue item's own bookkeeping entry records the volume id list so it
   is never re-scraped by a later keyword match.
3. For each volume, fetch its IIIF manifest (URL built deterministically
   from the volume id) and enumerate every page/canvas.
4. Skip the whole volume if it has fewer than --min-pages-per-book pages
   (default 5) - recorded as "skipped_too_short", never rechecked.
5. Otherwise walk every page. If the volume has MORE than
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
6. Progress is checkpointed after every page to processed_items.json,
   resuming a killed run mid-volume. A page-level error pauses the volume
   (does NOT advance next_page_index) for retry on the next run, up to
   --max-page-retries attempts, before the volume is marked
   "failed_permanent".
7. Every page decision (kept/deleted/skipped/error) is appended to
   page_log.jsonl with full provenance: generic source/source_item_id
   fields alongside the library-specific waseda_volume_id/catalog_item_id
   fields, the manifest URL, canvas id/label, page index/number, the IIIF
   image service URL, the exact requested image URL, the downloaded image's
   actual pixel dimensions (read from YOLO's own decode - never assumed to
   equal the requested IIIF width), the source canvas's width/height from
   the manifest when available, the blob name when uploaded, and - for
   illustration-negative pages - whether the page was selected for the
   negative audit sample and the deterministic sampling value used.
8. Volume-level bibliographic metadata, the manifest's rights/attribution,
   page counts, illustration-detection totals, and an illustration_density
   ratio are written to books.jsonl, one line per volume (plus one line per
   "skipped_no_iiif" catalogue item, for visibility into what was checked
   and rejected).
9. processed_items.json/books.jsonl/page_log.jsonl/run_metadata.json are
   ALSO pushed to Azure under --state-azure-prefix every
   --state-upload-interval seconds (and once more at the end of the run),
   each upload overwriting the previous one - a RunPod pod can die without
   warning.
10. run_metadata.json is written once, the first time --output-dir is used,
    and never regenerated on a resumed run (even with different CLI flags) -
    it records which YOLO model (path + SHA-256 of the weights file), class
    names, imgsz, confidence threshold, illustration keywords, and
    Python/PyTorch/Ultralytics/CUDA versions actually produced the corpus in
    that directory. See load_or_create_run_metadata().

Be a good citizen (there is no API to be a citizen of - be a good guest)
-----------------------------------------------------------------------------
Because search here means scraping HTML meant for a browser (see above),
this pipeline defaults to a MORE conservative --sleep than most of its
siblings (matching RMDA's posture), and sets an identifying User-Agent via
WASEDA_CONTACT_EMAIL (see below) purely so a research contact is visible if
anyone at Waseda University Library ever looks at their access logs. Every
volume's manifest carries Waseda's own reuse policy URL
(https://www.waseda.jp/library/user/using-images/) into books.jsonl's
rights_statement field so provenance - and which reuse terms apply - travels
with the data.

Environment
-----------
Credentials are read from environment variables (never hardcode them in
this file, since it goes to GitHub). Put them in a local `.env` (gitignored)
or set them as RunPod pod environment variables / secrets.

    WASEDA_CONTACT_EMAIL="you@example.org"   # optional but good practice

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
This script and the other seven illustration pipelines write to fully
separate local directories and Azure prefixes (see "IMPORTANT" above), so
it's safe to run this in its own tmux session alongside all the others on
the same pod:

    tmux new -s waseda_illustration
    conda activate botany_yolo   # or whatever env has torch/ultralytics
    python waseda_illustration_yolo_pipeline.py
    # detach: Ctrl-b d

    tmux attach -t waseda_illustration   # reattach to check on it
    tmux ls                              # list all sessions

This script defaults --device to the same auto-detected cuda:0 the other
pipelines use, so on a single-GPU pod every concurrently-running
illustration pipeline shares that one GPU's compute/VRAM (they'll interleave
GPU time rather than truly run in parallel), but if you have a multi-GPU
pod, pass --device cuda:1/cuda:2/... to spread them out.

Example
-------
python waseda_illustration_yolo_pipeline.py \
    --azure-container botany-pages \
    --max-books-per-keyword 100 \
    --output-dir waseda_illustration_yolo_run

Dry run (no Azure credentials needed, still deletes local files):
python waseda_illustration_yolo_pipeline.py --dry-run --max-books-per-keyword 2
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

WASEDA_BASE = "https://www.wul.waseda.ac.jp/kotenseki"
WASEDA_SEARCH_URL = f"{WASEDA_BASE}/search.php"
WASEDA_IIIF_MANIFEST_BASE = "https://iiif.archive.waseda.jp/iiif/manifest/ktnsk"
SCRIPT_DIR = Path(__file__).resolve().parent

ITEM_LINK_RE = re.compile(
    r'href="(https?://(?:www\.)?wul\.waseda\.ac\.jp/kotenseki/html/'
    r'([A-Za-z0-9]+)/([A-Za-z0-9_]+)/index\.html)"',
    re.IGNORECASE,
)
RESULT_COUNT_RE = re.compile(
    r"ヒット件数：</FONT><FONT COLOR=\"red\" SIZE=\"3\">([\d,]+)"
)
MANIFEST_LINK_RE = re.compile(
    r"https://iiif\.archive\.waseda\.jp/iiif/manifest/ktnsk/"
    r"([A-Za-z0-9_]+)/manifest\.json"
)

# Generic source identifier used in page_log.jsonl/run_metadata.json so
# records from different library pipelines can eventually be pooled and
# distinguished by a single consistent field name.
SOURCE_NAME = "waseda"
SOURCE_DISPLAY_NAME = (
    "Waseda University Library Kotenseki Sogo Database "
    "(Classical Text Database)"
)

_contact = os.environ.get("WASEDA_CONTACT_EMAIL", "").strip()
USER_AGENT = (
    "Phyto-Vision-Waseda-Illustration-YOLO-Pipeline/1.0 "
    f"(research; contact: {_contact})"
    if _contact
    else "Phyto-Vision-Waseda-Illustration-YOLO-Pipeline/1.0 "
    "(research; set WASEDA_CONTACT_EMAIL for a contact address)"
)

# Same keyword list as ndl_illustration_yolo_pipeline.py/
# rmda_illustration_yolo_pipeline.py, per user instruction.
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

DEFAULT_YOLO_MODEL = str(SCRIPT_DIR / "page_layout_best_new.pt")
DEFAULT_IMGSZ = 640
DEFAULT_CONF_THRESHOLD = 0.30
DEFAULT_ILLUSTRATION_CLASS_KEYWORDS = ["illustration"]

DEFAULT_MAX_BOOKS_PER_KEYWORD = 100
WASEDA_RESULTS_PER_PAGE = 500  # the largest "表示" option search.php offers, confirmed live.
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 5
DEFAULT_MAX_PAGE_RETRIES = 5
DEFAULT_MAX_BOOK_RETRIES = 3
DEFAULT_MIN_PAGES_PER_BOOK = 5
DEFAULT_EDGE_SKIP_THRESHOLD = 20
DEFAULT_EDGE_SKIP_COUNT = 5
DEFAULT_STATE_AZURE_PREFIX = "state_backups/illustration_runs/waseda_illustration_yolo_run"
DEFAULT_STATE_UPLOAD_INTERVAL = 3000.0
DEFAULT_IIIF_WIDTH = 1200
DEFAULT_NEGATIVE_SAMPLE_RATE = 0.02

# Terminal statuses for a volume/book-level state entry - once reached, the
# volume is never reprocessed, only its keywords_matched list is extended if
# a later keyword search happens to match its catalogue item again.
BOOK_TERMINAL_STATUSES = ("completed", "failed_permanent", "skipped_too_short")


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
            "Search the Waseda University Library Kotenseki Sogo Database "
            "(Classical Text Database) for illustration-rich books, triage "
            "every page with a GPU YOLO model, and stream "
            "illustration-positive pages to Azure Blob Storage."
        )
    )

    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Override the built-in (Japanese/Sino-Japanese) keyword list "
        "(same default list as ndl_illustration_yolo_pipeline.py/"
        "rmda_illustration_yolo_pipeline.py).",
    )
    parser.add_argument(
        "--max-books-per-keyword",
        type=int,
        default=DEFAULT_MAX_BOOKS_PER_KEYWORD,
        help="Maximum number of Waseda search results (catalogue items, "
        "before IIIF-volume expansion) to fetch per keyword.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Search result page to start from for each keyword (1-based, "
        f"matching search.php's own `page_no` convention; "
        f"{WASEDA_RESULTS_PER_PAGE} results per page).",
    )
    parser.add_argument(
        "--limit-pages-per-book",
        type=int,
        default=0,
        help="Cap on pages processed per volume. 0 means process every page.",
    )
    parser.add_argument(
        "--min-pages-per-book",
        type=int,
        default=DEFAULT_MIN_PAGES_PER_BOOK,
        help="Volumes with fewer than this many pages are skipped entirely "
        "(recorded as status 'skipped_too_short', no pages downloaded or "
        f"run through YOLO). Default: {DEFAULT_MIN_PAGES_PER_BOOK}",
    )
    parser.add_argument(
        "--edge-skip-threshold",
        type=int,
        default=DEFAULT_EDGE_SKIP_THRESHOLD,
        help="Volumes with more than this many pages have their leading/"
        "trailing pages (see --edge-skip-count) skipped rather than "
        f"downloaded and run through YOLO. Default: {DEFAULT_EDGE_SKIP_THRESHOLD}",
    )
    parser.add_argument(
        "--edge-skip-count",
        type=int,
        default=DEFAULT_EDGE_SKIP_COUNT,
        help="Number of pages to skip at the start AND at the end of a "
        "volume whose page count exceeds --edge-skip-threshold. Skipped "
        "pages still get an entry in page_log.jsonl (action "
        "'skipped_edge_page') recording their IIIF image URL, but are never "
        f"downloaded or run through YOLO. Default: {DEFAULT_EDGE_SKIP_COUNT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("waseda_illustration_yolo_run"),
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
        default="illustrations/waseda",
        help="Blob name prefix for kept page images. Defaults to "
        "'illustrations/waseda' (NOT the container root), matching the "
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
        default="illustrations/waseda/negative_audit",
        help="Blob prefix for the negative audit sample. Default: "
        "illustrations/waseda/negative_audit",
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
        "without cloud credentials.",
    )

    # HTTP
    parser.add_argument(
        "--iiif-width",
        type=int,
        default=DEFAULT_IIIF_WIDTH,
        help="Requested IIIF image width in pixels (0 = full resolution). "
        f"Default: {DEFAULT_IIIF_WIDTH}.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds for Waseda search/detail-page/IIIF "
        "requests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Delay after each successful search/detail-page/manifest/image "
        "request. Kept more conservative than this project's IIIF-API-"
        "backed pipelines since Waseda has no documented search API at all "
        "and search means scraping browser-facing HTML - see module "
        "docstring.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Maximum HTTP retries per request (search, detail-page, "
        "manifest, and image downloads all use the same exponential "
        "backoff, honouring a Retry-After header on 429s).",
    )
    parser.add_argument(
        "--max-page-retries",
        type=int,
        default=DEFAULT_MAX_PAGE_RETRIES,
        help="Maximum times a single page is retried across separate runs "
        "before its volume is marked failed_permanent and skipped like a "
        f"completed volume. Default: {DEFAULT_MAX_PAGE_RETRIES}",
    )
    parser.add_argument(
        "--max-book-retries",
        type=int,
        default=DEFAULT_MAX_BOOK_RETRIES,
        help="Maximum times a volume-level or catalogue-item-level failure "
        "(manifest/detail-page couldn't be fetched, no pages could be "
        "resolved, etc. - as opposed to a page-level failure, see "
        "--max-page-retries) is retried across separate runs before being "
        "marked failed_permanent and stops being re-discovered by future "
        f"searches. Default: {DEFAULT_MAX_BOOK_RETRIES}",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------
# Waseda search (HTML scrape) / IIIF helpers
# --------------------------------------------------------------------------


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/json,image/jpeg,image/png,image/*,*/*;q=0.8",
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


def fetch_text_with_retry(
    session: requests.Session,
    url: str,
    timeout: int,
    sleep_seconds: float,
    retries: int,
    *,
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> str:
    """
    GET or POST a wul.waseda.ac.jp page and return its body as text, ALWAYS
    forced to UTF-8 decoding - see module docstring's "CONFIRMED LIVE
    GOTCHA" section: every page this pipeline scrapes sends
    `Content-Type: text/html` with no charset parameter even though the
    content is genuinely UTF-8, so `requests`' default ISO-8859-1 fallback
    would otherwise silently mojibake every Japanese character.
    """
    delay = 2.0

    for attempt in range(retries):
        try:
            if method == "POST":
                response = session.post(url, data=data, timeout=timeout)
            else:
                response = session.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = "utf-8"
            text = response.text
            if sleep_seconds:
                time.sleep(sleep_seconds)
            return text
        except Exception as exc:
            if is_permanent_http_error(exc) or attempt == retries - 1:
                raise
            time.sleep(retry_delay_seconds(exc, delay))
            delay = min(delay * 2, 30.0)

    raise RuntimeError("Unreachable text retry state.")


def parse_search_results_html(html: str) -> tuple[list[dict[str, Any]], int | None]:
    """
    Extract catalogue item results (in first-seen order, de-duplicated by
    item id) and the total hit count from one Waseda search-results HTML
    page - see module docstring's API-reference section for why this is a
    scrape rather than a JSON API call. Returns (results, total_count) where
    each result is {"item_id": ..., "detail_url": ...}; total_count is None
    if the "ヒット件数：N件" text couldn't be found (treated as "unknown,
    keep paging until a page comes back empty").
    """
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for match in ITEM_LINK_RE.finditer(html):
        detail_url, _collection, item_id = match.group(1), match.group(2), match.group(3)
        if item_id in seen:
            continue
        seen.add(item_id)
        # Confirmed live: http:// links from search results 301-redirect to
        # the identical https:// path - upgrade the scheme up front to save
        # that extra round trip on every detail-page fetch.
        if detail_url.startswith("http://"):
            detail_url = "https://" + detail_url[len("http://") :]
        results.append({"item_id": item_id, "detail_url": detail_url})

    total_count: int | None = None
    count_match = RESULT_COUNT_RE.search(html)
    if count_match:
        try:
            total_count = int(count_match.group(1).replace(",", ""))
        except ValueError:
            total_count = None

    return results, total_count


def search_waseda_paginated(
    session: requests.Session,
    keyword: str,
    max_results: int,
    timeout: int,
    sleep_seconds: float,
    retries: int = DEFAULT_RETRIES,
    start_page: int = 1,
) -> list[dict[str, Any]]:
    """
    Pages through Waseda's search-results HTML (1-based `page_no`, fixed
    WASEDA_RESULTS_PER_PAGE per page - see module docstring). `cndiiif` is
    deliberately left blank (searches the WHOLE catalogue, not just
    IIIF-flagged items - see "IIIF availability is scarce" in the module
    docstring for why). Stops once max_results item ids are collected, a
    page yields no NEW item ids (covers both "ran out of results" and
    "template changed and broke parsing" - either way there's nothing
    useful left to do), or the parsed total hit count has been reached.
    """
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    page_no = start_page
    total_count: int | None = None

    while len(results) < max_results:
        data = {
            "cndbn": keyword,
            "cndiiif": "",
            "szlmt": str(WASEDA_RESULTS_PER_PAGE),
            "page_no": str(page_no),
        }

        html: str | None = None
        try:
            html = fetch_text_with_retry(
                session,
                WASEDA_SEARCH_URL,
                timeout=timeout,
                sleep_seconds=0.0,  # sleep once below, after a successful page
                retries=retries,
                method="POST",
                data=data,
            )
        except Exception as exc:
            print(
                f"[warn] search failed for keyword={keyword!r} "
                f"page_no={page_no}: {exc}"
            )
            break

        page_results, page_total = parse_search_results_html(html)
        if page_total is not None:
            total_count = page_total

        new_results = [r for r in page_results if r["item_id"] not in seen]
        if not new_results:
            break

        for result in new_results:
            seen.add(result["item_id"])
            results.append(result)

        if sleep_seconds:
            time.sleep(sleep_seconds)

        page_no += 1
        already_fetched = (page_no - start_page) * WASEDA_RESULTS_PER_PAGE
        if total_count is not None and already_fetched >= total_count:
            break

    return results[:max_results]


def discover_manifest_urls(detail_html: str) -> list[tuple[str, str]]:
    """
    Scrape every distinct IIIF manifest link out of one catalogue item's
    detail page HTML - see module docstring's "Catalogue items vs. IIIF
    volumes" section. Returns a list of (volume_id, manifest_url) pairs, in
    first-seen order, de-duplicated by volume id (each volume's manifest
    link legitimately appears twice on the page - once as the "IIIF
    Image(UV)" viewer link, once as the "Manifest" link - both pointing at
    the same manifest.json URL).
    """
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for match in MANIFEST_LINK_RE.finditer(detail_html):
        volume_id = match.group(1)
        if volume_id in seen:
            continue
        seen.add(volume_id)
        pairs.append((volume_id, match.group(0)))
    return pairs


def manifest_url_for_volume(volume_id: str) -> str:
    """
    IIIF manifest URL for a KNOWN volume id is fully deterministic - see
    module docstring point 4 - so a resumed/swept volume can reconstruct
    this directly without re-scraping its catalogue item's detail page.
    """
    return f"{WASEDA_IIIF_MANIFEST_BASE}/{volume_id}/manifest.json"


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
    from the other six illustration pipelines. Confirmed live that Waseda
    manifests are a MIX of Presentation 2 (top-level "sequences") and
    Presentation 3 (top-level "items") depending on volume - see module
    docstring point 5 - so this pipeline is the second in the family (after
    RMDA) to actually exercise the v3 branch in practice. Each returned page
    dict also carries canvas_width/canvas_height (the canvas's own declared
    dimensions, i.e. the SOURCE image size per the manifest - not
    necessarily what gets downloaded, see process_page's downloaded_width/
    height) when the manifest provides them, which Waseda's do.
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


def language_map_values(value: Any, prefer_language: str = "ja") -> list[str]:
    """
    Waseda's Presentation-3 manifests use the native IIIF language-map shape
    for label/value text - {"ja": ["..."], "en": ["..."]} - while its
    Presentation-2 manifests use plain strings (see module docstring point
    5). Ported from rmda_illustration_yolo_pipeline.py: returns the string
    list for the preferred language, falling back to English, then "none"
    (IIIF's own placeholder for language-unspecified text), then whichever
    language happens to be present first. Passes plain strings/lists through
    UNCHANGED, which is exactly what lets one extract_book_metadata() below
    handle both manifest generations without a version branch.
    """
    if isinstance(value, str):
        return [value] if value else []

    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]

    if isinstance(value, dict):
        for lang in (prefer_language, "en", "none"):
            texts = value.get(lang)
            if isinstance(texts, list) and texts:
                return [t for t in texts if isinstance(t, str)]
        for texts in value.values():
            if isinstance(texts, list):
                found = [t for t in texts if isinstance(t, str)]
                if found:
                    return found

    return []


# Waseda manifest metadata label vocabulary - confirmed live to differ
# between the two manifest generations (see module docstring point 6):
# "Publisher"/"Publication Date"/"Descriptopn" (Waseda's own typo, kept
# verbatim) appear on Presentation-2 volumes, "Imprint"/"Publication"/
# "Description" on Presentation-3 volumes. A trailing space on the "Notes "
# label (Presentation-2) is stripped before this lookup, so a single "Notes"
# entry below covers both.
METADATA_LABELS = {
    "author": ["Creator"],
    "imprint": ["Publisher", "Imprint"],
    "date": ["Publication Date", "Publication"],
    "physical_description": ["Descriptopn", "Description"],
    "notes": ["Notes"],
}


def first_dict_or_list_item(value: Any) -> dict[str, Any] | None:
    """requiredStatement/provider are sometimes a single object, sometimes a
    single-element list of one, in the manifests seen live - normalise."""
    if isinstance(value, list):
        return value[0] if value and isinstance(value[0], dict) else None
    if isinstance(value, dict):
        return value
    return None


def extract_book_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """
    Best-effort bibliographic metadata extraction from a manifest's
    top-level "metadata" list of {"label": ..., "value": ...} pairs (either
    plain strings or IIIF v3 language maps - see language_map_values()).
    Kept as manifest-only (not the catalogue detail-page HTML) so metadata
    extraction behaves identically whether a volume is processed fresh or
    resumed via sweep_paused_books() - exactly like every sibling pipeline.

    rights_statement combines every rights/attribution signal confirmed
    live across both manifest generations (see module docstring point 6):
    Presentation-2's plain "attribution"/"license" strings, and
    Presentation-3's "requiredStatement" (Waseda's reuse-policy text) /
    "provider" (publisher agent name) / "rights" (a license URL, e.g.
    Creative Commons).
    """
    by_label: dict[str, list[str]] = {}
    for entry in manifest.get("metadata", []) or []:
        if not isinstance(entry, dict):
            continue
        label_texts = language_map_values(entry.get("label"))
        label = label_texts[0].strip() if label_texts else None
        if not label:
            continue

        values = language_map_values(entry.get("value"))
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

    title = strip_html(" ".join(language_map_values(manifest.get("label")))) or None

    attribution = strip_html(str(manifest.get("attribution") or "")) or None

    required_statement = first_dict_or_list_item(manifest.get("requiredStatement"))
    required_statement_text = None
    if required_statement is not None:
        required_statement_text = (
            strip_html(" ".join(language_map_values(required_statement.get("value"))))
            or None
        )

    provider = first_dict_or_list_item(manifest.get("provider"))
    provider_label = None
    if provider is not None:
        provider_label = (
            strip_html(" ".join(language_map_values(provider.get("label")))) or None
        )

    license_url = manifest.get("license") if isinstance(manifest.get("license"), str) else None
    rights_url = manifest.get("rights") if isinstance(manifest.get("rights"), str) else None

    rights_statement = (
        " | ".join(
            p
            for p in (attribution, required_statement_text, provider_label, license_url, rights_url)
            if p
        )
        or None
    )

    return {
        "title": title,
        "author": first(*METADATA_LABELS["author"]),
        "imprint": first(*METADATA_LABELS["imprint"]),
        "date": first(*METADATA_LABELS["date"]),
        "physical_description": first(*METADATA_LABELS["physical_description"]),
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
    count toward a page/volume's illustration signal.
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
    catalogue item so a long run doesn't spend all its time re-uploading the
    (growing) page_log.jsonl. --state-upload-interval controls the minimum
    gap between uploads; `force=True` (used at the very end of main())
    bypasses it for a final guaranteed backup.
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
            item_id = record.get("source_item_id")
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
# Per-page / per-volume pipeline
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
    volume_id: str,
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
        # workflow step 7) - the same field names other library pipelines
        # in this project also write, so records can eventually be pooled.
        "source": SOURCE_NAME,
        "source_item_id": volume_id,
        "waseda_volume_id": volume_id,
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
            blob_name = build_blob_name(args.azure_prefix, volume_id, page_index)

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
                SOURCE_NAME, volume_id, page_identifier
            )
            is_selected = sample_value < args.negative_sample_rate
            record["negative_audit_sample_value"] = round(sample_value, 8)
            record["negative_audit_selected"] = is_selected

            if is_selected:
                blob_name = build_blob_name(
                    args.negative_audit_prefix, volume_id, page_index
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
    volume_id: str,
    keyword: str,
    manifest_url: str | None,
    page: dict[str, Any],
    page_index: int,
    log_path: Path,
) -> dict[str, Any]:
    """
    Record a leading/trailing page of a long volume WITHOUT downloading it
    or running YOLO - see --edge-skip-threshold/--edge-skip-count. The
    page's IIIF image URL and position are still written to page_log.jsonl
    so its existence is never lost, it's just never classified (so
    downloaded_width/height and the negative-audit fields stay null - there
    was no download and no detection to base them on).
    """
    record: dict[str, Any] = {
        "source": SOURCE_NAME,
        "source_item_id": volume_id,
        "waseda_volume_id": volume_id,
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


def record_failure(
    state: dict[str, Any],
    entry_id: str,
    error: str,
    keyword: str,
    max_book_retries: int,
    *,
    extra_fields: dict[str, Any] | None = None,
) -> str:
    """
    Shared bookkeeping for a volume-level OR catalogue-item-level failure -
    a manifest/detail-page couldn't be fetched, no pages could be resolved
    from a manifest, or any other exception happened before/instead of the
    per-page loop (as opposed to a page-level failure, which pauses a volume
    "in_progress" and is capped by a separate mechanism - see
    --max-page-retries). Tracks book_retry_count across separate runs the
    same way page-level retries are tracked: once max_book_retries is
    reached, the entry is marked "failed_permanent" (a terminal status, like
    "completed") so it stops being re-discovered and re-attempted on every
    future run. Without this cap, a search result whose manifest/detail page
    is permanently missing or broken on Waseda's end would be retried
    forever, since a plain "failed" status is NOT terminal. Returns the
    final status assigned, for logging.
    """
    previous = state.get(entry_id) or {}
    keywords_matched = set(previous.get("keywords_matched", []))
    keywords_matched.add(keyword)
    retry_count = previous.get("book_retry_count", 0) + 1
    status = "failed_permanent" if retry_count >= max_book_retries else "failed"

    state[entry_id] = {
        **previous,
        **(extra_fields or {}),
        "status": status,
        "error": error,
        "book_retry_count": retry_count,
        "keywords_matched": sorted(keywords_matched),
    }
    return status


def process_book(
    *,
    volume_id: str,
    manifest_url: str,
    catalog_item_id: str,
    catalog_url: str | None,
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
    volume_state = state.get(volume_id)

    if volume_state and volume_state.get("status") in BOOK_TERMINAL_STATUSES:
        keywords_matched = volume_state.setdefault("keywords_matched", [])
        if keyword not in keywords_matched:
            keywords_matched.append(keyword)
            save_state_atomic(args.state_path, state)
            book_record = books.get(volume_id)
            if book_record is not None:
                book_record["keywords_matched"] = keywords_matched
                save_books_atomic(args.books_path, books)
        print(f"[skip] {volume_id} already {volume_state.get('status')}")
        return

    print(f"\n[volume] {volume_id} (catalog_item={catalog_item_id!r}, keyword={keyword!r})")

    metadata: dict[str, Any] = {}
    total_pages = volume_state.get("page_count", 0) if volume_state else 0

    def flush_book_record(status: str) -> None:
        current = state.get(volume_id, {})
        pages_kept = current.get("pages_kept", 0)
        books[volume_id] = {
            "source": SOURCE_NAME,
            "source_item_id": volume_id,
            "waseda_volume_id": volume_id,
            "catalog_item_id": catalog_item_id,
            "catalog_url": catalog_url,
            "manifest_url": manifest_url,
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "imprint": metadata.get("imprint"),
            "date": metadata.get("date"),
            "physical_description": metadata.get("physical_description"),
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
        manifest_data = get_json_with_retry(
            session, manifest_url, args.timeout, args.sleep, args.retries
        )
        metadata = extract_book_metadata(manifest_data)
        pages = parse_iiif_manifest(manifest_data, width=args.iiif_width)

        if not pages:
            print(f"[skip] {volume_id}: no page images could be resolved")
            status = record_failure(
                state, volume_id, "no page images found", keyword, args.max_book_retries,
                extra_fields={"kind": "volume", "catalog_item_id": catalog_item_id, "catalog_url": catalog_url},
            )
            save_state_atomic(args.state_path, state)
            flush_book_record(status)
            return

        raw_total_pages = len(pages)

        # Rule: skip volumes with fewer than --min-pages-per-book pages
        # entirely - too short to be worth any GPU/bandwidth time. Nothing
        # is downloaded; just record it as a terminal state so it isn't
        # re-checked the next time a keyword happens to match its catalogue
        # item again.
        if raw_total_pages < args.min_pages_per_book:
            print(
                f"[skip] {volume_id}: only {raw_total_pages} page(s) "
                f"(< --min-pages-per-book={args.min_pages_per_book}); "
                "skipping volume"
            )
            state[volume_id] = {
                "kind": "volume",
                "status": "skipped_too_short",
                "page_count": raw_total_pages,
                "catalog_item_id": catalog_item_id,
                "catalog_url": catalog_url,
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

        # Rule: for volumes with more than --edge-skip-threshold pages,
        # don't spend download/GPU time on the leading/trailing
        # --edge-skip-count pages (title pages, flyleaves, colophons are
        # reliably low-yield for illustration content). Their IIIF URL is
        # still logged (see log_skipped_edge_page) so the page isn't lost,
        # just never fetched or classified. Computed against the volume's
        # real length (raw_total_pages), not a --limit-pages-per-book cap.
        edge_skip_indices: set[int] = set()
        if raw_total_pages > args.edge_skip_threshold and args.edge_skip_count > 0:
            edge_skip_indices = set(range(0, args.edge_skip_count)) | set(
                range(raw_total_pages - args.edge_skip_count, raw_total_pages)
            )

        start_index = volume_state.get("next_page_index", 0) if volume_state else 0
        pages_kept = volume_state.get("pages_kept", 0) if volume_state else 0
        pages_deleted = volume_state.get("pages_deleted", 0) if volume_state else 0
        pages_negative_audited = (
            volume_state.get("pages_negative_audited", 0) if volume_state else 0
        )
        pages_skipped_edge = (
            volume_state.get("pages_skipped_edge", 0) if volume_state else 0
        )
        total_illustration_detections = (
            volume_state.get("total_illustration_detections", 0) if volume_state else 0
        )
        book_max_confidence = (
            volume_state.get("max_confidence", 0.0) if volume_state else 0.0
        )
        failed_pages = list(volume_state.get("failed_pages", [])) if volume_state else []
        failed_page_retry = (
            dict(volume_state.get("failed_page_retry", {})) if volume_state else {}
        )
        keywords_matched = (
            list(volume_state.get("keywords_matched", [])) if volume_state else []
        )
        if keyword not in keywords_matched:
            keywords_matched.append(keyword)

        tmp_dir = args.output_dir / "tmp_page" / safe_id(volume_id)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        def save_progress(status: str) -> None:
            state[volume_id] = {
                "kind": "volume",
                "status": status,
                "next_page_index": next_page_index,
                "page_count": total_pages,
                "catalog_item_id": catalog_item_id,
                "catalog_url": catalog_url,
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
                    volume_id=volume_id,
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
                    volume_id=volume_id,
                    keyword=keyword,
                    manifest_url=manifest_url,
                    page=pages[page_index],
                    page_index=page_index,
                    tmp_dir=tmp_dir,
                    log_path=log_path,
                )

            # A page-level error must NOT advance next_page_index - the
            # volume stops here so the same page is retried (not silently
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
                        f"[error] {volume_id}: page {page_index + 1} failed "
                        f"{retry_count} times; giving up on this volume "
                        f"(failed_permanent): {record['error']}"
                    )
                    save_progress("failed_permanent")
                else:
                    print(
                        f"[warn] {volume_id}: page {page_index + 1} failed "
                        f"(attempt {retry_count}/{args.max_page_retries}); "
                        f"stopping volume for later retry: {record['error']}"
                    )
                    save_progress("in_progress")

                flush_book_record(state[volume_id]["status"])
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

        state[volume_id]["status"] = "completed"
        save_state_atomic(args.state_path, state)
        flush_book_record("completed")

        shutil.rmtree(tmp_dir, ignore_errors=True)

        print(
            f"[done] {volume_id}: kept={pages_kept} "
            f"deleted={pages_deleted} skipped_edge={pages_skipped_edge} "
            f"total={total_pages}"
        )

    except Exception as exc:
        print(f"[error] {volume_id}: {exc}")
        status = record_failure(
            state, volume_id, str(exc), keyword, args.max_book_retries,
            extra_fields={"kind": "volume", "catalog_item_id": catalog_item_id, "catalog_url": catalog_url},
        )
        if status == "failed_permanent":
            print(
                f"[error] {volume_id}: giving up after repeated volume-level "
                "failures (failed_permanent)"
            )
        save_state_atomic(args.state_path, state)
        flush_book_record(status)


def process_catalog_item(
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
    """
    Handle one Waseda search hit (a catalogue item, e.g. "bunko30_a0002" -
    NOT yet a processable "book"). See module docstring's "Catalogue items
    vs. IIIF volumes" section: this is resolved into zero or more IIIF
    volumes (each becomes an independent process_book() call) exactly once
    per catalogue item, via a lightweight bookkeeping entry in `state` keyed
    by the catalogue item id with kind="catalog_item" - a later keyword
    match against the same catalogue item skips straight to re-dispatching
    its already-known volume ids (or, if it has none, is recorded as a
    no-op skip) without re-fetching the detail page.
    """
    catalog_item_id = result["item_id"]
    detail_url = result["detail_url"]

    catalog_state = state.get(catalog_item_id)
    if catalog_state and catalog_state.get("kind") == "catalog_item":
        keywords_matched = catalog_state.setdefault("keywords_matched", [])
        if keyword not in keywords_matched:
            keywords_matched.append(keyword)
            save_state_atomic(args.state_path, state)
            book_record = books.get(catalog_item_id)
            if book_record is not None and catalog_state.get("status") == "skipped_no_iiif":
                book_record["keywords_matched"] = keywords_matched
                save_books_atomic(args.books_path, books)

        if catalog_state.get("status") == "skipped_no_iiif":
            print(f"[skip] {catalog_item_id} already skipped_no_iiif")
            return
        if catalog_state.get("status") == "failed_permanent":
            print(f"[skip] {catalog_item_id} already failed_permanent")
            return

        for volume_id in catalog_state.get("volume_ids", []):
            process_book(
                volume_id=volume_id,
                manifest_url=manifest_url_for_volume(volume_id),
                catalog_item_id=catalog_item_id,
                catalog_url=detail_url,
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
        return

    print(f"\n[catalog] {catalog_item_id} (keyword={keyword!r})")

    try:
        detail_html = fetch_text_with_retry(
            session, detail_url, args.timeout, args.sleep, args.retries
        )
    except Exception as exc:
        print(f"[warn] {catalog_item_id}: failed to fetch detail page: {exc}")
        status = record_failure(
            state, catalog_item_id, str(exc), keyword, args.max_book_retries,
            extra_fields={"kind": "catalog_item", "detail_url": detail_url},
        )
        save_state_atomic(args.state_path, state)
        if status == "failed_permanent":
            books[catalog_item_id] = {
                "source": SOURCE_NAME,
                "source_item_id": catalog_item_id,
                "waseda_catalog_item_id": catalog_item_id,
                "catalog_url": detail_url,
                "status": status,
                "error": str(exc),
                "keywords_matched": state[catalog_item_id]["keywords_matched"],
                "last_updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            save_books_atomic(args.books_path, books)
        return

    volume_pairs = discover_manifest_urls(detail_html)

    if not volume_pairs:
        state[catalog_item_id] = {
            "kind": "catalog_item",
            "status": "skipped_no_iiif",
            "detail_url": detail_url,
            "keywords_matched": [keyword],
        }
        save_state_atomic(args.state_path, state)
        books[catalog_item_id] = {
            "source": SOURCE_NAME,
            "source_item_id": catalog_item_id,
            "waseda_catalog_item_id": catalog_item_id,
            "catalog_url": detail_url,
            "status": "skipped_no_iiif",
            "keywords_matched": [keyword],
            "last_updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_books_atomic(args.books_path, books)
        print(f"[skip] {catalog_item_id}: no IIIF manifest available")
        return

    volume_ids = [volume_id for volume_id, _ in volume_pairs]
    state[catalog_item_id] = {
        "kind": "catalog_item",
        "status": "expanded",
        "detail_url": detail_url,
        "volume_ids": volume_ids,
        "keywords_matched": [keyword],
    }
    save_state_atomic(args.state_path, state)
    print(f"[catalog] {catalog_item_id}: {len(volume_ids)} IIIF volume(s) discovered")

    for volume_id, manifest_url in volume_pairs:
        process_book(
            volume_id=volume_id,
            manifest_url=manifest_url,
            catalog_item_id=catalog_item_id,
            catalog_url=detail_url,
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
    Resume every volume currently paused ("in_progress") after a page-level
    error. A volume only pauses because download_image_with_retry already
    exhausted its own backoff and still failed - the fix is elapsed
    wall-clock time (transient server issues tend to clear up), not another
    immediate retry. Calling this between keyword searches (and once more at
    the end of the run) gives every paused volume that gap naturally,
    instead of leaving it stuck until its catalogue item happens to
    resurface under a later keyword's search results.

    A resumed volume's manifest URL is reconstructed deterministically from
    its volume id (see manifest_url_for_volume() / module docstring point
    4), and its catalogue item id/URL come from its own stored state -
    neither requires re-scraping the catalogue item's detail page. Only
    entries with kind == "volume" are swept here; kind == "catalog_item"
    bookkeeping entries are never themselves "in_progress" (they resolve to
    a terminal status - skipped_no_iiif, expanded, or failed_permanent - the
    same request they're first seen in).
    """
    paused_ids = [
        entry_id
        for entry_id, entry_state in state.items()
        if entry_state.get("kind") == "volume" and entry_state.get("status") == "in_progress"
    ]

    if not paused_ids:
        return

    print(f"\n=== Resuming {len(paused_ids)} paused volume(s) ===")

    for volume_id in paused_ids:
        volume_state = state.get(volume_id, {})
        keywords_matched = volume_state.get("keywords_matched") or []
        keyword = keywords_matched[-1] if keywords_matched else "resume"

        process_book(
            volume_id=volume_id,
            manifest_url=manifest_url_for_volume(volume_id),
            catalog_item_id=volume_state.get("catalog_item_id", volume_id),
            catalog_url=volume_state.get("catalog_url"),
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
            "[warn] WASEDA_CONTACT_EMAIL is not set; the User-Agent sent to "
            "Waseda has no contact address. Good practice for any harvest, "
            "especially given Waseda has no documented search API at all "
            "(see module docstring) - consider setting it."
        )

    state = load_state(args.state_path)
    books = load_books(args.books_path)
    session = make_session()

    keywords = args.keywords or DEFAULT_KEYWORDS
    print(f"Keywords ({len(keywords)}): {', '.join(keywords)}")
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
    print(
        "[note] Waseda's IIIF rollout is confirmed live to be very small "
        "(around 90 volumes site-wide as of 2026-08-29) - expect most "
        "keyword hits below to end up 'skipped_no_iiif'. See module "
        "docstring point 7."
    )

    for keyword in keywords:
        print(f"\n=== Keyword: {keyword!r} ===")
        try:
            search_results = search_waseda_paginated(
                session=session,
                keyword=keyword,
                max_results=args.max_books_per_keyword,
                timeout=args.timeout,
                sleep_seconds=args.sleep,
                retries=args.retries,
                start_page=args.start_page,
            )
        except Exception as exc:
            print(f"[warn] search failed for {keyword!r}: {exc}")
            continue

        print(f"Found {len(search_results)} candidate catalogue item(s) for {keyword!r}")

        for result in search_results:
            process_catalog_item(
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

    # One more pass once every keyword has been searched, in case a volume
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

    volume_states = [v for v in state.values() if v.get("kind") == "volume"]
    catalog_states = [v for v in state.values() if v.get("kind") == "catalog_item"]

    completed = sum(1 for v in volume_states if v.get("status") == "completed")
    in_progress = sum(1 for v in volume_states if v.get("status") == "in_progress")
    failed = sum(1 for v in volume_states if v.get("status") == "failed")
    failed_permanent = sum(1 for v in volume_states if v.get("status") == "failed_permanent")
    skipped_too_short = sum(1 for v in volume_states if v.get("status") == "skipped_too_short")
    kept_total = sum(v.get("pages_kept", 0) for v in volume_states)
    deleted_total = sum(v.get("pages_deleted", 0) for v in volume_states)
    audited_total = sum(v.get("pages_negative_audited", 0) for v in volume_states)
    skipped_edge_total = sum(v.get("pages_skipped_edge", 0) for v in volume_states)
    detections_total = sum(v.get("total_illustration_detections", 0) for v in volume_states)

    catalog_expanded = sum(1 for v in catalog_states if v.get("status") == "expanded")
    catalog_no_iiif = sum(1 for v in catalog_states if v.get("status") == "skipped_no_iiif")

    print("\n=== Run complete ===")
    print(f"Catalogue items checked: {len(catalog_states)} "
          f"(expanded into IIIF volumes: {catalog_expanded}, no IIIF available: {catalog_no_iiif})")
    print(f"Volumes completed: {completed}")
    print(f"Volumes paused for retry (in_progress): {in_progress}")
    print(f"Volumes failed (will retry next run): {failed}")
    print(f"Volumes failed_permanent (gave up, page retries exhausted): {failed_permanent}")
    print(f"Volumes skipped_too_short (< --min-pages-per-book): {skipped_too_short}")
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
