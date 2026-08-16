#!/usr/bin/env python3
"""
loc_vlm_triage.py

Large-scale Library of Congress book triage with a vision-language model.

Workflow
--------
1. Search the Library of Congress Books endpoint (or a specified collection).
2. Fetch detailed item metadata for each book.
3. Extract the IIIF Presentation manifest URL.
4. Fetch the manifest and discover page/image information.
5. Sample a small number of representative pages.
6. Download only those sampled page images temporarily.
7. Send each page to Gemma through Together AI.
8. If the first sample is inconclusive, adaptively sample more pages.
9. Save every processed book, its metadata, manifest information, sampled pages,
   VLM annotations, and final triage decision to JSONL.
10. Save raw LOC item JSON and IIIF manifest JSON for reproducibility.

The script does NOT download an entire book unless you explicitly change the
sampling logic.

Environment
-----------
export TOGETHER_API_KEY="your_key_here"

Install
-------
pip install requests together

Example
-------
python loc_vlm_triage.py \
    --query "botany" \
    --limit 20 \
    --output-dir loc_triage \
    --initial-pages 5 \
    --additional-pages 5 \
    --max-pages 15

For a broader corpus:
python loc_vlm_triage.py --limit 1000 --output-dir loc_triage

Notes
-----
- The exact availability/structure of IIIF manifests varies between LOC items.
  The script supports common IIIF Presentation 2 and 3 structures and has a
  fallback to LOC's item/resource "files" structure.
- Images are kept after annotation by default.
- The script is resumable: a book with status "completed" is skipped on later
  runs unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from together import AsyncTogether


LOC_BASE = "https://www.loc.gov"
DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_CONCURRENCY = 3
DEFAULT_RETRIES = 5
DEFAULT_TIMEOUT = 60
DEFAULT_INITIAL_PAGES = 5
DEFAULT_ADDITIONAL_PAGES = 5
DEFAULT_MAX_PAGES = 15

USER_AGENT = (
    "Phyto-Vision-LOC-Triage/1.0 "
    "(research; contact information should be added by researcher)"
)

# These are intentionally broad triage labels rather than a sophisticated
# art-historical taxonomy. They are meant to identify books/pages worth
# sending to later Phyto-Vision stages.
ALLOWED_TAGS = [
    "botanical",
    "herbal_or_medicinal",
    "agriculture",
    "horticulture_or_gardening",
    "natural_history",
    "plant_diagram_or_botanical_diagram",
    "plant_illustration",
    "general_illustration",
    "map_or_chart",
    "mostly_text",
    "uncertain",
    "non_botanical",
]

SYSTEM_PROMPT = f"""
You are a careful visual-corpus triage assistant for a Digital Humanities
research project studying historical books.

The page image you receive is one page from a historical book. Your task is
NOT to identify the exact book from the image and NOT to invent bibliographic
facts. Judge only what is visibly present on the page.

The project's current research interest is botanical and plant-related visual
material, but the corpus is intentionally broad. The purpose of this first
pass is high-recall corpus triage: it is preferable to flag an uncertain page
for later review rather than confidently discard potentially relevant material.

Return ONLY a JSON object with exactly these keys:
- "relevant": boolean
- "tags": list of 1 to 5 strings chosen ONLY from the allowed tags below
- "confidence": number from 0 to 1
- "reason": one short sentence explaining the visible evidence

Allowed tags:
{json.dumps(ALLOWED_TAGS, ensure_ascii=False)}

Use "botanical" when the page visibly contains meaningful plant/botanical
content. Use "plant_illustration" when an actual botanical/plant illustration
is visible. Use "herbal_or_medicinal" when plants are visibly connected to
medicine, remedies, pharmacology, or herbal practice. Use "agriculture" or
"horticulture_or_gardening" only when there is visible evidence for those
contexts.

Set "relevant" to true when the page contains meaningful botanical/plant
material or another clearly useful plant-related context for this research.
Set it to false for pages that are clearly unrelated. If the evidence is
ambiguous, use "uncertain" and reduce confidence.

Do not infer relevance merely because the page looks old or illustrated.
"""

USER_PROMPT = """
Analyze this historical book page for corpus triage.

Return only the required JSON object. Do not include Markdown fences or any
text outside the JSON object.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive LOC book triage using Gemma through Together AI."
    )

    parser.add_argument(
        "--query",
        default="",
        help="LOC search query. Empty means browse the LOC Books endpoint.",
    )
    parser.add_argument(
        "--collection",
        default="",
        help="Optional LOC collection slug, e.g. selected-digitized-books.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of books to process in this run.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="LOC search result page to start from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("loc_triage"),
        help="Directory for records, raw metadata, manifests, and temporary images.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Together model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TOGETHER_API_KEY"),
        help="Together API key. Defaults to TOGETHER_API_KEY.",
    )
    parser.add_argument(
        "--initial-pages",
        type=int,
        default=DEFAULT_INITIAL_PAGES,
        help="Number of pages in the first sample.",
    )
    parser.add_argument(
        "--additional-pages",
        type=int,
        default=DEFAULT_ADDITIONAL_PAGES,
        help="Number of extra pages sampled when the first pass is inconclusive.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Maximum number of pages sent to the VLM for one book.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum concurrent Together requests. Default: {DEFAULT_CONCURRENCY}",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Maximum API and image-download retries per page.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds for LOC/IIIF image requests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Delay after each successful LOC/image request.",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        default=True,
        help="Keep sampled page images instead of deleting them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess books already marked completed.",
    )

    return parser.parse_args()


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


def search_books(
    session: requests.Session,
    query: str,
    collection: str,
    start_page: int,
    limit: int,
    timeout: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    endpoint = (
        f"{LOC_BASE}/collections/{collection}/"
        if collection
        else f"{LOC_BASE}/books/"
    )

    params: dict[str, Any] = {
        "fo": "json",
        "c": min(limit, 100),
        "sp": start_page,
    }
    if query:
        params["q"] = query

    response = session.get(endpoint, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    if not isinstance(results, list):
        return []

    return results[:limit]


def item_json_url(result: dict[str, Any]) -> str | None:
    item_id = extract_item_id(result)
    if not item_id:
        return None

    return f"{LOC_BASE}/item/{item_id}/?fo=json"


def recursive_find_manifest_url(obj: Any) -> str | None:
    """
    Find a likely IIIF Presentation manifest URL anywhere in an LOC JSON
    response. LOC metadata schemas vary, so this deliberately searches
    recursively rather than assuming one exact nesting path.
    """
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
        return url[:-len("/info.json")]

    return None


def fallback_page_images(item_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Fallback for LOC item JSON where a manifest URL cannot be extracted.
    This follows the same useful idea as the user's earlier LOC downloader:
    inspect resources/files and turn IIIF file URLs into IIIF image services.
    """
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

        # Some IIIF objects expose an id directly as the image resource.
        body_id = body.get("id") or body.get("@id")
        if isinstance(body_id, str) and (
            "iiif" in body_id.lower() or "image-services" in body_id.lower()
        ):
            return extract_image_base_from_url(body_id) or body_id.rstrip("/")

    return None


def parse_iiif_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Supports common IIIF Presentation 2 and Presentation 3 structures.
    Returns a normalized list of page dictionaries.
    """
    pages: list[dict[str, Any]] = []

    # IIIF Presentation 3:
    # manifest.items[].items[].items[].body
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
            canvas_items = canvas.get("items", [])

            for annotation_page in canvas_items:
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
                        "image_url": (
                            f"{service_base}/full/1200,/0/default.jpg"
                        ),
                    }
                )

        if pages:
            return pages

    # IIIF Presentation 2:
    # manifest.sequences[0].canvases[].images[0].resource.service
    sequences = manifest.get("sequences", [])
    if isinstance(sequences, list) and sequences:
        canvases = sequences[0].get("canvases", [])

        for canvas_index, canvas in enumerate(canvases):
            if not isinstance(canvas, dict):
                continue

            label = canvas.get(
                "label", f"Page {canvas_index + 1}"
            )

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
                        "image_url": (
                            f"{service_base}/full/1200,/0/default.jpg"
                        ),
                    }
                )

    return pages


def sample_indices(
    total_pages: int,
    desired_count: int,
    excluded: set[int] | None = None,
) -> list[int]:
    """
    Select evenly distributed page indices.

    The first sample intentionally covers the whole book rather than only
    title/front matter pages, which is important for sparse botanical content.
    """
    excluded = excluded or set()

    if total_pages <= 0 or desired_count <= 0:
        return []

    available = [i for i in range(total_pages) if i not in excluded]
    if not available:
        return []

    desired_count = min(desired_count, len(available))

    if desired_count == len(available):
        return available

    # Evenly distribute across the complete page range.
    raw = [
        round(i * (total_pages - 1) / (desired_count - 1))
        for i in range(desired_count)
    ] if desired_count > 1 else [total_pages // 2]

    result: list[int] = []
    for idx in raw:
        if idx not in excluded and idx not in result:
            result.append(idx)

    # Fill any gaps caused by rounding/exclusion.
    for idx in available:
        if len(result) >= desired_count:
            break
        if idx not in result:
            result.append(idx)

    return sorted(result)


def encode_image_as_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def validate_annotation(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("VLM response is not a JSON object.")

    relevant = payload.get("relevant")
    tags = payload.get("tags")
    confidence = payload.get("confidence")
    reason = payload.get("reason")

    if not isinstance(relevant, bool):
        raise ValueError("'relevant' must be boolean.")

    if not isinstance(tags, list):
        raise ValueError("'tags' must be a list.")

    normalized_tags: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError("Every tag must be a string.")
        tag = tag.strip()
        if tag not in ALLOWED_TAGS:
            raise ValueError(f"Unknown tag: {tag}")
        if tag not in normalized_tags:
            normalized_tags.append(tag)

    if not isinstance(confidence, (int, float)):
        raise ValueError("'confidence' must be numeric.")

    confidence = max(0.0, min(1.0, float(confidence)))

    if not isinstance(reason, str):
        raise ValueError("'reason' must be a string.")

    return {
        "relevant": relevant,
        "tags": normalized_tags[:5],
        "confidence": confidence,
        "reason": reason.strip(),
    }


async def request_annotation(
    client: AsyncTogether,
    model: str,
    image_path: Path,
) -> dict[str, Any]:
    image_data_url = encode_image_as_data_url(image_path)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    payload = json.loads(content)
    return validate_annotation(payload)


async def request_annotation_with_retry(
    client: AsyncTogether,
    model: str,
    image_path: Path,
    retries: int,
) -> dict[str, Any]:
    delay = 2.0

    for attempt in range(retries):
        try:
            return await request_annotation(client, model, image_path)
        except Exception as exc:
            error_text = str(exc).lower()
            if (
                "credit limit exceeded" in error_text
                or "credit_limit" in error_text
                or "error code: 402" in error_text
            ):
                raise

            if attempt == retries - 1:
                raise

            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    raise RuntimeError("Unreachable retry state.")


def download_image(
    session: requests.Session,
    image_url: str,
    output_path: Path,
    timeout: int,
    sleep_seconds: float,
) -> None:
    partial_path = output_path.with_name(output_path.name + ".part")

    try:
        response = session.get(
            image_url,
            timeout=timeout,
            stream=True,
        )
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


async def annotate_pages(
    client: AsyncTogether,
    model: str,
    session: requests.Session,
    pages: list[dict[str, Any]],
    indices: list[int],
    image_dir: Path,
    concurrency: int,
    retries: int,
    timeout: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one_page(index: int) -> dict[str, Any]:
        page = pages[index]
        page_id = safe_id(str(page.get("id", index)))
        image_path = image_dir / f"page_{index + 1:05d}_{page_id}.jpg"

        result: dict[str, Any] = {
            "page_index": index,
            "page_number": index + 1,
            "label": page.get("label"),
            "image_url": page.get("image_url"),
            "annotation_status": "pending",
        }

        try:
            download_image_with_retry(
                session=session,
                image_url=page["image_url"],
                output_path=image_path,
                timeout=timeout,
                sleep_seconds=sleep_seconds,
                retries=retries,
            )

            async with semaphore:
                annotation = await request_annotation_with_retry(
                    client=client,
                    model=model,
                    image_path=image_path,
                    retries=retries,
                )

            result["annotation"] = annotation
            result["annotation_status"] = "completed"
            return result

        except Exception as exc:
            result["annotation_status"] = "failed"
            result["error"] = str(exc)
            return result

    tasks = [asyncio.create_task(one_page(index)) for index in indices]
    results = await asyncio.gather(*tasks)

    return sorted(results, key=lambda x: x["page_index"])


def summarize_annotations(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        x for x in annotations
        if x.get("annotation_status") == "completed"
        and isinstance(x.get("annotation"), dict)
    ]

    if not completed:
        return {
            "completed_pages": 0,
            "relevant_pages": 0,
            "relevance_ratio": None,
            "max_relevance_confidence": None,
            "tags": [],
            "decision": "error_or_no_evidence",
        }

    relevant = [
        x for x in completed
        if x["annotation"].get("relevant") is True
    ]

    confidences = [
        float(x["annotation"].get("confidence", 0.0))
        for x in completed
    ]

    tags: list[str] = []
    for page in completed:
        for tag in page["annotation"].get("tags", []):
            if tag not in tags:
                tags.append(tag)

    ratio = len(relevant) / len(completed)
    max_confidence = max(confidences) if confidences else 0.0

    # High-recall triage rule:
    # one convincing relevant page is enough to keep a book for downstream
    # inspection. Multiple pages increase confidence but are not required.
    if any(
        x["annotation"].get("relevant") is True
        and float(x["annotation"].get("confidence", 0.0)) >= 0.70
        for x in completed
    ):
        decision = "candidate_keep"
    elif any(
        x["annotation"].get("relevant") is True
        for x in completed
    ):
        decision = "candidate_review"
    elif any(
        "uncertain" in x["annotation"].get("tags", [])
        for x in completed
    ):
        decision = "uncertain"
    else:
        decision = "candidate_reject"

    return {
        "completed_pages": len(completed),
        "relevant_pages": len(relevant),
        "relevance_ratio": ratio,
        "max_relevance_confidence": max_confidence,
        "tags": tags,
        "decision": decision,
    }


def should_sample_more(
    annotations: list[dict[str, Any]],
    total_pages: int,
    sampled_count: int,
    max_pages: int,
) -> bool:
    """
    Adaptive rule.

    Continue sampling if:
    - there is no convincing positive evidence yet, OR
    - the current evidence is ambiguous.

    Stop early when there is strong positive evidence, because the purpose of
    this stage is triage rather than exhaustive annotation.
    """
    summary = summarize_annotations(annotations)

    if summary["completed_pages"] == 0:
        return False

    if summary["decision"] == "candidate_keep":
        return False

    if sampled_count >= min(max_pages, total_pages):
        return False

    # If there is only weak evidence, collect another batch.
    return True


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    temp_path.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

    return records


def record_key(record: dict[str, Any]) -> str | None:
    return record.get("loc_item_id")


def record_is_completed(record: dict[str, Any]) -> bool:
    triage = record.get("triage")
    completed_pages = (
        triage.get("completed_pages") if isinstance(triage, dict) else None
    )
    return (
        record.get("status") == "completed"
        and isinstance(triage, dict)
        and isinstance(completed_pages, int)
        and completed_pages > 0
    )


def extract_item_id(result: dict[str, Any]) -> str | None:
    value = result.get("id") or result.get("url")
    if not isinstance(value, str):
        return None

    match = re.search(r"/item/([^/?#]+)/?", value)
    if match:
        return match.group(1)

    return safe_id(value)


def save_raw_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


async def process_book(
    *,
    result: dict[str, Any],
    session: requests.Session,
    client: AsyncTogether,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    record_map: dict[str, int],
) -> dict[str, Any]:
    loc_item_id = extract_item_id(result)

    if not loc_item_id:
        return {
            "loc_item_id": None,
            "status": "failed",
            "error": "Could not determine LOC item ID from search result.",
            "search_result": result,
        }

    if (
        not args.overwrite
        and loc_item_id in record_map
        and record_is_completed(records[record_map[loc_item_id]])
    ):
        print(f"[skip] {loc_item_id} already completed")
        return records[record_map[loc_item_id]]

    print(f"\n[book] {loc_item_id}")

    record: dict[str, Any] = {
        "loc_item_id": loc_item_id,
        "status": "started",
        "loc_item_url": f"{LOC_BASE}/item/{loc_item_id}/",
        "processed_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "search_result": result,
        "item_metadata": None,
        "iiif_manifest_url": None,
        "iiif_manifest_local_path": None,
        "page_count": None,
        "sampled_pages": [],
        "triage": None,
        "error": None,
    }

    try:
        metadata_url = item_json_url(result)
        if not metadata_url:
            raise RuntimeError("Could not construct LOC item JSON URL.")

        item_data = get_json(
            session=session,
            url=metadata_url,
            timeout=args.timeout,
            sleep_seconds=args.sleep,
        )

        item_metadata = item_data.get("item", item_data)
        record["item_metadata"] = item_metadata
        record["metadata_api_url"] = metadata_url

        raw_path = args.output_dir / "raw_items" / f"{safe_id(loc_item_id)}.json"
        save_raw_json(raw_path, item_data)
        record["raw_item_json_path"] = str(raw_path)

        manifest_url = recursive_find_manifest_url(item_data)

        if manifest_url and manifest_url.startswith("//"):
            manifest_url = "https:" + manifest_url

        if manifest_url:
            record["iiif_manifest_url"] = manifest_url

        pages: list[dict[str, Any]] = []

        if manifest_url:
            try:
                manifest_data = get_json(
                    session=session,
                    url=manifest_url,
                    timeout=args.timeout,
                    sleep_seconds=args.sleep,
                )

                manifest_path = (
                    args.output_dir
                    / "manifests"
                    / f"{safe_id(loc_item_id)}.json"
                )
                save_raw_json(manifest_path, manifest_data)

                record["iiif_manifest_local_path"] = str(manifest_path)
                pages = parse_iiif_manifest(manifest_data)
            except Exception as exc:
                record["iiif_manifest_error"] = str(exc)
                print(
                    f"[warn] {loc_item_id}: manifest unavailable; "
                    "using LOC item image fallback"
                )

        # Fallback to LOC item resources/files if the manifest is absent or
        # doesn't expose usable page services.
        if not pages:
            pages = fallback_page_images(item_data)

        if not pages:
            raise RuntimeError(
                "No IIIF page/image services could be extracted from this item."
            )

        record["page_count"] = len(pages)

        image_dir = (
            args.output_dir
            / "sampled_images"
            / safe_id(loc_item_id)
        )
        image_dir.mkdir(parents=True, exist_ok=True)

        sampled_indices: set[int] = set()
        annotations: list[dict[str, Any]] = []

        # Pass 1: broad coverage over the entire book.
        next_indices = sample_indices(
            total_pages=len(pages),
            desired_count=min(args.initial_pages, args.max_pages),
            excluded=sampled_indices,
        )

        while next_indices:
            print(
                f"[sample] {loc_item_id}: "
                f"adding {len(next_indices)} pages "
                f"(currently {len(sampled_indices)})"
            )

            sampled_indices.update(next_indices)

            batch_annotations = await annotate_pages(
                client=client,
                model=args.model,
                session=session,
                pages=pages,
                indices=next_indices,
                image_dir=image_dir,
                concurrency=args.concurrency,
                retries=args.retries,
                timeout=args.timeout,
                sleep_seconds=args.sleep,
            )

            annotations.extend(batch_annotations)
            record["sampled_pages"] = annotations
            record["triage"] = summarize_annotations(annotations)

            # Save after each batch so a long run is resumable.
            record["status"] = "in_progress"

            existing_index = record_map.get(loc_item_id)
            if existing_index is None:
                records.append(record)
                record_map[loc_item_id] = len(records) - 1
            else:
                records[existing_index] = record

            write_jsonl_atomic(
                args.output_dir / "book_records.jsonl",
                records,
            )

            if not should_sample_more(
                annotations=annotations,
                total_pages=len(pages),
                sampled_count=len(sampled_indices),
                max_pages=args.max_pages,
            ):
                break

            remaining = min(
                args.additional_pages,
                args.max_pages - len(sampled_indices),
                len(pages) - len(sampled_indices),
            )

            if remaining <= 0:
                break

            next_indices = sample_indices(
                total_pages=len(pages),
                desired_count=remaining,
                excluded=sampled_indices,
            )

        if not any(
            page.get("annotation_status") == "completed"
            for page in annotations
        ):
            raise RuntimeError(
                "All sampled page annotations failed; no triage evidence was produced."
            )

        # Remove temporary directory unless requested otherwise.
        if not args.keep_images:
            shutil.rmtree(image_dir, ignore_errors=True)

        record["status"] = "completed"
        record["triage"] = summarize_annotations(annotations)

    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)

    existing_index = record_map.get(loc_item_id)
    if existing_index is None:
        records.append(record)
        record_map[loc_item_id] = len(records) - 1
    else:
        records[existing_index] = record

    write_jsonl_atomic(
        args.output_dir / "book_records.jsonl",
        records,
    )

    print(
        f"[done] {loc_item_id}: "
        f"{record['status']} / "
        f"{(record.get('triage') or {}).get('decision')}"
    )

    return record


async def async_main() -> int:
    args = parse_args()

    if not args.api_key:
        print(
            "Missing Together API key. Set TOGETHER_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    if args.limit <= 0:
        print("--limit must be positive.", file=sys.stderr)
        return 1

    if args.initial_pages <= 0:
        print("--initial-pages must be positive.", file=sys.stderr)
        return 1

    if args.additional_pages <= 0:
        print("--additional-pages must be positive.", file=sys.stderr)
        return 1

    if args.max_pages <= 0:
        print("--max-pages must be positive.", file=sys.stderr)
        return 1

    if args.initial_pages > args.max_pages:
        print("--initial-pages cannot exceed --max-pages.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records_path = args.output_dir / "book_records.jsonl"
    records = load_jsonl(records_path)
    record_map = {
        key: index
        for index, record in enumerate(records)
        if (key := record_key(record))
    }

    session = make_session()
    client = AsyncTogether(api_key=args.api_key)

    print(f"LOC query: {args.query or '[all books]'}")
    print(f"LOC collection: {args.collection or '[books endpoint]'}")
    print(f"Limit: {args.limit}")
    print(f"Model: {args.model}")
    print(
        "Sampling: "
        f"{args.initial_pages} initial + "
        f"{args.additional_pages} adaptive, "
        f"maximum {args.max_pages}"
    )
    print(f"Output: {args.output_dir}")

    try:
        search_results = search_books(
            session=session,
            query=args.query,
            collection=args.collection,
            start_page=args.start_page,
            limit=args.limit,
            timeout=args.timeout,
            sleep_seconds=args.sleep,
        )
    except Exception as exc:
        print(f"LOC search failed: {exc}", file=sys.stderr)
        return 2

    if not search_results:
        print("No LOC search results found.")
        return 0

    completed = 0
    failed = 0
    skipped = 0

    for result in search_results:
        item_id = extract_item_id(result)

        if (
            item_id
            and not args.overwrite
            and item_id in record_map
            and record_is_completed(records[record_map[item_id]])
        ):
            skipped += 1
            print(f"[skip] {item_id} already completed")
            continue

        record = await process_book(
            result=result,
            session=session,
            client=client,
            args=args,
            records=records,
            record_map=record_map,
        )

        if record.get("status") == "completed":
            completed += 1
        else:
            failed += 1

    print("\n=== Run complete ===")
    print(f"Search results: {len(search_results)}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Records: {records_path}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
