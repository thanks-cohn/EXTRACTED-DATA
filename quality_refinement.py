#!/usr/bin/env python3
"""Second-pass cleanup for structurally detected metadata records."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from dateutil import parser as date_parser

from ocr_backends import backend_diagnostics, read_line

DATE_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"\d{1,2}(?:,\s+\d{4})?(?:,?\s+\d{1,2}:\d{2}\s*(?:AM|PM))?",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b\d{4}\b")
TRIM_CHARS = " \t\r\n|,;:[](){}<>«»‘’“”€¢=+-_"
REJECT_CHIPS = {"by", "and", "or", "the"}
KNOWN_LANGUAGES = {
    "english", "japanese", "chinese", "korean", "spanish", "french",
    "german", "italian", "portuguese", "russian", "thai", "vietnamese",
}
SCRIPT_LANGUAGE_PATTERNS = (
    (re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]"), "Japanese"),
    (re.compile(r"[\uac00-\ud7af]"), "Korean"),
    (re.compile(r"[\u0400-\u04ff]"), "Russian"),
    (re.compile(r"[\u0e00-\u0e7f]"), "Thai"),
)


def _clean_chip(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(TRIM_CHARS)
    value = re.sub(r"^[^\w]+|[^\w)]+$", "", value, flags=re.UNICODE)
    return value.strip()


def _validate_language(*candidates: str) -> tuple[str | None, str | None]:
    for candidate in candidates:
        candidate = (candidate or "").strip()
        if not candidate:
            continue
        for pattern, language in SCRIPT_LANGUAGE_PATTERNS:
            if pattern.search(candidate):
                return language, candidate
        latin = re.sub(r"[^A-Za-z ]", " ", candidate)
        latin = re.sub(r"\s+", " ", latin).strip().casefold()
        if latin in KNOWN_LANGUAGES:
            return latin.title(), candidate
    return None, None


def _plausible_header(value: str) -> str | None:
    value = re.sub(r"\s+", " ", value or "").strip(" |,;:-")
    if not value or len(value) < 2 or len(value) > 220:
        return None
    if value.casefold() in {"group", "type", "language", "series", "characters", "tags"}:
        return None
    if not any(ch.isalnum() for ch in value):
        return None
    return value


def _reference_datetime(data: dict[str, Any]) -> datetime:
    processed_at = data.get("extraction", {}).get("processed_at")
    if processed_at:
        try:
            return datetime.fromisoformat(str(processed_at).replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now().astimezone()


def _extract_date_from_group(data: dict[str, Any], fields: dict[str, Any], work: dict[str, Any]) -> None:
    group = fields.get("group", {})
    group_text = group.get("value") or ""
    match = DATE_RE.search(group_text)
    if not match:
        return

    raw_date = match.group(0).strip()
    remaining = f"{group_text[:match.start()]} {group_text[match.end():]}"
    group_value = re.sub(r"\s+", " ", remaining).strip(" ,;:-")
    group.update({
        "status": "available" if group_value else "absent",
        "value": group_value or None,
        "raw_text": group_value or None,
    })

    reference = _reference_datetime(data)
    has_explicit_year = bool(YEAR_RE.search(raw_date))
    default = reference.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    try:
        parsed = date_parser.parse(raw_date, default=default)
        parsed_iso = parsed.isoformat()
    except (ValueError, OverflowError):
        parsed_iso = None

    work["date"] = {
        "status": "available",
        "raw_text": raw_date,
        "parsed_iso": parsed_iso,
        "box": None,
        "source": "split_from_group_row",
        "year_inferred": not has_explicit_year,
        "inferred_year": reference.year if not has_explicit_year else None,
    }


def _header_fallback(image, data: dict[str, Any], lang: str, ocr_engine: str) -> tuple[str | None, str | None, str | None]:
    group_box = data.get("detected_region", {}).get("anchors", {}).get("group")
    region = data.get("detected_region", {}).get("box")
    if not group_box or not region:
        return None, None, None
    x = group_box["x"]
    right = min(image.shape[1], region["x"] + region["width"])
    width = max(1, right - x)
    group_y = group_box["y"]
    title_box = {"x": x, "y": max(0, group_y - 58), "width": width, "height": 28}
    creator_box = {"x": x, "y": max(0, group_y - 30), "width": width, "height": 24}
    title, _, title_engine = read_line(image, title_box, lang, ocr_engine)
    creator, _, creator_engine = read_line(image, creator_box, lang, ocr_engine)
    engine_used = title_engine if title_engine == creator_engine else f"{title_engine}+{creator_engine}"
    return _plausible_header(title), _plausible_header(creator), engine_used


def refine(data: dict[str, Any], source: Path, lang: str, ocr_engine: str = "auto") -> dict[str, Any]:
    if data.get("extraction", {}).get("status") != "complete":
        return data

    image = cv2.imread(str(source))
    fields = data.setdefault("fields", {})
    work = data.setdefault("work", {})
    warnings = data.setdefault("warnings", [])
    engines_used: set[str] = set()

    _extract_date_from_group(data, fields, work)

    for field_name in ("characters", "tags"):
        field = fields.get(field_name, {})
        cleaned_items = []
        seen = set()
        for item in field.get("items", []):
            value = _clean_chip(item.get("value", ""))
            folded = value.casefold()
            if not value or folded in REJECT_CHIPS or len(value) < 3:
                continue
            if field_name == "tags" and value.isdigit():
                continue
            if folded in seen:
                continue
            seen.add(folded)
            item["value"] = value
            item["raw_text"] = item.get("raw_text") or value
            cleaned_items.append(item)
        field["items"] = cleaned_items
        field["values"] = [item["value"] for item in cleaned_items]
        field["status"] = "available" if cleaned_items else "absent"

    language = fields.get("language", {})
    label_box = language.get("label_box")
    detected_box = data.get("detected_region", {}).get("box")
    candidate = ""
    candidate_confidence = None
    if image is not None and label_box and detected_box:
        crop_box = {
            "x": label_box["x"] + label_box["width"] + 8,
            "y": label_box["y"] - 5,
            "width": max(1, min(360, detected_box["x"] + detected_box["width"] - (label_box["x"] + label_box["width"] + 8))),
            "height": label_box["height"] + 10,
        }
        candidate, candidate_confidence, engine_used = read_line(image, crop_box, lang, ocr_engine)
        engines_used.add(engine_used)

    validated, validated_raw = _validate_language(candidate, language.get("value") or "")
    if validated:
        language.update({"status": "available", "value": validated, "raw_text": validated_raw, "ocr_candidate": candidate or None, "ocr_confidence": candidate_confidence, "validation": "known-language-or-script"})
    else:
        original = language.get("raw_text") or language.get("value")
        language.update({"status": "uncertain", "value": None, "raw_text": original, "ocr_candidate": candidate or None, "ocr_confidence": candidate_confidence, "validation": "failed"})
        if "language_ocr_unreliable" not in warnings:
            warnings.append("language_ocr_unreliable")

    if image is not None and (work.get("title", {}).get("status") != "available" or work.get("creator", {}).get("status") != "available"):
        title, creator, header_engine = _header_fallback(image, data, lang, ocr_engine)
        if header_engine:
            engines_used.update(header_engine.split("+"))
        if title:
            work["title"] = {"status": "available", "value": title, "source": "geometry_anchored_header_ocr"}
        if creator:
            work["creator"] = {"status": "available", "value": creator, "source": "geometry_anchored_header_ocr"}

    block_confidence = float(data.get("detected_region", {}).get("confidence") or 0.0)
    weighted_checks = [
        (bool(fields.get("group", {}).get("value")), 1.0),
        (bool(fields.get("type", {}).get("value")), 1.0),
        (fields.get("language", {}).get("status") == "available", 1.0),
        (bool(fields.get("series", {}).get("value")), 1.0),
        (bool(fields.get("characters", {}).get("values")), 1.0),
        (bool(fields.get("tags", {}).get("values")), 1.0),
        (work.get("date", {}).get("status") == "available", 0.8),
        (work.get("title", {}).get("status") == "available", 1.2),
        (work.get("creator", {}).get("status") == "available", 0.8),
    ]
    earned = sum(weight for passed, weight in weighted_checks if passed)
    possible = sum(weight for _, weight in weighted_checks)
    field_confidence = earned / possible
    thumbnail = data.get("thumbnail", {})
    thumb_confidence = float(thumbnail.get("confidence") or 0.0)
    if thumbnail.get("status") != "detected":
        thumb_confidence = 0.0

    extraction = data["extraction"]
    extraction["block_confidence"] = round(block_confidence, 3)
    extraction["field_confidence"] = round(field_confidence, 3)
    extraction["thumbnail_confidence"] = round(thumb_confidence, 3)
    extraction["overall_confidence"] = round(0.40 * block_confidence + 0.50 * field_confidence + 0.10 * thumb_confidence, 3)
    extraction["refinement"] = "field-cleanup-v2.4"
    extraction["requested_ocr_engine"] = ocr_engine
    extraction["refinement_ocr_engines_used"] = sorted(engines_used)
    extraction["ocr_backend_diagnostics"] = backend_diagnostics()

    for name in ("title", "creator"):
        if work.get(name, {}).get("status") != "available":
            warning = f"{name}_not_detected"
            if warning not in warnings:
                warnings.append(warning)
    return data
