#!/usr/bin/env python3
"""Second-pass cleanup for structurally detected metadata records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import pytesseract
from dateutil import parser as date_parser

DATE_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"\d{1,2},\s+\d{4}(?:,?\s+\d{1,2}:\d{2}\s*(?:AM|PM))?",
    re.IGNORECASE,
)
TRIM_CHARS = " \t\r\n|,;:[](){}<>«»‘’“”€¢=+-_"
REJECT_CHIPS = {"by", "and", "or", "the"}
KNOWN_LANGUAGES = {
    "english", "japanese", "chinese", "korean", "spanish", "french",
    "german", "italian", "portuguese", "russian", "thai", "vietnamese",
}


def _clean_chip(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(TRIM_CHARS)
    value = re.sub(r"^[^\w]+|[^\w)]+$", "", value, flags=re.UNICODE)
    return value.strip()


def _ocr_line(image, box: dict[str, int], lang: str) -> str:
    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
    pad_x, pad_y = 8, 5
    y0, y1 = max(0, y - pad_y), min(image.shape[0], y + h + pad_y)
    x0, x1 = max(0, x - pad_x), min(image.shape[1], x + w + pad_x)
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return ""
    roi = cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return re.sub(
        r"\s+", " ",
        pytesseract.image_to_string(gray, lang=lang, config="--oem 3 --psm 7")
    ).strip()


def refine(data: dict[str, Any], source: Path, lang: str) -> dict[str, Any]:
    if data.get("extraction", {}).get("status") != "complete":
        return data

    image = cv2.imread(str(source))
    fields = data.setdefault("fields", {})
    work = data.setdefault("work", {})
    warnings = data.setdefault("warnings", [])

    # Split a date accidentally captured at the end of Group.
    group = fields.get("group", {})
    group_text = group.get("value") or ""
    match = DATE_RE.search(group_text)
    if match:
        raw_date = match.group(0)
        group_value = group_text[:match.start()].strip(" ,;:-")
        group.update({"value": group_value or None, "raw_text": group_value or None})
        try:
            parsed_iso = date_parser.parse(raw_date).isoformat()
        except (ValueError, OverflowError):
            parsed_iso = None
        work["date"] = {
            "status": "available",
            "raw_text": raw_date,
            "parsed_iso": parsed_iso,
            "box": None,
            "source": "split_from_group_row",
        }

    # Clean chip OCR and reject obvious non-chip debris.
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

    # Re-OCR the language row from its own enlarged crop, then validate it.
    language = fields.get("language", {})
    label_box = language.get("label_box")
    detected_box = data.get("detected_region", {}).get("box")
    if image is not None and label_box and detected_box:
        crop_box = {
            "x": label_box["x"] + label_box["width"] + 8,
            "y": label_box["y"] - 5,
            "width": max(1, min(360, detected_box["x"] + detected_box["width"] - (label_box["x"] + label_box["width"] + 8))),
            "height": label_box["height"] + 10,
        }
        candidate = _ocr_line(image, crop_box, lang)
        normalized = re.sub(r"[^A-Za-z ]", "", candidate).strip().casefold()
        if normalized in KNOWN_LANGUAGES:
            language.update({"status": "available", "value": normalized.title(), "raw_text": candidate})
        elif language.get("value") and not re.search(r"[A-Za-z]{4,}", language.get("value", "")):
            language.update({"status": "uncertain", "value": None})
            warnings.append("language_ocr_unreliable")

    # Confidence now distinguishes block detection from field extraction quality.
    block_confidence = float(data.get("detected_region", {}).get("confidence") or 0.0)
    checks = []
    for name in ("group", "type", "series"):
        checks.append(bool(fields.get(name, {}).get("value")))
    checks.append(bool(fields.get("characters", {}).get("values")))
    checks.append(bool(fields.get("tags", {}).get("values")))
    checks.append(work.get("date", {}).get("status") == "available")
    field_confidence = sum(checks) / max(1, len(checks))
    data["extraction"]["block_confidence"] = round(block_confidence, 3)
    data["extraction"]["field_confidence"] = round(field_confidence, 3)
    data["extraction"]["overall_confidence"] = round(
        0.45 * block_confidence + 0.55 * field_confidence, 3
    )
    data["extraction"]["refinement"] = "field-cleanup-v2"
    return data
