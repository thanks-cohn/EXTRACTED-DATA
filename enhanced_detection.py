#!/usr/bin/env python3
"""Adaptive OCR and metadata-boundary patches for small gallery screenshots."""

from __future__ import annotations

import difflib
import re
from typing import Any, Iterable

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

import extracted_data as engine

LABELS = ("group", "type", "language", "series", "characters", "tags")
GENDER_SYMBOLS = {"♀": "female", "♂": "male"}
TRAILING_SYMBOL_OCR = {"♀", "♂", "¢", "=", "«", "+", "5", "s"}
_ORIGINAL_DETECT_FILLED_CHIPS = engine.detect_filled_chips


def _clean(value: str) -> str:
    return re.sub(r"[^a-z]", "", (value or "").casefold())


def fuzzy_canonical_label(value: str) -> str | None:
    cleaned = _clean(value)
    if not cleaned:
        return None

    aliases = {
        "group": "group",
        "groups": "group",
        "type": "type",
        "language": "language",
        "langauge": "language",
        "series": "series",
        "characters": "characters",
        "character": "characters",
        "charactcrs": "characters",
        "tags": "tags",
        "tag": "tags",
    }
    if cleaned in aliases:
        return aliases[cleaned]

    match = difflib.get_close_matches(cleaned, LABELS, n=1, cutoff=0.72)
    return match[0] if match else None


def _tokens_from_patch(
    patch: np.ndarray,
    lang: str,
    *,
    scale: float,
    offset_x: int,
    offset_y: int,
    psm: int,
) -> list[engine.OCRToken]:
    if patch.size == 0:
        return []

    enlarged = (
        cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if scale != 1.0
        else patch
    )
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    data = pytesseract.image_to_data(
        normalized,
        lang=lang,
        config=f"--oem 3 --psm {psm}",
        output_type=Output.DICT,
    )

    result: list[engine.OCRToken] = []
    for index, raw in enumerate(data["text"]):
        text = engine.normalize(raw)
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if confidence < 0:
            continue

        x = int(round(int(data["left"][index]) / scale)) + offset_x
        y = int(round(int(data["top"][index]) / scale)) + offset_y
        width = max(1, int(round(int(data["width"][index]) / scale)))
        height = max(1, int(round(int(data["height"][index]) / scale)))
        result.append(
            engine.OCRToken(
                text=text,
                confidence=confidence / 100.0,
                box=engine.Box(x, y, width, height),
            )
        )
    return result


def _deduplicate(tokens: Iterable[engine.OCRToken]) -> list[engine.OCRToken]:
    kept: list[engine.OCRToken] = []
    for token in sorted(tokens, key=lambda item: item.confidence, reverse=True):
        normalized = _clean(token.text)
        duplicate = False
        for existing in kept:
            if _clean(existing.text) != normalized:
                continue
            dx = abs(existing.box.x - token.box.x)
            dy = abs(existing.box.y - token.box.y)
            if dx <= max(8, token.box.width // 2) and dy <= max(6, token.box.height):
                duplicate = True
                break
        if not duplicate:
            kept.append(token)
    return kept


def adaptive_ocr_tokens(image: np.ndarray, lang: str, psm: int = 6) -> list[engine.OCRToken]:
    """OCR the page at several scales and strategic crops."""
    height, width = image.shape[:2]
    passes: list[tuple[np.ndarray, float, int, int, int]] = [
        (image, 1.0, 0, 0, 11),
        (image, 2.0, 0, 0, 11),
    ]

    crop_specs = [
        (0, int(height * 0.18), width, height),
        (int(width * 0.22), int(height * 0.18), width, int(height * 0.82)),
        (int(width * 0.35), int(height * 0.25), width, int(height * 0.72)),
    ]
    for left, top, right, bottom in crop_specs:
        crop = image[top:bottom, left:right]
        passes.append((crop, 2.5, left, top, 11))
        passes.append((crop, 3.25, left, top, 6))

    all_tokens: list[engine.OCRToken] = []
    for patch, scale, offset_x, offset_y, pass_psm in passes:
        all_tokens.extend(
            _tokens_from_patch(
                patch,
                lang,
                scale=scale,
                offset_x=offset_x,
                offset_y=offset_y,
                psm=pass_psm,
            )
        )
    return _deduplicate(all_tokens)


def panel_delimited_bottom(image: np.ndarray, tags_box: engine.Box, right_edge: int) -> int:
    """Stop metadata above a comic-panel edge.

    A valid delimiter is a meaningful white-space gap after the tag rows,
    followed by one or more long, dark, horizontally adjacent pixel runs.
    This prevents Korean dialogue or artwork inside the first comic panel from
    being mistaken for extra tag chips.
    """
    height, width = image.shape[:2]
    left = max(0, tags_box.x)
    right = min(width, right_edge)
    start = max(0, tags_box.bottom)
    end = min(height, start + max(360, tags_box.height * 24))
    if right <= left or end <= start:
        return min(height, tags_box.bottom + 12)

    gray = cv2.cvtColor(image[start:end, left:right], cv2.COLOR_BGR2GRAY)
    strip_width = max(1, gray.shape[1])

    dark = gray < 72
    nonwhite_fraction = np.mean(gray < 242, axis=1)
    min_gap = max(12, int(round(tags_box.height * 0.8)))
    min_line_width = max(140, int(round(strip_width * 0.28)))
    max_gap_activity = 0.012

    blank_run = 0
    seen_tag_activity = False
    fallback_last_active = 0

    for row_index in range(gray.shape[0]):
        activity = float(nonwhite_fraction[row_index])
        if activity > 0.015:
            seen_tag_activity = True
            fallback_last_active = row_index

        if seen_tag_activity and activity <= max_gap_activity:
            blank_run += 1
            continue

        if blank_run >= min_gap:
            line_band_top = row_index
            line_band_bottom = min(gray.shape[0], row_index + max(8, tags_box.height))
            consecutive_dark_rows = 0
            for probe in range(line_band_top, line_band_bottom):
                row = dark[probe].astype(np.uint8)
                if row.size == 0:
                    continue
                padded = np.pad(row, (1, 1))
                transitions = np.diff(padded)
                starts = np.where(transitions == 1)[0]
                ends = np.where(transitions == -1)[0]
                longest_run = int(np.max(ends - starts)) if starts.size and ends.size else 0
                if longest_run >= min_line_width:
                    consecutive_dark_rows += 1
                    if consecutive_dark_rows >= 2:
                        delimiter_y = start + line_band_top
                        return max(tags_box.bottom + 4, delimiter_y - 2)
                else:
                    consecutive_dark_rows = 0

        blank_run = 0

    fallback = start + fallback_last_active + 10
    hard_cap = tags_box.y + max(220, tags_box.height * 14)
    return min(height, max(tags_box.bottom + 10, min(fallback, hard_cap)))


def _classify_gender_symbol(text: str) -> tuple[str | None, str | None]:
    for symbol, meaning in GENDER_SYMBOLS.items():
        if symbol in text:
            return symbol, meaning
    return None, None


def _trailing_symbol_candidate(raw_text: str) -> str | None:
    match = re.search(r"\s+(\S)\s*$", raw_text, flags=re.UNICODE)
    if not match:
        return None
    candidate = match.group(1)
    return candidate if candidate in TRAILING_SYMBOL_OCR else None


def _detect_chip_symbol(image: np.ndarray, item: dict[str, Any]) -> dict[str, Any] | None:
    """Read and preserve the small symbol at the far right of a tag chip."""
    raw_text = str(item.get("raw_text") or item.get("value") or "")
    symbol, meaning = _classify_gender_symbol(raw_text)
    if symbol:
        return {
            "status": "available",
            "value": symbol,
            "meaning": meaning,
            "raw_ocr": symbol,
            "source": "full_chip_ocr",
        }

    box_data = item.get("box") or {}
    try:
        box = engine.Box(**box_data)
    except (TypeError, ValueError):
        return None

    marker_width = min(box.width, max(16, int(round(box.height * 1.15))))
    left = max(box.x, box.right - marker_width)
    roi = image[box.y:box.bottom, left:box.right]
    marker_ocr = ""
    if roi.size:
        enlarged = cv2.resize(roi, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)
        marker_ocr = engine.normalize(
            pytesseract.image_to_string(
                gray,
                config="--oem 3 --psm 10 -c tessedit_char_whitelist=♀♂",
            )
        )
        symbol, meaning = _classify_gender_symbol(marker_ocr)
        if symbol:
            return {
                "status": "available",
                "value": symbol,
                "meaning": meaning,
                "raw_ocr": marker_ocr,
                "source": "right_edge_symbol_ocr",
                "box": {
                    "x": left,
                    "y": box.y,
                    "width": box.right - left,
                    "height": box.height,
                },
            }

    raw_symbol = _trailing_symbol_candidate(raw_text)
    if raw_symbol:
        return {
            "status": "uncertain",
            "value": None,
            "meaning": None,
            "raw_ocr": raw_symbol,
            "source": "trailing_chip_ocr",
            "box": {
                "x": left,
                "y": box.y,
                "width": box.right - left,
                "height": box.height,
            },
        }
    return None


def detect_filled_chips_with_symbols(
    image: np.ndarray,
    search_box: engine.Box,
    label_right: int,
    lang: str,
) -> list[dict[str, Any]]:
    items = _ORIGINAL_DETECT_FILLED_CHIPS(image, search_box, label_right, lang)
    for item in items:
        raw_text = str(item.get("raw_text") or item.get("value") or "")
        trailing = _trailing_symbol_candidate(raw_text)
        symbol = _detect_chip_symbol(image, item)
        if symbol is not None:
            item["symbol"] = symbol
        if trailing:
            clean_value = re.sub(r"\s+\S\s*$", "", raw_text, flags=re.UNICODE).strip()
            if clean_value:
                item["value"] = clean_value
    return items


def install() -> None:
    engine.canonical_label = fuzzy_canonical_label
    engine.ocr_tokens = adaptive_ocr_tokens
    engine.detect_bottom_from_content = panel_delimited_bottom
    engine.detect_filled_chips = detect_filled_chips_with_symbols
