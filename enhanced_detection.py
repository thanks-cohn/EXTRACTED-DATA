#!/usr/bin/env python3
"""Adaptive OCR patches for screenshots where metadata text is small.

This module keeps the original two-stage design, but improves stage one by
running multiple OCR passes at several scales and page regions before label
clustering. Coordinates are mapped back to the original image.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

import extracted_data as engine

LABELS = ("group", "type", "language", "series", "characters", "tags")


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

    # OCR frequently drops or substitutes one character in these small labels.
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

    if scale != 1.0:
        enlarged = cv2.resize(
            patch,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        enlarged = patch

    # Mild contrast normalization helps gray labels without assuming a fixed color.
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
    """OCR the page at several scales and strategic crops.

    Full screenshots make the metadata lettering tiny. The right and lower page
    passes enlarge the likely content area while preserving original coordinates.
    """
    height, width = image.shape[:2]
    passes: list[tuple[np.ndarray, float, int, int, int]] = [
        (image, 1.0, 0, 0, 11),
        (image, 2.0, 0, 0, 11),
    ]

    # Metadata panels in the target screenshots occupy the page body, commonly
    # right of the thumbnail and below browser/error chrome. These are broad
    # candidate regions, not fixed metadata coordinates.
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


def install() -> None:
    engine.canonical_label = fuzzy_canonical_label
    engine.ocr_tokens = adaptive_ocr_tokens
