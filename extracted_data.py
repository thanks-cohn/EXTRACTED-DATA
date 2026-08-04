#!/usr/bin/env python3
"""Extract a gallery metadata block from a larger screenshot.

The program deliberately uses two phases:

1. Detect and validate the metadata block from stable labels such as Group,
   Type, Language, Series, Characters, and Tags.
2. Extract title, creator, date, fields, character chips, tag chips, and
   thumbnail evidence only inside or immediately beside that validated block.

The source image is never modified. JSON is always written beside the source as:

    <image-stem>-EXTRACTED-DATA.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pytesseract
from dateutil import parser as date_parser
from pytesseract import Output

EXPECTED_LABELS = ("group", "type", "language", "series", "characters", "tags")
REQUIRED_LABELS = ("group", "tags")
DATE_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"\d{1,2},\s+\d{4}(?:,?\s+\d{1,2}:\d{2}\s*(?:AM|PM))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    def padded(self, px: int, image_width: int, image_height: int) -> "Box":
        left = max(0, self.x - px)
        top = max(0, self.y - px)
        right = min(image_width, self.right + px)
        bottom = min(image_height, self.bottom + px)
        return Box(left, top, right - left, bottom - top)


@dataclass
class OCRToken:
    text: str
    confidence: float
    box: Box


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def canonical_label(value: str) -> str | None:
    cleaned = re.sub(r"[^a-z]", "", normalize(value).casefold())
    aliases = {
        "group": "group",
        "type": "type",
        "language": "language",
        "series": "series",
        "characters": "characters",
        "character": "characters",
        "tags": "tags",
        "tag": "tags",
    }
    return aliases.get(cleaned)


def box_dict(box: Box | None) -> dict[str, int] | None:
    return asdict(box) if box else None


def ocr_tokens(image: np.ndarray, lang: str, psm: int = 6) -> list[OCRToken]:
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=f"--oem 3 --psm {psm}",
        output_type=Output.DICT,
    )
    tokens: list[OCRToken] = []
    for i, raw in enumerate(data["text"]):
        text = normalize(raw)
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1.0
        if not text or confidence < 0:
            continue
        tokens.append(
            OCRToken(
                text=text,
                confidence=confidence / 100.0,
                box=Box(
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["width"][i]),
                    int(data["height"][i]),
                ),
            )
        )
    return tokens


def find_label_cluster(tokens: list[OCRToken], image_width: int) -> tuple[dict[str, OCRToken], float]:
    candidates: dict[str, list[OCRToken]] = {name: [] for name in EXPECTED_LABELS}
    for token in tokens:
        label = canonical_label(token.text)
        if label:
            candidates[label].append(token)

    clusters: list[tuple[dict[str, OCRToken], float]] = []
    for group in candidates["group"]:
        cluster: dict[str, OCRToken] = {"group": group}
        for label in EXPECTED_LABELS[1:]:
            viable = [
                t
                for t in candidates[label]
                if t.box.y >= group.box.y - 8
                and t.box.y - group.box.y <= max(500, group.box.height * 25)
                and abs(t.box.x - group.box.x) <= max(60, group.box.width * 2)
            ]
            if viable:
                cluster[label] = min(viable, key=lambda t: (abs(t.box.x - group.box.x), t.box.y))

        if not all(name in cluster for name in REQUIRED_LABELS):
            continue

        ordered = [cluster[name].box.y for name in EXPECTED_LABELS if name in cluster]
        correct_order = ordered == sorted(ordered)
        xs = [token.box.x for token in cluster.values()]
        alignment_spread = max(xs) - min(xs) if xs else image_width
        score = 0.0
        score += 3.0 if "group" in cluster else 0
        score += 4.0 if "tags" in cluster else 0
        score += 2.0 if "characters" in cluster else 0
        score += 0.8 * sum(1 for name in EXPECTED_LABELS if name in cluster)
        score += 3.0 if correct_order else -4.0
        score += max(0.0, 3.0 - alignment_spread / 20.0)
        score += sum(t.confidence for t in cluster.values()) / max(1, len(cluster))
        clusters.append((cluster, score))

    if not clusters:
        raise ValueError("metadata_label_cluster_not_found")
    cluster, score = max(clusters, key=lambda item: item[1])
    confidence = max(0.0, min(1.0, score / 17.0))
    return cluster, confidence


def horizontal_filled_bands(image: np.ndarray, group_box: Box) -> list[Box]:
    """Find up to two wide header bands immediately above Group."""
    search_top = max(0, group_box.y - max(140, group_box.height * 8))
    search_bottom = group_box.y
    region = image[search_top:search_bottom, group_box.x :]
    if region.size == 0:
        return []
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    non_white = cv2.threshold(gray, 242, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    joined = cv2.morphologyEx(non_white, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < max(180, image.shape[1] * 0.20):
            continue
        if not 8 <= h <= 60:
            continue
        candidates.append(Box(x + group_box.x, y + search_top, w, h))
    candidates.sort(key=lambda b: b.y, reverse=True)
    near: list[Box] = []
    cursor = group_box.y
    for box in candidates:
        gap = cursor - box.bottom
        if -5 <= gap <= 24:
            near.append(box)
            cursor = box.y
        if len(near) == 2:
            break
    return sorted(near, key=lambda b: b.y)


def detect_bottom_from_content(image: np.ndarray, tags_box: Box, right_edge: int) -> int:
    """Extend through wrapped chip rows, stopping at a blank gap or strong divider."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start = max(0, tags_box.y - 4)
    end = min(image.shape[0], start + max(220, tags_box.height * 14))
    strip = gray[start:end, tags_box.x:right_edge]
    if strip.size == 0:
        return min(image.shape[0], tags_box.bottom + 10)
    activity = np.mean(strip < 242, axis=1)
    min_active = 0.015
    seen = False
    blank_run = 0
    last_active = tags_box.bottom - start
    for i, value in enumerate(activity):
        if value > min_active:
            seen = True
            blank_run = 0
            last_active = i
        elif seen:
            blank_run += 1
            if blank_run >= max(10, tags_box.height):
                break
    return min(image.shape[0], start + last_active + 10)


def establish_metadata_region(
    image: np.ndarray,
    labels: dict[str, OCRToken],
    padding: int = 10,
) -> tuple[Box, list[Box]]:
    h, w = image.shape[:2]
    header_bands = horizontal_filled_bands(image, labels["group"].box)
    top_anchor = header_bands[0].y if header_bands else labels["group"].box.y
    left = max(0, min(token.box.x for token in labels.values()) - padding)
    top = max(0, top_anchor - padding)

    # Start broad enough to include values and the date but avoid assuming a fixed site width.
    right = w
    bottom = detect_bottom_from_content(image, labels["tags"].box, right)
    return Box(left, top, right - left, max(1, bottom - top)), header_bands


def row_text(tokens: Iterable[OCRToken], top: int, bottom: int, min_x: int, max_x: int) -> str:
    selected = [
        token
        for token in tokens
        if token.box.center_y >= top
        and token.box.center_y < bottom
        and token.box.x >= min_x
        and token.box.right <= max_x
    ]
    selected.sort(key=lambda token: (token.box.y, token.box.x))
    return normalize(" ".join(token.text for token in selected))


def field_value_ranges(labels: dict[str, OCRToken], region: Box) -> dict[str, tuple[int, int]]:
    ordered = sorted(labels.items(), key=lambda item: item[1].box.y)
    ranges: dict[str, tuple[int, int]] = {}
    for index, (name, token) in enumerate(ordered):
        top = token.box.y - 3
        if index + 1 < len(ordered):
            bottom = ordered[index + 1][1].box.y - 2
        else:
            bottom = region.bottom
        ranges[name] = (top, bottom)
    return ranges


def detect_filled_chips(
    image: np.ndarray,
    search_box: Box,
    label_right: int,
    lang: str,
) -> list[dict[str, Any]]:
    patch = image[search_box.y:search_box.bottom, search_box.x:search_box.right]
    if patch.size == 0:
        return []
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # Any compact non-white filled block is eligible; no exact gray is calibrated.
    mask = cv2.inRange(gray, 20, 235)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        original = Box(x + search_box.x, y + search_box.y, w, h)
        if original.x <= label_right:
            continue
        if not 8 <= h <= 42 or not 14 <= w <= 320 or w <= h:
            continue
        roi = image[original.y:original.bottom, original.x:original.right]
        if roi.size == 0:
            continue
        border_pixels = np.concatenate((roi[0], roi[-1], roi[:, 0], roi[:, -1]), axis=0)
        if float(np.mean(border_pixels)) > 242:
            continue
        boxes.append(original)

    # Deduplicate nested contours.
    boxes.sort(key=lambda b: b.width * b.height, reverse=True)
    kept: list[Box] = []
    for box in boxes:
        if any(
            box.x >= other.x
            and box.y >= other.y
            and box.right <= other.right
            and box.bottom <= other.bottom
            for other in kept
        ):
            continue
        kept.append(box)
    kept.sort(key=lambda b: (b.y, b.x))

    items: list[dict[str, Any]] = []
    for box in kept:
        roi = image[box.y:box.bottom, box.x:box.right]
        text = normalize(
            pytesseract.image_to_string(roi, lang=lang, config="--oem 3 --psm 7")
        ).strip(" |,;:[]()")
        if not text or len(text) > 80 or not any(ch.isalnum() for ch in text):
            continue
        if canonical_label(text):
            continue
        items.append({"value": text, "raw_text": text, "box": box_dict(box)})
    return items


def detect_date(tokens: list[OCRToken], region: Box) -> dict[str, Any]:
    line_candidates: list[tuple[str, list[OCRToken]]] = []
    by_line: dict[int, list[OCRToken]] = {}
    for token in tokens:
        if region.x <= token.box.x <= region.right and region.y <= token.box.y <= region.bottom:
            key = round(token.box.center_y / max(8, token.box.height))
            by_line.setdefault(key, []).append(token)
    for line in by_line.values():
        line.sort(key=lambda t: t.box.x)
        line_candidates.append((normalize(" ".join(t.text for t in line)), line))
    for text, line in line_candidates:
        match = DATE_RE.search(text)
        if not match:
            continue
        raw = match.group(0)
        parsed_iso = None
        try:
            parsed_iso = date_parser.parse(raw).isoformat()
        except (ValueError, OverflowError):
            pass
        box = Box(
            min(t.box.x for t in line),
            min(t.box.y for t in line),
            max(t.box.right for t in line) - min(t.box.x for t in line),
            max(t.box.bottom for t in line) - min(t.box.y for t in line),
        )
        return {"status": "available", "raw_text": raw, "parsed_iso": parsed_iso, "box": box_dict(box)}
    return {"status": "absent", "raw_text": None, "parsed_iso": None, "box": None}


def detect_thumbnail(
    image: np.ndarray,
    metadata_region: Box,
    tokens: list[OCRToken],
) -> dict[str, Any]:
    left_limit = max(0, metadata_region.x)
    search = image[metadata_region.y:metadata_region.bottom, 0:left_limit]
    if search.size == 0:
        return {"status": "absent", "reason": "no_space_left_of_metadata"}

    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 130)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = Box(x, y + metadata_region.y, w, h)
        area = w * h
        if area < 2500 or w < 50 or h < 50:
            continue
        if w > left_limit or box.bottom > metadata_region.bottom:
            continue
        candidates.append(box)

    read_tokens = [t for t in tokens if normalize(t.text).casefold() in {"read", "online", "download"}]
    best: tuple[Box, float, bool, bool] | None = None
    for box in candidates:
        below = [t for t in read_tokens if box.x - 15 <= t.box.x <= box.right + 15 and box.bottom <= t.box.y <= box.bottom + 100]
        words = {normalize(t.text).casefold() for t in below}
        read_online = "read" in words and "online" in words
        download = "download" in words
        score = math.log1p(box.width * box.height) + (4 if read_online else 0) + (4 if download else 0)
        if best is None or score > best[1]:
            best = (box, score, read_online, download)

    if best is None:
        return {"status": "absent", "reason": "rectangular_image_candidate_not_found"}
    box, score, read_online, download = best
    return {
        "status": "detected",
        "box": box_dict(box),
        "size": {"width_px": box.width, "height_px": box.height},
        "aspect_ratio": round(box.width / box.height, 6),
        "validation": {
            "left_of_metadata": True,
            "rectangular_image_region": True,
            "read_online_found_below": read_online,
            "download_found_below": download,
        },
        "confidence": round(min(1.0, 0.45 + 0.25 * read_online + 0.25 * download), 3),
        "extraction": {
            "requested": False,
            "performed": False,
            "destination_mode": None,
            "output_path": None,
        },
    }


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix}: ").strip().casefold()
    if not answer:
        return default
    return answer in {"y", "yes"}


def choose_thumbnail_path(source: Path) -> tuple[Path, str]:
    choice = input("Use default location or choose a new one? [D/n]: ").strip().casefold()
    filename = f"{source.stem}-THUMBNAIL.png"
    if choice in {"", "d", "default"}:
        return source.with_name(filename), "default"
    custom = input("Enter destination directory: ").strip()
    directory = Path(custom).expanduser().resolve() if custom else source.parent
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename, "custom"


def extract(source: Path, lang: str, interactive: bool, force_thumbnail: bool | None) -> dict[str, Any]:
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source}")
    h, w = image.shape[:2]
    tokens = ocr_tokens(image, lang=lang, psm=6)

    labels, cluster_confidence = find_label_cluster(tokens, w)
    region, header_bands = establish_metadata_region(image, labels)
    ranges = field_value_ranges(labels, region)
    value_x = max(token.box.right for token in labels.values()) + 8

    fields: dict[str, Any] = {}
    for name in ("group", "type", "language", "series"):
        if name not in labels:
            fields[name] = {"status": "absent", "value": None}
            continue
        top, bottom = ranges[name]
        text = row_text(tokens, top, bottom, value_x, region.right)
        fields[name] = {
            "status": "available" if text else "absent",
            "value": text or None,
            "raw_text": text or None,
            "label_box": box_dict(labels[name].box),
        }

    char_top, char_bottom = ranges.get("characters", (0, 0))
    tag_top, tag_bottom = ranges.get("tags", (0, 0))
    character_items = detect_filled_chips(
        image,
        Box(region.x, char_top, region.width, max(1, char_bottom - char_top)),
        labels.get("characters", labels["tags"]).box.right,
        lang,
    ) if "characters" in labels else []
    tag_items = detect_filled_chips(
        image,
        Box(region.x, tag_top, region.width, max(1, tag_bottom - tag_top)),
        labels["tags"].box.right,
        lang,
    )
    fields["characters"] = {
        "status": "available" if character_items else "absent",
        "values": [item["value"] for item in character_items],
        "items": character_items,
    }
    fields["tags"] = {
        "status": "available" if tag_items else "absent",
        "values": [item["value"] for item in tag_items],
        "items": tag_items,
        "row_count": len({round(item["box"]["y"] / max(1, item["box"]["height"])) for item in tag_items}),
    }

    title_text = None
    creator_text = None
    if header_bands:
        title_text = row_text(tokens, header_bands[0].y, header_bands[0].bottom, header_bands[0].x, header_bands[0].right)
    if len(header_bands) > 1:
        creator_text = row_text(tokens, header_bands[1].y, header_bands[1].bottom, header_bands[1].x, header_bands[1].right)

    thumbnail = detect_thumbnail(image, region, tokens)
    extract_thumb = force_thumbnail
    if extract_thumb is None and interactive and thumbnail.get("status") == "detected":
        extract_thumb = ask_yes_no("Extract thumbnail?", default=False)
    extract_thumb = bool(extract_thumb)
    if thumbnail.get("status") == "detected":
        thumbnail["extraction"]["requested"] = extract_thumb
        if extract_thumb:
            output_path, mode = choose_thumbnail_path(source) if interactive else (source.with_name(f"{source.stem}-THUMBNAIL.png"), "default")
            b = Box(**thumbnail["box"])
            crop = image[b.y:b.bottom, b.x:b.right]
            ok = cv2.imwrite(str(output_path), crop)
            thumbnail["extraction"].update(
                {
                    "performed": bool(ok),
                    "destination_mode": mode,
                    "output_path": str(output_path) if ok else None,
                }
            )

    result: dict[str, Any] = {
        "schema": "gallery-metadata-extraction/v1",
        "source": {
            "image_filename": source.name,
            "image_path": str(source.resolve()),
            "image_width_px": w,
            "image_height_px": h,
        },
        "output": {
            "json_filename": f"{source.stem}-EXTRACTED-DATA.json",
            "saved_beside_source_image": True,
        },
        "extraction": {
            "status": "complete",
            "method": "two-stage-anchor-stack-and-bounded-layout-extraction",
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "overall_confidence": round(cluster_confidence, 3),
            "ocr_language": lang,
        },
        "detected_region": {
            "status": "found",
            "box": box_dict(region),
            "confidence": round(cluster_confidence, 3),
            "anchors": {name: box_dict(token.box) for name, token in labels.items()},
            "header_bands": [box_dict(box) for box in header_bands],
        },
        "work": {
            "title": {"status": "available" if title_text else "absent", "value": title_text},
            "creator": {"status": "available" if creator_text else "absent", "value": creator_text},
            "date": detect_date(tokens, region),
        },
        "fields": fields,
        "thumbnail": thumbnail,
        "warnings": [],
        "errors": [],
    }
    return result


def build_failure(source: Path, exc: Exception) -> dict[str, Any]:
    return {
        "schema": "gallery-metadata-extraction/v1",
        "source": {"image_filename": source.name, "image_path": str(source.resolve())},
        "output": {
            "json_filename": f"{source.stem}-EXTRACTED-DATA.json",
            "saved_beside_source_image": True,
        },
        "extraction": {
            "status": "failed",
            "reason": str(exc),
            "processed_at": datetime.now(timezone.utc).isoformat(),
        },
        "detected_region": {"status": "not-found", "box": None},
        "work": {},
        "fields": {
            name: {"status": "not-extracted"}
            for name in ("group", "type", "language", "series", "characters", "tags")
        },
        "thumbnail": {"status": "not-extracted"},
        "warnings": [],
        "errors": [str(exc)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Screenshot to analyze")
    parser.add_argument("--ocr-lang", default="eng", help="Tesseract language string, default: eng")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--extract-thumbnail", action="store_true", help="Extract the detected thumbnail without prompting")
    group.add_argument("--no-extract-thumbnail", action="store_true", help="Never extract the thumbnail")
    parser.add_argument("--non-interactive", action="store_true", help="Do not ask questions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.image.expanduser().resolve()
    output = source.with_name(f"{source.stem}-EXTRACTED-DATA.json")
    forced: bool | None = None
    if args.extract_thumbnail:
        forced = True
    elif args.no_extract_thumbnail or args.non_interactive:
        forced = False

    try:
        data = extract(source, args.ocr_lang, not args.non_interactive, forced)
    except Exception as exc:  # Always leave a machine-readable record beside the image.
        data = build_failure(source, exc)

    data["output"]["json_path"] = str(output)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return 0 if data.get("extraction", {}).get("status") == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
