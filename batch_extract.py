#!/usr/bin/env python3
"""Process one image or a directory of images with EXTRACTED-DATA.

Examples:
    python batch_extract.py image.png
    python batch_extract.py /path/to/images
    python batch_extract.py /path/to/images --recursive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from extracted_data import build_failure, extract

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

GENERATED_SUFFIXES = (
    "-thumbnail",
    "-extracted-thumbnail",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process one image or a directory and save JSON beside every source image."
    )
    parser.add_argument("source", type=Path, help="Image file or directory to process")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search the selected directory and all subdirectories",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing *-EXTRACTED-DATA.json files",
    )
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="Tesseract language string, default: eng",
    )
    thumbnail = parser.add_mutually_exclusive_group()
    thumbnail.add_argument(
        "--extract-thumbnail",
        action="store_true",
        help="Extract every detected thumbnail beside its source image",
    )
    thumbnail.add_argument(
        "--no-extract-thumbnail",
        action="store_true",
        help="Do not extract thumbnails",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not ask questions; thumbnail extraction defaults to no",
    )
    return parser.parse_args()


def is_source_image(path: Path) -> bool:
    if not path.is_file() or path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        return False
    stem = path.stem.casefold()
    return not any(stem.endswith(suffix) for suffix in GENERATED_SUFFIXES)


def discover_images(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        if not is_source_image(source):
            raise ValueError(f"Unsupported image file: {source}")
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(f"Source does not exist: {source}")

    iterator: Iterable[Path] = source.rglob("*") if recursive else source.glob("*")
    return sorted(path for path in iterator if is_source_image(path))


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix}: ").strip().casefold()
    if not answer:
        return default
    return answer in {"y", "yes"}


def write_result(source: Path, data: dict) -> Path:
    output = source.with_name(f"{source.stem}-EXTRACTED-DATA.json")
    data.setdefault("output", {})
    data["output"]["json_filename"] = output.name
    data["output"]["json_path"] = str(output)
    data["output"]["saved_beside_source_image"] = True
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()

    try:
        images = discover_images(source, args.recursive)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not images:
        print("No supported images found.")
        return 1

    forced_thumbnail: bool | None
    if args.extract_thumbnail:
        forced_thumbnail = True
    elif args.no_extract_thumbnail or args.non_interactive:
        forced_thumbnail = False
    elif len(images) > 1:
        forced_thumbnail = ask_yes_no("Extract thumbnails for all detected images?", default=False)
    else:
        forced_thumbnail = None

    succeeded = 0
    failed = 0
    skipped = 0

    print(f"Found {len(images)} image(s).")

    for index, image in enumerate(images, start=1):
        output = image.with_name(f"{image.stem}-EXTRACTED-DATA.json")
        prefix = f"[{index}/{len(images)}] {image}"

        if output.exists() and not args.overwrite:
            print(f"{prefix}  SKIPPED (JSON already exists)")
            skipped += 1
            continue

        try:
            data = extract(
                image,
                args.ocr_lang,
                interactive=(not args.non_interactive and len(images) == 1),
                force_thumbnail=forced_thumbnail,
            )
        except Exception as exc:
            data = build_failure(image, exc)

        written = write_result(image, data)
        if data.get("extraction", {}).get("status") == "complete":
            print(f"{prefix}  OK -> {written.name}")
            succeeded += 1
        else:
            reason = data.get("extraction", {}).get("reason", "unknown failure")
            print(f"{prefix}  FAILED ({reason}) -> {written.name}")
            failed += 1

    print("\nCompleted:")
    print(f"  succeeded: {succeeded}")
    print(f"  failed:    {failed}")
    print(f"  skipped:   {skipped}")

    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
