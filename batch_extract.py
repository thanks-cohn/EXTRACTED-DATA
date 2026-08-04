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

import enhanced_detection

enhanced_detection.install()

from extracted_data import build_failure, extract  # noqa: E402
from quality_refinement import refine  # noqa: E402

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
}
GENERATED_SUFFIXES = ("-thumbnail", "-extracted-thumbnail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process one image or a directory and save JSON beside every source image."
    )
    parser.add_argument("source", type=Path, help="Image file or directory to process")
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search the selected directory and all subdirectories",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing *-EXTRACTED-DATA.json files",
    )
    parser.add_argument(
        "--ocr-lang", default="eng",
        help="OCR language string, default: eng",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=("auto", "paddle", "tesseract"),
        default="auto",
        help="Refinement OCR backend. auto prefers local PaddleOCR and falls back to Tesseract.",
    )
    thumbnail = parser.add_mutually_exclusive_group()
    thumbnail.add_argument(
        "--extract-thumbnail", action="store_true",
        help="Extract every detected thumbnail beside its source image",
    )
    thumbnail.add_argument(
        "--no-extract-thumbnail", action="store_true",
        help="Do not extract thumbnails",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
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
    iterator = source.rglob("*") if recursive else source.glob("*")
    return sorted(path for path in iterator if is_source_image(path))


def clickable_uri(path: Path) -> str:
    """Return an absolute file URI that supported terminals can open."""
    return path.expanduser().resolve().as_uri()


def print_clickable_paths(source: Path, output: Path) -> None:
    print(f"      source: {clickable_uri(source)}")
    print(f"      json:   {clickable_uri(output)}")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix}: ").strip().casefold()
    if not answer:
        return default
    return answer in {"y", "yes"}


def choose_thumbnail_policy(args: argparse.Namespace, image_count: int) -> bool | None:
    if args.extract_thumbnail:
        return True
    if args.no_extract_thumbnail or args.non_interactive:
        return False
    if image_count > 1:
        return ask_yes_no("Extract thumbnails for all detected images?", default=False)
    return None


def write_result(source: Path, data: dict) -> Path:
    output = source.with_name(f"{source.stem}-EXTRACTED-DATA.json")
    data.setdefault("output", {})
    data["output"]["json_filename"] = output.name
    data["output"]["json_path"] = str(output)
    data["output"]["json_uri"] = clickable_uri(output)
    data["output"]["saved_beside_source_image"] = True
    data.setdefault("source", {})
    data["source"]["image_uri"] = clickable_uri(source)
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

    print(f"Found {len(images)} image(s).")
    if not images:
        return 0

    thumbnail_policy = choose_thumbnail_policy(args, len(images))
    succeeded = failed = skipped = 0

    for index, image in enumerate(images, start=1):
        output = image.with_name(f"{image.stem}-EXTRACTED-DATA.json")
        prefix = f"[{index}/{len(images)}] {image.name}"
        if output.exists() and not args.overwrite:
            skipped += 1
            print(f"{prefix}  SKIPPED (JSON already exists)")
            print_clickable_paths(image, output)
            continue

        try:
            data = extract(
                image,
                args.ocr_lang,
                interactive=(not args.non_interactive and len(images) == 1),
                force_thumbnail=thumbnail_policy,
            )
            data = refine(data, image, args.ocr_lang, args.ocr_engine)
        except Exception as exc:
            data = build_failure(image, exc)

        written = write_result(image, data)
        status = data.get("extraction", {}).get("status")
        if status == "complete":
            succeeded += 1
            confidence = data.get("extraction", {}).get("overall_confidence")
            engines = data.get("extraction", {}).get("refinement_ocr_engines_used", [])
            engine_note = f" ocr={'+'.join(engines)}" if engines else ""
            confidence_note = f" confidence={confidence}" if confidence is not None else ""
            print(f"{prefix}  OK{confidence_note}{engine_note}")
        else:
            failed += 1
            reason = data.get("extraction", {}).get("reason", "unknown_error")
            print(f"{prefix}  FAILED ({reason})")
        print_clickable_paths(image, written)

    print("\nCompleted:")
    print(f"  succeeded: {succeeded}")
    print(f"  failed:    {failed}")
    print(f"  skipped:   {skipped}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
