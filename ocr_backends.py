#!/usr/bin/env python3
"""Local OCR backend abstraction with Paddle-first automatic fallback."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import pytesseract

_BACKEND_DIAGNOSTICS: dict[str, Any] = {
    "paddle_attempted": False,
    "paddle_available": None,
    "paddle_error": None,
    "fallback_to_tesseract": False,
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _crop(image: np.ndarray, box: dict[str, int]) -> np.ndarray:
    x = max(0, int(box["x"]))
    y = max(0, int(box["y"]))
    right = min(image.shape[1], x + max(1, int(box["width"])))
    bottom = min(image.shape[0], y + max(1, int(box["height"])))
    return image[y:bottom, x:right]


def _prepare_line(roi: np.ndarray) -> np.ndarray:
    if roi.size == 0:
        return roi
    roi = cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def tesseract_line(image: np.ndarray, box: dict[str, int], lang: str) -> tuple[str, float | None]:
    roi = _prepare_line(_crop(image, box))
    if roi.size == 0:
        return "", None
    text = pytesseract.image_to_string(roi, lang=lang, config="--oem 3 --psm 7")
    return _normalise(text), None


def _paddle_language(lang: str) -> str:
    first = (lang or "eng").split("+")[0].casefold()
    return {
        "eng": "en", "en": "en", "jpn": "japan", "japanese": "japan",
        "chi_sim": "ch", "chi_tra": "chinese_cht", "kor": "korean",
    }.get(first, "en")


@lru_cache(maxsize=8)
def _paddle_reader(lang: str):
    _BACKEND_DIAGNOSTICS["paddle_attempted"] = True
    try:
        import paddle  # type: ignore  # noqa: F401
        from paddleocr import PaddleOCR  # type: ignore

        kwargs = {
            "lang": _paddle_language(lang),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "engine": "paddle",
        }
        try:
            reader = PaddleOCR(**kwargs)
        except TypeError:
            kwargs.pop("engine", None)
            reader = PaddleOCR(**kwargs)
        _BACKEND_DIAGNOSTICS.update({"paddle_available": True, "paddle_error": None})
        return reader
    except Exception as exc:
        _BACKEND_DIAGNOSTICS.update({
            "paddle_available": False,
            "paddle_error": f"{type(exc).__name__}: {exc}",
        })
        raise


def _walk_paddle_payload(value: Any) -> list[tuple[str, float | None]]:
    found: list[tuple[str, float | None]] = []
    if value is None:
        return found
    if hasattr(value, "json"):
        try:
            payload = value.json() if callable(value.json) else value.json
            return _walk_paddle_payload(payload)
        except Exception:
            pass
    if isinstance(value, dict):
        texts = value.get("rec_texts") or value.get("texts")
        scores = value.get("rec_scores") or value.get("scores")
        if isinstance(texts, (list, tuple)):
            for index, text in enumerate(texts):
                cleaned = _normalise(str(text))
                if not cleaned:
                    continue
                score = None
                if isinstance(scores, (list, tuple)) and index < len(scores):
                    try:
                        score = float(scores[index])
                    except (TypeError, ValueError):
                        pass
                found.append((cleaned, score))
            return found
        for child in value.values():
            found.extend(_walk_paddle_payload(child))
        return found
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[1], (list, tuple)) and value[1] and isinstance(value[1][0], str):
            score = None
            if len(value[1]) > 1:
                try:
                    score = float(value[1][1])
                except (TypeError, ValueError):
                    pass
            return [(_normalise(value[1][0]), score)]
        for child in value:
            found.extend(_walk_paddle_payload(child))
    return found


def paddle_line(image: np.ndarray, box: dict[str, int], lang: str) -> tuple[str, float | None]:
    roi = _crop(image, box)
    if roi.size == 0:
        return "", None
    reader = _paddle_reader(lang)
    result = list(reader.predict(roi))
    items = _walk_paddle_payload(result)
    if not items:
        return "", None
    text = _normalise(" ".join(text for text, _ in items))
    scores = [score for _, score in items if score is not None]
    return text, (sum(scores) / len(scores) if scores else None)


def backend_diagnostics() -> dict[str, Any]:
    return dict(_BACKEND_DIAGNOSTICS)


def paddle_available(lang: str = "eng") -> bool:
    try:
        _paddle_reader(lang)
        return True
    except Exception:
        return False


def read_line(image: np.ndarray, box: dict[str, int], lang: str, engine: str = "auto") -> tuple[str, float | None, str]:
    engine = (engine or "auto").casefold()
    if engine not in {"auto", "paddle", "tesseract"}:
        raise ValueError(f"Unsupported OCR engine: {engine}")

    if engine in {"auto", "paddle"}:
        try:
            text, confidence = paddle_line(image, box, lang)
            if text or engine == "paddle":
                return text, confidence, "paddle"
            _BACKEND_DIAGNOSTICS["paddle_error"] = "PaddleOCR returned no text for this crop"
        except Exception:
            if engine == "paddle":
                raise
        _BACKEND_DIAGNOSTICS["fallback_to_tesseract"] = True

    text, confidence = tesseract_line(image, box, lang)
    return text, confidence, "tesseract"
