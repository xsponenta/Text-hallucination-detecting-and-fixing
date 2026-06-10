from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class QualityGateResult:
    accepted: bool
    text_improved: bool
    outside_mse: float
    reason: str


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + int(char_a != char_b)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def text_similarity(a: str, b: str) -> float:
    maximum = max(len(a), len(b), 1)
    return 1.0 - levenshtein_distance(a.lower(), b.lower()) / maximum


def outside_mask_mse(before: Image.Image, after: Image.Image, mask: Image.Image) -> float:
    before_arr = np.asarray(before.convert("RGB"), dtype=np.float32) / 255.0
    after_arr = np.asarray(after.convert("RGB").resize(before.size), dtype=np.float32) / 255.0
    mask_arr = np.asarray(mask.convert("L").resize(before.size), dtype=np.float32) / 255.0
    outside = mask_arr < 0.1
    if outside.sum() == 0:
        return 1.0
    diff = before_arr[outside] - after_arr[outside]
    return float(np.mean(diff * diff))


def decide_quality(
    original_image: Image.Image,
    repaired_image: Image.Image,
    repair_mask: Image.Image,
    expected_text: str,
    ocr_before: str,
    ocr_after: str,
    max_outside_mse: float = 0.004,
) -> QualityGateResult:
    before_score = text_similarity(expected_text, ocr_before)
    after_score = text_similarity(expected_text, ocr_after)
    text_improved = after_score > before_score
    mse = outside_mask_mse(original_image, repaired_image, repair_mask)
    if not text_improved:
        return QualityGateResult(False, False, mse, "OCR similarity did not improve")
    if mse > max_outside_mse:
        return QualityGateResult(False, True, mse, "Photo changed too much outside the repair mask")
    return QualityGateResult(True, True, mse, "Accepted")
