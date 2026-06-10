from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OCRDetection:
    text: str
    confidence: float
    box: tuple[int, int, int, int]


def _quad_to_box(points: list[list[float]] | tuple[tuple[float, float], ...]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


class EasyOCRDetector:
    def __init__(self, languages: list[str] | None = None, gpu: bool = True) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError("Install easyocr to use EasyOCRDetector: pip install easyocr") from exc
        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    def detect(self, image_path: str | Path) -> list[OCRDetection]:
        detections: list[OCRDetection] = []
        for points, text, confidence in self.reader.readtext(str(image_path)):
            detections.append(OCRDetection(text=text, confidence=float(confidence), box=_quad_to_box(points)))
        return detections


def detections_to_region_rows(detections: list[OCRDetection]) -> list[dict]:
    return [{"text": item.text, "confidence": item.confidence, "box": item.box} for item in detections]
