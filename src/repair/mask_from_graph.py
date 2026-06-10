from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


def prediction_to_mask(
    boxes: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    image_size: tuple[int, int],
    threshold: float = 0.35,
    padding: int = 8,
    blur_radius: float = 2.0,
) -> Image.Image:
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    error_scores = torch.sigmoid(prediction["error_logits"]).detach().cpu().flatten()
    mask_scores = torch.sigmoid(prediction["mask_logits"]).detach().cpu().flatten()
    offsets = prediction["offset"].detach().cpu()

    for index, box_tensor in enumerate(boxes.detach().cpu()):
        score = max(float(error_scores[index]), float(mask_scores[index]))
        if score < threshold:
            continue
        x0, y0, x1, y1 = [float(value) for value in box_tensor]
        dx = float(offsets[index, 0]) * width
        dy = float(offsets[index, 1]) * height
        source_box = (x0 - padding, y0 - padding, x1 + padding, y1 + padding)
        target_box = (x0 + dx - padding, y0 + dy - padding, x1 + dx + padding, y1 + dy + padding)
        fill = int(np.clip(score * 255, 0, 255))
        draw.rectangle(source_box, fill=fill)
        draw.rectangle(target_box, fill=fill)

    return mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def save_mask(mask: Image.Image, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output)
