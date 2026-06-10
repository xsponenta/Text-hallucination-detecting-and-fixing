from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


@dataclass
class InpaintRequest:
    image_path: str
    mask_path: str
    expected_text: str
    prompt: str
    output_path: str
    strength: float = 0.28
    guidance_scale: float = 5.5
    steps: int = 25


def load_inpaint_pipeline(model_id: str = "runwayml/stable-diffusion-inpainting", device: str | None = None):
    from diffusers import StableDiffusionInpaintPipeline

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if target_device == "cuda" else torch.float32
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipeline = pipeline.to(target_device)
    return pipeline


def run_inpainting(pipeline, request: InpaintRequest) -> Image.Image:
    image = Image.open(request.image_path).convert("RGB")
    mask = Image.open(request.mask_path).convert("L")
    prompt = f"{request.prompt}. Correct readable text: {request.expected_text}. Preserve the original photo, lighting, camera, and background."
    negative_prompt = "warped letters, misspelled text, blurry text, changed background, low quality"
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        strength=request.strength,
        guidance_scale=request.guidance_scale,
        num_inference_steps=request.steps,
    ).images[0]
    output = Path(request.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    return result


def debug_overlay_repair(image_path: str, mask_path: str, expected_text: str, output_path: str) -> Image.Image:
    """Fast local fallback for debugging masks when diffusion is unavailable."""

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        return image
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, fill=(245, 245, 235))
    font = ImageFont.load_default()
    draw.text((bbox[0] + 4, bbox[1] + 4), expected_text, fill=(20, 20, 20), font=font)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return image
