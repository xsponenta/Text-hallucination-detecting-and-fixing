from __future__ import annotations

import numpy as np
from PIL import Image

from src.repair.quality_gate import levenshtein_distance, outside_mask_mse


def ocr_accuracy(expected: str, predicted: str) -> float:
    """Exact OCR match accuracy for one sample."""

    return exact_match(expected, predicted)


def _f1_from_units(expected_units: list[str], predicted_units: list[str]) -> float:
    if not expected_units and not predicted_units:
        return 1.0
    if not expected_units or not predicted_units:
        return 0.0
    expected_counts: dict[str, int] = {}
    predicted_counts: dict[str, int] = {}
    for unit in expected_units:
        expected_counts[unit] = expected_counts.get(unit, 0) + 1
    for unit in predicted_units:
        predicted_counts[unit] = predicted_counts.get(unit, 0) + 1
    true_positive = sum(min(expected_counts.get(unit, 0), predicted_counts.get(unit, 0)) for unit in expected_counts)
    precision = true_positive / max(1, len(predicted_units))
    recall = true_positive / max(1, len(expected_units))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def ocr_f1(expected: str, predicted: str, level: str = "character") -> float:
    """OCR F1 at character or word level."""

    if level == "word":
        return _f1_from_units(expected.lower().split(), predicted.lower().split())
    expected_chars = [char for char in expected.lower() if not char.isspace()]
    predicted_chars = [char for char in predicted.lower() if not char.isspace()]
    return _f1_from_units(expected_chars, predicted_chars)


def word_accuracy(expected_words: list[str], predicted_words: list[str]) -> float:
    """Percentage of complete words generated correctly."""

    if not expected_words:
        return 1.0 if not predicted_words else 0.0
    correct = 0
    for expected, predicted in zip(expected_words, predicted_words):
        correct += int(expected.strip().lower() == predicted.strip().lower())
    return correct / len(expected_words)


def word_accuracy_from_text(expected: str, predicted: str) -> float:
    return word_accuracy(expected.split(), predicted.split())


def character_error_rate(expected: str, predicted: str) -> float:
    return levenshtein_distance(expected, predicted) / max(1, len(expected))


def exact_match(expected: str, predicted: str) -> float:
    return float(expected.strip().lower() == predicted.strip().lower())


def mask_iou(predicted: Image.Image, target: Image.Image, threshold: int = 128) -> float:
    pred = np.asarray(predicted.convert("L"), dtype=np.uint8) >= threshold
    gt = np.asarray(target.convert("L").resize(predicted.size), dtype=np.uint8) >= threshold
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def background_preservation_score(before: Image.Image, after: Image.Image, mask: Image.Image) -> float:
    mse = outside_mask_mse(before, after, mask)
    return max(0.0, 1.0 - mse / 0.01)


def lpips_outside_mask(before: Image.Image, after: Image.Image, mask: Image.Image, net: str = "alex") -> float:
    """LPIPS on the preserved area around text.

    Requires optional dependency: `pip install lpips`.
    """

    try:
        import lpips
        import torch
    except ImportError as exc:
        raise ImportError("Install optional dependency to compute LPIPS: pip install lpips") from exc

    before_arr = np.asarray(before.convert("RGB"), dtype=np.float32) / 255.0
    after_arr = np.asarray(after.convert("RGB").resize(before.size), dtype=np.float32) / 255.0
    mask_arr = np.asarray(mask.convert("L").resize(before.size), dtype=np.float32) / 255.0
    outside = (mask_arr < 0.1).astype(np.float32)[..., None]
    before_arr = before_arr * outside
    after_arr = after_arr * outside
    before_tensor = torch.from_numpy(before_arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    after_tensor = torch.from_numpy(after_arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    loss_fn = lpips.LPIPS(net=net)
    with torch.no_grad():
        return float(loss_fn(before_tensor, after_tensor).item())


def clip_text_alignment_score(image: Image.Image, text: str, model_name: str = "openai/clip-vit-base-patch32") -> float:
    """CLIP image-text alignment for repaired image and prompt/expected text.

    Requires optional dependency: `pip install transformers`.
    """

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("Install transformers and torch to compute CLIP alignment") from exc

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    inputs = processor(text=[text], images=image.convert("RGB"), return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return float((image_embeds @ text_embeds.T).item())


def frechet_distance(mu_a: np.ndarray, sigma_a: np.ndarray, mu_b: np.ndarray, sigma_b: np.ndarray) -> float:
    try:
        from scipy.linalg import sqrtm
    except ImportError as exc:
        raise ImportError("Install scipy to compute FID: pip install scipy") from exc

    covmean = sqrtm(sigma_a @ sigma_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(sigma_a + sigma_b - 2.0 * covmean))


def fid_from_features(real_features: np.ndarray, repaired_features: np.ndarray, eps: float = 1e-6) -> float:
    """FID from precomputed feature vectors.

    Feature extraction is intentionally separate so the project can use InceptionV3,
    CLIP, or cached dataset features depending on the experiment.
    """

    real_features = np.asarray(real_features, dtype=np.float64)
    repaired_features = np.asarray(repaired_features, dtype=np.float64)
    mu_real = real_features.mean(axis=0)
    mu_repaired = repaired_features.mean(axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_repaired = np.cov(repaired_features, rowvar=False)
    sigma_real = sigma_real + np.eye(sigma_real.shape[0]) * eps
    sigma_repaired = sigma_repaired + np.eye(sigma_repaired.shape[0]) * eps
    return frechet_distance(mu_real, sigma_real, mu_repaired, sigma_repaired)
