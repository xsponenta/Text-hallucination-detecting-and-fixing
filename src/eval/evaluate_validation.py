from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from src.eval.metrics import (
    background_preservation_score,
    character_error_rate,
    clip_text_alignment_score,
    fid_from_features,
    lpips_outside_mask,
    mask_iou,
    ocr_accuracy,
    ocr_f1,
    word_accuracy_from_text,
)
from src.models.gnn_text_repair import GNNTextRepair
from src.repair.mask_from_graph import prediction_to_mask, save_mask


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def load_metadata(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def load_repaired_ocr(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["sample_id"])] = str(row.get("ocr_text") or row.get("repaired_text") or "")
    return rows


def load_model(checkpoint_path: Path) -> GNNTextRepair:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = GNNTextRepair(
        node_dim=config["model"].get("node_dim", 12),
        edge_dim=config["model"].get("edge_dim", 5),
        hidden_dim=config["model"].get("hidden_dim", 128),
        layers=config["model"].get("layers", 3),
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate_text_pair(expected: str, predicted: str) -> dict[str, float]:
    return {
        "wa": word_accuracy_from_text(expected, predicted),
        "ocr_accuracy": ocr_accuracy(expected, predicted),
        "ocr_f1": ocr_f1(expected, predicted),
        "cer": character_error_rate(expected, predicted),
    }


def print_block(title: str, values: dict[str, float]) -> None:
    print(title)
    for key, value in values.items():
        print(f"  {key}: {value:.4f}")


def append_group(groups: dict[str, dict[str, list[float]]], group_name: str, values: dict[str, float]) -> None:
    group = groups.setdefault(group_name, {})
    for key, value in values.items():
        group.setdefault(key, []).append(value)


def print_groups(title: str, groups: dict[str, dict[str, list[float]]]) -> None:
    if not groups:
        return
    print(title)
    for group_name in sorted(groups):
        values = {key: mean(items) for key, items in groups[group_name].items()}
        formatted = ", ".join(f"{key}={value:.4f}" for key, value in values.items())
        print(f"  {group_name}: {formatted}")


def image_feature_vector(image: Image.Image) -> np.ndarray:
    """Small no-download image feature vector for FID-style local estimates."""

    resized = image.convert("RGB").resize((64, 64))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    hist_features: list[float] = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0.0, 1.0), density=True)
        hist_features.extend(hist.astype(np.float32).tolist())
    gray = arr.mean(axis=2)
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    return np.concatenate([means, stds, np.array(hist_features + [grad_x, grad_y], dtype=np.float32)])


def simple_rectangular_mask(row: dict, boxes: torch.Tensor, image_size: tuple[int, int], padding: int = 8) -> Image.Image:
    """Baseline mask: rectangle around observed corrupted boxes, no offsets."""

    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    box_values = boxes.detach().cpu().numpy()
    x0 = float(box_values[:, 0].min())
    y0 = float(box_values[:, 1].min())
    x1 = float(box_values[:, 2].max())
    y1 = float(box_values[:, 3].max())
    rectangle = (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )
    draw.rectangle(rectangle, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=2.0))


def evaluate(args: argparse.Namespace) -> None:
    metadata = load_metadata(args.metadata)
    repaired_ocr = load_repaired_ocr(args.repaired_ocr)
    model = load_model(args.checkpoint)
    graph_paths = sorted(args.graph_dir.glob("*.pt"))
    if args.limit is not None:
        graph_paths = graph_paths[: args.limit]
    if not graph_paths:
        raise FileNotFoundError(f"No graph files found in {args.graph_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_scores: list[float] = []
    offset_errors: list[float] = []
    node_error_accs: list[float] = []
    baseline_mask_scores: list[float] = []
    baseline_offset_errors: list[float] = []
    baseline_node_error_accs: list[float] = []
    preservation_scores: list[float] = []
    clip_scores_before: list[float] = []
    clip_scores_after: list[float] = []
    lpips_scores: list[float] = []
    fid_real_features: list[np.ndarray] = []
    fid_repaired_features: list[np.ndarray] = []
    before_text_metrics: dict[str, list[float]] = {"wa": [], "ocr_accuracy": [], "ocr_f1": [], "cer": []}
    after_text_metrics: dict[str, list[float]] = {"wa": [], "ocr_accuracy": [], "ocr_f1": [], "cer": []}
    benchmark_groups: dict[str, dict[str, list[float]]] = {}
    case_groups: dict[str, dict[str, list[float]]] = {}
    after_available = False

    for graph_path in graph_paths:
        graph = torch.load(graph_path, map_location="cpu")
        sample_id = str(graph["sample_id"])
        row = metadata.get(sample_id, {})
        with torch.no_grad():
            prediction = model(graph["node_features"], graph["edge_index"], graph["edge_features"])

        image = Image.open(graph["image_path"]).convert("RGB")
        target_mask = Image.open(graph["mask_path"]).convert("L")
        pred_mask = prediction_to_mask(graph["boxes"], prediction, image.size)
        save_mask(pred_mask, args.output_dir / f"{sample_id}_pred_mask.png")
        mask_scores.append(mask_iou(pred_mask, target_mask))

        baseline_mask = simple_rectangular_mask(row, graph["boxes"], image.size)
        save_mask(baseline_mask, args.output_dir / f"{sample_id}_simple_baseline_mask.png")
        baseline_mask_scores.append(mask_iou(baseline_mask, target_mask))

        offset_errors.append(float(torch.mean(torch.abs(prediction["offset"] - graph["offset_targets"])).item()))
        baseline_offset_errors.append(float(torch.mean(torch.abs(torch.zeros_like(graph["offset_targets"]) - graph["offset_targets"])).item()))
        predicted_errors = (torch.sigmoid(prediction["error_logits"]) >= 0.5).float()
        node_error_accs.append(float((predicted_errors == graph["error_targets"]).float().mean().item()))
        all_bad_baseline = torch.ones_like(graph["error_targets"])
        baseline_node_error_accs.append(float((all_bad_baseline == graph["error_targets"]).float().mean().item()))

        clean = None
        if row.get("clean_image") and Path(row["clean_image"]).exists():
            clean = Image.open(row["clean_image"]).convert("RGB")
            preservation_scores.append(background_preservation_score(image, clean, target_mask))

        expected = str(row.get("expected_text") or graph.get("expected_text") or "")
        corrupted = str(row.get("corrupted_text") or row.get("detected_text") or "")
        if expected and corrupted:
            metrics = evaluate_text_pair(expected, corrupted)
            for key, value in metrics.items():
                before_text_metrics[key].append(value)

        repaired_text = repaired_ocr.get(sample_id)
        if repaired_text is None and args.use_clean_target_as_repaired_text:
            repaired_text = expected
        if expected and repaired_text is not None:
            after_available = True
            metrics = evaluate_text_pair(expected, repaired_text)
            for key, value in metrics.items():
                after_text_metrics[key].append(value)

        repaired_image = clean if args.use_clean_target_as_repaired_text else None
        if args.repaired_image_dir is not None:
            candidate = args.repaired_image_dir / f"{sample_id}_repaired.png"
            if candidate.exists():
                repaired_image = Image.open(candidate).convert("RGB")

        if args.compute_clip and expected:
            try:
                clip_scores_before.append(clip_text_alignment_score(image, expected))
                if repaired_image is not None:
                    clip_scores_after.append(clip_text_alignment_score(repaired_image, expected))
            except ImportError as exc:
                print(f"\nCLIP Score unavailable: {exc}")
                args.compute_clip = False

        if args.compute_lpips and repaired_image is not None:
            try:
                lpips_scores.append(lpips_outside_mask(image, repaired_image, target_mask))
            except ImportError as exc:
                print(f"\nLPIPS unavailable: {exc}")
                args.compute_lpips = False

        if args.compute_fid and clean is not None and repaired_image is not None:
            fid_real_features.append(image_feature_vector(clean))
            fid_repaired_features.append(image_feature_vector(repaired_image))

        group_values = {
            "mask_iou": mask_scores[-1],
            "offset_mae": offset_errors[-1],
            "node_error_acc": node_error_accs[-1],
        }
        if expected and corrupted:
            before_metrics = evaluate_text_pair(expected, corrupted)
            group_values.update({f"before_{key}": value for key, value in before_metrics.items()})
        if expected and repaired_text is not None:
            after_metrics = evaluate_text_pair(expected, repaired_text)
            group_values.update({f"after_{key}": value for key, value in after_metrics.items()})
        append_group(benchmark_groups, str(row.get("benchmark") or "unknown"), group_values)
        append_group(case_groups, str(row.get("case_type") or "unknown"), group_values)

    print("\nValidation report")
    print(f"  samples: {len(graph_paths)}")
    print(f"  checkpoint: {args.checkpoint}")

    print_block(
        "\nGNN geometry metrics",
        {
            "mask_iou": mean(mask_scores),
            "offset_mae_normalized": mean(offset_errors),
            "node_error_accuracy": mean(node_error_accs),
        },
    )

    baseline_values = {
        "simple_mask_iou": mean(baseline_mask_scores),
        "simple_offset_mae_normalized": mean(baseline_offset_errors),
        "simple_all_bad_node_accuracy": mean(baseline_node_error_accs),
        "gnn_mask_iou_gain": mean(mask_scores) - mean(baseline_mask_scores),
        "gnn_offset_mae_reduction": mean(baseline_offset_errors) - mean(offset_errors),
        "gnn_node_accuracy_gain": mean(node_error_accs) - mean(baseline_node_error_accs),
    }
    print_block("\nSimple baseline comparison", baseline_values)

    if any(before_text_metrics.values()):
        print_block("\nBefore repair text metrics", {key: mean(values) for key, values in before_text_metrics.items()})
    else:
        print("\nBefore repair text metrics")
        print("  not available: metadata needs corrupted_text or detected_text")

    if after_available:
        title = "\nAfter repair text metrics"
        if args.use_clean_target_as_repaired_text and not repaired_ocr:
            title += " (synthetic clean-target upper bound)"
        print_block(title, {key: mean(values) for key, values in after_text_metrics.items()})
    else:
        print("\nAfter repair text metrics")
        print("  not available: pass --repaired-ocr path/to/repaired_ocr.jsonl or --use-clean-target-as-repaired-text")

    if preservation_scores:
        print_block("\nPhoto preservation proxy", {"background_preservation_score": mean(preservation_scores)})

    print_groups("\nBy benchmark", benchmark_groups)
    print_groups("\nBy case type", case_groups)

    heavy_metrics: dict[str, float] = {}
    if clip_scores_before:
        heavy_metrics["clip_score_before"] = mean(clip_scores_before)
    if clip_scores_after:
        heavy_metrics["clip_score_after"] = mean(clip_scores_after)
    if lpips_scores:
        heavy_metrics["lpips_outside_mask"] = mean(lpips_scores)
    if args.compute_fid and len(fid_real_features) >= 2 and len(fid_repaired_features) >= 2:
        try:
            heavy_metrics["fid_lightweight_feature_estimate"] = max(
                0.0,
                fid_from_features(np.vstack(fid_real_features), np.vstack(fid_repaired_features)),
            )
        except ImportError as exc:
            print(f"\nFID unavailable: {exc}")

    if heavy_metrics:
        print_block("\nHeavy benchmark metrics", heavy_metrics)
    else:
        print("\nHeavy benchmark metrics")
        print("  not computed. Add --compute-clip, --compute-lpips, and/or --compute-fid")
        print("  CLIP/LPIPS may require optional model packages; FID here uses a lightweight local feature estimate.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print validation metrics for trained GNN text repair.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/gnn_text_repair_best.pt"))
    parser.add_argument("--metadata", type=Path, default=Path("data/synthetic/metadata.jsonl"))
    parser.add_argument("--graph-dir", type=Path, default=Path("data/processed_graphs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--repaired-ocr", type=Path, default=None)
    parser.add_argument("--repaired-image-dir", type=Path, default=None)
    parser.add_argument("--use-clean-target-as-repaired-text", action="store_true")
    parser.add_argument("--compute-clip", action="store_true")
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--compute-fid", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
