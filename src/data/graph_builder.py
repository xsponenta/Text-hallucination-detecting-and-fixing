from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class GraphSample:
    sample_id: str
    image_path: str
    mask_path: str
    expected_text: str
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    error_targets: torch.Tensor
    offset_targets: torch.Tensor
    mask_targets: torch.Tensor
    boxes: torch.Tensor
    target_boxes: torch.Tensor


def _char_feature(char: str) -> list[float]:
    code = ord(char[0]) if char else 0
    return [
        min(code, 127) / 127.0,
        float(char.isalpha()),
        float(char.isdigit()),
        float(char.isupper()),
    ]


def _box_center(box: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
    return (float(box[0] + box[2]) * 0.5, float(box[1] + box[3]) * 0.5)


def build_graph_from_metadata(row: dict, image_size: int = 512, k_nearest: int = 2) -> GraphSample:
    char_rows = row["char_boxes"]
    node_features: list[list[float]] = []
    boxes: list[list[float]] = []
    target_boxes: list[list[float]] = []
    error_targets: list[float] = []
    offset_targets: list[list[float]] = []
    mask_targets: list[float] = []

    for index, char_row in enumerate(char_rows):
        box = char_row["box"]
        target_box = char_row["target_box"]
        cx, cy = _box_center(box)
        tx, ty = _box_center(target_box)
        width = max(1.0, float(box[2] - box[0]))
        height = max(1.0, float(box[3] - box[1]))
        target_width = max(1.0, float(target_box[2] - target_box[0]))
        target_height = max(1.0, float(target_box[3] - target_box[1]))
        expected_index = index / max(1, len(char_rows) - 1)
        feature = [
            cx / image_size,
            cy / image_size,
            width / image_size,
            height / image_size,
            expected_index,
            target_width / image_size,
            target_height / image_size,
            1.0,
        ]
        feature.extend(_char_feature(char_row["char"]))
        node_features.append(feature)
        boxes.append([float(value) for value in box])
        target_boxes.append([float(value) for value in target_box])
        error_targets.append(float(char_row["is_corrupted"]))
        offset_targets.append([(tx - cx) / image_size, (ty - cy) / image_size])
        mask_targets.append(float(char_row["is_corrupted"]))

    edges: set[tuple[int, int]] = set()
    for index in range(len(char_rows) - 1):
        edges.add((index, index + 1))
        edges.add((index + 1, index))

    centers = [_box_center(item["box"]) for item in char_rows]
    for src, center in enumerate(centers):
        ranked = sorted(
            ((dst, (center[0] - other[0]) ** 2 + (center[1] - other[1]) ** 2) for dst, other in enumerate(centers) if dst != src),
            key=lambda item: item[1],
        )
        for dst, _ in ranked[:k_nearest]:
            edges.add((src, dst))

    if not edges and len(char_rows) == 1:
        edges.add((0, 0))

    edge_index_values = sorted(edges)
    edge_features: list[list[float]] = []
    for src, dst in edge_index_values:
        sx, sy = centers[src]
        dx, dy = centers[dst]
        source_box = char_rows[src]["box"]
        dest_box = char_rows[dst]["box"]
        edge_features.append(
            [
                (dx - sx) / image_size,
                (dy - sy) / image_size,
                abs(dst - src) / max(1, len(char_rows) - 1),
                float(abs(dst - src) == 1),
                max(1.0, dest_box[3] - dest_box[1]) / max(1.0, source_box[3] - source_box[1]),
            ]
        )

    edge_index = torch.tensor(edge_index_values, dtype=torch.long).t().contiguous()
    return GraphSample(
        sample_id=row["sample_id"],
        image_path=row["corrupted_image"],
        mask_path=row["repair_mask"],
        expected_text=row["expected_text"],
        node_features=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_features=torch.tensor(edge_features, dtype=torch.float32),
        error_targets=torch.tensor(error_targets, dtype=torch.float32).unsqueeze(-1),
        offset_targets=torch.tensor(offset_targets, dtype=torch.float32),
        mask_targets=torch.tensor(mask_targets, dtype=torch.float32).unsqueeze(-1),
        boxes=torch.tensor(boxes, dtype=torch.float32),
        target_boxes=torch.tensor(target_boxes, dtype=torch.float32),
    )


def save_graph(sample: GraphSample, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sample_id": sample.sample_id,
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
            "expected_text": sample.expected_text,
            "node_features": sample.node_features,
            "edge_index": sample.edge_index,
            "edge_features": sample.edge_features,
            "error_targets": sample.error_targets,
            "offset_targets": sample.offset_targets,
            "mask_targets": sample.mask_targets,
            "boxes": sample.boxes,
            "target_boxes": sample.target_boxes,
        },
        output_path,
    )


def build_graph_dataset(metadata_path: Path, output_dir: Path, image_size: int, limit: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            row = json.loads(line)
            graph = build_graph_from_metadata(row, image_size=image_size)
            save_graph(graph, output_dir / f"{graph.sample_id}.pt")
    print(f"Wrote graph files to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GNN graph tensors from synthetic metadata.")
    parser.add_argument("--metadata", type=Path, default=Path("data/synthetic/metadata.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_graphs"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build_graph_dataset(args.metadata, args.output_dir, args.image_size, args.limit)


if __name__ == "__main__":
    main()
