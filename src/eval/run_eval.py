from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.eval.metrics import mask_iou
from src.models.gnn_text_repair import GNNTextRepair
from src.repair.mask_from_graph import prediction_to_mask, save_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Create predicted masks for graph validation samples.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, default=Path("data/processed_graphs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/debug_graphs"))
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores: list[float] = []
    for graph_path in sorted(args.graph_dir.glob("*.pt"))[: args.limit]:
        graph = torch.load(graph_path, map_location="cpu")
        with torch.no_grad():
            pred = model(graph["node_features"], graph["edge_index"], graph["edge_features"])
        image = Image.open(graph["image_path"]).convert("RGB")
        pred_mask = prediction_to_mask(graph["boxes"], pred, image.size)
        output_path = args.output_dir / f"{graph['sample_id']}_mask.png"
        save_mask(pred_mask, output_path)
        target_mask = Image.open(graph["mask_path"]).convert("L")
        scores.append(mask_iou(pred_mask, target_mask))
    if scores:
        print(f"mean_mask_iou={sum(scores) / len(scores):.4f}")


if __name__ == "__main__":
    main()
