from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW

from src.models.gnn_text_repair import GNNTextRepair, repair_loss


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_graph(path: Path, device: torch.device) -> dict:
    graph = torch.load(path, map_location=device)
    for key in [
        "node_features",
        "edge_index",
        "edge_features",
        "error_targets",
        "offset_targets",
        "mask_targets",
        "boxes",
        "target_boxes",
    ]:
        graph[key] = graph[key].to(device)
    return graph


def split_files(files: list[Path], validation_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    return shuffled[validation_count:], shuffled[:validation_count]


def run_epoch(model: GNNTextRepair, files: list[Path], optimizer: AdamW | None, device: torch.device) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    for path in files:
        graph = load_graph(path, device)
        with torch.set_grad_enabled(training):
            prediction = model(graph["node_features"], graph["edge_index"], graph["edge_features"])
            loss, parts = repair_loss(prediction, graph)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / max(1, len(files)) for key, value in totals.items()}


def train(config: dict) -> None:
    graph_dir = Path(config["data"]["graph_dir"])
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(graph_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No graph files found in {graph_dir}")

    device_name = config["training"].get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    train_files, val_files = split_files(files, config["training"].get("validation_fraction", 0.1), config["training"].get("seed", 7))
    model = GNNTextRepair(
        node_dim=config["model"].get("node_dim", 12),
        edge_dim=config["model"].get("edge_dim", 5),
        hidden_dim=config["model"].get("hidden_dim", 128),
        layers=config["model"].get("layers", 3),
        dropout=config["model"].get("dropout", 0.1),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config["training"].get("learning_rate", 0.001), weight_decay=config["training"].get("weight_decay", 0.0001))
    epochs = config["training"].get("epochs", 20)
    min_epochs = config["training"].get("min_epochs", 1)
    min_delta = config["training"].get("min_delta", 0.0)
    best_val = float("inf")
    patience = config["training"].get("patience", 5)
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_files, optimizer, device)
        val_metrics = run_epoch(model, val_files, None, device)
        print(f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}")
        if val_metrics["loss"] < best_val - min_delta:
            best_val = val_metrics["loss"]
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "config": config, "val_metrics": val_metrics}, output_dir / "gnn_text_repair_best.pt")
        else:
            stale_epochs += 1
            if epoch >= min_epochs and stale_epochs >= patience:
                print("Early stopping")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GNN text repair model.")
    parser.add_argument("--config", type=Path, default=Path("configs/gnn_repair.yaml"))
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
