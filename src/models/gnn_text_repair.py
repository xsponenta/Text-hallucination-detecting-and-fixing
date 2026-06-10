from __future__ import annotations

import torch
from torch import nn


class EdgeMessageLayer(nn.Module):
    """Small edge-aware message passing layer implemented with plain PyTorch."""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x
        src, dst = edge_index
        message_input = torch.cat([x[src], x[dst], edge_features], dim=-1)
        messages = self.message(message_input)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)
        degree = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst, torch.ones(messages.size(0), 1, device=x.device, dtype=x.dtype))
        aggregated = aggregated / degree.clamp_min(1.0)
        return x + self.update(torch.cat([x, aggregated], dim=-1))


class GNNTextRepair(nn.Module):
    def __init__(
        self,
        node_dim: int = 12,
        edge_dim: int = 5,
        hidden_dim: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList([EdgeMessageLayer(hidden_dim, edge_dim, dropout) for _ in range(layers)])
        self.error_head = nn.Linear(hidden_dim, 1)
        self.offset_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.mask_head = nn.Linear(hidden_dim, 1)
        self.transform_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.region_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.node_encoder(node_features)
        for layer in self.layers:
            x = layer(x, edge_index, edge_features)

        region_embedding = x.mean(dim=0, keepdim=True)
        region_values = torch.sigmoid(self.region_head(region_embedding)).squeeze(0)
        return {
            "error_logits": self.error_head(x),
            "offset": self.offset_head(x),
            "mask_logits": self.mask_head(x),
            "scale_rotation": self.transform_head(x),
            "region_inpaint_strength": region_values[0],
            "region_keep_photo_weight": region_values[1],
        }


def repair_loss(prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    bce = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss()
    error_loss = bce(prediction["error_logits"], batch["error_targets"])
    offset_loss = smooth_l1(prediction["offset"], batch["offset_targets"])
    mask_loss = bce(prediction["mask_logits"], batch["mask_targets"])
    transform_target = torch.zeros_like(prediction["scale_rotation"])
    transform_loss = smooth_l1(prediction["scale_rotation"], transform_target)
    preservation_loss = prediction["region_inpaint_strength"] * (1.0 - prediction["region_keep_photo_weight"])
    total = (
        0.30 * error_loss
        + 0.30 * offset_loss
        + 0.15 * transform_loss
        + 0.15 * mask_loss
        + 0.10 * preservation_loss
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "error": float(error_loss.detach().cpu()),
        "offset": float(offset_loss.detach().cpu()),
        "mask": float(mask_loss.detach().cpu()),
        "transform": float(transform_loss.detach().cpu()),
        "preservation": float(preservation_loss.detach().cpu()),
    }
    return total, parts
