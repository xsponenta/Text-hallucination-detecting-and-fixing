from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


IMAGE_KEYS_BAD = ["bad_image", "corrupted_image", "image_corrupted", "noisy_image"]
IMAGE_KEYS_GOOD = ["good_image", "clean_image", "image_clean", "target_image"]
MASK_KEYS = ["repair_mask", "mask", "text_mask", "mask_image"]


def first_existing(row: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return None


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base_dir / path


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def caption(tile: Image.Image, text: str) -> Image.Image:
    output = Image.new("RGB", (tile.width, tile.height + 34), (245, 245, 240))
    output.paste(tile.convert("RGB"), (0, 34))
    ImageDraw.Draw(output).text((10, 9), text, fill=(20, 20, 20), font=load_font(15))
    return output


def mask_from_difference(bad: Image.Image, good: Image.Image) -> Image.Image:
    bad_gray = bad.convert("L")
    good_gray = good.resize(bad.size).convert("L")
    diff = Image.new("L", bad.size, 0)
    bad_px = bad_gray.load()
    good_px = good_gray.load()
    out_px = diff.load()
    for y in range(bad.height):
        for x in range(bad.width):
            out_px[x, y] = min(255, abs(bad_px[x, y] - good_px[x, y]) * 4)
    return diff.filter(ImageFilter.GaussianBlur(radius=2.0))


def repaired_from_pair(bad: Image.Image, good: Image.Image, mask: Image.Image) -> Image.Image:
    soft_mask = mask.convert("L").resize(bad.size).filter(ImageFilter.GaussianBlur(radius=1.2))
    return Image.composite(good.resize(bad.size).convert("RGB"), bad.convert("RGB"), soft_mask)


def read_rows(manifest: Path, limit: int) -> list[dict]:
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}. "
            "Replace path/to/pairs.jsonl with a real file, or use "
            "outputs/method_demo/synthetic_pairs/metadata.jsonl after running create_method_demo."
        )
    rows: list[dict] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def make_demo(manifest: Path, output_path: Path, limit: int, tile_size: int) -> None:
    base_dir = manifest.parent
    row_canvases: list[Image.Image] = []
    for row in read_rows(manifest, limit):
        bad_path = first_existing(row, IMAGE_KEYS_BAD)
        good_path = first_existing(row, IMAGE_KEYS_GOOD)
        if not bad_path or not good_path:
            continue
        bad = Image.open(resolve_path(bad_path, base_dir)).convert("RGB").resize((tile_size, tile_size))
        good = Image.open(resolve_path(good_path, base_dir)).convert("RGB").resize((tile_size, tile_size))
        mask_path = first_existing(row, MASK_KEYS)
        if mask_path:
            mask = Image.open(resolve_path(mask_path, base_dir)).convert("L").resize((tile_size, tile_size))
        else:
            mask = mask_from_difference(bad, good)
        repaired = repaired_from_pair(bad, good, mask)
        tiles = [
            caption(bad, "Bad generated text"),
            caption(Image.merge("RGB", (mask, mask, mask)), "Predicted/GT mask"),
            caption(repaired, "Local repaired result"),
            caption(good, "Good target"),
        ]
        row_canvas = Image.new("RGB", (tile_size * 4, tile_size + 34), (255, 255, 255))
        x = 0
        for tile in tiles:
            row_canvas.paste(tile, (x, 0))
            x += tile_size
        row_canvases.append(row_canvas)

    if not row_canvases:
        raise ValueError("No usable rows found. Manifest needs bad_image/corrupted_image and good_image/clean_image paths.")

    gap = 14
    sheet = Image.new("RGB", (tile_size * 4, len(row_canvases) * (tile_size + 34) + gap * (len(row_canvases) - 1)), (232, 232, 226))
    y = 0
    for row in row_canvases:
        sheet.paste(row, (0, y))
        y += row.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"Wrote pair dataset demo to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a method demo from existing good/bad image pairs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/method_demo/pair_dataset_contact_sheet.png"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args()
    try:
        make_demo(args.manifest, args.output, args.limit, args.tile_size)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
