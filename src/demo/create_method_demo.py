from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.data.synth_text_dataset import generate_dataset


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


def blend_repair_from_pair(corrupted: Image.Image, clean: Image.Image, mask: Image.Image) -> Image.Image:
    """Synthetic-pair repair used only for method visualization.

    In the real pipeline the GNN predicts this mask and diffusion inpainting fills it.
    For a local demo we use the clean target under the mask to show the intended repair.
    """

    soft_mask = mask.convert("L").filter(ImageFilterFallback.gaussian(1.2))
    return Image.composite(clean.convert("RGB"), corrupted.convert("RGB"), soft_mask)


class ImageFilterFallback:
    @staticmethod
    def gaussian(radius: float):
        from PIL import ImageFilter

        return ImageFilter.GaussianBlur(radius=radius)


def add_caption(tile: Image.Image, caption: str) -> Image.Image:
    caption_height = 34
    output = Image.new("RGB", (tile.width, tile.height + caption_height), (245, 245, 240))
    output.paste(tile.convert("RGB"), (0, caption_height))
    draw = ImageDraw.Draw(output)
    draw.text((10, 9), caption, fill=(24, 24, 24), font=load_font(15))
    return output


def make_contact_sheet(rows: list[dict], output_path: Path, tile_size: int) -> None:
    captions = ["Bad generated text", "GNN/inpaint mask", "Repaired result", "Clean target"]
    row_images: list[Image.Image] = []
    for row in rows:
        corrupted = Image.open(row["corrupted_image"]).convert("RGB").resize((tile_size, tile_size))
        clean = Image.open(row["clean_image"]).convert("RGB").resize((tile_size, tile_size))
        mask = Image.open(row["repair_mask"]).convert("L").resize((tile_size, tile_size))
        mask_rgb = Image.merge("RGB", (mask, mask, mask))
        repaired = blend_repair_from_pair(corrupted, clean, mask)
        tiles = [corrupted, mask_rgb, repaired, clean]
        captioned = [add_caption(tile, captions[index]) for index, tile in enumerate(tiles)]
        row_width = sum(tile.width for tile in captioned)
        row_height = max(tile.height for tile in captioned)
        row_canvas = Image.new("RGB", (row_width, row_height), (255, 255, 255))
        x = 0
        for tile in captioned:
            row_canvas.paste(tile, (x, 0))
            x += tile.width
        row_images.append(row_canvas)

    gap = 14
    sheet_width = max(row.width for row in row_images)
    sheet_height = sum(row.height for row in row_images) + gap * (len(row_images) - 1)
    sheet = Image.new("RGB", (sheet_width, sheet_height), (232, 232, 226))
    y = 0
    for row in row_images:
        sheet.paste(row, (0, y))
        y += row.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a visual method demo from synthetic bad/good text pairs.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/method_demo"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--prompt-manifest", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.output_dir / "synthetic_pairs"
    generate_dataset(dataset_dir, args.count, args.image_size, args.seed, None, args.prompt_manifest, cycle_manifest=True)
    rows: list[dict] = []
    with (dataset_dir / "metadata.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    make_contact_sheet(rows, args.output_dir / "method_contact_sheet.png", tile_size=args.image_size)
    print(f"Wrote demo to {args.output_dir / 'method_contact_sheet.png'}")


if __name__ == "__main__":
    main()
