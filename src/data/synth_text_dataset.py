from __future__ import annotations

import argparse
import json
import random
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass
class CharacterBox:
    char: str
    box: tuple[int, int, int, int]
    target_box: tuple[int, int, int, int]
    is_corrupted: int


@dataclass
class SyntheticSample:
    sample_id: str
    expected_text: str
    corrupted_text: str
    clean_image: str
    corrupted_image: str
    repair_mask: str
    text_region_box: tuple[int, int, int, int]
    char_boxes: list[CharacterBox]
    background_type: str
    font_id: str
    prompt: str
    benchmark: str
    case_type: str


TEXT_MARKER = "displaying the text:"

DEFAULT_TEXTS = [
    "the library",
    "hotel",
    "open cafe",
    "fresh bakery",
    "city market",
    "gallery",
    "book store",
    "summer sale",
    "train station",
    "coffee bar",
    "pharmacy",
    "studio 24",
    "airport gate",
]

SCENE_PROMPTS = [
    "realistic street photo of a small hotel sign on a plaster wall",
    "close up product photo of a cardboard box with a printed label",
    "photo of a cafe window with a painted sign",
    "realistic poster on a concrete wall with soft daylight",
    "storefront sign photographed from the street",
]


def load_font(size: int, font_path: str | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def extract_text_from_annotation(annotation: str) -> tuple[str, str | None]:
    idx = annotation.find(TEXT_MARKER)
    if idx == -1:
        return annotation, None
    prefix = annotation[: idx + len(TEXT_MARKER)]
    text = annotation[idx + len(TEXT_MARKER) :].strip()
    return prefix, text


def load_prompt_manifest(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "original_text" not in row and "annotation" in row:
                _, text = extract_text_from_annotation(row["annotation"])
                row["original_text"] = text
            rows.append(row)
    return rows


def corrupt_text(text: str, rng: random.Random) -> str:
    if len(text) < 2:
        return text
    chars = list(text)
    operations = ["substitute", "delete", "transpose", "repeat", "space"]
    for _ in range(rng.randint(1, 2)):
        op = rng.choice(operations)
        non_space = [i for i, char in enumerate(chars) if char != " "]
        if not non_space:
            continue
        i = rng.choice(non_space)
        if op == "substitute":
            alphabet = string.ascii_lowercase if chars[i].islower() else string.ascii_uppercase
            chars[i] = rng.choice(alphabet + "0123456789")
        elif op == "delete" and len(chars) > 3:
            chars.pop(i)
        elif op == "transpose" and i < len(chars) - 1:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "repeat":
            chars.insert(i, chars[i])
        elif op == "space":
            if " " in chars and rng.random() < 0.5:
                chars = [char for char in chars if char != " "]
            else:
                chars.insert(min(len(chars), i + 1), " ")
    corrupted = "".join(chars)
    return corrupted if corrupted != text else text + rng.choice("x7")


def choose_text_pair(prompt_rows: list[dict], rng: random.Random) -> tuple[str, str, str, dict]:
    if prompt_rows:
        row = rng.choice(prompt_rows)
        expected = str(row.get("original_text") or row.get("expected_text") or "").strip()
        corrupted = str(row.get("corrupted_text") or "").strip()
        prompt = str(row.get("corrupted_annotation") or row.get("annotation") or rng.choice(SCENE_PROMPTS))
        if expected:
            return expected, corrupted or corrupt_text(expected, rng), prompt, row

    expected = rng.choice(DEFAULT_TEXTS)
    prompt = f"{rng.choice(SCENE_PROMPTS)} displaying the text: {expected}"
    return expected, corrupt_text(expected, rng), prompt, {"benchmark": "synthetic", "case_type": "standard"}


def make_lit_texture(size: tuple[int, int], rng: random.Random, base_a: tuple[int, int, int], base_b: tuple[int, int, int]) -> Image.Image:
    width, height = size
    a = np.array(base_a, dtype=np.float32)
    b = np.array(base_b, dtype=np.float32)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    light = 0.70 * yy + 0.30 * xx
    arr = (1.0 - light) * a + light * b
    noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(0, 5.5, (height, width, 3))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.35))


def add_camera_finish(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    arr = np.asarray(image).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    distance = ((xx - width * 0.52) / width) ** 2 + ((yy - height * 0.48) / height) ** 2
    vignette = np.clip(1.0 - distance * rng.uniform(0.35, 0.65), 0.72, 1.0)[..., None]
    arr *= vignette
    arr += np.random.default_rng(rng.randint(0, 1_000_000)).normal(0, 1.8, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.25)))


def draw_scene_base(size: int, rng: random.Random, case_type: str = "standard") -> tuple[Image.Image, tuple[int, int, int, int], str]:
    style_pool = ["storefront_sign", "paper_poster", "product_label", "window_sign"]
    if case_type in {"dense_label", "tiny_text", "phone_clutter"}:
        style_pool = ["product_label", "paper_poster"]
    elif case_type in {"low_contrast", "occlusion", "phone_shadow", "phone_glare", "phone_blur"}:
        style_pool = ["window_sign", "storefront_sign", "paper_poster"]
    style = rng.choice(style_pool)
    image = make_lit_texture((size, size), rng, (115, 105, 92), (188, 178, 160))
    draw = ImageDraw.Draw(image, mode="RGBA")

    if style == "storefront_sign":
        wall = make_lit_texture((size, size), rng, (120, 116, 105), (194, 188, 172))
        image.paste(wall)
        for x in range(-40, size, 34):
            draw.line((x, 0, x + rng.randint(-12, 12), size), fill=(80, 80, 80, 24), width=1)
        sign = (int(size * 0.15), int(size * 0.20), int(size * 0.85), int(size * 0.42))
        draw.rounded_rectangle((sign[0] + 7, sign[1] + 9, sign[2] + 7, sign[3] + 9), radius=8, fill=(0, 0, 0, 70))
        draw.rounded_rectangle(sign, radius=8, fill=(235, 226, 202, 255), outline=(62, 56, 47, 190), width=3)
        for x in (sign[0] + 25, sign[2] - 25):
            draw.ellipse((x - 4, sign[1] + 8, x + 4, sign[1] + 16), fill=(70, 65, 58, 220))
        return image, sign, style

    if style == "paper_poster":
        wall = make_lit_texture((size, size), rng, (95, 101, 104), (175, 178, 176))
        image.paste(wall)
        poster = (int(size * 0.22), int(size * 0.12), int(size * 0.78), int(size * 0.82))
        draw.rectangle((poster[0] + 5, poster[1] + 7, poster[2] + 5, poster[3] + 7), fill=(0, 0, 0, 55))
        draw.rectangle(poster, fill=(232, 229, 213, 255), outline=(210, 205, 188, 255), width=2)
        for _ in range(18):
            y = rng.randint(poster[1] + 12, poster[3] - 12)
            draw.line((poster[0] + 18, y, poster[2] - 18, y + rng.randint(-2, 2)), fill=(170, 160, 135, 28), width=1)
        return image, (poster[0] + 24, poster[1] + 70, poster[2] - 24, poster[1] + 165), style

    if style == "product_label":
        table = make_lit_texture((size, size), rng, (80, 67, 54), (168, 140, 112))
        image.paste(table)
        box = (int(size * 0.18), int(size * 0.18), int(size * 0.82), int(size * 0.78))
        draw.rounded_rectangle((box[0] + 9, box[1] + 14, box[2] + 9, box[3] + 14), radius=13, fill=(0, 0, 0, 55))
        draw.rounded_rectangle(box, radius=12, fill=(197, 151, 92, 255), outline=(113, 82, 48, 180), width=2)
        label = (box[0] + 35, box[1] + 92, box[2] - 35, box[1] + 190)
        draw.rounded_rectangle(label, radius=6, fill=(242, 237, 221, 248), outline=(116, 100, 82, 130), width=2)
        return image, label, style

    glass = make_lit_texture((size, size), rng, (47, 75, 87), (111, 150, 164))
    image.paste(glass)
    for _ in range(5):
        x0 = rng.randint(-80, size)
        draw.rectangle((x0, 0, x0 + rng.randint(20, 55), size), fill=(255, 255, 255, rng.randint(18, 34)))
    frame = (int(size * 0.12), int(size * 0.15), int(size * 0.88), int(size * 0.78))
    draw.rectangle(frame, outline=(35, 42, 44, 230), width=8)
    sign = (int(size * 0.22), int(size * 0.36), int(size * 0.78), int(size * 0.52))
    draw.rounded_rectangle((sign[0] + 5, sign[1] + 5, sign[2] + 5, sign[3] + 5), radius=6, fill=(0, 0, 0, 52))
    draw.rounded_rectangle(sign, radius=6, fill=(245, 239, 211, 235), outline=(50, 47, 40, 150), width=2)
    return image, sign, style


def adjust_panel_for_case(box: tuple[int, int, int, int], size: int, case_type: str) -> tuple[int, int, int, int]:
    if case_type in {"tiny_text", "phone_tiny"}:
        width = max(90, int((box[2] - box[0]) * 0.58))
        height = max(28, int((box[3] - box[1]) * 0.48))
        cx = int((box[0] + box[2]) * 0.5)
        cy = int((box[1] + box[3]) * 0.5)
        return (max(0, cx - width // 2), max(0, cy - height // 2), min(size, cx + width // 2), min(size, cy + height // 2))
    if case_type in {"long_text", "phone_long_text"}:
        return (max(0, box[0] - 20), box[1], min(size, box[2] + 20), box[3])
    return box


def fit_font(text: str, box: tuple[int, int, int, int], font_path: str | None, rng: random.Random) -> ImageFont.ImageFont:
    max_width = max(20, box[2] - box[0] - 18)
    max_height = max(18, box[3] - box[1] - 12)
    for size in range(max_height, 11, -1):
        font = load_font(size, font_path)
        bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            return font
    return load_font(max(12, min(max_height, 18)), font_path)


def centered_origin(text: str, font: ImageFont.ImageFont, box: tuple[int, int, int, int]) -> tuple[int, int]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return (int(box[0] + (box[2] - box[0] - text_width) * 0.5), int(box[1] + (box[3] - box[1] - text_height) * 0.5 - bbox[1]))


def text_boxes(text: str, font: ImageFont.ImageFont, origin: tuple[int, int]) -> list[tuple[str, tuple[int, int, int, int]]]:
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    cursor_x, origin_y = origin
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for char in text:
        bbox = draw.textbbox((cursor_x, origin_y), char, font=font)
        boxes.append((char, bbox))
        cursor_x += max(1, int(draw.textlength(char, font=font)))
    return boxes


def draw_scene_text(
    image: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    origin: tuple[int, int],
    rng: random.Random,
    broken: bool = False,
    case_type: str = "standard",
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    if case_type in {"low_contrast", "phone_shadow"}:
        ink = rng.choice([(185, 180, 164), (205, 201, 184), (98, 112, 116)])
    else:
        ink = rng.choice([(28, 28, 25), (55, 42, 34), (18, 45, 61), (95, 28, 32), (230, 235, 224)])
    shadow = tuple(max(0, value - 45) for value in ink)
    draw.text((origin[0] + 2, origin[1] + 2), text, font=font, fill=shadow)
    draw.text(origin, text, font=font, fill=ink)
    if broken:
        boxes = text_boxes(text, font, origin)
        for _, box in boxes:
            if rng.random() < 0.35:
                draw.line((box[0], box[1] + rng.randint(2, 8), box[2], box[1] + rng.randint(0, 8)), fill=(235, 235, 220), width=rng.randint(1, 3))
            if rng.random() < 0.25:
                draw.rectangle((box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), outline=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)), width=1)
        if rng.random() < 0.65:
            result = result.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.55)))
    return result


def add_case_artifacts(image: Image.Image, text_region: tuple[int, int, int, int], rng: random.Random, case_type: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result, mode="RGBA")
    x0, y0, x1, y1 = text_region
    if case_type in {"occlusion", "phone_occlusion"}:
        for _ in range(rng.randint(1, 3)):
            ox0 = rng.randint(x0, max(x0, x1 - 8))
            oy0 = rng.randint(y0, max(y0, y1 - 8))
            ox1 = min(x1, ox0 + rng.randint(12, max(14, (x1 - x0) // 3)))
            oy1 = min(y1, oy0 + rng.randint(8, max(10, (y1 - y0) // 2)))
            draw.rounded_rectangle((ox0, oy0, ox1, oy1), radius=3, fill=(30, 30, 30, rng.randint(95, 145)))
    elif case_type in {"glare", "phone_glare"}:
        draw.polygon(
            [(x0 - 20, y1), (x0 + 40, y0), (x0 + 90, y0), (x0 + 25, y1)],
            fill=(255, 255, 255, 65),
        )
    elif case_type in {"motion_blur", "phone_blur"}:
        result = result.filter(ImageFilter.GaussianBlur(radius=0.8))
    elif case_type in {"dense_label", "phone_clutter"}:
        for y in range(y0 - 35, y1 + 55, 13):
            draw.line((x0 - 10, y, x1 + 10, y + rng.randint(-1, 1)), fill=(80, 75, 65, 35), width=1)
        if case_type == "phone_clutter":
            for _ in range(12):
                bx0 = rng.randint(max(0, x0 - 55), min(result.width - 10, x1 + 20))
                by0 = rng.randint(max(0, y0 - 65), min(result.height - 10, y1 + 65))
                draw.rectangle((bx0, by0, bx0 + rng.randint(12, 50), by0 + rng.randint(3, 12)), fill=(25, 25, 25, rng.randint(18, 48)))
    elif case_type == "phone_shadow":
        draw.polygon(
            [(0, y0 - 60), (result.width, y0 - 10), (result.width, y1 + 80), (0, y1 + 30)],
            fill=(0, 0, 0, 70),
        )
    elif case_type == "phone_compression":
        arr = np.asarray(result).astype(np.float32)
        arr = np.round(arr / 18.0) * 18.0
        noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(0, 3.0, arr.shape)
        result = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return result


def build_char_records(
    expected_text: str,
    corrupted_text: str,
    font: ImageFont.ImageFont,
    target_origin: tuple[int, int],
    corrupted_origin: tuple[int, int],
) -> list[CharacterBox]:
    target_boxes = text_boxes(expected_text, font, target_origin)
    corrupted_boxes = text_boxes(corrupted_text, font, corrupted_origin)
    records: list[CharacterBox] = []
    for index, (char, target_box) in enumerate(target_boxes):
        if index < len(corrupted_boxes):
            bad_char, bad_box = corrupted_boxes[index]
            is_corrupted = int(bad_char != char or bad_box != target_box or len(corrupted_text) != len(expected_text))
            source_box = bad_box
        else:
            is_corrupted = 1
            source_box = target_box
        records.append(CharacterBox(char=char, box=source_box, target_box=target_box, is_corrupted=is_corrupted))
    return records


def padded_region(boxes: list[tuple[str, tuple[int, int, int, int]]], size: int, padding: int = 12) -> tuple[int, int, int, int]:
    xs = [value for _, box in boxes for value in (box[0], box[2])]
    ys = [value for _, box in boxes for value in (box[1], box[3])]
    return (max(0, min(xs) - padding), max(0, min(ys) - padding), min(size, max(xs) + padding), min(size, max(ys) + padding))


def create_sample(
    sample_id: str,
    output_dir: Path,
    rng: random.Random,
    image_size: int,
    font_path: str | None,
    prompt_rows: list[dict],
) -> SyntheticSample:
    expected_text, corrupted_text, prompt, prompt_row = choose_text_pair(prompt_rows, rng)
    expected_text = re.sub(r"\s+", " ", expected_text.strip())
    corrupted_text = re.sub(r"\s+", " ", corrupted_text.strip())
    benchmark = str(prompt_row.get("benchmark") or "synthetic")
    case_type = str(prompt_row.get("case_type") or "standard")
    base, text_panel, style = draw_scene_base(image_size, rng, case_type=case_type)
    text_panel = adjust_panel_for_case(text_panel, image_size, case_type)
    font = fit_font(expected_text.upper() if rng.random() < 0.45 else expected_text, text_panel, font_path, rng)

    render_expected = expected_text.upper() if rng.random() < 0.65 else expected_text
    render_corrupted = corrupted_text.upper() if render_expected.isupper() else corrupted_text
    target_origin = centered_origin(render_expected, font, text_panel)
    jitter = (rng.randint(-5, 5), rng.randint(-4, 4))
    corrupted_origin = (target_origin[0] + jitter[0], target_origin[1] + jitter[1])

    target_boxes = text_boxes(render_expected, font, target_origin)
    corrupted_boxes = text_boxes(render_corrupted, font, corrupted_origin)
    text_region = padded_region(target_boxes + corrupted_boxes, image_size, padding=16)
    clean = draw_scene_text(base, render_expected, font, target_origin, rng, broken=False, case_type=case_type)
    corrupted = draw_scene_text(base, render_corrupted, font, corrupted_origin, rng, broken=True, case_type=case_type)
    clean = add_case_artifacts(clean, text_region, rng, case_type)
    corrupted = add_case_artifacts(corrupted, text_region, rng, case_type)
    clean = add_camera_finish(clean, rng)
    corrupted = add_camera_finish(corrupted, rng)

    repair_mask = Image.new("L", (image_size, image_size), 0)
    mask_draw = ImageDraw.Draw(repair_mask)
    mask_draw.rounded_rectangle(text_region, radius=6, fill=255)
    repair_mask = repair_mask.filter(ImageFilter.GaussianBlur(radius=2.0))
    char_records = build_char_records(render_expected, render_corrupted, font, target_origin, corrupted_origin)

    clean_path = output_dir / "images_clean" / f"{sample_id}.png"
    corrupted_path = output_dir / "images_corrupted" / f"{sample_id}.png"
    mask_path = output_dir / "masks" / f"{sample_id}.png"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    corrupted_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    clean.save(clean_path, quality=95)
    corrupted.save(corrupted_path, quality=95)
    repair_mask.save(mask_path)
    return SyntheticSample(
        sample_id=sample_id,
        expected_text=render_expected,
        corrupted_text=render_corrupted,
        clean_image=str(clean_path),
        corrupted_image=str(corrupted_path),
        repair_mask=str(mask_path),
        text_region_box=text_region,
        char_boxes=char_records,
        background_type=style,
        font_id=font_path or "system_default",
        prompt=prompt,
        benchmark=benchmark,
        case_type=case_type,
    )


def generate_dataset(
    output_dir: Path,
    count: int,
    image_size: int,
    seed: int,
    font_path: str | None,
    prompt_manifest: Path | None,
    cycle_manifest: bool = False,
) -> None:
    rng = random.Random(seed)
    prompt_rows = load_prompt_manifest(prompt_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            active_rows = prompt_rows
            if cycle_manifest and prompt_rows:
                active_rows = [prompt_rows[index % len(prompt_rows)]]
            sample = create_sample(f"sample_{index:06d}", output_dir, rng, image_size, font_path, active_rows)
            row = asdict(sample)
            row["char_boxes"] = [asdict(item) for item in sample.char_boxes]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {count} photo-like samples to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate photo-like synthetic text repair data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--font-path", type=str, default=None)
    parser.add_argument("--prompt-manifest", type=Path, default=None)
    parser.add_argument("--cycle-manifest", action="store_true")
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.count, args.image_size, args.seed, args.font_path, args.prompt_manifest, args.cycle_manifest)


if __name__ == "__main__":
    main()
