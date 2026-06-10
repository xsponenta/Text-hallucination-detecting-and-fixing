# Architecture and File Naming

## System Overview

The implementation combines three declared approaches:

```text
RL feedback:
  OCR/LLM feedback compares generated text with ground truth.

Refinement:
  mask and inpaint only the zone that contains text.

GNN:
  characters or character parts are graph nodes; spatial relations are edges.
  After training, the GNN predicts a deformed mask and dx/dy offsets.
```

The implementation is split into five layers:

```text
data generation
    -> graph construction
    -> GNN prediction
    -> mask/inpainting repair
    -> evaluation and quality gate
```

The GNN predicts text-structure corrections. The inpainting model handles pixels. The quality gate protects the photo from unnecessary changes.

Main benchmark tracks:

```text
TextAtlas5 / TextAtlas5M
Google DrawBench
```

## Repository Layout

```text
configs/
  gnn_repair.yaml

data/
  benchmark_manifests/
  prompt_manifests/
  benchmarks/
  synthetic/
  real_generated/
  processed_graphs/

src/
  data/
    synth_text_dataset.py
    graph_builder.py
    ocr_detector.py
  models/
    gnn_text_repair.py
  repair/
    mask_from_graph.py
    inpaint_repair.py
    quality_gate.py
  eval/
    metrics.py
    run_eval.py
    evaluate_validation.py
  demo/
    create_method_demo.py
    create_pair_dataset_demo.py
  train_gnn.py

outputs/
  checkpoints/
  debug_graphs/
  repaired_images/
```

## File Responsibilities

### `src/data/synth_text_dataset.py`

Creates supervised synthetic samples for fast GNN training.

Outputs:

```text
data/synthetic/metadata.jsonl
data/synthetic/images_clean/{sample_id}.png
data/synthetic/images_corrupted/{sample_id}.png
data/synthetic/masks/{sample_id}.png
```

Each metadata row contains expected text, corrupted image path, clean target path, repair mask path, text-region box, character boxes, and target character boxes.

### `src/data/graph_builder.py`

Converts synthetic metadata into graph tensors.

Outputs:

```text
data/processed_graphs/{sample_id}.pt
```

Each graph file contains:

```text
node_features
edge_index
edge_features
error_targets
offset_targets
mask_targets
boxes
target_boxes
image_path
mask_path
expected_text
```

### `src/data/ocr_detector.py`

Optional OCR adapter for real generated images. The current adapter supports EasyOCR when it is installed.

Output:

```text
OCRDetection(text, confidence, box)
```

### `src/models/gnn_text_repair.py`

Defines the pure-PyTorch GNN model.

Main class:

```text
GNNTextRepair
```

Predictions:

```text
error_logits
offset
mask_logits
scale_rotation
region_inpaint_strength
region_keep_photo_weight
```

### `src/train_gnn.py`

Trains the GNN from `.pt` graph files.

Output:

```text
outputs/checkpoints/gnn_text_repair_best.pt
```

### `src/repair/mask_from_graph.py`

Converts GNN predictions into soft inpainting masks.

Output naming:

```text
outputs/debug_graphs/{sample_id}_mask.png
```

### `src/repair/inpaint_repair.py`

Contains the Stable Diffusion inpainting wrapper and a debug overlay fallback.

Recommended final image naming:

```text
outputs/repaired_images/{sample_id}_repaired.png
```

### `src/repair/quality_gate.py`

Accepts or rejects a repaired image using:

```text
OCR improvement
outside-mask image difference
photo preservation threshold
```

### `src/eval/metrics.py`

Shared metrics:

```text
word accuracy
OCR accuracy
OCR F1
character error rate
exact match
mask IoU
CLIP Score
background preservation score
CLIP-Text Alignment Score
LPIPS outside text mask
FID from precomputed features
```

### `src/eval/run_eval.py`

Loads a trained checkpoint, predicts masks for graph samples, saves debug masks, and prints mean mask IoU.

### `src/eval/evaluate_validation.py`

Loads a trained checkpoint and prints validation metrics:

```text
mask IoU
offset MAE
node error accuracy
Word Accuracy
OCR Accuracy
OCR F1
CER
photo preservation proxy
```

## Naming Conventions

Use `snake_case` for Python files and functions.

Use these suffixes consistently:

```text
*_clean.png          clean target image
*_corrupted.png      corrupted input image
*_mask.png           repair or predicted mask
*_repaired.png       final repaired image
*_graph.pt           graph tensor file when manually exported
*_best.pt            best checkpoint
*_latest.pt          latest checkpoint if added later
```

Current generated graph files are named:

```text
{sample_id}.pt
```

because `sample_id` already encodes the sample identity.

## Standard Commands

Generate a small synthetic dataset:

```bash
python3 -m src.data.synth_text_dataset --count 1000 --output-dir data/synthetic --prompt-manifest data/prompt_manifests/noisy_text_examples.jsonl
```

Generate benchmark-style hard cases:

```bash
python3 -m src.data.synth_text_dataset --count 600 --output-dir data/benchmarks/textatlas5_stress --prompt-manifest data/benchmark_manifests/textatlas5_stress_prompts.jsonl --cycle-manifest
python3 -m src.data.synth_text_dataset --count 600 --output-dir data/benchmarks/drawbench_text_stress --prompt-manifest data/benchmark_manifests/drawbench_text_stress_prompts.jsonl --cycle-manifest
python3 -m src.data.synth_text_dataset --count 350 --output-dir data/benchmarks/phone_realworld_stress --prompt-manifest data/benchmark_manifests/phone_realworld_stress_prompts.jsonl --cycle-manifest
```

Build graph tensors:

```bash
python3 -m src.data.graph_builder --metadata data/synthetic/metadata.jsonl --output-dir data/processed_graphs
```

Train the GNN:

```bash
python3 -m src.train_gnn --config configs/gnn_repair.yaml
```

This training step is where the project uses learning. The GNN is trained for a few hours to predict repair masks, node error probabilities, and text-node offsets. The inpainting model is not trained; it is used as a pretrained repair backend.

Evaluate predicted masks:

```bash
python3 -m src.eval.run_eval --checkpoint outputs/checkpoints/gnn_text_repair_best.pt
```

Print validation metrics:

```bash
python3 -m src.eval.evaluate_validation --checkpoint outputs/checkpoints/gnn_text_repair_best.pt --metadata data/synthetic/metadata.jsonl --graph-dir data/processed_graphs --use-clean-target-as-repaired-text
```

Print all available reported metrics:

```bash
python3 -m src.eval.evaluate_validation --checkpoint outputs/checkpoints/gnn_text_repair_best.pt --metadata data/synthetic/metadata.jsonl --graph-dir data/processed_graphs --use-clean-target-as-repaired-text --compute-fid
```

Optional flags:

```text
--compute-clip
--compute-lpips
--compute-fid
```

`--compute-fid` uses a lightweight local feature estimate by default. Benchmark-scale FID should be computed with cached Inception features.

The validation report prints overall metrics and grouped metrics by:

```text
benchmark
case_type
```

Create a visual method demo:

```bash
python3 -m src.demo.create_method_demo --prompt-manifest data/prompt_manifests/noisy_text_examples.jsonl
```

Create a demo from real good/bad image pairs:

```bash
python3 -m src.demo.create_pair_dataset_demo --manifest path/to/pairs.jsonl
```

## First Prototype Flow

1. Generate 1,000-5,000 synthetic samples.
2. Build graph tensors.
3. Train `GNNTextRepair`.
4. Generate predicted masks.
5. Run local inpainting on masks.
6. Accept only repairs that improve OCR and keep outside-mask change low.

## Training and Validation Usage

Training is used for the GNN correction model:

```text
input: corrupted image graph + expected text layout
target: bad-node labels + dx/dy offsets + repair mask
output: trained checkpoint for mask/offset prediction
```

Short training schedule:

```text
2-4 hours: train on 2,000-8,000 synthetic photo-like samples
30 minutes: validate on 300-800 held-out samples
30-60 minutes: run inpainting on validation repairs
```

Validation metrics reported:

```text
WA: percent of fully correct words
OCR Accuracy: percent of OCR outputs matching expected text
OCR F1: precision/recall balance for recognized text tokens or characters
CER: character edit error rate
CLIP Score: image/text semantic match on TextAtlas5 and DrawBench prompts
LPIPS: outside-mask preservation around text
FID: overall realism after correction
```

Expected validation conclusion:

```text
The approach works well if WA increases, CER decreases, CLIP alignment stays stable or improves, LPIPS outside the mask remains low, and FID does not degrade after repair.
```
