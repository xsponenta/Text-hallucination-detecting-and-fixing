# Text Hallucination Detection And Repair

Prototype pipeline for detecting and repairing hallucinated text in generated images. The method treats bad text as a localized structural problem: detect the corrupted text region, build a graph over character/text components, predict a repair mask and geometric offsets with a GNN, then use local inpainting to fix the text while preserving the surrounding image.

## Project Idea

Generated images often look realistic but fail on text: letters are swapped, merged, misspelled, shifted, or unreadable. Instead of regenerating the whole image, this project focuses on a targeted repair loop:

```text
prompt + expected text
  -> generated/corrupted image
  -> OCR or synthetic labels
  -> text-component graph
  -> GNN mask and offset prediction
  -> local inpainting repair
  -> OCR / CLIP / LPIPS / FID quality gate
```

The diffusion model is used as a pretrained inpainting backend. The trainable component is the GNN that plans where and how to repair text.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `src/data/` | Synthetic text data generation, OCR adapter, graph construction |
| `src/models/` | GNN text repair model |
| `src/repair/` | Mask generation, inpainting wrapper, quality gate |
| `src/eval/` | OCR/text/image quality metrics and validation entry points |
| `src/demo/` | Demo dataset and method visualization scripts |
| `configs/` | Training/repair/evaluation configuration |
| `data/prompt_manifests/` | Prompt manifests for synthetic data generation |
| `data/benchmark_manifests/` | Stress-test prompt manifests for benchmark-like splits |
| `notebooks/` | Exploratory notebook for noisy text prompt generation |

Generated datasets, graph tensors, checkpoints, repaired images, and debug outputs are intentionally ignored by git.

## Main Components

- `src/data/synth_text_dataset.py` creates clean/corrupted synthetic image pairs, masks, text boxes, and metadata.
- `src/data/graph_builder.py` converts metadata into graph tensors with node/edge features and repair targets.
- `src/models/gnn_text_repair.py` defines the pure-PyTorch GNN for error classification, offset prediction, and mask logits.
- `src/train_gnn.py` trains the graph model from processed graph tensors.
- `src/repair/mask_from_graph.py` converts predictions into soft inpainting masks.
- `src/repair/inpaint_repair.py` wraps Stable Diffusion inpainting and includes a debug overlay fallback.
- `src/repair/quality_gate.py` accepts or rejects repairs based on OCR improvement and photo preservation.
- `src/eval/evaluate_validation.py` reports validation metrics and baseline comparisons.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional dependencies enable stronger OCR and perceptual metrics:

```bash
pip install easyocr torch-geometric lpips
```

## Quick Start

Generate a synthetic training split:

```bash
python3 -m src.data.synth_text_dataset \
  --count 1000 \
  --output-dir data/synthetic \
  --prompt-manifest data/prompt_manifests/noisy_text_examples.jsonl
```

Build graph tensors:

```bash
python3 -m src.data.graph_builder \
  --metadata data/synthetic/metadata.jsonl \
  --output-dir data/processed_graphs
```

Train the GNN:

```bash
python3 -m src.train_gnn --config configs/gnn_repair.yaml
```

Evaluate the trained checkpoint:

```bash
python3 -m src.eval.evaluate_validation \
  --checkpoint outputs/checkpoints/gnn_text_repair_best.pt \
  --metadata data/synthetic/metadata.jsonl \
  --graph-dir data/processed_graphs \
  --use-clean-target-as-repaired-text
```

For heavier report metrics, add optional flags when dependencies are installed:

```bash
python3 -m src.eval.evaluate_validation \
  --checkpoint outputs/checkpoints/gnn_text_repair_best.pt \
  --metadata data/synthetic/metadata.jsonl \
  --graph-dir data/processed_graphs \
  --use-clean-target-as-repaired-text \
  --compute-fid
```

`--compute-clip` and `--compute-lpips` are also available.

## Benchmark Stress Sets

The repo includes manifests for three benchmark-style validation tracks:

- TextAtlas5 / TextAtlas5M-style text stress cases;
- Google DrawBench-style prompt-following text cases;
- phone-like real-world stress cases with glare, blur, small labels, and occlusion.

Example:

```bash
python3 -m src.data.synth_text_dataset \
  --count 600 \
  --output-dir data/benchmarks/textatlas5_stress \
  --prompt-manifest data/benchmark_manifests/textatlas5_stress_prompts.jsonl \
  --cycle-manifest
```

Then build graphs and evaluate with the same trained checkpoint:

```bash
python3 -m src.data.graph_builder \
  --metadata data/benchmarks/textatlas5_stress/metadata.jsonl \
  --output-dir data/benchmarks/textatlas5_graphs

python3 -m src.eval.evaluate_validation \
  --checkpoint outputs/checkpoints/gnn_text_repair_best.pt \
  --metadata data/benchmarks/textatlas5_stress/metadata.jsonl \
  --graph-dir data/benchmarks/textatlas5_graphs \
  --use-clean-target-as-repaired-text
```

## Demo Scripts

Create a visual synthetic method demo:

```bash
python3 -m src.demo.create_method_demo \
  --prompt-manifest data/prompt_manifests/noisy_text_examples.jsonl
```

Create a demo from real good/bad image pairs:

```bash
python3 -m src.demo.create_pair_dataset_demo --manifest path/to/pairs.jsonl
```

For the synthetic demo generated by this repo:

```bash
python3 -m src.demo.create_pair_dataset_demo \
  --manifest outputs/method_demo/synthetic_pairs/metadata.jsonl
```

## Metrics

The validation pipeline reports:

- Word Accuracy;
- OCR Accuracy;
- OCR F1;
- Character Error Rate;
- CLIP Score / CLIP-text alignment;
- LPIPS outside the text mask;
- FID for global realism;
- mask IoU, offset MAE, and node error accuracy for the GNN.

The expected repair behavior is: text metrics improve, CER decreases, CLIP alignment stays stable, and outside-mask perceptual change remains low.
