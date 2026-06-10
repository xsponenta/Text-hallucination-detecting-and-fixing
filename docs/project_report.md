# Project Report: GNN-Guided Text Hallucination Repair

## Goal

The goal is to reduce text hallucination in generated images while preserving the realism of the surrounding photo.

The method combines three ideas:

```text
RL feedback:
  OCR/LLM feedback compares generated text with ground truth.

Refinement:
  only the text zone is masked and repaired with inpainting.

GNN:
  symbols or symbol parts are graph nodes, spatial relations are edges,
  and the trained model predicts masks and dx/dy offsets.
```

## Architecture Scheme

```text
Prompt + expected text
        |
Generated / corrupted image
        |
OCR or synthetic labels
        |
Text boxes / character boxes / connected components
        |
Graph construction
        |
GNN correction model
        |
Bad-node probabilities + dx/dy offsets + soft repair mask
        |
Local text-region inpainting / refinement
        |
OCR + CLIP + photo-quality validation
        |
Accept / reject / refine again
```

## How It Works

The system treats text hallucination as a **localized structural error** rather than as a full-image generation failure. Most of the image can already be visually acceptable: the background, object layout, lighting, and style may look good. The main problem is that the letters are wrong, merged, shifted, or unreadable. Because of that, the method focuses on detecting and repairing only the text area.

### Step 1: Generate or Receive an Image

The input is an image that should contain a known text string.

Example:

```text
prompt: "A storefront sign displaying the text: open cafe"
expected text: "open cafe"
generated text: "opne cafe"
```

The generated image may contain realistic visual content, but the text can be corrupted.

### Step 2: Detect the Text Region

The system needs a rough location of the text. This can come from:

```text
OCR boxes
synthetic training labels
manual benchmark labels
connected components in the text area
```

In synthetic training, the text boxes are known because the dataset generator renders both the clean and corrupted text. For real generated images, OCR such as EasyOCR or PaddleOCR can provide text boxes and confidence scores.

### Step 3: Convert Text Into a Graph

Instead of treating the text region as a flat image patch, the method builds a graph.

Each node represents a symbol-level unit:

```text
one character box
one part of a character
one connected component
one OCR text component
```

Each node stores features such as:

```text
x/y position
width and height
character index
expected character embedding
detected character embedding
OCR confidence
local visual quality
```

Edges describe spatial and reading-order relations:

```text
neighboring characters
same word
same line
nearest spatial components
baseline alignment
```

This is useful because text errors are structured. If one letter is shifted, its relation to neighboring letters changes. If characters are merged, spacing and box overlap become abnormal.

### Step 4: Predict the Correction Field With GNN

The GNN reads the graph and predicts what should be repaired.

For each node, it predicts:

```text
is this node corrupted?
how far should it move in x/y?
should this area be part of the mask?
does this local region need stronger or weaker inpainting?
```

The key outputs are:

```text
bad-node probability
dx/dy displacement
soft mask probability
photo-preservation weight
```

The GNN does not generate pixels. It acts as a geometry and mask planner.

### Step 5: Build a Deformed Repair Mask

The predicted bad nodes and offsets are converted into a soft repair mask.

The mask includes:

```text
the corrupted text location
the predicted corrected location
small padding around character boxes
blurred mask edges for smoother inpainting
```

This gives an inpainting model a precise target: repair only the broken text area.

### Step 6: Refinement With Inpainting

The masked region is passed to an inpainting model. The expected text is included in the prompt or conditioning.

The goal is:

```text
replace corrupted text with readable target text
keep background unchanged
preserve lighting and material style
avoid global image regeneration
```

This is why LPIPS is measured outside the mask: if the repair changes the rest of the image, it should be rejected.

### Step 7: Feedback and Quality Gate

After repair, the image is evaluated again.

The feedback system checks:

```text
did OCR improve?
did CER decrease?
did Word Accuracy increase?
did CLIP Score stay stable or improve?
did LPIPS outside the mask stay low?
did the image still look realistic?
```

If the answer is yes, the repair is accepted. If not, the system can reject the result or try another refinement pass. This is the RL-style feedback part of the approach.

### Why GNN Helps

A simple rectangular mask can erase too much or too little. A GNN can learn text structure:

```text
letters should follow a baseline
spacing between neighboring letters should be consistent
characters in the same word should align
wrong symbols often break local graph relations
```

Therefore the GNN can produce a better repair mask and more accurate displacement field than a simple OCR bounding box.

## Main Components

### Base Generation Model

The project assumes a Stable Diffusion-family generator.

```text
base image generation:
  Stable Diffusion / SDXL

local repair:
  Stable Diffusion Inpainting

project contribution:
  GNN mask and offset predictor before inpainting
```

The diffusion model is not trained from scratch. It is used as a pretrained image generator and inpainting backend. The trained component is the GNN, which predicts where and how text should be repaired.

### 1. RL Feedback

The feedback module compares generated text against the expected text.

Signals:

```text
OCR Accuracy
Word Accuracy
OCR F1
CER
OCR confidence
CLIP Score
photo preservation score
```

In the current implementation, this is represented as a reward/quality-gating mechanism. It can be extended into a full iterative RL loop.

### 2. Refinement / Inpainting

Instead of regenerating the whole image, the system repairs only the text region.

This protects:

```text
background
lighting
composition
object identity
photo realism
```

### 3. GNN Geometry Model

The GNN models text as a graph.

Nodes:

```text
characters
symbol parts
OCR boxes
connected components
```

Edges:

```text
left-right text order
nearest spatial neighbors
same-word relations
same-line relations
baseline alignment
```

Predictions:

```text
bad character probability
node dx/dy displacement
soft repair mask
region inpainting strength
photo-preservation weight
```

## Dataset

### Training Dataset

The training dataset is synthetic but photo-like. It contains paired examples:

```text
corrupted image
clean target image
expected text
corrupted text
repair mask
character boxes
target character boxes
benchmark label
case type
```

Generated data includes signs, posters, labels, storefronts, and window text.

### Prompt Sources

The project supports prompt manifests inspired by TextAtlas-style annotations:

```text
original_text
corrupted_text
annotation
corrupted_annotation
ops
benchmark
case_type
```

Current manifests:

```text
data/prompt_manifests/noisy_text_examples.jsonl
data/benchmark_manifests/textatlas5_stress_prompts.jsonl
data/benchmark_manifests/drawbench_text_stress_prompts.jsonl
data/benchmark_manifests/phone_realworld_stress_prompts.jsonl
```

## Dataset Splits

### Main Synthetic Split

Used for training and initial validation.

```text
dataset size: 5,000 samples
training split: 90%
validation split: 10%
split method: random graph-file split
```

### Benchmark Splits

Used for stress validation after training.

```text
TextAtlas5-style stress split
DrawBench-style text stress split
Phone-like real-world stress split
```

Each benchmark split contains hard cases such as:

```text
tiny text
long text
low contrast
dense labels
occlusion
glare
motion blur
compression artifacts
handheld phone-like conditions
```

## Training Setup

Only the GNN is trained. The inpainting model remains pretrained.

Training uses paired samples:

```text
input graph:
  corrupted character boxes and spatial relations

targets:
  corrupted-node labels
  clean target boxes
  dx/dy offsets
  repair mask
```

The model learns to answer:

```text
which text parts are wrong?
where should those parts move?
which pixels should be edited?
how conservative should the repair be?
```

Training objective:

```text
L = BCE(error_pred, error_gt)
  + SmoothL1(offset_pred, offset_gt)
  + SmoothL1(scale_rotation_pred, scale_rotation_gt)
  + BCE/Dice(mask_pred, mask_gt)
  + photo_preservation_penalty
```

Current config:

```text
epochs: 30
min_epochs: 12
batching: one graph at a time in the prototype
optimizer: AdamW
learning rate: 0.001
validation fraction: 0.1
early stopping patience: 5
device: auto
```

Expected training time:

```text
small prototype: minutes
report-scale run: 2-4 hours
maximum target: under 10 hours
```

The current prototype trains quickly because the model is lightweight and the synthetic labels are clean. Larger benchmark runs can increase data size and case complexity to make the training closer to a real experimental setup.

## Validation Metrics

### Text Metrics

```text
Word Accuracy (WA):
  percentage of complete words generated correctly.

OCR Accuracy:
  exact OCR match with expected text.

OCR F1:
  character-level or word-level precision/recall balance.

Character Error Rate (CER):
  edit distance divided by expected text length.
```

Additional text-comparison metrics for method comparison:

```text
Exact Match Accuracy:
  OCR_text == expected_text.

Character Accuracy:
  1 - CER.

Normalized Levenshtein Similarity:
  1 - edit_distance / max(len(expected), len(predicted)).

Text Preservation Gain:
  metric_after - metric_before.
```

These metrics can compare:

```text
before repair
simple rectangular-mask baseline
GNN-guided repair
```

### Geometry Metrics

```text
mask IoU:
  overlap between predicted repair mask and target repair mask.

offset MAE:
  average dx/dy error for predicted node displacement.

node error accuracy:
  accuracy of corrupted-node detection.
```

### Image Quality Metrics

```text
CLIP Score:
  image-text semantic alignment.

LPIPS outside text mask:
  checks whether the surrounding photo remains visually intact.

FID:
  estimates global image realism after repair.
```

The implementation includes a lightweight local FID estimate. Final benchmark-scale FID should use cached Inception features.

## Validation Procedure

Validation is performed in two modes.

### 1. Geometry Validation

This checks whether the GNN learned the repair structure.

```text
predicted mask vs target mask -> mask IoU
predicted dx/dy vs target dx/dy -> offset MAE
predicted bad-node labels vs ground truth -> node error accuracy
```

This stage does not require diffusion inpainting. It directly validates the learned graph model.

### 2. Repair Quality Validation

This checks text and photo quality before and after repair.

Current prototype mode:

```text
before repair: corrupted synthetic text
after repair: clean target text upper bound
```

Final end-to-end mode:

```text
before repair: generated/corrupted image
after repair: actual inpainted image
OCR runs on both images
metrics compare real before/after OCR results
```

The current results prove that the GNN predicts the necessary correction region and displacement. The final end-to-end evaluation should replace the clean-target upper bound with OCR from actual inpainted outputs.

## Baseline Comparison

The method is compared with a simple baseline.

### Simple Baseline

```text
mask:
  one rectangular mask around observed corrupted character boxes

offset:
  zero dx/dy prediction

node prediction:
  all text nodes are treated as corrupted

graph reasoning:
  not used
```

This baseline represents a simple refinement strategy: find the observed corrupted text boxes and inpaint that rectangle without predicting corrected target positions.

### GNN Method

```text
mask:
  soft deformed mask created from bad-node scores and predicted offsets

offset:
  learned dx/dy prediction for each node

node prediction:
  learned corrupted-node probability

graph reasoning:
  uses spatial and reading-order relations between symbols
```

Comparison metrics:

```text
GNN mask IoU vs simple mask IoU
GNN offset MAE vs zero-offset MAE
GNN node accuracy vs all-bad baseline accuracy
```

The purpose of this comparison is to show whether graph reasoning improves the repair plan beyond a basic rectangular OCR mask.

## Benchmarks

### TextAtlas5 / TextAtlas5M

Used for text-image validation and OCR-based evaluation.

Metrics:

```text
CLIP Score
OCR Accuracy
OCR F1
CER
WA
```

### Google DrawBench

Used for prompt-following and text rendering stress tests.

Metrics:

```text
CLIP Score
OCR Accuracy
OCR F1
CER
LPIPS
FID
```

### Phone-Like Real-World Stress

Used to approximate difficult real-world phone images.

Cases:

```text
handheld blur
glare
shadow
tiny label
cluttered packaging
partial occlusion
compression artifacts
```

## Current Results

### Main Synthetic Validation

```text
samples: 5000
mask IoU: 0.6571
offset MAE normalized: 0.0077
node error accuracy: 0.9964

Before repair:
  WA: 0.3043
  OCR Accuracy: 0.0000
  OCR F1: 0.9419
  CER: 0.1557

After repair upper bound:
  WA: 1.0000
  OCR Accuracy: 1.0000
  OCR F1: 1.0000
  CER: 0.0000

CLIP before: 0.3039
CLIP after: 0.3143
LPIPS outside mask: 0.0132
background preservation: 0.9714
```

### TextAtlas5-Style Stress Validation

```text
samples: 600
mask IoU: 0.6253
offset MAE normalized: 0.0081
node error accuracy: 0.9974

Before repair:
  WA: 0.2222
  OCR Accuracy: 0.0000
  OCR F1: 0.9125
  CER: 0.1283

After repair upper bound:
  WA: 1.0000
  OCR Accuracy: 1.0000
  OCR F1: 1.0000
  CER: 0.0000

CLIP before: 0.2907
CLIP after: 0.3022
LPIPS outside mask: 0.0127
background preservation: 0.9705
```

## Interpretation

The GNN learns the geometry of text corruption well:

```text
node error accuracy is close to 1.0
offset error is around 4 pixels on 512x512 images
mask IoU is stable across normal and stress validation
```

The image-preservation metrics are also strong:

```text
LPIPS outside the mask is low
background preservation score is high
CLIP Score improves slightly after repair
```

This means the method is doing the intended job at the planning level:

```text
it finds where text is broken
it predicts where corrected text should be located
it creates a local mask instead of changing the full image
it keeps the area outside the mask visually stable
```

The strongest evidence is the combination of high node error accuracy and low offset error. The GNN is not only detecting that the text is wrong; it is also learning the spatial correction needed for refinement.

Important limitation:

```text
The current "after repair" text scores use the synthetic clean target as an upper bound.
For final real evaluation, these should be replaced with OCR results from actual inpainted repaired images.
```

## Commands

### Train

```bash
python3 -m src.data.synth_text_dataset \
  --count 5000 \
  --output-dir data/synthetic \
  --prompt-manifest data/prompt_manifests/noisy_text_examples.jsonl

python3 -m src.data.graph_builder \
  --metadata data/synthetic/metadata.jsonl \
  --output-dir data/processed_graphs

python3 -m src.train_gnn \
  --config configs/gnn_repair.yaml
```

### Validate Main Split

```bash
python3 -m src.eval.evaluate_validation \
  --checkpoint outputs/checkpoints/gnn_text_repair_best.pt \
  --metadata data/synthetic/metadata.jsonl \
  --graph-dir data/processed_graphs \
  --use-clean-target-as-repaired-text \
  --compute-clip \
  --compute-lpips \
  --compute-fid
```

### Validate Phone-Like Stress Split

```bash
python3 -m src.data.synth_text_dataset \
  --count 350 \
  --output-dir data/benchmarks/phone_realworld_stress \
  --prompt-manifest data/benchmark_manifests/phone_realworld_stress_prompts.jsonl \
  --cycle-manifest

python3 -m src.data.graph_builder \
  --metadata data/benchmarks/phone_realworld_stress/metadata.jsonl \
  --output-dir data/benchmarks/phone_realworld_graphs

python3 -m src.eval.evaluate_validation \
  --checkpoint outputs/checkpoints/gnn_text_repair_best.pt \
  --metadata data/benchmarks/phone_realworld_stress/metadata.jsonl \
  --graph-dir data/benchmarks/phone_realworld_graphs \
  --use-clean-target-as-repaired-text \
  --compute-fid
```

## Next Step

The next improvement is to replace the clean-target upper bound with actual inpainted outputs:

```text
GNN predicted mask
    -> Stable Diffusion inpainting
    -> OCR on repaired image
    -> real WA / OCR Accuracy / OCR F1 / CER
    -> CLIP / LPIPS / FID on actual repaired images
```

This will turn the current prototype validation into a full end-to-end repair evaluation.
