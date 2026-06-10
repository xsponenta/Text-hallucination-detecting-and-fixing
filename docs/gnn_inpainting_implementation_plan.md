# GNN-Guided Text Hallucination Repair Plan

## Goal

Fix hallucinated text in generated images while keeping the whole photo visually natural.

The proposed system does not ask the GNN to generate pixels. Instead, the GNN predicts where the text geometry is wrong and how the local text region should move, stretch, or be masked. A pretrained inpainting model then repairs only the necessary pixels, and a quality gate rejects fixes that improve OCR but damage the photo.

Working name: **Scene-Preserving Graph Text Repair**.

## Core Idea

Generated text usually fails in structured ways:

- characters are shifted, merged, duplicated, or missing;
- strokes are malformed;
- spacing between letters is inconsistent;
- text baseline is curved or broken;
- the surrounding photo is often good and should not be regenerated.

This makes the problem a good fit for a graph model. Text is not just pixels; it has structure. Each character, stroke component, or OCR box can be a node, and spatial/reading-order relationships can be edges.

The GNN predicts a **correction field**:

- which symbols are wrong;
- which regions need inpainting;
- where corrected text should be placed;
- how much each text component should move;
- how much the repair is allowed to modify the original image.

Then inpainting performs the visual repair inside the predicted mask.

## Declared Method Components

The full method uses three connected components:

```text
RL feedback:
  OCR/LLM feedback compares generated text with the expected ground truth.

Refinement:
  during correction, mask and inpaint only the region that contains text.

GNN:
  symbols or symbol parts are graph nodes, and spatial relations are edges.
  After training, the GNN predicts a deformed mask and local dx/dy shifts.
```

This gives the system a feedback stage, a local repair stage, and a learned geometry stage.

## Pipeline

```text
Prompt + expected text
        |
Generated image
        |
OCR + text region detection
        |
Character/component graph construction
        |
GNN correction model
        |
Repair mask + text layout + offset field
        |
Text-conditioned inpainting
        |
Photo quality + OCR quality gate
        |
Final corrected image
```

## Model Architecture

### 1. Text Region Detector

Use OCR to find candidate text regions.

Recommended first version:

- EasyOCR or PaddleOCR for text boxes;
- optionally CRAFT for better character-level text localization;
- synthetic labels during training when available.

Output:

```text
text boxes
character boxes or connected components
OCR text
OCR confidence
```

### 2. Graph Builder

Create one graph per text region.

Node options:

- first prototype: one node per OCR character box;
- stronger version: one node per connected component or stroke cluster;
- fallback: one node per small patch inside the text box.

Node features:

```text
normalized x, y, width, height
OCR confidence
character index in expected text
detected character embedding
expected character embedding
small visual crop embedding
foreground/background contrast
local blur score
```

Edge types:

```text
left-right reading order edges
nearest spatial neighbor edges
same-word edges
same-line edges
baseline alignment edges
expected-text sequence edges
```

Edge features:

```text
relative dx, dy
distance
angle
box overlap
height ratio
same_line flag
sequence distance
```

### 3. GNN Correction Model

Recommended architecture:

```text
small CNN patch encoder
        +
GraphSAGE or GATv2 layers
        +
node heads + region heads
```

Use **GATv2** if possible because attention can learn which neighboring characters matter most. Use **GraphSAGE** if training speed becomes more important than accuracy.

Predictions:

```text
node_error_probability
node_offset_dx_dy
node_scale_delta
node_rotation_delta
node_mask_weight
region_inpaint_strength
region_keep_photo_weight
```

The important extra output is `region_keep_photo_weight`. It penalizes unnecessary image changes and helps protect the photo from being over-edited.

### 4. Inpainting Repair

Use an existing pretrained image inpainting model instead of training a diffusion model from scratch.

Recommended:

- Stable Diffusion inpainting;
- SDXL inpainting if hardware allows;
- ControlNet inpainting if better layout control is needed.

Inputs to inpainting:

```text
original generated image
GNN repair mask
expected text
estimated text box/layout
low denoising strength
```

Start with a conservative denoising range:

```text
0.20 - 0.40 for text region only
0.05 - 0.15 for final full-image harmonization
```

Do not regenerate the whole image unless the text region is too damaged.

## Dataset Requirements

The dataset must teach both text correctness and photo preservation.

### Minimum Dataset Size

For a prototype under 10 training hours:

```text
10,000 - 30,000 synthetic text-region samples
1,000 - 3,000 full-image samples with text
500 - 1,000 validation samples
```

This is enough for the GNN because it learns geometry and masks, not full image generation.

### Required Fields Per Sample

Each training sample should contain:

```text
image_corrupted: generated or synthetically corrupted image
image_clean: target image or clean text rendering
expected_text: correct text string
detected_text: OCR result from corrupted image
text_region_box: bounding box around text
char_boxes_corrupted: character/component boxes before repair
char_boxes_target: expected character/component boxes after repair
repair_mask_gt: region that should be edited
preserve_mask_gt: region that should not be changed
font_id or font_style if synthetic
background_type
```

### Synthetic Data Generation

Generate clean text on realistic backgrounds, then corrupt the text.

Background sources:

- generated photos from the existing Stable Diffusion pipeline;
- public image datasets with no important text;
- simple scene-like backgrounds for fast pretraining.

Text rendering variations:

```text
fonts: sans, serif, handwritten, bold, condensed
sizes: small signs to large posters
colors: high contrast, low contrast, neon, shadowed
perspective: flat, tilted, curved surface approximation
lighting: blur, noise, compression, shadow
```

Corruptions:

```text
character shift
character swap
missing character
extra character
merged characters
broken strokes
random stroke blobs
baseline wave
wrong spacing
local blur
low contrast
partial occlusion
```

Ground truth targets:

- `error_probability = 1` for corrupted nodes;
- `offset_dx_dy = target_center - corrupted_center`;
- `scale_delta` and `rotation_delta` from target box transform;
- `repair_mask_gt` from the difference between corrupted and clean text region;
- `preserve_mask_gt` outside the repair region.

### Real Generated Data

Synthetic data is necessary, but the validation set must include real failures from the image generator.

Collect:

```text
prompt
expected_text
generated_image
OCR output
manual label: pass/fail
manual text region mask for a small subset
```

Manual labels do not need to be large. Even 200-500 real generated examples can expose whether the synthetic training transfers.

## Where Training Is Used

Training is used only for the lightweight GNN correction model. The inpainting model stays pretrained.

The GNN is trained to predict:

- corrupted character/component probability;
- local `dx/dy` offset for each text node;
- soft repair mask weights;
- conservative inpainting strength;
- photo-preservation weight around the text area.

This makes the method more than a hand-written mask heuristic, but still keeps the experiment short. The GNN learns text layout and geometry; diffusion inpainting performs the pixel-level repair.

Recommended short training setup:

```text
training data: 2,000-8,000 synthetic good/bad pairs
validation data: 300-800 held-out synthetic pairs
real check set: 50-200 generated/photo-like examples if available
training time: 2-4 hours on one GPU
maximum time: under 10 hours
```

For the report, the important point is that training is used to improve mask and offset prediction before inpainting. We do not fine-tune Stable Diffusion.

## Training Plan Under 10 Hours

Assumption: one consumer GPU with 12-24 GB VRAM. If only CPU is available, train on fewer samples and keep the CNN very small.

### Hour 0-1: Dataset Builder

- Render synthetic text on photo-like backgrounds.
- Save clean and corrupted versions.
- Save character boxes and displacement targets.
- Export graph samples in PyTorch Geometric format.

### Hour 1-2: OCR and Graph Construction

- Integrate OCR for generated images.
- Build graph edges from spatial proximity and reading order.
- Verify graph visualization on 20-50 samples.

### Hour 2-4: GNN Implementation

- Implement CNN crop encoder.
- Implement 3-layer GATv2 or GraphSAGE.
- Add node heads for error, offset, scale, rotation, and mask weight.
- Add region head for inpaint strength.

### Hour 4-7: GNN Training

Train only the GNN and small CNN encoder.

Loss:

```text
L = 0.30 * BCE(error_pred, error_gt)
  + 0.30 * SmoothL1(offset_pred, offset_gt)
  + 0.15 * SmoothL1(scale_rotation_pred, scale_rotation_gt)
  + 0.15 * DiceBCE(mask_pred, mask_gt)
  + 0.10 * photo_preservation_penalty
```

Suggested settings:

```text
epochs: 20-40
batch size: 16-64 graphs
optimizer: AdamW
learning rate: 1e-3 to 3e-4
mixed precision: enabled
early stopping: validation OCR/layout score
```

Shorter report-friendly version:

```text
2-4 hours: train GNN on synthetic photo-like pairs
30 minutes: validate masks and offsets
30-60 minutes: run inpainting and final metrics
```

### Hour 7-8: Inpainting Integration

- Convert GNN predictions to a soft inpainting mask.
- Warp or reposition the expected text layout.
- Run local inpainting on only the text region.
- Run optional low-strength full-image harmonization.

### Hour 8-10: Evaluation and Tuning

Evaluate both text quality and photo quality.

Text metrics:

```text
Word Accuracy (WA)
OCR exact match
Character Error Rate (CER)
word error rate
OCR confidence
mean node offset error
mask IoU
```

Photo quality metrics:

```text
CLIP-Text Alignment Score
LPIPS outside text mask
SSIM outside text mask
FID
CLIP-IQA or aesthetic score
background preservation score
manual visual pass/fail
```

Final reported validation metrics:

```text
WA: percentage of fully correct generated words after repair.
OCR Accuracy: percentage of samples where OCR matches the expected text.
OCR F1: precision/recall balance for recognized text at token or character level.
CER: character-level edit distance divided by expected text length.
CLIP Score / CLIP-Text Alignment Score: checks whether the repaired image still matches the prompt/expected text semantics.
LPIPS: computed outside the text mask to verify that surrounding photo content remains intact.
FID: compares repaired images with clean/realistic target distribution to estimate global realism after correction.
```

Benchmark tracks:

```text
TextAtlas5 / TextAtlas5M:
  CLIP Score and OCR metrics: Accuracy, F1, CER.

Google DrawBench:
  prompt following, CLIP Score, OCR metrics, LPIPS, and FID after correction.
```

Final acceptance rule:

```text
accept repair only if:
  WA improves or CER decreases
  AND CLIP-Text Alignment does not decrease significantly
  AND outside-mask LPIPS stays below threshold
  AND FID does not get worse on the validation set
  AND no large color/lighting mismatch appears
```

## Implementation Milestones

### Milestone 1: GNN Mask Predictor

Input:

```text
corrupted image
expected text
character/component boxes
```

Output:

```text
repair mask
bad character probabilities
```

Success:

```text
mask IoU > simple OCR-box baseline
```

### Milestone 2: Offset Field Predictor

Output:

```text
per-node dx/dy correction
```

Success:

```text
mean offset error decreases on synthetic validation set
```

### Milestone 3: Inpainting Repair

Output:

```text
corrected image
```

Success:

```text
OCR improves while outside-text LPIPS remains low
```

### Milestone 4: Quality Gate

Reject bad repairs automatically.

Success:

```text
fewer cases where text improves but photo quality becomes worse
```

## Recommended Repository Structure

```text
data/
  synthetic/
  real_generated/
  processed_graphs/

src/
  data/
    synth_text_dataset.py
    graph_builder.py
  models/
    gnn_text_repair.py
  repair/
    mask_from_graph.py
    inpaint_repair.py
    quality_gate.py
  eval/
    metrics.py
    run_eval.py

configs/
  gnn_repair.yaml

outputs/
  debug_graphs/
  repaired_images/
```

## Main Risks

1. OCR character boxes may be unreliable on heavily hallucinated text.
   - Mitigation: train with connected components and fallback patch nodes.

2. Inpainting may change the photo too much.
   - Mitigation: use soft local masks, low denoising, and a preservation quality gate.

3. Synthetic data may not transfer to real generated failures.
   - Mitigation: include real generated validation samples and fine-tune the GNN on a small manually labeled set.

4. Correct text may look pasted on.
   - Mitigation: estimate font color, lighting, perspective, blur, and use low-strength harmonization after inpainting.

## Best First Experiment

Start with one-line signs, posters, labels, and product text.

Keep the first prototype narrow:

```text
image size: 512x512
text regions: 1-2 per image
text length: 3-20 characters
language: English uppercase/lowercase
GNN nodes: character boxes or connected components
inpainting: local text region only
training time: 3-5 hours for GNN
total experiment time: under 10 hours
```

This version is small enough to build quickly, but it still tests the real hypothesis: **a graph model can understand text layout errors better than a plain mask or OCR-box heuristic, while inpainting keeps the image photorealistic**.
