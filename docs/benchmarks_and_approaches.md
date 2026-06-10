# Declared Approaches and Benchmarks

## Declared Approaches

The project combines three complementary approaches.

### 1. RL Feedback

Use OCR and LLM feedback to compare the generated image text with the expected ground truth text.

Role in the pipeline:

```text
generated image
    -> OCR reads visible text
    -> compare OCR text with expected text
    -> LLM/OCR critic scores the result
    -> choose whether to accept, regenerate, or refine
```

The RL-style part does not need a full reinforcement learning agent for the first version. It can be implemented as a feedback loop where the reward decides which image candidate or repair candidate is best.

Reward signals:

```text
OCR accuracy
Word Accuracy
Character Error Rate
OCR confidence
CLIP text-image alignment
photo preservation score
```

### 2. Refinement / Inpainting

During correction, only the zone that contains text is masked and repaired.

Role in the pipeline:

```text
bad generated image
    -> text region mask
    -> local inpainting
    -> optional low-strength harmonization
    -> quality gate
```

The goal is to fix text without damaging the rest of the photo. This is why LPIPS outside the text mask is important.

### 3. GNN Geometry Model

The GNN represents characters or character parts as graph nodes. Edges describe spatial relations between symbols.

Role in the pipeline:

```text
OCR boxes / connected components
    -> graph nodes
    -> spatial and reading-order edges
    -> trained GNN
    -> deformed repair mask + dx/dy offsets
    -> inpainting
```

After training, the GNN predicts:

```text
bad character probability
node dx/dy displacement
soft repair mask
region inpainting strength
photo-preservation weight
```

## Benchmarks

### TextAtlas5 / TextAtlas5M

Use TextAtlas as the main text-image benchmark because it contains images with text annotations and can be adapted into clean/corrupted text pairs.

Use it for:

```text
text correctness
OCR-based validation
CLIP Score
clean/corrupted prompt generation
synthetic-to-real transfer checks
```

Reported metrics:

```text
CLIP Score
OCR Accuracy
OCR F1
CER
Word Accuracy
```

Local stress manifest:

```text
data/benchmark_manifests/textatlas5_stress_prompts.jsonl
```

### DrawBench

Use Google's DrawBench as a prompt benchmark for general text-to-image quality and prompt following.

Use it for:

```text
prompt following
general realism after correction
image-text alignment
stress testing text prompts
```

Reported metrics:

```text
CLIP Score
OCR Accuracy
OCR F1
CER
LPIPS outside text mask
FID
```

Local stress manifest:

```text
data/benchmark_manifests/drawbench_text_stress_prompts.jsonl
```

The local DrawBench-style split includes harder prompt cases such as glare, motion blur, tiny labels, low contrast, and long text.

### Phone-Like Real-World Stress

Use this local stress split to approximate harder phone-camera conditions before running on real photos.

Use it for:

```text
handheld blur
window glare
uneven shadows
tiny labels
cluttered packaging
partial occlusion
compression artifacts
```

Local stress manifest:

```text
data/benchmark_manifests/phone_realworld_stress_prompts.jsonl
```

## Validation Protocol

Evaluate every method before and after correction.

```text
baseline image
    -> OCR and CLIP metrics
    -> GNN mask/offset prediction
    -> inpainting refinement
    -> OCR, CLIP, LPIPS, FID metrics
    -> compare before vs after
```

### Text Metrics

```text
Word Accuracy (WA):
  percentage of complete words generated correctly.

OCR Accuracy:
  percentage of samples where OCR output matches expected text.

OCR F1:
  token/character-level precision and recall balance for recognized text.

Character Error Rate (CER):
  edit distance between OCR text and expected text divided by expected length.
```

Expected result:

```text
WA increases
OCR Accuracy increases
OCR F1 increases
CER decreases
```

### Alignment Metric

```text
CLIP Score / CLIP-Text Alignment Score:
  measures whether the repaired image still matches the prompt and expected text semantics.
```

Expected result:

```text
CLIP Score stays stable or improves after repair.
```

### Photo Quality Metrics

```text
LPIPS:
  computed outside the text mask to check that the surrounding image remains visually intact.

FID:
  computed over repaired images to estimate whether global image realism is preserved.
```

Expected result:

```text
LPIPS outside the mask stays low
FID does not degrade after correction
```

Implementation note:

```text
WA, OCR Accuracy, OCR F1, CER, mask IoU, offset MAE, and photo preservation proxy are computed by default.
CLIP Score and LPIPS are available through optional flags when their model dependencies are installed.
FID is available as a lightweight local feature estimate; final benchmark-scale FID should use cached Inception features.
```

## Short Training Usage

Training is used for the GNN, not for the diffusion inpainting model.

```text
2-4 hours:
  train the GNN on synthetic TextAtlas-style good/bad text pairs.

30 minutes:
  validate mask IoU, dx/dy offset error, WA, and CER.

30-60 minutes:
  run inpainting on validation images and compute CLIP Score, LPIPS, and FID.
```

This gives a learned component without making the project too heavy.

## Final Claim

The approach works well if:

```text
OCR Accuracy increases
OCR F1 increases
Word Accuracy increases
CER decreases
CLIP Score stays stable or improves
LPIPS outside the text mask stays low
FID does not degrade
```

This shows that the method improves text quality while preserving the realism and visual integrity of the generated photo.
