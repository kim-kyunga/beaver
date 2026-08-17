# Preprocessing — WSI → H-optimus-1 features

Both training and inference consume per-slide `.h5` feature bags produced by the
**same** pipeline. At inference this runs inside the container
(`inference/pipeline/preprocess.py`); for training we pre-extract features once.

## Pipeline

1. **Tissue segmentation** — [TRIDENT](https://github.com/mahmoodlab/TRIDENT)
   (pinned commit `0b926f3`) segments tissue from each WSI. A saturation-Otsu /
   thumbnail fallback recovers slides with broken tiles or missing mpp
   (mirrors `inference/pipeline/preprocess.py`).
2. **Patching** — tissue is tiled at **20×** (mpp ≈ 0.5).
3. **Encoding** — each patch is embedded with the frozen **H-optimus-1** encoder
   (`huggingface.co/bioptimus/H-optimus-1`, gated, **1536-d**). Per-slide patch
   features are stored as an `.h5` bag `{stem}.h5` under `FEATURES_DIR`.

## Requirements

- TRIDENT + its tissue-segmentation checkpoint.
- H-optimus-1 in `$HF_HOME` (accept the HF license first; we do not redistribute).
- `openslide`, `timm`, `h5py`.

## Output

```
FEATURES_DIR/
  PIT_xx_xxxxx_xx.h5     # patch features [N_patches, 1536] + coords, one per slide
  ...
```
`FEATURES_DIR` is what `training/slot_abmil` (`data.train_data_path`) and the
retrieval-DB build consume. Feature extraction is deterministic given the same
TRIDENT checkpoint and encoder weights.

> The exact runner we used wraps the TRIDENT batch scripts
> (`run_batch_of_slides` → segmentation/patching → H-optimus-1 encoding).
> `inference/pipeline/preprocess.py` is the canonical, self-contained
> implementation of this same pipeline for a single slide.
