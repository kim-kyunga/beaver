# Reproducing BEAVER

Two phases: **(A) train the weights** from WSIs, then **(B) run inference**.
The `README.md` has the short version; this file lists the exact inputs, run
order, and outputs for each stage under [`training/`](training/).

## Prerequisites

- **H-optimus-1** patch encoder — gated on Hugging Face
  (`huggingface.co/bioptimus/H-optimus-1`); accept the licence, then it lives in
  `$HF_HOME`. We do not redistribute it.
- **TRIDENT** (pinned commit `0b926f3`) + its tissue-segmentation checkpoint, for
  segmentation/patching.
- The REG2026 **training data**: per-slide WSIs + `train_from_CoT.json`
  (each slide's chain-of-thought and slot labels).
- Python deps: see [`requirements.txt`](requirements.txt) plus `openslide`,
  `timm`, `h5py`.

## Phase A — training

### 0. Features — `training/preprocess/`
WSI → H-optimus-1 `.h5` patch bags (20×, mpp ≈ 0.5). Same pipeline as inference
(`pipeline/preprocess.py`). See [`training/preprocess/README.md`](training/preprocess/README.md).
→ output: `FEATURES_DIR/{stem}.h5` (one bag per slide, `[N_patches, 1536]`).

### 1. Slot / ABMIL model — `training/slot_abmil/` (core)
Hydra config: [`src/conf/config.yaml`](training/slot_abmil/src/conf/config.yaml).
Paths are supplied through environment variables:

Run **from `training/slot_abmil/`** with `-m src.train` (module form, so
`import src` resolves). Four data paths are required:

```bash
cd training/slot_abmil
export TRAIN_JSON=/path/to/train_from_CoT.json     # per-slide CoT + labels
export FEATURES_DIR=/path/to/features_hoptimus1    # .h5 bags from step 0
export SLOT_COT_DIR=/path/to/train_CoT_organ       # per-organ slot codebooks (required)
export EXPERIMENT_DIR=/path/to/output              # checkpoints written here
export HF_HOME=/path/to/hoptimus1_hf               # H-optimus-1 cache

# smoke test (verify the loop runs, ~1 min) — 1 epoch, 128 slides, eval off:
python -m src.train training.epochs=1 data.max_num_datapoints=128 \
    eval.run_embedding=false model.train_device=cuda:0 model.eval_device=cuda:0
#  -> $EXPERIMENT_DIR/hopt1run2/checkpoints/network_epoch_001.pth

# full submitted checkpoint — 150 epochs:
bash bash_scripts/train.sh
```
→ output: `sloton/checkpoints/network_epoch_150.pth` (run name `hopt1run2`).

> `eval.run_embedding=true` (default in `train.sh`) loads a large OpenBioLLM-8B
> report evaluator on `eval_device`; set it to `false` (as in the smoke command)
> to skip it. On a single GPU, point both `train_device`/`eval_device` at `cuda:0`.

### 2. Sequence heads — `training/seq_head/`
Per-organ linear heads that predict chain structure/edges from the ABMIL
embedding `z`.
```bash
python training/seq_head/build_seq_labels.py    # train_from_CoT.json -> seq_labels.json
python training/seq_head/train_seq_head.py       # z + seq_labels -> seq_heads.pt (300 epochs, tiny)
```
→ output: `cot_artifacts/seq_heads.pt`, `seq_vocab.json`.

### 3. Retrieval index — `training/retrieval_db/`
Builds the reference corpus (embeddings + reports + chains) that branch-matched
retrieval reads at inference.
```bash
python training/retrieval_db/build_full_db.py
```
→ output: `cot_artifacts/db/` (`db_z.npy`, `db_reports.json`, `db_chains.json`, `db_organ.json`).

### 4. Interface-0 gate — `training/metric_b_gate/`
Generates the tissue/background ROIs for the visual-grounding gate (Metric B).
→ output: `metricB/`.

> **Note.** The `seq_head/` and `retrieval_db/` scripts were written against our
> internal layout (`from nar2_common import ART, ORGANS`, absolute `/srv/...`
> paths). Set `ART` / the paths at the top of each script to your `cot_artifacts`
> directory before running. `slot_abmil/` and `preprocess/` are the fully
> parameterised, self-contained stages.

## Phase B — inference

The trained artifacts assemble into the `/opt/ml/model` layout in the main
[`README.md`](README.md#weights-and-data). Then either:

- **Container (recommended, exact):** build and run per
  [Build and run](README.md#build-and-run) — `inference.py` runs
  `pipeline/ (preprocess → lean → retrieval → assemble_cot)` and writes the CoT +
  report to `/output`.
- A worked example output is in
  [`example/sample_cot_output.json`](example/sample_cot_output.json).
