<p align="center">
  <img src="assets/beaver.jpg" width="200" alt="BEAVER">
</p>

# BEAVER

**BEAVER**, a branch-aware, confidence-gated reasoner, is a
perception-grounded, structured chain-of-thought system for whole-slide
pathology images. This repository contains the **inference code** for our
submission to the REG2026 challenge.

## Team

**PathWise**

## Method overview

Given a whole-slide image (WSI), BEAVER produces a structured diagnostic
chain-of-thought (CoT) and a final pathology report:

```
WSI
 └─ tissue segmentation + patching            (pipeline/preprocess.py)
     └─ patch encoder  → patch features
         └─ attention-based MIL (ABMIL)        (model_lib/src/network)
             └─ slide embedding z
                 ├─ slot classifiers           (per-attribute predictions)
                 └─ retrieval over a reference corpus  (pipeline/retrieval.py)
                     └─ chain-of-thought assembly      (pipeline/assemble_cot.py)
```

Two auxiliary reasoning components:
- **Self-verification** (`COT_SELF_VERIFY`): the #1-diagnosis-anchored consistency
  check/repair over slot predictions, emitting a reasoning trace.
- **Counterfactual reasoning** (`pipeline/counterfactual.py`, `COT_COUNTERFACTUAL`):
  attribute-constrained retrieval that re-derives the conclusion and downstream
  chain when a hypothetical modification is injected. It is **fully dormant**
  (byte-identical output) during normal inference.

## Repository layout

```
inference.py              Grand Challenge entry point
core.py                   shared I/O helpers
src/interf0/              interface 0: visual-context (tissue/background) response
src/interf1/              interface 1: chain-of-thought (runs the pipeline in a subprocess)
pipeline/                 CoT inference pipeline
  preprocess.py             WSI -> segmentation -> patch features
  lean.py                   single forward pass (ABMIL -> z)
  retrieval.py              report / chain retrieval over the reference corpus
  assemble_cot.py           chain-of-thought assembly + slot filling
  counterfactual.py         Metric C: attribute-constrained retrieval
  common.py / infer.py      config + subprocess entry
model_lib/src/            model definition (ABMIL + decoder), imported as `src.*`
Dockerfile                container recipe
requirements.txt          extra dependencies (beyond the base image)
```

## Weights and data

Our **trained weights are included in this repository** (`weights/`, 44 MB): the
ABMIL + slot checkpoint, the reference retrieval corpus, answer codebooks, and the
interface-0 gate. Only the public foundation models (the H-optimus-1 patch encoder
and the trident tissue segmenter) are downloaded from their original sources — see
[Downloading the weights](#downloading-the-weights) below.

At runtime everything is assembled under `/opt/ml/model` (the Grand Challenge
convention). Expected layout:

```
/opt/ml/model/
  sloton/                 ABMIL / slot checkpoint (+ config)      [in repo]
  cot_artifacts/          retrieval corpus (embeddings, reports, chains)  [in repo]
  codebooks/              answer codebooks                        [in repo]
  metricB/                tissue/background gate classifier (interface 0) [in repo]
  hoptimus1_hf/           H-optimus-1 encoder HF cache (offline)  [public download]
  trident_seg/            tissue-segmentation checkpoint          [public / auto]
```

Environment variables (see `Dockerfile`) override every path; defaults point to
the mount above.

### Downloading the weights

BEAVER needs three groups of weights. The parts we trained are small and shipped
**in this repository**; the two public/large components are fetched separately.
Assemble them into one directory (`./model` below), which is mounted at
`/opt/ml/model`.

**1. Trained weights — in this repo (44 MB).** ABMIL + slot heads, the retrieval
index, knowledge bases, answer codebooks, and the interface-0 gate:

```bash
mkdir -p ./model
tar xzf weights/containerA7_model_slim.tar.gz -C ./model
#  -> sloton/  cot_artifacts/  codebooks/  metricB/
```

**2. Patch encoder — H-optimus-1 (public, ~4.3 GB).** Fetched into an offline HF
cache that the container reads through `HF_HOME`:

```bash
HF_HOME=./model/hoptimus1_hf huggingface-cli download bioptimus/H-optimus-1
```

**3. Tissue-segmentation checkpoint (public).** BEAVER uses TRIDENT's default
DeepLabV3 tissue segmenter (`deeplabv3_seg_v4.ckpt`), which the
[trident](https://github.com/mahmoodlab/TRIDENT) package downloads automatically
on first use — no separate download or token needed. For a fully offline run (as
on Grand Challenge), copy that checkpoint to
`./model/trident_seg/deeplabv3_seg_v4.ckpt`; the container repoints trident's
registry to that path.

**4. Report retrieval (public).** The report and reasoning-chain retrieval index
ships in this repository (step 1); no separate download is required.

After these steps `./model` matches the `/opt/ml/model` layout shown above.
Everything the reproducer needs is either in this repository (trained weights) or
downloaded from its public source (H-optimus-1, the trident segmenter) — no
private access beyond this repository is required.

## Build and run

The main `Dockerfile` builds on a base image (`feature-extraction:v1`)
that provides the WSI feature-extraction stack — PyTorch, the
[trident](https://github.com/mahmoodlab/TRIDENT) toolkit, `timm`, and
`openslide`. That base is **not public**, so `base.Dockerfile` reconstructs an
equivalent from public sources (pinned versions). Build the base first, then the
app:

```bash
# 1) build the base image from public sources
docker build -f base.Dockerfile -t feature-extraction:v1 .

# 2) build BEAVER on top of it
docker build -f Dockerfile -t beaver .

# 3) run (weights mounted read-only)
docker run --rm --gpus all --network none \
  --volume /path/to/input:/input:ro \
  --volume /path/to/output:/output \
  --volume /path/to/weights:/opt/ml/model:ro \
  beaver
```

The container guarantees exact reproduction of the reported results given the
same weights and reference corpus.

## Training (reproducing the weights)

All training code is under [`training/`](training/). Each stage is independent and
writes into the artifact layout the container consumes:

| folder | stage | output |
|---|---|---|
| [`preprocess/`](training/preprocess/) | WSI → H-optimus-1 `.h5` patch features (TRIDENT + encoder) | `FEATURES_DIR/*.h5` |
| [`slot_abmil/`](training/slot_abmil/) | **ABMIL + slot model** — the submitted checkpoint | `sloton/checkpoints/network_epoch_150.pth` |
| [`seq_head/`](training/seq_head/) | per-organ sequence heads (chain structure / edges) | `cot_artifacts/seq_heads.pt` |
| [`retrieval_db/`](training/retrieval_db/) | reference retrieval index (embeddings, reports, chains) | `cot_artifacts/db/` |
| [`metric_b_gate/`](training/metric_b_gate/) | interface-0 tissue/background gate | `metricB/` |

The core model is **`slot_abmil/`** — the file to run is
[`src/train.py`](training/slot_abmil/src/train.py) (Hydra config
[`src/conf/config.yaml`](training/slot_abmil/src/conf/config.yaml)). Set four data
paths, then run **from the `slot_abmil/` folder** with `-m src.train` (so
`import src` resolves):

```bash
cd training/slot_abmil
export TRAIN_JSON=/path/to/train_from_CoT.json     # per-slide CoT + labels
export FEATURES_DIR=/path/to/features_hoptimus1    # .h5 bags from preprocess/
export SLOT_COT_DIR=/path/to/train_CoT_organ       # per-organ slot codebooks
export EXPERIMENT_DIR=/path/to/output
export HF_HOME=/path/to/hoptimus1_hf               # H-optimus-1 cache

# smoke test — confirm the loop runs (~1 min: 1 epoch, 128 slides, eval off):
python -m src.train training.epochs=1 data.max_num_datapoints=128 \
    eval.run_embedding=false model.train_device=cuda:0 model.eval_device=cuda:0
#  -> $EXPERIMENT_DIR/hopt1run2/checkpoints/network_epoch_001.pth

# full submitted checkpoint — 150 epochs:
bash bash_scripts/train.sh
```

Both commands are verified end-to-end. See [`REPRODUCE.md`](REPRODUCE.md) for the
exact data files, run order, and the downstream stages (`seq_head`,
`retrieval_db`, `metric_b_gate`).

## Example output

`example/sample_cot_output.json` is a real chain-of-thought produced by the
pipeline for one breast core-needle-biopsy slide. Each step is
`{"question", "answer", "next_question"}`:

```json
[
  {"question": "What is the organ?", "answer": "Breast",
   "next_question": "Is there any abnormality present?"},
  {"question": "What is the procedure?", "answer": "Core needle biopsy",
   "next_question": "Is there any abnormality present?"}
]
```

## Citation

A REG2026 challenge paper describing this work is in preparation; the citation
will be added here once it is available.

## License

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE) and
[NOTICE](NOTICE).
