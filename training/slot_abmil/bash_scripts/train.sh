#!/usr/bin/env bash
# BEAVER — train the slot/ABMIL model that produced the submitted checkpoint
# (run name: hopt1run2 -> sloton/checkpoints/network_epoch_150.pth).
#
# Set the data locations first (see ../../../REPRODUCE.md). Hydra reads config
# from src/conf/config.yaml; ${PLACEHOLDER}s there resolve to these env vars.
set -e

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

# --- required paths (edit to your layout) ---
export TRAIN_JSON="${TRAIN_JSON:?set TRAIN_JSON=/path/to/train_from_CoT.json}"
export FEATURES_DIR="${FEATURES_DIR:?set FEATURES_DIR=/path/to/features_hoptimus1 (.h5 bags)}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:?set EXPERIMENT_DIR=/path/to/output}"
export SLOT_COT_DIR="${SLOT_COT_DIR:?set SLOT_COT_DIR=/path/to/train_CoT_organ}"

# dataset-stats figures are written here (created up-front so the first run works)
mkdir -p imgs

# Run from this folder with `-m src.train` so `import src` resolves.
# Full run = config default training.epochs=150 -> network_epoch_150.pth.
python3 -m src.train \
    data.experiment_name="hopt1run2" \
    training.bag_size=256 \
    training.seed=42 \
    eval.run_embedding=true \
    model.train_device="cuda:0" \
    model.eval_device="cuda:1"
# Single-GPU: set eval.run_text=false eval.run_embedding=false and both devices to cuda:0.
#
# Smoke test (verify the loop runs in ~1 min, no OpenBioLLM eval needed):
#   python3 -m src.train training.epochs=1 data.max_num_datapoints=128 \
#       eval.run_embedding=false model.train_device=cuda:0 model.eval_device=cuda:0
#   -> $EXPERIMENT_DIR/hopt1run2/checkpoints/network_epoch_001.pth
