# Builds the base image that the main Dockerfile builds upon.
# This reconstructs the WSI feature-extraction environment (PyTorch + trident +
# openslide + timm) from public sources, so the project is reproducible without
# any private base image.
#
# Build and tag it as the name expected by the main Dockerfile:
#   docker build -f base.Dockerfile -t feature-extraction:v1 .
#   docker build -f Dockerfile      -t beaver .
#
# Versions below are pinned to the environment used for our submission.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# system libraries for whole-slide image I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# WSI feature-extraction stack
RUN pip install --no-cache-dir \
        timm==0.9.16 \
        openslide-python==1.4.3 openslide-bin==4.0.0.13 \
        opencv-python-headless==4.13.0.92 \
        h5py==3.16.0 einops==0.8.2 einops-exts==0.0.4 \
        tifffile==2023.2.28 pandas==3.0.3 matplotlib==3.10.9 tqdm \
        segmentation-models-pytorch \
        transformers==4.57.6 huggingface_hub==0.36.2

# trident: tissue segmentation + patch-encoder toolkit
RUN pip install --no-cache-dir \
        "git+https://github.com/mahmoodlab/TRIDENT.git@0b926f3"
