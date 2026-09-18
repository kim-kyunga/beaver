FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        timm==0.9.16 \
        openslide-python==1.4.3 openslide-bin==4.0.0.13 \
        opencv-python-headless==4.13.0.92 \
        h5py==3.16.0 einops==0.8.2 einops-exts==0.0.4 \
        tifffile==2023.2.28 pandas==3.0.3 matplotlib==3.10.9 tqdm \
        segmentation-models-pytorch \
        transformers==4.57.6 huggingface_hub==0.36.2

RUN pip install --no-cache-dir \
        "git+https://github.com/mahmoodlab/TRIDENT.git@0b926f3"
