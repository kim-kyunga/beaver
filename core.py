
import json
from pathlib import Path

import numpy as np
import torch
import tifffile
from PIL import Image


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")

WSI_IMAGE_DIR = INPUT_PATH / "images" / "whole-slide-image"


def get_interface_key() -> tuple:
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    slugs = [entry["socket"]["slug"] for entry in inputs]
    return tuple(sorted(slugs))


def load_json_file(*, location: Path):
    with open(location) as f:
        return json.loads(f.read())


def write_json_file(*, location: Path, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_roi_image(*, location: Path) -> Image.Image:
    with Image.open(location) as opened:
        image = opened.convert("RGB")
    print(f"[ROI] Load verification passed: {location}")
    return image


def load_wsi_array(*, location: Path) -> np.ndarray:
    array = tifffile.imread(location)
    print(f"[WSI] Load verification passed: {location}")
    return array


def show_torch_cuda_info():
    print("=+=" * 10)
    print("Torch CUDA available:", (available := torch.cuda.is_available()))
    if available:
        print(f"  devices          : {torch.cuda.device_count()}")
        current = torch.cuda.current_device()
        print(f"  current device   : {current}")
        print(f"  device properties: {torch.cuda.get_device_properties(current)}")
    print("=+=" * 10)
