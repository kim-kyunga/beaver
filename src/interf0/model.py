from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from core import load_json_file, load_roi_image, MODEL_PATH

ANSWER_BACKGROUND = (
    "This region is non-informative background with no assessable tissue; "
    "no diagnostic or morphologic assessment can be made."
)
ANSWER_TISSUE = (
    "Tissue is visible in this region; histologic structures are present and assessable."
)

ROI_SIZE = 224

GATE_CLF_PATH = os.environ.get("GATE_CLF_PATH", str(MODEL_PATH / "metricB" / "gate_clf_hopt.joblib"))
HOPT_HF = os.environ.get("HOPT_HF", str(MODEL_PATH / "hoptimus1_hf"))

_HOPT_MEAN = np.array((0.707223, 0.578729, 0.703617), dtype=np.float32)
_HOPT_STD = np.array((0.211883, 0.230117, 0.177517), dtype=np.float32)

B2_SAT_THR = float(os.environ.get("COT_B2_SAT_THR", "0.05"))

_ENC = None
_CLF = None
_DEVICE = None


def _lazy_load():
    global _ENC, _CLF, _DEVICE
    if _ENC is not None:
        return
    import torch
    import joblib
    import timm

    os.environ.setdefault("HF_HOME", HOPT_HF)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _ENC = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1", pretrained=True,
        init_values=1e-5, dynamic_img_size=False).eval().to(_DEVICE)
    _CLF = joblib.load(GATE_CLF_PATH)
    print(f"[interf0] H-opt gate loaded (device={_DEVICE}, clf={GATE_CLF_PATH})")


def _embed(roi_image):
    import torch

    img = roi_image.convert("RGB").resize((ROI_SIZE, ROI_SIZE), Image.BICUBIC)
    arr = ((np.asarray(img, np.float32) / 255.0 - _HOPT_MEAN) / _HOPT_STD).transpose(2, 0, 1)
    x = torch.from_numpy(arr).unsqueeze(0).to(_DEVICE)
    with torch.inference_mode():
        if _DEVICE == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                feat = _ENC(x)
        else:
            feat = _ENC(x)
    return feat.float().cpu().numpy()


def _tissue_frac(roi_image) -> float:
    a = np.asarray(roi_image.convert("RGB"), dtype=np.float32)
    mx = a.max(2); mn = a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    val = mx / 255.0
    return float(((sat > 0.10) & (val > 0.20) & (val < 0.92)).mean())


def _is_tissue(roi_image) -> bool:
    if int(_CLF.predict(_embed(roi_image))[0]) == 1:
        return True
    return _tissue_frac(roi_image) >= B2_SAT_THR


def predict_visual_context_response(
    *,
    question_path: Path,
    roi_image_path: Path,
) -> str:
    _ = load_json_file(location=question_path)
    roi_image = load_roi_image(location=roi_image_path)

    _lazy_load()
    if _is_tissue(roi_image):
        return ANSWER_TISSUE
    return ANSWER_BACKGROUND
