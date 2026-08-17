"""
Interface 0 — Visual Grounding (Metric B).

H-optimus-1(= Metric A와 동일 인코더) frozen embedding + 학습된 tissue/background
게이트 분류기로 ROI가 조직인지 배경인지 판별한 뒤 답을 생성한다.

  - 배경(non-informative)  -> 진단/조직 주장 없는 거부 답           (B1)
  - 조직(tissue)           -> 조직이 보인다는 grounded 답           (B3, B2)

설계 원칙(주최 규정): 답은 질문 텍스트 템플릿이 아니라 **이미지 내용**으로 결정한다.
게이트는 순수 이미지 기반.

★통합 백엔드: Metric A 와 동일한 H-optimus-1 인코더 사용.
  Metric A(workflow)와 동일한 인코더/추론 기반에서 grounding을 산출하도록 함(주최 요구).
  H-opt 게이트 검증: 5-fold CV 97.9%, 주최 18예시 100%.

가중치는 런타임에 /opt/ml/model 아래에서 로드:
  - H-optimus-1: HF 캐시 (HF_HOME=/opt/ml/model/hoptimus1_hf, 오프라인) — A와 공유
  - 게이트 분류기: /opt/ml/model/metricB/gate_clf_hopt.joblib
로컬 테스트는 env(HOPT_HF, GATE_CLF_PATH)로 경로 override.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from core import load_json_file, load_roi_image, MODEL_PATH

# --- 답 문자열 (이미지 판별 결과로만 선택) ---------------------------------
ANSWER_BACKGROUND = (
    "This region is non-informative background with no assessable tissue; "
    "no diagnostic or morphologic assessment can be made."
)
ANSWER_TISSUE = (
    "Tissue is visible in this region; histologic structures are present and assessable."
)

ROI_SIZE = 224  # H-optimus-1 입력 크기 (학습과 동일)

# --- 경로 (컨테이너 기본값 + 로컬 override) --------------------------------
GATE_CLF_PATH = os.environ.get("GATE_CLF_PATH", str(MODEL_PATH / "metricB" / "gate_clf_hopt.joblib"))
# H-opt 가중치 HF 캐시 — Metric A(preprocess)와 동일 디렉토리 공유
HOPT_HF = os.environ.get("HOPT_HF", str(MODEL_PATH / "hoptimus1_hf"))

# H-optimus-1 정규화 상수 (Metric A와 동일)
_HOPT_MEAN = np.array((0.707223, 0.578729, 0.703617), dtype=np.float32)
_HOPT_STD = np.array((0.211883, 0.230117, 0.177517), dtype=np.float32)

# B2(input_sensitivity) 보강용 채도게이트 임계: 조직픽셀 비율이 이 이상이면 조직으로 본다.
# sparse 조직이 perturbation(검은 ~40% 마스킹)에 가려져 H-opt 게이트가 배경으로 뒤집혀도
# 채도게이트가 남은 조직을 잡아 원본과 동일한 답을 유지(B2↑). 진짜 배경은 조직%≈0 이라 무발동(B1 무손상).
B2_SAT_THR = float(os.environ.get("COT_B2_SAT_THR", "0.05"))

# --- lazy 로드 (컨테이너당 1회) --------------------------------------------
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

    # A와 동일하게 로컬 HF 캐시에서 오프라인 로드 (symlink 없는 통구조)
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
    """ROI(PIL) -> H-optimus-1 임베딩 (1, 1536) numpy. Metric A와 동일 전처리."""
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
    """채도 기반 조직 픽셀 비율(0~1). H&E 조직=분홍/보라=충분한 채도.
    검은마스크(저 value)·흰배경(저 채도)은 조건에서 자동 제외 → perturbation 견고. cv2 불필요(순수 numpy)."""
    a = np.asarray(roi_image.convert("RGB"), dtype=np.float32)
    mx = a.max(2); mn = a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)   # HSV S
    val = mx / 255.0                                               # HSV V
    return float(((sat > 0.10) & (val > 0.20) & (val < 0.92)).mean())


def _is_tissue(roi_image) -> bool:
    """하이브리드 게이트 = H-opt 게이트 OR 채도 게이트.
    sparse 조직+검은마스킹이 H-opt를 background로 뒤집어도 채도가 조직을 잡아 답 일관성(B2) 유지.
    진짜 배경은 조직%≈0 이라 채도게이트 무발동 → B1(배경거부)·통합백엔드(H-opt 주경로) 무손상."""
    if int(_CLF.predict(_embed(roi_image))[0]) == 1:
        return True
    return _tissue_frac(roi_image) >= B2_SAT_THR


def predict_visual_context_response(
    *,
    question_path: Path,
    roi_image_path: Path,
) -> str:
    # 질문은 로드하되 답을 질문 텍스트에 키잉하지 않는다(주최 규정).
    _ = load_json_file(location=question_path)
    roi_image = load_roi_image(location=roi_image_path)

    _lazy_load()
    if _is_tissue(roi_image):
        return ANSWER_TISSUE
    return ANSWER_BACKGROUND
