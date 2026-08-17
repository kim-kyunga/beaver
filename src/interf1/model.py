"""
Interface 1 — Workflow Reasoning (Metric A): WSI -> chain-of-thought.

trident(seg+coords+features) + PIPELINE(slot-direct CoT) 통합.
MODEL 패키지가 `src` 라는 이름을 써서 제출 템플릿의 `src`(interf0/interf1)와
같은 프로세스에서 충돌 → MODEL forward 는 subprocess(infer.py)로 격리한다.
(컨테이너=케이스 1개라 프로세스 1회 기동 비용은 문제 없음.)

코드(이미지 /opt/app):  model_lib(src), pipeline(cot)
가중치(런타임 /opt/ml/model):  sloton(config/idx/ckpt), cot_artifacts, codebooks,
                              hoptimus1_hf, trident_seg(HEST)
모두 env override 가능(로컬 테스트).
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict

from core import MODEL_PATH


class ChainOfThoughtStep(TypedDict):
    question: str
    answer: str
    next_question: str


# 코드 위치 (이미지에 COPY)
MODEL_TRAIN_DIR = os.environ.get("MODEL_TRAIN_DIR", "/opt/app/model_lib")
COT_PIPELINE_DIR = os.environ.get("COT_PIPELINE_DIR", "/opt/app/pipeline")
# 가중치 위치 (model.tar.gz -> /opt/ml/model)
SLOTON_DIR = os.environ.get("SLOTON_DIR", str(MODEL_PATH / "sloton"))
COT_CKPT = os.environ.get(
    "COT_CKPT", str(MODEL_PATH / "sloton" / "checkpoints" / "network_epoch_150.pth"))
CODEBOOK_DIR = os.environ.get("CODEBOOK_DIR", str(MODEL_PATH / "codebooks"))
COT_ART = os.environ.get("COT_ART", str(MODEL_PATH / "cot_artifacts"))
TRAIN_JSON = os.environ.get("TRAIN_JSON", str(MODEL_PATH / "sloton" / "train_from_CoT.json"))

# 안전망: WSI 처리가 어떤 이유로든(OOM/읽기실패/에러) 죽어도 유효한 CoT 를 반드시 내놓는다.
# OOM 은 자식 subprocess 만 죽이고 부모(inference.py)는 살아 여기서 잡힌다 → 그 케이스는
# "실패(crash)" 대신 "낮은 점수" 가 되어 제출 전체가 유효해진다. (컨테이너 전체 타임아웃은 못 잡음)
_FALLBACK_COT: list[ChainOfThoughtStep] = [
    {"question": "What is the organ?",
     "answer": "Unspecified", "next_question": "What is the procedure?"},
    {"question": "What is the procedure?",
     "answer": "Surgical resection", "next_question": "Is there any abnormality present?"},
    {"question": "Is there any abnormality present?",
     "answer": "Yes, there is an abnormality.",
     "next_question": "What is the final pathology report?"},
    {"question": "What is the final pathology report?",
     "answer": "Microscopic examination of the submitted specimen shows tissue "
               "consistent with the clinical context.",
     "next_question": ""},
]


def predict_chain_of_thought(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    try:
        return _predict_impl(wsi_path=wsi_path)
    except Exception:
        import traceback
        print("[interf1] pipeline failed; returning fallback CoT.\n" + traceback.format_exc())
        return _FALLBACK_COT


def _predict_impl(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    tmpdir = tempfile.mkdtemp(prefix="interf1_")
    out_json = os.path.join(tmpdir, "cot.json")

    env = dict(os.environ)
    # MODEL src 만 보이게 (템플릿 src 제외)
    env["PYTHONPATH"] = f"{MODEL_TRAIN_DIR}:{COT_PIPELINE_DIR}"
    env["SLOTON_DIR"] = SLOTON_DIR
    env["COT_CKPT"] = COT_CKPT
    env["CODEBOOK_DIR"] = CODEBOOK_DIR
    env["COT_ART"] = COT_ART
    env["TRAIN_JSON"] = TRAIN_JSON
    env.setdefault("HF_HOME", str(MODEL_PATH / "hf_cache"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TQDM_DISABLE", "1")

    # 진입 스크립트(하위호환): 기본=infer.py(c16 등 동작 불변).
    script = os.path.join(COT_PIPELINE_DIR, os.environ.get("COT_INFER_SCRIPT", "infer.py"))
    # ⏱ 슬라이드 deadline: 느린/스트립형 WSI 가 seg/읽기에서 무한정 늘어져 GC 컨테이너
    # 타임아웃(=제출 실패)을 내는 것을 방지. deadline 초과 → 자식 프로세스그룹 강제 kill →
    # RuntimeError → predict_chain_of_thought 가 잡아 _FALLBACK_COT 반환(=실패 대신 낮은점수).
    # (signal.alarm 은 C 단일 거대 read 를 못 끊어 무용 → 프로세스 kill 만이 확실.)
    deadline = float(os.environ.get("COT_SLIDE_DEADLINE", "300"))
    print(f"[interf1] running PIPELINE inference: {wsi_path} (deadline {deadline:.0f}s)", flush=True)
    from collections import deque
    import threading
    tail_buf: deque[str] = deque(maxlen=60)
    proc = subprocess.Popen(
        [sys.executable, script, str(wsi_path), out_json, tmpdir],
        cwd=MODEL_TRAIN_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True,   # 프로세스그룹 분리 → 타임아웃시 자식까지 한 번에 kill
    )

    def _drain():                              # 별 스레드로 출력 흘림(블로킹 read 가 deadline 을 막지 않게)
        try:
            for line in proc.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                tail_buf.append(line)
        except Exception:
            pass
    _reader = threading.Thread(target=_drain, daemon=True)
    _reader.start()

    try:
        proc.wait(timeout=deadline)
    except subprocess.TimeoutExpired:
        import signal as _sg
        try:
            os.killpg(os.getpgid(proc.pid), _sg.SIGKILL)   # 자식 프로세스그룹 통째로 강제종료
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        tail = "".join(tail_buf) or "(no output)"
        raise RuntimeError(
            f"[interf1] PIPELINE subprocess TIMEOUT (>{deadline:.0f}s) — killed.\n"
            f"----- subprocess output (tail) -----\n{tail}"
        )
    _reader.join(timeout=5)
    if proc.returncode != 0:
        tail = "".join(tail_buf) or "(no output)"
        raise RuntimeError(
            f"[interf1] PIPELINE subprocess failed (exit {proc.returncode}).\n"
            f"----- subprocess output (tail) -----\n{tail}"
        )

    with open(out_json) as f:
        steps = json.load(f)
    print(f"[interf1] CoT {len(steps)} steps")
    return steps
