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


MODEL_TRAIN_DIR = os.environ.get("MODEL_TRAIN_DIR", "/opt/app/model_lib")
COT_PIPELINE_DIR = os.environ.get("COT_PIPELINE_DIR", "/opt/app/pipeline")
SLOTON_DIR = os.environ.get("SLOTON_DIR", str(MODEL_PATH / "sloton"))
COT_CKPT = os.environ.get(
    "COT_CKPT", str(MODEL_PATH / "sloton" / "checkpoints" / "network_epoch_150.pth"))
CODEBOOK_DIR = os.environ.get("CODEBOOK_DIR", str(MODEL_PATH / "codebooks"))
COT_ART = os.environ.get("COT_ART", str(MODEL_PATH / "cot_artifacts"))
TRAIN_JSON = os.environ.get("TRAIN_JSON", str(MODEL_PATH / "sloton" / "train_from_CoT.json"))

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

    script = os.path.join(COT_PIPELINE_DIR, os.environ.get("COT_INFER_SCRIPT", "infer.py"))
    deadline = float(os.environ.get("COT_SLIDE_DEADLINE", "300"))
    print(f"[interf1] running PIPELINE inference: {wsi_path} (deadline {deadline:.0f}s)", flush=True)
    from collections import deque
    import threading
    tail_buf: deque[str] = deque(maxlen=60)
    proc = subprocess.Popen(
        [sys.executable, script, str(wsi_path), out_json, tmpdir],
        cwd=MODEL_TRAIN_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True,
    )

    def _drain():
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
            os.killpg(os.getpgid(proc.pid), _sg.SIGKILL)
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
