"""
REG2026 Challenge — Algorithm Entry Point
=========================================

This file is the container's entrypoint (see Dockerfile).
It detects which interface is active and calls the right handler.

  Interface 0 — Visual Grounding (Metric B)
    Input  : paths to ROI thumbnail (.jpeg) + question (JSON)
    Output : visual-context-response.json  — a plain JSON string

  Interface 1 — Workflow Reasoning (Metric A)
    Input  : WSI at /input/images/whole-slide-image/<uid>.tiff  (uid = opaque hash)
    Output : chain-of-thought.json  — a JSON array of {question, answer, next_question}

Where to add YOUR code
-----------------------
  - src/interf0/model.py  →  predict_visual_context_response()
  - src/interf1/model.py  →  predict_chain_of_thought()

  The functions in core.py (I/O helpers, path constants, interface detection)
  do not need to be changed.

See README.md for a full walkthrough.
"""

from core import (
    INPUT_PATH,
    OUTPUT_PATH,
    get_interface_key,
    write_json_file,
    show_torch_cuda_info,
)

# ---------------------------------------------------------------------------
# Import your inference functions from src/
# ---------------------------------------------------------------------------
from src.interf0.model import predict_visual_context_response
from src.interf1.model import predict_chain_of_thought


# ---------------------------------------------------------------------------
# Entry point — dispatches to the correct handler automatically
# ---------------------------------------------------------------------------

def run():
    interface_key = get_interface_key()

    handler = {
        (
            "histopathology-region-of-interest-thumbnail",
            "visual-context-question",
        ): interf0_handler,
        ("whole-slide-image",): interf1_handler,
    }[interface_key]

    return handler()


# ---------------------------------------------------------------------------
# Interface 0 — Visual Grounding
# ---------------------------------------------------------------------------

def interf0_handler():
    # --- Fixed input paths (do not change) -----------------------------------
    question_path   = INPUT_PATH / "visual-context-question.json"
    roi_image_path  = INPUT_PATH / "histopathology-region-of-interest-thumbnail.jpeg"
    output_path     = OUTPUT_PATH / "visual-context-response.json"

    print(f"[interf0] Question path : {question_path}")
    print(f"[interf0] ROI path      : {roi_image_path}")

    # --- Run inference (try-except: 실패해도 안전한 기본답을 반드시 출력) ------
    try:
        # predict_visual_context_response lives in src/interf0/model.py
        answer = predict_visual_context_response(
            question_path=question_path,
            roi_image_path=roi_image_path,
        )
    except Exception:
        import traceback
        # 실패 시 안전한 기본 = 배경(거짓 진단주장 안 함) → B1 안전, hallucination 방지
        from src.interf0.model import ANSWER_BACKGROUND
        print("[interf0] handler-level failure; writing safe background answer.\n"
              + traceback.format_exc(), flush=True)
        answer = ANSWER_BACKGROUND

    # --- Write output (항상 실행) --------------------------------------------
    write_json_file(location=output_path, content=answer)
    print(f"[interf0] Answer written: {answer}")
    return 0


# ---------------------------------------------------------------------------
# Interface 1 — Workflow Reasoning
# ---------------------------------------------------------------------------

def interf1_handler():
    # 주최 권고: 단일 케이스의 크래시/타임아웃이 제출 전체를 실패시키지 않도록 핸들러 전체를
    # try-except 로 감싸 *항상* 유효한 chain-of-thought.json 을 쓴다(실패 시 fallback CoT).
    output_path = OUTPUT_PATH / "chain-of-thought.json"
    try:
        # --- WSI path (platform: /input/images/whole-slide-image/<uid>.<ext>) ----
        # 실제 데이터는 .tiff 지만 Try-out/플랫폼에 따라 .mha/.png 로 올 수 있어 확장자 무관하게 찾는다.
        wsi_dir = INPUT_PATH / "images" / "whole-slide-image"
        wsi_files = sorted(p for p in wsi_dir.iterdir() if p.is_file())
        if not wsi_files:
            raise FileNotFoundError(f"No image files found in {wsi_dir}")
        wsi_path = wsi_files[0]
        show_torch_cuda_info()
        # predict_chain_of_thought 는 내부에서 subprocess deadline + 예외 → fallback 처리
        chain_of_thought = predict_chain_of_thought(wsi_path=wsi_path)
    except Exception:
        import traceback
        from src.interf1.model import _FALLBACK_COT
        print("[interf1] handler-level failure; writing fallback CoT.\n"
              + traceback.format_exc(), flush=True)
        chain_of_thought = _FALLBACK_COT

    # --- Write output (항상 실행) --------------------------------------------
    write_json_file(location=output_path, content=chain_of_thought)
    print(f"[interf1] Chain-of-thought written ({len(chain_of_thought)} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
