
from core import (
    INPUT_PATH,
    OUTPUT_PATH,
    get_interface_key,
    write_json_file,
    show_torch_cuda_info,
)

from src.interf0.model import predict_visual_context_response
from src.interf1.model import predict_chain_of_thought


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


def interf0_handler():
    question_path   = INPUT_PATH / "visual-context-question.json"
    roi_image_path  = INPUT_PATH / "histopathology-region-of-interest-thumbnail.jpeg"
    output_path     = OUTPUT_PATH / "visual-context-response.json"

    print(f"[interf0] Question path : {question_path}")
    print(f"[interf0] ROI path      : {roi_image_path}")

    try:
        answer = predict_visual_context_response(
            question_path=question_path,
            roi_image_path=roi_image_path,
        )
    except Exception:
        import traceback
        from src.interf0.model import ANSWER_BACKGROUND
        print("[interf0] handler-level failure; writing safe background answer.\n"
              + traceback.format_exc(), flush=True)
        answer = ANSWER_BACKGROUND

    write_json_file(location=output_path, content=answer)
    print(f"[interf0] Answer written: {answer}")
    return 0


def interf1_handler():
    output_path = OUTPUT_PATH / "chain-of-thought.json"
    try:
        wsi_dir = INPUT_PATH / "images" / "whole-slide-image"
        wsi_files = sorted(p for p in wsi_dir.iterdir() if p.is_file())
        if not wsi_files:
            raise FileNotFoundError(f"No image files found in {wsi_dir}")
        wsi_path = wsi_files[0]
        show_torch_cuda_info()
        chain_of_thought = predict_chain_of_thought(wsi_path=wsi_path)
    except Exception:
        import traceback
        from src.interf1.model import _FALLBACK_COT
        print("[interf1] handler-level failure; writing fallback CoT.\n"
              + traceback.format_exc(), flush=True)
        chain_of_thought = _FALLBACK_COT

    write_json_file(location=output_path, content=chain_of_thought)
    print(f"[interf1] Chain-of-thought written ({len(chain_of_thought)} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
