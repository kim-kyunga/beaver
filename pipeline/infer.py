"""
단일 WSI(.tiff) -> chain-of-thought (JSON list).  interf1 에서 subprocess 로 호출.

격리 이유: 제출 템플릿의 `src`(interf0/interf1) 와 MODEL 의 `src`(network/utils...) 가
패키지명이 같아 같은 프로세스에서 충돌 → 별 프로세스로 실행하고 PYTHONPATH/cwd 로 MODEL src 만 보이게.

흐름: preprocess(seg+features) -> lean(MODEL forward) -> assemble_cot(CoT).
env: SLOTON_DIR, COT_CKPT, CODEBOOK_DIR, COT_ART, HF_HOME, HF_HUB_OFFLINE, TRIDENT_PATH.
"""
import sys
import os
import json
import tempfile


def main():
    wsi_path = sys.argv[1]
    out_json = sys.argv[2]
    job_dir = sys.argv[3] if len(sys.argv) > 3 else tempfile.mkdtemp(prefix="pipeline_")

    import time
    import torch
    from preprocess import extract_features
    from lean import load_model, run_forward
    from assemble_cot import assemble

    gpu = int(os.environ.get("COT_GPU", "0"))
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    _T0 = time.time()

    # 1) WSI -> H-optimus-1 patch features (trident: seg -> coords -> encoder)
    _t = time.time()
    _h5, feats, _coords = extract_features(wsi_path, job_dir, gpu=gpu)
    print(f"[timing] STAGE1 trident(seg+coords+feat) = {time.time()-_t:.1f}s | features {feats.shape}", flush=True)

    # 2) MODEL(sloton) 단일 forward (lean)
    _t = time.time()
    handle = load_model(device=device)
    print(f"[timing] STAGE2a MODEL load = {time.time()-_t:.1f}s", flush=True)
    _t = time.time()
    rec = run_forward(handle, feats)
    print(f"[timing] STAGE2b MODEL forward = {time.time()-_t:.1f}s", flush=True)

    # 3) record -> CoT  (retrieval-based chain assembly)
    _t = time.time()
    cot = assemble(rec)
    with open(out_json, "w") as f:
        json.dump(cot, f, ensure_ascii=False, indent=2)
    print(f"[timing] STAGE3 assemble(retrieval) = {time.time()-_t:.1f}s | {len(cot)} steps", flush=True)
    print(f"[timing] === GRAND TOTAL (subprocess) = {time.time()-_T0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
