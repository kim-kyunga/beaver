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

    _t = time.time()
    _h5, feats, _coords = extract_features(wsi_path, job_dir, gpu=gpu)
    print(f"[timing] STAGE1 trident(seg+coords+feat) = {time.time()-_t:.1f}s | features {feats.shape}", flush=True)

    _t = time.time()
    handle = load_model(device=device)
    print(f"[timing] STAGE2a MODEL load = {time.time()-_t:.1f}s", flush=True)
    _t = time.time()
    rec = run_forward(handle, feats)
    print(f"[timing] STAGE2b MODEL forward = {time.time()-_t:.1f}s", flush=True)

    _t = time.time()
    cot = assemble(rec)
    with open(out_json, "w") as f:
        json.dump(cot, f, ensure_ascii=False, indent=2)
    print(f"[timing] STAGE3 assemble(retrieval) = {time.time()-_t:.1f}s | {len(cot)} steps", flush=True)
    print(f"[timing] === GRAND TOTAL (subprocess) = {time.time()-_T0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
