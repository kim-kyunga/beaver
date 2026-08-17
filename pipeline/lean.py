"""
PIPELINE lean 단일슬라이드 forward — hydra/RegDataset/vocab 빌드 없이.

기존 pipeline_extract.py(full 모드)와 동일한 record를 내지만:
  - hydra 대신 OmegaConf 로 sloton/config.yaml 직접 로드
  - RegDataset/DataLoader 대신 features(N,768) 를 직접 (img,lens) 로 만들어 forward
  - get_vocabulary(train_json) 제거 (모델은 idx_to_word=sloton/idx_to_word.json 만 사용)

제출 interf1 에서 load_model()(1회) + run_forward(feats) 로 호출.
경로는 env override 가능: SLOTON_DIR, COT_CKPT.
"""
import os
from collections import OrderedDict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.utils.general_utils import process_config
from src.utils.text_utils import read_idx_mappings, decode_text_pred
from src.network.model import get_model
from src.evaluation.evaluation_utils.inference import InferenceRunner

SLOTON_DIR = os.environ.get(
    "SLOTON_DIR", "/opt/ml/model/sloton/train")
COT_CKPT = os.environ.get(
    "COT_CKPT", os.path.join(SLOTON_DIR, "checkpoints", "network_epoch_150.pth"))
FEATURE_DIM = 1536


def slot_num_classes_from_ckpt(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    out = {}
    for k, v in sd.items():
        if k.startswith("slot_classifiers.") and k.endswith(".weight"):
            _, organ, qid, _ = k.split(".")
            out.setdefault(organ, {})[qid] = v.shape[0]
    return {org: dict(sorted(d.items())) for org, d in out.items()}


def load_model(sloton_dir=SLOTON_DIR, ckpt=COT_CKPT, device="cuda"):
    """모델 1회 로드. 반환: 추론에 필요한 핸들 dict."""
    cfg = OmegaConf.load(os.path.join(sloton_dir, "config.yaml"))
    cfg.model.model_checkpoint = ckpt
    cfg.model.train_device = device
    cfg.model.eval_device = device
    # process_config 가 experiment_path/experiment_name 으로 결과폴더를 mkdir 한다.
    # config default path may be read-only for the non-root container user; redirect to writable /tmp.
    cfg.data.experiment_path = os.environ.get("COT_EXP_DIR", "/tmp/pipeline_exp")
    # get_model 이 train_json 으로 decoder corpus 를 만든다 → 컨테이너 경로로 override
    train_json = os.environ.get("TRAIN_JSON")
    if train_json:
        cfg.data.train_json_path = train_json
    # 데이터 경로 의존 제거: input feature shape 은 직접 줄 것이므로 검증만 회피
    cfg = process_config(cfg, is_eval=True)

    idx_to_word, idx_to_site, idx_to_extraction = read_idx_mappings(Path(sloton_dir))
    slot_ncls = slot_num_classes_from_ckpt(ckpt)
    net = get_model(cfg, idx_to_word, idx_to_site, idx_to_extraction,
                    (1, FEATURE_DIM), slot_num_classes=slot_ncls)
    net.eval()
    infer = InferenceRunner(
        net, device=device,
        precision=torch.float16 if cfg.training.precision == "fp16" else torch.float32)
    return {
        "infer": infer, "device": device, "pad_idx": cfg.text.pad_idx,
        "idx_to_word": idx_to_word, "idx_to_site": idx_to_site,
        "idx_to_extraction": idx_to_extraction,
    }


@torch.inference_mode()
def run_forward(handle, feats):
    """feats: np.ndarray [N,768] -> record dict (assemble_cot 입력 형식과 동일)."""
    device = handle["device"]
    img = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)   # [1,N,768]
    lens = torch.tensor([feats.shape[0]], device=device)
    out = handle["infer"](img, lens)

    z = out.embedding[0].float().cpu().numpy()
    site_i = out.site_logits.argmax(-1).item()
    em_i = out.em_logits.argmax(-1).item()
    slot_logits = (out.extra or {}).get("slot_logits", {})
    slot_codes = {org: {qid: lg.argmax(-1).item() for qid, lg in qd.items()}
                  for org, qd in slot_logits.items()}
    text = decode_text_pred(out.text_logits[0].cpu(), handle["idx_to_word"], handle["pad_idx"])
    return {
        "z": z, "site": site_i, "em": em_i,
        "site_name": handle["idx_to_site"][site_i],
        "em_name": handle["idx_to_extraction"][em_i],
        "report_body": text, "slot_codes": slot_codes,
    }
