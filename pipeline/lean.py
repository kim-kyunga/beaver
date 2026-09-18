import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.utils.general_utils import process_config
from src.utils.text_utils import read_idx_mappings, decode_text_pred
from src.network.model import get_model
from src.evaluation.evaluation_utils.inference import InferenceRunner

SLOTON_DIR = os.environ.get("SLOTON_DIR", "/opt/ml/model/sloton/train")
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
    cfg = OmegaConf.load(os.path.join(sloton_dir, "config.yaml"))
    cfg.model.model_checkpoint = ckpt
    cfg.model.train_device = device
    cfg.model.eval_device = device
    cfg.data.experiment_path = os.environ.get("COT_EXP_DIR", "/tmp/pipeline_exp")
    train_json = os.environ.get("TRAIN_JSON")
    if train_json:
        cfg.data.train_json_path = train_json
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
    device = handle["device"]
    img = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    lens = torch.tensor([feats.shape[0]], device=device)
    out = handle["infer"](img, lens)

    z = out.embedding[0].float().cpu().numpy()
    site_i = out.site_logits.argmax(-1).item()
    em_i = out.em_logits.argmax(-1).item()
    slot_logits = (out.extra or {}).get("slot_logits", {})
    slot_codes = {org: {qid: lg.argmax(-1).item() for qid, lg in qd.items()}
                  for org, qd in slot_logits.items()}
    slot_conf = {org: {qid: float(torch.softmax(lg, dim=-1).max()) for qid, lg in qd.items()}
                 for org, qd in slot_logits.items()}
    text = decode_text_pred(out.text_logits[0].cpu(), handle["idx_to_word"], handle["pad_idx"])
    return {
        "z": z, "site": site_i, "em": em_i,
        "site_name": handle["idx_to_site"][site_i],
        "em_name": handle["idx_to_extraction"][em_i],
        "report_body": text, "slot_codes": slot_codes, "slot_conf": slot_conf,
    }
