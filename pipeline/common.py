import csv
import json
import os
import re
from collections import OrderedDict

import torch

SLOTON_DIR = os.environ.get(
    "SLOTON_DIR", "/opt/ml/model/sloton/train")
CKPT = os.environ.get(
    "COT_CKPT", os.path.join(SLOTON_DIR, "checkpoints", "network_epoch_150.pth"))
CODEBOOK_DIR = os.environ.get(
    "CODEBOOK_DIR", "/opt/ml/model/codebooks")
TRAIN_COT = os.environ.get("TRAIN_COT", "/opt/ml/model/sloton/train_CoT.json")
ART = os.environ.get("COT_ART", "/opt/ml/model/cot_artifacts")

REPORT_Q_PATTERN = re.compile(r"final\s+(pathology\s+)?report", re.I)
FINAL_Q = "What is the final pathology report?"

ORGAN_TO_SLOTKEY = {
    "breast": "breast", "colon": "colon", "lung": "lung",
    "prostate": "prostate", "stomach": "stomach",
    "bladder": "urinary_bladder", "cervix": "uterine_cervix",
}
SITE_TO_ORGAN = {
    "Breast": "breast", "Nipple": "breast",
    "Colon": "colon", "Rectum": "colon", "Anus": "colon",
    "Lung": "lung", "Prostate": "prostate", "Stomach": "stomach",
    "Urinary bladder": "bladder", "Uterine cervix": "cervix",
}
ORGANS = ["breast", "colon", "lung", "prostate", "stomach", "bladder", "cervix"]


def slot_num_classes_from_ckpt(ckpt_path=CKPT):
    sd = torch.load(ckpt_path, map_location="cpu")
    out = {}
    for k, v in sd.items():
        if k.startswith("slot_classifiers.") and k.endswith(".weight"):
            _, organ, qid, _ = k.split(".")
            out.setdefault(organ, {})[qid] = v.shape[0]
    return {org: OrderedDict(sorted(d.items())) for org, d in out.items()}


def load_codebook(organ_key):
    path = os.path.join(CODEBOOK_DIR, f"{organ_key}_label_codebook.csv")
    code2ans = {}
    qid2qtext = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row["question_id"]
            code2ans.setdefault(qid, {})[int(row["code"])] = row["answer_text"]
            qid2qtext[qid] = row["question_text"]
    return code2ans, qid2qtext


def load_all_codebooks():
    out = {}
    for organ in ORGANS:
        slotkey = ORGAN_TO_SLOTKEY[organ]
        code2ans, qid2qtext = load_codebook(slotkey)
        qtext2qid = {}
        for qid, qt in qid2qtext.items():
            qtext2qid.setdefault(qt.strip(), qid)
        out[organ] = {"code2ans": code2ans, "qid2qtext": qid2qtext,
                      "qtext2qid": qtext2qid, "slotkey": slotkey}
    return out


def read_idx_maps(sloton_dir=SLOTON_DIR):
    idx_to_site = {int(k): v for k, v in json.load(
        open(os.path.join(sloton_dir, "idx_to_site.json"))).items()}
    idx_to_extr = {int(k): v for k, v in json.load(
        open(os.path.join(sloton_dir, "idx_to_extraction.json"))).items()}
    idx_to_word = {int(k): v for k, v in json.load(
        open(os.path.join(sloton_dir, "idx_to_word.json"))).items()}
    return idx_to_site, idx_to_extr, idx_to_word


def chain_questions(chain):
    return [s.get("question", "").strip() for s in chain if s.get("question", "").strip()]
