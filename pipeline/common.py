"""
PIPELINE 공통 헬퍼 — slot-direct CoT 파이프라인.

핵심 아이디어: MODEL(sloton) 한 forward로
  embedder(ABMIL) -> z[1024]
  site_classifier  -> organ (CoT Q01)
  extraction_classifier -> procedure (CoT Q02)
  slot_classifiers[organ][qid] -> 중간 질문 답 (CoT 본문)
  decoder          -> final pathology report (CoT 마지막)
를 모두 산출. Qwen-7B(Model W) 완전 제거.

질문 시퀀스(Edge_F1/BPV, Metric A의 0.35)는 z 위에 새로 학습한
per-organ sequence head가 담당. (train_seq_head.py)
"""
import csv
import json
import os
import re
from collections import OrderedDict

import torch

# ---- 경로 (env override 가능; 기본값=dev, 컨테이너=/opt/ml/model) ----
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

# train_CoT.json 의 organ 필드 -> slot/codebook organ_key
ORGAN_TO_SLOTKEY = {
    "breast": "breast", "colon": "colon", "lung": "lung",
    "prostate": "prostate", "stomach": "stomach",
    "bladder": "urinary_bladder", "cervix": "uterine_cervix",
}
# idx_to_site.json 의 site 이름 -> train_CoT organ 필드 (추론 라우팅)
SITE_TO_ORGAN = {
    "Breast": "breast", "Nipple": "breast",
    "Colon": "colon", "Rectum": "colon", "Anus": "colon",
    "Lung": "lung", "Prostate": "prostate", "Stomach": "stomach",
    "Urinary bladder": "bladder", "Uterine cervix": "cervix",
}
ORGANS = ["breast", "colon", "lung", "prostate", "stomach", "bladder", "cervix"]


def slot_num_classes_from_ckpt(ckpt_path=CKPT):
    """체크포인트 shape 에서 slot_num_classes = {organ_key: {qid: n_classes}} 복원.
    이렇게 하면 load_slot_data(CSV) 없이도 slots-ON 모델을 정확히 만들 수 있다."""
    sd = torch.load(ckpt_path, map_location="cpu")
    out = {}
    for k, v in sd.items():
        # slot_classifiers.<organ>.<qid>.weight  -> [n_classes, 1024]
        if k.startswith("slot_classifiers.") and k.endswith(".weight"):
            _, organ, qid, _ = k.split(".")
            out.setdefault(organ, {})[qid] = v.shape[0]
    # qid 순서 정렬(Q03, Q04, ...)
    return {org: OrderedDict(sorted(d.items())) for org, d in out.items()}


def load_codebook(organ_key):
    """{qid: {code:int -> answer_text}} 와 {qid: question_text} 반환.
    organ_key 는 slot key (urinary_bladder/uterine_cervix)."""
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
    """beaver_organ -> (code2ans, qid2qtext, qtext2qid)."""
    out = {}
    for organ in ORGANS:
        slotkey = ORGAN_TO_SLOTKEY[organ]
        code2ans, qid2qtext = load_codebook(slotkey)
        qtext2qid = {}
        for qid, qt in qid2qtext.items():
            qtext2qid.setdefault(qt.strip(), qid)  # 첫 등장 qid 우선
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
    """CoT chain -> 질문 문자열 리스트 (final report 질문은 제외하지 않음)."""
    return [s.get("question", "").strip() for s in chain if s.get("question", "").strip()]
