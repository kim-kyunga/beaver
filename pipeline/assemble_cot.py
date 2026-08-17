"""
PIPELINE 추출 record -> chain-of-thought.json  (Qwen 없음).

record(full 모드): {z, site_name, em_name, report_body, slot_codes{slotkey:{qid:code}}}
  organ      = SITE_TO_ORGAN[site_name]
  sequence   = seq_vocab[organ][ argmax(seq_head[organ] · z) ]
  각 질문 답:
    Q01(organ)    -> site_name
    Q02(procedure)-> em_name (codebook Q02 로 정규 표기 보정)
    final report  -> "{site}, {em};{report_body}"
    그 외(중간)   -> slot code -> codebook 정답 텍스트
"""
import os
import re
import json
import numpy as np
import torch

from common import (ART, REPORT_Q_PATTERN, FINAL_Q, SITE_TO_ORGAN,
                         ORGAN_TO_SLOTKEY, load_all_codebooks)
from retrieval import build_query_template, retrieve_report, retrieve_chain

_SEQ_VOCAB = json.load(open(f"{ART}/seq_vocab.json"))
_HEADS = torch.load(f"{ART}/seq_heads.pt", map_location="cpu")
_CB = load_all_codebooks()

import os
# [c10] 리포트 라우팅: 특정 organ은 '검색'(이웃 리포트 복사) 대신 모델의 리포트 슬롯
# (메뉴판 직접 분류, 예: breast Q24)을 쓴다. held-out 공식채점(S-PubMedBert) 기준
# breast/bladder/stomach 에서 직접분류>검색 → 리포트점수 0.9202→0.9252(+0.0050),
# prostate 등은 검색이 나아 제외(라우팅 안 함). 플래그 미설정 시 c9 동작(전부 검색) 유지.
_REPORT_ROUTING = os.environ.get("COT_REPORT_ROUTING", "0") == "1"
_ROUTE_ORGANS = {o.strip() for o in os.environ.get(
    "COT_ROUTE_ORGANS", "breast,bladder,stomach").split(",") if o.strip()}
_REP_QID_CACHE = {}


def _report_slot_qid(organ, cb):
    """codebook 에서 'final pathology report' 질문의 qid(예: breast Q24) 1회 탐색·캐시."""
    if organ not in _REP_QID_CACHE:
        rq = None
        for qid, qt in cb["qid2qtext"].items():
            if REPORT_Q_PATTERN.search(qt):
                rq = qid
                break
        _REP_QID_CACHE[organ] = rq
    return _REP_QID_CACHE[organ]


def _procedure_answer(em_name, organ):
    """extraction_classifier 이름을 codebook Q02 정규 표기로 보정(대소문자 등)."""
    q2 = _CB[organ]["code2ans"].get("Q02", {})
    for ans in q2.values():
        if ans.strip().lower() == em_name.strip().lower():
            return ans
    return em_name


def predict_sequence(z, organ):
    h = _HEADS[organ]
    logits = torch.from_numpy(np.asarray(z, dtype=np.float32)) @ h["W"].T + h["b"]
    seq_id = int(logits.argmax(-1).item())
    return _SEQ_VOCAB[organ][seq_id]


def _answer_for(q, site, proc, report, cb, slot_codes, fallback=""):
    """질문 문자열 -> 우리 예측 답. (organ/procedure/report 특수처리, 그외 slot codebook)
    슬롯 미커버 시 fallback(검색이웃 답) 사용."""
    if REPORT_Q_PATTERN.search(q):
        return report
    ql = q.strip().lower()
    if ql == "what is the organ?":
        return site
    if ql == "what is the procedure?":
        return proc
    qid = cb["qtext2qid"].get(q.strip())
    if qid is not None and qid in slot_codes:
        return cb["code2ans"].get(qid, {}).get(slot_codes[qid], "") or fallback
    return fallback


# ===== 에이전틱 자기검증 (COT_SELF_VERIFY=1) =====
# 진단 슬롯 간 모순을 감지→#1진단(앵커)으로 추론·해소→일관된 CoT + 추론 trace.
# (앵커 해소는 점수중립 검증됨 = A 안깎임. 에이전트 '과정'만 추가.)
_SELF_VERIFY = os.environ.get("COT_SELF_VERIFY", "0") == "1"
# [containerA4] Counterfactual Reasoning(Metric C) 경로. 기본 off=완전 dormant(정상추론 무변경).
_COUNTERFACTUAL = os.environ.get("COT_COUNTERFACTUAL", "0") == "1"
_MAL_KW = ("carcinoma", "sarcoma", "melanoma", "lymphoma", "blastoma", "leukemia", "malignant", "adenocarcinoma")
def _sv_low(t): return re.sub(r"\s+", " ", str(t).strip().lower())
def _sv_neg(t):
    t = _sv_low(t)
    return t.startswith("no") or "there is no" in t or "no evidence" in t or "negative for" in t or "no residual" in t
def _sv_mal(t): return (not _sv_neg(t)) and any(k in _sv_low(t) for k in _MAL_KW)
def _sv_inv(t): t2 = _sv_low(t); return (not _sv_neg(t)) and ("invasive" in t2 or "invasion" in t2)
def _sv_situ(t): t2 = _sv_low(t); return ("in situ" in t2 or "intraepithelial" in t2) and "invasive" not in t2
def _sv_slottext(cb, slot_codes, qtext):
    qid = cb["qtext2qid"].get(qtext)
    if qid and qid in slot_codes:
        return cb["code2ans"].get(qid, {}).get(slot_codes[qid], "")
    return None
def _sv_opt(cb, qtext, positive):
    """codebook 에서 qtext 질문의 긍정/부정 답 옵션 텍스트."""
    qid = cb["qtext2qid"].get(qtext)
    if not qid:
        return None
    for txt in cb["code2ans"].get(qid, {}).values():
        if positive and not _sv_neg(txt):
            return txt
        if (not positive) and _sv_neg(txt):
            return txt
    return None
def self_verify(cb, slot_codes):
    """모순 감지→#1진단 앵커로 해소. 반환 (overrides{질문소문자: 교정답}, trace[추론문]). """
    def g(sub):
        for qt in cb["qtext2qid"]:
            if sub in _sv_low(qt):
                v = _sv_slottext(cb, slot_codes, qt)
                if v is not None:
                    return v
        return None
    dx1 = g("#1 diagnosis")
    overrides, trace = {}, []
    if not dx1:
        return overrides, trace
    trace.append(f"자기검증: '#1 진단={dx1}' 을 앵커로 슬롯 일관성 점검.")
    checks = [
        ("is there any invasion present?", "is there any invasion present", _sv_inv(dx1), True, "진단이 침습성인데 '침습 없음'"),
        ("is there any abnormality present?", "is there any abnormality present", _sv_mal(dx1), True, "악성 진단인데 '이상 없음'"),
        ("is there any neoplasm present?", "is there any neoplasm present", _sv_mal(dx1), True, "악성 진단인데 '종양 없음'"),
    ]
    for qkey, sub, anchor_pos, want_pos, desc in checks:
        cur = g(sub)
        if cur and anchor_pos and _sv_neg(cur):       # 앵커는 양성인데 슬롯은 음성 = 모순
            corr = _sv_opt(cb, qkey, want_pos)
            if corr:
                overrides[_sv_low(qkey)] = corr
                trace.append(f"  ⚠️모순: {desc}('{cur}'). → 앵커(진단)에 근거해 '{corr}'로 교정.")
    if not overrides:
        trace.append("  ✓ 슬롯 예측 일관됨 (모순 없음).")
    return overrides, trace


# ===== A7: per-branch KB resolvers + grade override (LLM-free, deterministic) =====
# Independently-predicted slots can disagree across the branches of a multi-diagnosis chain
# (invasion / papillary / behavior) or with the model's own grade slot. Each resolver fills
# the answer from its branch diagnosis via a train-learned knowledge base, or from the
# confident grade slot (prostate grade). Knowledge bases live under COT_ART (cot_artifacts/).
_INV_BRANCH = os.environ.get("COT_INV_BRANCH", "0") == "1"
_COT_CONSISTENCY = os.environ.get("COT_CONSISTENCY", "0") == "1"
# #1-diagnosis<-report unification: OFF in A7 (degrades #1-diagnosis accuracy on held-out).
_COT_DXFIX = os.environ.get("COT_DXFIX", "1") == "1"
_GRADE_OVERRIDE = os.environ.get("COT_GRADE_OVERRIDE", "0") == "1"
_BEH_PURITY_THR = float(os.environ.get("COT_BEH_PURITY_THR", "0.9"))
_INV_KB = None
_PAP_KB = None
_BEH_KB = None
_INV_BENIGN = ("hyperplasia", "adenosis", "atypia", "fibroadenoma", "papilloma", "adenoma",
    "fibroepithelial", "columnar cell", "phyllodes", "fibromatosis", "cell change", "cyst")


def _load_inv_kb():
    global _INV_KB
    if _INV_KB is None:
        try: _INV_KB = json.load(open(f"{ART}/inv_kb.json")).get("mapping", {})
        except Exception: _INV_KB = {}
    return _INV_KB


def _load_pap_kb():
    global _PAP_KB
    if _PAP_KB is None:
        try: _PAP_KB = json.load(open(f"{ART}/pap_kb.json")).get("mapping", {})
        except Exception: _PAP_KB = {}
    return _PAP_KB


def _load_beh_kb():
    global _BEH_KB
    if _BEH_KB is None:
        try: _BEH_KB = json.load(open(f"{ART}/beh_kb.json")).get("mapping", {})
        except Exception: _BEH_KB = {}
    return _BEH_KB


def _svr_primary(rep):
    r = rep.replace("\\n", "\n")
    m = re.search(r"\b1\.\s*([^\n]+)", r)
    if m: return m.group(1).strip()
    if ";" in r: return r.split(";", 1)[1].strip().split("\n")[0].strip()
    return ""


def _branch_dx(skel, p):
    """First 'histologic type' answer after step p in the skeleton = that branch's diagnosis."""
    for j in range(p + 1, min(p + 7, len(skel))):
        if skel[j].get("question", "").strip().lower().startswith("what is the histologic type"):
            return skel[j].get("answer", "").strip()
    return None


def _kw_invasion(dx):
    d = _sv_low(dx)
    if not d: return None
    if "in situ" in d or "non-invasive" in d or "noninvasive" in d or "intraepithelial" in d: return "No"
    if any(b in d for b in _INV_BENIGN): return "No"
    if any(m in d for m in ("carcinoma", "sarcoma", "lymphoma", "melanoma", "malignant")): return "Yes"
    return None


def _inv_option(cb, positive):
    for _qt, _qid in cb.get("qtext2qid", {}).items():
        if "is there any invasion present" in _qt.lower():
            for _txt in cb.get("code2ans", {}).get(_qid, {}).values():
                if positive and not _sv_neg(_txt): return _txt
                if (not positive) and _sv_neg(_txt): return _txt
    return "Yes, there is a invasion." if positive else "No, there is no invasion."


def _pap_option(cb, positive):
    for _qt, _qid in cb.get("qtext2qid", {}).items():
        if "is there any papillary lesion present" in _qt.lower():
            for _txt in cb.get("code2ans", {}).get(_qid, {}).values():
                if positive and not _sv_neg(_txt): return _txt
                if (not positive) and _sv_neg(_txt): return _txt
    return "Yes, there is a papillary lesion." if positive else "No, there is no papillary lesion."


def _beh_option(cb, want):
    for _qt, _qid in cb.get("qtext2qid", {}).items():
        if _sv_low(_qt) == "what is the behavior of neoplasm?":
            for _txt in cb.get("code2ans", {}).get(_qid, {}).values():
                if _sv_low(_txt) == _sv_low(want): return _txt
    return want


def _inv_branch_resolve(steps, skel, organ, cb):
    """Resolve each invasion step from its branch diagnosis via the KB (breast / bladder)."""
    if organ not in ("breast", "bladder"): return steps, []
    okb = _load_inv_kb().get("Breast" if organ == "breast" else "Urinary bladder", {})
    trace = []
    for i, s in enumerate(steps):
        if _sv_low(s.get("question", "")) != "is there any invasion present?": continue
        dx = _branch_dx(skel, i) if i < len(skel) else None
        if not dx: continue
        inv = okb.get(_sv_low(dx)) or _kw_invasion(dx)
        if inv is None: continue
        new = _inv_option(cb, inv == "Yes")
        if _sv_low(new) != _sv_low(s.get("answer", "")): trace.append(f"invasion branch '{dx}'->{inv}")
        s["answer"] = new
    return steps, trace


def _pap_branch_resolve(steps, skel, organ, cb):
    if organ != "bladder": return steps, []
    okb = _load_pap_kb().get("Urinary bladder", {}); trace = []
    for i, s in enumerate(steps):
        if _sv_low(s.get("question", "")) != "is there any papillary lesion present?": continue
        dx = _branch_dx(skel, i) if i < len(skel) else None
        if not dx: continue
        pap = okb.get(_sv_low(dx))
        if pap is None: continue
        new = _pap_option(cb, pap == "Yes")
        if _sv_low(new) != _sv_low(s.get("answer", "")): trace.append(f"papillary branch '{dx}'->{pap}")
        s["answer"] = new
    return steps, trace


def _beh_branch_resolve(steps, skel, site, cb):
    """Behavior from branch diagnosis, applied only when the KB mapping is pure (>= COT_BEH_PURITY_THR).
    Ambiguous mappings (e.g. cervical SIL, purity ~0.5 mixing LSIL/HSIL) are left to the slot."""
    okb = _load_beh_kb().get(site, {})
    if not okb: return steps, []
    trace = []
    for i, s in enumerate(steps):
        if _sv_low(s.get("question", "")) != "what is the behavior of neoplasm?": continue
        dx = _branch_dx(skel, i) if i < len(skel) else None
        if not dx: continue
        ent = okb.get(_sv_low(dx))
        if not ent: continue
        if isinstance(ent, dict): beh = ent.get("behavior"); pur = ent.get("purity", 1.0)
        else: beh = ent; pur = 1.0
        if not beh or pur < _BEH_PURITY_THR: continue
        opt = _beh_option(cb, beh)
        if _sv_low(opt) != _sv_low(s.get("answer", "")): trace.append(f"behavior branch '{dx}'->{beh}(p{pur})")
        s["answer"] = opt
    return steps, trace


def _cot_consistency_fix(steps, report):
    rp = _svr_primary(report)
    if not rp: return steps, []
    trace = []
    for s in steps:
        if re.match(r"what is the #1 diagnosis\??$", _sv_low(s.get("question", ""))):
            if _sv_low(s.get("answer", "")) != _sv_low(rp):
                trace.append(f"#1 diagnosis '{s['answer']}'->report '{rp}'"); s["answer"] = rp
    return steps, trace


def _gg_from_gs(gs):
    m = re.search(r"\((\d)\+(\d)\)", gs or "")
    if not m: return None
    a, b = int(m.group(1)), int(m.group(2)); t = a + b
    return {(3, 3): 1, (3, 4): 2, (4, 3): 3}.get((a, b),
        4 if t == 8 else (5 if t >= 9 else (1 if t == 6 else None)))


def _grade_override(report, cb, slot_codes, organ):
    """Replace the routed prostate report grade with the model's Gleason-score slot (+ derived group)."""
    if organ != "prostate" or not report or "gleason" not in report.lower(): return report, []
    gs = None
    for _qt in cb.get("qtext2qid", {}):
        if "gleason score" in _sv_low(_qt): gs = _sv_slottext(cb, slot_codes, _qt); break
    if not gs: return report, []
    gg = _gg_from_gs(gs)
    if not gg: return report, []
    new = re.sub(r"[Gg]leason's score \d+ \(\d\+\d\), grade group \d+",
                 "Gleason's score %s, grade group %d" % (gs, gg), report)
    return (new, ["grade->slot %s/GG%d" % (gs, gg)]) if new != report else (report, [])


def _sanitize_cot(steps):
    """출력 구조 보증(GC 'structure' 검증 통과용):
      - 빈 question("") 스텝 제거 — GT 체인 DB 일부(124/9654, 폐 NSCLC subtype)가 question="" 결손
        스텝을 가짐(질문 텍스트가 직전 next_question 에만 존재). 그대로 내보내면 구조 위반.
        (assemble 루프에서 직전 next_question 으로 복원하지만, 복원 불가분은 여기서 제거.)
      - 빈 answer 보강, 마지막 next_question="" 강제.
    물음표 누락 등 표기 차이는 GT 에도 1489건 존재 → 허용이므로 건드리지 않음(EdgeF1 보존)."""
    out = [s for s in steps if str(s.get("question", "")).strip()]
    if not out:
        return steps
    for s in out:
        if not str(s.get("answer", "")).strip():
            s["answer"] = "Not specified."
    out[-1]["next_question"] = ""
    return out


def assemble(record):
    site = record["site_name"]
    em = record["em_name"]
    organ = SITE_TO_ORGAN.get(site, "breast")
    if organ not in _HEADS:
        organ = "breast"
    slotkey = ORGAN_TO_SLOTKEY[organ]
    cb = _CB[organ]
    slot_codes = record.get("slot_codes", {}).get(slotkey, {})
    proc = _procedure_answer(em, organ)

    # [containerA4] Counterfactual Reasoning(Metric C): 속성수정 주입시에만 조건부검색으로
    #   결론·하류 체인 재유도(수정속성에 매칭되는 실제 GT 케이스=내부일관). 수정 없으면 즉시 통과
    #   → 정상 추론(일반 리더보드 채점)엔 영향 0 = A/B 무손상.
    if _COUNTERFACTUAL:
        try:
            import counterfactual as _CF
            _spec = _CF.parse_cf_spec()
            if _spec:
                _base = _CF.baseline_from_slots(cb, slot_codes, organ)
                _rep, _chain = _CF.apply(ART, record["z"], organ, _base, _spec)
                if _chain:
                    return _sanitize_cot([{"question": s.get("question", ""),
                                           "answer": s.get("answer", ""),
                                           "next_question": s.get("next_question", "")} for s in _chain])
        except Exception as _e:
            print(f"[counterfactual] skip ({_e})", flush=True)  # 실패시 정상경로로 안전 폴백

    # --- final report: 검색(+슬롯 재랭킹), 예측 organ 으로 후보 제한(cross-organ 오류 방지) ---
    query = build_query_template(site, proc, cb, slot_codes)
    report = retrieve_report(ART, record["z"], query, organ=organ)
    if not report:
        report = f"{site}, {em};{record.get('report_body','')}"

    # [c10] 리포트 라우팅: 일부 organ 은 모델 리포트 슬롯(직접 분류)으로 교체
    if _REPORT_ROUTING and organ in _ROUTE_ORGANS:
        rq = _report_slot_qid(organ, cb)
        if rq is not None and rq in slot_codes:
            slot_rep = cb["code2ans"].get(rq, {}).get(slot_codes[rq], "")
            if slot_rep:
                report = slot_rep

    # [A7] grade override: replace the routed prostate report grade with the grade slot value
    if _GRADE_OVERRIDE:
        report, _go = _grade_override(report, cb, slot_codes, organ)
        if _go:
            print("[grade-override] " + "  ".join(_go), flush=True)

    # --- 에이전틱 자기검증: 진단 슬롯 모순 감지→#1진단 앵커로 해소 (env on) ---
    _sv_over, _sv_trace = ({}, [])
    if _SELF_VERIFY:
        _sv_over, _sv_trace = self_verify(cb, slot_codes)
        if _sv_trace:
            print("[self-verify] " + "  ".join(_sv_trace), flush=True)

    # --- CoT 구조: top-K GT 체인 중 우리 슬롯답과 분기일치 최대 골격 + 우리 답. (path2_1 레버1) ---
    skel = retrieve_chain(ART, record["z"], organ=organ, cb=cb, slot_codes=slot_codes)
    if skel:
        steps = []
        prev_nq = ""
        for st in skel:
            q = st.get("question", "")
            if not q.strip() and prev_nq.strip():   # 결손 스텝: 질문 텍스트가 직전 next_question 에만 있음 → 복원
                q = prev_nq
            a = _answer_for(q, site, proc, report, cb, slot_codes,
                            fallback=st.get("answer", ""))
            if _SELF_VERIFY and _sv_low(q) in _sv_over:       # 자기검증 교정 적용
                a = _sv_over[_sv_low(q)]
            steps.append({"question": q, "answer": a,
                          "next_question": st.get("next_question", "")})
            prev_nq = st.get("next_question", "")
        # [A7] per-branch invasion resolution (breast/bladder)
        if _INV_BRANCH:
            steps, _ib = _inv_branch_resolve(steps, skel, organ, cb)
            if _ib:
                print("[inv-branch] " + "  ".join(_ib), flush=True)
        # [A7] per-branch papillary + behavior (purity-gated); #1dx<-report only if COT_DXFIX (off in A7)
        if _COT_CONSISTENCY:
            steps, _pc = _pap_branch_resolve(steps, skel, organ, cb)
            steps, _bc = _beh_branch_resolve(steps, skel, site, cb)
            _cc = []
            if _COT_DXFIX:
                steps, _cc = _cot_consistency_fix(steps, report)
            _cons = _pc + _bc + _cc
            if _cons:
                print("[cot-consistency] " + "  ".join(_cons), flush=True)
        return _sanitize_cot(steps)

    # --- fallback: 체인 DB 없으면 seq head 템플릿 ---
    questions = predict_sequence(record["z"], organ)
    if not any(REPORT_Q_PATTERN.search(q) for q in questions):
        questions = questions + [FINAL_Q]
    pairs = [(q, _answer_for(q, site, proc, report, cb, slot_codes, "")) for q in questions]
    steps = []
    for i, (q, a) in enumerate(pairs):
        nq = pairs[i + 1][0] if i + 1 < len(pairs) else ""
        steps.append({"question": q, "answer": a, "next_question": nq})
    return _sanitize_cot(steps)
