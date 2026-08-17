"""[containerA4] Counterfactual Reasoning 경로 (Metric C) — 조건부 검색.
설계: 정상 추론은 안 건드림. counterfactual 수정(속성 override)이 주입되면,
그 속성에 매칭되는 *실제 GT 케이스*만으로 검색풀을 좁혀 z-최근접을 골라
결론(=실제 리포트)과 하류 체인(=실제·내부일관)을 가져온다.
  → 결론이 속성의 함수가 됨, 실제 GT텍스트라 품질↑, 하류 자동 일관.

지원 속성(held-out 검증, 반응/일관):
  - malignancy(=abnormality/neoplasm/malignant 그룹, 의미상 동일축): 전 organ 100%/100%
  - invasion: breast·bladder 100%/100%
  - grade(organ별 등급체계): 92~100%/100%
  - subtype(histologic type): 96~100%/100%
relaxation 순서: 가장 약한 조건(subtype→grade→invasion)부터 풀어 malignancy를 가장 오래 유지.

수정 주입 형식(미공개 → 잠정): env `COT_CF_SPEC` = JSON, 예:
  {"malignancy":"absent"}  {"invasion":"present"}  {"grade":"grade 3"}  {"subtype":"adenocarcinoma"}
대회 인터페이스 확정되면 parse_cf_spec() 만 1곳 조정.
"""
import os, re, json
import numpy as np
from retrieval import load_db, load_chains, load_organs, _candidates

def _norm(t): return re.sub(r"\s+", " ", str(t).strip().lower())
_NEG = lambda t: _norm(t).startswith(
    ("no", "there is no", "negative", "no evidence", "no residual", "absent", "not identified"))
_MAL_KW = ("carcinoma", "sarcoma", "melanoma", "lymphoma", "blastoma", "leukemia",
           "malignant", "adenocarcinoma", "malignancy")
def _is_mal(t): t = str(t).lower(); return (not _NEG(t)) and any(k in t for k in _MAL_KW)
def _is_inv(t): t = str(t).lower(); return (not _NEG(t)) and ("invasive" in t or "invasion" in t)

# organ별 등급 질문 키워드 / subtype 질문 키워드
_GRADE_KEYS = {
    "breast": ["grade of neoplasm", "overall score"],
    "prostate": ["gleason score", "grade group"],
    "colon": ["grade of dysplasia"], "bladder": ["grade of dysplasia"],
    "cervix": ["grade of dysplasia", "grade of atypia"],
    "stomach": ["grade of neoplasm"], "lung": ["grade of neoplasm"],
}
_SUB_KEYS = ["histologic type of neoplasm"]
_ALL_GRADE = sorted({k for v in _GRADE_KEYS.values() for k in v})

_ROMAN = {"iv": "4", "iii": "3", "ii": "2", "i": "1"}
def _canon_grade(t, organ=""):
    """organ별 grade 표기를 canonical로 정규화 → 어떤 표기로 주입돼도 매칭.
    breast: Nottingham총점(3-9)↔Grade I/II/III(tier). prostate: Gleason↔grade group. 그 외: low/high·분화도·숫자."""
    if not t: return None
    t = _norm(t); o = _norm(organ)
    if o == "breast":
        m = re.search(r"grade\s+(iv|iii|ii|i)\b", t)
        if m: return "g" + _ROMAN.get(m.group(1), m.group(1))
        m = re.search(r"grade\s+([1-3])\b", t)
        if m: return "g" + m.group(1)
        m = re.search(r"\b([3-9])\b", t)               # Nottingham 총점 → tier
        if m:
            sc = int(m.group(1)); return "g" + ("1" if sc <= 5 else "2" if sc <= 7 else "3")
        return t
    if o == "prostate":
        m = re.search(r"grade group\s+([1-5])", t)
        if m: return "gg" + m.group(1)
        m = re.search(r"gleason.*?\b(\d{1,2})\b", t) or re.search(r"\b(6|7|8|9|10)\b", t)
        if m:
            sc = int(m.group(1)); return "gg" + {6: "1", 7: "2", 8: "4", 9: "5", 10: "5"}.get(sc, str(sc))
        return t
    if "high grade" in t or "high-grade" in t: return "high"
    if "low grade" in t or "low-grade" in t: return "low"
    if "poorly" in t: return "g3"
    if "moderately" in t: return "g2"
    if "well differ" in t: return "g1"
    m = re.search(r"grade\s+(iv|iii|ii|i)\b", t)
    if m: return "g" + _ROMAN.get(m.group(1), m.group(1))
    m = re.search(r"grade\s+([1-4])\b", t)
    if m: return "g" + m.group(1)
    return t

def _chain_val(chain, keys):
    """체인서 keys 매칭 질문의 답(normalized). 없으면 None."""
    for st in chain:
        q = _norm(st.get("question", ""))
        if any(kk in q for kk in keys):
            a = st.get("answer", "")
            if a: return _norm(a)
    return None

def _chain_attrs(chain):
    """GT 체인서 (malignant, invasive) 추론. None=불명. (malignancy=abnormality/neoplasm/malignancy 그룹)"""
    mal = inv = None
    for st in chain:
        q = str(st.get("question", "")).lower(); a = str(st.get("answer", ""))
        if "invasion" in q and a: inv = (not _NEG(a))
        if ("abnormality" in q or "neoplasm" in q or "malignancy" in q) and a: mal = (not _NEG(a))
    dx = chain[-1].get("answer", "") if chain else ""
    if _is_mal(dx): mal = True
    if _is_inv(dx): inv = True
    if dx and _NEG(dx): mal = False
    return mal, inv

_ATTR = None
def _attr_index(art_dir):
    """DB 각 케이스의 (mal, inv, grade, subtype) 인덱스 (1회 캐시, db_z 와 정렬)."""
    global _ATTR
    if _ATTR is not None: return _ATTR
    chains = load_chains(art_dir); orgs = load_organs(art_dir)
    n = len(chains)
    mal = np.empty(n, dtype=object); inv = np.empty(n, dtype=object)
    grade = np.empty(n, dtype=object); sub = np.empty(n, dtype=object)
    for i, ch in enumerate(chains):
        m, v = _chain_attrs(ch); mal[i] = m; inv[i] = v
        o = _norm(orgs[i]) if (orgs and i < len(orgs)) else ""
        gk = _GRADE_KEYS.get(o, _ALL_GRADE)
        grade[i] = _canon_grade(_chain_val(ch, gk), o); sub[i] = _chain_val(ch, _SUB_KEYS)
    _ATTR = (mal, inv, grade, sub)
    return _ATTR

def cond_retrieve(art_dir, z, organ, want, lock=("mal",)):
    """organ + 속성조건 매칭 GT 중 z-최근접의 (report, chain).
    want={'mal','inv','grade','sub'}(None=무시). lock=수정속성(가장 오래 유지). malignancy는 항상 lock.
    완화: 비-lock 약한속성(subtype→grade→invasion)부터 풀어, *바꾸려는 속성*은 끝까지 유지."""
    Z, reports = load_db(art_dir)
    chains = load_chains(art_dir)
    if Z is None or not chains: return None, None
    mal, inv, grade, sub = _attr_index(art_dir)
    lock = set(lock) | {"mal"}
    wv = {"mal": want.get("mal"), "inv": want.get("inv"),
          "grade": _canon_grade(want.get("grade"), organ) if want.get("grade") else None,
          "sub": _norm(want.get("sub")) if want.get("sub") else None}
    arr = {"mal": mal, "inv": inv, "grade": grade, "sub": sub}
    cand = _candidates(art_dir, organ)
    pool = np.arange(len(reports)) if (cand is None or len(cand) == 0) else np.asarray(cand)
    def _sel(active):
        m = np.ones(len(pool), dtype=bool)
        for k in active:
            if wv[k] is not None: m &= np.array([arr[k][i] == wv[k] for i in pool])
        return pool[m]
    # 완화순서: 비-lock 약한순(sub→grade→inv) 먼저, 그 다음 lock(mal 제외)
    weak = ["sub", "grade", "inv"]
    order = [k for k in weak if k not in lock] + [k for k in weak if k in lock]
    active = {"mal", "inv", "grade", "sub"}
    sel = _sel(active)
    for k in order:
        if len(sel): break
        active.discard(k); sel = _sel(active)
    if len(sel) == 0: sel = pool
    q = np.asarray(z, dtype=np.float32); q = q / (np.linalg.norm(q) + 1e-8)
    best = int(sel[int(np.argmax(Z[sel] @ q))])
    return reports[best], chains[best]

# ---- counterfactual 수정 명세 파싱 ----
_TRUE = ("present", "yes", "positive", "true", "1", "invasive", "malignant")
_FALSE = ("absent", "no", "negative", "false", "0", "benign", "in situ", "none")
def _to_bool(v):
    s = str(v).strip().lower()
    if s in _TRUE: return True
    if s in _FALSE: return False
    return None

def parse_cf_spec():
    """env COT_CF_SPEC(JSON) → {'malignancy':bool, 'invasion':bool, 'grade':str, 'subtype':str} 또는 None.
    malignancy 그룹 = abnormality/neoplasm/malignancy (의미상 동일축). 대회 인터페이스 확정시 여기만 조정."""
    raw = os.environ.get("COT_CF_SPEC", "").strip()
    if not raw: return None
    try:
        d = json.loads(raw)
    except Exception:
        return None
    out = {}
    for k, v in d.items():
        kl = str(k).lower()
        if "invasion" in kl or "invasive" in kl: out["invasion"] = _to_bool(v)
        elif "malig" in kl or "neoplasm" in kl or "abnormal" in kl: out["malignancy"] = _to_bool(v)
        elif "grade" in kl or "gleason" in kl: out["grade"] = str(v)
        elif "subtype" in kl or "histologic" in kl or "type" in kl: out["subtype"] = str(v)
    return out or None

def _find_qid(cb, *subs):
    for sub in subs:
        for qt, qid in cb["qtext2qid"].items():
            if sub in _norm(qt): return qid
    return None
def baseline_from_slots(cb, slot_codes, organ=""):
    """모델 슬롯예측에서 baseline {mal, inv, grade, subtype}."""
    def _ans(qid): return cb["code2ans"].get(qid, {}).get(slot_codes.get(qid), "") if qid else ""
    o = _norm(organ); b = {"mal": None, "inv": None, "grade": None, "subtype": None}
    a = _ans(_find_qid(cb, "is there any abnormality present", "is there any neoplasm present", "is there any malignancy"))
    if a: b["mal"] = (not _NEG(a))
    a = _ans(_find_qid(cb, "is there any invasion present"))
    if a: b["inv"] = (not _NEG(a))
    dx = _ans(_find_qid(cb, "#1 diagnosis", "final diagnosis", "what is the diagnosis"))
    if _is_mal(dx): b["mal"] = True
    if _is_inv(dx): b["inv"] = True
    gk = _GRADE_KEYS.get(o, _ALL_GRADE)
    gq = _find_qid(cb, *gk);
    if gq: b["grade"] = _ans(gq) or None
    sq = _find_qid(cb, *_SUB_KEYS)
    if sq: b["subtype"] = _ans(sq) or None
    return b

def apply(art_dir, z, organ, baseline, spec):
    """spec(수정) 적용 → 조건부검색으로 (report, chain). spec/baseline 없으면 (None,None).
    수정된 속성(spec에 있는 것)을 lock=최우선유지 → 나머지 baseline 속성부터 완화."""
    if not spec or baseline is None: return None, None
    want = {"mal": spec.get("malignancy", baseline.get("mal")),
            "inv": spec.get("invasion", baseline.get("inv")),
            "grade": spec.get("grade", baseline.get("grade")),
            "sub": spec.get("subtype", baseline.get("subtype"))}
    if want["mal"] is False:   # 악성아니면 침습·등급·subtype 무의미
        want["inv"] = want["grade"] = want["sub"] = None
    lock = {"mal"}
    if "invasion" in spec: lock.add("inv")
    if "grade" in spec: lock.add("grade")
    if "subtype" in spec: lock.add("sub")
    return cond_retrieve(art_dir, z, organ, want, lock)
