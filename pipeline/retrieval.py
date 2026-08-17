"""
검색 기반 final report 생성 (디코더 자유생성 대체).

원리(held-out 측정: ranking 0.609 슬롯템플릿 → 0.867 검색 → 0.894 검색+재랭킹):
  WSI 임베딩 z 로 train DB(실제 병리 GT 리포트) 에서 코사인 top-K 최근접을 찾고,
  그 중 '예측 슬롯 내용(진단 템플릿)' 과 토큰 Jaccard 가 최대인 리포트를 고른다.
  → 이미지 유사도 + 예측 진단내용 결합. 반환 리포트는 실제 병리 문장이라
     평가기(key=의학개체명 Jaccard 0.4, emb=OpenBioLLM cos 0.3)에서 높게 나온다.

추론 비용: DB 로드 1회 ~22ms, 케이스당 검색+재랭킹 ~0.3ms (5분 예산 무관).
DB 파일: {ART}/db/db_z.npy [N,1024] L2정규화, db_reports.json [N] (model.tar.gz 동봉).
"""
import json
import os
import re

import numpy as np

_TOKRE = re.compile(r"\b\w+\b")
_DIAG_RE = re.compile(r"#(\d+)\s+diagnosis", re.I)
_NUMDIAG_RE = re.compile(r"number of diagnoses", re.I)

_DB = None       # (Z[N,1024] 정규화, reports[N])
_CHAINS = None   # [N] 각 = GT 체인 steps (db_z 와 인덱스 정렬)
_ORG = None      # [N] 각 케이스 organ (db_z 와 인덱스 정렬)
_ORG_IDX = None  # {organ: np.array(indices)} 캐시


def _toks(t):
    return set(_TOKRE.findall(t.lower()))


def load_organs(art_dir):
    """{art_dir}/db/db_organ.json 로드(1회 캐시). 없으면 []."""
    global _ORG
    if _ORG is not None:
        return _ORG
    p = os.path.join(art_dir, "db", "db_organ.json")
    _ORG = json.load(open(p)) if os.path.exists(p) else []
    return _ORG


def _candidates(art_dir, organ):
    """예측 organ 의 DB 인덱스 배열 반환(검색 후보 제한 = cross-organ 오류 방지).
    organ 정보 없거나 해당 organ 후보 없으면 None(전체 사용)."""
    global _ORG_IDX
    if not organ:
        return None
    orgs = load_organs(art_dir)
    if not orgs:
        return None
    if _ORG_IDX is None:
        _ORG_IDX = {}
        arr = np.asarray(orgs)
        for o in set(orgs):
            _ORG_IDX[o] = np.where(arr == o)[0]
    idx = _ORG_IDX.get(organ)
    return idx if (idx is not None and len(idx) > 0) else None


def load_db(art_dir):
    """{art_dir}/db/ 에서 검색 DB 로드(1회 캐시). 없으면 None."""
    global _DB
    if _DB is not None:
        return _DB
    ddir = os.path.join(art_dir, "db")
    zpath = os.path.join(ddir, "db_z.npy")
    rpath = os.path.join(ddir, "db_reports.json")
    if not (os.path.exists(zpath) and os.path.exists(rpath)):
        _DB = (None, None)
        return _DB
    Z = np.load(zpath).astype(np.float32)
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    reports = json.load(open(rpath))
    _DB = (Z, reports)
    return _DB


def load_chains(art_dir):
    """{art_dir}/db/db_chains.json 로드(1회 캐시, db_z 와 인덱스 정렬). 없으면 None."""
    global _CHAINS
    if _CHAINS is not None:
        return _CHAINS
    p = os.path.join(art_dir, "db", "db_chains.json")
    _CHAINS = json.load(open(p)) if os.path.exists(p) else []
    return _CHAINS


def retrieve_chain(art_dir, z, organ=None, cb=None, slot_codes=None, K=20):
    """z 로 DB 최근접 GT 체인 steps 반환(next_question 분기구조 보존).
    organ 주어지면 같은 organ 후보로 제한(cross-organ 오류 방지).
    [레버1, path2_1] cb+slot_codes 주어지면 top-K 후보 중 '우리 슬롯답과
      분기결정이 가장 일치하는' 골격 선택 → 진단개수/유무분기 불일치로 인한 구조붕괴 방지.
      held-out: BPV 0.807->0.874, EdgeF1 0.945->0.966, MESS 0.947->0.969, MetricA 0.929->0.944.
      미제공 시 기존 top-1 동작(하위호환). DB/체인 없으면 None."""
    Z, _ = load_db(art_dir)
    chains = load_chains(art_dir)
    if Z is None or not chains:
        return None
    q = np.asarray(z, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    cand = _candidates(art_dir, organ)
    pool = np.arange(len(chains)) if (cand is None or len(cand) == 0) else cand
    sims = Z[pool] @ q
    if cb is not None and slot_codes:
        order = np.argsort(-sims)[:K]
        topk = [int(pool[i]) for i in order]
        def _n(s):
            return re.sub(r"\s+", " ", str(s).strip().lower())
        best_nn, best_key = topk[0], (-1.0, -1.0)
        for ci in topk:
            agree = tot = 0
            for st in chains[ci]:
                qid = cb["qtext2qid"].get(st.get("question", "").strip())
                if qid is None or qid not in slot_codes:
                    continue  # 우리가 예측하는 슬롯 질문만 비교(분기 결정 변수)
                tot += 1
                our = cb["code2ans"].get(qid, {}).get(slot_codes[qid], "")
                if _n(our) == _n(st.get("answer", "")):
                    agree += 1
            key = (agree / tot if tot else 0.0, float(Z[ci] @ q))  # 일치율 우선, 동률이면 z유사도
            if key > best_key:
                best_key, best_nn = key, ci
        return chains[best_nn]
    nn = int(pool[int(np.argmax(sims))])
    return chains[nn]


def diag_qids(cb):
    """codebook -> {'num': numdiag_qid, 'dx': {n: #n_diagnosis_qid}}."""
    num_qid, dxmap = None, {}
    for qid, qt in cb["qid2qtext"].items():
        if _NUMDIAG_RE.search(qt):
            num_qid = qid
        m = _DIAG_RE.search(qt)
        if m:
            dxmap[int(m.group(1))] = qid
    return {"num": num_qid, "dx": dxmap}


def build_query_template(site_name, proc_name, cb, slot_codes):
    """예측 슬롯으로 슬롯-템플릿 리포트 문자열 생성(검색 재랭킹 쿼리).
    GT 와 동일 포맷: '{Site}, {procedure};\\n  1. {dx1}\\n  2. {dx2} ...' (literal \\n)."""
    def ans(qid):
        if qid in slot_codes:
            return cb["code2ans"].get(qid, {}).get(slot_codes[qid], None)
        return None
    dq = diag_qids(cb)
    n = None
    if dq["num"]:
        try:
            n = int(str(ans(dq["num"])).strip())
        except (TypeError, ValueError):
            n = None
    dx_list = []
    maxn = max(dq["dx"].keys()) if dq["dx"] else 0
    for i in range(1, maxn + 1):
        a = ans(dq["dx"].get(i))
        if not a:
            break
        dx_list.append(a)
        if n is not None and len(dx_list) >= n:
            break
    body = "".join(f"\\n  {i}. {dx}" for i, dx in enumerate(dx_list, 1))
    return f"{site_name}, {proc_name.lower()};{body}"


def retrieve_report(art_dir, z, query_text, K=10, organ=None):
    """z(WSI 임베딩) 로 DB top-K 검색 → query_text 와 토큰 Jaccard 최대 리포트 반환.
    organ 주어지면 같은 organ 후보로 제한(cross-organ 오류 방지: 예측 organ 과 리포트 organ 일치 보장).
    DB 없으면 None(호출측이 fallback)."""
    Z, reports = load_db(art_dir)
    if Z is None:
        return None
    q = np.asarray(z, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    cand = _candidates(art_dir, organ)
    if cand is None:
        sims = Z @ q
        pool = np.arange(len(reports))
    else:
        sims = Z[cand] @ q
        pool = cand
    K = min(K, len(pool))
    sub = np.argpartition(-sims, K - 1)[:K]
    sub = sub[np.argsort(-sims[sub])]
    topk = [int(pool[i]) for i in sub]
    my_t = _toks(query_text)
    best, best_sc = topk[0], -1.0
    for idx in topk:
        rt = _toks(reports[idx])
        u = len(my_t | rt)
        j = (len(my_t & rt) / u) if u else 0.0
        if j > best_sc:
            best_sc, best = j, idx
    return reports[best]
