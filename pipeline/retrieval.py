import json
import os
import re

import numpy as np

_TOKRE = re.compile(r"\b\w+\b")
_DIAG_RE = re.compile(r"#(\d+)\s+diagnosis", re.I)
_NUMDIAG_RE = re.compile(r"number of diagnoses", re.I)

_DB = None
_CHAINS = None
_ORG = None
_ORG_IDX = None


def _toks(t):
    return set(_TOKRE.findall(t.lower()))


def load_organs(art_dir):
    global _ORG
    if _ORG is not None:
        return _ORG
    p = os.path.join(art_dir, "db", "db_organ.json")
    _ORG = json.load(open(p)) if os.path.exists(p) else []
    return _ORG


def _candidates(art_dir, organ):
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
    global _CHAINS
    if _CHAINS is not None:
        return _CHAINS
    p = os.path.join(art_dir, "db", "db_chains.json")
    _CHAINS = json.load(open(p)) if os.path.exists(p) else []
    return _CHAINS


def retrieve_chain(art_dir, z, organ=None, cb=None, slot_codes=None, K=20):
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
                    continue
                tot += 1
                our = cb["code2ans"].get(qid, {}).get(slot_codes[qid], "")
                if _n(our) == _n(st.get("answer", "")):
                    agree += 1
            key = (agree / tot if tot else 0.0, float(Z[ci] @ q))
            if key > best_key:
                best_key, best_nn = key, ci
        return chains[best_nn]
    nn = int(pool[int(np.argmax(sims))])
    return chains[nn]


def diag_qids(cb):
    num_qid, dxmap = None, {}
    for qid, qt in cb["qid2qtext"].items():
        if _NUMDIAG_RE.search(qt):
            num_qid = qid
        m = _DIAG_RE.search(qt)
        if m:
            dxmap[int(m.group(1))] = qid
    return {"num": num_qid, "dx": dxmap}


def build_query_template(site_name, proc_name, cb, slot_codes):
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
