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
    if not t: return None
    t = _norm(t); o = _norm(organ)
    if o == "breast":
        m = re.search(r"grade\s+(iv|iii|ii|i)\b", t)
        if m: return "g" + _ROMAN.get(m.group(1), m.group(1))
        m = re.search(r"grade\s+([1-3])\b", t)
        if m: return "g" + m.group(1)
        m = re.search(r"\b([3-9])\b", t)
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
    for st in chain:
        q = _norm(st.get("question", ""))
        if any(kk in q for kk in keys):
            a = st.get("answer", "")
            if a: return _norm(a)
    return None

def _chain_attrs(chain):
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

_TRUE = ("present", "yes", "positive", "true", "1", "invasive", "malignant")
_FALSE = ("absent", "no", "negative", "false", "0", "benign", "in situ", "none")
def _to_bool(v):
    s = str(v).strip().lower()
    if s in _TRUE: return True
    if s in _FALSE: return False
    return None

def parse_cf_spec():
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
    if not spec or baseline is None: return None, None
    want = {"mal": spec.get("malignancy", baseline.get("mal")),
            "inv": spec.get("invasion", baseline.get("inv")),
            "grade": spec.get("grade", baseline.get("grade")),
            "sub": spec.get("subtype", baseline.get("subtype"))}
    if want["mal"] is False:
        want["inv"] = want["grade"] = want["sub"] = None
    lock = {"mal"}
    if "invasion" in spec: lock.add("inv")
    if "grade" in spec: lock.add("grade")
    if "subtype" in spec: lock.add("sub")
    return cond_retrieve(art_dir, z, organ, want, lock)
