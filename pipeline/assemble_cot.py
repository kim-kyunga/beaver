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

_REPORT_ROUTING = os.environ.get("COT_REPORT_ROUTING", "0") == "1"
_ROUTE_ORGANS = {o.strip() for o in os.environ.get(
    "COT_ROUTE_ORGANS", "breast,bladder,stomach").split(",") if o.strip()}
_REP_QID_CACHE = {}


def _report_slot_qid(organ, cb):
    if organ not in _REP_QID_CACHE:
        rq = None
        for qid, qt in cb["qid2qtext"].items():
            if REPORT_Q_PATTERN.search(qt):
                rq = qid
                break
        _REP_QID_CACHE[organ] = rq
    return _REP_QID_CACHE[organ]


def _procedure_answer(em_name, organ):
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


_MAL_KW = ("carcinoma", "sarcoma", "melanoma", "lymphoma", "blastoma", "leukemia", "malignant", "adenocarcinoma")
def _sv_low(t): return re.sub(r"\s+", " ", str(t).strip().lower())
def _sv_neg(t):
    t = _sv_low(t)
    return t.startswith("no") or "there is no" in t or "no evidence" in t or "negative for" in t or "no residual" in t
def _sv_mal(t): return (not _sv_neg(t)) and any(k in _sv_low(t) for k in _MAL_KW)
def _sv_inv(t): t2 = _sv_low(t); return (not _sv_neg(t)) and ("invasive" in t2 or "invasion" in t2)
def _sv_slottext(cb, slot_codes, qtext):
    qid = cb["qtext2qid"].get(qtext)
    if qid and qid in slot_codes:
        return cb["code2ans"].get(qid, {}).get(slot_codes[qid], "")
    return None
def _sv_opt(cb, qtext, positive):
    qid = cb["qtext2qid"].get(qtext)
    if not qid:
        return None
    for txt in cb["code2ans"].get(qid, {}).values():
        if positive and not _sv_neg(txt):
            return txt
        if (not positive) and _sv_neg(txt):
            return txt
    return None


_SELF_VERIFY = os.environ.get("COT_SELF_VERIFY", "0") == "1"
_COUNTERFACTUAL = os.environ.get("COT_COUNTERFACTUAL", "0") == "1"


def self_verify(cb, slot_codes):
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
    trace.append(f"self-verify: anchor on '#1 diagnosis={dx1}'.")
    checks = [
        ("is there any invasion present?", "is there any invasion present", _sv_inv(dx1), True, "invasive diagnosis but 'no invasion'"),
        ("is there any abnormality present?", "is there any abnormality present", _sv_mal(dx1), True, "malignant diagnosis but 'no abnormality'"),
        ("is there any neoplasm present?", "is there any neoplasm present", _sv_mal(dx1), True, "malignant diagnosis but 'no neoplasm'"),
    ]
    for qkey, sub, anchor_pos, want_pos, desc in checks:
        cur = g(sub)
        if cur and anchor_pos and _sv_neg(cur):
            corr = _sv_opt(cb, qkey, want_pos)
            if corr:
                overrides[_sv_low(qkey)] = corr
                trace.append(f"  contradiction: {desc} ('{cur}') -> corrected to '{corr}'.")
    if not overrides:
        trace.append("  slots consistent.")
    return overrides, trace


_SV_REPORT = os.environ.get("COT_SV_REPORT", "0") == "1"
_SV_REPORT_THR = float(os.environ.get("COT_SV_REPORT_THR", "0.95"))
_DX1_HEADS = None


def _load_dx1_heads():
    global _DX1_HEADS
    if _DX1_HEADS is None:
        try:
            import pickle
            _DX1_HEADS = pickle.load(open(f"{ART}/dx1_heads.pkl", "rb"))
        except Exception:
            _DX1_HEADS = {}
    return _DX1_HEADS


def _dx1_conf(z, organ):
    h = _load_dx1_heads().get(organ)
    if h is None:
        return 1.0
    lg = np.asarray(z, dtype=np.float32) @ np.asarray(h["W"]).T + np.asarray(h["b"])
    e = np.exp(lg - lg.max())
    return float((e / e.sum()).max())


def _svr_dx1(cb, slot_codes):
    for qt, qid in cb.get("qtext2qid", {}).items():
        if re.match(r"what is the #1 diagnosis\??$", _sv_low(qt)) and qid in slot_codes:
            return cb["code2ans"].get(qid, {}).get(slot_codes[qid], "")
    return ""


def _svr_fix(rep, new_dx):
    r = rep.replace("\\n", "\n")
    if re.search(r"\b1\.\s*[^\n]+", r):
        r2 = re.sub(r"(\b1\.\s*)[^\n]+", lambda m: m.group(1) + new_dx, r, count=1)
    elif ";" in r:
        head, tail = r.split(";", 1); L = tail.strip().split("\n"); L[0] = new_dx
        r2 = head + ";\n  " + "\n".join(L)
    else:
        r2 = r
    return r2.replace("\n", "\\n")


def _sv_report_fix(report, cb, slot_codes, z=None, organ=None):
    dx1 = _svr_dx1(cb, slot_codes)
    if not dx1 or not report:
        return report, False
    if (not _sv_mal(_svr_primary(report))) and _sv_mal(dx1):
        if z is not None and organ is not None and _dx1_conf(z, organ) < _SV_REPORT_THR:
            return report, False
        return _svr_fix(report, dx1), True
    return report, False


# ----- per-node confidence gate: accept the slot answer if its softmax confidence
# clears the per-(organ,node) tau, otherwise defer to the case-grounded retrieval answer.
_CONF_GATE = os.environ.get("COT_CONF_GATE", "0") == "1"
try:
    _TAU = json.load(open(f"{ART}/conf_tau.json"))
except Exception:
    _TAU = {}
_GATE_NODES = [("#1dx", "what is the #1 diagnosis"),
               ("invasion", "is there any invasion present"),
               ("histtype", "what is the histologic type of neoplasm"),
               ("abnormality", "is there any abnormality present"),
               ("neoplasm", "is there any neoplasm present"),
               ("behavior", "what is the behavior of neoplasm")]


def _gate_node(q):
    ql = _sv_low(q)
    for node, sub in _GATE_NODES:
        if sub in ql:
            return node
    return None


_INV_BRANCH = os.environ.get("COT_INV_BRANCH", "0") == "1"
_COT_CONSISTENCY = os.environ.get("COT_CONSISTENCY", "0") == "1"
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
    slot_conf = record.get("slot_conf", {}).get(slotkey, {})
    proc = _procedure_answer(em, organ)

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
            print(f"[counterfactual] skip ({_e})", flush=True)

    query = build_query_template(site, proc, cb, slot_codes)
    report = retrieve_report(ART, record["z"], query, organ=organ)
    if not report:
        report = f"{site}, {em};{record.get('report_body','')}"

    if _REPORT_ROUTING and organ in _ROUTE_ORGANS:
        rq = _report_slot_qid(organ, cb)
        if rq is not None and rq in slot_codes:
            slot_rep = cb["code2ans"].get(rq, {}).get(slot_codes[rq], "")
            if slot_rep:
                report = slot_rep

    if _GRADE_OVERRIDE:
        report, _go = _grade_override(report, cb, slot_codes, organ)
        if _go:
            print("[grade-override] " + "  ".join(_go), flush=True)

    if _SV_REPORT:
        report, _svr_ch = _sv_report_fix(report, cb, slot_codes, record["z"], organ)
        if _svr_ch:
            print("[sv-report] report primary <- confident #1-diagnosis slot", flush=True)

    _sv_over, _sv_trace = ({}, [])
    if _SELF_VERIFY:
        _sv_over, _sv_trace = self_verify(cb, slot_codes)
        if _sv_trace:
            print("[self-verify] " + "  ".join(_sv_trace), flush=True)

    skel = retrieve_chain(ART, record["z"], organ=organ, cb=cb, slot_codes=slot_codes)
    if skel:
        steps = []
        prev_nq = ""
        for st in skel:
            q = st.get("question", "")
            if not q.strip() and prev_nq.strip():
                q = prev_nq
            a = _answer_for(q, site, proc, report, cb, slot_codes,
                            fallback=st.get("answer", ""))
            if _CONF_GATE and _TAU:
                nd = _gate_node(q)
                if nd:
                    qid = cb["qtext2qid"].get(q.strip())
                    tau = _TAU.get(organ, {}).get(nd)
                    conf = slot_conf.get(qid) if qid else None
                    if qid in slot_codes and tau is not None and conf is not None and conf < tau:
                        a = st.get("answer", "")
            if _SELF_VERIFY and _sv_low(q) in _sv_over:
                a = _sv_over[_sv_low(q)]
            steps.append({"question": q, "answer": a,
                          "next_question": st.get("next_question", "")})
            prev_nq = st.get("next_question", "")
        if _INV_BRANCH:
            steps, _ib = _inv_branch_resolve(steps, skel, organ, cb)
            if _ib:
                print("[inv-branch] " + "  ".join(_ib), flush=True)
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

    questions = predict_sequence(record["z"], organ)
    if not any(REPORT_Q_PATTERN.search(q) for q in questions):
        questions = questions + [FINAL_Q]
    pairs = [(q, _answer_for(q, site, proc, report, cb, slot_codes, "")) for q in questions]
    steps = []
    for i, (q, a) in enumerate(pairs):
        nq = pairs[i + 1][0] if i + 1 < len(pairs) else ""
        steps.append({"question": q, "answer": a, "next_question": nq})
    return _sanitize_cot(steps)
