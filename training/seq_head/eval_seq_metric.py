"""
NARWHAL2 시퀀스 메트릭(BPV, Edge_F1) — eval_workflow_reasoning 의 edge 로직 그대로.
held-out 10% (seed 42, train_seq_head 와 동일 split) 에서 측정.

주의: Pathwise 의 보고치(breast BPV 0.955)는 *gold report* 입력 가정(상한).
NARWHAL2 는 이미지 임베딩 z 에서 직접 예측 → 배포 조건과 동일.
"""
import json, pickle, re
import numpy as np
import torch
from nar2_common import ART, ORGANS

REPORT_Q = re.compile(r"final\s+(pathology\s+)?report", re.I)
def canon(t): return re.sub(r"\s+", " ", t.strip().lower()).rstrip(" .?!:;,")
def edges(qs):
    tr = []
    for q in qs:
        if REPORT_Q.search(q): break
        tr.append(q)
    return set((canon(tr[i]), canon(tr[i+1])) for i in range(len(tr)-1))

emb = pickle.load(open(f"{ART}/train_embed.pkl", "rb"))
lab = json.load(open(f"{ART}/seq_labels.json"))
vocab = json.load(open(f"{ART}/seq_vocab.json"))
heads = torch.load(f"{ART}/seq_heads.pt", map_location="cpu")
g = torch.Generator().manual_seed(42)

rows = []
for organ in ORGANS:
    stems = [s for s in lab if lab[s]["organ"] == organ and s in emb]
    perm = torch.randperm(len(stems), generator=g)
    n_val = max(1, len(stems)//10)
    val = [stems[i] for i in perm[:n_val].tolist()]
    h = heads[organ]; W, b = h["W"], h["b"]
    bpv, ef1 = [], []
    for s in val:
        z = torch.from_numpy(emb[s]["z"].astype(np.float32))
        pid = int((z @ W.T + b).argmax(-1).item())
        gid = lab[s]["seq_id"]
        Ep, Eg = edges(vocab[organ][pid]), edges(vocab[organ][gid])
        bpv.append(1.0 if Ep == Eg else 0.0)
        tp = len(Ep & Eg); fp = len(Ep-Eg); fn = len(Eg-Ep)
        pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
        ef1.append(2*pr*rc/(pr+rc) if pr+rc else 0.0)
    rows.append((organ, len(val), float(np.mean(bpv)), float(np.mean(ef1))))
    print(f"  {organ:9s} n={len(val):4d}  BPV={np.mean(bpv):.3f}  Edge_F1={np.mean(ef1):.3f}")

tot = sum(r[1] for r in rows)
wbpv = sum(r[1]*r[2] for r in rows)/tot
wef1 = sum(r[1]*r[3] for r in rows)/tot
print(f"[seq-metric] weighted  BPV={wbpv:.3f}  Edge_F1={wef1:.3f}")
print(f"[seq-metric] 시퀀스 기여분 0.05*BPV+0.30*Edge_F1 = {0.05*wbpv+0.30*wef1:.3f} / 0.35")
