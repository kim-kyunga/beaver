"""
per-organ 시퀀스 분류 헤드 학습.
입력: NARWHAL ABMIL 임베딩 z[1024]  (train_embed.pkl)
타깃: GT seq_id  (seq_labels.json, build_seq_labels.py 산출)

작은 어휘(organ당 7~58 시퀀스) → 단순 linear + CE 로 충분.
출력: artifacts/seq_heads.pt = {organ: {"W":tensor, "b":tensor, "n_seq":int}}
"""
import json
import pickle

import numpy as np
import torch
import torch.nn as nn

from nar2_common import ART, ORGANS

EMB = f"{ART}/train_embed.pkl"
LAB = f"{ART}/seq_labels.json"

emb = pickle.load(open(EMB, "rb"))
lab = json.load(open(LAB))
print(f"[seq] embeddings={len(emb)}  labels={len(lab)}")

g = torch.Generator().manual_seed(42)
heads = {}
summary = []
for organ in ORGANS:
    stems = [s for s in lab if lab[s]["organ"] == organ and s in emb]
    if not stems:
        print(f"  {organ}: no data"); continue
    X = torch.tensor(np.stack([emb[s]["z"] for s in stems]), dtype=torch.float32)
    y = torch.tensor([lab[s]["seq_id"] for s in stems], dtype=torch.long)
    n_seq = int(y.max().item()) + 1

    # 90/10 split
    perm = torch.randperm(len(stems), generator=g)
    n_val = max(1, len(stems) // 10)
    vi, ti = perm[:n_val], perm[n_val:]
    Xt, yt, Xv, yv = X[ti], y[ti], X[vi], y[vi]

    head = nn.Linear(1024, n_seq)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best_acc, best_state = -1, None
    for ep in range(300):
        head.train(); opt.zero_grad()
        loss = lossf(head(Xt), yt); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == 299:
            head.eval()
            with torch.no_grad():
                va = (head(Xv).argmax(-1) == yv).float().mean().item()
            if va >= best_acc:
                best_acc = va
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    with torch.no_grad():
        ta = (head(Xt).argmax(-1) == yt).float().mean().item()
        # majority baseline
        maj = torch.bincount(yt).max().item() / len(yt)
    heads[organ] = {"W": head.weight.detach(), "b": head.bias.detach(), "n_seq": n_seq}
    summary.append((organ, len(stems), n_seq, ta, best_acc, maj))
    print(f"  {organ:9s} n={len(stems):5d} n_seq={n_seq:3d} "
          f"train_acc={ta:.3f} val_acc={best_acc:.3f} (majority={maj:.3f})")

torch.save(heads, f"{ART}/seq_heads.pt")
wa = np.average([s[4] for s in summary], weights=[s[1] for s in summary])
print(f"[seq] wrote seq_heads.pt | weighted val_acc={wa:.3f}")
