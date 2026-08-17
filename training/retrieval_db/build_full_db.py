"""
배포용 검색 DB(인덱스 정렬) 재생성: db_z.npy / db_reports.json / db_chains.json / db_organ.json.
- db_z.npy [N,1024] L2정규화 (검색 키)
- db_reports.json [N] (리포트 검색용; 기존과 동일 내용/순서)
- db_chains.json [N] 각 = GT 체인 steps [{question, answer, next_question}] (★체인구조 검색용; next_question 분기 보존)
- db_organ.json [N]
배포 DB = train 전체(임베딩된 9654). 테스트 슬라이드는 DB에 없으므로 leakage 아님.
"""
import json, os, pickle, sys
import numpy as np

sys.path.insert(0, "/srv/HDD24_1/NARWHAL2/cot")
import nar2_common as C

OUT = "/srv/HDD24_1/NARWHAL2/artifacts/db"
os.makedirs(OUT, exist_ok=True)

emb = pickle.load(open(os.path.join(C.ART, "train_embed.pkl"), "rb"))
cot = json.load(open(C.TRAIN_COT))
gt_report, gt_chain, gt_organ = {}, {}, {}
for c in cot:
    organ = c.get("organ", "").strip().lower()
    if organ not in C.ORGANS:
        continue
    stem = c["id"].replace(".tiff", "")
    chain = c["chain-of-thought"]
    gt_report[stem] = chain[-1]["answer"]
    gt_chain[stem] = [{"question": s.get("question", ""), "answer": s.get("answer", ""),
                       "next_question": s.get("next_question", "")} for s in chain]
    gt_organ[stem] = organ

stems = [s for s in emb if s in gt_report]   # 정렬 기준(고정)
Z = np.stack([emb[s]["z"] for s in stems]).astype(np.float32)
Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
np.save(os.path.join(OUT, "db_z.npy"), Z)
json.dump([gt_report[s] for s in stems], open(os.path.join(OUT, "db_reports.json"), "w"), ensure_ascii=False)
json.dump([gt_chain[s] for s in stems], open(os.path.join(OUT, "db_chains.json"), "w"), ensure_ascii=False)
json.dump([gt_organ[s] for s in stems], open(os.path.join(OUT, "db_organ.json"), "w"))
print(f"N={len(stems)}")
for f in ["db_z.npy", "db_reports.json", "db_chains.json", "db_organ.json"]:
    print(f"  {f}: {os.path.getsize(os.path.join(OUT,f))/1e6:.1f} MB")
