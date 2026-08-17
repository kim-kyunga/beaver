"""
배포용 컴팩트 검색 DB 아티팩트 생성 + 단일 쿼리(추론 1케이스) 시간 측정.
산출: report_eval/db/db_z.npy (float32 [N,1024], L2정규화), db_reports.json, db_organ.json
"""
import json, os, pickle, sys, time
import numpy as np

sys.path.insert(0, "/srv/HDD24_1/NARWHAL2/cot")
import nar2_common as C

OUT = "/srv/HDD24_1/NARWHAL2/report_eval/db"
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
emb = pickle.load(open(os.path.join(C.ART, "train_embed.pkl"), "rb"))
cot = json.load(open(C.TRAIN_COT))
gt_report, gt_organ = {}, {}
for c in cot:
    organ = c.get("organ", "").strip().lower()
    if organ not in C.ORGANS:
        continue
    stem = c["id"].replace(".tiff", "")
    gt_report[stem] = c["chain-of-thought"][-1]["answer"]
    gt_organ[stem] = organ
print(f"source load {time.time()-t0:.1f}s")

stems = [s for s in emb if s in gt_report]
Z = np.stack([emb[s]["z"] for s in stems]).astype(np.float32)
Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
reports = [gt_report[s] for s in stems]
organs = [gt_organ[s] for s in stems]
np.save(os.path.join(OUT, "db_z.npy"), Z)
json.dump(reports, open(os.path.join(OUT, "db_reports.json"), "w"), ensure_ascii=False)
json.dump(organs, open(os.path.join(OUT, "db_organ.json"), "w"))
print(f"DB built: N={len(stems)}  z.npy={os.path.getsize(OUT+'/db_z.npy')/1e6:.1f}MB "
      f"reports.json={os.path.getsize(OUT+'/db_reports.json')/1e6:.1f}MB")

# ================= 추론 시간 측정 =================
import re
TOKRE = re.compile(r"\b\w+\b")
def toks(t): return set(TOKRE.findall(t.lower()))

# --- 컨테이너 시작 시 1회: DB 로드 ---
t = time.time()
DBZ = np.load(os.path.join(OUT, "db_z.npy"))
DBR = json.load(open(os.path.join(OUT, "db_reports.json")))
load_t = time.time() - t
print(f"\n[추론] DB 로드(컨테이너당 1회): {load_t*1000:.1f} ms  (z {DBZ.shape}, reports {len(DBR)})")

# --- 케이스당: 쿼리 z + slot 템플릿 으로 검색+재랭킹 ---
# (z 와 slot 템플릿은 NARWHAL forward 산출물; 여기선 임의 케이스로 시뮬레이션)
rng = np.random.RandomState(0)
times = []
for _ in range(50):
    qi = rng.randint(len(DBZ))
    z = DBZ[qi].copy()                     # 쿼리 임베딩(정규화돼 있음)
    my_t = toks(DBR[qi])                   # slot 템플릿 토큰 대용
    t = time.time()
    sims = DBZ @ (z / (np.linalg.norm(z) + 1e-8))
    topk = np.argpartition(-sims, 10)[:10]
    topk = topk[np.argsort(-sims[topk])]
    best, best_sc = int(topk[0]), -1.0
    for idx in topk:
        rt = toks(DBR[int(idx)])
        u = len(my_t | rt)
        j = (len(my_t & rt) / u) if u else 0.0
        if j > best_sc:
            best_sc, best = j, int(idx)
    _ = DBR[best]
    times.append(time.time() - t)
times = np.array(times) * 1000
print(f"[추론] 케이스당 검색+재랭킹: 평균 {times.mean():.2f} ms, 최대 {times.max():.2f} ms (50회)")
print(f"\n=> 검색이 5분 예산에 더하는 시간: 로드 {load_t*1000:.0f}ms + 쿼리 ~{times.mean():.0f}ms ≈ {(load_t*1000+times.mean())/1000:.2f}s (무시 가능)")
