"""
Metric B - WSI 조직/배경 ROI 자동 생성 + CONCH 임베딩 데이터셋
- 기존 trident seg(geojson) 재사용: /srv/HDD24_1/MICCAI_2026/output/contours_geojson
- 폴리곤 내부=조직, 외부(조직 bbox 주변)=배경, 5x/256
- mpp는 WSI 메타데이터에서 읽고 정상범위만 사용
- 출력: embeddings.npy, labels.npy, meta.json, 일부 ROI 이미지(점검용)
- 정직한 평가: 슬라이드 단위 group CV
"""
import sys, os, json, glob, random, warnings, argparse
import numpy as np
from PIL import Image
warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ.get("TRIDENT_DIR", os.path.expanduser("~/src/trident")))
random.seed(0); np.random.seed(0)

import torch, openslide
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

ap = argparse.ArgumentParser()
ap.add_argument("--n_slides", type=int, default=200)
ap.add_argument("--per_class", type=int, default=12)   # 슬라이드당 클래스별
ap.add_argument("--out", default="/srv/HDD24_2/REG2026/metricB_data")
args = ap.parse_args()

ROI_PX, TARGET_MAG = 256, 5.0
GJ_DIR = "/srv/HDD24_1/MICCAI_2026/output/contours_geojson"
WSI_DIR = "/srv/HDD24_1/MICCAI_2026/reg2026/train"
os.makedirs(args.out, exist_ok=True)
os.makedirs(os.path.join(args.out, "sample_rois"), exist_ok=True)

def get_mpp(slide):
    try:
        m = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "nan"))
        if 0.15 <= m <= 0.7:   # 20x~40x 정상범위만
            return m
    except Exception:
        pass
    return None

def load_polys(gj_path):
    d = json.load(open(gj_path))
    feats = d["features"] if "features" in d else [d]
    polys = []
    for f in feats:
        g = f["geometry"]; cs = g["coordinates"]
        try:
            if g["type"] == "Polygon": polys.append(Polygon(cs[0]))
            elif g["type"] == "MultiPolygon":
                for p in cs: polys.append(Polygon(p[0]))
        except Exception: pass
    polys = [p for p in polys if p.is_valid and p.area > 0]
    return unary_union(polys) if polys else None

def crop(slide, cx, cy, cl0):
    x0, y0 = int(cx - cl0//2), int(cy - cl0//2)
    return slide.read_region((x0, y0), 0, (cl0, cl0)).convert("RGB").resize((ROI_PX, ROI_PX), Image.BILINEAR)

def mask_patches(img, grid=8, frac=0.3):
    """예시 perturbation 재현: grid 셀 중 frac 비율을 검은 사각형으로 가림."""
    a = np.array(img).copy()
    cell = ROI_PX // grid
    cells = [(i, j) for i in range(grid) for j in range(grid)]
    random.shuffle(cells)
    for (i, j) in cells[:int(grid*grid*frac)]:
        a[i*cell:(i+1)*cell, j*cell:(j+1)*cell] = 0
    return Image.fromarray(a)

def sample_slide(sid):
    wsi = os.path.join(WSI_DIR, sid + ".tiff")
    gj = os.path.join(GJ_DIR, sid + ".geojson")
    if not (os.path.exists(wsi) and os.path.exists(gj)): return []
    try: slide = openslide.OpenSlide(wsi)
    except Exception: return []
    mpp = get_mpp(slide)
    if mpp is None: return []
    poly = load_polys(gj)
    if poly is None or poly.area <= 0: return []
    cl0 = int(ROI_PX * (2.0 / mpp))          # 5x crop in level0 px
    W, H = slide.dimensions
    minx, miny, maxx, maxy = poly.bounds
    pad = (maxx - minx) * 0.12
    bx = (max(0, minx-pad), max(0, miny-pad), min(W, maxx+pad), min(H, maxy+pad))
    h = cl0//2
    def ok(x, y): return h <= x <= W-h and h <= y <= H-h
    out, tries = [], 0
    nt = nb = 0
    while (nt < args.per_class or nb < args.per_class) and tries < args.per_class*300:
        tries += 1
        if nt < args.per_class:
            x, y = random.uniform(minx, maxx), random.uniform(miny, maxy)
            if poly.contains(Point(x, y)) and ok(x, y):
                out.append((crop(slide, x, y, cl0), 1, sid, int(x), int(y), mpp)); nt += 1
        if nb < args.per_class:
            x, y = random.uniform(bx[0], bx[2]), random.uniform(bx[1], bx[3])
            if (not poly.contains(Point(x, y))) and ok(x, y):
                out.append((crop(slide, x, y, cl0), 0, sid, int(x), int(y), mpp)); nb += 1
    return out

# 슬라이드 선택: prefix 섞어서 다양성
ids = sorted(os.path.basename(g)[:-8] for g in glob.glob(GJ_DIR + "/*.geojson"))
random.shuffle(ids)

print("=== conch_v15 로드 ===")
from trident.patch_encoder_models.load import encoder_factory
os.environ.setdefault("HF_HOME", "/srv/HDD24_1/hf_cache")
enc = encoder_factory("conch_v15").to("cuda").eval()
prec = getattr(enc, "precision", torch.float16)

@torch.no_grad()
def embed(imgs, bs=64):
    o = []
    for i in range(0, len(imgs), bs):
        xb = torch.stack([enc.eval_transforms(im) for im in imgs[i:i+bs]]).cuda()
        with torch.autocast("cuda", dtype=prec):
            o.append(enc(xb).float().cpu().numpy())
    return np.concatenate(o, 0)

print(f"=== {args.n_slides} 슬라이드에서 ROI 샘플링 ===")
rois, used = [], 0
for sid in ids:
    if used >= args.n_slides: break
    r = sample_slide(sid)
    if len(r) >= 2:
        rois += r; used += 1
        if used % 25 == 0: print(f"  {used} slides, {len(rois)} ROIs")
print(f"사용 슬라이드 {used} / 총 ROI {len(rois)} (마스킹 전)")

# 마스킹 augmentation: 조직 ROI마다 가린 복사본 추가 (label=조직 유지) → B2 대비
aug = []
for r in rois:
    if r[1] == 1:
        aug.append((mask_patches(r[0]), 1, r[2], r[3], r[4], r[5]))
rois = rois + aug
print(f"마스킹 조직 복사본 {len(aug)}개 추가 → 총 {len(rois)}")

imgs = [r[0] for r in rois]
y = np.array([r[1] for r in rois])
meta = [{"slide": r[2], "x": r[3], "y": r[4], "mpp": r[5], "label": int(r[1])} for r in rois]
print("=== CONCH 임베딩 ===")
X = embed(imgs)
np.save(os.path.join(args.out, "embeddings.npy"), X)
np.save(os.path.join(args.out, "labels.npy"), y)
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"))
# 점검용 ROI 일부 저장
for i in list(range(0, len(imgs), max(1, len(imgs)//40)))[:40]:
    imgs[i].save(os.path.join(args.out, "sample_rois", f"{meta[i]['label']}_{meta[i]['slide']}_{i}.jpg"))
print(f"저장: {args.out} (X={X.shape}, tissue={int(y.sum())}, bg={int((1-y).sum())})")

# === 정직한 평가: 슬라이드 단위 group CV ===
print("\n=== 슬라이드 단위 Group 5-fold CV (누수 없음) ===")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
groups = np.array([m["slide"] for m in meta])
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
accs = []
for tr, te in GroupKFold(5).split(X, y, groups):
    clf.fit(X[tr], y[tr]); accs.append((clf.predict(X[te]) == y[te]).mean())
accs = np.array(accs)
print(f"  Group-CV accuracy: {accs.mean()*100:.1f}% (+/- {accs.std()*100:.1f})")

# 예시 18개 검증
VG = "/srv/HDD24_1/MICCAI_2026/visualgrounding_ex"
rows = [l.split("\t") for l in open(f"{VG}/anonymous_rois_mapping.txt").read().strip().splitlines()]
hdr = rows[0]; li, ii = hdr.index("label"), hdr.index("image")
ex = [(Image.open(f"{VG}/rois/{r[ii]}").convert("RGB").resize((256,256)), 1 if r[li]=="tissue" else 0) for r in rows[1:]]
Xe = embed([e[0] for e in ex]); ye = np.array([e[1] for e in ex])
clf.fit(X, y)
print(f"  예시 18개 정확도: {(clf.predict(Xe)==ye).mean()*100:.0f}%")
import joblib; joblib.dump(clf, os.path.join(args.out, "gate_clf.joblib"))
print("  분류기 저장: gate_clf.joblib")
