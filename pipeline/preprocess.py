"""
단일 WSI(TIFF) -> H-optimus-1 patch feature (N x 1536).  trident API 직접 호출.
핵심: 플랫폼 TIFF는 해상도 메타데이터가 망가져 있으므로(mpp=1000) **mpp=0.5 강제 주입**.
(dev 추출이 custom_list_of_wsis CSV로 mpp=0.5 준 것과 동일 — MODEL 학습 feature 기준값)

run_single_slide.py 의 process_slide 와 동일한 단계: seg(hest) -> coords(mag20/512) -> H-optimus-1.
제출 interf1 에서 이 함수를 그대로 호출한다.
"""
import os
import sys
import json

TRIDENT_PATH = os.environ.get("TRIDENT_PATH", "/trident")
if TRIDENT_PATH not in sys.path:
    sys.path.insert(0, TRIDENT_PATH)

import h5py
import numpy as np

from trident import load_wsi

# 제출 컨테이너는 /dev/shm 이 작아(기본 64MB) DataLoader multiprocessing worker 가
# 죽는다("worker exited unexpectedly"). 플랫폼 docker 플래그를 제어할 수 없으므로
# trident WSI 메서드(seg/feature 둘 다)가 쓰는 worker 수를 0 으로 강제한다.
import trident.wsi_objects.WSI as _wsi_mod
_wsi_mod.get_num_workers = lambda *a, **k: 0

# GC 가 주는 tiff 는 단일레벨(피라미드 없음) 타일형이라, openslide 로 읽으면 타일을
# 하나씩 기어가며 읽어 seg 가 150초+ 걸린다(타임아웃 원인). tiffslide(=tifffile 백엔드)는
# 같은 슬라이드를 3배 빨리 읽는다(75s→26s) → 기본적으로 tiffslide 를 강제한다.
# (COT_READER=openslide 로 끄면 원래 openslide 사용)
import openslide as _osl
_real_openslide = _osl.OpenSlide
_READER_PREF = os.environ.get("COT_READER", "tiffslide").lower()
def _openslide_or_tiffslide(path, *a, **k):
    if _READER_PREF == "tiffslide":
        try:
            import tiffslide
            return tiffslide.TiffSlide(path)            # 빠른 경로(기본)
        except Exception as e:
            print(f"[trident] tiffslide 실패 → openslide 로 재시도: {path} ({e})")
            return _real_openslide(path, *a, **k)
    try:
        return _real_openslide(path, *a, **k)           # COT_READER=openslide
    except Exception:
        import tiffslide
        print(f"[trident] openslide 실패 → tiffslide 로 재시도: {path}")
        return tiffslide.TiffSlide(path)
_osl.OpenSlide = _openslide_or_tiffslide

# 손상 JPEG 타일 내성(핵심): 일부 플랫폼 tiff 는 level-0 타일 일부가 손상돼('Bogus marker
# length'/'two SOI markers' 등) read_region 이 통째로 터진다 → seg 뿐 아니라 패치
# 추출(trident extract_patch_features)도 죽는다. read_region 을 한 곳에서 감싸, 실패하면
# 256 서브타일 단위로 다시 읽어 손상분만 흰색(255)으로 채우고 정상 픽셀은 보존한다.
# 이 한 패치로 모든 읽기(썸네일 밴드/패치추출)가 손상 슬라이드를 견딘다. (1500여개 복구에서 검증)
import tiffslide as _ts
_real_read_region = _ts.TiffSlide.read_region
def _tolerant_read_region(self, location, level, size, *a, **k):
    try:
        return _real_read_region(self, location, level, size, *a, **k)
    except Exception:
        from PIL import Image
        x0, y0 = location; w, h = size
        out = np.full((h, w, 3), 255, np.uint8)
        TS = 256
        for yy in range(0, h, TS):
            for xx in range(0, w, TS):
                sw, sh = min(TS, w - xx), min(TS, h - yy)
                try:
                    sub = _real_read_region(self, (x0 + xx, y0 + yy), level, (sw, sh)).convert("RGB")
                    out[yy:yy + sh, xx:xx + sw] = np.asarray(sub)
                except Exception:
                    pass  # 손상 서브타일 → 흰색 유지
        return Image.fromarray(out)
_ts.TiffSlide.read_region = _tolerant_read_region

# 타일형 슬라이드보다 큰 단일레벨/스트립형은 네이티브 get_thumbnail 이 풀해상도(~15GB)를
# 통째로 RAM 에 올려 32GB 도 OOM 난다. _seg_thumbnail 이 구조(타일/스트립)·크기로 분기해
# 네이티브(타일형 빠름) vs 밴드형(스트립/초거대 OOM안전)을 고른다. (아래 _seg_thumbnail 참고)
_NATIVE_MAX_PIXELS = int(os.environ.get("COT_NATIVE_MAX_PIXELS", str(3_500_000_000)))
# 단일층(피라미드 없음) native get_thumbnail 은 전체 픽셀을 읽으므로 *작을 때만* 안전.
# (피라미드 있는 슬라이드는 _seg_thumbnail 위쪽 pyramid 경로가 먼저 처리.) 큰 단일층은
# 아래 밴드-예산 경로로 보내 부분 seg(시간한도 내) → hang/OOM 회피.
_SINGLE_NATIVE_MAX = int(os.environ.get("COT_SINGLE_NATIVE_MAX", str(300_000_000)))
# 단일층 거대 *타일* 슬라이드(피라미드 없음) seg 시 디코딩할 타일 수 상한. seg 는 저배율(ds=16)이라
# 모든 타일을 풀해상도로 읽을 필요 없음 → 이 개수까지만 stride 로 듬성듬성 읽어 빠른 coarse seg.
_SEG_MAX_TILES = int(os.environ.get("COT_SEG_MAX_TILES", "6000"))

FORCED_MPP = 0.5
MAG = 20
PATCH = 512

# === H-optimus-1 인코더 ===
import torch
from PIL import Image as _PILImage
_HOPT_MODEL = None
_HOPT_MEAN = np.array((0.707223, 0.578729, 0.703617), dtype=np.float32)
_HOPT_STD = np.array((0.211883, 0.230117, 0.177517), dtype=np.float32)


def _load_hopt(gpu):
    global _HOPT_MODEL
    if _HOPT_MODEL is None:
        os.environ.setdefault("HF_HOME", "/opt/ml/model/hoptimus1_hf")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import timm
        _HOPT_MODEL = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1", pretrained=True,
            init_values=1e-5, dynamic_img_size=False).eval().to(f"cuda:{gpu}")
    return _HOPT_MODEL


def _ycbcr_to_rgb(arr):
    """JPEG가 YCbCr인데 tiff photometric=RGB로 잘못 태깅된 슬라이드 색교정.
    리더가 변환을 생략해 읽힌 값이 사실 YCbCr → 표준 YCbCr→RGB 적용하면 정상색."""
    a = np.asarray(arr, np.float32)
    Y, Cb, Cr = a[..., 0], a[..., 1], a[..., 2]
    R = Y + 1.402 * (Cr - 128.0)
    G = Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    B = Y + 1.772 * (Cb - 128.0)
    return np.clip(np.stack([R, G, B], -1), 0, 255).astype(np.uint8)


def _is_pink_bg(arr):
    """배경(밝은영역)이 흰색 아니라 핑크면 = YCbCr 오태깅 슬라이드. 정상은 배경=흰색(R≈G≈B)이라 False."""
    a = np.asarray(arr, np.float32).reshape(-1, 3)
    lum = a.mean(1)
    bright = a[lum > np.percentile(lum, 80)]      # 밝은 20% = 배경
    if len(bright) == 0:
        return False
    R, G, B = bright[:, 0].mean(), bright[:, 1].mean(), bright[:, 2].mean()
    return (R - max(G, B) > 25.0) and (R > 170.0)


def _hopt_extract_features(slide, coords_h5, gpu, batch=None, time_budget=None, fix_color=False):
    """좌표의 512px 조각 → 224 리사이즈 → H-optimus-1 → fp32 (학습과 동일).
    견고성 3종:
      ① **tiffslide tolerant read** (검증된 견고성 — openslide는 일부 슬라이드서 잘못읽어 폐기)
      ② 조각별 try/except(깨진타일→빈패치)  ③ 시간예산 초과시 읽은 만큼만(timeout 방지).
    PATCH_CAP=1000 + 부분조각이어도 정확도 동일(모델 bag학습, cap1000 vs 2000 진단 0.9200=0.9200 검증)."""
    import time as _tt
    batch = int(os.environ.get("COT_HOPT_BATCH", "32")) if batch is None else batch
    time_budget = float(os.environ.get("COT_HOPT_TIME_BUDGET", "180")) if time_budget is None else time_budget
    with h5py.File(coords_h5, "r") as f:
        coords = f["coords"][:]
    m = _load_hopt(gpu)
    blank = np.zeros((3, 224, 224), dtype=np.float32)
    imgs = []
    t0 = _tt.time()
    for j, (x, y) in enumerate(coords):
        if _tt.time() - t0 > time_budget:
            print(f"[hopt] 시간예산 {time_budget:.0f}s 초과 → {j}/{len(coords)}조각만 사용 (timeout 방지)", flush=True)
            break
        try:
            p = slide.read_region((int(x), int(y)), 0, (PATCH, PATCH)).convert("RGB").resize((224, 224), _PILImage.BICUBIC)
            pa = np.asarray(p, np.float32)
            if fix_color:                                  # YCbCr 오태깅 슬라이드 색교정
                pa = _ycbcr_to_rgb(pa).astype(np.float32)
            imgs.append(((pa / 255.0 - _HOPT_MEAN) / _HOPT_STD).transpose(2, 0, 1))
        except Exception:
            imgs.append(blank)   # 깨진 조각 → 빈 패치
    if not imgs:
        imgs = [blank]
    out = []
    with torch.autocast("cuda", dtype=torch.float16), torch.inference_mode():
        for i in range(0, len(imgs), batch):
            X = torch.tensor(np.stack(imgs[i:i + batch]), dtype=torch.float16).to(f"cuda:{gpu}")
            out.append(m(X).float().cpu().numpy())
            del X
    return np.concatenate(out).astype(np.float32)

# 실제 채점은 .tiff(WSI) 지만 플랫폼/Try-out 은 .mha/.png 로 줄 수 있다 → 형식별로 연다.
OPENSLIDE_EXTS = {".tif", ".tiff", ".svs", ".ndpi", ".scn", ".mrxs", ".vms", ".vmu", ".bif", ".svslide"}
ITK_EXTS = {".mha", ".mhd", ".nii", ".nrrd"}   # SimpleITK 로 읽는 의료영상 포맷


def _prepare_input(slide_path, job_dir):
    """입력 파일 -> trident 가 읽을 수 있는 (path, reader_type).
    모델은 픽셀만 보므로, 여기서 어떤 형식이든 동일한 이미지로 만들어주면 이후 단계는 같다."""
    name = str(slide_path).lower()
    ext = os.path.splitext(name)[1]
    if ext in OPENSLIDE_EXTS:
        return str(slide_path), "openslide"          # 진짜 WSI (실제 채점 데이터)
    # 의료영상: .nii.gz 처럼 .gz 로 끝나는 것도 잡히게 파일명 끝(endswith)으로 검사
    if name.endswith((".mha", ".mhd", ".nii", ".nii.gz", ".nrrd")):
        import SimpleITK as sitk
        from PIL import Image
        arr = np.squeeze(np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(slide_path)))))
        if arr.ndim == 2:                                            # 흑백 -> RGB 3채널
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = np.moveaxis(arr, 0, -1)                            # (채널,H,W) -> (H,W,채널)
        arr = arr[..., :3]
        if arr.dtype != np.uint8:                                    # 0~255 uint8 로 맞추기
            a = arr.astype(np.float32); a -= a.min()
            mx = a.max(); a = a / mx if mx > 0 else a
            arr = (a * 255).astype(np.uint8)
        os.makedirs(job_dir, exist_ok=True)
        png = os.path.join(job_dir, "converted_input.png")
        Image.fromarray(arr).save(png)
        return png, "image"                          # 변환된 png 를 일반 이미지로 로드
    return str(slide_path), "image"                  # .png/.jpg 등은 그대로 ImageWSI


PATCH_CAP = int(os.environ.get("COT_PATCH_CAP", "2000"))  # 보험②: 조각 상한(feat 시간∝패치수)
RAST_DS = 64       # geojson 래스터화 다운샘플(픽셀)
MIN_FRAC = 0.10    # 패치 내 조직 비율 이 이상이면 채택


def _ensure_patches(slide, coords_h5):
    """보험①: 조직 조각이 0개면 전체 이미지를 격자로 덮는 fallback coords 생성 (크래시 방지)."""
    n = 0
    if os.path.exists(coords_h5):
        with h5py.File(coords_h5, "r") as f:
            n = f["coords"].shape[0] if "coords" in f else 0
    if n > 0:
        return
    W, H = slide.dimensions
    xs = list(range(0, max(1, W - PATCH + 1), PATCH)) or [0]
    ys = list(range(0, max(1, H - PATCH + 1), PATCH)) or [0]
    coords = np.array([[x, y] for y in ys for x in xs], dtype=np.int64)
    os.makedirs(os.path.dirname(coords_h5), exist_ok=True)
    with h5py.File(coords_h5, "w") as f:
        d = f.create_dataset("coords", data=coords)
        for k, v in {"patch_size": PATCH, "patch_size_level0": PATCH,
                     "target_magnification": MAG, "level0_magnification": MAG,
                     "overlap": 0, "name": slide.name,
                     "level0_width": W, "level0_height": H}.items():
            d.attrs[k] = v
    print(f"[trident] zero-tissue fallback: grid {len(coords)} patches")


def _write_coords_h5(coords_h5, coords, slide, W, H):
    """coords 배열을 trident 가 읽는 h5 포맷(attrs 포함)으로 저장."""
    os.makedirs(os.path.dirname(coords_h5), exist_ok=True)
    with h5py.File(coords_h5, "w") as f:
        d = f.create_dataset("coords", data=np.asarray(coords, dtype=np.int64))
        for k, v in {"patch_size": PATCH, "patch_size_level0": PATCH,
                     "target_magnification": MAG, "level0_magnification": MAG,
                     "overlap": 0, "name": slide.name,
                     "level0_width": W, "level0_height": H}.items():
            d.attrs[k] = v


def _coords_from_geojson(geojson_path, W, H):
    """seg 가 만든 geojson 폴리곤을 빠르게 래스터화 → 512px 격자에서 조직 패치 좌표.
    (make_coords_feat.py 와 동일 로직: cv2.fillPoly, ~0.1초). trident 좌표와 Jaccard 0.99."""
    import cv2
    d = json.load(open(geojson_path))
    mw, mh = int(np.ceil(W / RAST_DS)), int(np.ceil(H / RAST_DS))
    mask = np.zeros((mh, mw), np.uint8)
    for f in d.get("features", []):
        geom = f.get("geometry") or {}
        if geom.get("type") == "Polygon":
            rings = geom["coordinates"]
        elif geom.get("type") == "MultiPolygon":
            rings = [r for poly in geom["coordinates"] for r in poly]
        else:
            continue
        for ring in rings:
            pts = (np.array(ring, np.float32) / RAST_DS).round().astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
    pb = PATCH / RAST_DS
    nx, ny = int(np.ceil(W / PATCH)), int(np.ceil(H / PATCH))
    coords = []
    for iy in range(ny):
        y0, y1 = int(iy * pb), int((iy + 1) * pb)
        row = mask[y0:min(y1, mh), :]
        if row.size == 0:
            continue
        for ix in range(nx):
            x0, x1 = int(ix * pb), int((ix + 1) * pb)
            blk = row[:, x0:min(x1, mw)]
            if blk.size and blk.mean() >= MIN_FRAC:
                coords.append([ix * PATCH, iy * PATCH])
    return np.array(coords, dtype=np.int64)


def _cap_patches(coords_h5):
    """보험②: coords 가 PATCH_CAP 보다 많으면 랜덤 부분집합으로 줄인다.
    ABMIL 은 학습 때 bag_size=256 랜덤 샘플로 훈련 → 랜덤 2000 은 학습 방식과 일치."""
    with h5py.File(coords_h5, "r") as f:
        n = f["coords"].shape[0]
        if n <= PATCH_CAP:
            return
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)
    idx = np.sort(np.random.default_rng(0).choice(n, PATCH_CAP, replace=False))
    with h5py.File(coords_h5, "w") as f:
        d = f.create_dataset("coords", data=coords[idx])
        for k, v in attrs.items():
            d.attrs[k] = v
    print(f"[trident] patch cap: {n} -> {PATCH_CAP}")


SEG_MPP = float(os.environ.get("COT_SEG_MPP", "8.0"))   # Otsu seg 해상도(µm/px); 0.5 대비 16배 축소


def _seg_thumbnail(path, reader, W, H, time_budget=None):
    """seg 용 저배율 썸네일을 읽어 (thumb_rgb, real_ds) 반환. 구조로 분기:
      - 피라미드(다중레벨) → get_thumbnail 이 낮은 레벨 읽어 *크기 무관* 빠름 (최우선)
      - 단일층 타일형 & 크지않음 → 네이티브 get_thumbnail
      - 단일층 스트립형/초거대 → 가로 밴드별 읽기 (time_budget 내 읽은 만큼 = 부분 seg, 면적 축소)
    time_budget: 밴드읽기가 이 시간(초) 넘기면 상단 읽은 만큼으로 진행(격자 쓰레기 회피)."""
    import cv2
    from PIL import Image
    ds = max(1.0, SEG_MPP / FORCED_MPP)
    out_w = max(1, int(W / ds)); out_h = max(1, int(H / ds))
    if reader == "image":                                  # png/mha 변환입력(작음): 그대로 축소
        img = np.asarray(Image.open(path).convert("RGB"))
        out = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA) if ds > 1 else img
        return out, W / out.shape[1]
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            is_tiled = bool(tf.pages[0].is_tiled)
    except Exception:
        is_tiled = False
    import tiffslide
    s = tiffslide.TiffSlide(path)
    # ① 피라미드(다중 레벨) 있으면 get_thumbnail 이 *낮은 레벨* 을 읽어 크기 무관 빠르다.
    #    huge 슬라이드 seg-hang 주원인 = 피라미드 있는데 안 쓰고 level0 밴드읽기였음 (360s→1.3s 검증).
    _nlev = getattr(s, "level_count", 1)
    if _nlev > 1:
        try:
            thumb = np.asarray(s.get_thumbnail((out_w, out_h)).convert("RGB"))
            s.close()
            print(f"[trident] 썸네일 pyramid {thumb.shape[1]}x{thumb.shape[0]} (levels={_nlev})", flush=True)
            return thumb, W / thumb.shape[1]
        except Exception as e:
            print(f"[trident] pyramid get_thumbnail 실패({e}) → 폴백", flush=True)
    if is_tiled and W * H <= _SINGLE_NATIVE_MAX:           # 단일층 타일형 & 작음 → 네이티브(빠름); 큰 단일층은 아래 밴드-예산
        try:
            thumb = np.asarray(s.get_thumbnail((out_w, out_h)).convert("RGB"))
            s.close()
            print(f"[trident] 썸네일 native(tiled) {thumb.shape[1]}x{thumb.shape[0]}", flush=True)
            return thumb, W / thumb.shape[1]
        except Exception as e:
            # JPEG 타일 손상(예: 'Bogus marker length')으로 native 가 전체 디코딩에 실패 →
            # 타일 단위로 읽어 손상분만 건너뛰는 폴백(1500여개 복구에서 검증).
            print(f"[trident] native get_thumbnail 실패({e}) → tile-tolerant 폴백", flush=True)
            thumb = _read_thumb_tiletolerant(s, W, H, ds, out_w, out_h)
            s.close()
            return thumb, W / thumb.shape[1]
    # 단일층 거대 *타일* 슬라이드(피라미드 없음): 타일을 stride 로 듬성듬성 읽어 빠른 저배율 seg.
    #   밴드로 level0 전체(768~4600Mpx)를 읽으면 90s+ 캡(검증). seg 는 저해상도면 충분하므로
    #   디코딩 타일 수를 _SEG_MAX_TILES 로 제한 → 전 슬라이드 균등 coarse seg (상단-부분 seg 보다 coverage 우수).
    if is_tiled:
        thumb = _read_thumb_sparse(s, W, H, ds, out_w, out_h, time_budget=time_budget)
        s.close()
        return thumb, W / thumb.shape[1]
    # 스트립형(비타일) → 가로 밴드별 읽기 (OOM 안전; 스트립은 빠름).
    # 밴드 읽기가 손상으로 실패하면 그 밴드만 tile-tolerant 로 메운다.
    import time as _bt
    out = np.full((out_h, out_w, 3), 255, np.uint8)
    band = 256
    y = 0
    bad_bands = 0
    _b0 = _bt.time()
    while y < H:
        if time_budget is not None and _bt.time() - _b0 > time_budget:
            # 예산 소진 → 여기까지(상단) 읽은 만큼으로 진행 = 면적 축소(진짜 seg, 격자 쓰레기 아님)
            print(f"[trident] seg 예산 {time_budget:.0f}s 소진 → 상단 {100*y//max(1,H)}%만 읽고 부분 seg", flush=True)
            break
        bh = min(band, H - y)
        oy0 = int(y / ds); oy1 = min(int((y + bh) / ds), out_h)
        try:
            reg = np.asarray(s.read_region((0, y), 0, (W, bh)).convert("RGB"))
            if oy1 > oy0:
                out[oy0:oy1] = cv2.resize(reg, (out_w, oy1 - oy0), interpolation=cv2.INTER_AREA)
            del reg
        except Exception:
            bad_bands += 1
            sub = _read_thumb_tiletolerant(s, W, bh, ds, out_w, max(1, oy1 - oy0), y0=y, quiet=True)
            if oy1 > oy0:
                out[oy0:oy1] = sub[:oy1 - oy0]
        y += bh
    s.close()
    tag = f"banded(strip/huge){' +tile-tolerant' if bad_bands else ''}"
    print(f"[trident] 썸네일 {tag} {out_w}x{out_h}" + (f" (손상밴드 {bad_bands})" if bad_bands else ""), flush=True)
    return out, W / out.shape[1]


def _read_thumb_tiletolerant(s, W, H, ds, out_w, out_h, y0=0, quiet=False):
    """손상 JPEG 타일을 견디는 썸네일 읽기: native 256타일 단위로 read_region 하며
    디코딩 실패 타일은 흰색(255)으로 건너뛰고 나머지로 조립한다. (1500여개 복구에서 검증)
    y0: 슬라이드 내 세로 오프셋(밴드 보정용). 반환 크기는 (out_h, out_w, 3)."""
    import cv2
    out = np.full((out_h, out_w, 3), 255, np.uint8)
    TS = 256
    nx, ny = (W + TS - 1) // TS, (H + TS - 1) // TS
    ok = bad = 0
    for ty in range(ny):
        for tx in range(nx):
            x, y = tx * TS, ty * TS
            w, h = min(TS, W - x), min(TS, H - y)
            try:
                reg = np.asarray(s.read_region((x, y0 + y), 0, (w, h)).convert("RGB"))
                ow_, oh_ = max(1, int(w / ds)), max(1, int(h / ds))
                sub = cv2.resize(reg, (ow_, oh_), interpolation=cv2.INTER_AREA)
                ox0, oy0 = int(x / ds), int(y / ds)
                ph = min(sub.shape[0], out_h - oy0); pw = min(sub.shape[1], out_w - ox0)
                if ph > 0 and pw > 0:
                    out[oy0:oy0 + ph, ox0:ox0 + pw] = sub[:ph, :pw]
                ok += 1
            except Exception:
                bad += 1
    if not quiet:
        print(f"[trident] tile-tolerant 썸네일 {out_w}x{out_h} "
              f"(타일 {ok} ok / {bad} 손상 {100*bad/max(1,ok+bad):.1f}%)", flush=True)
    return out


def _read_thumb_sparse(s, W, H, ds, out_w, out_h, tile=256, time_budget=None):
    """단일층 거대 타일슬라이드(피라미드 없음)용 빠른 저배율 seg 썸네일.
    get_thumbnail/밴드읽기는 level0 전체(수천 Mpx)를 디코딩 → 90s+. seg(조직마스크)는 저해상도면
    충분하므로 타일을 stride 간격으로만 디코딩(<= _SEG_MAX_TILES 개)하고 각 stride 블록을 그 대표타일로
    채운다. 전 슬라이드 균등 coverage (상단만 읽는 부분 seg 보다 우수). 손상타일은 흰색(bg) 유지."""
    import cv2, math, time as _bt
    nx = (W + tile - 1) // tile
    ny = (H + tile - 1) // tile
    stride = max(1, int(math.ceil(math.sqrt((nx * ny) / max(1, _SEG_MAX_TILES)))))
    out = np.full((out_h, out_w, 3), 255, np.uint8)
    _b0 = _bt.time(); n_read = n_bad = 0
    for ty in range(0, ny, stride):
        if time_budget is not None and _bt.time() - _b0 > time_budget:
            print(f"[trident] sparse seg 예산 {time_budget:.0f}s 소진 → 상단 "
                  f"{100*ty//max(1,ny)}%까지", flush=True)
            break
        y = ty * tile
        bh = min(tile, H - y)                       # 대표타일 1개 높이
        blk_h = min(tile * stride, H - y)           # stride 블록 전체 높이(출력 채울 범위)
        oy0 = int(y / ds); oy1 = min(int((y + blk_h) / ds), out_h)
        if oy1 <= oy0:
            continue
        for tx in range(0, nx, stride):
            x = tx * tile
            bw = min(tile, W - x)
            blk_w = min(tile * stride, W - x)
            ox0 = int(x / ds); ox1 = min(int((x + blk_w) / ds), out_w)
            if ox1 <= ox0:
                continue
            try:
                reg = np.asarray(s.read_region((x, y), 0, (bw, bh)).convert("RGB"))
                out[oy0:oy1, ox0:ox1] = cv2.resize(reg, (ox1 - ox0, oy1 - oy0),
                                                   interpolation=cv2.INTER_AREA)
                n_read += 1
            except Exception:
                n_bad += 1                          # 손상타일 → 흰색(bg) 유지
    print(f"[trident] 썸네일 sparse-tile {out_w}x{out_h} (stride={stride}, "
          f"읽은타일 {n_read}" + (f", 손상 {n_bad}" if n_bad else "") + ")", flush=True)
    return out


def _otsu_segment(arr, sat_min=0.07, white_thr=220, black_thr=15, min_area_frac=0.0005):
    """썸네일에서 조직 마스크(채도 Otsu). 딥세그 대신 고전적 방법: 손상타일 견디고, deeplabv3 와
    99% 일치(검증). 조직=채도충분 AND 너무밝지않음(유리X) AND 너무어둡지않음(손상blank X)."""
    import cv2
    rgb = arr.astype(np.float32)
    gray = rgb.mean(2)
    mx = rgb.max(2); mn = rgb.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    sat_u8 = np.clip(sat * 255, 0, 255).astype(np.uint8)
    otsu_t, _ = cv2.threshold(sat_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu_t / 255.0, sat_min)
    keep = _otsu_keep(sat, gray, thr, white_thr, black_thr, min_area_frac, arr.shape)
    # 옅은 염색 슬라이드 구제: 배경이 대부분 흰색이면 Otsu 가 임계를 과하게 높게 잡아(예 0.31)
    # 옅은 조직을 통째로 날린다. 결과가 거의 비면 바닥 임계(sat_min)로 재시도해 더 잡히면 채택.
    # (정상 슬라이드는 첫 결과로 끝나 영향 없음)
    if (keep > 0).mean() < 0.003 and thr > sat_min:
        keep2 = _otsu_keep(sat, gray, sat_min, white_thr, black_thr, min_area_frac, arr.shape)
        if (keep2 > 0).mean() > (keep > 0).mean():
            print(f"[trident] 옅은조직 폴백: Otsu thr={thr:.3f}로 조직 0% → "
                  f"바닥 thr={sat_min:.2f}로 재검출 {100*(keep2>0).mean():.2f}%", flush=True)
            keep = keep2

    # 펜마크 제거 (선택, COT_REMOVE_PENMARKS=1): 고채도 *순색*(빨강/파랑/초록 잉크)을 조직마스크서 제외.
    # H&E 조직은 저채도라 안 걸림(검증: 빨간마커 정확히 잡고 조직 안건드림). 기본 off=production 무영향.
    if os.environ.get("COT_REMOVE_PENMARKS", "0") == "1":
        sc = np.sort(rgb, axis=2)                       # 0-255 오름차순
        dom = (sc[..., 2] - sc[..., 1]) / 255.0         # 순색도(최상위채널-둘째) 0-1
        ink = (sat > 0.45) & (dom > 0.25) & (mx < 230)  # 고채도 + 순색 + 너무밝지않음
        ink = cv2.dilate(ink.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        _b = (keep > 0).mean()
        keep[ink > 0] = 0
        print(f"[trident] 펜마크제거: 잉크 {100*(ink>0).mean():.2f}% → 조직 "
              f"{100*_b:.1f}%->{100*(keep>0).mean():.1f}%", flush=True)
    return keep


def _otsu_keep(sat, gray, thr, white_thr, black_thr, min_area_frac, shape):
    """주어진 채도 임계로 조직 마스크 생성(형태학 + 작은덩어리 제거)."""
    import cv2
    tissue = ((sat >= thr) & (gray < white_thr) & (gray > black_thr)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, k, iterations=2)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, k, iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(tissue, 8)
    min_area = min_area_frac * shape[0] * shape[1]
    keep = np.zeros_like(tissue)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return keep


def _mask_to_geojson_file(mask, downsample, slide_name, out_path, min_pts=10):
    """조직 마스크 -> trident 호환 geojson(level-0 Polygon) 파일로 저장."""
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feats = []
    for tid, c in enumerate(c for c in contours if len(c) >= min_pts):
        ring = [[float(x) * downsample, float(y) * downsample] for [[x, y]] in c]
        ring.append(ring[0])
        feats.append({"type": "Feature", "properties": {"tissue_id": tid},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    gj = {"type": "FeatureCollection", "name": slide_name,
          "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
          "features": feats}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(gj, open(out_path, "w"))
    return len(feats)


def extract_features(slide_path, job_dir, gpu=0):
    import time as _t
    # ⏱ deadline 연동: seg+feat 합이 interf1 의 hard kill(COT_SLIDE_DEADLINE) 을 넘지 않도록
    # 안쪽 deadline(= deadline − 모델로드/model/write 여유)을 둔다. feat 예산은 "남은 시간"으로
    # 동적 산정 → seg 가 오래 걸려도 합이 안쪽 deadline 안에 들어가 *부분이라도 진짜 CoT* 를 남긴다.
    # (seg/feat 둘 다 maxout 해도 hard kill 로 결과를 통째로 버리는 일이 없게.)
    _extract_start = _t.time()
    _slide_deadline = float(os.environ.get("COT_SLIDE_DEADLINE", "300"))
    _reserve = float(os.environ.get("COT_DEADLINE_RESERVE", "45"))   # 로드+model+write+margin
    _inner_deadline = max(30.0, _slide_deadline - _reserve)
    path, reader = _prepare_input(slide_path, job_dir)
    print(f"[trident] reader={reader} reader_pref={_READER_PREF} path={path}", flush=True)
    with load_wsi(slide_path=path, reader_type=reader,
                  lazy_init=False, custom_mpp_keys=None, mpp=FORCED_MPP) as slide:
        W, H = slide.dimensions
        save_coords = os.path.join(job_dir, f"{float(MAG):g}x_{PATCH}px_0px_overlap")
        coords_h5 = os.path.join(save_coords, "patches", f"{slide.name}_patches.h5")
        geojson_path = os.path.join(job_dir, "contours_geojson", f"{slide.name}.geojson")

        # === seg: deeplabv3 대신 Otsu 썸네일 seg (빠름/OOM없음/손상내성, deeplabv3와 99% 일치) ===
        # 느린 슬라이드 대책(격자 쓰레기 대신 *진짜* seg): ①피라미드 있으면 낮은 레벨 read(즉시)
        # ②단일층 거대 슬라이드는 time_budget 내 읽은 만큼으로 부분 seg(=면적 축소). _seg_thumbnail 내부 처리.
        _t0 = _t.time()
        # seg 상한(기본 90s)을 둬 feat 가 굶지 않게: seg ≤ min(상한, 남은시간)
        _seg_budget = max(5.0, min(float(os.environ.get("COT_SEG_TIME_BUDGET", "90")),
                                   _inner_deadline - (_t.time() - _extract_start)))
        thumb, real_ds = _seg_thumbnail(path, reader, W, H, time_budget=_seg_budget)
        # YCbCr 오태깅(핑크배경) 슬라이드 자동 색교정 (COT_FIX_YCBCR=1). seg·패치 둘 다 적용.
        fix_color = (os.environ.get("COT_FIX_YCBCR", "0") == "1") and _is_pink_bg(thumb)
        if fix_color:
            thumb = _ycbcr_to_rgb(thumb)
            print("[trident] ⚠️ YCbCr 색교정 적용 (핑크배경 감지 = JPEG 오태깅 슬라이드)", flush=True)
        mask = _otsu_segment(thumb)
        nfeat = _mask_to_geojson_file(mask, real_ds, slide.name, geojson_path)
        _t_seg = _t.time() - _t0
        print(f"[trident-timing] seg(otsu) = {_t_seg:.1f}s | 썸네일 {thumb.shape[1]}x{thumb.shape[0]} "
              f"| 조직 {(mask>0).mean()*100:.1f}% | geojson {nfeat}개", flush=True)

        # === coords: geojson 래스터화로 직접 생성 (trident 느린 coords 우회) ===
        _t0 = _t.time()
        coords = _coords_from_geojson(geojson_path, W, H) if os.path.exists(geojson_path) else np.empty((0, 2), np.int64)
        if len(coords):
            _write_coords_h5(coords_h5, coords, slide, W, H)
        _t_coords = _t.time() - _t0

        # 조각 위치(coords)를 인코딩 전에 손본다: 0개면 격자 채우고, 너무 많으면 상한.
        _ensure_patches(slide, coords_h5)   # 보험①
        _cap_patches(coords_h5)             # 보험②
        with h5py.File(coords_h5, "r") as _f:
            _npatch = _f["coords"].shape[0]
        print(f"[trident-timing] coords = {_t_coords:.1f}s | patches = {_npatch}", flush=True)

        # === H-optimus-1 인코더로 피처 추출 ===
        # 512px 조각을 좌표에서 읽어 224 리사이즈 → H-opt forward → fp32 (학습과 동일 방식).
        _t0 = _t.time()
        # feat 예산 = min(고정 HOPT예산, 안쪽 deadline 까지 남은 시간) → seg 가 오래 걸렸으면 feat 가
        # 그만큼 줄어 합이 deadline 을 안 넘김(부분이라도 진짜 CoT 를 남기고 hard-kill 회피).
        _feat_budget = max(10.0, min(float(os.environ.get("COT_HOPT_TIME_BUDGET", "180")),
                                     _inner_deadline - (_t.time() - _extract_start)))
        feats = _hopt_extract_features(slide, coords_h5, gpu, time_budget=_feat_budget, fix_color=fix_color)
        _t_feat = _t.time() - _t0
        print(f"[trident-timing] feat(hoptimus1) = {_t_feat:.1f}s | TOTAL seg+coords+feat = "
              f"{_t_seg + _t_coords + _t_feat:.1f}s", flush=True)
        with h5py.File(coords_h5, "r") as _f:
            coords = _f["coords"][:]
    return None, feats, coords


if __name__ == "__main__":
    import time
    sp = sys.argv[1]
    job = sys.argv[2] if len(sys.argv) > 2 else "/tmp/preprocess"
    t = time.time()
    h5, feats, coords = extract_features(sp, job)
    print(f"[trident] {feats.shape} feats, {coords.shape} coords in {time.time()-t:.1f}s -> {h5}")
