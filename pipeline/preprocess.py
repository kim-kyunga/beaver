import os
import sys
import json

TRIDENT_PATH = os.environ.get("TRIDENT_PATH", "/trident")
if TRIDENT_PATH not in sys.path:
    sys.path.insert(0, TRIDENT_PATH)

import h5py
import numpy as np

from trident import load_wsi

import trident.wsi_objects.WSI as _wsi_mod
_wsi_mod.get_num_workers = lambda *a, **k: 0

import openslide as _osl
_real_openslide = _osl.OpenSlide
_READER_PREF = os.environ.get("COT_READER", "tiffslide").lower()
def _openslide_or_tiffslide(path, *a, **k):
    if _READER_PREF == "tiffslide":
        try:
            import tiffslide
            return tiffslide.TiffSlide(path)
        except Exception as e:
            print(f"[trident] tiffslide 실패 → openslide 로 재시도: {path} ({e})")
            return _real_openslide(path, *a, **k)
    try:
        return _real_openslide(path, *a, **k)
    except Exception:
        import tiffslide
        print(f"[trident] openslide 실패 → tiffslide 로 재시도: {path}")
        return tiffslide.TiffSlide(path)
_osl.OpenSlide = _openslide_or_tiffslide

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
                    pass
        return Image.fromarray(out)
_ts.TiffSlide.read_region = _tolerant_read_region

_NATIVE_MAX_PIXELS = int(os.environ.get("COT_NATIVE_MAX_PIXELS", str(3_500_000_000)))
_SINGLE_NATIVE_MAX = int(os.environ.get("COT_SINGLE_NATIVE_MAX", str(300_000_000)))
_SEG_MAX_TILES = int(os.environ.get("COT_SEG_MAX_TILES", "6000"))

FORCED_MPP = 0.5
MAG = 20
PATCH = 512

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
    a = np.asarray(arr, np.float32)
    Y, Cb, Cr = a[..., 0], a[..., 1], a[..., 2]
    R = Y + 1.402 * (Cr - 128.0)
    G = Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    B = Y + 1.772 * (Cb - 128.0)
    return np.clip(np.stack([R, G, B], -1), 0, 255).astype(np.uint8)


def _is_pink_bg(arr):
    a = np.asarray(arr, np.float32).reshape(-1, 3)
    lum = a.mean(1)
    bright = a[lum > np.percentile(lum, 80)]
    if len(bright) == 0:
        return False
    R, G, B = bright[:, 0].mean(), bright[:, 1].mean(), bright[:, 2].mean()
    return (R - max(G, B) > 25.0) and (R > 170.0)


def _hopt_extract_features(slide, coords_h5, gpu, batch=None, time_budget=None, fix_color=False):
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
            if fix_color:
                pa = _ycbcr_to_rgb(pa).astype(np.float32)
            imgs.append(((pa / 255.0 - _HOPT_MEAN) / _HOPT_STD).transpose(2, 0, 1))
        except Exception:
            imgs.append(blank)
    if not imgs:
        imgs = [blank]
    out = []
    with torch.autocast("cuda", dtype=torch.float16), torch.inference_mode():
        for i in range(0, len(imgs), batch):
            X = torch.tensor(np.stack(imgs[i:i + batch]), dtype=torch.float16).to(f"cuda:{gpu}")
            out.append(m(X).float().cpu().numpy())
            del X
    return np.concatenate(out).astype(np.float32)

OPENSLIDE_EXTS = {".tif", ".tiff", ".svs", ".ndpi", ".scn", ".mrxs", ".vms", ".vmu", ".bif", ".svslide"}
ITK_EXTS = {".mha", ".mhd", ".nii", ".nrrd"}


def _prepare_input(slide_path, job_dir):
    name = str(slide_path).lower()
    ext = os.path.splitext(name)[1]
    if ext in OPENSLIDE_EXTS:
        return str(slide_path), "openslide"
    if name.endswith((".mha", ".mhd", ".nii", ".nii.gz", ".nrrd")):
        import SimpleITK as sitk
        from PIL import Image
        arr = np.squeeze(np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(slide_path)))))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = np.moveaxis(arr, 0, -1)
        arr = arr[..., :3]
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32); a -= a.min()
            mx = a.max(); a = a / mx if mx > 0 else a
            arr = (a * 255).astype(np.uint8)
        os.makedirs(job_dir, exist_ok=True)
        png = os.path.join(job_dir, "converted_input.png")
        Image.fromarray(arr).save(png)
        return png, "image"
    return str(slide_path), "image"


PATCH_CAP = int(os.environ.get("COT_PATCH_CAP", "2000"))
RAST_DS = 64
MIN_FRAC = 0.10


def _ensure_patches(slide, coords_h5):
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
    os.makedirs(os.path.dirname(coords_h5), exist_ok=True)
    with h5py.File(coords_h5, "w") as f:
        d = f.create_dataset("coords", data=np.asarray(coords, dtype=np.int64))
        for k, v in {"patch_size": PATCH, "patch_size_level0": PATCH,
                     "target_magnification": MAG, "level0_magnification": MAG,
                     "overlap": 0, "name": slide.name,
                     "level0_width": W, "level0_height": H}.items():
            d.attrs[k] = v


def _coords_from_geojson(geojson_path, W, H):
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


SEG_MPP = float(os.environ.get("COT_SEG_MPP", "8.0"))


def _seg_thumbnail(path, reader, W, H, time_budget=None):
    import cv2
    from PIL import Image
    ds = max(1.0, SEG_MPP / FORCED_MPP)
    out_w = max(1, int(W / ds)); out_h = max(1, int(H / ds))
    if reader == "image":
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
    _nlev = getattr(s, "level_count", 1)
    if _nlev > 1:
        try:
            thumb = np.asarray(s.get_thumbnail((out_w, out_h)).convert("RGB"))
            s.close()
            print(f"[trident] 썸네일 pyramid {thumb.shape[1]}x{thumb.shape[0]} (levels={_nlev})", flush=True)
            return thumb, W / thumb.shape[1]
        except Exception as e:
            print(f"[trident] pyramid get_thumbnail 실패({e}) → 폴백", flush=True)
    if is_tiled and W * H <= _SINGLE_NATIVE_MAX:
        try:
            thumb = np.asarray(s.get_thumbnail((out_w, out_h)).convert("RGB"))
            s.close()
            print(f"[trident] 썸네일 native(tiled) {thumb.shape[1]}x{thumb.shape[0]}", flush=True)
            return thumb, W / thumb.shape[1]
        except Exception as e:
            print(f"[trident] native get_thumbnail 실패({e}) → tile-tolerant 폴백", flush=True)
            thumb = _read_thumb_tiletolerant(s, W, H, ds, out_w, out_h)
            s.close()
            return thumb, W / thumb.shape[1]
    if is_tiled:
        thumb = _read_thumb_sparse(s, W, H, ds, out_w, out_h, time_budget=time_budget)
        s.close()
        return thumb, W / thumb.shape[1]
    import time as _bt
    out = np.full((out_h, out_w, 3), 255, np.uint8)
    band = 256
    y = 0
    bad_bands = 0
    _b0 = _bt.time()
    while y < H:
        if time_budget is not None and _bt.time() - _b0 > time_budget:
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
        bh = min(tile, H - y)
        blk_h = min(tile * stride, H - y)
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
                n_bad += 1
    print(f"[trident] 썸네일 sparse-tile {out_w}x{out_h} (stride={stride}, "
          f"읽은타일 {n_read}" + (f", 손상 {n_bad}" if n_bad else "") + ")", flush=True)
    return out


def _otsu_segment(arr, sat_min=0.07, white_thr=220, black_thr=15, min_area_frac=0.0005):
    import cv2
    rgb = arr.astype(np.float32)
    gray = rgb.mean(2)
    mx = rgb.max(2); mn = rgb.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    sat_u8 = np.clip(sat * 255, 0, 255).astype(np.uint8)
    otsu_t, _ = cv2.threshold(sat_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu_t / 255.0, sat_min)
    keep = _otsu_keep(sat, gray, thr, white_thr, black_thr, min_area_frac, arr.shape)
    if (keep > 0).mean() < 0.003 and thr > sat_min:
        keep2 = _otsu_keep(sat, gray, sat_min, white_thr, black_thr, min_area_frac, arr.shape)
        if (keep2 > 0).mean() > (keep > 0).mean():
            print(f"[trident] 옅은조직 폴백: Otsu thr={thr:.3f}로 조직 0% → "
                  f"바닥 thr={sat_min:.2f}로 재검출 {100*(keep2>0).mean():.2f}%", flush=True)
            keep = keep2

    if os.environ.get("COT_REMOVE_PENMARKS", "0") == "1":
        sc = np.sort(rgb, axis=2)
        dom = (sc[..., 2] - sc[..., 1]) / 255.0
        ink = (sat > 0.45) & (dom > 0.25) & (mx < 230)
        ink = cv2.dilate(ink.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        _b = (keep > 0).mean()
        keep[ink > 0] = 0
        print(f"[trident] 펜마크제거: 잉크 {100*(ink>0).mean():.2f}% → 조직 "
              f"{100*_b:.1f}%->{100*(keep>0).mean():.1f}%", flush=True)
    return keep


def _otsu_keep(sat, gray, thr, white_thr, black_thr, min_area_frac, shape):
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
    _extract_start = _t.time()
    _slide_deadline = float(os.environ.get("COT_SLIDE_DEADLINE", "300"))
    _reserve = float(os.environ.get("COT_DEADLINE_RESERVE", "45"))
    _inner_deadline = max(30.0, _slide_deadline - _reserve)
    path, reader = _prepare_input(slide_path, job_dir)
    print(f"[trident] reader={reader} reader_pref={_READER_PREF} path={path}", flush=True)
    with load_wsi(slide_path=path, reader_type=reader,
                  lazy_init=False, custom_mpp_keys=None, mpp=FORCED_MPP) as slide:
        W, H = slide.dimensions
        save_coords = os.path.join(job_dir, f"{float(MAG):g}x_{PATCH}px_0px_overlap")
        coords_h5 = os.path.join(save_coords, "patches", f"{slide.name}_patches.h5")
        geojson_path = os.path.join(job_dir, "contours_geojson", f"{slide.name}.geojson")

        _t0 = _t.time()
        _seg_budget = max(5.0, min(float(os.environ.get("COT_SEG_TIME_BUDGET", "90")),
                                   _inner_deadline - (_t.time() - _extract_start)))
        thumb, real_ds = _seg_thumbnail(path, reader, W, H, time_budget=_seg_budget)
        fix_color = (os.environ.get("COT_FIX_YCBCR", "0") == "1") and _is_pink_bg(thumb)
        if fix_color:
            thumb = _ycbcr_to_rgb(thumb)
            print("[trident] ⚠️ YCbCr 색교정 적용 (핑크배경 감지 = JPEG 오태깅 슬라이드)", flush=True)
        mask = _otsu_segment(thumb)
        nfeat = _mask_to_geojson_file(mask, real_ds, slide.name, geojson_path)
        _t_seg = _t.time() - _t0
        print(f"[trident-timing] seg(otsu) = {_t_seg:.1f}s | 썸네일 {thumb.shape[1]}x{thumb.shape[0]} "
              f"| 조직 {(mask>0).mean()*100:.1f}% | geojson {nfeat}개", flush=True)

        _t0 = _t.time()
        coords = _coords_from_geojson(geojson_path, W, H) if os.path.exists(geojson_path) else np.empty((0, 2), np.int64)
        if len(coords):
            _write_coords_h5(coords_h5, coords, slide, W, H)
        _t_coords = _t.time() - _t0

        _ensure_patches(slide, coords_h5)
        _cap_patches(coords_h5)
        with h5py.File(coords_h5, "r") as _f:
            _npatch = _f["coords"].shape[0]
        print(f"[trident-timing] coords = {_t_coords:.1f}s | patches = {_npatch}", flush=True)

        _t0 = _t.time()
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
