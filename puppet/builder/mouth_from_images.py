"""口の差分画像(同じ絵柄で口だけ描き分けた拡大図)から、口の部品を作る。

手順:
  1. 差分どうしを「口より上(鼻・耳・横髪)」の特徴点で基準の差分に合わせる(相似変換)
  2. 基準の差分を、鼻先〜あご先の長さで全身画像の大きさに合わせ、口の線の中心どうしを重ねる
  3. 肌の色を全身画像に合わせてから、肌との色の差で「口だけ」を切り出す
     (見た目が肌の上で元と同じになるよう、色を逆算して半透明にする)
  4. 拡大しても粗くならないよう、全身画像の T 倍の解像度で持つ

rig_config.json の "口の差分" で使う。座標は、差分画像はその画素、全身画像は元画像(baseWidth)基準。
"""
import os

import cv2
import numpy as np
from PIL import Image


def _load(path):
    return np.array(Image.open(path).convert("RGB"))


def _align_to_base(base, img):
    """img を base に合わせる相似変換(2×3)。口より上と横の髪だけを使う"""
    h, w = base.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[: int(h * 0.235)] = 255
    mask[int(h * 0.235): int(h * 0.365), : int(w * 0.15)] = 255
    mask[int(h * 0.235): int(h * 0.365), int(w * 0.85):] = 255
    sift = cv2.SIFT_create(nfeatures=6000)
    kb, db = sift.detectAndCompute(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY), mask)
    ki, di = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), mask)
    pairs = cv2.BFMatcher().knnMatch(di, db, k=2)
    good = [p for p, q in pairs if p.distance < 0.7 * q.distance]
    if len(good) < 6:
        raise RuntimeError("口の差分の位置合わせに失敗しました(特徴点が足りません)")
    src = np.float32([ki[g.queryIdx].pt for g in good])
    dst = np.float32([kb[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
    return M, int(inl.sum())


def build(cfg, base_dir, full_rgb, S, T=3):
    """full_rgb: 全身画像(白背景に合成、0〜1)。S: 元画像→全身画像の倍率。
    戻り値: 名前 → dict(hi=高解像度RGBA(0〜1), origin=(x0,y0), T=倍率, full=全身サイズRGBA)"""
    mc = cfg["口の差分"]
    folder = os.path.join(base_dir, mc["フォルダ"])
    base_img = _load(os.path.join(folder, mc["基準"]))
    H, W = full_rgb.shape[:2]

    # 大きさ: 鼻先〜あご先の長さの比
    (cn, cc), (fn, fc) = mc["差分の鼻先とあご先"], mc["全身の鼻先とあご先"]
    d_close = np.hypot(cc[0] - cn[0], cc[1] - cn[1])
    d_full = np.hypot(fc[0] - fn[0], fc[1] - fn[1]) * S
    s = d_full / d_close
    mcx_c, mcy_c = mc["差分の口の線の中心"]
    mcx_f, mcy_f = [v * S for v in mc["全身の口の線の中心"]]
    to_full = np.array([[s, 0, mcx_f - s * mcx_c], [0, s, mcy_f - s * mcy_c]])

    r = mc["範囲"]
    rx0, ry0 = int(round(mcx_f + r[0] * S)), int(round(mcy_f + r[1] * S))
    rx1, ry1 = int(round(mcx_f + r[2] * S)), int(round(mcy_f + r[3] * S))
    rw, rh = rx1 - rx0, ry1 - ry0
    to_hi = np.array([[T, 0, -T * rx0], [0, T, -T * ry0], [0, 0, 1]])

    # 肌の色: 全身画像の口のまわり(明るい画素の中央値)
    ring = full_rgb[ry0:ry1, rx0:rx1].reshape(-1, 3)
    skin_full = np.median(ring[ring.min(axis=1) > 0.75], axis=0)
    ringc = base_img[int(mcy_c - 90): int(mcy_c + 110), int(mcx_c - 190): int(mcx_c + 190)].reshape(-1, 3) / 255.0
    skin_close = np.median(ringc[ringc.min(axis=1) > 0.78], axis=0)
    gain = skin_full / skin_close

    # 切り出し範囲(楕円)。縁はなめらかに0へ
    yy, xx = np.mgrid[0:rh * T, 0:rw * T].astype(np.float32)
    ex, ey = (xx / T + rx0 - mcx_f) / (max(-r[0], r[2]) * S), (yy / T + ry0 - (ry0 + ry1) / 2) / ((r[3] - r[1]) * S / 2)
    rr = np.sqrt(ex ** 2 + ey ** 2)
    ell = np.clip((1.0 - rr) / 0.18, 0, 1)

    out = {}
    for name, fname in mc["形"].items():
        img = _load(os.path.join(folder, fname))
        if fname == mc["基準"]:
            M = np.array([[1, 0, 0], [0, 1, 0]], float)
        else:
            M, n = _align_to_base(base_img, img)
            print(f"  口「{name}」を基準に合わせた(対応点 {n})")
        A = to_hi @ np.vstack([to_full, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]])
        sigma = 0.45 / (T * s)
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        hi = cv2.warpAffine(blurred, A[:2], (rw * T, rh * T), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.0
        hi = np.clip(hi * gain, 0, 1)
        # 肌との色の差から不透明度を決め、肌の上で元の見た目になる色を逆算する
        diff = np.abs(hi - skin_full).max(axis=2)
        alpha = np.clip((diff - 0.025) / 0.16, 0, 1) * ell
        col = skin_full + (hi - skin_full) / np.maximum(alpha, 0.02)[..., None]
        col = np.where(alpha[..., None] > 0.02, np.clip(col, 0, 1), skin_full)
        rgba_hi = np.dstack([col, alpha]).astype(np.float32)
        small = cv2.resize(rgba_hi * np.dstack([alpha] * 3 + [np.ones_like(alpha)]), (rw, rh), interpolation=cv2.INTER_AREA)
        a_s = small[..., 3]
        c_s = small[..., :3] / np.maximum(a_s, 1e-4)[..., None]
        full = np.zeros((H, W, 4), np.float32)
        full[ry0:ry1, rx0:rx1] = np.dstack([np.clip(c_s, 0, 1), a_s])
        out[name] = dict(hi=rgba_hi, origin=(rx0, ry0), T=T, full=full, rect=(rx0, ry0, rx1, ry1))
    info = dict(scale=s, center=(mcx_f, mcy_f), gain=gain, skin=skin_full)
    return out, info
