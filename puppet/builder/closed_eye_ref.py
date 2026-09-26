"""同じ絵柄・同じ構図で「目だけ閉じた」全身画像から、手描きの閉じ目を取り出す。

手順:
  1. 目と口を除いた特徴点(前髪・輪郭・耳・体)で、閉じ目の画像を元の立ち絵に合わせる(相似変換)
  2. 目のまわりだけを Real-ESRGAN で4倍にしてから、組み立て用の解像度へ写す(線がぼけないように)
  3. 呼び出し側で、目を消して塗り直した肌との「差」から閉じ目の線だけを部品にする

rig_config.json の "閉じ目の差分" で使う。
"""
import os

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")


def _on_white(path):
    im = np.array(Image.open(path).convert("RGBA")).astype(np.float32)
    a = im[..., 3:4] / 255
    return (im[..., :3] * a + 255 * (1 - a)).astype(np.uint8)


def align(base, img, exclude):
    """img を base に合わせる相似変換(2×3)。exclude の範囲(目・口)の特徴点は使わない"""
    h, w = base.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[: int(h * 0.3)] = 255          # 顔と髪のあたり(体は描き直しでずれることがあるので使わない)
    for (x0, y0, x1, y1) in exclude:
        mask[int(y0):int(y1), int(x0):int(x1)] = 0
    sift = cv2.SIFT_create()
    kb, db = sift.detectAndCompute(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY), mask)
    ki, di = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
    pairs = cv2.BFMatcher().knnMatch(di, db, k=2)
    good = [p for p, q in pairs if p.distance < 0.75 * q.distance]
    if len(good) < 8:
        raise RuntimeError("閉じ目の画像の位置合わせに失敗しました(特徴点が足りません)")
    src = np.float32([ki[g.queryIdx].pt for g in good])
    dst = np.float32([kb[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    res = np.linalg.norm(src @ M[:, :2].T + M[:, 2] - dst, axis=1)[inl[:, 0] > 0]
    return M, int(inl.sum()), float(res.mean())


def _upscale_crop(img, box, key):
    """img の box の範囲を4倍にする(結果は .cache に保存して使い回す)"""
    x0, y0, x1, y1 = box
    path = os.path.join(CACHE, f"閉じ目_{key}_{x0}_{y0}_{x1}_{y1}_x4.png")
    if os.path.exists(path):
        return np.array(Image.open(path).convert("RGB"))
    import torch
    import upscale
    torch.set_num_threads(os.cpu_count() or 4)
    crop = img[y0:y1, x0:x1].astype(np.float32) / 255
    out = (upscale.run_x4(upscale.load_model(), crop) * 255).round().astype(np.uint8)
    os.makedirs(CACHE, exist_ok=True)
    Image.fromarray(out).save(path)
    return out


def build(cfg, base_dir, H, W, S):
    """閉じ目の画像を、組み立て用の解像度(H×W、元画像の S 倍)に合わせた RGB(0〜1)にして返す。
    戻り値: (画像, 有効な範囲の真偽配列, 情報)"""
    cc = cfg["閉じ目の差分"]
    base = _on_white(os.path.join(base_dir, cfg["fallbackSource"]))
    img = np.array(Image.open(os.path.join(base_dir, cc["ファイル"])).convert("RGB"))
    exclude = [cfg["目"][k]["範囲"] for k in ("R", "L")]
    exclude = [(x0 - 6, y0 - 8, x1 + 6, y1 + 10) for (x0, y0, x1, y1) in exclude]
    m = cfg["口"]["消す範囲"]
    exclude.append((m[0] - 4, m[1] - 4, m[2] + 4, m[3] + 6))
    M, n, res = align(base, img, exclude)

    # 目のまわり(元画像の座標)を、閉じ目の画像の座標に戻して切り出す範囲を決める
    xs = [v for k in ("R", "L") for v in (cfg["目"][k]["範囲"][0], cfg["目"][k]["範囲"][2])]
    ys = [v for k in ("R", "L") for v in (cfg["目"][k]["範囲"][1], cfg["目"][k]["範囲"][3])]
    Mi = cv2.invertAffineTransform(M)
    corners = np.array([[min(xs) - 20, min(ys) - 20], [max(xs) + 20, max(ys) + 20]], np.float32)
    c = corners @ Mi[:, :2].T + Mi[:, 2]
    box = (int(max(c[:, 0].min(), 0)), int(max(c[:, 1].min(), 0)),
           int(min(c[:, 0].max() + 1, img.shape[1])), int(min(c[:, 1].max() + 1, img.shape[0])))
    up = _upscale_crop(img, box, os.path.splitext(os.path.basename(cc["ファイル"]))[0])

    # 4倍の切り出し → 閉じ目の画像 → 元の立ち絵 → 組み立て用の解像度 をひとつの変換にまとめる
    # 画素の中心の位置も合わせる(4倍の画素 u の中心は、元の u/4 - 0.375。S倍では S*x + (S-1)/2)
    A = np.array([[0.25, 0, box[0] - 0.375], [0, 0.25, box[1] - 0.375], [0, 0, 1]], float)
    B = np.vstack([M, [0, 0, 1]])
    C = np.array([[S, 0, (S - 1) / 2], [0, S, (S - 1) / 2], [0, 0, 1]], float)
    T = (C @ B @ A)[:2]
    f = float(np.hypot(T[0, 0], T[1, 0]))   # 縮める倍率
    src = up.astype(np.float32)
    if f < 1:   # 縮めるときは、先に少しぼかしてギザギザ(折り返し)を防ぐ
        src = cv2.GaussianBlur(src, (0, 0), 0.5 * (1 / f - 1) + 1e-3)
    out = cv2.warpAffine(src, T, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    valid = cv2.warpAffine(np.ones(up.shape[:2], np.uint8), T, (W, H), flags=cv2.INTER_NEAREST) > 0
    info = {"scale": float(np.hypot(M[0, 0], M[1, 0])), "points": n, "residual": res}
    return np.clip(out / 255, 0, 1).astype(np.float32), valid, info
