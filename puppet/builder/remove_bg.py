#!/usr/bin/env python3
"""白い背景の立ち絵から、背景を透明にする(アニメ絵向けの切り抜きモデル isnet-anime を使う)。

モデル: https://github.com/SkyTNT/anime-segmentation(rembg が配布している isnet-anime.onnx)
初回に .cache へダウンロードする。

使い方: python3 remove_bg.py 入力.png 出力.png
      python3 remove_bg.py 切り抜き済み.png 出力.png すき間だけ   (白いすき間の処理だけ)
"""
import os
import sys
import urllib.request

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx"


def load_session():
    import onnxruntime as ort
    path = os.path.join(CACHE, "isnet-anime.onnx")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        print("切り抜きモデルをダウンロード中:", URL)
        urllib.request.urlretrieve(URL, path)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def alpha_of(rgb, sess=None):
    """rgb: 高さ×幅×3(0〜255)→ 透明度(0〜1)"""
    sess = sess or load_session()
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - 0.5) / 1.0
    x = x.transpose(2, 0, 1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    out = (out - out.min()) / max(out.max() - out.min(), 1e-6)
    a = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    # 中間の値を締める(縁は1〜2画素のなめらかさだけ残す)
    return np.clip((a - 0.15) / 0.7, 0, 1)


def unmix_white(rgb, a):
    """縁の半透明の画素から、背景の白を差し引いた色にする(白いふちが出ないように)"""
    c = rgb.astype(np.float32) / 255
    aa = np.maximum(a, 1e-3)[..., None]
    fg = (c - (1 - a[..., None])) / aa
    fg = np.where(a[..., None] > 0.02, np.clip(fg, 0, 1), c)
    return fg


def clear_white_gaps(rgba, top_ratio=0.2, thr=228):
    """頭のあたり(上から top_ratio の範囲)で、外の背景とつながっている白いすき間を透明にする。
    髪の毛のすき間から見えていた白い背景が、切り抜きで不透明のまま残ることがあるため。
    rgba: 高さ×幅×4(0〜255)。白衣は頭より下にあるので範囲に入らない"""
    from scipy import ndimage
    out = rgba.copy()
    a = out[..., 3].astype(np.float32) / 255
    comp = out[..., :3].astype(np.float32) * a[..., None] + 255 * (1 - a[..., None])   # 白い背景に重ねた見た目
    whiteish = comp.min(axis=2) > thr
    passable = whiteish | (a < 0.05)
    passable[int(out.shape[0] * top_ratio):] = False
    lab, n = ndimage.label(passable)
    outside = np.unique(lab[(a < 0.02) & passable])
    outside = outside[outside > 0]
    remove = np.isin(lab, outside) & (a > 0)
    out[remove, 3] = 0
    return out, int(remove.sum())


def fill_enclosed(rgba, thr=235):
    """輪郭線で囲まれているのに半透明になった所(背景に近い白い服など)を不透明に戻す。
    囲まれた所のうち、切り抜きモデルがある程度不透明とみなした所(平均0.2より上)だけ。
    本当の背景のすき間(腕と体の間など)はモデルがほぼ0にするので、そのまま残る"""
    from scipy import ndimage
    out = rgba.copy()
    a = out[..., 3].astype(np.float32) / 255
    comp = out[..., :3].astype(np.float32) * a[..., None] + 255 * (1 - a[..., None])
    wall = (a > 0.9) | ((comp.min(axis=2) < thr) & (a > 0.3))
    holes = ndimage.binary_fill_holes(wall) & ~wall
    lab, n = ndimage.label(holes)
    if n == 0:
        return out, 0
    mean = ndimage.mean(a, lab, np.arange(1, n + 1))
    fill = np.isin(lab, np.where(mean > 0.2)[0] + 1)
    out[fill, 3] = 255
    return out, int(fill.sum())


def main():
    src, dst = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == "すき間だけ":
        # 切り抜き済みの画像(高解像度化した後など)に、白いすき間の処理だけをかける
        rgba = np.array(Image.open(src).convert("RGBA"))
        out, n = clear_white_gaps(rgba)
        Image.fromarray(out, "RGBA").save(dst)
        print(f"保存: {dst}(白いすき間 {n} 画素を透明に)")
        return
    rgb = np.array(Image.open(src).convert("RGB"))
    a = alpha_of(rgb)
    fg = unmix_white(rgb, a)
    out = (np.dstack([fg, a]) * 255).round().astype(np.uint8)
    out, m = fill_enclosed(out)
    out, n = clear_white_gaps(out)
    Image.fromarray(out, "RGBA").save(dst)
    print(f"保存: {dst}(不透明な画素 {(a > 0.5).mean() * 100:.1f}%、囲まれた半透明 {m} 画素を不透明に、白いすき間 {n} 画素を透明に)")


if __name__ == "__main__":
    main()
