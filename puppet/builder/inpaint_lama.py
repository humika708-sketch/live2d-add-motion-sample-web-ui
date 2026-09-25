"""画像の穴埋め(塗り足し)を、LaMa という学習済みモデルで行う。

ギットハブで公開されている IOPaint(https://github.com/Sanster/IOPaint)が配布している
アニメ・漫画向けの LaMa(anime-manga-big-lama)を使う。重みの扱いは配布元の条件に従う(このリポジトリには含めない)。
重みは初回にギットハブの配布ページから自動でダウンロードする(約200MB)。

周りの線や陰影をつないで描き足してくれるので、髪の下の白衣のしわや襟の線、
前髪の下の肌などが、単純なぼかし塗りよりずっと自然になる。
"""
import os
import urllib.request

import cv2
import numpy as np
import torch

URL = "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
_model = None


def _load():
    global _model
    if _model is None:
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, os.path.basename(URL))
        if not os.path.exists(path):
            print("塗り足しモデルをダウンロード中:", URL)
            urllib.request.urlretrieve(URL, path)
        torch.set_num_threads(os.cpu_count() or 4)
        _model = torch.jit.load(path, map_location="cpu").eval()
    return _model


@torch.no_grad()
def _run(img, mask):
    """img: 高さ×幅×3(0〜255の整数)、mask: 高さ×幅(真=塗る)。8の倍数に広げて推論する"""
    h, w = mask.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    img_p = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
    mask_p = np.pad(mask, ((0, ph), (0, pw)), mode="symmetric")
    x = torch.from_numpy(img_p.transpose(2, 0, 1)).float()[None] / 255
    m = torch.from_numpy(mask_p.astype(np.float32))[None, None]
    x = x * (1 - m)   # 塗る部分の元の絵を消してから渡す(消さないと元の絵がうっすら残る)
    out = _load()(x, m)[0].permute(1, 2, 0).numpy()
    return np.clip(out * 255, 0, 255)[:h, :w]


def inpaint(rgb, region, context=96, max_side=1024):
    """rgb: 高さ×幅×3(0〜1の実数)。region の画素を塗り足した画像を返す。
    穴のまとまりごとに周りを少し含めて切り出し、大きすぎるときは縮小して処理してから戻す。"""
    out = rgb.copy()
    if not region.any():
        return out
    H, W = region.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(region.astype(np.uint8), np.ones((25, 25), np.uint8)), connectivity=8)
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        x0, y0 = max(x - context, 0), max(y - context, 0)
        x1, y1 = min(x + w + context, W), min(y + h + context, H)
        sub_mask = region[y0:y1, x0:x1] & (lab[y0:y1, x0:x1] == i)
        if not sub_mask.any():
            continue
        sub = (np.clip(rgb[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)
        scale = min(1.0, max_side / max(x1 - x0, y1 - y0))
        if scale < 1:
            sw, sh = int((x1 - x0) * scale), int((y1 - y0) * scale)
            small = cv2.resize(sub, (sw, sh), interpolation=cv2.INTER_AREA)
            small_m = cv2.resize(sub_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
            small_m = cv2.dilate(small_m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            res = _run(small, small_m)
            res = cv2.resize(res, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
        else:
            res = _run(sub, sub_mask)
        patch = out[y0:y1, x0:x1]
        patch[sub_mask] = res[sub_mask] / 255
        out[y0:y1, x0:x1] = patch
    return out
