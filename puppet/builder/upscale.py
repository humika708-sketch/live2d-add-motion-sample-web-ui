#!/usr/bin/env python3
"""アニメ絵向けの超解像(高解像度化)。

ギットハブで公開されている Real-ESRGAN のアニメ絵用モデル
(RealESRGAN_x4plus_anime_6B、BSD-3ライセンス)を使う。
ネットワーク構造(RRDBNet)は公開されている定義をここで再実装しているので、
必要なのは PyTorch と重みファイルだけ。重みは初回に自動でダウンロードする。

透明部分があるPNGにも対応する:
  ・色は、透明部分へ輪郭の色をにじませてから拡大する(縁が黒ずむのを防ぐ)
  ・透明度は、灰色画像として同じモデルで拡大する(輪郭がぼやけない)

使い方: python3 upscale.py 入力.png 出力.png [倍率(既定2)]
"""
import os
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

WEIGHTS_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nb=6, nf=64, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_model():
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, os.path.basename(WEIGHTS_URL))
    if not os.path.exists(path):
        print("重みをダウンロード中:", WEIGHTS_URL)
        urllib.request.urlretrieve(WEIGHTS_URL, path)
    model = RRDBNet()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("params_ema", state))
    return model.eval()


@torch.no_grad()
def run_x4(model, rgb, tile=192, pad=12):
    """rgb: 高さ×幅×3 の 0〜1 の配列 → 4倍の配列。メモリ節約のためタイルに分けて処理する。"""
    h, w, _ = rgb.shape
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float()[None]
    out = torch.zeros(1, 3, h * 4, w * 4)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
            yb, xb = min(y1 + pad, h), min(x1 + pad, w)
            o = model(x[:, :, ya:yb, xa:xb])
            out[:, :, y0 * 4:y1 * 4, x0 * 4:x1 * 4] = \
                o[:, :, (y0 - ya) * 4:(y0 - ya) * 4 + (y1 - y0) * 4, (x0 - xa) * 4:(x0 - xa) * 4 + (x1 - x0) * 4]
        print(f"  {min(y0 + tile, h)}/{h} 行", flush=True)
    return out[0].clamp(0, 1).numpy().transpose(1, 2, 0)


def bleed_colors(rgba):
    """透明部分に輪郭の色をにじませる(拡大時に縁が暗くならないように)"""
    rgb = rgba[..., :3].copy()
    a = rgba[..., 3]
    hole = (a < 8).astype(np.uint8)
    # 遠い透明部分は時間短縮のため縮小して塗ってから戻す
    small = cv2.resize(rgb, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small_hole = cv2.resize(hole, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    small = cv2.inpaint(small, small_hole, 3, cv2.INPAINT_TELEA)
    far = cv2.resize(small, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    near = cv2.inpaint(rgb, hole, 3, cv2.INPAINT_TELEA)
    dist = cv2.distanceTransform(hole, cv2.DIST_L2, 3)
    t = np.clip(dist / 12.0, 0, 1)[..., None]
    return (near * (1 - t) + far * t).astype(np.uint8)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    torch.set_num_threads(os.cpu_count() or 4)
    rgba = np.array(Image.open(src).convert("RGBA"))
    h, w = rgba.shape[:2]
    model = load_model()
    print("色を拡大中")
    rgb4 = run_x4(model, bleed_colors(rgba).astype(np.float32) / 255)
    print("透明度を拡大中")
    a = rgba[..., 3].astype(np.float32) / 255
    a4 = run_x4(model, np.repeat(a[..., None], 3, axis=2)).mean(axis=2)
    out = np.dstack([rgb4, a4])
    size = (round(w * scale), round(h * scale))
    out = cv2.resize((out * 255).round().astype(np.uint8), size, interpolation=cv2.INTER_AREA)
    Image.fromarray(out, "RGBA").save(dst)
    print(f"保存: {dst} ({size[0]}×{size[1]})")


if __name__ == "__main__":
    main()
