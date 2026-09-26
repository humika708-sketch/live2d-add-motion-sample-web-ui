#!/usr/bin/env python3
"""三面図(横に人物が並んだ透明背景の画像)から人物を1人ずつ切り出し、
正面の立ち絵と同じ大きさ・同じ頭の高さで、同じ画面の大きさに置き直す。

・人物は、透明でない列のかたまりで見分ける(左から順に 名前1, 名前2, ... として保存)
・大きさは「頭の上端から足先までの高さ」を正面の立ち絵に合わせ、頭の中心を正面の頭の中心(設定の 頭.中心)に置く
・--高解像度化 を付けると、切り出した人物を Real-ESRGAN で拡大してから置き、2倍の画面(_x2)も作る

使い方:
  python3 cut_turnaround.py 三面図.png 正面の立ち絵.png 出力フォルダ 名前1,名前2,... [--高解像度化]
  例: python3 cut_turnaround.py kurisu_turnaround.png kurisu_front.png views 横左,後ろ,横右 --高解像度化
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))


def figures(alpha, min_width=20):
    """透明でない列のかたまり → [(x0, x1), ...](左から)"""
    lab, n = ndimage.label((alpha > 20).any(axis=0))
    out = []
    for i in range(1, n + 1):
        xs = np.where(lab == i)[0]
        if len(xs) >= min_width:
            out.append((int(xs.min()), int(xs.max())))
    return out


def extent(alpha):
    """頭の上端・足先・頭の中心x(上から130画素ぶんの横の範囲の中央)"""
    ys = np.where((alpha > 20).any(axis=1))[0]
    top, bottom = int(ys.min()), int(ys.max())
    head = alpha[top:top + max(20, (bottom - top) // 8)]
    xs = np.where((head > 20).any(axis=0))[0]
    return top, bottom, (xs.min() + xs.max()) / 2


def main():
    src, front, out_dir, names = sys.argv[1:5]
    upscale = "--高解像度化" in sys.argv
    names = names.split(",")
    cfg = json.load(open(os.path.join(HERE, "rig_config.json"), encoding="utf-8"))
    head_cx = cfg["頭"]["中心"][0]
    fa = np.array(Image.open(front).convert("RGBA"))[..., 3]
    f_top, f_bottom, _ = extent(fa)
    W, H = fa.shape[1], fa.shape[0]
    im = Image.open(src).convert("RGBA")
    figs = figures(np.array(im)[..., 3])
    if len(figs) != len(names):
        raise SystemExit(f"人物の数({len(figs)})と名前の数({len(names)})が合いません")
    os.makedirs(out_dir, exist_ok=True)
    for (x0, x1), name in zip(figs, names):
        crop = im.crop((max(x0 - 6, 0), 0, min(x1 + 7, im.width), im.height))
        a = np.array(crop)[..., 3]
        top, bottom, cx = extent(a)
        k = (f_bottom - f_top) / (bottom - top)
        crop_path = os.path.join(out_dir, f"_切り出し_{name}.png")
        crop.save(crop_path)
        versions = [(1, crop)]
        if upscale:
            import subprocess
            up_path = os.path.join(out_dir, f"_切り出し_{name}_拡大.png")
            subprocess.run([sys.executable, os.path.join(HERE, "upscale.py"), crop_path, up_path, str(2 * k)], check=True)
            versions.append((2, Image.open(up_path).convert("RGBA")))
        for scale, img in versions:
            size = (round(crop.width * k * scale), round(crop.height * k * scale))
            if img.size != size:
                img = img.resize(size, Image.LANCZOS)
            canvas = Image.new("RGBA", (W * scale, H * scale), (0, 0, 0, 0))
            ox, oy = round((head_cx - cx * k) * scale), round((f_top - top * k) * scale)
            canvas.alpha_composite(img, (ox, oy))
            path = os.path.join(out_dir, f"kurisu_{name}{'_x2' if scale == 2 else ''}.png")
            canvas.save(path)
            print(f"保存: {path}(倍率 {k * scale:.3f}、位置 {ox},{oy})")


if __name__ == "__main__":
    main()
