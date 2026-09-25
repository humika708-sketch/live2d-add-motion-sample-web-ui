#!/usr/bin/env python3
"""build_rig.py で分けた部品を、パーツ分けPSDとして書き出す。

レイヤー名は see-through(一枚絵を自動でパーツ分けする研究)の出力と同じ名前にしてあるので、
次のような道具にそのまま読み込める:
  ・Anime2.5DRig(ブラウザでPSDを自動で動かす道具。https://github.com/852wa/Anime2.5DRig)
  ・Live2D Cubism Editor(手作業できちんとリグを組む場合)
  ・Stretchy Studio / Inochi2D など、PSDを読み込めるリグ作成ツール

閉じ目・開き口は、自作エンジン(runtime/puppet.js)で描いた画像を使う
(models/kurisu/layers/閉じ目.png・開き口.png。作り方は説明.md を参照)。

使い方: python3 export_psd.py [出力先.psd] [切り抜き: 全身 または x0,y0,x1,y1(元画像基準の座標)]
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "rig_config.json"), encoding="utf-8"))
LAYER_DIR = os.path.join(HERE, CFG["output"], "layers")


def load(name):
    return np.array(Image.open(os.path.join(LAYER_DIR, f"{name}.png")).convert("RGBA"))


def has(name):
    return os.path.exists(os.path.join(LAYER_DIR, f"{name}.png"))


def merge(*names):
    """同じ役割の部品(左右の目など)を1枚に重ねる"""
    out = None
    for n in names:
        im = Image.fromarray(load(n))
        out = im if out is None else Image.alpha_composite(out, im)
    return np.array(out)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, CFG["output"], "kurisu_parts.psd")
    crop = sys.argv[2] if len(sys.argv) > 2 else "全身"

    # 白目は「目の形」の内側だけにする(白目の画像は動かしたとき用に少し外側まで塗ってあるため)
    white = merge("白目R", "白目L").astype(np.float32)
    shape = merge("目の形R", "目の形L")[..., 3:4].astype(np.float32) / 255
    white[..., 3:4] *= shape
    white = white.astype(np.uint8)

    # 下から上への重ね順(PSDでは下のレイヤーほど奥)
    stack = [
        ("back hair", load("PSD_後髪")),
        ("topwear", load("体")),
        ("side hair", load("PSD_前の房")),
        ("face", load("顔")),
        ("eyewhite", white),
        ("irides", merge("瞳R", "瞳L")),
        ("eyelash", merge("まつ毛R", "まつ毛L")),
        ("eye_close", load("閉じ目")),
        # 口の差分画像から作った口があればそれを使う。無ければ描き起こしの口
        ("mouth_close", load("口_閉じ") if has("口_閉じ") else load("口の線")),
        # 開き口は、口パクに向く「中開き」を優先する
        ("mouth_open", load("口_中開き") if has("口_中開き") else load("口_あ") if has("口_あ") else load("開き口")),
        ("front hair", load("PSD_前髪")),
    ]
    h, w = stack[0][1].shape[:2]
    if crop != "全身":
        s = w / CFG["baseWidth"]
        x0, y0, x1, y1 = [int(round(float(v) * s)) for v in crop.split(",")]
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    psd = PSDImage.new("RGBA", (x1 - x0, y1 - y0))
    for name, arr in stack:
        im = Image.fromarray(arr[y0:y1, x0:x1], "RGBA")
        box = im.getbbox()
        if not box:
            continue
        layer = PixelLayer.frompil(im.crop(box), psd, name, top=box[1], left=box[0])
        psd.append(layer)
    psd.save(out_path)
    print(f"書き出し: {out_path}({x1 - x0}×{y1 - y0}、レイヤー {len(stack)} 枚)")


if __name__ == "__main__":
    main()
