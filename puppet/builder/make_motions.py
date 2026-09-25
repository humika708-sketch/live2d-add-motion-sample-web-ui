#!/usr/bin/env python3
"""パペット用のモーションを作る(ライブ2Dと同じ motion3.json 形式)。

カーブの作り方はこのリポジトリの tools/gen_motions.py の curve() / motion() をそのまま使うので、
本家ライブ2Dモデル向けと同じ書き方でモーションを追加できる。
パラメータ名もライブ2Dの標準名(ParamAngleX など)に合わせてあるため、
ここで作ったモーションは標準的なライブ2Dモデルでも、ほぼそのまま再生できる。

使い方: python3 make_motions.py [出力先(既定: ../models/kurisu/motions)]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
from gen_motions import curve, motion  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "models", "kurisu", "motions")


def define():
    M = {}

    # ---------------------------------------------------------------- 微笑み
    D = 3.0
    M["smile"] = motion(D, [
        curve("ParamEyeLOpen", [(0, 1), (0.35, 0), (2.3, 0), (2.7, 1), (D, 1)]),
        curve("ParamEyeROpen", [(0, 1), (0.35, 0), (2.3, 0), (2.7, 1), (D, 1)]),
        curve("ParamEyeLSmile", [(0, 0), (0.35, 1), (2.3, 1), (2.7, 0), (D, 0)]),
        curve("ParamEyeRSmile", [(0, 0), (0.35, 1), (2.3, 1), (2.7, 0), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (0.35, 1), (2.3, 1), (2.8, 0), (D, 0)]),
        curve("ParamMouthOpenY", [(0, 0), (0.4, 0.35), (1.2, 0.2), (2.2, 0.25), (2.7, 0), (D, 0)]),
        curve("ParamCheek", [(0, 0), (0.5, 0.6), (2.3, 0.6), (2.9, 0), (D, 0)]),
        curve("ParamAngleZ", [(0, 0), (0.5, 8), (2.3, 8), (2.9, 0), (D, 0)]),
        curve("ParamAngleY", [(0, 0), (0.4, 5), (2.3, 3), (2.9, 0), (D, 0)]),
    ])

    # ---------------------------------------------------------------- うなずき
    D = 2.0
    M["nod"] = motion(D, [
        curve("ParamAngleY", [(0, 0), (0.3, -24), (0.6, -3), (0.9, -18), (1.3, 0), (D, 0)]),
        curve("ParamBodyAngleX", [(0, 0), (0.35, 0), (D, 0)]),
        curve("ParamEyeLOpen", [(0, 1), (0.3, 0.6), (0.6, 0.9), (0.9, 0.65), (1.3, 1), (D, 1)]),
        curve("ParamEyeROpen", [(0, 1), (0.3, 0.6), (0.6, 0.9), (0.9, 0.65), (1.3, 1), (D, 1)]),
        curve("ParamMouthForm", [(0, 0), (0.3, 0.4), (1.3, 0.4), (1.8, 0), (D, 0)]),
    ])

    # ---------------------------------------------------------------- 首かしげ
    D = 3.0
    M["tilt"] = motion(D, [
        curve("ParamAngleZ", [(0, 0), (0.6, -22), (2.2, -22), (2.8, 0), (D, 0)]),
        curve("ParamAngleX", [(0, 0), (0.6, 8), (2.2, 8), (2.8, 0), (D, 0)]),
        curve("ParamBodyAngleZ", [(0, 0), (0.7, -4), (2.2, -4), (2.8, 0), (D, 0)]),
        curve("ParamEyeBallX", [(0, 0), (0.6, 0.5), (2.2, 0.5), (2.8, 0), (D, 0)]),
        curve("ParamEyeBallY", [(0, 0), (0.6, 0.3), (2.2, 0.3), (2.8, 0), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (0.6, -0.3), (2.2, -0.3), (2.8, 0), (D, 0)]),
        curve("ParamEyeLOpen", [(0, 1), (1.3, 1), (1.4, 0), (1.5, 1), (D, 1)]),
        curve("ParamEyeROpen", [(0, 1), (1.3, 1), (1.4, 0), (1.5, 1), (D, 1)]),
    ])

    # ---------------------------------------------------------------- 照れ隠し
    D = 4.0
    M["shy"] = motion(D, [
        curve("ParamCheek", [(0, 0), (0.5, 1), (3.3, 1), (D, 0)]),
        curve("ParamAngleX", [(0, 0), (0.6, 22), (3.3, 20), (D, 0)]),
        curve("ParamAngleY", [(0, 0), (0.6, -10), (3.3, -10), (D, 0)]),
        curve("ParamAngleZ", [(0, 0), (0.7, 6), (3.3, 6), (D, 0)]),
        curve("ParamBodyAngleX", [(0, 0), (0.7, 4), (3.3, 4), (D, 0)]),
        curve("ParamEyeBallX", [(0, 0), (0.4, 0.9), (1.8, 0.9), (2.1, -0.3), (2.4, 0.9), (3.3, 0.9), (3.8, 0), (D, 0)]),
        curve("ParamEyeBallY", [(0, 0), (0.4, -0.5), (3.3, -0.5), (3.8, 0), (D, 0)]),
        curve("ParamEyeLOpen", [(0, 1), (0.4, 0.75), (1.2, 0.75), (1.3, 0), (1.4, 0.75), (3.3, 0.75), (3.8, 1), (D, 1)]),
        curve("ParamEyeROpen", [(0, 1), (0.4, 0.75), (1.2, 0.75), (1.3, 0), (1.4, 0.75), (3.3, 0.75), (3.8, 1), (D, 1)]),
        curve("ParamMouthForm", [(0, 0), (0.4, -0.7), (3.3, -0.7), (3.8, 0), (D, 0)]),
        curve("ParamMouthOpenY", [(0, 0), (0.5, 0.15), (1.0, 0.35), (1.3, 0.05), (1.6, 0.25), (2.0, 0), (D, 0)]),
        curve("ParamEyeForm", [(0, 0), (0.5, -0.6), (3.3, -0.6), (3.8, 0), (D, 0)]),
    ])

    # ---------------------------------------------------------------- びっくり
    D = 2.5
    M["surprised"] = motion(D, [
        curve("ParamEyeLOpen", [(0, 1), (0.08, 0.2), (0.2, 1), (D, 1)]),
        curve("ParamEyeROpen", [(0, 1), (0.08, 0.2), (0.2, 1), (D, 1)]),
        curve("ParamAngleY", [(0, 0), (0.15, 14), (0.5, 9), (1.5, 9), (2.1, 0), (D, 0)]),
        curve("ParamBodyAngleX", [(0, 0), (0.2, -3), (1.5, -2), (2.1, 0), (D, 0)]),
        curve("ParamEyeBallY", [(0, 0), (0.2, 0.2), (1.5, 0.2), (2.1, 0), (D, 0)]),
        curve("ParamMouthOpenY", [(0, 0), (0.2, 0.75), (1.4, 0.7), (2.1, 0), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (0.2, -1), (1.4, -1), (2.1, 0), (D, 0)]),
        curve("ParamEyeBallForm", [(0, 0), (0.2, -1), (1.3, -1), (2.0, 0), (D, 0)]),
    ])

    # ---------------------------------------------------------------- ウィンク
    D = 2.0
    M["wink"] = motion(D, [
        curve("ParamEyeLOpen", [(0, 1), (0.25, 0), (1.2, 0), (1.5, 1), (D, 1)]),
        curve("ParamEyeLSmile", [(0, 0), (0.25, 1), (1.2, 1), (1.5, 0), (D, 0)]),
        curve("ParamEyeROpen", [(0, 1), (D, 1)]),
        curve("ParamAngleZ", [(0, 0), (0.3, -12), (1.3, -12), (D, 0)]),
        curve("ParamAngleX", [(0, 0), (0.3, -6), (1.3, -6), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (0.25, 1), (1.3, 1), (1.7, 0), (D, 0)]),
        curve("ParamMouthOpenY", [(0, 0), (0.3, 0.4), (1.1, 0.25), (1.5, 0), (D, 0)]),
    ])

    # ---------------------------------------------------------------- 考え中
    D = 4.0
    M["thinking"] = motion(D, [
        curve("ParamAngleX", [(0, 0), (0.8, -14), (3.2, -14), (D, 0)]),
        curve("ParamAngleY", [(0, 0), (0.8, 8), (3.2, 8), (D, 0)]),
        curve("ParamAngleZ", [(0, 0), (0.8, 14), (3.2, 14), (D, 0)]),
        curve("ParamBodyAngleZ", [(0, 0), (0.9, 3), (3.2, 3), (D, 0)]),
        curve("ParamEyeBallX", [(0, 0), (0.7, -0.8), (2.0, -0.8), (2.4, -0.4), (3.2, -0.8), (D, 0)]),
        curve("ParamEyeBallY", [(0, 0), (0.7, 0.8), (3.2, 0.8), (D, 0)]),
        curve("ParamEyeLOpen", [(0, 1), (0.8, 0.75), (2.2, 0.75), (2.3, 0), (2.4, 0.75), (3.2, 0.75), (D, 1)]),
        curve("ParamEyeROpen", [(0, 1), (0.8, 0.75), (2.2, 0.75), (2.3, 0), (2.4, 0.75), (3.2, 0.75), (D, 1)]),
        curve("ParamMouthForm", [(0, 0), (0.8, -0.6), (3.2, -0.6), (D, 0)]),
        curve("ParamEyeForm", [(0, 0), (0.8, 0.35), (3.2, 0.35), (D, 0)]),
    ])

    # ---------------------------------------------------------------- やれやれ(首ふり)
    D = 2.8
    M["sigh"] = motion(D, [
        curve("ParamAngleX", [(0, 0), (0.35, -20), (0.75, 18), (1.15, -14), (1.55, 8), (2.0, 0), (D, 0)]),
        curve("ParamAngleY", [(0, 0), (0.3, -8), (2.0, -6), (2.5, 0), (D, 0)]),
        curve("ParamBodyAngleX", [(0, 0), (0.4, -3), (0.8, 3), (1.2, -2), (1.6, 1), (2.1, 0), (D, 0)]),
        curve("ParamEyeLOpen", [(0, 1), (0.3, 0.45), (1.8, 0.45), (2.3, 1), (D, 1)]),
        curve("ParamEyeROpen", [(0, 1), (0.3, 0.45), (1.8, 0.45), (2.3, 1), (D, 1)]),
        curve("ParamMouthForm", [(0, 0), (0.3, -0.8), (1.8, -0.8), (2.3, 0), (D, 0)]),
        curve("ParamEyeForm", [(0, 0), (0.3, 0.45), (1.8, 0.45), (2.3, 0), (D, 0)]),
    ])

    # ---------------------------------------------------------------- 話す(口パクの見本)
    # 口の形は「普通」の列(閉じ → え → 中開き → 大開き)を中心に、ときどき笑顔寄りにする
    D = 3.6
    M["talk"] = motion(D, [
        curve("ParamMouthOpenY", [(0, 0), (0.15, 0.45), (0.3, 0.15), (0.45, 0.7), (0.62, 0.2), (0.8, 0.5), (0.95, 0.1),
                                  (1.2, 0.05), (1.35, 0.6), (1.5, 0.3), (1.7, 0.8), (1.88, 0.25), (2.05, 0.55),
                                  (2.25, 0.1), (2.45, 0.4), (2.65, 0.7), (2.85, 0.2), (3.1, 0), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (0.6, 0.1), (1.2, 0), (1.9, 0.35), (2.6, 0.1), (3.2, 0), (D, 0)]),
        curve("ParamAngleX", [(0, 0), (0.6, 5), (1.3, -3), (2.1, 4), (2.9, 0), (D, 0)]),
        curve("ParamAngleY", [(0, 0), (0.4, 3), (1.1, -2), (1.8, 3), (2.6, 0), (D, 0)]),
        curve("ParamBodyAngleX", [(0, 0), (0.8, 2), (1.8, -1), (2.9, 0), (D, 0)]),
        curve("ParamEyeLOpen", [(0, 1), (1.95, 1), (2.05, 0), (2.15, 1), (D, 1)]),
        curve("ParamEyeROpen", [(0, 1), (1.95, 1), (2.05, 0), (2.15, 1), (D, 1)]),
    ])

    # ---------------------------------------------------------------- 怒る
    D = 3.2
    M["angry"] = motion(D, [
        curve("ParamEyeForm", [(0, 0), (0.25, 1), (2.6, 1), (3.0, 0), (D, 0)]),
        curve("ParamEyeLOpen", [(0, 1), (0.25, 0.85), (2.6, 0.85), (3.0, 1), (D, 1)]),
        curve("ParamEyeROpen", [(0, 1), (0.25, 0.85), (2.6, 0.85), (3.0, 1), (D, 1)]),
        curve("ParamMouthForm", [(0, 0), (0.25, -1), (2.6, -1), (3.0, 0), (D, 0)]),
        curve("ParamMouthOpenY", [(0, 0), (0.3, 0.55), (0.5, 0.2), (0.7, 0.7), (0.9, 0.15), (1.1, 0.5), (1.35, 0), (D, 0)]),
        curve("ParamAngleY", [(0, 0), (0.2, -6), (0.5, 2), (0.8, -5), (1.3, -3), (2.6, -3), (3.0, 0), (D, 0)]),
        curve("ParamAngleZ", [(0, 0), (0.3, -6), (2.6, -6), (3.0, 0), (D, 0)]),
        curve("ParamAngleX", [(0, 0), (1.3, 0), (1.7, 16), (2.6, 16), (3.0, 0), (D, 0)]),
        curve("ParamEyeBallX", [(0, 0), (1.3, 0), (1.6, 0.8), (2.6, 0.8), (3.0, 0), (D, 0)]),
        curve("ParamBodyAngleX", [(0, 0), (0.3, -3), (1.3, -3), (1.7, 3), (2.6, 3), (3.0, 0), (D, 0)]),
    ])

    # ---------------------------------------------------------------- 悲しむ
    D = 4.5
    M["sad"] = motion(D, [
        curve("ParamEyeForm", [(0, 0), (0.6, -1), (3.8, -1), (4.3, 0), (D, 0)]),
        curve("ParamEyeLOpen", [(0, 1), (0.6, 0.7), (2.0, 0.7), (2.15, 0), (2.35, 0.7), (3.8, 0.7), (4.3, 1), (D, 1)]),
        curve("ParamEyeROpen", [(0, 1), (0.6, 0.7), (2.0, 0.7), (2.15, 0), (2.35, 0.7), (3.8, 0.7), (4.3, 1), (D, 1)]),
        curve("ParamEyeBallY", [(0, 0), (0.6, -0.6), (3.8, -0.6), (4.3, 0), (D, 0)]),
        curve("ParamEyeBallX", [(0, 0), (0.6, 0.3), (3.8, 0.3), (4.3, 0), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (0.6, -1), (3.8, -1), (4.3, 0), (D, 0)]),
        curve("ParamAngleY", [(0, 0), (0.8, -16), (3.8, -14), (4.4, 0), (D, 0)]),
        curve("ParamAngleZ", [(0, 0), (0.9, 7), (3.8, 7), (4.4, 0), (D, 0)]),
        curve("ParamBodyAngleZ", [(0, 0), (1.0, 3), (3.8, 3), (4.4, 0), (D, 0)]),
    ])

    # ---------------------------------------------------------------- 笑う
    D = 3.0
    M["laugh"] = motion(D, [
        curve("ParamEyeLOpen", [(0, 1), (0.25, 0), (2.4, 0), (2.8, 1), (D, 1)]),
        curve("ParamEyeROpen", [(0, 1), (0.25, 0), (2.4, 0), (2.8, 1), (D, 1)]),
        curve("ParamEyeLSmile", [(0, 0), (0.25, 1), (2.4, 1), (2.8, 0), (D, 0)]),
        curve("ParamEyeRSmile", [(0, 0), (0.25, 1), (2.4, 1), (2.8, 0), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (0.25, 1), (2.4, 1), (2.8, 0), (D, 0)]),
        curve("ParamMouthOpenY", [(0, 0), (0.25, 0.9), (0.45, 0.5), (0.65, 0.95), (0.85, 0.45), (1.05, 0.9), (1.3, 0.5),
                                  (1.55, 0.8), (2.0, 0.4), (2.4, 0.2), (2.8, 0), (D, 0)]),
        curve("ParamCheek", [(0, 0), (0.4, 0.7), (2.4, 0.7), (2.9, 0), (D, 0)]),
        curve("ParamAngleY", [(0, 0), (0.25, 8), (0.45, 3), (0.65, 9), (0.85, 3), (1.05, 8), (1.3, 3), (1.6, 6), (2.4, 2), (2.9, 0), (D, 0)]),
        curve("ParamAngleZ", [(0, 0), (0.4, -6), (2.4, -4), (2.9, 0), (D, 0)]),
        curve("ParamBodyAngleX", [(0, 0), (0.4, 2), (1.2, -1), (2.4, 1), (2.9, 0), (D, 0)]),
    ])

    # ---------------------------------------------------------------- 待機(ゆっくりあたりを見る)
    D = 8.0
    M["idle_look"] = motion(D, [
        curve("ParamAngleX", [(0, 0), (1.5, 8), (3.5, 8), (5.0, -6), (6.8, -6), (D, 0)]),
        curve("ParamAngleY", [(0, 0), (1.5, -3), (3.5, -2), (5.0, 2), (6.8, 2), (D, 0)]),
        curve("ParamEyeBallX", [(0, 0), (1.2, 0.5), (3.5, 0.5), (4.8, -0.4), (6.8, -0.4), (D, 0)]),
        curve("ParamBodyAngleX", [(0, 0), (2.0, 2), (5.5, -2), (D, 0)]),
    ])
    M["idle_breath"] = motion(6.0, [
        curve("ParamAngleZ", [(0, 0), (2.0, 4), (4.0, -3), (6.0, 0)]),
        curve("ParamBodyAngleZ", [(0, 0), (2.2, 1.5), (4.2, -1.5), (6.0, 0)]),
    ])

    MANIFEST = [
        ("smile", "微笑み", 0.4, 0.5, False), ("nod", "うなずき", 0.3, 0.4, False),
        ("tilt", "首かしげ", 0.4, 0.5, False), ("shy", "照れ隠し", 0.4, 0.5, False),
        ("surprised", "びっくり", 0.15, 0.5, False), ("wink", "ウィンク", 0.3, 0.4, False),
        ("thinking", "考え中", 0.5, 0.5, False), ("sigh", "やれやれ", 0.3, 0.5, False),
        ("talk", "話す", 0.2, 0.3, False),
        ("laugh", "笑う", 0.2, 0.4, False), ("angry", "怒る", 0.2, 0.4, False), ("sad", "悲しむ", 0.5, 0.6, False),
        ("idle_look", "待機(見回す)", 1.0, 1.0, True), ("idle_breath", "待機(ゆらぐ)", 1.0, 1.0, True),
    ]
    return M, MANIFEST


# 表情: モーションとは別に、その場で切り替えて保つ値の組(参照シートの「表情参考」をもとに作成)
# 口の開き・目の開きは自動の動き(口パク・まばたき)と重なるので、ここでは形だけを決める
EXPRESSIONS = [
    {"name": "通常", "params": {}},
    {"name": "微笑", "params": {"ParamMouthForm": 1, "ParamEyeLSmile": 0.35, "ParamEyeRSmile": 0.35, "ParamCheek": 0.25}},
    {"name": "閉じ目", "params": {"ParamEyeLOpen": 0, "ParamEyeROpen": 0}},
    {"name": "驚き", "params": {"ParamEyeBallForm": -1, "ParamMouthForm": -1, "ParamMouthOpenY": 0.4}},
    {"name": "怒り", "params": {"ParamEyeForm": 1, "ParamEyeLOpen": 0.85, "ParamEyeROpen": 0.85, "ParamMouthForm": -1}},
    {"name": "悲しみ", "params": {"ParamEyeForm": -1, "ParamEyeLOpen": 0.75, "ParamEyeROpen": 0.75, "ParamMouthForm": -1, "ParamEyeBallY": -0.4}},
    {"name": "照れ", "params": {"ParamEyeForm": -0.6, "ParamCheek": 1, "ParamMouthForm": 0.3, "ParamEyeBallX": 0.5, "ParamEyeBallY": -0.3}},
    {"name": "ジト目", "params": {"ParamEyeLOpen": 0.55, "ParamEyeROpen": 0.55, "ParamEyeForm": 0.4, "ParamMouthForm": -0.6}},
]


def main():
    motions, manifest = define()
    os.makedirs(OUT, exist_ok=True)
    index = []
    for key, name, fin, fout, idle in manifest:
        data = motions[key]
        data["Meta"]["FadeInTime"] = fin
        data["Meta"]["FadeOutTime"] = fout
        fn = f"{key}.motion3.json"
        with open(os.path.join(OUT, fn), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        index.append({"file": fn, "name": name, "idle": idle, "duration": data["Meta"]["Duration"]})
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "expressions.json"), "w", encoding="utf-8") as fh:
        json.dump(EXPRESSIONS, fh, ensure_ascii=False, indent=1)
    print(f"表情 {len(EXPRESSIONS)} 種を書き出し")
    print(f"モーション {len(index)} 本を書き出し: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
