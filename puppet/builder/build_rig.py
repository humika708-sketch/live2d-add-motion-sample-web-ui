#!/usr/bin/env python3
"""正面立ち絵1枚から、ライブ2D風に動くパペット(puppet.json + texture.png)を組み立てる。

流れ:
  1. 髪を色で見分け、「後ろ髪」「前の房」「前髪」に分ける
  2. 顔・目(白目・瞳・まつ毛・目の形の切り抜き)・口・体に分ける
  3. 部品を動かしたときに見える「隠れていた部分」を周りの色から塗り足す
  4. 部品ごとにメッシュ(三角形の網)を作り、1枚のテクスチャにまとめる
  5. 頭の向き・まばたき・口・呼吸・体の傾き・髪揺れの変形(キーフォーム)を計算で作る

使い方: python3 build_rig.py [設定ファイル(既定: rig_config.json)]
出力先は設定の "output"(既定 ../models/kurisu)。途中経過の画像は debug/ に出る。
"""
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from inpaint_lama import inpaint as lama_inpaint
import mouth_from_images
import eye_parts as eyelib

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================ 読み込み

def load_config():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "rig_config.json")
    cfg = json.load(open(path, encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(path))
    src = os.path.join(base, cfg["source"])
    if not os.path.exists(src):
        print(f"高解像度版が無いため元画像を使う: {src}")
        src = os.path.join(base, cfg["fallbackSource"])
    return cfg, base, src


CFG, BASE, SRC = load_config()
IMG = np.array(Image.open(SRC).convert("RGBA")).astype(np.float32) / 255.0
H, W = IMG.shape[:2]
S = W / CFG["baseWidth"]  # 設定の座標(元画像基準)→ 実際の画素への倍率
OUT = os.path.join(BASE, CFG["output"])
DEBUG = os.path.join(OUT, "debug")
os.makedirs(DEBUG, exist_ok=True)

ALPHA = np.where(IMG[..., 3] > 0.94, 1.0, IMG[..., 3])   # 背景除去でわずかに透けている部分を不透明に戻す
RGB = IMG[..., :3]

# 元の絵の眉(前髪の上に透けて描かれた細い線)を取り出し、絵からは消しておく(眉は別の部品として動かす)
import brow_parts
BROW_ALPHAS, BROW_INFO = [], []
if "眉" in CFG:
    _ORIG_RGB = RGB.copy()
    _boxes = [tuple(int(round(v * (IMG.shape[1] / CFG["baseWidth"]))) for v in b) for b in CFG["眉"]["範囲"]]
    RGB, BROW_ALPHAS, BROW_INFO = brow_parts.extract(RGB, RGB.max(axis=2) * 255, IMG.shape[1] / CFG["baseWidth"], _boxes)
OPAQUE = ALPHA > 0.5
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def P(v):
    """設定の座標を実際の画素座標へ"""
    if isinstance(v, (list, tuple)):
        return [P(x) for x in v]
    return v * S


def save_debug(name, arr):
    a = np.clip(arr * 255 if arr.dtype != np.uint8 else arr, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(os.path.join(DEBUG, name))


def polygon_mask(points):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in points], fill=255)
    return np.array(m) > 127


def ellipse_mask(center, radius):
    cx, cy = center
    rx, ry = radius
    return ((XX - cx) / rx) ** 2 + ((YY - cy) / ry) ** 2 <= 1.0


def soft(mask, sigma=0.6):
    """二値マスクの縁を少しぼかす(部品の境目がギザギザにならないように)"""
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma * S)


def decontaminate(rgb, alpha, core_thresh=0.9):
    """半透明の縁の画素の色を、内側(不透明部分)の色に置き換える。
    縁に背景や下の部品の色(白衣の白など)が混ざっていると、動かしたときに白い縁取りが見えるため"""
    core = alpha >= core_thresh
    if not core.any():
        return rgb
    _, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
    edge = (alpha > 0) & ~core
    out = rgb.copy()
    out[edge] = rgb[iy[edge], ix[edge]]
    return out


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def remove_small(mask, min_area):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_area
    return keep[lab]


def diffuse_fill(rgb, known, region, iters=None):
    """region の画素を、known の画素の色から塗り足す。
    多重解像度で「粗い段階で穴を埋めてから細かい段階へ戻す」方式(プッシュ・プル法)なので、
    大きな穴でもブロック状にならず、なめらかにつながる。iters は互換のため残している(未使用)。"""
    out = rgb.copy()
    if not region.any() or not known.any():
        return out
    ys, xs = np.where(region)
    pad = int(64 * S)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
    c = rgb[y0:y1, x0:x1].astype(np.float32)
    w = known[y0:y1, x0:x1].astype(np.float32)
    cs, ws = [c * w[..., None]], [w]
    while min(cs[-1].shape[:2]) > 2:
        h2, w2 = (cs[-1].shape[0] + 1) // 2, (cs[-1].shape[1] + 1) // 2
        cs.append(cv2.resize(cs[-1], (w2, h2), interpolation=cv2.INTER_AREA))
        ws.append(cv2.resize(ws[-1], (w2, h2), interpolation=cv2.INTER_AREA))
    col = cs[-1] / np.maximum(ws[-1], 1e-6)[..., None]
    for k in range(len(cs) - 2, -1, -1):
        up = cv2.resize(col, (cs[k].shape[1], cs[k].shape[0]), interpolation=cv2.INTER_LINEAR)
        wk = np.clip(ws[k], 0, 1)[..., None]
        col = cs[k] / np.maximum(ws[k], 1e-6)[..., None] * wk + up * (1 - wk)
    sub = out[y0:y1, x0:x1]
    reg = region[y0:y1, x0:x1]
    sub[reg] = col[reg]
    out[y0:y1, x0:x1] = sub
    return out


# ============================================================ 1. 髪の判定と振り分け

hsv = cv2.cvtColor((RGB * 255).astype(np.uint8), cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
HUE = hsv[..., 0] * 360 / 256
SAT = hsv[..., 1]
VAL = hsv[..., 2]
hc = CFG["髪の色"]
hair_color = (((HUE >= hc["色相の下限"]) & (HUE <= hc["色相の上限"]) & (SAT > hc["彩度の下限"]) & (VAL > hc["明度の下限"]))
              | ((HUE <= 30) & (SAT > 150) & (VAL > 25)))
HAIR = remove_small(hair_color & (ALPHA > 0.05), int(40 * S * S))
# 髪の中の小さな穴(ハイライトなど)を埋める
HAIR = HAIR | (remove_small(~HAIR, int(30 * S * S)) ^ ~HAIR)
HAIR &= ALPHA > 0.02

head = CFG["頭"]
jaw = P(head["あごの線"])
HEAD_POLY = polygon_mask(jaw + [[jaw[-1][0], 0], [jaw[0][0], 0]])
# 頭の範囲の中の暗い筋(髪の影の線)は、目と口の周り以外なら髪として扱う
_feature = np.zeros((H, W), bool)
for _e in CFG["目"].values():
    _x0, _y0, _x1, _y1 = [int(round(v)) for v in P(_e["範囲"])]
    _feature[_y0 - int(4 * S):_y1 + int(3 * S), _x0 - int(3 * S):_x1 + int(3 * S)] = True
_m = CFG["口"]["消す範囲"]
_feature[int(P(_m[1])):int(P(_m[3])), int(P(_m[0])):int(P(_m[2]))] = True
_dark_strand = HEAD_POLY & (VAL < 120) & (ALPHA > 0.5) & ~_feature
HAIR = HAIR | remove_small(_dark_strand, int(3 * S * S))
# 髪のすぐ外側の暗い線(髪の輪郭線)も髪に含める。含めないと、髪が動いたときに体の側へ線が残る
_near_hair = ndimage.binary_dilation(HAIR, iterations=int(3 * S))
HAIR = HAIR | (_near_hair & (VAL < 110) & (ALPHA > 0.05) & ~_feature)

# 前髪 = 顔の上にかかる髪
bg = CFG["前髪の範囲"]
BANGS = HAIR & ellipse_mask(P(bg["中心"]), P(bg["半径"])) & (YY < P(bg["下端"])) & HEAD_POLY

# 前の房 = 体(白衣・シャツ)の手前に垂れている髪。
# 設定の範囲(房の通り道)の中の髪のうち、肩より下では「その行の体の左端〜右端の内側」にあるものを前とみなす。
# 肩より上は後ろ髪とひと続きで色では区別できないため、範囲の多角形で決める。
fr = CFG["前の房"]
body0 = OPAQUE & ~HAIR & ~HEAD_POLY
lock_area = np.zeros((H, W), bool)
for poly in fr["範囲"]:
    lock_area |= polygon_mask(P(poly))
inside_body = np.zeros((H, W), bool)
for y in range(H):
    xs = np.where(body0[y])[0]
    if len(xs) >= 2:
        inside_body[y, xs.min():xs.max() + 1] = True
y_start = int(P(fr["開始の高さ"]))
FRONT = HAIR & lock_area & (YY >= P(fr["根元の高さ"])) & ((YY < y_start) | inside_body)
# 大きい塊2つ(左右の房)だけ残す
lab, n = ndimage.label(FRONT)
if n > 2:
    sizes = ndimage.sum(FRONT, lab, range(1, n + 1))
    keep = np.argsort(sizes)[::-1][:2] + 1
    FRONT = np.isin(lab, keep)
FRONT &= ~BANGS
BACK = HAIR & ~BANGS & ~FRONT

# 顔 = 頭の範囲(あごの輪郭線ぶん少し下へ広げる)の中の、髪ではない部分(耳を含む)
HEAD_POLY = HEAD_POLY | (np.roll(HEAD_POLY, int(3 * S), axis=0) & (YY > P(200)))
FACE = OPAQUE & ~HAIR & HEAD_POLY
# 体 = 残り全部(首・白衣・脚など)
BODY = (ALPHA > 0.02) & ~HAIR & ~FACE

vis = np.full((H, W, 3), 1.0, np.float32)
for m, c in [(BODY, (0.85, 0.85, 0.9)), (FACE, (1.0, 0.85, 0.7)), (BACK, (0.55, 0.2, 0.1)),
             (FRONT, (0.95, 0.45, 0.1)), (BANGS, (0.95, 0.8, 0.1))]:
    vis[m] = c
save_debug("01_振り分け.png", vis)
print(f"画像 {W}×{H}(倍率 {S:.2f})  髪 {HAIR.sum()} 画素、前の房 {FRONT.sum()}、前髪 {BANGS.sum()}")


# ============================================================ 2. 目と口の抽出

def extract_eye(key):
    """目の範囲から、白目+瞳の「開いている部分」、まつ毛、瞳を取り出す"""
    e = CFG["目"][key]
    x0, y0, x1, y1 = [int(round(v)) for v in P(e["範囲"])]
    box = np.zeros((H, W), bool)
    box[y0:y1, x0:x1] = True
    dark = box & (VAL < 95) & ~hair_color
    r8, b8 = RGB[..., 0] * 255, RGB[..., 2] * 255
    bluish = box & ((b8 - r8) > -14) & (VAL > 60) & ~hair_color & ~BANGS   # 白目・瞳は青み、肌は赤み
    opening = bluish
    opening = ndimage.binary_closing(opening, iterations=max(1, int(S)))
    opening = ndimage.binary_fill_holes(opening) & box
    opening = remove_small(opening, int(20 * S * S))
    # 瞳の中の暗い部分(瞳孔・縁取り)も開いている部分に含める
    iris_ell = ellipse_mask(P(e["瞳の中心"]), [r * 1.15 for r in P(e["瞳の半径"])])
    opening |= (dark & iris_ell & ndimage.binary_dilation(opening, iterations=int(2 * S)))
    lab, n = ndimage.label(opening)
    if n > 1:   # 瞳の楕円に重なる塊だけ残す
        hit = np.unique(lab[iris_ell & opening])
        opening = np.isin(lab, hit[hit > 0])
    opening = ndimage.binary_fill_holes(opening)
    # 列ごとの上端・下端
    cols = np.where(opening.any(axis=0))[0]
    top = {x: np.where(opening[:, x])[0].min() for x in cols}
    bot = {x: np.where(opening[:, x])[0].max() for x in cols}
    # まつ毛 = 開いている部分の上に接する暗い帯(と目尻の跳ね)
    lash = np.zeros((H, W), bool)
    for x in range(x0, x1):
        t = top.get(x)
        if t is None:
            # 目頭・目尻の外側: 近い列の上端を使う
            near = min(cols, key=lambda c: abs(c - x))
            if abs(near - x) > 6 * S:
                continue
            t = top[near]
        ys = np.arange(max(y0, int(t - 7 * S)), int(t + 2 * S))
        lash[ys, x] = dark[ys, x]
    lash = remove_small(lash & ~BANGS, int(6 * S * S))
    lash = ndimage.binary_closing(lash, iterations=1)
    iris = opening & iris_ell
    return dict(box=(x0, y0, x1, y1), opening=opening, lash=lash, iris=iris, cols=cols, top=top, bot=bot,
                iris_ell=iris_ell)


def extract_eye_v2(key):
    """目を、開いている部分(白目+瞳)と上まつ毛に分ける(eye_parts.py)"""
    e = CFG["目"][key]
    box = [int(round(v)) for v in P(e["範囲"])]
    bright_hair = hair_color & (VAL > 150)
    r = eyelib.extract(RGB, ALPHA, np.dstack([hsv[..., 0], SAT, VAL]), box, P(e["瞳の中心"]), P(e["瞳の半径"]), S,
                       BANGS | bright_hair)
    r["lash"] = r["upper"]
    r["iris"] = r["opening"] & r["iris_ell"]
    return r


EYES = {k: extract_eye_v2(k) for k in ("R", "L")}

m = CFG["口"]
mx0, my0, mx1, my1 = [int(round(v)) for v in P(m["消す範囲"])]
MOUTH_BOX = np.zeros((H, W), bool)
MOUTH_BOX[my0:my1, mx0:mx1] = True

vis = RGB.copy()
for k, e in EYES.items():
    vis[e["opening"]] = vis[e["opening"]] * 0.4 + np.array([0, 0.6, 1.0]) * 0.6
    vis[e["lash"]] = [1, 0, 0]
    vis[e["iris"] & e["opening"]] = vis[e["iris"] & e["opening"]] * 0.5 + np.array([0, 1, 0]) * 0.5
vis[MOUTH_BOX] = vis[MOUTH_BOX] * 0.5 + np.array([1, 0, 1]) * 0.5
cx, cy = P(head["中心"])
crop = vis[int(cy - 70 * S):int(cy + 60 * S), int(cx - 80 * S):int(cx + 80 * S)]
save_debug("02_目と口.png", cv2.resize(crop, None, fx=4 / S, fy=4 / S, interpolation=cv2.INTER_NEAREST))


# ============================================================ 3. 部品の画像を作る(隠れる部分の塗り足し込み)

LAYERS = {}   # 名前 → 高さ×幅×4 の画像(0〜1、ストレートアルファ)

def layer(name, rgb, alpha):
    LAYERS[name] = np.dstack([np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)]).astype(np.float32)


hair_soft = np.clip(soft(HAIR, 0.5), 0, 1)

# --- 後ろ髪: 体・顔・前髪の下にも髪の色を塗り足しておく(頭や髪が動いても隙間が出ない)
# 塗り足すのはキャラクターの輪郭の内側だけ(外側に塗ると背景に色が出てしまう)
under_back = ((ndimage.binary_dilation(BACK, iterations=int(30 * S)) | HEAD_POLY) & (ALPHA > 0.5)) & ~BACK
_hair_ctx = diffuse_fill(RGB, HAIR & (ALPHA > 0.9), ~HAIR & ndimage.binary_dilation(HAIR, iterations=int(60 * S)))
# 顔の後ろ: LaMa は「髪に囲まれた穴」に顔を描いてしまうので使わず、なめらかな髪の色を縦方向にならして毛の流れを出す
_behind_head = under_back & ndimage.binary_dilation(HEAD_POLY, iterations=int(10 * S))
_v = cv2.blur(_hair_ctx, (max(3, int(3 * S)) | 1, max(3, int(40 * S)) | 1))
rgb_back = np.where(_behind_head[..., None], _v, _hair_ctx)
# 白衣の下の髪: こちらは LaMa で毛の質感を描き足す
print("後ろ髪の隠れた部分を描き足し中(LaMa)")
rgb_back = lama_inpaint(rgb_back, under_back & ~_behind_head)
# 胸の前の房の下は、房そのものの絵を後ろ髪にも入れておく(房が揺れたとき、下から同じ髪の質感が見えるように。
# 塗り足しだと暗い色になり、房の横に黒っぽいくさびが見えてしまう)
rgb_back = np.where(FRONT[..., None], RGB, rgb_back)
a_back = np.where(BACK, ALPHA * np.clip(soft(BACK, 0.5) * 1.5, 0, 1), 0)
a_back = np.maximum(a_back, under_back.astype(np.float32))
layer("後ろ髪", decontaminate(rgb_back, a_back), a_back)

# --- 体: 前の房の下を塗り足し、首をあごの下へ延ばす
_lock_all = ndimage.binary_dilation(FRONT, iterations=int(6 * S)) & (ALPHA > 0.5)   # 房全体(毛先の薄い部分も含める)
under_body = _lock_all & inside_body & (YY >= y_start - P(10))
_collar_psd = P(CFG.get("首", {}).get("襟の高さ", 272)) + P(6)
# 首の延長: 各列で、あごのすぐ下の首を「あごの線で折り返して」上へ映す。
# 境目で色がそのままつながり、あごの影の濃淡も続くので、頭が動いて首が見えても継ぎ目が出ない。
# 首の幅は、あごの下に肌がある列だけを使うことで自動的に決まる(髪のある列には伸ばさない)
skin_like_body = (VAL > 100) & ((RGB[..., 0] - RGB[..., 2]) * 255 > 18)
ext = P(CFG["首の延長"])
neck = np.zeros((H, W), bool)
neck_rgb = np.zeros((H, W, 3), np.float32)
pcx = P(head["首の回転軸"][0])
for x in range(int(pcx - P(60)), int(pcx + P(60))):
    col = np.where(HEAD_POLY[:, x])[0]
    if len(col) == 0:
        continue
    y_jaw = col.max() + 1                       # 頭の範囲のすぐ下
    ys = np.arange(y_jaw, min(y_jaw + int(10 * S), H))
    ok = ys[BODY[ys, x] & skin_like_body[ys, x]]
    if len(ok) < 3:
        continue
    # あごの輪郭線の名残(輪郭線と肌の間の数画素)も塗り替えるため、少し下のきれいな首から折り返す
    y0 = ok[0] + int(2 * S)
    # 下向きに肌が続く長さ(襟に当たるまで)
    y_end = y0
    while y_end + 1 < H and BODY[y_end + 1, x] and skin_like_body[y_end + 1, x]:
        y_end += 1
    y_top = int(max(y_jaw - ext, 0))
    for y in range(y_top, y0):
        src = min(y0 + (y0 - 1 - y), y_end)     # あごの線で折り返した位置(襟より下には行かない)
        neck_rgb[y, x] = RGB[src, x]
    neck[y_top:y0, x] = True
neck &= HEAD_POLY | (YY < P(270))
# 横方向に少しならして筋っぽさを消す
neck_rgb = cv2.GaussianBlur(neck_rgb, (0, 0), 1.2 * S)
_wt = cv2.GaussianBlur(neck.astype(np.float32), (0, 0), 1.2 * S)
neck_rgb = neck_rgb / np.maximum(_wt, 1e-4)[..., None]
known_body = BODY & (ALPHA > 0.9) & ~FRONT
print("胸の房の下の白衣を描き足し中(LaMa)")
# 房を丸ごと消してから描き足し、使うのは白衣がある部分だけ(肩の上の髪の色がにじまないように)
_filled = lama_inpaint(RGB, _lock_all)
rgb_body = np.where(under_body[..., None], _filled, RGB)
rgb_body = np.where(neck[..., None], neck_rgb, rgb_body)
a_body = np.where(BODY, ALPHA * (1 - hair_soft * (~BODY)), 0)
a_body = np.where(BODY, ALPHA, 0) * np.clip(1.0 - soft(HAIR & ~BODY, 0.4), 0, 1)
# 左右の端だけぼかす(上下は顔に隠れる)
# 左右の端は広めにぼかして、後ろ髪へなじませる(端の影の色が首を傾けたときに灰色のしみに見えるため)
neck_soft = np.clip(cv2.GaussianBlur(neck.astype(np.float32), (0, 0), 2.5 * S) * 1.5 - 0.15, 0, 1) * neck
a_body = np.maximum(a_body, np.maximum(under_body.astype(np.float32), neck_soft))
layer("体", rgb_body, a_body)
# パーツ分けPSD用の「首」: あごの下から襟までの肌と、あごの下に隠れる首の延長(他の道具で首を頭に追従させるため)
_neck_zone = (np.abs(XX - P(head["首の回転軸"][0])) < P(60)) & (YY < _collar_psd) & (YY > P(head["中心"][1]))
_neck_skin = (BODY | neck) & skin_like_body & _neck_zone
_neck_skin = ndimage.binary_closing(_neck_skin, iterations=int(2 * S))
_lab_n, _n_n = ndimage.label(_neck_skin)
if _n_n > 1:
    _neck_skin = _lab_n == (np.argmax(ndimage.sum(_neck_skin, _lab_n, range(1, _n_n + 1))) + 1)
_a_neck = np.clip(soft(_neck_skin, 0.6) * 1.4, 0, 1) * np.maximum(np.where(neck, 0, a_body), neck_soft)
layer("PSD_首", decontaminate(rgb_body, _a_neck), _a_neck)
# パーツ分けPSD用の「体」: 首の延長(あごの下に隠れる部分)は首の層だけに入れる。
# 体に残すと、顔が動いたとき体に残った首の延長が四角く見えてしまう
layer("PSD_体", rgb_body, a_body * (1 - neck.astype(np.float32)))

# --- 前の房
a_front = np.where(ndimage.binary_dilation(FRONT, iterations=1), ALPHA * np.clip(soft(FRONT, 0.5) * 1.6, 0, 1), 0)
layer("前の房", decontaminate(RGB, a_front), a_front)

# --- 顔: 前髪の下の肌、目と口を消した肌を塗り足す
eye_erase = np.zeros((H, W), bool)
for e in EYES.values():
    e["erase"] = eyelib.erase_region(e, S, H, W) & ~BANGS
    eye_erase |= e["erase"]
skin_like = (VAL > 190) & ((RGB[..., 0] - RGB[..., 2]) * 255 > 12)
# 色の手本は、目のまわりの暗い線だけを除いた肌(まぶたの陰の色は残して、塗り足しが周りの陰となじむようにする)
_near_eye_dark = ndimage.binary_dilation(eye_erase, iterations=int(4 * S)) & (VAL < 175)
skin_known = FACE & skin_like & ~eye_erase & ~MOUTH_BOX & ~_near_eye_dark & (ALPHA > 0.9)
_skin = FACE & skin_like
_lab, _n = ndimage.label(_skin)
if _n > 1:  # いちばん大きい肌の塊(顔)だけ。耳は除く
    _sizes = ndimage.sum(_skin, _lab, range(1, _n + 1))
    _skin = _lab == (np.argmax(_sizes) + 1)
_ys, _xs = np.where(_skin)
_hull = cv2.convexHull(np.stack([_xs, _ys], axis=1).astype(np.int32))
FACE_HULL = np.zeros((H, W), np.uint8)
cv2.fillPoly(FACE_HULL, [_hull], 1)
FACE_HULL = FACE_HULL.astype(bool)
# 額は前髪に隠れて肌が見えないので、凸包を少し上へ延ばす
FACE_HULL |= np.roll(FACE_HULL, -int(12 * S), axis=0) & HEAD_POLY
face_fill = (BANGS & FACE_HULL) | eye_erase | MOUTH_BOX
face_fill &= HEAD_POLY
rgb_face = diffuse_fill(RGB, skin_known, face_fill, iters=80)
# 目のあった所は、各列で上下の肌の色を直線でつないで塗り直す(まぶたの陰の濃淡がそのまま続く)
_eye_known = FACE & ~eye_erase & ~BANGS & (ALPHA > 0.9) & (VAL > 150)
_vfill, _vdone = eyelib.fill_vertical(RGB, eye_erase, _eye_known, S)
rgb_face = np.where((_vdone & eye_erase)[..., None], _vfill, rgb_face)
EYE_BASE = rgb_face.copy()   # まつ毛・下まぶたの線を「差」で取り出すときの下地
a_face = np.where(FACE, ALPHA, 0) * np.clip(1.0 - soft(HAIR & ~BANGS, 0.4), 0, 1)
# 顔の下端(あごの輪郭線の少し下で多角形に切った所)は、ぼかして階段状のギザギザを消す
a_face *= np.clip(soft(HEAD_POLY, 0.9) * 1.25, 0, 1)
a_face = np.maximum(a_face, face_fill.astype(np.float32))
layer("顔", rgb_face, a_face)

# --- 目の部品
for k, e in EYES.items():
    op = e["opening"]
    # 白目: 開いている部分を、白目の画素だけから取った「行ごとの色」で塗る(上まぶたの影の濃淡を残す)。
    # 瞳のあった所もこの色で塗るので、瞳を動かしたり小さくしたりしても元の瞳の跡が出ない
    grow = ndimage.binary_dilation(op, iterations=int(2 * S))
    sclera = op & (SAT < 30) & (VAL > 150) & ~ndimage.binary_dilation(e["iris_ell"], iterations=int(1 * S))
    # 白目の色は、白目の画素から面として塗り広げる(明るい所と、上まぶたの陰の濃淡をそのまま残す)
    rgb_w = diffuse_fill(RGB, sclera, grow & ~sclera)
    layer(f"白目{k}", rgb_w, grow.astype(np.float32))
    # 瞳: 楕円全体まで延ばす(まぶたの下に隠れていた部分も描いておく)
    icx_, icy_ = P(CFG["目"][k]["瞳の中心"])
    irx_, iry_ = P(CFG["目"][k]["瞳の半径"])
    big_ell = ((XX - icx_) / (irx_ * 1.2)) ** 2 + ((YY - icy_) / (iry_ * 1.2)) ** 2 <= 1
    pale_px = (SAT < 30) & (VAL > 150)
    # 瞳の範囲 = 暗い縁取りの輪の内側(白目の画素を含めない。視線を動かしたとき白目がついて来ないように)
    iris_reg = ndimage.binary_fill_holes(e["opening"] & big_ell & ~pale_px)
    lab_, n_ = ndimage.label(iris_reg)
    if n_ > 1:
        iris_reg = lab_ == lab_[int(icy_), int(icx_)] if lab_[int(icy_), int(icx_)] else lab_ == (np.argmax(ndimage.sum(iris_reg, lab_, range(1, n_ + 1))) + 1)
    # まぶたの下に隠れていた部分(開いている部分の外)も、瞳の色で塗り足しておく(上下を見たとき用)
    hidden = e["iris_ell"] & ~e["opening"]
    iris_full = iris_reg | hidden
    rgb_i = diffuse_fill(RGB, iris_reg, iris_full & ~iris_reg, iters=30)
    a_i = np.clip(soft(iris_full, 0.35) * 1.4, 0, 1)
    layer(f"瞳{k}", rgb_i, a_i)
    # 上まつ毛: 不透明度は暗さから、色は背景の明るさを差し引いて逆算する(縁がなめらかになる)
    # 上まつ毛・下まぶたの線は、元の絵と「目を消して塗り直した肌」の差で作る
    # (肌の上に重ねると元の絵どおりに見え、縁もなめらかになる)
    mid_full = np.interp(np.arange(W), e["cols"], ((e["top"] + e["bot"]) / 2)[e["cols"]])
    upper_side = YY <= mid_full[None, :]
    # 開いている部分の上の縁(2S)も、まつ毛の影としてまつ毛に含める(まつ毛と白目の間に白い筋が出ないように)
    top_full = np.interp(np.arange(W), e["cols"], e["top"][e["cols"]])
    # ただし青みのある画素(瞳の上の影・瞳孔の先)は除く(閉じたとき、まつ毛の下に青い影が残るため)
    not_blue = (RGB[..., 2] - RGB[..., 0]) < 0.02
    top_edge = e["opening"] & (YY <= top_full[None, :] + 2 * S) & not_blue & ~e["iris_ell"]
    lash_region = e["erase"] & (~e["opening"] | top_edge) & (upper_side | ndimage.binary_dilation(e["lash"], iterations=int(S)))
    col_l, a_l = eyelib.diff_layer(RGB, EYE_BASE, lash_region, strength=0.30, darker_only=True)
    layer(f"まつ毛{k}", col_l, a_l * ALPHA)
    e["lash"] = a_l > 0.2
    lower_region = e["erase"] & ~e["opening"] & ~lash_region
    col_d, a_d = eyelib.diff_layer(RGB, EYE_BASE, lower_region, strength=0.18, darker_only=True)
    layer(f"下まぶた{k}", col_d, a_d * ALPHA)
    # 目の形(切り抜き用。見た目には描かない)
    layer(f"目の形{k}", np.ones((H, W, 3), np.float32), op.astype(np.float32))

# --- 口: 元の口の線を「肌との差」で切り出す
diff = np.abs(RGB - rgb_face).max(axis=2)
a_m = np.where(MOUTH_BOX, np.clip(diff / 0.18, 0, 1), 0)
layer("口の線", RGB, a_m)

# --- 口の差分画像があれば、それを口の部品にする
HIRES = {}      # 名前 → 高解像度の画像(テクスチャだけ高解像度で持つ部品)
USE_MOUTH_IMAGES = "口の差分" in CFG
if USE_MOUTH_IMAGES:
    print("口の差分画像から口を作成中")
    _white = RGB * ALPHA[..., None] + (1 - ALPHA[..., None])
    MOUTHS, _minfo = mouth_from_images.build(CFG, BASE, _white, S)
    print(f"  倍率 {_minfo['scale']:.4f}、肌の色の補正 {np.round(_minfo['gain'], 3)}")
    for _n, _v in MOUTHS.items():
        LAYERS[f"口_{_n}"] = _v["full"]
        HIRES[f"口_{_n}"] = _v

# --- 前髪
a_bangs = np.where(ndimage.binary_dilation(BANGS, iterations=1), ALPHA * np.clip(soft(BANGS, 0.5) * 1.6, 0, 1), 0)
layer("前髪", decontaminate(RGB, a_bangs), a_bangs)

# --- パーツ分けPSD用の髪(see-through と同じ分け方)
# 「前髪」= 頭の髪すべて(頭頂〜顔まわり)。下端は切り口が目立たないよう、ぼかして消す
_cut0, _cut1 = P(CFG["PSD用"]["頭の髪の下端"][0]), P(CFG["PSD用"]["頭の髪の下端"][1])
_fade = 1 - np.clip((YY - _cut0) / (_cut1 - _cut0), 0, 1)
# 胸の房の根元も含める(除くと、房の始まる高さで前髪に四角い切り欠きができる)。房とはぼかし合わせでつなぐ
a_headhair = ALPHA * np.clip(soft(HAIR, 0.5) * 1.6, 0, 1) * _fade
layer("PSD_前髪", decontaminate(RGB, np.where(HAIR, ALPHA, 0)), a_headhair)
# 「後髪」= 髪全体+隠れた部分の描き足し。頭の上のほうは少し内側に縮めて、前髪がずれても二重に見えないようにする
# 縮める量は高さに応じてなめらかに減らす(一定の高さで切り替えると輪郭に段差ができる)
_depth_in = ndimage.distance_transform_edt(HAIR | under_back)
_shrink = P(8) * (1 - smoothstep(P(150), _cut1, YY))
_inner = _depth_in > _shrink
a_backall = np.where(_inner, 1.0, 0) * np.where(HAIR & (YY >= _cut0), ALPHA, 1.0)
# 髪の見えている部分は元の絵のまま入れておく(前髪・房が少しずれても、下から同じ髪が見えるだけで済む)
layer("PSD_後髪", decontaminate(np.where(HAIR[..., None], RGB, rgb_back), a_backall), a_backall)
# 「横髪」(胸の前の房)は、根元の切り口が目立たないよう上端をぼかして消す
_r0 = P(fr["根元の高さ"])
_lock_fade = np.clip((YY - _r0) / P(30), 0, 1)
layer("PSD_前の房", decontaminate(RGB, a_front), a_front * _lock_fade)

# --- 描き起こす部品(口の中・下唇・頬の赤み)。細かく描くため4倍で作ってから縮める
def painted(name, box, draw_fn, blur=0.8):
    x0, y0, x1, y1 = box
    k = 4
    w, h = int((x1 - x0) * k), int((y1 - y0) * k)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(im), w, h, k)
    im = im.filter(ImageFilter.GaussianBlur(blur * k))
    im = im.resize((int(round(x1 - x0)), int(round(y1 - y0))), Image.LANCZOS)
    arr = np.array(im).astype(np.float32) / 255
    full = np.zeros((H, W, 4), np.float32)
    xi, yi = int(round(x0)), int(round(y0))
    full[yi:yi + arr.shape[0], xi:xi + arr.shape[1]] = arr
    LAYERS[name] = full


mcx, mcy = P(m["中心"])
mw = P(m["幅"])
open_h = mw * 0.36
INNER_BOX = (mcx - mw * 0.36, mcy - open_h * 0.35, mcx + mw * 0.36, mcy + open_h * 0.65 + P(1))

def draw_inner(d, w, h, k):
    d.ellipse([0, 0, w - 1, h - 1], fill=(92, 28, 38, 255))
    d.ellipse([w * 0.18, h * 0.55, w * 0.82, h * 1.25], fill=(196, 92, 104, 255))   # 舌
    d.rectangle([w * 0.12, 0, w * 0.88, h * 0.16], fill=(246, 244, 244, 255))        # 上の歯

painted("口の中", INNER_BOX, draw_inner, blur=0.35)
# 楕円の外側を切り落とす
x0, y0, x1, y1 = [int(round(v)) for v in INNER_BOX]
ell = np.zeros((H, W), np.float32)
ell_m = ellipse_mask(((x0 + x1) / 2, (y0 + y1) / 2), ((x1 - x0) / 2, (y1 - y0) / 2))
LAYERS["口の中"][..., 3] *= soft(ell_m, 0.3)

LOWER_BOX = (mcx - mw * 0.3, INNER_BOX[3] - P(2.5), mcx + mw * 0.3, INNER_BOX[3] + P(1.5))

def draw_lower(d, w, h, k):
    d.arc([0, -h * 1.2, w - 1, h * 0.8], 20, 160, fill=(150, 80, 80, 200), width=int(1.3 * k * S))

painted("下唇", LOWER_BOX, draw_lower, blur=0.25)

for i, c in enumerate(CFG["頬"]):
    ccx, ccy = P(c)
    bw_, bh_ = P(16), P(7)
    box = (ccx - bw_, ccy - bh_, ccx + bw_, ccy + bh_)

    def draw_cheek(d, w, h, k):
        d.ellipse([w * 0.05, h * 0.1, w * 0.95, h * 0.9], fill=(255, 120, 130, 110))
        for j in range(3):
            xx = w * (0.35 + j * 0.13)
            d.line([(xx, h * 0.62), (xx + w * 0.07, h * 0.36)], fill=(235, 90, 110, 140), width=int(0.8 * k * S))

    painted(f"頬{'RL'[i]}", box, draw_cheek, blur=0.9)

# --- 部品シートの素材(感情マーク・汗・涙・眼鏡・眉・頬)
MAT = CFG.get("素材", {})
MAT_DIR = os.path.join(BASE, MAT.get("フォルダ", ""))
MAT_INFO = {}   # 名前 → (中心x, 中心y, 幅, 高さ)(実際の画素)


def load_material(key, crop=None):
    m = MAT.get(key)
    if not m:
        return None
    path = os.path.join(MAT_DIR, m["ファイル"])
    if not os.path.exists(path):
        print(f"  素材が無いので飛ばす: {path}")
        return None
    im = Image.open(path).convert("RGBA")
    if crop:
        im = im.crop(tuple(crop))
    # 高解像度化で残ったごく薄い透明度のノイズを消す(残すと素材のまわりに四角い影が出て、大きさの計算もずれる)
    arr = np.array(im).astype(np.float32)
    a = arr[..., 3] / 255
    arr[..., 3] = np.clip((a - 0.08) / 0.92, 0, 1) * 255
    im = Image.fromarray(arr.round().astype(np.uint8), "RGBA")
    box = Image.fromarray((arr[..., 3] > 20).astype(np.uint8) * 255).getbbox()
    return im.crop(box) if box else None


def place_material(name, im, center, width, alpha_scale=1.0, yscale=1.0):
    """素材を、元画像の座標で指定した中心・幅に置いた全体サイズの層にする(yscale: 縦の倍率)"""
    w = P(width)
    h = w * im.height / im.width * yscale
    cx, cy = P(center)
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    im2 = im.resize((max(1, int(round(w))), max(1, int(round(h)))), Image.LANCZOS)
    arr = np.array(im2).astype(np.float32) / 255
    full = np.zeros((H, W, 4), np.float32)
    ys, xs = slice(max(y0, 0), min(y0 + arr.shape[0], H)), slice(max(x0, 0), min(x0 + arr.shape[1], W))
    full[ys, xs] = arr[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    full[..., 3] *= alpha_scale
    LAYERS[name] = full
    MAT_INFO[name] = (cx, cy, w, h)


if MAT:
    print("部品シートの素材を配置中")
    # 頬: 描き起こしの頬を、素材の頬(左右に分かれた2つの塊)に置き換える
    blush = load_material("頬")
    if blush is not None:
        # 左右の塊の境目は、真ん中付近で最も薄い列にする(真ん中で切ると、はみ出したぼかしが直線で切れる)
        ba = np.array(blush)[..., 3].astype(np.float32).sum(axis=0)
        q = blush.width // 4
        half = q + int(np.argmin(ba[q:3 * q]))
        for i, (part, c) in enumerate(zip((blush.crop((0, 0, half, blush.height)), blush.crop((half, 0, blush.width, blush.height))), CFG["頬"])):
            # 切った側の縁を、なめらかに0へ落とす
            pa = np.array(part).astype(np.float32)
            n = pa.shape[1]
            ramp = np.clip(np.arange(n) / (n * 0.18), 0, 1)
            ramp = ramp[::-1] if i == 0 else ramp
            pa[..., 3] *= ramp[None, :]
            part = Image.fromarray(pa.round().astype(np.uint8), "RGBA")
            part = part.crop(part.getbbox())
            place_material(f"頬{'RL'[i]}", part, c, MAT["頬"]["幅"])
    for key in ("眼鏡_黒", "眼鏡_赤", "怒りマーク", "汗"):
        im = load_material(key)
        if im is not None:
            place_material(key, im, MAT[key]["中心"], MAT[key]["幅"], yscale=MAT[key].get("縦の倍率", 1.0))
    tear = load_material("涙")
    if tear is not None:
        for i, pos in enumerate(MAT["涙"]["位置"]):
            place_material(f"涙{'RL'[i]}", tear, pos, MAT["涙"]["幅"])


# 元の眉: 元の絵と、眉を消した絵との差で取り出す(肌や前髪の上に重ねると元どおりに見える)
for i, (ba, info) in enumerate(zip(BROW_ALPHAS, BROW_INFO)):
    if info is None:
        continue
    _reg = ndimage.binary_dilation(ba > 0.02, iterations=max(1, int(S)))
    _col, _a = eyelib.diff_layer(_ORIG_RGB, RGB, _reg, strength=0.25, darker_only=True)
    layer(f"眉{'RL'[i]}", _col, _a * ALPHA)
    MAT_INFO[f"眉{'RL'[i]}"] = (info[0], info[1], info[2], P(6))

# 部品を元の位置のまま全体サイズでも保存する(PSD書き出し export_psd.py で使う)
os.makedirs(os.path.join(OUT, "layers"), exist_ok=True)
for name, arr in LAYERS.items():
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8), "RGBA").save(os.path.join(OUT, "layers", f"{name}.png"))

for name, arr in LAYERS.items():
    ys, xs = np.where(arr[..., 3] > 0.004)
    if len(ys):
        c = arr[max(ys.min() - 2, 0):ys.max() + 3, max(xs.min() - 2, 0):xs.max() + 3]
        Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8), "RGBA").save(os.path.join(DEBUG, f"部品_{name}.png"))


# ============================================================ 4. メッシュとテクスチャ

def crop_bbox(arr, pad=2):
    ys, xs = np.where(arr[..., 3] > 0.004)
    return (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad + 1, W), min(ys.max() + pad + 1, H))


def grid_mesh(arr, bbox, cell):
    """部品の範囲を格子に切り、中身のあるマスだけ三角形にする"""
    x0, y0, x1, y1 = bbox
    nx = max(1, math.ceil((x1 - x0) / cell))
    ny = max(1, math.ceil((y1 - y0) / cell))
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    a = arr[..., 3] > 0.004
    used = np.zeros((ny, nx), bool)
    for j in range(ny):
        for i in range(nx):
            ya, yb = int(ys[j]) - 1, int(math.ceil(ys[j + 1])) + 1
            xa, xb = int(xs[i]) - 1, int(math.ceil(xs[i + 1])) + 1
            used[j, i] = a[max(ya, 0):yb, max(xa, 0):xb].any()
    index = {}
    verts = []
    tris = []
    def vid(i, j):
        if (i, j) not in index:
            index[(i, j)] = len(verts)
            verts.append((float(xs[i]), float(ys[j])))
        return index[(i, j)]
    for j in range(ny):
        for i in range(nx):
            if used[j, i]:
                a0, b0, c0, d0 = vid(i, j), vid(i + 1, j), vid(i, j + 1), vid(i + 1, j + 1)
                tris += [a0, b0, c0, b0, d0, c0]
    return verts, tris


def strip_mesh(xs, top, bot):
    """上下の縁で挟まれた帯状のメッシュ(目の形など)"""
    verts, tris = [], []
    for x, t, b in zip(xs, top, bot):
        verts += [(x, t), (x, b)]
    for i in range(len(xs) - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        tris += [a, c, b, c, d, b]
    return verts, tris


PARTS = []   # 描画部品の定義(メッシュ・親・キーフォームなど)

def add_part(name, parent, order, cell=None, mesh=None, **extra):
    arr = LAYERS[name]
    bbox = crop_bbox(arr)
    verts, tris = mesh if mesh else grid_mesh(arr, bbox, cell)
    PARTS.append(dict(name=name, parent=parent, order=order, bbox=bbox, verts=verts, tris=tris, **extra))
    return PARTS[-1]


# ============================================================ 5. 変形(キーフォーム)の計算

DEFORMERS = []

def warp(id_, parent, rect, cols, rows, keys, disp_fn):
    """disp_fn(組み合わせの値の辞書, xs, ys) → (dx, dy) の配列。全組み合わせのキーフォームを作る"""
    x, y, w, h = rect
    gx, gy = np.meshgrid(np.linspace(x, x + w, cols + 1), np.linspace(y, y + h, rows + 1))
    gx, gy = gx.ravel(), gy.ravel()
    forms = []
    for combo in combos(keys):
        dx, dy = disp_fn(combo, gx, gy)
        forms.append(np.round(np.stack([dx, dy], axis=1).ravel(), 2).tolist())
    DEFORMERS.append(dict(id=id_, type="warp", parent=parent, rect=[round(v, 2) for v in rect],
                          cols=cols, rows=rows, keys=keys, forms=forms))


def combos(keys):
    """キーの全組み合わせ(先頭のキーが最も速く変わる順)"""
    out = [{}]
    for k in keys:
        out = [dict(c) for c in out]
        out = [dict(c, **{k["param"]: v}) for v in k["values"] for c in out]
        # 先頭が速く変わる順に並べ替え
    idx = []
    def rec(i, cur):
        if i < 0:
            idx.append(dict(cur))
            return
        for v in keys[i]["values"]:
            cur[keys[i]["param"]] = v
            rec(i - 1, cur)
    rec(len(keys) - 1, {})
    return idx




# --- 頭の立体的な回転(球に貼った絵を回すと考える)
HCX, HCY = P(head["中心"])
HRX, HRY = P(head["半径"])
YAW_MAX, PITCH_MAX, ROLL_MAX = 10.0, 7.0, 8.0   # 大きくすると顔の形がゆがみ、首や髪の根元の塗り足しが見えやすくなる。足りない分は体の動きで補う
CHIN_Y = max(p[1] for p in P(head["あごの線"]))      # あご先の高さ
CHIN_PIN = 0.7                                       # あご先の動きを打ち消す割合(1で完全に固定)
PX, PY = P(head["首の回転軸"])

def head_warp_disp(ax, ay, xs, ys, depth_bias=0.0):
    """頭の向き(左右・上下)による変形。あご先の動きの大部分を打ち消し、首のところを支点に回るようにする
    (頭全体がずれて首が伸びたり、あごの下が見えたりしないように)"""
    dx, dy = _head_warp_raw(ax, ay, xs, ys, depth_bias)
    cdx, cdy = _head_warp_raw(ax, ay, np.array([HCX]), np.array([CHIN_Y]), depth_bias)
    return dx - CHIN_PIN * cdx[0], dy - CHIN_PIN * cdy[0]


def _head_warp_raw(ax, ay, xs, ys, depth_bias=0.0):
    th = math.radians(ax / 30 * YAW_MAX)
    ph = math.radians(ay / 30 * PITCH_MAX)
    u = (xs - HCX) / HRX
    v = (ys - HCY) / HRY
    z = np.clip(1 - u * u - v * v, 0, None) + depth_bias
    x1 = u * math.cos(th) + z * math.sin(th)
    z1 = -u * math.sin(th) + z * math.cos(th)
    y1 = v * math.cos(ph) - z1 * math.sin(ph)
    X = HCX + HRX * (x1 + 0.22 * math.sin(th))
    Y = HCY + HRY * (y1 - 0.12 * math.sin(ph))
    return X - xs, Y - ys


def roll(az, xs, ys):
    a = math.radians(-az / 30 * ROLL_MAX)
    dx, dy = xs - PX, ys - PY
    return PX + dx * math.cos(a) - dy * math.sin(a) - xs, PY + dx * math.sin(a) + dy * math.cos(a) - ys


def head_total_disp(ax, ay, az, xs, ys):
    dx1, dy1 = head_warp_disp(ax, ay, xs, ys)
    dx2, dy2 = roll(az, xs + dx1, ys + dy1)
    return dx1 + dx2, dy1 + dy2


K3 = [-30, 0, 30]
KEY_AX = {"param": "ParamAngleX", "values": K3}
KEY_AY = {"param": "ParamAngleY", "values": K3}
KEY_AZ = {"param": "ParamAngleZ", "values": [-30, -15, 0, 15, 30]}

# 体: 左右の傾き・回転・呼吸
bd = CFG["体"]
HIP = P(bd["腰の高さ"])
SHOULDER = P(bd["肩の高さ"])
BCX = P(head["首の回転軸"][0])

def body_disp(c, xs, ys):
    bx, bz, br = c["ParamBodyAngleX"], c["ParamBodyAngleZ"], c["ParamBreath"]
    up = smoothstep(HIP + P(60), P(80), ys)            # 腰から上ほど大きく動く
    dx = bx / 10 * P(12) * up ** 1.3
    dy = np.zeros_like(xs)
    a = math.radians(-bz / 10 * 3.5)
    rx, ry = xs - BCX, ys - HIP
    dx = dx + (rx * math.cos(a) - ry * math.sin(a) - rx) * up
    dy = dy + (rx * math.sin(a) + ry * math.cos(a) - ry) * up
    chest = smoothstep(P(600), P(330), ys)
    dy = dy - br * P(2.2) * chest
    dx = dx + br * (xs - BCX) * 0.006 * smoothstep(P(560), P(300), ys) * smoothstep(P(200), P(300), ys)
    return dx, dy

warp("体", None, [0, 0, W, H], 8, 16,
     [{"param": "ParamBodyAngleX", "values": [-10, 0, 10]},
      {"param": "ParamBodyAngleZ", "values": [-10, 0, 10]},
      {"param": "ParamBreath", "values": [0, 1]}], body_disp)

# 首の回転(頭の傾き)
DEFORMERS.append(dict(id="首", type="rotation", parent="体", origin=[PX, PY], keys=[KEY_AZ],
                      forms=[{"angle": -v / 30 * ROLL_MAX} for v in KEY_AZ["values"]]))

# 首: あごのところでは頭と一緒に動き、襟に向かって動きが0になる(首を振ったとき、首が顔についていく)。
# あごより上(顔に隠れている首の延長)は頭と完全に一緒に動くので、首を振っても見えてこない
_jaw_arr = np.array(jaw)
_collar = P(CFG.get("首", {}).get("襟の高さ", 272))


def neck_disp(c, xs, ys):
    jy = np.interp(xs, _jaw_arr[:, 0], _jaw_arr[:, 1])
    dx, dy = head_total_disp(c["ParamAngleX"], c["ParamAngleY"], c["ParamAngleZ"], xs, np.minimum(ys, jy))
    wy = 1 - smoothstep(jy, _collar, ys)
    wx = 1 - smoothstep(P(44), P(60), np.abs(xs - PX))
    return dx * wy * wx, dy * wy * wx


warp("首の追従", "体", [PX - P(62), P(185), P(124), P(115)], 16, 16, [KEY_AX, KEY_AY, KEY_AZ], neck_disp)

# 頭の向き(左右・上下)
hx0, hy0, hx1, hy1 = P(head["変形の範囲"])
warp("頭", "首", [hx0, hy0, hx1 - hx0, hy1 - hy0], 10, 10, [KEY_AX, KEY_AY],
     lambda c, xs, ys: head_warp_disp(c["ParamAngleX"], c["ParamAngleY"], xs, ys))

# 前髪: 顔より手前にあるぶん少し大きく動く + 揺れ
bangs_box = crop_bbox(LAYERS["前髪"], pad=int(4 * S))
bx0, by0, bx1, by1 = bangs_box
BCX_, BCY_ = P(bg["中心"])
BRX_, BRY_ = P(bg["半径"])

def bangs_weight(xs, ys):
    r = np.sqrt(((xs - BCX_) / BRX_) ** 2 + ((ys - BCY_) / BRY_) ** 2)
    edge = 1 - smoothstep(0.55, 0.95, np.where(ys > BCY_, np.abs(xs - BCX_) / BRX_, r))
    return edge

warp("前髪の奥行き", "頭", [bx0, by0, bx1 - bx0, by1 - by0], 8, 8, [KEY_AX, KEY_AY],
     lambda c, xs, ys: tuple(0.16 * d * bangs_weight(xs, ys) for d in head_warp_disp(c["ParamAngleX"], c["ParamAngleY"], xs, ys)))
warp("前髪の揺れ", "前髪の奥行き", [bx0, by0, bx1 - bx0, by1 - by0], 8, 8,
     [{"param": "ParamHairFront", "values": [-1, 0, 1]}],
     lambda c, xs, ys: (c["ParamHairFront"] * P(4.5) * smoothstep(by0 + (by1 - by0) * 0.45, by1, ys) ** 1.4 * bangs_weight(xs, ys),
                        -abs(c["ParamHairFront"]) * P(0.8) * smoothstep(by0 + (by1 - by0) * 0.45, by1, ys) * bangs_weight(xs, ys)))

# 長い髪: 上のほうは頭に付いていき、下へいくほど物理演算の揺れが効く。
# 頭への追従は、後ろ髪と前の房で「同じ1つの変形器」を使う(別々だと格子の違いで境目がずれる)
bk = CFG["後ろ髪"]
# あごの高さより上の髪は、顔とまったく同じに動かす(顔の縁が髪の上をすべって、ぎざぎざに見えないように)
FOLLOW_START = P(head["首の回転軸"][1]) - P(12)   # ここより上は頭と一緒に動く
FOLLOW_END = FOLLOW_START + P(95)                 # ここより下は頭に付いていかない(肩の上の髪が肩から浮かないように短めにする)
_b1, _b2 = crop_bbox(LAYERS["後ろ髪"], pad=int(6 * S)), crop_bbox(LAYERS["前の房"], pad=int(6 * S))
fx0, fy0 = min(_b1[0], _b2[0]), min(_b1[1], _b2[1])
fx1 = max(_b1[2], _b2[2])
fy1 = max(_b1[3], _b2[3])
_cols = max(8, int((fx1 - fx0) / P(24)))
_rows = max(8, int((fy1 - fy0) / P(30)))

_head_half = (P(head["あごの線"][-1][0]) - P(head["あごの線"][0][0])) / 2


def follow(c, xs, ys):
    wgt = 1 - smoothstep(FOLLOW_START, FOLLOW_END, ys)
    # あごより下で、頭の幅より外(肩にかかる髪)は、頭の動きにほとんど付いていかない
    # (頭を傾けたとき肩の髪が上下にずれて、白衣の縁に影の帯が見えないように)
    side = smoothstep(_head_half - P(8), _head_half + P(40), np.abs(xs - PX)) * smoothstep(FOLLOW_START - P(10), FOLLOW_START + P(25), ys)
    wgt = wgt * (1 - 0.85 * side)
    dx, dy = head_total_disp(c["ParamAngleX"], c["ParamAngleY"], c["ParamAngleZ"], xs, ys)
    return dx * wgt, dy * wgt

warp("髪の追従", "体", [fx0, fy0, fx1 - fx0, fy1 - fy0], _cols, _rows, [KEY_AX, KEY_AY, KEY_AZ], follow)


def hair_sway(name, param, root, tip, amp, cols, rows):
    x0, y0, x1, y1 = crop_bbox(LAYERS[name], pad=int(6 * S))

    def sway(c, xs, ys):
        t = smoothstep(root, tip, ys)
        dx = c[param] * amp * t ** 1.5
        return dx, -np.abs(dx) * 0.12

    warp(f"{name}の揺れ", "髪の追従", [x0, y0, x1 - x0, y1 - y0], cols, rows, [{"param": param, "values": [-1, 0, 1]}], sway)
    return f"{name}の揺れ"

back_parent = hair_sway("後ろ髪", "ParamHairBack", P(bk["根元の高さ"]), P(bk["先端の高さ"]), P(30), 10, 18)
front_parent = hair_sway("前の房", "ParamHairSide", P(fr["根元の高さ"]), P(fr["先端の高さ"]), P(12), 8, 14)


# ============================================================ 6. 部品の登録とキーフォーム

add_part("後ろ髪", back_parent, 0, cell=P(14))
add_part("体", "首の追従", 10, cell=P(10))
add_part("前の房", front_parent, 20, cell=P(10))
face_part = add_part("顔", "頭", 30, cell=P(8))
if USE_MOUTH_IMAGES:
    # 差分の「大開き」では、あご先が口の線からあご先までの長さの約1割下がっていたので、それに合わせる
    _jaw_cx, _jaw_y0 = P(CFG["口の差分"]["全身の口の線の中心"])
    _chin = P(CFG["口の差分"]["全身の鼻先とあご先"][1])
    _drop = (_chin[1] - _jaw_y0) * 0.11
    _half = P(46)
    face_part["keys"] = [{"param": "ParamMouthOpenY", "values": [0, 1]}]
    face_part["forms"] = [[0.0] * (2 * len(face_part["verts"]))]
    f = []
    for (vx, vy) in face_part["verts"]:
        w = smoothstep(_jaw_y0 - P(2), _chin[1] - P(4), vy) * max(0.0, 1 - abs(vx - _jaw_cx) / _half) ** 1.2
        f += [0.0, round(float(_drop * w), 2)]
    face_part["forms"].append(f)

EYE_KEYS = [{"param": None, "values": [0, 1]}, {"param": None, "values": [0, 1]}]

def eye_parts(k, order):
    """目の部品(目の形・白目・瞳・上まつ毛)と、その動き。
    まぶたは「上の縁の線」として扱い、まつ毛は列ごとにその線に沿って動かす(形がくずれない)"""
    e = EYES[k]
    open_id = f"ParamEye{k}Open"
    smile_id = f"ParamEye{k}Smile"
    keys = [{"param": open_id, "values": [0, 1]}, {"param": smile_id, "values": [0, 1]},
            {"param": "ParamEyeForm", "values": [-1, 0, 1]}]
    cols = e["cols"]
    (xa, ya), (xb, yb) = e["corners"]
    n = 21
    xs = np.linspace(xa - 0.5, xb + 0.5, n)
    top = np.interp(xs, cols, e["top"][cols]) - 0.5
    bot = np.interp(xs, cols, e["bot"][cols]) + 0.5
    # 列ごとのばらつきをならす(閉じたときの線がガタつかないように)
    k_ = np.array([1, 2, 3, 2, 1], float); k_ /= k_.sum()
    top = np.convolve(np.pad(top, 2, mode="edge"), k_, mode="valid")
    bot = np.convolve(np.pad(bot, 2, mode="edge"), k_, mode="valid")
    hgt = bot - top
    hc = float(hgt.max())
    tt = np.linspace(0, 1, n)
    bump = np.sin(np.pi * tt)                              # 中央ほど大きい
    corner_line = ya + (yb - ya) * (xs - xa) / max(xb - xa, 1)
    closed_line = corner_line + 0.16 * hc * bump           # 閉じた目: 目頭と目尻を結ぶ線から、ゆるく下へ弧を描く
    smile_line = corner_line + 0.06 * hc - 0.30 * hc * bump  # 笑い目(^^): 上へ弧を描く
    # 目頭側ほど1(右目=画面の左の目は右側が目頭、左目は左側が目頭)
    t_inner = tt if k == "R" else 1 - tt

    def shape(op, sm, fm=0):
        """目の開き・笑い・形(-1 困り目 〜 1 怒り目)に応じた上端・下端"""
        # 怒り目は目頭側の上まぶたを、困り目は目尻側の上まぶたを下げる(参照シートの表情参考より)
        lid = np.where(fm > 0, fm * hgt * 0.40 * t_inner ** 1.4, -fm * hgt * 0.30 * (1 - t_inner) ** 1.4) + abs(fm) * hgt * 0.04
        t_open = np.minimum(top + lid, bot - hgt * 0.25)
        b_open = bot - sm * hgt * 0.28 * bump              # 笑うと下まぶたが上がる
        t_closed = closed_line * (1 - sm) + smile_line * sm
        b_closed = t_closed + 0.3
        t = t_open * op + t_closed * (1 - op)
        b = b_open * op + b_closed * (1 - op)
        return t, np.maximum(b, t + 0.3)

    verts, tris = strip_mesh(xs, top, bot)
    forms = []
    for c in combos(keys):
        t, b = shape(c[open_id], c[smile_id], c["ParamEyeForm"])
        d = []
        for i in range(n):
            d += [0.0, float(t[i] - top[i]), 0.0, float(b[i] - bot[i])]
        forms.append(np.round(d, 2).tolist())
    mask = add_part(f"目の形{k}", "頭", order, mesh=(verts, tris), keys=keys, forms=forms, maskOnly=True)

    add_part(f"白目{k}", "頭", order + 1, cell=P(4), masks=[mask["name"]])
    # 瞳: 視線で動く
    ir = add_part(f"瞳{k}", "頭", order + 2, cell=P(4), masks=[mask["name"]])
    ew = xb - xa
    ir["keys"] = [{"param": "ParamEyeBallX", "values": [-1, 0, 1]}, {"param": "ParamEyeBallY", "values": [-1, 0, 1]},
                  {"param": "ParamEyeBallForm", "values": [-1, 0, 1]}]
    ir["forms"] = []
    icx, icy = P(CFG["目"][k]["瞳の中心"])
    for c in combos(ir["keys"]):
        dx, dy = c["ParamEyeBallX"] * ew * 0.17, -c["ParamEyeBallY"] * hc * 0.18
        sc = {-1: 0.8, 0: 1.0, 1: 1.1}[c["ParamEyeBallForm"]]   # 驚いたときは瞳が小さくなる(参照シートの驚き顔に合わせて控えめに)
        f = []
        for (vx, vy) in ir["verts"]:
            f += [round(dx + (vx - icx) * (sc - 1), 2), round(dy + (vy - icy) * (sc - 1), 2)]
        ir["forms"].append(f)

    # 下まぶたの線: 動かさず、目を閉じるにつれて消す(笑い目のときも消す)
    if LAYERS[f"下まぶた{k}"][..., 3].max() > 0.01:
        low = add_part(f"下まぶた{k}", "頭", order + 3, cell=P(3))
        low["opacity"] = {"keys": [{"param": open_id, "values": [0, 0.35, 0.7]}, {"param": smile_id, "values": [0, 1]}],
                          "values": [0, 0, 1, 0, 0, 0.3]}

    # 上まつ毛: 各列で、まつ毛の下の縁を「まぶたの線」に合わせて動かし、閉じるほど薄くする
    lash = add_part(f"まつ毛{k}", "頭", order + 4, cell=P(1.5))
    lm = e["lash"]
    lcols = np.where(lm.any(axis=0))[0]
    lb = np.array([np.where(lm[:, x])[0].max() for x in lcols], float)   # まつ毛の下の縁
    # 目尻の縁の線のように下へ伸びる部分で下の縁が急に下がらないよう、前後9列の中央値にする
    lb = ndimage.median_filter(lb, size=9, mode="nearest")
    lash["keys"] = keys
    lash["forms"] = []
    for c in combos(keys):
        t, b = shape(c[open_id], c[smile_id], c["ParamEyeForm"])
        d = []
        for (vx, vy) in lash["verts"]:
            u = float(np.clip(vx, xs[0], xs[-1]))
            tv = float(np.interp(u, xs, t))
            if c[open_id] >= 1:
                # 開いた目: まぶたの動き(怒り目など)の分だけ動かす。形はそのまま
                d += [0.0, tv - float(np.interp(u, xs, top))]
            else:
                # 閉じた目: まつ毛の下の縁を、なめらかな閉じ線にぴったり合わせ、厚みを55%にする
                edge = float(np.interp(u, lcols, lb))   # 目の幅の外は端の列の値を使い、跳ねの形を保つ
                d += [0.0, (tv - edge) + (vy - edge) * (0.55 - 1)]
        lash["forms"].append(np.round(d, 2).tolist())

eye_parts("R", 40)
eye_parts("L", 45)

# 口
MOUTH_KEYS = [{"param": "ParamMouthOpenY", "values": [0, 1]}, {"param": "ParamMouthForm", "values": [-1, 0, 1]}]

def mouth_forms(part, fn):
    part["keys"] = MOUTH_KEYS
    part["forms"] = []
    for c in combos(MOUTH_KEYS):
        d = []
        for (vx, vy) in part["verts"]:
            dx, dy = fn(c["ParamMouthOpenY"], c["ParamMouthForm"], vx, vy)
            d += [round(dx, 2), round(dy, 2)]
        part["forms"].append(d)

def corner_lift(form, vx):
    # 口角の上げ下げ(両端ほど大きく、中央は逆に少し動く)
    u = (vx - mcx) / (mw / 2)
    return -form * P(2.2) * (u * u) + form * P(0.6) * (1 - u * u)

OPEN_STEPS = [0, 0.33, 0.67, 1]
MOUTH_GRID = [{"param": "ParamMouthOpenY", "values": OPEN_STEPS}, {"param": "ParamMouthForm", "values": [-1, 0, 1]}]
# 開き(4段階)×形(への字・普通・笑顔)の各点で表示する口。差分に無い形は近い形で代用する
MOUTH_ROWS = {-1: ["閉じ", "お小", "お大", "大開き"],
              0: ["閉じ", "え", "中開き", "大開き"],
              1: ["笑顔", "あ", "あ", "あ"]}
MOUTH_ROWS_FALLBACK = {"え": "お小", "中開き": "お大", "大開き": "お大"}
# 各形が「ちょうどの大きさ」で見える開きの値(これより小さい開きでは縦に縮めて見せる)
MOUTH_MAIN_OPEN = {"お小": 0.33, "え": 0.33, "お大": 0.67, "中開き": 0.67, "あ": 0.67, "大開き": 1.0}


def add_hires_part(name, parent, order, cell):
    """テクスチャだけ高解像度で持つ部品(メッシュは全身画像の座標)"""
    v = HIRES[name]
    rx0, ry0, rx1, ry1 = v["rect"]
    verts, tris = grid_mesh(LAYERS[name], (rx0, ry0, rx1, ry1), cell)
    PARTS.append(dict(name=name, parent=parent, order=order, bbox=(rx0, ry0, rx1, ry1), verts=verts, tris=tris, hires=True))
    return PARTS[-1]


if USE_MOUTH_IMAGES:
    _mcx, _mcy = _minfo["center"]
    _hw = P(18)
    _names = [n for n in ["閉じ", "笑顔", "お小", "お大", "あ", "え", "中開き", "大開き"] if f"口_{n}" in HIRES]
    MOUTH_TABLE = {}
    for fm, row in MOUTH_ROWS.items():
        for op, n in zip(OPEN_STEPS, row):
            MOUTH_TABLE[(fm, op)] = n if f"口_{n}" in HIRES else MOUTH_ROWS_FALLBACK.get(n, n)
    # 形ごとの口の上端(縦に縮めるときの基準)
    _mouth_top = {}
    for n in _names:
        a = LAYERS[f"口_{n}"][..., 3]
        band = a[:, int(_mcx - P(4)):int(_mcx + P(4))] > 0.4
        rows = np.where(band.any(axis=1))[0]
        _mouth_top[n] = float(rows.min()) if len(rows) else _mcy

    def _mouth_shape(name, op, fm, vx, vy):
        """形ごとの変形: 開きかけは口の上端を基準に縦に縮め、への字は閉じ口の口角を下げる"""
        dy = 0.0
        if name in MOUTH_MAIN_OPEN:
            squash = float(np.clip(1 + (op - MOUTH_MAIN_OPEN[name]) * 1.0, 0.3, 1.12))
            top = _mouth_top[name]
            dy = (top + (vy - top) * squash) - vy
        u = np.clip((vx - _mcx) / _hw, -1, 1)
        if name == "閉じ" and fm < 0:
            dy += -fm * P(1.3) * u * u
        return 0.0, dy

    for i, name in enumerate(_names):
        part = add_hires_part(f"口_{name}", "頭", 50 + i, cell=P(2))
        part["keys"] = MOUTH_GRID
        part["forms"] = []
        vals = []
        for c in combos(MOUTH_GRID):
            op, fm = c["ParamMouthOpenY"], c["ParamMouthForm"]
            vals.append(1 if MOUTH_TABLE[(fm, op)] == name else 0)
            d = []
            for (vx, vy) in part["verts"]:
                dx, dy = _mouth_shape(name, op, fm, vx, vy)
                d += [round(float(dx), 2), round(float(dy), 2)]
            part["forms"].append(d)
        # 同じ組の口は、重みを鋭くして(ほぼ切り替え)重ねる。表示エンジンが不透明度を正しく混ぜ直す
        part["opacity"] = {"keys": MOUTH_GRID, "values": vals}
        part["blendGroup"] = "口"
        part["blendSharpen"] = 3

def add_drawn_mouth():
    """口の差分画像が無いときの、描き起こしの口"""
    line_y = mcy
    inner = add_part("口の中", "頭", 50, cell=P(3))
    mouth_forms(inner, lambda op, fm, vx, vy: (
        0.0,
        (line_y + (vy - line_y) * max(op, 0.02)) - vy + corner_lift(fm, vx) * 0.8,
    ))
    inner["opacity"] = {"keys": [{"param": "ParamMouthOpenY", "values": [0, 0.12, 1]}], "values": [0, 1, 1]}
    iy1 = INNER_BOX[3]
    lower = add_part("下唇", "頭", 51, cell=P(3))
    mouth_forms(lower, lambda op, fm, vx, vy: (0.0, (iy1 - line_y) * (op - 1) + corner_lift(fm, vx) * 0.5))
    lower["opacity"] = {"keys": [{"param": "ParamMouthOpenY", "values": [0, 0.25, 1]}], "values": [0, 1, 1]}
    upper = add_part("口の線", "頭", 52, cell=P(3))
    mouth_forms(upper, lambda op, fm, vx, vy: (0.0, -op * P(1.2) + corner_lift(fm, vx)))


if not USE_MOUTH_IMAGES:
    add_drawn_mouth()

for i, s in enumerate("RL"):
    ch = add_part(f"頬{s}", "頭", 60 + i, cell=P(6))
    ch["opacity"] = {"keys": [{"param": "ParamCheek", "values": [0, 1]}], "values": [0, 1]}

# 涙: 目尻の下に出て、頬を伝って落ちる
def transform_forms(part, keys, fn):
    """keys の全組み合わせで、fn(組み合わせ, x, y) → (dx, dy) の変形を作る"""
    part["keys"] = keys
    part["forms"] = []
    for c in combos(keys):
        d = []
        for (vx, vy) in part["verts"]:
            dx, dy = fn(c, vx, vy)
            d += [round(float(dx), 2), round(float(dy), 2)]
        part["forms"].append(d)


for i, sd in enumerate("RL"):
    if f"涙{sd}" in LAYERS:
        tp = add_part(f"涙{sd}", "頭", 62 + i, cell=P(3))
        fall = P(MAT["涙"]["落ちる距離"])
        tk = [{"param": "ParamTear", "values": [0, 0.3, 1]}]
        tp["opacity"] = {"keys": tk, "values": [0, 1, 0.85]}
        cx_, cy_, w_, h_ = MAT_INFO[f"涙{sd}"]
        transform_forms(tp, tk, lambda c, vx, vy: (0.0, {0: -P(2), 0.3: 0.0, 1: fall}[c["ParamTear"]]
                                                    + (vy - cy_) * {0: -0.4, 0.3: 0.0, 1: 0.25}[c["ParamTear"]]))

# 眼鏡: 目の上、前髪の下に描く(前髪の毛先が眼鏡の上にかかる)
for key, order in (("眼鏡_黒", 64), ("眼鏡_赤", 65)):
    if key in LAYERS:
        gp = add_part(key, "頭", order, cell=P(4))
        gp["opacity"] = {"keys": [{"param": MAT[key]["パラメータ"], "values": [0, 1]}], "values": [0, 1]}

add_part("前髪", "前髪の揺れ", 70, cell=P(6))

# 眉: 元の絵の眉(前髪の上に透けて描かれている)。上下と傾きで表情を付ける
for i, sd in enumerate("RL"):
    if f"眉{sd}" in LAYERS:
        bp = add_part(f"眉{sd}", "前髪の奥行き", 72 + i, cell=P(3))
        cx_, cy_, w_, h_ = MAT_INFO[f"眉{sd}"]
        sign = 1 if sd == "R" else -1          # 右眉(画面の左)は目頭側が右
        bk = [{"param": "ParamBrowY", "values": [-1, 0, 1]}, {"param": "ParamBrowAngle", "values": [-1, 0, 1]}]

        def brow_fn(c, vx, vy, cx_=cx_, cy_=cy_, sign=sign):
            th = np.radians(12 * c["ParamBrowAngle"] * sign)    # 1: 怒り(目頭側が下がる)、-1: 困り(目頭側が上がる)
            dx, dy = vx - cx_, vy - cy_
            rx, ry = dx * np.cos(th) - dy * np.sin(th), dx * np.sin(th) + dy * np.cos(th)
            return rx - dx, ry - dy - c["ParamBrowY"] * P(5)
        transform_forms(bp, bk, brow_fn)
        bp["opacity"] = {"keys": [{"param": "ParamBrowVisible", "values": [0, 1]}], "values": [0, 1]}

# 感情マーク・汗: 頭の外側に描く
if "怒りマーク" in LAYERS:
    ap = add_part("怒りマーク", "頭", 80, cell=P(6))
    cx_, cy_, w_, h_ = MAT_INFO["怒りマーク"]
    ak = [{"param": "ParamAnger", "values": [0, 0.5, 1]}]
    ap["opacity"] = {"keys": ak, "values": [0, 1, 1]}
    transform_forms(ap, ak, lambda c, vx, vy: ((vx - cx_) * ({0: 0.6, 0.5: 0.92, 1: 1.12}[c["ParamAnger"]] - 1),
                                               (vy - cy_) * ({0: 0.6, 0.5: 0.92, 1: 1.12}[c["ParamAnger"]] - 1)))
if "汗" in LAYERS:
    sp = add_part("汗", "頭", 81, cell=P(3))
    sk = [{"param": "ParamSweat", "values": [0, 0.4, 1]}]
    sp["opacity"] = {"keys": sk, "values": [0, 1, 1]}
    transform_forms(sp, sk, lambda c, vx, vy: (0.0, {0: -P(3), 0.4: 0.0, 1: P(9)}[c["ParamSweat"]]))


# ============================================================ 7. テクスチャにまとめて書き出す

def pack(parts, max_w=4096):
    """棚詰めで1枚に並べる"""
    order = sorted(parts, key=lambda p: -(p["bbox"][3] - p["bbox"][1]) * (HIRES[p["name"]]["T"] if p.get("hires") else 1))
    x = y = shelf = 0
    pad = 4
    for p in order:
        k = HIRES[p["name"]]["T"] if p.get("hires") else 1
        w = (p["bbox"][2] - p["bbox"][0]) * k
        h = (p["bbox"][3] - p["bbox"][1]) * k
        if x + w + pad > max_w:
            x, y, shelf = 0, y + shelf + pad, 0
        p["atlas"] = (x, y)
        x += w + pad
        shelf = max(shelf, h)
    total_h = y + shelf
    size = 1
    while size < max(total_h, 16):
        size *= 2
    return max_w, size


visible = [p for p in PARTS]
aw, ah = pack(visible)
if ah > 4096:
    sys.exit(f"テクスチャが大きすぎます({aw}×{ah})。元画像を小さくしてください。")
atlas = np.zeros((ah, aw, 4), np.float32)
for p in visible:
    x0, y0, x1, y1 = p["bbox"]
    ax, ay = p["atlas"]
    if p.get("hires"):
        hi = HIRES[p["name"]]["hi"]
        atlas[ay:ay + hi.shape[0], ax:ax + hi.shape[1]] = hi
    else:
        atlas[ay:ay + (y1 - y0), ax:ax + (x1 - x0)] = LAYERS[p["name"]][y0:y1, x0:x1]
Image.fromarray((np.clip(atlas, 0, 1) * 255).round().astype(np.uint8), "RGBA").save(os.path.join(OUT, "texture.png"), optimize=True)

PARAMS = [
    ("ParamAngleX", "顔の向き 左右", -30, 30, 0), ("ParamAngleY", "顔の向き 上下", -30, 30, 0),
    ("ParamAngleZ", "頭の傾き", -30, 30, 0),
    ("ParamEyeLOpen", "左目の開き", 0, 1, 1), ("ParamEyeLSmile", "左目の笑い", 0, 1, 0),
    ("ParamEyeROpen", "右目の開き", 0, 1, 1), ("ParamEyeRSmile", "右目の笑い", 0, 1, 0),
    ("ParamEyeBallX", "視線 左右", -1, 1, 0), ("ParamEyeBallY", "視線 上下", -1, 1, 0),
    ("ParamEyeForm", "目の形(困り目↔怒り目)", -1, 1, 0), ("ParamEyeBallForm", "瞳の大きさ", -1, 1, 0),
    ("ParamMouthOpenY", "口の開き", 0, 1, 0), ("ParamMouthForm", "口の形(笑顔↔への字)", -1, 1, 0),
    ("ParamCheek", "頬の赤み", 0, 1, 0),
    ("ParamBrowVisible", "眉を表示", 0, 1, 1), ("ParamBrowY", "眉の上下", -1, 1, 0),
    ("ParamBrowAngle", "眉の傾き(困り↔怒り)", -1, 1, 0),
    ("ParamAnger", "怒りマーク", 0, 1, 0), ("ParamSweat", "汗", 0, 1, 0), ("ParamTear", "涙", 0, 1, 0),
    ("ParamGlasses", "眼鏡", 0, 1, 0),
    ("ParamBodyAngleX", "体の傾き 左右", -10, 10, 0), ("ParamBodyAngleZ", "体の回転", -10, 10, 0),
    ("ParamBreath", "呼吸", 0, 1, 0),
    ("ParamHairFront", "前髪の揺れ", -1, 1, 0), ("ParamHairSide", "前の房の揺れ", -1, 1, 0),
    ("ParamHairBack", "後ろ髪の揺れ", -1, 1, 0),
]

drawables = []
for p in PARTS:
    x0, y0, _, _ = p["bbox"]
    ax, ay = p["atlas"]
    pos, uvs = [], []
    for (vx, vy) in p["verts"]:
        pos += [round(vx, 2), round(vy, 2)]
        k = HIRES[p["name"]]["T"] if p.get("hires") else 1
        uvs += [round((ax + (vx - x0) * k) / aw, 6), round((ay + (vy - y0) * k) / ah, 6)]
    d = dict(id=p["name"], parent=p["parent"], order=p["order"], texture=0,
             positions=pos, uvs=uvs, indices=p["tris"])
    for key in ("keys", "forms", "opacity", "masks", "maskOnly", "blendGroup", "blendSharpen"):
        if key in p:
            d[key] = p[key]
    drawables.append(d)

physics = [
    dict(id="前髪", output="ParamHairFront", inputs=[{"param": "ParamAngleX", "weight": 0.6}, {"param": "ParamAngleZ", "weight": 0.5},
                                                     {"param": "ParamBodyAngleX", "weight": 0.3}],
         stiffness=110, damping=9, scale=2.4, wind=0.06, windSpeed=1.7, phase=0.0),
    dict(id="前の房", output="ParamHairSide", inputs=[{"param": "ParamAngleX", "weight": 0.5}, {"param": "ParamAngleZ", "weight": 0.5},
                                                     {"param": "ParamBodyAngleX", "weight": 0.5}, {"param": "ParamBodyAngleZ", "weight": 0.4}],
         stiffness=55, damping=5.5, scale=2.8, wind=0.08, windSpeed=1.1, phase=1.3),
    dict(id="後ろ髪", output="ParamHairBack", inputs=[{"param": "ParamAngleX", "weight": 0.5}, {"param": "ParamAngleZ", "weight": 0.5},
                                                     {"param": "ParamBodyAngleX", "weight": 0.6}, {"param": "ParamBodyAngleZ", "weight": 0.5}],
         stiffness=35, damping=4.5, scale=2.8, wind=0.1, windSpeed=0.8, phase=2.1),
]

model = dict(
    version=1, name=CFG["name"],
    canvas=dict(width=W, height=H),
    anchors=dict(face=[round(HCX, 1), round(HCY, 1)], head=[round(v, 1) for v in P(head["変形の範囲"])]),
    textures=["texture.png"],
    parameters=[dict(id=i, name=n, min=a, max=b, default=d) for i, n, a, b, d in PARAMS],
    deformers=DEFORMERS,
    drawables=drawables,
    physics=physics,
)
with open(os.path.join(OUT, "puppet.json"), "w", encoding="utf-8") as fh:
    json.dump(model, fh, ensure_ascii=False, separators=(",", ":"),
              default=lambda o: o.item() if hasattr(o, "item") else float(o))  # numpyの数値を普通の数値へ
nv = sum(len(p["verts"]) for p in PARTS)
print(f"書き出し: {OUT}/puppet.json(部品 {len(PARTS)}、頂点 {nv}、変形器 {len(DEFORMERS)})")
print(f"書き出し: {OUT}/texture.png({aw}×{ah})")
