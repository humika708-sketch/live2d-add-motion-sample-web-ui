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


EYES = {k: extract_eye(k) for k in ("R", "L")}

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
a_back = np.where(BACK, ALPHA * np.clip(soft(BACK, 0.5) * 1.5, 0, 1), 0)
a_back = np.maximum(a_back, under_back.astype(np.float32))
layer("後ろ髪", decontaminate(rgb_back, a_back), a_back)

# --- 体: 前の房の下を塗り足し、首をあごの下へ延ばす
_lock_all = ndimage.binary_dilation(FRONT, iterations=int(6 * S)) & (ALPHA > 0.5)   # 房全体(毛先の薄い部分も含める)
under_body = _lock_all & inside_body & (YY >= y_start - P(10))
# 首の延長: 各列で「あごのすぐ下の首の色」を上へ伸ばす。
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
    y0 = ok[0]
    # あごのすぐ下は影で暗いので、少し下の明るい首の色を使う
    c = RGB[y0 + int(10 * S):y0 + int(24 * S), x].mean(axis=0)
    y_top = int(max(y_jaw - ext, 0))
    neck[y_top:y0, x] = True
    neck_rgb[y_top:y0, x] = c
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
neck_soft = np.clip(cv2.GaussianBlur(neck.astype(np.float32), (0, 0), 1.0 * S) * 1.6, 0, 1) * neck
a_body = np.maximum(a_body, np.maximum(under_body.astype(np.float32), neck_soft))
layer("体", rgb_body, a_body)

# --- 前の房
a_front = np.where(ndimage.binary_dilation(FRONT, iterations=1), ALPHA * np.clip(soft(FRONT, 0.5) * 1.6, 0, 1), 0)
layer("前の房", decontaminate(RGB, a_front), a_front)

# --- 顔: 前髪の下の肌、目と口を消した肌を塗り足す
eye_erase = np.zeros((H, W), bool)
for e in EYES.values():
    eye_erase |= ndimage.binary_dilation(e["opening"] | e["lash"], iterations=int(2 * S))
skin_like = (VAL > 190) & ((RGB[..., 0] - RGB[..., 2]) * 255 > 12)
skin_known = FACE & skin_like & ~eye_erase & ~MOUTH_BOX & (ALPHA > 0.9)
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
a_face = np.where(FACE, ALPHA, 0) * np.clip(1.0 - soft(HAIR & ~BANGS, 0.4), 0, 1)
a_face = np.maximum(a_face, face_fill.astype(np.float32))
layer("顔", rgb_face, a_face)

# --- 目の部品
for k, e in EYES.items():
    op = e["opening"]
    # 白目: 開いている部分から瞳を消して白で塗り、少し外側まで延ばす
    white_known = op & ~ndimage.binary_dilation(e["iris"], iterations=max(1, int(S)))
    grow = ndimage.binary_dilation(op, iterations=int(2 * S))
    rgb_w = diffuse_fill(RGB, white_known, (grow & ~white_known), iters=30)
    layer(f"白目{k}", rgb_w, grow.astype(np.float32))
    # 瞳: 楕円全体まで延ばす(まぶたの下に隠れていた部分も描いておく)
    iris_full = e["iris_ell"]
    rgb_i = diffuse_fill(RGB, e["iris"], iris_full & ~e["iris"], iters=30)
    a_i = np.clip(soft(iris_full, 0.5) * 1.3, 0, 1)
    layer(f"瞳{k}", rgb_i, a_i)
    # まつ毛
    a_l = np.clip(soft(e["lash"], 0.45) * 1.5, 0, 1) * ALPHA
    layer(f"まつ毛{k}", RGB, a_l)
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
YAW_MAX, PITCH_MAX, ROLL_MAX = 11.0, 8.0, 6.0   # 大きくすると首や髪の根元の塗り足しが見えやすくなる
PX, PY = P(head["首の回転軸"])

def head_warp_disp(ax, ay, xs, ys, depth_bias=0.0):
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
FOLLOW_START = P(bk["根元の高さ"]) - P(30)   # ここより上は頭と一緒に動く
FOLLOW_END = P(bk["根元の高さ"]) + P(150)    # ここより下は頭に付いていかない
_b1, _b2 = crop_bbox(LAYERS["後ろ髪"], pad=int(6 * S)), crop_bbox(LAYERS["前の房"], pad=int(6 * S))
fx0, fy0 = min(_b1[0], _b2[0]), min(_b1[1], _b2[1])
fx1 = max(_b1[2], _b2[2])
fy1 = max(_b1[3], _b2[3])
_cols = max(8, int((fx1 - fx0) / P(24)))
_rows = max(8, int((fy1 - fy0) / P(30)))

def follow(c, xs, ys):
    wgt = 1 - smoothstep(FOLLOW_START, FOLLOW_END, ys)
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
add_part("体", "体", 10, cell=P(20))
add_part("前の房", front_parent, 20, cell=P(10))
add_part("顔", "頭", 30, cell=P(8))

EYE_KEYS = [{"param": None, "values": [0, 1]}, {"param": None, "values": [0, 1]}]

def eye_parts(k, order):
    e = EYES[k]
    open_id = f"ParamEye{k}Open"
    smile_id = f"ParamEye{k}Smile"
    keys = [{"param": open_id, "values": [0, 1]}, {"param": smile_id, "values": [0, 1]}]
    cols = sorted(e["cols"])
    n = 17
    xs = np.linspace(cols[0] - 0.5, cols[-1] + 0.5, n)
    def edge(d, x):
        near = min(d.keys(), key=lambda c: abs(c - x))
        return float(d[near])
    top = np.array([edge(e["top"], x) - 0.5 for x in xs])
    bot = np.array([edge(e["bot"], x) + 0.5 for x in xs])
    # 列ごとのばらつきをならす(閉じたときの線がガタつかないように)
    k_ = np.array([1, 2, 3, 2, 1], float); k_ /= k_.sum()
    top = np.convolve(np.pad(top, 2, mode="edge"), k_, mode="valid")
    bot = np.convolve(np.pad(bot, 2, mode="edge"), k_, mode="valid")
    hgt = bot - top
    tt = np.linspace(0, 1, n)
    bump = np.sin(np.pi * tt)          # 中央ほど大きい
    arc_y = bot - hgt * 0.35 - hgt * 0.45 * bump   # 笑い目(^^)の弧

    def shape(op, sm):
        """目の開き・笑いの度合いに応じた上端・下端"""
        t_open = top
        b_open = bot - sm * hgt * 0.28 * bump            # 笑うと下まぶたが上がる
        close_line = bot - hgt * 0.30 + hgt * 0.08 * bump   # 閉じた目はゆるく下に弧を描く
        t_closed = close_line * (1 - sm) + arc_y * sm
        b_closed = t_closed + 0.3
        t = t_open * op + t_closed * (1 - op)
        b = b_open * op + b_closed * (1 - op)
        return t, np.maximum(b, t + 0.3)

    verts, tris = strip_mesh(xs, top, bot)
    forms = []
    for c in combos(keys):
        t, b = shape(c[open_id], c[smile_id])
        d = []
        for i in range(n):
            d += [0.0, float(t[i] - top[i]), 0.0, float(b[i] - bot[i])]
        forms.append(np.round(d, 2).tolist())
    mask = add_part(f"目の形{k}", "頭", order, mesh=(verts, tris), keys=keys, forms=forms, maskOnly=True)

    add_part(f"白目{k}", "頭", order + 1, cell=P(4), masks=[mask["name"]])
    # 瞳: 視線で動く
    ir = add_part(f"瞳{k}", "頭", order + 2, cell=P(4), masks=[mask["name"]])
    ew = cols[-1] - cols[0]
    ir["keys"] = [{"param": "ParamEyeBallX", "values": [-1, 0, 1]}, {"param": "ParamEyeBallY", "values": [-1, 0, 1]}]
    ir["forms"] = []
    for c in combos(ir["keys"]):
        dx, dy = c["ParamEyeBallX"] * ew * 0.17, -c["ParamEyeBallY"] * float(hgt.mean()) * 0.18
        ir["forms"].append([round(dx, 2), round(dy, 2)] * len(ir["verts"]))

    # まつ毛: 目の上端の動きに合わせて下がる
    lash = add_part(f"まつ毛{k}", "頭", order + 3, cell=P(2))
    lash["keys"] = keys
    lash["forms"] = []
    lx0, ly0, lx1, ly1 = lash["bbox"]
    for c in combos(keys):
        t, b = shape(c[open_id], c[smile_id])
        d = []
        for (vx, vy) in lash["verts"]:
            dt = float(np.interp(vx, xs, t - top))
            # 閉じるほど上下に薄くする(まつ毛の下端を基準に縮める)
            squash = 1 - 0.62 * (1 - c[open_id])
            vy2 = ly1 + (vy - ly1) * squash
            d += [0.0, dt + (vy2 - vy)]
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

MOUTH_GRID = [{"param": "ParamMouthOpenY", "values": [0, 0.5, 1]}, {"param": "ParamMouthForm", "values": [-1, 0, 1]}]
# 開き(0・0.5・1)×形(への字・普通・笑顔)の各点で表示する口
MOUTH_TABLE = {(-1, 0): "閉じ", (-1, 0.5): "お小", (-1, 1): "お大",
               (0, 0): "閉じ", (0, 0.5): "お小", (0, 1): "お大",
               (1, 0): "笑顔", (1, 0.5): "あ", (1, 1): "あ"}


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

    def _mouth_shape(name, op, fm, vx, vy):
        """形ごとの変形: 開きかけは上唇の線を基準に縦に縮め、への字は口角を下げる"""
        top = _mcy - P(1)
        squash = {"あ": {0: 0.3, 0.5: 0.62, 1: 1.0}, "お大": {0: 0.4, 0.5: 0.72, 1: 1.0},
                  "お小": {0: 0.55, 0.5: 1.0, 1: 1.15}}.get(name, {0: 1, 0.5: 1, 1: 1})[op]
        dy = (top + (vy - top) * squash) - vy
        u = np.clip((vx - _mcx) / _hw, -1, 1)
        if name == "閉じ":
            dy += -fm * P(1.3) * u * u if fm < 0 else 0.0
        return 0.0, dy

    for i, name in enumerate(["閉じ", "笑顔", "お小", "お大", "あ"]):
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

add_part("前髪", "前髪の揺れ", 70, cell=P(6))


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
    ("ParamMouthOpenY", "口の開き", 0, 1, 0), ("ParamMouthForm", "口の形(笑顔↔への字)", -1, 1, 0),
    ("ParamCheek", "頬の赤み", 0, 1, 0),
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
