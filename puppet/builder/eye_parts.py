"""目を部品に分ける(上まつ毛・下まぶたの線・白目・瞳・目の形)。

考え方:
  ・「開いている部分」= 白目と瞳(青み・色の薄い明るい画素)の塊
  ・「線」= 開いている部分のまわりの暗い画素。真っ黒なまつ毛から、つながっている暗い画素を
    一定の距離までたどって集める(目尻の跳ね・目頭の線は赤茶色で髪の色に近いので、色ではなく
    つながりで見分ける)
  ・線のうち、開いている部分の中心線より上 = 上まつ毛、下 = 下まぶたの線
  ・線の不透明度は暗さから決め(線の縁がなめらかになる)、色は背景の明るさを差し引いて逆算する
"""
import cv2
import numpy as np
from scipy import ndimage


def extract(rgb, alpha, hsv, box, iris_center, iris_radius, S, exclude, corner_reach=1.0, join_parts=False, hint=None):
    """rgb: 0〜1、hsv: HSV_FULL(0〜255)、box: (x0,y0,x1,y1)、exclude: 目の部品にしない画素(前髪など)
    corner_reach: 目頭・目尻の外へまつ毛をたどる距離の倍率(斜め向きの手前の目など、目尻が長い絵で大きくする)
    join_parts: 細い髪の毛が白目を横切って、白目が分かれている絵のとき True(分かれた白目もつなぐ)
    hint: 開いている部分として必ず含める範囲(影で色が肌に近い白目など、色で見分けられないとき)"""
    H, W = alpha.shape
    V, Sat = hsv[..., 2].astype(np.float32), hsv[..., 1].astype(np.float32)
    x0, y0, x1, y1 = box
    pad = int(8 * S)
    bx0, by0, bx1, by1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
    inbox = np.zeros((H, W), bool)
    inbox[by0:by1, bx0:bx1] = True
    core_box = np.zeros((H, W), bool)
    core_box[y0:y1, x0:x1] = True

    # 開いている部分: 青みがある(白目・瞳)か、ほぼ無彩色の明るい画素(白目の陰)。肌は彩度40前後なので入らない
    r, b = rgb[..., 0] * 255, rgb[..., 2] * 255
    bluish = ((b - r) > -14) & (V > 60)
    pale = (Sat < 30) & (V > 150)
    opening = core_box & (bluish | pale) & ~exclude
    opening = ndimage.binary_opening(opening, iterations=1)
    if hint is not None:
        opening |= hint & ~exclude
    cx, cy = iris_center
    yy, xx = np.mgrid[0:H, 0:W]
    iris_ell = ((xx - cx) / (iris_radius[0] * 1.15)) ** 2 + ((yy - cy) / (iris_radius[1] * 1.15)) ** 2 <= 1
    # 瞳の中の暗い画素(瞳孔・縁取り)も開いている部分に含める
    opening |= iris_ell & (V < 175) & ndimage.binary_dilation(opening, iterations=int(2 * S))
    opening = ndimage.binary_fill_holes(ndimage.binary_closing(opening, iterations=1))
    lab, n = ndimage.label(opening)
    if n > 1:
        sizes = ndimage.sum(opening, lab, range(1, n + 1))
        if join_parts:
            # ある程度の大きさの塊はすべて残し、髪の毛1本ぶんのすき間を埋めてつなぐ
            opening = np.isin(lab, [i + 1 for i, s_ in enumerate(sizes) if s_ >= 0.08 * sizes.max()])
            opening = ndimage.binary_fill_holes(ndimage.binary_closing(opening, iterations=max(1, int(2 * S)))) & core_box
            lab, n = ndimage.label(opening)
            if n > 1:
                sizes = ndimage.sum(opening, lab, range(1, n + 1))
                opening = lab == (np.argmax(sizes) + 1)
        else:
            opening = lab == (np.argmax(sizes) + 1)

    cols = np.where(opening.any(axis=0))[0]
    top = np.full(W, np.nan)
    bot = np.full(W, np.nan)
    for x in cols:
        ys = np.where(opening[:, x])[0]
        top[x], bot[x] = ys.min(), ys.max()
    xa, xb = int(cols.min()), int(cols.max())
    ya, yb = (top[xa] + bot[xa]) / 2, (top[xb] + bot[xb]) / 2

    # 上まつ毛: 真っ黒な画素を種に、暗い画素をたどる。ただし「目の上の縁に沿った帯」と
    # 「目頭・目尻のまわり」の中だけ(前髪の毛先の暗い線に広がらないように)
    top_full = np.interp(np.arange(W), cols, top[cols])
    band = (yy >= top_full[None, :] - 8 * S) & (yy <= top_full[None, :] + 3 * S) & (xx >= xa) & (xx <= xb)
    for (qx, qy) in ((xa, ya), (xb, yb)):
        band |= ((xx - qx) ** 2 + (yy - qy) ** 2 <= (11 * S * corner_reach) ** 2) & (yy >= qy - 8 * S)
    band &= inbox & ~exclude
    black = band & (V < 70)
    dark = band & (V < 175)
    seeds = black & ndimage.binary_dilation(opening, iterations=int(3 * S))
    ink = seeds.copy()
    for _ in range(int(12 * S)):
        grown = ndimage.binary_dilation(ink) & dark
        if (grown == ink).all():
            break
        ink = grown
    # 開いている部分に食い込んだ真っ黒な画素(まつ毛の下の縁)は、まつ毛の側に入れる
    ink &= ~opening | (V < 70)
    # 目の範囲の列では、目の上の縁から上へ続く暗い画素を「まつ毛の厚み」ぶんだけ残す
    # (前髪の毛先の暗い線が、まつ毛から上へ伸びて見えるのを切る)
    runs = {}
    for x in range(xa, xb + 1):
        y = int(top_full[x]) + int(2 * S)
        while y > 0 and not ink[y, x] and y > top_full[x] - 3 * S:
            y -= 1
        y_bottom = y
        while y > 0 and ink[y, x]:
            y -= 1
        if ink[y_bottom, x]:
            runs[x] = (y + 1, y_bottom)
    lengths = [b_ - a_ + 1 for a_, b_ in runs.values()]
    lmax = np.median(lengths) * 1.5 if lengths else 6 * S
    keep = np.zeros_like(ink)
    for x, (a_, b_) in runs.items():
        keep[max(a_, int(b_ - lmax + 1)):b_ + 1, x] = True
    # 目頭・目尻の外側: 目のすぐそば(5S以内)か、目の端より上(外へ跳ねるまつ毛)だけ
    dist_open = ndimage.distance_transform_edt(~opening)
    outside_cols = (xx < xa) | (xx > xb)
    corner_ok = outside_cols & ink & (dist_open <= 9 * S * corner_reach) & (yy <= np.where(xx < xa, ya, yb) + 2 * S)
    upper = keep | corner_ok
    # 残した画素につながっていない小さな破片は捨てる
    lab, n = ndimage.label(ndimage.binary_dilation(upper))
    if n > 1:
        sizes = ndimage.sum(upper, lab, range(1, n + 1))
        upper &= np.isin(lab, [i + 1 for i, s_ in enumerate(sizes) if s_ >= 0.15 * max(sizes)])

    # 下まぶたの線と目尻の縁の線: 開いている部分のすぐ外(3S以内)で、上まつ毛より下にある暗めの画素。
    # 目を閉じるときは消す(閉じた目は上まつ毛の線だけにする)
    mid_full = np.interp(np.arange(W), cols, ((top + bot) / 2)[cols])
    near = ndimage.binary_dilation(opening, iterations=int(3 * S)) & ~opening
    lower = near & ~ndimage.binary_dilation(upper, iterations=1) & (V < 228) & (yy > mid_full[None, :]) & ~exclude
    lab, n = ndimage.label(lower)
    if n > 1:
        sizes = ndimage.sum(lower, lab, range(1, n + 1))
        lower &= np.isin(lab, [i + 1 for i, s_ in enumerate(sizes) if s_ >= 3 * S * S])
    # まつ毛に入れた画素を開いている部分から外し、上端・下端を求め直す
    opening = opening & ~upper
    lab, n = ndimage.label(opening)
    if n > 1:
        sizes = ndimage.sum(opening, lab, range(1, n + 1))
        opening = lab == (np.argmax(sizes) + 1)
    cols = np.where(opening.any(axis=0))[0]
    top = np.full(W, np.nan)
    bot = np.full(W, np.nan)
    for x in cols:
        ys = np.where(opening[:, x])[0]
        top[x], bot[x] = ys.min(), ys.max()
    return dict(opening=opening, upper=upper, lower=lower, ink=ink, iris_ell=iris_ell, cols=cols,
                top=top, bot=bot, corners=((xa, ya), (xb, yb)))


def line_layer(rgb, V, mask, S, bg_level=245.0, ink_level=25.0):
    """線の部品: 不透明度は暗さから、色は背景(明るい肌・白目)を差し引いて逆算する"""
    region = ndimage.binary_dilation(mask, iterations=max(1, int(0.8 * S)))
    a = np.clip((bg_level - V) / (bg_level - ink_level), 0, 1) * region
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 0.35 * S) * region
    bg = bg_level / 255.0
    col = (rgb - (1 - a[..., None]) * bg) / np.maximum(a[..., None], 0.05)
    col = np.clip(col, 0, 1)
    # 色が不安定な薄い縁は、線の内側の色で置き換える
    core = a > 0.6
    if core.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
        weak = a <= 0.6
        col[weak] = col[iy[weak], ix[weak]]
    return col, a


def erase_region(e, S, H, W):
    """顔から消す範囲: 開いている部分・上まつ毛・下まぶたの線の帯(開いている部分の下の縁から4Sまで)"""
    op, lash = e["opening"], e["upper"]
    region = ndimage.binary_dilation(op | lash, iterations=int(1.5 * S))
    cols = e["cols"]
    for x in cols:
        b_ = int(e["bot"][x])
        region[b_:min(b_ + int(4 * S), H), x] = True
    (xa, ya), (xb, yb) = e["corners"]
    # 目頭・目尻の外側も、端の高さのまわりを少し消す(目尻の縁の線)
    for x in range(max(xa - int(4 * S), 0), xa):
        region[int(ya) - int(2 * S):int(ya) + int(4 * S), x] |= True
    for x in range(xb + 1, min(xb + int(4 * S), W)):
        region[int(yb) - int(2 * S):int(yb) + int(4 * S), x] |= True
    return ndimage.binary_closing(region, iterations=int(S))


def fill_vertical(rgb, region, known, S):
    """region を、各列で「すぐ上の既知の画素」と「すぐ下の既知の画素」の色を直線でつないで塗る。
    まぶたの陰のような縦の濃淡がそのまま続く。最後に横方向になじませる"""
    out = rgb.copy()
    H, W = region.shape
    ys, xs = np.where(region)
    if len(xs) == 0:
        return out
    filled = np.zeros((H, W), bool)
    for x in np.unique(xs):
        col = np.where(region[:, x])[0]
        # 塗る範囲を連続した区間ごとに処理する
        splits = np.where(np.diff(col) > 1)[0] + 1
        for seg in np.split(col, splits):
            y0, y1 = seg.min(), seg.max()
            ya = y0 - 1
            while ya >= 0 and not known[ya, x] and y0 - ya < 12 * S:
                ya -= 1
            yb = y1 + 1
            while yb < H and not known[yb, x] and yb - y1 < 12 * S:
                yb += 1
            ok_a = ya >= 0 and known[ya, x]
            ok_b = yb < H and known[yb, x]
            if not (ok_a or ok_b):
                continue
            ca = rgb[ya, x] if ok_a else rgb[yb, x]
            cb = rgb[yb, x] if ok_b else rgb[ya, x]
            t = (np.arange(y0, y1 + 1) - ya) / max(yb - ya, 1)
            out[y0:y1 + 1, x] = ca[None, :] * (1 - t[:, None]) + cb[None, :] * t[:, None]
            filled[y0:y1 + 1, x] = True
    # 横方向になじませる(列ごとの塗りの筋を消す)
    k = int(2 * S) * 2 + 1
    blur = cv2.GaussianBlur(out, (k, 1), 1.5 * S)
    wgt = cv2.GaussianBlur(filled.astype(np.float32), (k, 1), 1.5 * S)
    sm = cv2.GaussianBlur(out * filled[..., None], (k, 1), 1.5 * S) / np.maximum(wgt, 1e-4)[..., None]
    out = np.where(filled[..., None], sm, out)
    return out, filled


def diff_layer(orig, base, region, strength=0.22, darker_only=False):
    """元の絵と、塗り足した下地との「差」だけを取り出した部品。下地の上に重ねると元の絵どおりに見える。
    不透明度は差の大きさから決め、色は下地を差し引いて逆算する。
    darker_only: 下地より暗い画素だけを取る(線だけを取り、白目のにじみなどの明るい画素を拾わない)"""
    if darker_only:
        lum = lambda c: c[..., 0] * 0.3 + c[..., 1] * 0.55 + c[..., 2] * 0.15
        d = np.clip(lum(base) - lum(orig), 0, None) * 1.4
    else:
        d = np.abs(orig - base).max(axis=2)
    a = np.clip(d / strength, 0, 1) * region
    col = base + (orig - base) / np.maximum(a[..., None], 1e-3)
    col = np.clip(col, 0, 1)
    col = np.where(a[..., None] > 1e-3, col, orig)
    return col, a
