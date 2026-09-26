"""元の絵に描かれている眉(前髪の上に透けて描かれた細い線)を取り出す。

・眉は「すぐ上とすぐ下より暗い、横に細い線」として見つける
・前髪の毛の線(縦に走る暗い線)を拾わないよう、列ごとの線の中心から2次曲線を当てはめ、
  その曲線のすぐ近くの画素だけを眉とする
・元の絵からは、各列で眉の上下の色を直線でつないで消す(前髪の毛並みがそのまま続く)
"""
import numpy as np
from scipy import ndimage


def extract(rgb, V, S, boxes):
    """boxes: [(x0, y0, x1, y1), ...](実際の画素)。
    戻り値: (眉を消した画像, [眉ごとの不透明度], [眉ごとの(中心x, 中心y, 幅)])"""
    H, W = V.shape
    out = rgb.copy()
    alphas, infos = [], []
    off = max(2, int(round(3 * S)))
    yy = np.mgrid[0:H, 0:W][0]
    for (x0, y0, x1, y1) in boxes:
        up = V[y0 - off:y1 - off, x0:x1]
        dn = V[y0 + off:y1 + off, x0:x1]
        c = V[y0:y1, x0:x1]
        d = np.clip(np.minimum(up, dn) - c, 0, None)
        a = np.clip((d - 12) / 50, 0, 1)
        # 列ごとの線の中心(暗さで重み付け)→ 2次曲線を当てはめる
        xs, ys, ws = [], [], []
        for i in range(a.shape[1]):
            col = a[:, i]
            if col.max() > 0.3:
                w_ = col ** 2
                xs.append(i); ys.append(float((np.arange(len(col)) * w_).sum() / w_.sum())); ws.append(col.max())
        if len(xs) < 5:
            alphas.append(np.zeros((H, W), np.float32)); infos.append(None)
            continue
        xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)
        coef = np.polyfit(xs, ys, 2, w=ws)
        # 当てはめから大きく外れた列(縦の毛の線)を除いてもう一度
        res = np.abs(np.polyval(coef, xs) - ys)
        ok = res < 1.5 * S
        coef = np.polyfit(xs[ok], ys[ok], 2, w=ws[ok])
        curve = np.polyval(coef, np.arange(a.shape[1]))
        near = np.abs(np.arange(a.shape[0])[:, None] - curve[None, :]) <= 1.6 * S
        a = a * near
        # 曲線に沿って眉のある範囲(左右の端)だけ残す
        colmax = a.max(axis=0)
        on = np.where(colmax > 0.25)[0]
        if len(on):
            a[:, :max(on.min() - 1, 0)] = 0
            a[:, on.max() + 2:] = 0
        full = np.zeros((H, W), np.float32)
        full[y0:y1, x0:x1] = a
        alphas.append(full)
        xa, xb = on.min() + x0, on.max() + x0
        infos.append(((xa + xb) / 2, float(np.polyval(coef, (xa + xb) / 2 - x0)) + y0, xb - xa))
        # 元の絵から消す: 各列で上下の色を直線でつなぐ
        reg = ndimage.binary_dilation(full > 0.05, iterations=max(1, int(S)))
        for x in np.unique(np.where(reg)[1]):
            col = np.where(reg[:, x])[0]
            for seg in np.split(col, np.where(np.diff(col) > 1)[0] + 1):
                a0, b0 = seg.min() - 1, seg.max() + 1
                t = (np.arange(a0 + 1, b0) - a0) / (b0 - a0)
                out[a0 + 1:b0, x] = rgb[a0, x][None] * (1 - t[:, None]) + rgb[b0, x][None] * t[:, None]
    return out, alphas, infos
