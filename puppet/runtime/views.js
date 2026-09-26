// 向き違いのパペット(正面・斜め左・斜め右など)を重ねて持ち、切り替える。
//
// 向きごとに別の絵から組み立てたパペットを、同じ場所に重ねたキャンバスで描く。
// 切り替えるときは、今の向きの首をその方向へ回しながら薄め、新しい向きは少し戻した所から
// 首を正面へ戻しながら濃くする(絵の差し替えではなく、振り向いたように見せる)。
//
// 使い方:
//   const views = new PuppetViews(document.getElementById("stage"), [
//     { name: "正面", url: "../models/kurisu/puppet.json", yaw: 0 },
//     { name: "斜め右", url: "../models/kurisu_右/puppet.json", yaw: 1 },
//     { name: "斜め左", url: "../models/kurisu_左/puppet.json", yaw: -1 },
//   ]);
//   await views.load("正面");
//   await views.setView("斜め右");          // 振り向いて切り替える
//   views.all((p) => p.setExpression({ ParamMouthForm: 1 }));   // 読み込み済みの全部の向きに
//
// yaw: その絵の向き(-1 画面の左 〜 0 正面 〜 1 画面の右)。切り替えで首を回す向きと量に使う。
import { PuppetView } from "./puppet.js";

// 切り替えの途中(t: 0〜1)の状態。dir: 首を回す向き(1 右 / -1 左)、turn: 首を回す量。
// 前半: 今の向きのまま、その方向へ首と体を回す。中ほどの短い間だけ重ねて入れ替え(二重に見える時間を短くする)、
// 後半: 新しい向きが、少し戻した所から正面(その絵の向き)へ首を戻す。
// 新しい向きを上に重ねて濃くしてから、下の古い向きを消す(途中で背景が透けて見えないように)
export function turnState(t, dir, turn = 30) {
  const sm = (a, b) => { const x = Math.min(1, Math.max(0, (t - a) / (b - a))); return x * x * (3 - 2 * x); };
  const turnOut = sm(0, 0.55), turnIn = sm(0.4, 1);
  return {
    fromAngle: dir * turn * turnOut, fromBody: dir * 6 * turnOut,
    toAngle: -dir * turn * 0.6 * (1 - turnIn), toBody: -dir * 4 * (1 - turnIn),
    toOpacity: sm(0.38, 0.52), fromOpacity: 1 - sm(0.5, 0.62),
  };
}

export class PuppetViews {
  constructor(container, views, opts = {}) {
    this.container = container;
    this.defs = views;
    this.opts = { follow: "window", ...opts };
    this.entries = new Map();          // 名前 → { view, puppet, canvas }
    this.currentName = null;
    this.auto = null;                  // 全部の向きで共有する自動の動きの設定
    this.onChange = null;              // 向きが変わったときに呼ぶ関数(名前, パペット)
    this._busy = null;
    if (getComputedStyle(container).position === "static") container.style.position = "relative";
  }

  get puppet() { return this.entries.get(this.currentName)?.puppet; }
  get view() { return this.entries.get(this.currentName)?.view; }
  get names() { return this.defs.map((d) => d.name); }
  loaded() { return [...this.entries.values()].map((e) => e.puppet); }
  // 読み込み済みの全部の向きに同じ操作をする(表情・パラメータ・小物など)
  all(fn) { for (const p of this.loaded()) fn(p); }

  async _ensure(name) {
    if (this.entries.has(name)) return this.entries.get(name);
    const def = this.defs.find((d) => d.name === name);
    if (!def) throw new Error(`向き「${name}」はありません`);
    const canvas = document.createElement("canvas");
    Object.assign(canvas.style, { position: "absolute", inset: "0", width: "100%", height: "100%",
                                  display: "block", opacity: "0", transition: "none", touchAction: "none" });
    this.container.appendChild(canvas);
    const view = new PuppetView(canvas, { ...this.opts, focus: this._focusFor ? null : this.opts.focus });
    const puppet = await view.load(def.url);
    view.stop();
    // 自動の動き(まばたき・視線など)の設定は全部の向きで共有する
    if (this.auto) puppet.auto = this.auto; else this.auto = puppet.auto;
    // 今の向きの状態(利用者の値・表情・口パク)を引き継ぐ
    const cur = this.puppet;
    if (cur) this._copyState(cur, puppet);
    const e = { view, puppet, canvas, def };
    this.entries.set(name, e);
    if (this._focusFor) this._applyFocus(e);
    return e;
  }

  _copyState(from, to) {
    for (const [k, v] of from.userParams) if (to.paramInfo.has(k)) to.setParam(k, v);
    if (from._expr) {
      const target = {};
      for (const [k, v] of Object.entries(from._expr.to)) target[k] = v;
      to.setExpression(target, 0.01);
    }
    to.lipSync = from.lipSync;
    to.lookTarget = { ...from.lookTarget };
    to.look = { ...from.look };
  }

  // 表示範囲: 名前("all" / "bust" / "face")を渡すと、向きごとの頭の位置から計算する。配列ならそのまま
  setFocus(focus) {
    this._focusFor = focus;
    for (const e of this.entries.values()) this._applyFocus(e);
  }
  _applyFocus(e) {
    const f = this._focusFor;
    let rect = null;
    if (Array.isArray(f)) rect = f;
    else if (f && f !== "all") {
      const h = e.puppet.json.anchors.head, hw = h[2] - h[0], hh = h[3] - h[1];
      if (f === "bust") rect = [h[0] - hw * 0.45, h[1], hw * 1.9, hh * 1.7];
      if (f === "face") rect = [h[0] + hw * 0.1, h[1] + hh * 0.2, hw * 0.8, hh * 0.75];
    }
    e.view.opts.focus = rect;
    e.puppet.fit(0.04, rect);
  }

  async load(name) {
    const e = await this._ensure(name);
    this.currentName = name;
    e.canvas.style.opacity = "1";
    e.view.start();
    if (this.onChange) this.onChange(name, e.puppet);
    return e.puppet;
  }

  // 向きを切り替える。duration: かける秒数、turn: 切り替えの途中で首を回す量(ParamAngleX の値)
  async setView(name, { duration = 0.6, turn = 30 } = {}) {
    if (name === this.currentName) return this.puppet;
    if (this._busy) await this._busy;
    const run = async () => {
      const from = this.entries.get(this.currentName);
      const to = await this._ensure(name);
      if (!from) return this.load(name);
      this._copyState(from.puppet, to.puppet);
      // 首を回す向き: 新しい向きが今より右なら右へ
      const dir = Math.sign((to.def.yaw ?? 0) - (from.def.yaw ?? 0)) || 1;
      to.view.start();
      to.canvas.style.zIndex = "2"; from.canvas.style.zIndex = "1";
      const t0 = performance.now();
      await new Promise((resolve) => {
        const step = (now) => {
          const t = Math.min(1, (now - t0) / 1000 / duration);
          const st = turnState(t, dir, turn);
          from.puppet.setOffset("ParamAngleX", st.fromAngle);
          from.puppet.setOffset("ParamBodyAngleX", st.fromBody);
          to.puppet.setOffset("ParamAngleX", st.toAngle);
          to.puppet.setOffset("ParamBodyAngleX", st.toBody);
          to.canvas.style.opacity = String(st.toOpacity);
          from.canvas.style.opacity = String(st.fromOpacity);
          if (t < 1) requestAnimationFrame(step); else resolve();
        };
        requestAnimationFrame(step);
      });
      from.view.stop();
      from.canvas.style.opacity = "0";
      from.puppet.setOffset("ParamAngleX", 0); from.puppet.setOffset("ParamBodyAngleX", 0);
      to.puppet.setOffset("ParamAngleX", 0); to.puppet.setOffset("ParamBodyAngleX", 0);
      this.currentName = name;
      if (this.onChange) this.onChange(name, to.puppet);
      return to.puppet;
    };
    this._busy = run();
    try { return await this._busy; } finally { this._busy = null; }
  }
}
