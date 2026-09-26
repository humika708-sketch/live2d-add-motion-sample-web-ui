// 簡易ライブ2D風パペット 表示エンジン(外部ライブラリなし・ESモジュール)
//
// 仕組みは本家ライブ2Dと同じ考え方:
//   パラメータ → キーフォーム補間 → 変形器(ワープ・回転)の連鎖 → 部品(メッシュ)の頂点
// モデルは builder/build_rig.py が生成する puppet.json と、テクスチャ画像1枚で構成される。
// モーションはライブ2Dと同じ motion3.json 形式をそのまま再生できる。
//
// 最小の使い方:
//   import { PuppetView } from "./puppet.js";
//   const view = await PuppetView.create(canvas, "models/kurisu/puppet.json");
//   view.playMotion("models/kurisu/motions/smile.motion3.json");

// ------------------------------------------------------------------ 補助関数

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

// キーフォームの重み計算。
// keys: [{param, values:[昇順の値...]}, ...]
// 戻り値: [[キーフォーム番号, 重み], ...](合計1)。先頭の束縛が最も速く変わる並び順。
function keyformWeights(keys, params) {
  let result = [[0, 1]];
  let stride = 1;
  for (const k of keys) {
    const vals = k.values;
    const v = params.get(k.param) ?? 0;
    let i0 = 0, i1 = 0, t = 0;
    if (vals.length > 1) {
      if (v <= vals[0]) { i0 = i1 = 0; }
      else if (v >= vals[vals.length - 1]) { i0 = i1 = vals.length - 1; }
      else {
        let i = 0;
        while (v > vals[i + 1]) i++;
        i0 = i; i1 = i + 1; t = (v - vals[i]) / (vals[i + 1] - vals[i]);
      }
    }
    const next = [];
    for (const [idx, w] of result) {
      if (i0 === i1 || t === 0) next.push([idx + i0 * stride, w]);
      else {
        next.push([idx + i0 * stride, w * (1 - t)]);
        next.push([idx + i1 * stride, w * t]);
      }
    }
    result = next;
    stride *= vals.length;
  }
  return result;
}

// 複数キーフォーム(数値配列)の重み付き和
function blendArrays(forms, weights, out) {
  out.fill(0);
  for (const [idx, w] of weights) {
    const f = forms[idx];
    if (!f || w === 0) continue;
    for (let i = 0; i < out.length; i++) out[i] += f[i] * w;
  }
  return out;
}

// ------------------------------------------------------------------ 変形器

class WarpDeformer {
  // 格子状の制御点を動かして中の点を変形させる(本家のワープデフォーマ相当)
  constructor(def) {
    this.id = def.id;
    this.parentId = def.parent ?? null;
    [this.x, this.y, this.w, this.h] = def.rect;
    this.rows = def.rows;
    this.cols = def.cols;
    this.keys = def.keys || [];
    this.forms = def.forms || [];
    const n = (this.rows + 1) * (this.cols + 1) * 2;
    this.delta = new Float32Array(n);
    this.grid = new Float32Array(n);
  }
  update(params) {
    if (this.forms.length) blendArrays(this.forms, keyformWeights(this.keys, params), this.delta);
    let k = 0;
    for (let r = 0; r <= this.rows; r++) {
      for (let c = 0; c <= this.cols; c++) {
        this.grid[k] = this.x + (this.w * c) / this.cols + this.delta[k];
        this.grid[k + 1] = this.y + (this.h * r) / this.rows + this.delta[k + 1];
        k += 2;
      }
    }
  }
  // 静止状態の座標 (px,py) を変形後の座標へ写す(格子の外は端のマスで線形に延長)
  map(px, py, out) {
    const u = ((px - this.x) / this.w) * this.cols;
    const v = ((py - this.y) / this.h) * this.rows;
    const c = clamp(Math.floor(u), 0, this.cols - 1);
    const r = clamp(Math.floor(v), 0, this.rows - 1);
    const fu = u - c, fv = v - r;
    const stride = (this.cols + 1) * 2;
    const a = r * stride + c * 2, b = a + 2, d = a + stride, e = d + 2;
    const g = this.grid;
    const w00 = (1 - fu) * (1 - fv), w10 = fu * (1 - fv), w01 = (1 - fu) * fv, w11 = fu * fv;
    out[0] = g[a] * w00 + g[b] * w10 + g[d] * w01 + g[e] * w11;
    out[1] = g[a + 1] * w00 + g[b + 1] * w10 + g[d + 1] * w01 + g[e + 1] * w11;
  }
}

class RotationDeformer {
  // 基準点を中心に回転・移動・拡大する(本家の回転デフォーマ相当)
  // キーフォーム: [角度(度), x移動, y移動, 拡大率]
  constructor(def) {
    this.id = def.id;
    this.parentId = def.parent ?? null;
    [this.ox, this.oy] = def.origin;
    this.keys = def.keys || [];
    this.forms = (def.forms || []).map((f) => [f.angle ?? 0, f.x ?? 0, f.y ?? 0, f.scale ?? 1]);
    this.value = new Float32Array(4);
  }
  update(params) {
    if (this.forms.length) blendArrays(this.forms, keyformWeights(this.keys, params), this.value);
    else this.value.set([0, 0, 0, 1]);
    const rad = (this.value[0] * Math.PI) / 180;
    this.cos = Math.cos(rad) * this.value[3];
    this.sin = Math.sin(rad) * this.value[3];
  }
  map(px, py, out) {
    const dx = px - this.ox, dy = py - this.oy;
    out[0] = this.ox + this.value[1] + dx * this.cos - dy * this.sin;
    out[1] = this.oy + this.value[2] + dx * this.sin + dy * this.cos;
  }
}

// ------------------------------------------------------------------ モーション(motion3.json)

function bezierAt(p0, p1, p2, p3, s) {
  const i = 1 - s;
  return i * i * i * p0 + 3 * i * i * s * p1 + 3 * i * s * s * p2 + s * s * s * p3;
}

// motion3.json の1カーブを「時刻→値」の関数に変換する
function compileCurve(seg, restricted) {
  const pieces = [];
  let t0 = seg[0], v0 = seg[1], i = 2;
  while (i < seg.length) {
    const type = seg[i];
    if (type === 1) {
      const [c1t, c1v, c2t, c2v, t1, v1] = seg.slice(i + 1, i + 7);
      // 区間の始点は必ず定数に写してから使う(t0・v0 はこの後で書き換わるため、そのまま閉じ込めると最後の区間の値になってしまう)
      const s0 = t0, a0 = v0;
      pieces.push({ t0: s0, t1, f: (t) => {
        let s = (t - s0) / (t1 - s0 || 1);
        if (!restricted) {
          // 時間軸もベジェなので、二分法で時刻に対応する媒介変数を求める
          let lo = 0, hi = 1;
          for (let k = 0; k < 20; k++) {
            const mid = (lo + hi) / 2;
            if (bezierAt(s0, c1t, c2t, t1, mid) < t) lo = mid; else hi = mid;
          }
          s = (lo + hi) / 2;
        }
        return bezierAt(a0, c1v, c2v, v1, s);
      } });
      t0 = t1; v0 = v1; i += 7;
    } else {
      const t1 = seg[i + 1], v1 = seg[i + 2], a = v0, s0 = t0;
      if (type === 0) pieces.push({ t0: s0, t1, f: (t) => a + ((v1 - a) * (t - s0)) / (t1 - s0 || 1) });
      else if (type === 2) pieces.push({ t0: s0, t1, f: () => a });   // 段差: 区間の終わりまで前の値
      else pieces.push({ t0: s0, t1, f: () => v1 });                  // 逆段差: 区間の頭から次の値
      t0 = t1; v0 = v1; i += 3;
    }
  }
  const first = seg[1];
  return (t) => {
    if (!pieces.length || t <= pieces[0].t0) return first;
    for (const p of pieces) if (t <= p.t1) return p.f(t);
    return v0;
  };
}

export class Motion {
  constructor(json, opts = {}) {
    const meta = json.Meta || {};
    this.duration = meta.Duration ?? 1;
    this.loop = opts.loop ?? meta.Loop ?? false;
    this.fadeIn = opts.fadeIn ?? json.Meta?.FadeInTime ?? 0.3;
    this.fadeOut = opts.fadeOut ?? json.Meta?.FadeOutTime ?? 0.3;
    const restricted = meta.AreBeziersRestricted ?? true;
    this.curves = (json.Curves || [])
      .filter((c) => c.Target === "Parameter")
      .map((c) => ({ id: c.Id, f: compileCurve(c.Segments, restricted),
                     fadeIn: c.FadeInTime, fadeOut: c.FadeOutTime }));
    this.ids = new Set(this.curves.map((c) => c.id));
  }
  static async load(url, opts) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`モーションを読み込めません: ${url}`);
    return new Motion(await res.json(), opts);
  }
}

// 再生中モーションの管理
// ・単発再生は「尺の終わり − フェードアウト秒」から自然に消えていく
// ・新しいモーションを再生すると、再生中のものはその時点からフェードアウトして重なる
class MotionPlayer {
  constructor() { this.active = []; this.time = 0; }
  play(motion) {
    for (const a of this.active) {
      if (a.fadeOutStart === null || a.fadeOutStart > this.time) a.fadeOutStart = this.time;
    }
    const m = motion;
    const entry = { motion: m, start: this.time,
                    fadeOutStart: m.loop ? null : this.time + Math.max(0, m.duration - m.fadeOut) };
    entry.finished = new Promise((r) => { entry.done = r; });
    this.active.push(entry);
    return entry.finished;
  }
  stop() {
    for (const a of this.active) {
      if (a.fadeOutStart === null || a.fadeOutStart > this.time) a.fadeOutStart = this.time;
    }
  }
  // フェードアウトに入っていないモーションがあるか
  get playing() { return this.active.some((a) => a.fadeOutStart === null || a.fadeOutStart > this.time); }
  // params に重み付きで書き込み、書き込んだパラメータ名の集合を返す
  update(time, params) {
    this.time = time;
    const touched = new Set();
    for (const a of this.active) {
      const m = a.motion;
      let t = time - a.start;
      t = m.loop ? t % m.duration : Math.min(t, m.duration);
      const fin = m.fadeIn > 0 ? clamp((time - a.start) / m.fadeIn, 0, 1) : 1;
      let fout = 1;
      if (a.fadeOutStart !== null && time >= a.fadeOutStart) {
        fout = m.fadeOut > 0 ? clamp(1 - (time - a.fadeOutStart) / m.fadeOut, 0, 1) : 0;
      }
      const w = ease(fin) * ease(fout);
      for (const c of m.curves) {
        const cur = params.get(c.id) ?? 0;
        params.set(c.id, cur + (c.f(t) - cur) * w);
        if (w > 0.001) touched.add(c.id);
      }
      if (a.fadeOutStart !== null && time >= a.fadeOutStart + m.fadeOut) a.dead = true;
    }
    for (const a of this.active) if (a.dead) a.done();
    this.active = this.active.filter((a) => !a.dead);
    return touched;
  }
}

const ease = (t) => 0.5 - 0.5 * Math.cos(Math.PI * t);

// ------------------------------------------------------------------ 物理演算(髪揺れなど)

// 入力パラメータの動きに遅れて付いていくばね。遅れの量を出力パラメータにする。
class PhysicsRig {
  constructor(defs, paramInfo) {
    this.settings = defs.map((d) => ({ ...d, pos: 0, vel: 0, prevTarget: null }));
    this.info = paramInfo;
  }
  norm(id, v) {
    const p = this.info.get(id);
    if (!p) return 0;
    const mid = (p.max + p.min) / 2, half = (p.max - p.min) / 2 || 1;
    return (v - mid) / half;
  }
  update(dt, params, time) {
    const steps = Math.max(1, Math.ceil(dt / (1 / 120)));
    const h = dt / steps;
    for (const s of this.settings) {
      let target = 0;
      for (const inp of s.inputs) target += this.norm(inp.param, params.get(inp.param) ?? 0) * inp.weight;
      if (s.prevTarget === null) { s.pos = target; s.prevTarget = target; }
      const k = s.stiffness ?? 60, c = s.damping ?? 8;
      for (let i = 0; i < steps; i++) {
        const acc = -k * (s.pos - target) - c * s.vel;
        s.vel += acc * h;
        s.pos += s.vel * h;
      }
      s.prevTarget = target;
      // 遅れ(先端と根元の差)と、ゆるい風の揺らぎを出力にする
      const wind = s.wind ? Math.sin(time * (s.windSpeed ?? 1.3) + (s.phase ?? 0)) * s.wind : 0;
      const out = this.info.get(s.output);
      if (!out) continue;
      const lag = (s.pos - target) * (s.scale ?? 1) + wind;
      const mid = (out.max + out.min) / 2, half = (out.max - out.min) / 2;
      params.set(s.output, clamp(mid + lag * half, out.min, out.max));
    }
  }
}

// ------------------------------------------------------------------ 描画(WebGL)

const VS = `
attribute vec2 aPos; attribute vec2 aUv;
uniform vec4 uView; // x移動, y移動, 拡大率(横), 拡大率(縦) → クリップ座標
varying vec2 vUv;
void main() {
  vUv = aUv;
  gl_Position = vec4(aPos.x * uView.z + uView.x, -(aPos.y * uView.w + uView.y), 0.0, 1.0);
}`;
const FS = `
precision mediump float;
varying vec2 vUv; uniform sampler2D uTex; uniform float uOpacity; uniform vec3 uMul; uniform float uMaskPass;
void main() {
  vec4 c = texture2D(uTex, vUv);
  if (uMaskPass > 0.5) { if (c.a < 0.5) discard; gl_FragColor = vec4(1.0); return; }
  gl_FragColor = vec4(c.rgb * uMul, c.a) * uOpacity;
}`;

class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext("webgl", { premultipliedAlpha: true, alpha: true, stencil: true,
                                            preserveDrawingBuffer: true, antialias: true });
    if (!gl) throw new Error("WebGLが使えません");
    this.gl = gl;
    const sh = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const pr = gl.createProgram();
    gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(pr);
    this.pr = pr;
    this.loc = {
      aPos: gl.getAttribLocation(pr, "aPos"), aUv: gl.getAttribLocation(pr, "aUv"),
      uView: gl.getUniformLocation(pr, "uView"), uTex: gl.getUniformLocation(pr, "uTex"),
      uOpacity: gl.getUniformLocation(pr, "uOpacity"), uMul: gl.getUniformLocation(pr, "uMul"),
      uMaskPass: gl.getUniformLocation(pr, "uMaskPass"),
    };
  }
  async loadTexture(url) {
    const gl = this.gl;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = url;
    await img.decode();
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    const pow2 = (n) => (n & (n - 1)) === 0;
    if (pow2(img.width) && pow2(img.height)) {
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    } else {
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }
  createMesh(d) {
    const gl = this.gl;
    const m = { count: d.indices.length };
    m.pos = gl.createBuffer();
    m.uv = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, m.uv);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(d.uvs), gl.STATIC_DRAW);
    m.idx = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.idx);
    const big = d.positions.length / 2 > 65535;
    m.type = big ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    if (big) gl.getExtension("OES_element_index_uint");
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, big ? new Uint32Array(d.indices) : new Uint16Array(d.indices), gl.STATIC_DRAW);
    return m;
  }
}

// ------------------------------------------------------------------ モデル本体

export class Puppet {
  // puppet.json を読み込んで描画できる状態にする
  static async load(canvas, url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`モデルを読み込めません: ${url}`);
    const json = await res.json();
    const p = new Puppet(canvas, json, url);
    await p._init();
    return p;
  }

  constructor(canvas, json, url) {
    this.canvas = canvas;
    this.json = json;
    this.base = new URL(url, location.href);
    this.width = json.canvas.width;
    this.height = json.canvas.height;
    this.paramInfo = new Map(json.parameters.map((p) => [p.id, p]));
    this.params = new Map();          // 現在のフレームの値
    this.userParams = new Map();      // 利用者が setParam で指定した値(毎フレームの基準値)
    for (const p of json.parameters) this.userParams.set(p.id, p.default);
    this.deformers = new Map();
    for (const d of json.deformers) {
      this.deformers.set(d.id, d.type === "rotation" ? new RotationDeformer(d) : new WarpDeformer(d));
    }
    // 親→子の順に更新できるよう並べる
    this.deformerOrder = [];
    const seen = new Set();
    const visit = (d) => {
      if (seen.has(d.id)) return;
      if (d.parentId) visit(this.deformers.get(d.parentId));
      seen.add(d.id); this.deformerOrder.push(d);
    };
    for (const d of this.deformers.values()) visit(d);
    this.drawables = json.drawables.map((d) => ({
      ...d,
      rest: new Float32Array(d.positions),
      delta: new Float32Array(d.positions.length),
      out: new Float32Array(d.positions.length),
      opacityNow: 1,
    }));
    this.byId = new Map(this.drawables.map((d) => [d.id, d]));
    this.drawOrder = [...this.drawables].sort((a, b) => a.order - b.order);
    this.physics = new PhysicsRig(json.physics || [], this.paramInfo);
    this.motions = new MotionPlayer();
    this.time = 0;
    // 自動動作の設定(アプリ側で切り替え可能)
    this.auto = { blink: true, breath: true, lookAt: true, physics: true, bodyFollow: true };
    this.lookTarget = { x: 0, y: 0 };
    this.look = { x: 0, y: 0, vx: 0, vy: 0 };
    this.lipSync = null;              // 0〜1 を返す関数、または数値
    this._blink = { next: 1.5, phase: -1 };
    this.view = { x: 0, y: 0, scale: 1 };  // 描画位置(キャンバスのピクセル単位)
    this.fit();
  }

  async _init() {
    this.renderer = new Renderer(this.canvas);
    this.textures = [];
    for (const t of this.json.textures) this.textures.push(await this.renderer.loadTexture(new URL(t, this.base).href));
    for (const d of this.drawables) d.mesh = this.renderer.createMesh(d);
  }

  // キャンバスに全身が収まるように配置する(margin: 余白の割合、focusY: 表示の中心にする高さ 0〜1)
  fit(margin = 0.04, focus = null) {
    const cw = this.canvas.width, ch = this.canvas.height;
    let bx = 0, by = 0, bw = this.width, bh = this.height;
    if (focus) [bx, by, bw, bh] = focus;
    const s = Math.min(cw / bw, ch / bh) * (1 - margin * 2);
    this.view.scale = s;
    this.view.x = cw / 2 - (bx + bw / 2) * s;
    this.view.y = ch / 2 - (by + bh / 2) * s;
  }

  // ---------------------------------------------------------------- 利用者向けの操作

  get parameterList() { return this.json.parameters; }
  setParam(id, value) {
    const p = this.paramInfo.get(id);
    if (p) this.userParams.set(id, clamp(value, p.min, p.max));
  }
  getParam(id) { return this.params.get(id) ?? this.userParams.get(id); }
  // 視線・顔の向きの目標(-1〜1。xは右が正、yは上が正)
  lookAt(x, y) { this.lookTarget.x = clamp(x, -1, 1); this.lookTarget.y = clamp(y, -1, 1); }
  // 口パク: 数値(0〜1)か、毎フレーム呼ばれる関数を渡す。null で解除。
  setLipSync(v) { this.lipSync = v; }
  // 音声(audio要素・メディアストリーム)の音量で口パクさせる
  connectAudio(source, { gain = 4, audioContext } = {}) {
    const ctx = audioContext || new (window.AudioContext || window.webkitAudioContext)();
    let node;
    if (source instanceof MediaStream) node = ctx.createMediaStreamSource(source);
    else if (source instanceof HTMLMediaElement) { node = ctx.createMediaElementSource(source); node.connect(ctx.destination); }
    else node = source;
    const an = ctx.createAnalyser();
    an.fftSize = 1024;
    node.connect(an);
    const buf = new Float32Array(an.fftSize);
    let smooth = 0;
    this.lipSync = () => {
      an.getFloatTimeDomainData(buf);
      let sum = 0;
      for (const v of buf) sum += v * v;
      const rms = Math.sqrt(sum / buf.length);
      smooth += (Math.min(1, rms * gain) - smooth) * 0.5;
      return smooth;
    };
    return ctx;
  }
  // 表情: 値の組(例 {ParamEyeForm: 1, ParamMouthForm: -1})へ fade 秒かけて切り替え、そのまま保つ。
  // 前の表情で動かしていた値は初期値へ戻る。まばたき・口パク・モーションはこの上に重なる。
  setExpression(params = {}, fade = 0.3) {
    const prev = this._exprNow || {};
    const keys = new Set([...Object.keys(prev), ...Object.keys(params)]);
    const e = { from: {}, to: {}, t: 0, dur: Math.max(fade, 1e-3) };
    for (const k of keys) {
      const info = this.paramInfo.get(k);
      if (!info) continue;
      e.from[k] = prev[k] ?? info.default;
      e.to[k] = clamp(params[k] ?? info.default, info.min, info.max);
    }
    this._expr = e;
  }

  // モーション再生(URL、motion3.json のオブジェクト、Motion のいずれか)。終了時に解決する Promise を返す。
  async playMotion(m, opts) {
    const motion = m instanceof Motion ? m : typeof m === "string" ? await Motion.load(m, opts) : new Motion(m, opts);
    return this.motions.play(motion);
  }
  stopMotion() { this.motions.stop(); }
  get isMotionPlaying() { return this.motions.playing; }

  // ---------------------------------------------------------------- 毎フレームの処理

  update(dt) {
    dt = clamp(dt, 0, 0.1);
    this.time += dt;
    const P = this.params;
    for (const [k, v] of this.userParams) P.set(k, v);

    // 表情(利用者の指定値の上に、なめらかに切り替わる値を置く)
    if (this._expr) {
      const e = this._expr;
      e.t += dt;
      const w = ease(Math.min(1, e.t / e.dur));
      const now = {};
      for (const k of Object.keys(e.to)) {
        now[k] = e.from[k] + (e.to[k] - e.from[k]) * w;
        const info = this.paramInfo.get(k);
        if (Math.abs(e.to[k] - info.default) > 1e-6 || w < 1) P.set(k, now[k]);
      }
      this._exprNow = now;
    }

    // モーション
    const touched = this.motions.update(this.time, P);

    // 視線追従(なめらかに目標へ寄せる)
    const L = this.look;
    for (const ax of ["x", "y"]) {
      const v = ax === "x" ? "vx" : "vy";
      const acc = (this.lookTarget[ax] - L[ax]) * 90 - L[v] * 18;
      L[v] += acc * dt; L[ax] += L[v] * dt;
    }
    if (this.auto.lookAt) {
      const add = (id, v) => { if (this.paramInfo.has(id)) P.set(id, (P.get(id) ?? 0) + v); };
      add("ParamAngleX", L.x * 24); add("ParamAngleY", L.y * 18);
      add("ParamAngleZ", -L.x * L.y * 8);
      add("ParamBodyAngleX", L.x * 4);
      if (!touched.has("ParamEyeBallX")) add("ParamEyeBallX", L.x);
      if (!touched.has("ParamEyeBallY")) add("ParamEyeBallY", L.y);
    }

    // 体の追従: 頭が動くと上半身も少しついていく(頭だけが動いて首が伸びて見えないように)
    if (this.auto.bodyFollow !== false) {
      const add = (id, v) => { if (this.paramInfo.has(id)) P.set(id, (P.get(id) ?? 0) + v); };
      add("ParamBodyAngleX", (P.get("ParamAngleX") ?? 0) * 0.22);
      add("ParamBodyAngleZ", (P.get("ParamAngleZ") ?? 0) * 0.18);
      add("ParamBreath", 0);
    }

    // 呼吸
    if (this.auto.breath) {
      const t = this.time;
      const add = (id, v) => { if (this.paramInfo.has(id)) P.set(id, (P.get(id) ?? 0) + v); };
      add("ParamAngleX", Math.sin(t * 2 * Math.PI / 6.5) * 1.5);
      add("ParamAngleY", Math.sin(t * 2 * Math.PI / 3.5) * 1.2);
      add("ParamAngleZ", Math.sin(t * 2 * Math.PI / 5.5) * 1.5);
      add("ParamBodyAngleX", Math.sin(t * 2 * Math.PI / 15.5) * 1.5);
      if (this.paramInfo.has("ParamBreath")) P.set("ParamBreath", 0.5 + 0.5 * Math.sin(t * 2 * Math.PI / 3.2));
    }

    // 自動まばたき(モーションが目を動かしていない間だけ)
    if (this.auto.blink && !touched.has("ParamEyeLOpen") && !touched.has("ParamEyeROpen")) {
      const b = this._blink;
      if (b.phase < 0 && this.time >= b.next) b.phase = 0;
      let f = 1;
      if (b.phase >= 0) {
        b.phase += dt;
        const close = 0.09, hold = 0.04, open = 0.14;
        if (b.phase < close) f = 1 - b.phase / close;
        else if (b.phase < close + hold) f = 0;
        else if (b.phase < close + hold + open) f = (b.phase - close - hold) / open;
        else { b.phase = -1; b.next = this.time + 2 + Math.random() * 4; }
      }
      for (const id of ["ParamEyeLOpen", "ParamEyeROpen"]) if (P.has(id)) P.set(id, P.get(id) * f);
    }

    // 口パク
    if (this.lipSync !== null && this.paramInfo.has("ParamMouthOpenY")) {
      const v = typeof this.lipSync === "function" ? this.lipSync() : this.lipSync;
      P.set("ParamMouthOpenY", Math.max(P.get("ParamMouthOpenY") ?? 0, clamp(v, 0, 1)));
    }

    // 物理演算
    if (this.auto.physics) this.physics.update(dt, P, this.time);

    // 範囲に収める
    for (const [id, p] of this.paramInfo) P.set(id, clamp(P.get(id) ?? p.default, p.min, p.max));

    this._deform();
  }

  _deform() {
    const P = this.params;
    for (const d of this.deformerOrder) d.update(P);
    const tmp = [0, 0];
    for (const d of this.drawables) {
      if (d.forms && d.forms.length) blendArrays(d.forms, keyformWeights(d.keys || [], P), d.delta);
      if (d.opacity) {
        const w = keyformWeights(d.opacity.keys, P);
        d.opacityNow = w.reduce((s, [i, x]) => s + d.opacity.values[i] * x, 0);
      }
      const chain = [];
      let id = d.parent;
      while (id) { const df = this.deformers.get(id); chain.push(df); id = df.parentId; }
      const n = d.rest.length;
      for (let i = 0; i < n; i += 2) {
        tmp[0] = d.rest[i] + d.delta[i];
        tmp[1] = d.rest[i + 1] + d.delta[i + 1];
        for (const df of chain) df.map(tmp[0], tmp[1], tmp);
        d.out[i] = tmp[0]; d.out[i + 1] = tmp[1];
      }
    }
    this._blendGroups();
  }

  // 差し替え用の組(口の形の描き分けなど)の不透明度を直す。
  // 重みを鋭くして(ほぼ切り替えに近づけて)から、下から順に重ねたときに
  // ちょうど重みどおりに混ざる不透明度(自分の重み ÷ ここまでの重みの合計)にする
  _blendGroups() {
    const groups = new Map();
    for (const d of this.drawOrder) {
      if (!d.blendGroup) continue;
      if (!groups.has(d.blendGroup)) groups.set(d.blendGroup, []);
      groups.get(d.blendGroup).push(d);
    }
    for (const list of groups.values()) {
      const pw = list[0].blendSharpen || 1;
      const ws = list.map((d) => Math.pow(Math.max(0, d.opacityNow), pw));
      const sum = ws.reduce((a, b) => a + b, 0) || 1;
      let acc = 0;
      list.forEach((d, i) => {
        const w = ws[i] / sum;
        acc += w;
        d.opacityNow = acc > 1e-6 ? w / acc : 0;
      });
    }
  }

  render() {
    const r = this.renderer, gl = r.gl, L = r.loc;
    const cw = this.canvas.width, ch = this.canvas.height;
    gl.viewport(0, 0, cw, ch);
    gl.clearColor(0, 0, 0, 0);
    gl.clearStencil(0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
    gl.useProgram(r.pr);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const v = this.view;
    gl.uniform4f(L.uView, (v.x / cw) * 2 - 1, (v.y / ch) * 2 - 1, (v.scale / cw) * 2, (v.scale / ch) * 2);
    gl.uniform1i(L.uTex, 0);
    gl.activeTexture(gl.TEXTURE0);
    const draw = (d, maskPass) => {
      gl.bindTexture(gl.TEXTURE_2D, this.textures[d.texture ?? 0]);
      gl.bindBuffer(gl.ARRAY_BUFFER, d.mesh.pos);
      gl.bufferData(gl.ARRAY_BUFFER, d.out, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(L.aPos);
      gl.vertexAttribPointer(L.aPos, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, d.mesh.uv);
      gl.enableVertexAttribArray(L.aUv);
      gl.vertexAttribPointer(L.aUv, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, d.mesh.idx);
      gl.uniform1f(L.uMaskPass, maskPass ? 1 : 0);
      gl.uniform1f(L.uOpacity, d.opacityNow);
      const m = d.multiply || [1, 1, 1];
      gl.uniform3f(L.uMul, m[0], m[1], m[2]);
      gl.drawElements(gl.TRIANGLES, d.mesh.count, d.mesh.type, 0);
    };
    for (const d of this.drawOrder) {
      if (d.maskOnly || d.opacityNow <= 0.001) continue;
      if (d.masks && d.masks.length) {
        // 切り抜き: マスク部品の形をステンシルに書き、その内側だけに描く
        gl.enable(gl.STENCIL_TEST);
        gl.clear(gl.STENCIL_BUFFER_BIT);
        gl.colorMask(false, false, false, false);
        gl.stencilFunc(gl.ALWAYS, 1, 0xff);
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
        for (const mid of d.masks) draw(this.byId.get(mid), true);
        gl.colorMask(true, true, true, true);
        gl.stencilFunc(d.invertMask ? gl.NOTEQUAL : gl.EQUAL, 1, 0xff);
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
        draw(d, false);
        gl.disable(gl.STENCIL_TEST);
      } else {
        draw(d, false);
      }
    }
  }
}

// ------------------------------------------------------------------ 画面に置くための補助

// キャンバスの大きさ合わせ・毎フレームの更新・マウス追従をまとめて面倒を見る
export class PuppetView {
  static async create(canvas, url, opts = {}) {
    const v = new PuppetView(canvas, opts);
    await v.load(url);
    return v;
  }
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = { follow: "window", pixelRatio: window.devicePixelRatio || 1, ...opts };
    this.running = false;
    this.onFrame = null;
    this.idle = [];
  }
  async load(url) {
    this.resize();
    this.puppet = await Puppet.load(this.canvas, url);
    this.puppet.fit(0.04, this.opts.focus || null);
    const ro = new ResizeObserver(() => { this.resize(); this.puppet.fit(0.04, this.opts.focus || null); });
    ro.observe(this.canvas);
    if (this.opts.follow) {
      const target = this.opts.follow === "window" ? window : this.canvas;
      target.addEventListener("pointermove", (e) => this._follow(e));
      target.addEventListener("pointerleave", () => this.puppet.lookAt(0, 0));
    }
    this.start();
    return this.puppet;
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    const pr = this.opts.pixelRatio;
    const w = Math.max(1, Math.round(r.width * pr)), h = Math.max(1, Math.round(r.height * pr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }
  _follow(e) {
    const p = this.puppet, r = this.canvas.getBoundingClientRect();
    const pr = this.opts.pixelRatio;
    // 顔の位置を基準に、指した方向を向く
    const face = p.json.anchors?.face || [p.width / 2, p.height * 0.15];
    const fx = (face[0] * p.view.scale + p.view.x) / pr + r.left;
    const fy = (face[1] * p.view.scale + p.view.y) / pr + r.top;
    const range = Math.max(r.width, r.height) * 0.5;
    p.lookAt((e.clientX - fx) / range, -(e.clientY - fy) / range);
  }
  // 何も再生していない間に順番に流す待機モーション
  setIdleMotions(list) { this.idle = list; }
  playMotion(m, opts) { return this.puppet.playMotion(m, opts); }
  start() {
    if (this.running) return;
    this.running = true;
    let last = performance.now();
    const loop = (now) => {
      if (!this.running) return;
      const dt = (now - last) / 1000; last = now;
      if (this.idle.length && !this.puppet.isMotionPlaying && !this._idleBusy) {
        this._idleBusy = true;
        const m = this.idle[Math.floor(Math.random() * this.idle.length)];
        this.puppet.playMotion(m).finally(() => { this._idleBusy = false; });
      }
      this.puppet.update(dt);
      this.puppet.render();
      if (this.onFrame) this.onFrame(this.puppet);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
  stop() { this.running = false; }
}
