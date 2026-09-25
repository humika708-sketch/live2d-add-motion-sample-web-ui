// 表示エンジンのモーション(motion3.json)のカーブ計算の試験
// 使い方: node puppet/tests/motion_curve.test.mjs
//
// ・ベジェ・直線・段差・逆段差を混ぜた合成のカーブで、キーフレームの値と区間の途中の値を確かめる
//   (以前、区間の始点を変数のまま閉じ込めていたため、全区間が最後の区間の値で計算される不具合があった)
// ・models/kurisu/motions があれば、そこの全モーションも同じように確かめる
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
globalThis.window = {};
globalThis.location = { href: "http://localhost/" };
const { Motion } = await import(path.join(here, "../runtime/puppet.js"));

let failures = 0;
function check(cond, msg) {
  if (!cond) { failures++; console.log("不合格:", msg); }
}

// 1. 合成のカーブ: 0秒 0 →(ベジェ)1秒 1 →(直線)2秒 3 →(段差)3秒 5 →(逆段差)4秒 2 →(ベジェ)5秒 2
const seg = [0, 0,
  1, 1 / 3, 0, 2 / 3, 1, 1, 1,
  0, 2, 3,
  2, 3, 5,
  3, 4, 2,
  1, 4 + 1 / 3, 2, 4 + 2 / 3, 2, 5, 2];
const m = new Motion({ Meta: { Duration: 5, AreBeziersRestricted: true }, Curves: [{ Target: "Parameter", Id: "P", Segments: seg }] });
const f = m.curves[0].f;
const near = (a, b) => Math.abs(a - b) < 1e-6;
check(near(f(0), 0), `0秒は0のはず(実際 ${f(0)})`);
check(near(f(1), 1), `1秒は1のはず(実際 ${f(1)})`);
check(near(f(0.5), 0.5), `平らな制御点のベジェの中点は0.5のはず(実際 ${f(0.5)})`);
check(near(f(1.5), 2), `直線の中点は2のはず(実際 ${f(1.5)})`);
check(near(f(2.5), 3), `段差の区間は前の値3のはず(実際 ${f(2.5)})`);
check(near(f(3.5), 2), `逆段差の区間は次の値2のはず(実際 ${f(3.5)})`);
check(near(f(4.5), 2), `平らなベジェは2のはず(実際 ${f(4.5)})`);
check(near(f(9), 2), `終わった後は最後の値2のはず(実際 ${f(9)})`);

// 2. 生成済みのモーションがあれば全部確かめる
const dir = path.join(here, "../models/kurisu/motions");
let n = 0;
if (fs.existsSync(dir)) {
  for (const file of fs.readdirSync(dir).filter((x) => x.endsWith(".motion3.json"))) {
    const j = JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
    const mo = new Motion(j);
    for (const c of j.Curves) {
      const s = c.Segments, keys = [[s[0], s[1]]];
      for (let i = 2; i < s.length;) {
        if (s[i] === 1) { keys.push([s[i + 5], s[i + 6]]); i += 7; } else { keys.push([s[i + 1], s[i + 2]]); i += 3; }
      }
      const cf = mo.curves.find((x) => x.id === c.Id).f;
      for (const [t, v] of keys) check(Math.abs(cf(t) - v) < 1e-3, `${file} ${c.Id} ${t}秒は${v}のはず(実際 ${cf(t)})`);
      for (let k = 0; k + 1 < keys.length; k++) {
        const [ta, va] = keys[k], [tb, vb] = keys[k + 1];
        const mid = cf((ta + tb) / 2);
        check(mid >= Math.min(va, vb) - 1e-6 && mid <= Math.max(va, vb) + 1e-6, `${file} ${c.Id} ${ta}〜${tb}秒の途中が範囲外(${mid})`);
      }
    }
    n++;
  }
}
console.log(failures ? `不合格 ${failures} 件` : `合格(合成のカーブ + 生成済みモーション ${n} 本)`);
process.exit(failures ? 1 : 0);
