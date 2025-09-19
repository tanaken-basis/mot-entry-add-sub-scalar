import { useMemo, useState, useEffect } from "react";
import { MathJax, MathJaxContext } from "better-react-mathjax";

// ---------- translations ----------
const translations = {
  ja: {
    title: "行列演算トレーナー",
    subtitle: "(成分, 足し算, 引き算, スカラー倍)", // "(成分, 足し算, 引き算, スカラー倍, 掛け算, 行列式, 逆行列)",
    newProblem: "新しい問題を作る",
    checkAnswer: "答え合わせ",
    hideAnswer: "答えを隠す",
    problem: "問題",
    instruction: "次の行列の演算を計算してください。",
    answerSteps: "答え と 途中経過",
    pressCheck: "「答え合わせ」を押すと表示されます。",
    allCorrect: "🎉 全て正解です！",
    correct: "🎉 正解です！",
    someIncorrect: "一部間違いがあります。赤枠(点線)のセルを見直しましょう。",
    incorrect: "不正解です。",
    hint: "ヒント: 行列の足し算・引き算・スカラー倍は要素ごとの計算です。",
    hintAddSubScalar: "ヒント: 足し算・引き算・スカラー倍は要素ごとの計算です。",
    hintMulDims: "ヒント: 行列の積 A(m × n) B(n × p) は n が一致する必要があり、結果は m × p 行列になります。",
    hintMulFormula: "成分は、行ベクトルと列ベクトルの内積です。",
    hintScalarFormula: "スカラー倍の成分は (kA)_{ij} = k A_{ij} です。",
    entryQuestion: "(${r}, ${c}) 成分の値は？", // (r, c) => '(${r}, ${c}) 成分の値は？',
    entryHint: "ヒント: (i, j) 成分は i 行 j 列の要素を指します（一番上が1行目, 一番左が1列目）。",
    detQuestion: "この行列の行列式の値は？",
    hintDet: "ヒント: det(A) は A の行列式。2×2 は ad - bc、3×3 はサラスの公式や余因子展開で計算できます。",
    hintInv: "ヒント: 2×2 では 簡単な公式があります。3×3 では 余因子行列の転置（随伴）を使います。",
    modeLabel: "出題モード",
    mode_random: "ランダム",
    mode_entry: "成分",
    mode_add: "足し算",
    mode_sub: "引き算",
    mode_times: "スカラー倍",
    mode_cdot: "掛け算",
    mode_det: "行列式",
    mode_inv: "逆行列",
    mode_row: "行基本変形",
    hintRow: "ヒント: 行の入れ替え / 行のスカラー倍 / 行の加法 のいずれか1回を適用します。",
},
  en: {
    title: "Matrix Operations Trainer",
    subtitle: "(Entry, Addition, Subtraction, Scalar multiplication)", // "(Entry, Addition, Subtraction, Scalar multiplication, Matrix multiplication, Determinant, Inverse matrix)",
    newProblem: "New Problem",
    checkAnswer: "Check Answer",
    hideAnswer: "Hide Answer",
    problem: "Problem",
    instruction: "Calculate the following matrix operation.",
    answerSteps: "Answer and Steps",
    pressCheck: "Press 'Check Answer' to reveal.",
    allCorrect: "🎉 All correct!",
    correct: "🎉 Correct!",
    someIncorrect: "Some answers are incorrect. Check the red dotted-frame cells.",
    incorrect: "Incorrect.",
    hint: "Hint: Addition, subtraction, and scalar multiplication are element-wise.",
    hintAddSubScalar: "Hint: Addition, subtraction, and scalar multiplication are element-wise.",
    hintMulDims: "Hint: For A(m × n) B(n × p), n must match; the result is m × p.",
    hintMulFormula: "Each entry is the dot product of a row and a column.",
    hintScalarFormula: "Scalar multiplication: (kA)_{ij} = k A_{ij}.",
    entryQuestion: "What is the (${r}, ${c})-entry?", // (r, c) => 'What is the (${r}, ${c})-entry?',
    entryHint: "Hint: The (i, j)-entry means the element in row i and column j (the top is the first row, the leftmost is the first column).",
    detQuestion: "What is the value of the determinant?",
    hintDet: "Hint: det(A) is the determinant of A. For 2×2, ad - bc; for 3×3, use Sarrus' rule or cofactor expansion.",
    hintInv: "Hint: For 2×2, there is a simple formula. For 3×3, use the adjugate (transpose of the cofactor matrix).",
    modeLabel: "Mode",
    mode_random: "Random",
    mode_entry: "Entry",
    mode_add: "Addition",
    mode_sub: "Subtraction",
    mode_times: "Scalar ×A",
    mode_cdot: "Matrix product",
    mode_det: "Determinant",
    mode_inv: "Inverse",
    mode_row: "Row operation",
    hintRow: "Hint: Apply exactly one of swap / scale / row addition once.",
  },
};

// ---------- helpers ----------
const randInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
const makeMatrix = (r, c, min = -10, max = 10) =>
  Array.from({ length: r }, () => Array.from({ length: c }, () => randInt(min, max)));
const zeros = (r, c) => Array.from({ length: r }, () => Array.from({ length: c }, () => 0));
const wrapNum = (x) => (x < 0 ? `(${x})` : `${x}`);

const toBMatrix = (M) => {
  const rows = M.map((row) => row.join(" & ")).join(" \\\\ ");
  return `\\begin{bmatrix} ${rows} \\end{bmatrix}`;
};
const toDetBMatrix = (M) => {
  const rows = M.map((row) => row.join(" & ")).join(" \\\\ ");
  return `\\operatorname{det}\\begin{bmatrix} ${rows} \\end{bmatrix}`;
};

// arithmetic operations
const add = (A, B) => A.map((row, r) => row.map((v, c) => v + B[r][c]));
const sub = (A, B) => A.map((row, r) => row.map((v, c) => v - B[r][c]));
const scalarMul = (A, k) => A.map((row) => row.map((v) => v * k));

// 行列積 C = A(m times n) cdot B(n times p)
const matrixMul = (A, B) => {
  const m = A.length, n = A[0].length, p = B[0].length;
  const C = Array.from({ length: m }, () => Array.from({ length: p }, () => 0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += A[i][k] * B[k][j];
      C[i][j] = s;
    }
  }
  return C;
};

// determinant (n = 1,2,3 を想定)
const det = (M) => {
  const n = M.length;
  if (n === 1) return M[0][0];
  if (n === 2) return M[0][0]*M[1][1] - M[0][1]*M[1][0];
  if (n === 3) {
    const a=M[0][0], b=M[0][1], c=M[0][2];
    const d=M[1][0], e=M[1][1], f=M[1][2];
    const g=M[2][0], h=M[2][1], i=M[2][2];
    return a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h; // サラスの公式
  }
  return 0;
};

// inverse (n = 2,3 を想定)
// ----- rational helpers -----
const igcd = (a, b) => { a = Math.abs(a); b = Math.abs(b); while (b) [a,b] = [b, a % b]; return a || 1; };
const norm = (n, d) => { // keep denominator > 0
  if (d < 0) { n = -n; d = -d; }
  const g = igcd(n, d);
  return { num: n / g, den: d / g };
};
const R = {
  fromInt: (x) => ({ num: x, den: 1 }),
  add: (a, b) => norm(a.num*b.den + b.num*a.den, a.den*b.den),
  sub: (a, b) => norm(a.num*b.den - b.num*a.den, a.den*b.den),
  mul: (a, b) => norm(a.num*b.num, a.den*b.den),
  div: (a, b) => norm(a.num*b.den, a.den*b.num),
  eq:  (a, b) => (a.num === b.num && a.den === b.den),
};
// parse " -3/5 " or " 2 " → rational
const parseRational = (s) => {
  if (typeof s === "number") return R.fromInt(s);
  const t = String(s ?? "").trim();
  if (t === "" || t === "-") return null; // 入力途中は未判定
  if (t.includes("/")) {
    const [ns, ds] = t.split("/").map(x => x.trim());
    const n = Number(ns), d = Number(ds);
    if (!Number.isFinite(n) || !Number.isFinite(d) || d === 0) return null;
    return norm(n, d);
  }
  const n = Number(t);
  if (!Number.isFinite(n)) return null;
  return norm(n, 1);
};
// TeX 文字列に（分母1は整数として）
const texR = (r) => r.den === 1 ? `${r.num}` : `\\frac{${r.num}}{${r.den}}`;
// 表示用文字列（入力欄初期値など）
const strR = (r) => r.den === 1 ? String(r.num) : `${r.num}/${r.den}`;

// adj(A)/det(A) をすべて有理数で
const detR2 = (A) => R.sub(R.mul(A[0][0], A[1][1]), R.mul(A[0][1], A[1][0]));
const detR3 = (A) => {
  const [a,b,c] = A[0], [d,e,f] = A[1], [g,h,i] = A[2];
  // a(ei − fh) − b(di − fg) + c(dh − eg)
  return R.add(
    R.sub(R.mul(a, R.sub(R.mul(e,i), R.mul(f,h))),
          R.mul(b, R.sub(R.mul(d,i), R.mul(f,g)))),
    R.mul(c, R.sub(R.mul(d,h), R.mul(e,g)))
  );
};
const inverseR2 = (A) => {
  const det = detR2(A);
  if (det.num === 0) return null;
  // (1/det) * [[d,-b],[-c,a]]
  const [[a,b],[c,d]] = A;
  const inv = [
    [d, {num:-b.num, den:b.den}],
    [{num:-c.num, den:c.den}, a],
  ].map(row => row.map(x => R.div(x, det)));
  return { inv, det };
};
const inverseR3 = (A) => {
  const det = detR3(A);
  if (det.num === 0) return null;
  // 余因子行列 → 随伴 → 1/det 倍
  const m = A;
  const C11 = R.sub(R.mul(m[1][1], m[2][2]), R.mul(m[1][2], m[2][1]));
  const C12 = { ...R.sub(R.mul(m[1][0], m[2][2]), R.mul(m[1][2], m[2][0])) }; C12.num *= -1;
  const C13 = R.sub(R.mul(m[1][0], m[2][1]), R.mul(m[1][1], m[2][0]));
  const C21 = { ...R.sub(R.mul(m[0][1], m[2][2]), R.mul(m[0][2], m[2][1])) }; C21.num *= -1;
  const C22 = R.sub(R.mul(m[0][0], m[2][2]), R.mul(m[0][2], m[2][0]));
  const C23 = { ...R.sub(R.mul(m[0][0], m[2][1]), R.mul(m[0][1], m[2][0])) }; C23.num *= -1;
  const C31 = R.sub(R.mul(m[0][1], m[1][2]), R.mul(m[0][2], m[1][1]));
  const C32 = { ...R.sub(R.mul(m[0][0], m[1][2]), R.mul(m[0][2], m[1][0])) }; C32.num *= -1;
  const C33 = R.sub(R.mul(m[0][0], m[1][1]), R.mul(m[0][1], m[1][0]));
  const cof = [[C11,C12,C13],[C21,C22,C23],[C31,C32,C33]];
  // 随伴（転置）
  const adj = [[cof[0][0],cof[1][0],cof[2][0]],
               [cof[0][1],cof[1][1],cof[2][1]],
               [cof[0][2],cof[1][2],cof[2][2]]];
  const inv = adj.map(row => row.map(x => R.div(x, det)));
  return { inv, det, cof, adj };
};
const toBMatrixR = (M) => {
  const rows = M.map(row => row.map(texR).join(" & ")).join(" \\\\ ");
  return `\\begin{bmatrix} ${rows} \\end{bmatrix}`;
};

// ----- row operation helpers -----
const swapRows = (A, i, j) => {
  const B = A.map(r => r.slice());
  [B[i], B[j]] = [B[j], B[i]];
  return B;
};
const scaleRow = (A, i, k) => A.map((row, r) => r === i ? row.map(x => k * x) : row.slice());
const addRows  = (A, i, j, k) => A.map((row, r) =>
  r === i ? row.map((x, c) => x + k * A[j][c]) : row.slice()
);

// 表示用の TeX ラベル, Elementary Row Operation（行基本変形）
const eroLabelTex = (type, i, j=null, k=null) => {
  if (type === "swap")  return `R_{${i+1}} \\,\\leftrightarrow\\, R_{${j+1}}`;
  if (type === "scale") return `R_{${i+1}} \\,\\leftarrow\\, ${wrapNum(k)}R_{${i+1}}`;
  if (type === "add")   return `R_{${i+1}} \\,\\leftarrow\\, R_{${i+1}} + ${wrapNum(k)}R_{${j+1}}`;
  return "";
};

// ---------- UI ----------
function NumberCell({ id, value, onChange, onKeyDown, isCorrect, showCheck }) {
  const border = showCheck ? (isCorrect ? "2px solid #10b981" : "2px dotted #ef4444") : "1px solid #cbd5e1";
  return (
    <input
      id={id}
      type="number"
      inputMode="numeric"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={onKeyDown}
      style={{
        width: 64,
        height: 44,
        textAlign: "center",
        fontSize: 16,
        borderRadius: 10,
        border,
        outline: "none",
      }}
    />
  );
}

/**
 * 分数スピン入力（分子は負OK、分母は1以上）。
 * 編集中は約分せず、blur のときだけ既約化します（normalizeModeで変更可）。
 *
 * Props:
 * - id?: string
 * - value: string | number          // "a/b" or "3"
 * - onChange: (val: string) => void // 親へ通知（編集中は非約分、blur時に既約）
 * - isCorrect?: boolean
 * - showCheck?: boolean
 * - numMin?: number  (default -99)
 * - numMax?: number  (default  99)
 * - denMin?: number  (default   1)
 * - denMax?: number  (default  99)
 * - step?: number    (default   1)
 * - normalizeMode?: 'blur' | 'live' | 'never'  // 既約化のタイミング（既定 'blur'）
 */
export function FractionCell({
  id,
  value,
  onChange,
  isCorrect,
  showCheck,
  numMin = -99,
  numMax =  99,
  denMin =   1,
  denMax =  99,
  step   =   1,
  normalizeMode = 'blur',
}) {
  // 文字列 → {num, den}
  const parse = (v) => {
    const t = String(v ?? "").trim();
    if (t === "" || t === "-") return { num: 0, den: 1 };
    if (t.includes("/")) {
      const [ns, ds] = t.split("/").map(s => s.trim());
      let n = Number.parseInt(ns, 10);
      let d = Number.parseInt(ds, 10);
      if (!Number.isFinite(n)) n = 0;
      if (!Number.isFinite(d) || d === 0) d = 1;
      return { num: n, den: d };
    }
    let n = Number.parseInt(t, 10);
    if (!Number.isFinite(n)) n = 0;
    return { num: n, den: 1 };
  };

  // 約分 & 符号正規化
  const gcd = (a, b) => { a = Math.abs(a); b = Math.abs(b); while (b) [a, b] = [b, a % b]; return a || 1; };
  const normalize = ({ num, den }) => {
    if (den < 0) { num = -num; den = -den; }
    if (num === 0) return { num: 0, den: 1 };
    const g = gcd(num, den);
    return { num: num / g, den: den / g };
  };
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const toStr = ({ num, den }) => (den === 1 ? String(num) : `${num}/${den}`);

  // 内部状態は「非約分」を保持（＝編集中に勝手に変えない）
  const [state, setState] = useState(() => parse(value));
  useEffect(() => { setState(parse(value)); }, [value]);

  // 親へ通知（normalizeMode に応じて）
  const notify = (nextRaw, when = 'change') => {
    const clamped = {
      num: clamp(nextRaw.num, numMin, numMax),
      den: clamp(nextRaw.den, denMin, denMax),
    };
    let out = clamped;
    if (normalizeMode === 'live') {
      out = normalize(clamped); // 常時既約
    } else if (normalizeMode === 'blur' && when === 'blur') {
      out = normalize(clamped); // blur のときだけ既約
    } // 'never' のときは常に非約分のまま
    onChange?.(toStr(out));
  };

  // ハンドラ
  const onNumChange = (e) => {
    const v = Number(e.target.value);
    const next = { ...state, num: Number.isFinite(v) ? v : state.num };
    setState(next);
    notify(next, 'change');
  };
  const onDenChange = (e) => {
    const v = Number(e.target.value);
    const safe = Number.isFinite(v) ? v : state.den;
    const next = { ...state, den: safe };
    setState(next);
    notify(next, 'change');
  };
  const onBlur = () => {
    notify(state, 'blur'); // blur 時に既約化（既定）
  };

  const border = showCheck
    ? (isCorrect ? "2px solid #10b981" : "2px dotted #ef4444")
    : "1px solid #cbd5e1";

  const preventWheel = (e) => e.currentTarget.blur();

  return (
    <div style={{
      display: "inline-flex",
      alignItems: "center",
      gap: 8,
      padding: 6,
      borderRadius: 10,
      border,
      background: "#0b1222",
    }}>
      {/* 分子（負OK） */}
      <input
        id={id ? `${id}-num` : undefined}
        type="number"
        inputMode="numeric"
        value={state.num}
        onChange={onNumChange}
        onBlur={onBlur}
        onWheel={preventWheel}
        min={numMin}
        max={numMax}
        step={step}
        style={{
          width: 88, height: 44, textAlign: "center",
          fontSize: 16, borderRadius: 10, border: "1px solid #475569",
          background: "#111827", color: "#e2e8f0", outline: "none",
        }}
      />
      <span style={{ opacity: 0.8, userSelect: "none" }}>/</span>
      {/* 分母（1以上） */}
      <input
        id={id ? `${id}-den` : undefined}
        type="number"
        inputMode="numeric"
        value={state.den}
        onChange={onDenChange}
        onBlur={onBlur}
        onWheel={preventWheel}
        min={denMin}
        max={denMax}
        step={step}
        style={{
          width: 88, height: 44, textAlign: "center",
          fontSize: 16, borderRadius: 10, border: "1px solid #475569",
          background: "#111827", color: "#e2e8f0", outline: "none",
        }}
      />
    </div>
  );
}

const Bracket = ({ side = "left", rows, cellHeight = 44, gap = 16, stroke = "#94a3b8", strokeWidth = 2 }) => {
  // 行列全体の高さを計算
  const totalHeight = rows * cellHeight + (rows - 1) * gap + 10;

  return (
    <svg
      aria-hidden
      style={{ height: totalHeight, width: 16, flex: "0 0 16px", display: "block" }}
      viewBox={`0 0 10 100`}
      preserveAspectRatio="none"
    >
      <path
        d={side === "left" ? "M9,0 H2 V100 H9" : "M1,0 H8 V100 H1"}
        fill="none"
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeLinecap="square"
        strokeLinejoin="miter"
      />
    </svg>
  );
};

const BracketBox = ({ rows, children }) => (
  <div style={{ display: "inline-flex", alignItems: "center", lineHeight: 0 }}>
    {/* <Bracket side="left" /> */}
    <Bracket side="left" rows={rows} />
      <div style={{ padding: "6px 8px", background: "#0b1222", borderRadius: 12, lineHeight: "normal" }}>{children}</div>
    {/* <Bracket side="right" /> */}
    <Bracket side="right" rows={rows} />
  </div>
);

function MatrixInput({ rows, cols, values, setValues, correctness, showCheck }) {
  const handleChange = (r, c, v) => {
    const num = v === "" || v === "-" ? v : Number(v);
    const next = values.map((row) => row.slice());
    next[r][c] = Number.isNaN(num) ? "" : num;
    setValues(next);
  };
  const handleKeyDown = (r, c, e) => {
    const { key } = e;
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(key)) {
      e.preventDefault();
      const flat = r * cols + c;
      let n = flat;
      if (key === "ArrowRight") n = Math.min(flat + 1, rows * cols - 1);
      if (key === "ArrowLeft") n = Math.max(flat - 1, 0);
      if (key === "ArrowDown") n = Math.min(flat + cols, rows * cols - 1);
      if (key === "ArrowUp") n = Math.max(flat - cols, 0);
      const nr = Math.floor(n / cols), nc = n % cols;
      const el = document.getElementById(`cell-${nr}-${nc}`);
      if (el) el.focus();
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, 68px)`, gap: 16 }}>
      {values.map((row, r) =>
        row.map((v, c) => (
          <NumberCell
            key={`${r}-${c}`}
            id={`cell-${r}-${c}`}
            value={v}
            onChange={(val) => handleChange(r, c, val)}
            onKeyDown={(e) => handleKeyDown(r, c, e)}
            isCorrect={correctness ? correctness[r][c] : false}
            showCheck={showCheck}
          />
        ))
      )}
    </div>
  );
}

export default function App() {
  // 言語
  const [lang, setLang] = useState("ja");
  // 問題種別
  const [mode, setMode] = useState("random");
  // 行列
  const [op, setOp] = useState("+");
  const [dims, setDims] = useState({ rows: 2, cols: 2 });
  const [A, setA] = useState(() => makeMatrix(2, 2));
  const [B, setB] = useState(() => makeMatrix(2, 2));
  const [k, setK] = useState(() => randInt(-5, 5));
  const [user, setUser] = useState(() => zeros(2, 2));
  const [checked, setChecked] = useState(false);
  const [rowAnswer, setRowAnswer] = useState(null); // { kind, i, j, kcoef, Cafter }
  // entry mode
  const [entryPos, setEntryPos] = useState({ r: 1, c: 1 });
  const [userEntry, setUserEntry] = useState("");
  // determinant
  const [userDet, setUserDet] = useState("");
  // 逆行列の正解（有理数行列）を保持
  const [AnsR, setAnsR] = useState(null);

  // const t = (key) => translations[lang][key];
  const t = (key, ...args) => {
    const val = translations[lang][key];
    // 1) 関数翻訳なら、そのまま呼ぶ（t("entryQuestion", r, c) など）
    if (typeof val === "function") return val(...args);
    // 2) 文字列翻訳なら、${...} プレースホルダを置換して返す
    if (typeof val === "string") {
      let vars = {};
      // オブジェクトで渡された場合: t("entryQuestion", { r: 2, c: 3 })
      if (args.length === 1 && typeof args[0] === "object" && args[0] !== null) {
        vars = args[0];
      // 後方互換: t("entryQuestion", 2, 3) でも動くように
      } else if (args.length >= 2) {
        vars = { r: args[0], c: args[1] };
      }
      return val.replace(/\$\{(\w+)\}/g, (_, k) => String(vars?.[k] ?? ""));
    }
    // 3) それ以外は文字列化して返す
    return String(val ?? "");
  };

  const C = useMemo(() => {
    if (op === "+") return add(A, B);
    if (op === "-") return sub(A, B);
    if (op === "times") return scalarMul(A, k);
    if (op === "cdot") return matrixMul(A, B);
    if (op === "entry" || op === "det" || op === "row" || op === "inv") return A; // 使わないのでダミー, 便宜的に A を返す
    return add(A, B);
  }, [A, B, k, op]);

  const detValue = useMemo(() => {
    if (op === "det") return det(A);
    return null;
  }, [A, op]);

  const correctness = useMemo(() => {
    if (!checked) return null;
    
    if (op === "det") {
      return Number(userDet) === detValue;
    }
    if (op === "inv") {
      if (!AnsR?.inv) return null;
      const n = AnsR.inv.length;
      return user.map((row, r) => row.map((v, c) => {
        const u = parseRational(v);
        if (!u) return false; // 入力途中などは不正とみなす（showCheck 前は枠色出ない）
        return R.eq(u, AnsR.inv[r][c]);
      }));
    }
    if (op === "row") {
      if (!rowAnswer) return null;
      const { Cafter } = rowAnswer;
      return user.map((row, r) => row.map((v, c) => v === Cafter[r][c]));
    }
    if (op === "entry") {
      const ans = A[entryPos.r - 1]?.[entryPos.c - 1];
      return Number(userEntry) === ans;
    }
    return user.map((row, r) => row.map((v, c) => v === C[r][c]));
  }, [user, C, checked, op, A, entryPos, userEntry, userDet, detValue]);

  const allCorrect = useMemo(() => {
    if (op === "entry" || op === "det") return !!correctness;
    return correctness && correctness.every((row) => row.every(Boolean));
  }, [correctness, op]);

  const newProblem = (overrideMode = null) => {
    if (overrideMode && typeof overrideMode !== 'string') overrideMode = null;

    // 1) まず演算を選ぶ
    const ops = ["entry", "+", "-", "times"]; // ops = ["entry", "+", "-", "times", "cdot", "det", "inv", "row"];
    const selected = overrideMode ?? mode;
    const chosen = selected === "random" ? ops[randInt(0, ops.length - 1)] : selected;
    setOp(chosen);
    setChecked(false);
  
    // 2) 演算に応じてサイズとオペランドを作る
    let a, b, scalar, resultRows, resultCols, pos;

    if (chosen === "row") {
      // サイズは少し小さめ（2〜3行、2〜3列）にしておくと見やすい
      const rows = randInt(2, 3);
      const cols = randInt(2, 3);
      const a = makeMatrix(rows, cols, -5, 5);
      // ランダムに操作を選択
      const kinds = ["swap", "scale", "add"];
      const kind = kinds[randInt(0, kinds.length - 1)];
      let i, j, kcoef;
      let Cafter = null;
      if (kind === "swap") {
        i = randInt(0, rows - 1);
        do { j = randInt(0, rows - 1); } while (j === i);
        Cafter = swapRows(a, i, j);
      } else if (kind === "scale") {
        i = randInt(0, rows - 1);
        const candidates = [-3,-2,-1,2,3];
        kcoef = candidates[randInt(0, candidates.length - 1)];
        Cafter = scaleRow(a, i, kcoef);
      } else { // "add"
        i = randInt(0, rows - 1);
        do { j = randInt(0, rows - 1); } while (j === i);
        const candidates = [-3,-2,-1,1,2,3];
        kcoef = candidates[randInt(0, candidates.length - 1)];
        Cafter = addRows(a, i, j, kcoef);
      }
      // 状態に反映
      setA(a);
      setB(a); // Bは使わないが既存構造に合わせて保持しておく（未使用）
      setDims({ rows, cols });
      setUser(zeros(rows, cols));
      setChecked(false);
      setOp("row");
      // 正解は C と同じスロットに入れる
      // 既存の C は useMemo で op により計算しているので、そこに合わせるため:
      // 1) 一時的に result を state に持たせてもOKだが、最小変更なら
      //    「row のときは C を a で埋めて、別に正解保持」とするのが楽。
      // ここでは簡単に正解を別 state に入れます。
      setRowAnswer({ kind, i, j, kcoef, Cafter }); // ← 追加の state（下に定義）
      return;
    }

    if (chosen === "inv") {
      const n = randInt(2, 3);
      // det ≠ 0 を満たすまでランダム生成
      let a;
      while (true) {
        a = makeMatrix(n, n, -5, 5);
        if (n === 2) {
          const d = a[0][0]*a[1][1] - a[0][1]*a[1][0];
          if (d !== 0) break;
        } else {
          // 簡易：数値detを見て非ゼロならOK
          const d = a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
                  - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
                  + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
          if (d !== 0) break;
        }
      }
      const aR = a.map(row => row.map(x => R.fromInt(x)));
      const result = (n === 2) ? inverseR2(aR) : inverseR3(aR);
      setAnsR(result); // det, inv, (3×3は cof/adj も)
      setA(a);
      setB(makeMatrix(n,n)); // B はダミー
      setDims({ rows: n, cols: n });
      setUser(Array.from({length:n}, ()=> Array.from({length:n}, ()=> ""))); // 文字列入力
      setChecked(false);
      setUserEntry("");
      setUserDet("");
      return;
    }
    
    if (chosen === "det") {
      const n = randInt(2, 3); // 2〜3 次
      a = makeMatrix(n, n);
      b = makeMatrix(n, n); // ダミー（未使用）
      resultRows = n;
      resultCols = n;
      setUserDet("");
      setA(a);
      setB(b);
      setDims({ rows: resultRows, cols: resultCols });
      setUser(zeros(resultRows, resultCols));
      return;
    }

    if (chosen === "cdot") {
      const rowsA = randInt(1, 3);
      const n = rowsA === 1 ? randInt(2, 3) : randInt(1, 3); // 共通次元
      const colsB = n === 1 ? randInt(2, 3) : randInt(1, 3);
      a = makeMatrix(rowsA, n);
      b = makeMatrix(n, colsB);
      resultRows = rowsA;
      resultCols = colsB;
      setA(a);
      setB(b);
      setDims({ rows: resultRows, cols: resultCols });
      setUser(zeros(resultRows, resultCols));
      return;
    }

    if ( (chosen === "+") || (chosen === "-") ) {
      const rows = randInt(1, 3);
      const cols = rows === 1 ? randInt(2, 3) : randInt(1, 3);
      a = makeMatrix(rows, cols);
      b = makeMatrix(rows, cols);
      resultRows = rows;
      resultCols = cols;
      setA(a);
      setB(b);
      setDims({ rows: resultRows, cols: resultCols });
      setUser(zeros(resultRows, resultCols));
      return;
    }

    if ( chosen === "times" ) {
      const rows = randInt(1, 3);
      const cols = rows === 1 ? randInt(2, 3) : randInt(1, 3);
      a = makeMatrix(rows, cols);
      b = makeMatrix(rows, cols);
      scalar = randInt(-5, 5);
      resultRows = rows;
      resultCols = cols;
      setA(a);
      setB(b);
      setK(scalar);
      setDims({ rows: resultRows, cols: resultCols });
      setUser(zeros(resultRows, resultCols));
      return;
    }

    if (chosen === "entry") {
      const rows = randInt(1, 3);
      const cols = rows === 1 ? randInt(2, 3) : randInt(1, 3);
      a = makeMatrix(rows, cols);
      b = makeMatrix(rows, cols);
      resultRows = rows;
      resultCols = cols;
      pos = { r: randInt(1, a.length), c: randInt(1, a[0].length) };
      resultRows = a.length; resultCols = a[0].length;
      setEntryPos(pos);
      setUserEntry("");
      setA(a);
      setB(b);
      setDims({ rows: resultRows, cols: resultCols });
      setUser(zeros(resultRows, resultCols));
    }
    
  };

  const revealAnswer = () => setChecked(prev => !prev);

  const texA = toBMatrix(A);
  const texB = toBMatrix(B);
  const texC = toBMatrix(C);

  const stepsAlignedTeX = useMemo(() => {
    // 行基本変形
    if (op === "row" && rowAnswer) {
      const { kind, i, j, kcoef, Cafter } = rowAnswer;
      const m = dims.rows, p = dims.cols;
      const lines = [];
      if (kind === "swap") {
        // 影響行 i, j の全列を式で表示
        for (let c = 0; c < p; c++) {
          lines.push(`& {\\small \\color{gray} ( ${i+1}, ${c+1} )} \\quad A'_{${i+1},${c+1}} = A_{${j+1},${c+1}} = ${A[j][c]} \\\\
                                & {\\small \\color{gray} ( ${j+1}, ${c+1} )} \\quad A'_{${j+1},${c+1}} = A_{${i+1},${c+1}} = ${A[i][c]}`);
        }
      } else if (kind === "scale") {
        for (let c = 0; c < p; c++) {
          const a = A[i][c], b = Cafter[i][c];
          lines.push(`& {\\small \\color{gray} ( ${i+1}, ${c+1} )} \\quad A'_{${i+1},${c+1}} = ${kcoef}\\cdot ${wrapNum(a)} = ${b}`);
        }
      } else { // add
        for (let c = 0; c < p; c++) {
          const a = A[i][c], b = A[j][c], out = Cafter[i][c];
          lines.push(`& {\\small \\color{gray} ( ${i+1}, ${c+1} )} \\quad A'_{${i+1},${c+1}} = ${wrapNum(a)} + ${kcoef}\\cdot ${wrapNum(b)} = ${out}`);
        }
      }
      return `\\begin{aligned}
        A &= ${toBMatrix(A)} \\\\
        \\text{Operation} &: \\; ${eroLabelTex(kind, i, j, kcoef)} \\\\
        \\text{Updated entries:} & \\\\
        ${lines.join(" \\\\ ")}
        \\end{aligned}`;
    }
    // 逆行列
    if (op === "inv") {
      const n = dims.rows;
      if (!AnsR) return "";
      if (n === 2) {
        const [[a,b],[c,d]] = A;
        const detNum = (a*d - b*c);
        const detTex = texR(AnsR.det); // 有理数としての det (整数ならそのまま)
        const adjTex = `\\begin{bmatrix} ${d} & ${-b} \\\\ ${-c} & ${a} \\end{bmatrix}`;
        const invTex = AnsR.inv.map(row => row.map(texR).join(" & ")).join(" \\\\ ");
        return `
          \\begin{aligned}
          A &= ${toBMatrix(A)},
          \\quad \\quad \\det(A) = ${detTex},
          \\quad \\quad \\operatorname{adj}(A) = ${adjTex} \\\\
          A^{-1} &= \\frac{1}{\\det(A)}\\cdot\\operatorname{adj}(A)
                  = \\begin{bmatrix} ${invTex} \\end{bmatrix}
          \\end{aligned}
        `;
      } else {
        const detTex = texR(AnsR.det);
        const cofTex = AnsR.cof.map(row => row.map(texR).join(" & ")).join(" \\\\ ");
        const adjTex = AnsR.adj.map(row => row.map(texR).join(" & ")).join(" \\\\ ");
        const invTex = AnsR.inv.map(row => row.map(texR).join(" & ")).join(" \\\\ ");
        return `
          \\begin{aligned}
          A &= ${toBMatrix(A)},
          \\quad \\quad \\det(A) = ${detTex},
          \\quad \\quad \\operatorname{adj}(A) = \\begin{bmatrix} ${adjTex} \\end{bmatrix} \\\\
          A^{-1} &= \\frac{1}{\\det(A)}\\cdot\\operatorname{adj}(A) = \\begin{bmatrix} ${invTex} \\end{bmatrix}
          \\end{aligned}
        `;
      }
    }
    // 行列式
    if (op === "det") {
      const n = A.length;
      if (n === 1) return `{ \\det(A) = ${A[0][0]} }`;
      if (A.length === 2) {
        const [[a,b],[c,d]] = A;
        return `
          \\begin{aligned}
          \\det ${toBMatrix(A)} &= ${wrapNum(a)}\\cdot${wrapNum(d)} - ${wrapNum(b)}\\cdot${wrapNum(c)} \\\\
              &= ${a*d - b*c}
          \\end{aligned}
        `;
      }
      if (A.length === 3) {
        const [[a,b,c],[d,e,f],[g,h,i]] = A;
        const pos = a*e*i + b*f*g + c*d*h;
        const neg = c*e*g + b*d*i + a*f*h;
        return `
          \\begin{aligned}
          \\det ${toBMatrix(A)} &= (${wrapNum(a)}\\cdot${wrapNum(wrapNum(e))}\\cdot${wrapNum(i)}) - (${wrapNum(a)}\\cdot${wrapNum(wrapNum(f))}\\cdot${wrapNum(h)}) \\\\
              &\\quad - (${wrapNum(b)}\\cdot${wrapNum(wrapNum(d))}\\cdot${wrapNum(i)}) + (${wrapNum(b)}\\cdot${wrapNum(wrapNum(f))}\\cdot${wrapNum(g)}) \\\\
              &\\quad + (${wrapNum(c)}\\cdot${wrapNum(wrapNum(d))}\\cdot${wrapNum(h)}) - (${wrapNum(c)}\\cdot${wrapNum(wrapNum(e))}\\cdot${wrapNum(g)}) \\\\
              &= ${pos - neg}
          \\end{aligned}
        `;
      }
      return "";
    }
    // 要素
    if (op === "entry") {
      const ans = A[entryPos.r - 1]?.[entryPos.c - 1];
      return `{ A_{${entryPos.r}${entryPos.c}} = ${ans} }`;
    }
    // そのほか
    const m = dims.rows, p = dims.cols;
    const lines = [];
    for (let r = 0; r < m; r++) {
      const rowParts = [];
      for (let c = 0; c < p; c++) {
        let expr = "";
        if (op === "+") {
          expr = `${wrapNum(A[r][c])} + ${wrapNum(B[r][c])} = ${(C[r][c])}`;
        } else if (op === "-") {
          expr = `${wrapNum(A[r][c])} - ${wrapNum(B[r][c])} = ${(C[r][c])}`;
        } else if (op === "times") {
          expr = `${wrapNum(k)} \\times ${wrapNum(A[r][c])} = ${(C[r][c])}`;
        } else if (op === "cdot") {
          const n = A[0].length;
          const terms = Array.from({ length: n }, (_, t) => `${wrapNum(A[r][t])} \\times ${wrapNum(B[t][c])}`);
          expr = `${terms.join(" + ")} = ${C[r][c]}`;
        }
        const label = `{\\small \\color{gray} (${r + 1}, ${c + 1})}`;
        rowParts.push(`& ${label}\\quad ${expr}`);
      }
      lines.push(rowParts.join(",& "));
    }
    return `\\begin{aligned} ${lines.join(" \\\\ ")} \\end{aligned}`;
  }, [A, B, C, k, dims.rows, dims.cols, op, entryPos]);

  const config = {
    loader: { load: ["input/tex", "output/chtml", "[tex]/color", "[tex]/textmacros"] },
    tex: {
      inlineMath: [["$", "$"], ["\\(", "\\)"]],
      displayMath: [["$$", "$$"], ["\\[", "\\]"]],
      packages: { "[+]": ["base", "ams", "color", "textmacros"] },
    },
  };

  return (
    <MathJaxContext version={3} config={config}>
      <div style={{
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        padding: 24,
        background: "#343434ff",
        color: "#e2e8f0",
        fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, Ubuntu, Cantarell, Helvetica Neue, Arial, 'Apple Color Emoji', 'Segoe UI Emoji'",
      }}>
        <div style={{ width: "min(1100px, 100%)", minWidth: "820px" }}>

          {/* タイトルと設定 */}
          <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <h1 style={{ fontSize: 28, fontWeight: 800 }}>{t("title")} <small>{t("subtitle")}</small></h1>
            <div style={{ display: "flex", gap: 8 }}>

              {/* 問題モード選択 */}
              <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: 14, opacity: 0.9 }}>{t("modeLabel")}:</span>
                <select
                  value={mode}
                  onChange={(e) => { const v = e.target.value; setMode(v); newProblem(v); }}
                  style={{ padding: "0px 12px", height: 36 }}
                >
                  <option value="entry">{t("mode_entry")}</option>
                  <option value="+">{t("mode_add")}</option>
                  <option value="-">{t("mode_sub")}</option>
                  <option value="times">{t("mode_times")}</option>
                  {/* <option value="cdot">{t("mode_cdot")}</option>
                  <option value="det">{t("mode_det")}</option>
                  <option value="inv">{t("mode_inv")}</option>
                  <option value="row">{t("mode_row")}</option> */}
                  <option value="random">{t("mode_random")}</option>
                </select>
              </label>

              {/* 新しい問題を作るボタン */}
              <button onClick={() => newProblem()} style={btnStyle}>{t("newProblem")}</button>

              {/* 言語選択 */}
              <select value={lang} onChange={(e) => setLang(e.target.value)} style={{padding: "0px 20px 0px 20px"}} >
                <option value="ja">日本語</option>
                <option value="en">English</option>
              </select>

            </div>
          </header>

          {/* 問題 */}
          <section style={cardStyle}>

            {/* 問題文 */}
            <h2 style={sectionTitle}>{t("problem")}</h2>
            <p style={{ marginBottom: 12 }}>
              {op === "entry" ? t("entryQuestion", { r: entryPos.r, c: entryPos.c })
                : op === "det" ? t("detQuestion")
                : t("instruction")}
            </p>

            {/* 問題の左辺 */}
            <div style={{ marginLeft: 48, display: "flex", alignItems: "center", gap: 12, overflowX: "hidden", overflowY: "hidden" }}>
              {     op === "row" ? (
                  <>
                    <MathJax dynamic>
                      {`\\[ 
                        ${toBMatrix(A)}
                        \\xrightarrow
                        {\\,${eroLabelTex(rowAnswer?.kind, rowAnswer?.i, rowAnswer?.j, rowAnswer?.kcoef)}\\,}
                      \\]`}
                    </MathJax>
                  </>
                ) : op === "inv" ? (
                  <>
                    <MathJax dynamic>{`\\[ A = ${toBMatrix(A)} \\]`}</MathJax>
                    <MathJax dynamic>{`\\[ \\quad , \\quad A^{-1}  \\]`}</MathJax>
                  </>
                ) : op === "det" ? (
                  <>
                    <MathJax dynamic>{`\\[ \\det ${toBMatrix(A)} \\]`}</MathJax>
                  </>
                ) : op === "cdot" ? (
                  <>
                    <MathJax dynamic>{`\\[ ${texA} \\]`}</MathJax>
                    <MathJax dynamic>{`\\[ ${texB} \\]`}</MathJax>
                  </>
                ) : op === "entry" ? (
                  <>
                    <MathJax dynamic>{`\\[ ${texA} \\]`}</MathJax>
                  </>
                ) : op === "times" ? (
                  <>
                    <MathJax dynamic>{`\\[ ${k} \\]`}</MathJax>
                    <MathJax dynamic>{`\\[ \\times \\]`}</MathJax>
                    <MathJax dynamic>{`\\[ ${texA} \\]`}</MathJax>
                  </>
                ) : (
                  <>
                    <MathJax dynamic>{`\\[ ${texA} \\]`}</MathJax>
                    <MathJax dynamic>{`\\[ ${op} \\]`}</MathJax>
                    <MathJax dynamic>{`\\[ ${texB} \\]`}</MathJax>
                  </>
                )
              }

              {/* 等号 */}
              {op === "entry" ? <MathJax dynamic>{`\\[ \\qquad\\qquad\\qquad\\qquad \\]`}</MathJax> : <MathJax dynamic>{`\\[ = \\]`}</MathJax>}
              
              {/* 問題の回答欄 */}
              {   op === "inv" ? (
                <div style={{ padding: 6, borderRadius: 12, background: "#0b1222" }}>
                  <BracketBox rows={dims.rows}>
                    <div style={{ display: "grid", gridTemplateColumns: `repeat(${dims.cols}, 190px)`, gap: 16 }}>
                      {user.map((row, r) =>
                        row.map((v, c) => (
                          <FractionCell
                            key={`inv-${r}-${c}`}
                            id={`inv-${r}-${c}`}
                            value={v}
                            onChange={(val) => {
                              const next = user.map(x => x.slice());
                              next[r][c] = val;
                              setUser(next);
                            }}
                            isCorrect={checked ? correctness?.[r]?.[c] : undefined}
                            showCheck={checked}
                          />
                        ))
                      )}
                    </div>
                  </BracketBox>
                </div>
              ) : op === "det" ? (
                <div style={{ marginTop: 12 }}>
                  <NumberCell
                    id={`det`}
                    value={userDet}
                    onChange={(val) => setUserDet(val)}
                    isCorrect={checked ? correctness : undefined}
                    showCheck={checked}
                  />
                </div>
              ) : op === "entry" ? (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <NumberCell
                    id={`entry`}
                    value={userEntry}
                    onChange={(val) => setUserEntry(val)}
                    isCorrect={checked ? correctness : undefined}
                    showCheck={checked}
                  />
                </div>
              ) : (
                <div style={{ padding: 6, borderRadius: 12, background: "#0b1222" }}>
                  <BracketBox rows={dims.rows}>
                    <MatrixInput
                      rows={dims.rows}
                      cols={dims.cols}
                      values={user}
                      setValues={setUser}
                      correctness={correctness}
                      showCheck={checked}
                    />
                  </BracketBox>
                </div>
              )}
            </div>
            
            {/* 答え合わせ・隠す */}
            <div style={{ display: "grid", justifyContent: "end" }}>
              <button onClick={revealAnswer} style={{ ...btnStyle, background: "#22c55e" }}>
                {checked ? t("hideAnswer") : t("checkAnswer")}
              </button>
            </div>
          </section>

          {/* 答え */}
          <section style={cardStyle}>

            {/* 正解・不正解 */}
            {checked &&
              <div style={{ marginTop: 6 }}>
                {allCorrect ? (
                  <p style={{ color: "#22c55e", fontWeight: 900, fontSize: 22 }}>
                    {(op === "entry" || op === "det") ? t("correct") : t("allCorrect")}
                  </p>
                ) : (
                  <p style={{ color: "#f43f5e", fontWeight: 900, fontSize: 22 }}>
                    {(op === "entry" || op === "det") ? t("incorrect") : t("someIncorrect")}
                  </p>
                )}
              </div>
            }

            <h2 style={sectionTitle}>{t("answerSteps")}</h2>
            {!checked && <p style={{ opacity: 0.85 }}>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{t("pressCheck")}</p>}
            {checked && (
              <div>

                {/* 答え */}
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12, overflowX: "hidden", overflowY: "hidden" }}>
                  <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                  {   op === "row" ? (
                    <>
                      <MathJax dynamic>{`\\[ ${toBMatrix(A)} \\]`}</MathJax>
                      <MathJax dynamic>{`\\[ \\xrightarrow{\\,${eroLabelTex(rowAnswer?.kind ?? "", rowAnswer?.i ?? 0, rowAnswer?.j ?? 0, rowAnswer?.kcoef ?? 0)}\\,} \\]`}</MathJax>
                    </>
                  ) : op === "inv" ? (
                    <>
                      <MathJax dynamic>{`\\[ A^{-1} \\]`}</MathJax>
                    </>
                  ) : op === "det" ? (
                    <>
                      <MathJax dynamic>{`\\[ \\det ${toBMatrix(A)} \\]`}</MathJax>
                    </>
                  ) : op === "entry" ? (
                    <>
                      <MathJax dynamic>{`\\[ ${texA} \\]`}</MathJax>
                      <MathJax dynamic>{`\\[ \\Rightarrow \\]`}</MathJax>
                      <MathJax dynamic>{`\\[ A_{${entryPos.r}${entryPos.c}} \\]`}</MathJax>
                    </>
                  ) : op === "times" ? (
                    <>
                      <MathJax dynamic>{`\\[ ${k} \\]`}</MathJax>
                      <MathJax dynamic>{`\\[ \\times \\]`}</MathJax>
                      <MathJax dynamic>{`\\[ ${texA} \\]`}</MathJax>
                    </>
                  ) : op === "cdot" ? (
                    <>
                      <MathJax dynamic>{`\\[ ${texA} \\]`}</MathJax>
                      <MathJax dynamic>{`\\[ ${texB} \\]`}</MathJax>
                    </>
                  ) : (
                    <>
                      <MathJax dynamic>{`\\[ ${texA} \\]`}</MathJax>
                      <MathJax dynamic>{`\\[ ${op} \\]`}</MathJax>
                      <MathJax dynamic>{`\\[ ${texB} \\]`}</MathJax>
                    </>
                  )}

                  <MathJax dynamic>{`\\[ = \\]`}</MathJax>

                  <MathJax dynamic>
                    {`\\[ ${
                      op === "row" ? (rowAnswer ? toBMatrix(rowAnswer.Cafter) : ""
                      ) : op === "inv" ? (AnsR?.inv ? toBMatrixR(AnsR.inv) : `\\text{(not invertible)}`
                      ) : op === "entry" ? (A[entryPos.r - 1][entryPos.c - 1]
                      ) : op === "det" ? detValue : texC
                    } \\]`}
                  </MathJax>
                </div>

                {/* 計算過程 */}
                <div style={{ borderTop: "1px dashed #334155", paddingTop: 12 }}>
                  <MathJax dynamic>{`\\[ ${stepsAlignedTeX} \\]`}</MathJax>
                </div>

              </div>
            )}
          </section>

          {/* ヒント */}
          <footer style={{ marginTop: 16, opacity: 0.8, fontSize: 14 }}>
            {   op === "row" ? (
              <>
                <p>{t("hintRow")}</p>
                <MathJax dynamic>
                  {`
                    \\[
                      \\text{Examples: } R_i \\leftrightarrow R_j,\\quad R_i \\leftarrow k R_i,\\quad R_i \\leftarrow R_i + k R_j
                    \\]
                  `}
                </MathJax>
              </>
            ) : op === "inv" ? (
              <>
                <p>{t("hintInv")}</p>
                <MathJax dynamic>
                  {`
                    \\[ 
                        A_{2} = ${toBMatrix([["a","b"],["c","d"]])}
                        \\, , \\quad
                        \\operatorname{det}(A_{2}) = ad - bc
                        \\, , \\quad
                        A_{2}^{-1} = \\frac{1}{\\operatorname{det}(A_{2})}\\cdot${toBMatrix([["d","-b"],["-c","a"]])} \\, .
                    \\]
                    \\[ \\small
                        A_{3} = ${toBMatrix([["a","b","c"],["d","e","f"],["g","h","i"]])}
                          \\, , \\quad
                        \\operatorname{det}(A_{3}) =
                          a\\cdot\\operatorname{det}${toBMatrix([["e","f"],["h","i"]])}
                          - b\\cdot\\operatorname{det}${toBMatrix([["d","f"],["g","i"]])}
                          + c\\cdot\\operatorname{det}${toBMatrix([["d","e"],["g","h"]])}
                          \\, , \\quad
                        \\operatorname{adj}(A_{3}) =
                          ${toBMatrix([
                            [toDetBMatrix([["e","f"],["h","i"]]), "-"+toDetBMatrix([["b","c"],["h","i"]]), toDetBMatrix([["b","c"],["e","f"]])],
                            ["-"+toDetBMatrix([["d","f"],["g","i"]]), toDetBMatrix([["a","c"],["g","i"]]), "-"+toDetBMatrix([["a","c"],["d","f"]])],
                            [toDetBMatrix([["d","e"],["g","h"]]), "-"+toDetBMatrix([["a","b"],["g","h"]]), toDetBMatrix([["a","b"],["d","e"]])]
                          ])}
                          \\, , \\quad
                        A_{3}^{-1} = \\frac{1}{\\operatorname{det}(A_{3})}\\cdot\\operatorname{adj}(A_{3}) \\, .
                    \\]
                  `}
                </MathJax>
              </>
            ) : op === "det" ? (
              <>
                <p>{t("hintDet")}</p>
                <MathJax dynamic>
                  {`
                    \\[ \\operatorname{det}${toBMatrix([["a","b"],["c","d"]])} = ad - bc
                        \\, , \\qquad
                        \\operatorname{det}${toBMatrix([["a","b","c"],["d","e","f"],["g","h","i"]])}
                        = a\\cdot\\operatorname{det}${toBMatrix([["e","f"],["h","i"]])}
                          - b\\cdot\\operatorname{det}${toBMatrix([["d","f"],["g","i"]])}
                          + c\\cdot\\operatorname{det}${toBMatrix([["d","e"],["g","h"]])} \\]
                  `}
                </MathJax>
              </>
            ) : op === "cdot" ? (
              <>
                <p>{t("hintMulDims")} {t("hintMulFormula")}</p>
                <MathJax dynamic>{`\\[
                  C = AB
                  \\, , \\qquad
                  C_{ij}=\\sum_{k=1}^{n} A_{ik}\\,B_{kj} 
                \\]`}</MathJax>
              </>
            ) : op === "entry" ? (
              <p>{t("entryHint")}</p>
            ) : (
              <p>{t("hintAddSubScalar")}</p>
            )}
          </footer>

        </div>
      </div>
    </MathJaxContext>
  );
}

const btnStyle = {
  background: "#3b82f6",
  color: "white",
  padding: "10px 14px",
  borderRadius: 10,
  border: "none",
  fontWeight: 700,
  cursor: "pointer",
  margin: "8px",
};

const cardStyle = {
  background: "#111827",
  border: "1px solid #1f2937",
  borderRadius: 16,
  padding: 16,
  marginBottom: 14,
};

const sectionTitle = {
  marginTop: 0,
  marginBottom: 10,
  fontSize: 18,
  fontWeight: 800,
};
