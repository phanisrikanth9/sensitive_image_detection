/* Browser port of detectors.py, the text classifier and the MobileNetV2 runner.
 * Kept line-for-line close to the Python so both give the same answers
 * (checked by tests/test_web_parity.js against the Python output).
 * Works in the browser (window.PrivacyEngine) and in Node (module.exports).
 */
(function (root) {
  "use strict";

  // ------------------------------------------------------------------ checksums
  function luhnValid(num) {
    let total = 0;
    for (let i = 0; i < num.length; i++) {
      let d = Number(num[num.length - 1 - i]);
      if (i % 2 === 1) { d *= 2; if (d > 9) d -= 9; }
      total += d;
    }
    return total % 10 === 0;
  }

  const VD = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]];
  const VP = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]];
  function verhoeffValid(num) {
    let c = 0;
    const r = num.split("").reverse();
    for (let i = 0; i < r.length; i++) c = VD[c][VP[i % 8][Number(r[i])]];
    return c === 0;
  }

  // ------------------------------------------------------------------ validators
  const digits = (s) => s.replace(/\D/g, "");

  function validCreditCard(s) {
    const d = digits(s);
    if (d.length < 13 || d.length > 19 || !luhnValid(d)) return false;
    return /^(4|5[1-5]|2(2[2-9]|[3-6]\d|7[01]|720)|3[47]|6(011|5|0|52)|35|3[068])/.test(d);
  }
  function validAadhaar(s) {
    const d = digits(s);
    return d.length === 12 && !"01".includes(d[0]) && verhoeffValid(d);
  }
  function validSsn(s) {
    const d = digits(s), area = d.slice(0, 3), group = d.slice(3, 5), serial = d.slice(5);
    return d.length === 9 && area !== "000" && area !== "666" && area[0] !== "9" && group !== "00" && serial !== "0000";
  }
  const validPan = (s) => "PCHFATBLJG".includes(s[3]);
  function validPhone(s) {
    let d = digits(s);
    if (d.length === 12 && d.startsWith("91")) d = d.slice(2);
    else if (d.length === 11 && d[0] === "0") d = d.slice(1);
    if (d.length === 10 && "6789".includes(d[0]) && new Set(d).size > 2) return true;
    if (d.length === 11 && d[0] === "1") d = d.slice(1);
    return d.length === 10 && "23456789".includes(d[0]) && "23456789".includes(d[3]);
  }

  // ------------------------------------------------------------------ patterns (same order as Python)
  const PATTERNS = [
    ["email", /\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b/g],
    ["credit_card", /\b(?:\d[ -]?){12,18}\d\b/g],
    ["aadhaar", /\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b/g],
    ["pan", /\b[A-Z]{5}\d{4}[A-Z]\b/g],
    ["ssn", /\b\d{3}[- ]\d{2}[- ]\d{4}\b|\b[Ss][Ss][Nn]\b\W{0,3}\d{9}\b/g],
    ["phone", /(?<!\w)(?:\+?91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}\b|(?:(?<!\w)\+?1[ .-]?)?(?:\(\d{3}\)\s?|\b\d{3}[ .-])\d{3}[ .-]\d{4}\b/g],
    ["credential", /\b(?:password|passwd|pwd|passcode|api[ _-]?key|secret key|token)\b\s*(?:is)?\s*[:=]?\s*(?=\S*[\d!@#$%^&*])[^\s,;]{4,}/gi],
    ["pin_otp_cvv", /\b(?:pin|otp|cvv|cvc|passcode|password)\b\s*(?:no\.?|code|number)?\s*[:#=-]?\s*\d{3,6}\b/gi],
  ];
  const PII_TYPES = PATTERNS.map((p) => p[0]);
  const VALIDATORS = { credit_card: validCreditCard, aadhaar: validAadhaar, ssn: validSsn, pan: validPan, phone: validPhone };
  const AADHAAR_CONTEXT = /aadha+r|\buid\b|unique identification|government of india/i;
  const LABEL = /^\D*?(?:pin|otp|cvv|cvc|passcode|password|passwd|pwd|key|token|code|number|no\.?)\b\s*(?:is)?\s*[:#=-]?\s*/i;

  const LABELS = {
    email: "Email address", credit_card: "Card number", aadhaar: "Aadhaar number", pan: "PAN",
    ssn: "Social Security number", phone: "Phone number", credential: "Password / key", pin_otp_cvv: "PIN / OTP / CVV",
  };

  function mask(value, keep = 4) {
    if (value.includes("@")) { const at = value.indexOf("@"); return value[0] + "***@" + value.slice(at + 1); }
    const alnum = [];
    for (let i = 0; i < value.length; i++) if (/[\p{L}\p{N}]/u.test(value[i])) alnum.push(i);
    const hide = new Set(alnum.length > keep ? alnum.slice(0, alnum.length - keep) : alnum);
    return value.split("").map((ch, i) => (hide.has(i) ? "X" : ch)).join("");
  }

  const normalise = (t) => (t || "").split(/\s+/).filter(Boolean).join(" ");

  function findPii(text) {
    text = normalise(text);
    if (!text) return [];
    const findings = [], taken = [];
    for (const [kind, re] of PATTERNS) {
      re.lastIndex = 0;
      let m;
      while ((m = re.exec(text)) !== null) {
        if (m[0].length === 0) { re.lastIndex++; continue; }
        const start = m.index, end = m.index + m[0].length, value = m[0].trim();
        if (taken.some(([s, e]) => start < e && end > s)) continue;
        const v = VALIDATORS[kind];
        if (kind === "aadhaar") {
          const labelled = AADHAAR_CONTEXT.test(text.slice(Math.max(0, start - 40), start));
          const spaced = !/^\d+$/.test(value);
          if (!(labelled || (spaced && v(value)))) continue;
        } else if (v && !v(value)) continue;
        let masked;
        if (kind === "pin_otp_cvv" || kind === "credential") {
          const lab = value.match(LABEL);
          const cut = lab ? lab[0].length : 0;
          masked = value.slice(0, cut) + "X".repeat(value.length - cut);
        } else masked = mask(value);
        findings.push({ type: kind, label: LABELS[kind], value, masked, start, end });
        taken.push([start, end]);
      }
    }
    return findings.sort((a, b) => a.start - b.start);
  }

  function maskText(text) {
    text = normalise(text);
    for (const f of findPii(text).sort((a, b) => b.start - a.start))
      text = text.slice(0, f.start) + f.masked + text.slice(f.end);
    return text;
  }

  // ------------------------------------------------------------------ text classifier
  // Re-implements sklearn's TfidfVectorizer (word + char_wb) and LogisticRegression.
  function wordNgrams(text, lo, hi) {
    const toks = text.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || [];
    const out = [];
    for (let n = lo; n <= hi; n++)
      for (let i = 0; i + n <= toks.length; i++) out.push(toks.slice(i, i + n).join(" "));
    return out;
  }
  function charWbNgrams(text, lo, hi) {
    const out = [];
    for (let w of text.toLowerCase().split(/\s+/).filter(Boolean)) {
      w = " " + w + " ";
      const len = [...w].length, chars = [...w];
      for (let n = lo; n <= hi; n++) {
        let offset = 0;
        out.push(chars.slice(offset, offset + n).join(""));
        while (offset + n < len) { offset++; out.push(chars.slice(offset, offset + n).join("")); }
        if (offset === 0) break;
      }
    }
    return out;
  }
  function tfidfDot(grams, vec) {
    const counts = new Map();
    for (const g of grams) if (vec.terms[g]) counts.set(g, (counts.get(g) || 0) + 1);
    let norm = 0, dot = 0;
    for (const [g, c] of counts) {
      const [idf, coef] = vec.terms[g];
      const v = (vec.sublinear_tf ? 1 + Math.log(c) : c) * idf;
      norm += v * v; dot += v * coef;
    }
    return norm > 0 ? dot / Math.sqrt(norm) : 0;
  }

  function makeTextModel(model) {
    const kw = Object.values(model.keyword_groups).map((p) => new RegExp(p, "gi"));
    return {
      threshold: model.threshold,
      score(text) {
        text = text || "";
        let z = model.intercept;
        z += tfidfDot(wordNgrams(text, ...model.words.ngram_range), model.words);
        z += tfidfDot(charWbNgrams(text, ...model.chars.ngram_range), model.chars);
        const counts = PII_TYPES.map((t) => 0);
        for (const f of findPii(text)) counts[PII_TYPES.indexOf(f.type)]++;
        for (const re of kw) counts.push((text.match(re) || []).length);
        counts.forEach((c, i) => { z += Math.log1p(c) * model.pattern_coef[i]; });
        return 1 / (1 + Math.exp(-z));
      },
    };
  }

  // ------------------------------------------------------------------ image model (needs global tf)
  async function loadImageModel(tf, graph, buffer) {
    const all = new Float32Array(buffer);
    const W = (layer) => layer.weights.map((w) => {
      const size = w.shape.reduce((a, b) => a * b, 1);
      return tf.tensor(all.subarray(w.offset, w.offset + size), w.shape);
    });
    const ops = graph.layers.map((l) => ({ ...l, w: W(l) }));
    function forward(x) {
      const out = { input_layer: x };
      for (const l of ops) {
        const a = out[l.inputs[0]];
        let y;
        switch (l.type) {
          case "Conv2D": y = tf.conv2d(a, l.w[0], l.strides, l.padding); if (l.use_bias) y = tf.add(y, l.w[1]); break;
          case "DepthwiseConv2D": y = tf.depthwiseConv2d(a, l.w[0], l.strides, l.padding); if (l.use_bias) y = tf.add(y, l.w[1]); break;
          case "BatchNormalization": y = tf.batchNorm(a, l.w[2], l.w[3], l.w[1], l.w[0], l.epsilon); break;
          case "ReLU": y = l.max_value != null ? tf.clipByValue(a, 0, l.max_value) : tf.relu(a); break;
          case "ZeroPadding2D": y = tf.pad(a, [[0, 0], l.padding[0], l.padding[1], [0, 0]]); break;
          case "Add": y = tf.add(a, out[l.inputs[1]]); break;
          case "GlobalAveragePooling2D": y = tf.mean(a, [1, 2]); break;
          case "Dense": y = tf.add(tf.matMul(a, l.w[0]), l.w[1]); if (l.activation === "sigmoid") y = tf.sigmoid(y); break;
          default: throw new Error("Unsupported layer " + l.type);
        }
        out[l.name] = y;
      }
      return out[ops[ops.length - 1].name];
    }
    return {
      // pixels: Float32Array of 224*224*3 RGB values in 0..255
      score(pixels) {
        return tf.tidy(() => {
          const x = tf.tensor(pixels, [1, 224, 224, 3]).div(255);
          return forward(x).dataSync()[0];
        });
      },
    };
  }

  const api = { findPii, maskText, mask, makeTextModel, loadImageModel, PII_TYPES, LABELS, luhnValid, verhoeffValid };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PrivacyEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
