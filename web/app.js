/* Front end for the Visual Privacy Risk Detector.
 *
 * Two modes, picked automatically:
 *  - server:  the page is served by app.py (Flask). Images go to /upload and the
 *             Python pipeline (PaddleOCR + TensorFlow + scikit-learn) answers.
 *  - browser: anywhere else (GitHub Pages, a plain file server). Everything runs on
 *             this device: Tesseract.js for OCR, engine.js for the ID checks and the
 *             text model, TensorFlow.js for the MobileNetV2 image model.
 */
(function () {
  "use strict";
  const E = window.PrivacyEngine;
  const $ = (id) => document.getElementById(id);
  const IMAGE_THRESHOLD = 0.5;
  const MAX_OCR_SIDE = 2000;
  const abs = (p) => new URL(p, location.href).href;

  let mode = "browser";
  let results = [];
  const engine = { ocr: null, text: null, image: null, loading: null };

  // ------------------------------------------------------------------ theme
  const themeKey = "vprd-theme";
  try { const t = localStorage.getItem(themeKey); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
  $("themeBtn").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem(themeKey, next); } catch (e) {}
  });

  // ------------------------------------------------------------------ status line
  function status(msg, frac) {
    $("status").hidden = false;
    $("status").classList.remove("error");
    $("statusMsg").textContent = msg;
    if (frac != null) $("statusBar").style.width = Math.round(frac * 100) + "%";
  }
  function statusError(msg) { status(msg); $("status").classList.add("error"); }
  const statusDone = () => { $("status").hidden = true; };

  // ------------------------------------------------------------------ mode detection
  async function detectMode() {
    if (location.protocol === "file:") return;
    try {
      const r = await fetch("health", { cache: "no-store" });
      const ct = r.headers.get("content-type") || "";
      if (r.ok && ct.includes("json")) {
        const h = await r.json();
        if ("image_model" in h) {
          mode = "server";
          $("modeText").textContent = "Connected to Python server";
          $("privacyNote").textContent = "Images are sent to the local Python server (app.py), analysed in memory and never saved.";
          $("ocrTech").textContent = "PaddleOCR";
        }
      }
    } catch (e) { /* static hosting: stay in browser mode */ }
  }

  // ------------------------------------------------------------------ lazy model loading (browser mode)
  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = () => rej(new Error("Could not load " + src));
      document.head.appendChild(s);
    });
  }

  function loadEngines() {
    if (engine.loading) return engine.loading;
    engine.loading = (async () => {
      status("Loading the text model…", 0.05);
      engine.text = E.makeTextModel(await (await fetch("web/text_model.json")).json());

      status("Loading the image model…", 0.15);
      await loadScript("web/vendor/tf.min.js");
      let gpu = false;
      try { gpu = await tf.setBackend("webgl"); } catch (e) { gpu = false; }
      if (!gpu) await tf.setBackend("cpu");
      const [graph, weights] = await Promise.all([
        fetch("web/image_model/graph.json").then((r) => r.json()),
        fetch("web/image_model/weights.bin").then((r) => r.arrayBuffer()),
      ]);
      engine.image = await E.loadImageModel(tf, graph, weights);

      status("Loading the OCR engine…", 0.4);
      await loadScript("web/vendor/tesseract.min.js");
      engine.ocr = await Tesseract.createWorker("eng", 1, {
        workerPath: abs("web/vendor/worker.min.js"),
        corePath: abs("web/vendor/tesseract-core"),
        langPath: abs("web/tessdata"),
        gzip: true,
        logger: (m) => {
          if (m.status === "recognizing text") status(currentLabel + ": reading text…", 0.5 + 0.4 * m.progress);
        },
      });
    })();
    engine.loading.catch(() => { engine.loading = null; });
    return engine.loading;
  }
  let currentLabel = "";

  // ------------------------------------------------------------------ browser pipeline
  async function analyzeInBrowser(file) {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, MAX_OCR_SIDE / Math.max(bmp.width, bmp.height));
    const w = Math.round(bmp.width * scale), h = Math.round(bmp.height * scale);
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
    ctx.drawImage(bmp, 0, 0, w, h);

    // 1) whole-image score (same as Python: stretch to 224x224, pixels / 255)
    const small = document.createElement("canvas");
    small.width = 224; small.height = 224;
    const sctx = small.getContext("2d");
    sctx.drawImage(bmp, 0, 0, 224, 224);
    const px = sctx.getImageData(0, 0, 224, 224).data;
    const rgb = new Float32Array(224 * 224 * 3);
    for (let i = 0, j = 0; i < px.length; i += 4) { rgb[j++] = px[i]; rgb[j++] = px[i + 1]; rgb[j++] = px[i + 2]; }
    const imageScore = engine.image.score(rgb);

    // 2) OCR -> words with boxes, joined into one text with known character offsets
    const { data } = await engine.ocr.recognize(canvas);
    const words = [];
    let text = "";
    for (const line of data.lines || []) {
      for (const wd of line.words || []) {
        const t = (wd.text || "").trim();
        if (!t || wd.confidence < 20) continue;
        if (text) text += " ";
        words.push({ start: text.length, end: text.length + t.length, bbox: wd.bbox });
        text += t;
      }
    }

    // 3) ID checks and 4) text model
    const findings = E.findPii(text);
    const textScore = text.trim().length >= 3 ? engine.text.score(text) : null;

    // 5) redaction: cover exactly the words that hold each finding (the secret part only
    //    for "PIN: 1234" / "password: x" so the label stays readable)
    let redactedUrl = null;
    if (findings.length) {
      ctx.fillStyle = "#000";
      for (const f of findings) {
        const tail = f.masked.search(/X+$/);
        const from = (f.type === "pin_otp_cvv" || f.type === "credential") && tail > 0 ? f.start + tail : f.start;
        for (const wd of words) {
          if (wd.start < f.end && wd.end > from) {
            const b = wd.bbox, pad = Math.max(3, (b.y1 - b.y0) * 0.15);
            ctx.fillRect(b.x0 - pad, b.y0 - pad, b.x1 - b.x0 + 2 * pad, b.y1 - b.y0 + 2 * pad);
          }
        }
      }
      redactedUrl = canvas.toDataURL("image/jpeg", 0.9);
    }
    return decide({
      imageScore, textScore, textThreshold: engine.text.threshold,
      pii: findings.map((f) => ({ type: f.type, label: f.label, masked: f.masked })),
      text: E.maskText(text), redactedUrl, wordCount: words.length,
    });
  }

  function decide(r) {
    const imageFlag = r.imageScore != null && r.imageScore >= IMAGE_THRESHOLD;
    const textFlag = r.textScore != null && r.textScore >= r.textThreshold;
    const reasons = [];
    if (r.pii.length) reasons.push("Found " + [...new Set(r.pii.map((p) => p.label.toLowerCase()))].join(", "));
    if (textFlag) reasons.push(`Text reads as sensitive (${pct(r.textScore)} confident)`);
    if (imageFlag) reasons.push(`Image looks like a sensitive document (${pct(r.imageScore)})`);
    const sensitive = imageFlag || textFlag || r.pii.length > 0;
    return { ...r, imageFlag, textFlag, reasons, sensitive, risk: r.pii.length ? "high" : sensitive ? "medium" : "low" };
  }
  const pct = (x) => Math.round(x * 100) + "%";

  // ------------------------------------------------------------------ server pipeline
  async function analyzeOnServer(files) {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    const r = await fetch("upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`Server answered ${r.status}`);
    const out = await r.json();
    return out.map((s) => s.error ? { error: s.error } : decide({
      imageScore: s.image ? s.image.score : null,
      textScore: s.text ? s.text.score : null,
      textThreshold: s.text && s.text.threshold != null ? s.text.threshold : 0.5,
      pii: (s.pii || []).map((p) => ({ type: p.type, label: E.LABELS[p.type] || p.type, masked: p.masked })),
      text: s.extracted_text || "", redactedUrl: s.redacted_image || null,
    }));
  }

  // ------------------------------------------------------------------ run
  let busy = false;
  async function run(files, labels) {
    files = files.filter((f) => f.type.startsWith("image/"));
    if (!files.length || busy) return;
    busy = true;
    try {
      if (mode === "server") {
        status(`Sending ${files.length} image${files.length > 1 ? "s" : ""} to the server…`, 0.3);
        const out = await analyzeOnServer(files);
        out.forEach((r, i) => addResult(files[i], labels ? labels[i] : files[i].name, r));
      } else {
        await loadEngines();
        for (let i = 0; i < files.length; i++) {
          currentLabel = files.length > 1 ? `Image ${i + 1} of ${files.length}` : "Analysing";
          status(currentLabel + "…", 0.5);
          const t0 = performance.now();
          let r;
          try { r = await analyzeInBrowser(files[i]); r.ms = Math.round(performance.now() - t0); }
          catch (e) { console.error(e); r = { error: "Could not read this image. Try a PNG or JPG." }; }
          addResult(files[i], labels ? labels[i] : files[i].name, r);
        }
      }
      statusDone();
    } catch (e) {
      console.error(e);
      statusError(mode === "server"
        ? "The server did not answer. Check that app.py is still running, then try again."
        : "The models could not load. Check your connection and reload the page.");
    } finally { busy = false; }
  }

  // ------------------------------------------------------------------ rendering
  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") n.className = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (v != null) n.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(String(k)));
    return n;
  }

  function meter(label, score, threshold) {
    if (score == null) return el("div", { class: "meter" }, el("span", {}, label), el("span", { class: "na" }, "not available"));
    const fill = el("span", { class: "fill" + (score >= threshold ? " on" : "") });
    fill.style.width = Math.max(2, score * 100) + "%";
    const tick = el("span", { class: "tick", title: "Threshold " + threshold });
    tick.style.left = `calc(${threshold * 100}% - 1px)`;
    return el("div", { class: "meter" }, el("span", {}, label), el("span", { class: "track" }, fill, tick), el("span", { class: "val" }, pct(score)));
  }

  const HEADLINE = { high: "Contains personal data", medium: "Possibly sensitive", low: "Looks safe to share" };

  function addResult(file, name, r) {
    const originalUrl = URL.createObjectURL(file);
    r.name = name;
    results.push(r);

    if (r.error) {
      $("results").prepend(el("article", { class: "card" },
        el("div", { class: "preview" }, el("img", { src: originalUrl, alt: "" })),
        el("div", { class: "details" }, el("div", { class: "verdict" }, el("h3", {}, name)), el("p", { class: "err" }, r.error))));
      updateSummary();
      return;
    }

    const img = el("img", { src: r.redactedUrl || originalUrl, alt: r.redactedUrl ? "Redacted copy of " + name : name });
    const previewKids = [img];
    if (r.redactedUrl) {
      const bRed = el("button", { type: "button", "aria-pressed": "true" }, "Redacted");
      const bOrig = el("button", { type: "button", "aria-pressed": "false" }, "Original");
      bRed.addEventListener("click", () => { img.src = r.redactedUrl; bRed.setAttribute("aria-pressed", "true"); bOrig.setAttribute("aria-pressed", "false"); });
      bOrig.addEventListener("click", () => { img.src = originalUrl; bOrig.setAttribute("aria-pressed", "true"); bRed.setAttribute("aria-pressed", "false"); });
      previewKids.unshift(el("div", { class: "seg", role: "group", "aria-label": "Preview" }, bRed, bOrig));
      previewKids.push(el("p", { class: "note" }, "Black boxes cover each detected value."));
    }

    const found = r.pii.length
      ? el("table", { class: "found" },
          el("thead", {}, el("tr", {}, el("th", {}, "Detected"), el("th", {}, "Value (masked)"))),
          el("tbody", {}, r.pii.map((p) => el("tr", {}, el("td", {}, p.label), el("td", {}, el("code", {}, p.masked))))))
      : el("p", { class: "none" }, "No ID numbers, phone numbers, emails or passwords found.");

    const actions = el("div", { class: "card-actions" });
    if (r.redactedUrl) {
      const base = name.replace(/\.[^.]+$/, "");
      actions.append(el("a", { class: "btn primary", href: r.redactedUrl, download: base + "-redacted.jpg" }, "Download redacted copy"));
    }

    const meta = [];
    if (r.wordCount != null) meta.push(`${r.wordCount} words read`);
    if (r.ms != null) meta.push(`${(r.ms / 1000).toFixed(1)} s`);

    $("results").prepend(el("article", { class: "card " + r.risk },
      el("div", { class: "preview" }, previewKids),
      el("div", { class: "details" },
        el("div", { class: "verdict" },
          el("span", { class: "badge " + r.risk }, r.risk + " risk"),
          el("h3", {}, HEADLINE[r.risk]),
          el("span", { class: "file" }, [name, ...meta].join(" · "))),
        r.reasons.length ? el("ul", { class: "reasons" }, r.reasons.map((t) => el("li", {}, t))) : null,
        found,
        el("div", { class: "meters" },
          meter("Text model", r.textScore, r.textThreshold),
          meter("Image model", r.imageScore, IMAGE_THRESHOLD),
          el("p", { class: "legend" }, "The tick on each bar is that model's decision threshold.")),
        el("details", { class: "text" },
          el("summary", {}, "Text found in the image (private values masked)"),
          el("pre", {}, r.text || "No readable text.")),
        actions.childNodes.length ? actions : null)));
    updateSummary();
  }

  function updateSummary() {
    const ok = results.filter((r) => !r.error);
    $("summary").hidden = results.length === 0;
    const n = (k) => ok.filter((r) => r.risk === k).length;
    $("cHigh").textContent = n("high") + " high";
    $("cMed").textContent = n("medium") + " medium";
    $("cLow").textContent = n("low") + " low";
  }

  // ------------------------------------------------------------------ exports
  function download(name, text, type) {
    const a = el("a", { href: URL.createObjectURL(new Blob([text], { type })), download: name });
    document.body.append(a); a.click(); a.remove();
  }
  const exportRows = () => results.map((r) => ({
    file: r.name, risk: r.error ? "error" : r.risk, sensitive: !!r.sensitive, reasons: (r.reasons || []).join("; "),
    detected: (r.pii || []).map((p) => `${p.label}: ${p.masked}`).join("; "),
    text_score: r.textScore == null ? "" : +r.textScore.toFixed(4),
    image_score: r.imageScore == null ? "" : +r.imageScore.toFixed(4),
    masked_text: r.text || "", error: r.error || "",
  }));
  $("dlJson").addEventListener("click", () => download("privacy-results.json", JSON.stringify(exportRows(), null, 2), "application/json"));
  $("dlCsv").addEventListener("click", () => {
    const rows = exportRows(), cols = Object.keys(rows[0] || { file: "" });
    const q = (v) => `"${String(v).replace(/"/g, '""')}"`;
    download("privacy-results.csv", [cols.join(","), ...rows.map((r) => cols.map((c) => q(r[c])).join(","))].join("\n"), "text/csv");
  });
  $("clearBtn").addEventListener("click", () => { results = []; $("results").replaceChildren(); updateSummary(); });

  // ------------------------------------------------------------------ input
  const drop = $("drop");
  $("fileInput").addEventListener("change", (e) => { run([...e.target.files]); e.target.value = ""; });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => run([...e.dataTransfer.files]));
  document.addEventListener("paste", (e) => {
    const files = [...(e.clipboardData ? e.clipboardData.files : [])];
    if (files.length) run(files.map((f, i) => new File([f], f.name && f.name !== "image.png" ? f.name : `pasted-${i + 1}.png`, { type: f.type })));
  });

  // ------------------------------------------------------------------ sample images (drawn, all data fake)
  const FONT = "Arial, Helvetica, sans-serif";
  function drawCard(c) {
    const x = c.getContext("2d"); c.width = 900; c.height = 600;
    x.fillStyle = "#eef1f5"; x.fillRect(0, 0, 900, 600);
    x.fillStyle = "#1f3b73"; roundRect(x, 70, 60, 760, 470, 28); x.fill();
    x.fillStyle = "#ffffff";
    x.font = `bold 40px ${FONT}`; x.fillText("SUNRISE BANK", 120, 140);
    x.font = `26px ${FONT}`; x.fillText("DEBIT CARD", 120, 185);
    x.fillStyle = "#d4af37"; roundRect(x, 120, 225, 90, 66, 10); x.fill();
    x.fillStyle = "#ffffff";
    x.font = `bold 50px ${FONT}`; x.fillText("4111 1111 1111 1111", 120, 370);
    x.font = `26px ${FONT}`; x.fillText("VALID THRU 09/29", 120, 425);
    x.font = `bold 32px ${FONT}`; x.fillText("RAHUL SHARMA", 120, 485);
  }
  function drawPoster(c) {
    const x = c.getContext("2d"); c.width = 900; c.height = 600;
    x.fillStyle = "#fff7e8"; x.fillRect(0, 0, 900, 600);
    x.fillStyle = "#c2410c"; x.fillRect(0, 0, 900, 22);
    x.fillStyle = "#1c1917";
    x.font = `bold 84px ${FONT}`; x.fillText("TECH FEST 2026", 70, 170);
    x.font = `40px ${FONT}`; x.fillText("Friday 4:30 PM", 70, 260);
    x.fillText("Room 1205, Brickyard", 70, 320);
    x.font = `bold 40px ${FONT}`; x.fillText("Free entry for all students", 70, 420);
    x.font = `30px ${FONT}`; x.fillText("Prizes worth 25000 rupees", 70, 490);
  }
  function drawChat(c) {
    const x = c.getContext("2d"); c.width = 900; c.height = 600;
    x.fillStyle = "#ffffff"; x.fillRect(0, 0, 900, 600);
    const bubble = (y, w, text, mine) => {
      x.fillStyle = mine ? "#dcf8c6" : "#f0f0f0";
      roundRect(x, mine ? 860 - w : 40, y, w, 74, 18); x.fill();
      x.fillStyle = "#111"; x.font = `34px ${FONT}`; x.fillText(text, (mine ? 860 - w : 40) + 26, y + 48);
    };
    bubble(50, 560, "hey can you login for me", false);
    bubble(160, 620, "my gmail password is sunshine", true);
    bubble(270, 470, "use it before 6 PM", true);
    bubble(380, 380, "ok, logging in", false);
  }
  function roundRect(x, a, b, w, h, r) { x.beginPath(); x.roundRect ? x.roundRect(a, b, w, h, r) : x.rect(a, b, w, h); }

  const SAMPLES = [["Debit card", "sample-debit-card.png", drawCard], ["Chat message", "sample-chat.png", drawChat], ["Event poster", "sample-poster.png", drawPoster]];
  for (const [label, file, draw] of SAMPLES) {
    const c = document.createElement("canvas"); draw(c);
    const btn = el("button", { class: "sample", type: "button", "aria-label": "Analyse the " + label.toLowerCase() + " sample" }, c, el("span", {}, label));
    btn.addEventListener("click", () => c.toBlob((b) => run([new File([b], file, { type: "image/png" })], [label + " (sample)"]), "image/png"));
    $("samples").append(btn);
  }

  detectMode();
})();
