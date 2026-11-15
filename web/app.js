/* ==========================================================================
   AstrocyteX — Flight Medical Telemetry Workstation Engine (v3.0)
   English Interface · 43-Channel Multimodal Fusion · AstrocyteX-Voice Studio ·
   Role-Adaptive Medical Advisor ('I am a doctor' / 'I am not a doctor')
   ========================================================================== */

(() => {
  /* -----------------------------------------------------------------------
     1. SUBTLE ATMOSPHERIC STARLIGHT LAYER
     ----------------------------------------------------------------------- */
  const starCanvas = document.getElementById("stars");
  const sCtx = starCanvas ? starCanvas.getContext("2d") : null;
  let stars = [];

  function createStars() {
    if (!starCanvas) return;
    starCanvas.width = window.innerWidth;
    starCanvas.height = window.innerHeight;
    const count = Math.floor((starCanvas.width * starCanvas.height) / 14000);
    stars = Array.from({ length: count }, () => ({
      x: Math.random() * starCanvas.width,
      y: Math.random() * (starCanvas.height * 0.78),
      radius: Math.random() * 0.9 + 0.25,
      alpha: Math.random() * 0.5 + 0.12,
      speed: Math.random() * 0.003 + 0.0008,
      direction: Math.random() < 0.5 ? 1 : -1,
    }));
  }

  window.addEventListener("resize", () => {
    createStars();
    redrawVisibleCanvases();
  });

  function animateStars() {
    if (!sCtx || !starCanvas) return;
    sCtx.clearRect(0, 0, starCanvas.width, starCanvas.height);
    for (const s of stars) {
      s.alpha += s.speed * s.direction;
      if (s.alpha >= 0.65) { s.alpha = 0.65; s.direction = -1; }
      if (s.alpha <= 0.10) { s.alpha = 0.10; s.direction = 1; }
      sCtx.beginPath();
      sCtx.arc(s.x, s.y, s.radius, 0, Math.PI * 2);
      sCtx.fillStyle = `rgba(238, 235, 248, ${s.alpha})`;
      sCtx.fill();
    }
    requestAnimationFrame(animateStars);
  }
  createStars();
  animateStars();

  /* -----------------------------------------------------------------------
     2. APPLICATION STATE & ENGLISH TERMINOLOGY
     ----------------------------------------------------------------------- */
  const currentLang = "en";
  localStorage.setItem("language", "en");

  const savedRole = localStorage.getItem("astrocytex_user_role");
  const state = {
    astronautId: "astro-1",
    missionDay: 90,
    profile: "iss_leo",
    scenario: "nominal",
    userRole: savedRole === "doctor" ? "doctor" : "non_doctor",
    subjectiveLog: "",
    values: {},
    sensorSeries: null,
    ecgResult: null,
    voiceResult: null,
    voiceSex: "female",
    voiceActiveSample: "calm_regulated",
    registry: [],
    paramByKey: {},
    categories: {},
    profiles: {},
    events: {},
    scenarios: [],
    assessment: null,
    gamification: null,
    pendingTasks: [],
    doneTasks: [],
    history: [],
    healthMeta: null,
    tableFilter: "all",
    historyColMode: "overall",
    historyBandFilter: "all",
    historySearch: "",
  };

  function escHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function t(key) {
    return (window.TRANSLATIONS.en && window.TRANSLATIONS.en[key]) || key;
  }
  window.agTranslate = t;

  function apiLang() {
    return "en";
  }

  function paramLabel(p) {
    return p.name || p.key;
  }

  function formatScenarioTitle(raw) {
    return String(raw || "")
      .split("_")
      .map((w) => (w ? w.charAt(0).toUpperCase() + w.slice(1) : ""))
      .join(" ");
  }

  function showToast(msg) {
    const box = document.getElementById("toastContainer");
    if (!box) return;
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = msg;
    box.appendChild(el);
    setTimeout(() => el.remove(), 3200);
  }

  /* -----------------------------------------------------------------------
     3. PIECEWISE-LINEAR LIVE RISK ENGINE
     ----------------------------------------------------------------------- */
  const LEVEL_COLORS = {
    gray: "#7c7a8c",
    green: "#29b377",
    yellow: "#df9f15",
    red: "#e5484d",
    black: "#f8fafc",
  };

  const LEVEL_NAMES = {
    gray: "Baseline",
    green: "Nominal",
    yellow: "Elevated",
    red: "High Risk",
    black: "Critical",
  };

  function riskLevel(pct) {
    if (pct === null || pct === undefined || Number.isNaN(pct)) return "gray";
    if (pct < 15) return "gray";
    if (pct <= 40) return "green";
    if (pct <= 70) return "yellow";
    if (pct <= 90) return "red";
    return "black";
  }

  function oneSideRisk(d, n, w, dg) {
    if (n === null || n === undefined) return 0;
    if (d <= 0) return 8.0;
    if (d <= n) return 8.0 + 32.0 * (d / Math.max(n, 1e-9));
    if (w !== null && w !== undefined && d <= w) return 40.0 + 30.0 * ((d - n) / Math.max(w - n, 1e-9));
    if (dg !== null && dg !== undefined && d <= dg) return 70.0 + 20.0 * ((d - w) / Math.max(dg - w, 1e-9));
    const span = Math.max((dg || w || n) - (w || n), 1e-9);
    return Math.min(100.0, 90.0 + 10.0 * ((d - (dg || w || n)) / span));
  }

  function liveRuleRisk(key, val) {
    const p = state.paramByKey[key];
    if (!p || val === null || val === undefined || Number.isNaN(Number(val))) return 8.0;
    const x = Number(val);
    let rLow = 0, rHigh = 0;
    if (p.normal[0] !== null && x < p.ideal) {
      rLow = oneSideRisk(
        p.ideal - x,
        p.ideal - p.normal[0],
        p.warn[0] !== null ? p.ideal - p.warn[0] : null,
        p.danger[0] !== null ? p.ideal - p.danger[0] : null
      );
    }
    if (p.normal[1] !== null && x > p.ideal) {
      rHigh = oneSideRisk(
        x - p.ideal,
        p.normal[1] - p.ideal,
        p.warn[1] !== null ? p.warn[1] - p.ideal : null,
        p.danger[1] !== null ? p.danger[1] - p.ideal : null
      );
    }
    if (x === p.ideal) return 8.0;
    const r = Math.max(rLow, rHigh);
    return r === 0 ? 8.0 : Math.min(100, Math.max(0, r));
  }

  function normalRangeText(p) {
    const lo = p.normal[0], hi = p.normal[1];
    const u = p.unit ? ` ${p.unit}` : "";
    if (lo !== null && hi !== null) return `${lo} - ${hi}${u}`;
    if (lo !== null) return `>= ${lo}${u}`;
    if (hi !== null) return `<= ${hi}${u}`;
    return "-";
  }

  window.agDotHtml = (key, val) => {
    const lv = riskLevel(liveRuleRisk(key, val));
    return `<span class="risk-dot ${lv}"></span>`;
  };
  window.agGetVal = (key) => state.values[key];
  window.agSetDietTotals = (totals) => {
    if (totals.calories_kcal > 0) setParamValue("calories_kcal", totals.calories_kcal);
    if (totals.protein_g > 0) setParamValue("protein_g", totals.protein_g);
    if (totals.fluid_intake_ml > 0) setParamValue("fluid_intake_ml", totals.fluid_intake_ml);
    showToast(t("dietSynced"));
    runAssessment();
  };

  /* -----------------------------------------------------------------------
     4. SIDEBAR NAVIGATION & ROLE SELECTOR
     ----------------------------------------------------------------------- */
  const sidebar = document.getElementById("sidebar");
  const mobileNavBtn = document.getElementById("mobileNavBtn");
  const topbarSectionTitle = document.getElementById("topbarSectionTitle");
  const menuItems = document.querySelectorAll(".menu-list button[data-page]");

  const PAGE_TITLES = {
    home: "Overview",
    "overall-health": "Health Matrix (43)",
    "mental-health": "Voice & Mood",
    "cardiovascular-statistics": "Cardiovascular & ECG",
    "biochemical-statistics": "Biochemistry & CBC",
    "space-radiation": "Space Environment",
    "exercise-tracker": "Exercise & Physiology",
    "health-history": "Health History",
    diet: "Nutrition & Hydration",
    "diet-history": "Nutrition Log",
    experiments: "Experiments",
    progress: "Trends & Advisor",
    rewards: "Countermeasures",
  };

  function showPage(pageId) {
    const id = pageId || "home";
    document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
    const target = document.getElementById(id) || document.getElementById("home");
    const resolvedId = target.id;
    target.classList.add("active");

    menuItems.forEach((btn) => btn.classList.toggle("current", btn.dataset.page === resolvedId));
    if (topbarSectionTitle) {
      topbarSectionTitle.textContent = PAGE_TITLES[resolvedId] || "Overview";
    }
    if (sidebar) sidebar.classList.remove("open");
    setTimeout(redrawVisibleCanvases, 35);
  }
  window.agShowPage = showPage;

  menuItems.forEach((item) => {
    item.addEventListener("click", () => {
      showPage(item.dataset.page);
      history.replaceState(null, "", "#" + item.dataset.page);
    });
  });

  document.querySelectorAll("[data-nav-target]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const dest = btn.dataset.navTarget;
      showPage(dest);
      history.replaceState(null, "", "#" + dest);
    });
  });

  if (mobileNavBtn && sidebar) {
    mobileNavBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      sidebar.classList.toggle("open");
    });
    document.addEventListener("click", (e) => {
      if (sidebar.classList.contains("open") && !sidebar.contains(e.target) && e.target !== mobileNavBtn) {
        sidebar.classList.remove("open");
      }
    });
  }

  function syncRoleUI() {
    document.querySelectorAll("[data-role-btn]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.roleBtn === state.userRole);
    });
    const desc = state.userRole === "doctor" ? t("roleDescDoctor") : t("roleDescNonDoctor");
    const h1 = document.getElementById("briefingRoleHint");
    const h2 = document.getElementById("chatRoleHint");
    if (h1) h1.textContent = desc;
    if (h2) h2.textContent = desc;
  }

  function setUserRole(role, notify = true) {
    state.userRole = role === "doctor" ? "doctor" : "non_doctor";
    localStorage.setItem("astrocytex_user_role", state.userRole);
    syncRoleUI();
    renderVoiceStudio();
    if (notify) {
      const msg = state.userRole === "doctor" ? t("roleDescDoctor") : t("roleDescNonDoctor");
      showToast(msg);
      appendSystemChatNotice(msg);
      runAssessment(true);
    }
  }

  function applyStaticTranslations() {
    document.documentElement.lang = "en";
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const txt = window.TRANSLATIONS.en[el.dataset.i18n];
      if (txt) el.textContent = txt;
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      const txt = window.TRANSLATIONS.en[el.dataset.i18nPlaceholder];
      if (txt) el.placeholder = txt;
    });
    syncRoleUI();
    document.dispatchEvent(new CustomEvent("app-language-changed"));
  }

  /* -----------------------------------------------------------------------
     5. PARAMETER INPUT CARD RENDERER
     ----------------------------------------------------------------------- */
  const CATEGORY_SCREEN_MAP = {
    cardiovascular: "cardiovascular-statistics",
    respiratory: "cardiovascular-statistics",
    ecg: "cardiovascular-statistics",
    stress_thermal: "mental-health",
    sleep_activity: "exercise-tracker",
    bone_muscle: "exercise-tracker",
    hydration_renal: "diet",
    nutrition: "diet",
    hemogram: "biochemical-statistics",
    biochemistry: "biochemical-statistics",
    radiation: "space-radiation",
    isolation: "space-radiation",
    distance: "space-radiation",
    gravity: "space-radiation",
    habitat: "space-radiation",
  };

  function assessedParam(key) {
    if (!state.assessment || !state.assessment.parameters) return null;
    return state.assessment.parameters.find((p) => p.key === key) || null;
  }

  function setParamValue(key, rawVal) {
    const p = state.paramByKey[key];
    if (!p) return;
    const clamped = Math.min(p.bounds[1], Math.max(p.bounds[0], Number(rawVal)));
    const factor = Math.pow(10, p.decimals || 0);
    const val = Math.round(clamped * factor) / factor;
    state.values[key] = val;

    document.querySelectorAll(`[data-param-card="${key}"]`).forEach((card) => {
      const sl = card.querySelector(".param-range");
      const nm = card.querySelector(".param-num");
      const dot = card.querySelector(".risk-dot");
      const badge = card.querySelector(".param-pct-badge");
      if (sl && Number(sl.value) !== val) sl.value = String(val);
      if (nm && Number(nm.value) !== val) nm.value = String(val);
      const rPct = liveRuleRisk(key, val);
      const lv = riskLevel(rPct);
      if (dot) dot.className = `risk-dot ${lv}`;
      if (badge) badge.textContent = `${rPct.toFixed(0)}%`;
    });
    if (window.AstroDietBridge) window.AstroDietBridge.updateDietAIBar();
  }

  function buildParamGrid(containerId, keys) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.replaceChildren();
    keys.forEach((key) => {
      const p = state.paramByKey[key];
      if (!p) return;
      const val = state.values[key] !== undefined ? state.values[key] : p.ideal;
      const ap = assessedParam(key);
      const pct = ap && ap.risk_pct !== null ? ap.risk_pct : liveRuleRisk(key, val);
      const lv = riskLevel(pct);
      const step = p.bounds[2] || (p.decimals > 0 ? Math.pow(10, -p.decimals) : 1);

      const card = document.createElement("div");
      card.className = "param-card";
      card.dataset.paramCard = key;
      card.innerHTML = `
        <div class="param-head">
          <div class="param-title-wrap">
            <span class="risk-dot ${lv}"></span>
            <span class="param-name" title="${paramLabel(p)}">${paramLabel(p)}</span>
          </div>
          <span class="param-pct-badge">${pct.toFixed(0)}%</span>
        </div>
        <div class="param-input-row">
          <input type="range" class="param-range" min="${p.bounds[0]}" max="${p.bounds[1]}" step="${step}" value="${val}">
          <input type="number" class="param-num" min="${p.bounds[0]}" max="${p.bounds[1]}" step="${step}" value="${val}">
        </div>
        <div class="param-foot">
          <span>Ref: ${normalRangeText(p)}</span>
          <span>${p.unit || ""}</span>
        </div>
      `;
      const sl = card.querySelector(".param-range");
      const nm = card.querySelector(".param-num");
      sl.addEventListener("input", () => setParamValue(key, sl.value));
      nm.addEventListener("change", () => setParamValue(key, nm.value));
      el.appendChild(card);
    });
  }

  /* -----------------------------------------------------------------------
     6. PRECISION LINEAR METER & HIGH-DPI CANVAS CHARTS
     ----------------------------------------------------------------------- */
  function linearMeterHtml(pct, level, subtitleHtml) {
    const safePct = Math.max(0, Math.min(100, Number(pct) || 0));
    const col = level === "black" ? "#e5484d" : (LEVEL_COLORS[level] || "#9b87f5");
    const bandLabel = LEVEL_NAMES[level] || "Baseline";
    return `
      <div class="meter-wrap">
        <div class="meter-readout-row">
          <div style="display:flex;align-items:center;gap:8px;">
            <span class="risk-dot lg ${level}"></span>
            <span class="meter-big">${safePct.toFixed(1)}%</span>
          </div>
          <span class="meter-band">${bandLabel}</span>
        </div>
        <div class="meter-track">
          <div class="meter-fill" style="width:${Math.max(2, safePct).toFixed(1)}%;background:${col};"></div>
        </div>
        <div class="meter-sub">${subtitleHtml}</div>
      </div>
    `;
  }

  function setupCanvas(canvas, cssHeight) {
    if (!canvas) return null;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(300, canvas.clientWidth || 600);
    const h = cssHeight || Number(canvas.getAttribute("height")) || 220;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const c = canvas.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    return { c, w, h };
  }

  function drawLineSeries(canvasId, seriesList, xLabels, yMin, yMax) {
    const canvas = document.getElementById(canvasId);
    const box = setupCanvas(canvas, Number(canvas?.getAttribute("height")) || 210);
    if (!box) return;
    const { c, w, h } = box;
    const pad = { l: 42, r: 16, t: 24, b: 26 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    c.strokeStyle = "rgba(255, 255, 255, 0.07)";
    c.fillStyle = "rgba(158, 155, 174, 0.78)";
    c.font = "11px Consolas, monospace";
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (ph * i) / 4;
      const val = yMax - ((yMax - yMin) * i) / 4;
      c.beginPath(); c.moveTo(pad.l, y); c.lineTo(w - pad.r, y); c.stroke();
      c.fillText(val.toFixed(0), 6, y + 4);
    }

    if (xLabels && xLabels.length > 1) {
      const step = Math.max(1, Math.floor((xLabels.length - 1) / 5));
      for (let i = 0; i < xLabels.length; i += step) {
        const x = pad.l + (pw * i) / (xLabels.length - 1);
        c.fillText(String(xLabels[i]), x - 12, h - 8);
      }
    }

    let lx = pad.l;
    seriesList.forEach(({ label, data, color }) => {
      if (!data || !data.length) return;
      c.strokeStyle = color;
      c.lineWidth = 1.75;
      c.beginPath();
      data.forEach((v, idx) => {
        const x = pad.l + (pw * idx) / Math.max(1, data.length - 1);
        const y = pad.t + ph * (1 - (v - yMin) / Math.max(1e-6, yMax - yMin));
        if (idx === 0) c.moveTo(x, y); else c.lineTo(x, y);
      });
      c.stroke();

      c.fillStyle = color;
      c.fillRect(lx, 8, 10, 2.5);
      c.fillStyle = "#efeff4";
      c.fillText(label, lx + 14, 13);
      lx += c.measureText(label).width + 30;
    });
  }

  function drawECGCanvas() {
    const canvas = document.getElementById("ecgCanvas");
    const box = setupCanvas(canvas, 265);
    if (!box) return;
    const { c, w, h } = box;

    c.strokeStyle = "rgba(255, 255, 255, 0.045)";
    c.lineWidth = 1;
    for (let x = 0; x < w; x += 16) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke(); }
    for (let y = 0; y < h; y += 16) { c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke(); }

    const ecg = state.ecgResult;
    const arr = (ecg && Array.isArray(ecg.lead_ii) && ecg.lead_ii.length) ? ecg.lead_ii : null;
    if (!arr) {
      c.fillStyle = "rgba(158, 155, 174, 0.78)";
      c.font = "12px Inter, Segoe UI, sans-serif";
      c.fillText("Select an ECG rhythm class above (NORMAL, AFIB, MI, PVC) or upload a 12-lead file.", 20, h / 2);
      return;
    }

    const strips = [
      { title: `Lead II — Rhythm Strip (${ecg.record_name || ecg.class_code})`, data: arr },
      { title: `Lead II — QRS-T Segment Detail`, data: arr.slice(0, Math.min(320, arr.length)) },
    ];
    const cellH = h / 2;
    strips.forEach((st, idx) => {
      const x0 = 14;
      const y0 = idx * cellH + 8;
      const cw = w - 28;
      const ch = cellH - 18;
      const mn = Math.min(...st.data), mx = Math.max(...st.data);
      const span = Math.max(0.2, mx - mn);

      c.fillStyle = "#9e9bae";
      c.font = "600 11px Consolas, monospace";
      c.fillText(st.title, x0 + 2, y0 + 12);

      c.strokeStyle = idx === 0 ? "#b4a5fa" : "#29b377";
      c.lineWidth = 1.5;
      c.beginPath();
      st.data.forEach((v, i) => {
        const x = x0 + (cw * i) / Math.max(1, st.data.length - 1);
        const y = y0 + ch - ((v - mn) / span) * (ch - 18) - 4;
        if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
      });
      c.stroke();
    });
  }

  function drawVoiceCanvas() {
    const canvas = document.getElementById("voiceCanvas");
    const box = setupCanvas(canvas, 215);
    if (!box) return;
    const { c, w, h } = box;

    const vr = state.voiceResult;
    if (!vr || !Array.isArray(vr.waveform_preview) || !vr.waveform_preview.length) {
      c.fillStyle = "rgba(158, 155, 174, 0.78)";
      c.font = "12px Inter, Segoe UI, sans-serif";
      c.fillText("Record microphone audio, upload a speech clip, or select a reference sample above.", 18, h / 2);
      return;
    }

    const topH = Math.floor(h * 0.44);
    const botY = topH + 10;
    const botH = h - botY - 8;
    const padX = 14;
    const innerW = w - padX * 2;

    c.fillStyle = "#9e9bae";
    c.font = "600 10.5px Consolas, monospace";
    c.fillText(`Acoustic Waveform (${(vr.duration_s || 0).toFixed(1)}s · 16 kHz)`, padX, 14);

    c.strokeStyle = "rgba(255, 255, 255, 0.08)";
    c.beginPath();
    c.moveTo(padX, 18 + (topH - 20) / 2);
    c.lineTo(w - padX, 18 + (topH - 20) / 2);
    c.stroke();

    const wf = vr.waveform_preview;
    c.strokeStyle = "#b4a5fa";
    c.lineWidth = 1.25;
    c.beginPath();
    wf.forEach((v, i) => {
      const x = padX + (innerW * i) / Math.max(1, wf.length - 1);
      const y = 18 + (topH - 20) * (0.5 - 0.46 * Math.max(-1, Math.min(1, v)));
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    });
    c.stroke();

    c.fillStyle = "#9e9bae";
    c.fillText("48-Band Log-Mel Spectrogram", padX, botY + 8);
    const spec = vr.mel_spectrogram_preview;
    if (Array.isArray(spec) && spec.length) {
      const nMels = spec.length;
      const nFrames = spec[0].length || 1;
      const sy0 = botY + 13;
      const sh = botH - 13;
      const cellW = innerW / nFrames;
      const cellH = sh / nMels;
      for (let m = 0; m < nMels; m++) {
        const row = spec[nMels - 1 - m];
        for (let tIdx = 0; tIdx < nFrames; tIdx++) {
          const z = Number(row[tIdx] || 0);
          const norm = Math.max(0, Math.min(1, (z + 2.2) / 4.4));
          const rCol = Math.round(12 + 184 * Math.pow(norm, 1.35));
          const gCol = Math.round(7 + 108 * Math.pow(norm, 1.8));
          const bCol = Math.round(24 + 205 * Math.pow(norm, 0.85));
          c.fillStyle = `rgb(${rCol},${gCol},${bCol})`;
          c.fillRect(padX + tIdx * cellW, sy0 + m * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
    }
  }

  function fmtCommDelay(der) {
    if (!der || der.comm_delay_s === undefined || der.comm_delay_s === null) return "0 s";
    const s = Number(der.comm_delay_s);
    return s < 60 ? `${s.toFixed(2)} s` : `${(s / 60).toFixed(1)} min`;
  }

  function fmtEvacTime(der) {
    if (!der || der.time_to_care_days === undefined || der.time_to_care_days === null) return "~6 h";
    const d = Number(der.time_to_care_days);
    return d < 1 ? `~${(d * 24).toFixed(0)} h` : `~${d.toFixed(0)} d`;
  }

  function redrawVisibleCanvases() {
    if (state.sensorSeries) {
      drawLineSeries("telemetryCanvas", [
        { label: "HR (bpm)", data: state.sensorSeries.heart_rate, color: "#e5484d" },
        { label: "SpO2 (%)", data: state.sensorSeries.spo2, color: "#b4a5fa" },
        { label: "Sys BP", data: state.sensorSeries.systolic_bp, color: "#df9f15" },
        { label: "Resp x4", data: (state.sensorSeries.resp_rate || []).map((x) => x * 4), color: "#29b377" },
      ], Array.from({ length: 96 }, (_, i) => `${Math.round((i - 95) * 15 / 60)}h`), 30, 185);
    }

    drawECGCanvas();
    drawVoiceCanvas();

    const A = state.assessment;
    if (A && A.physics && A.physics.projections) {
      const cur = A.physics.current || {};
      const p30 = A.physics.projections["+30d"] || {};
      const p90 = A.physics.projections["+90d"] || {};
      const p180 = A.physics.projections["+180d"] || {};
      drawLineSeries("odeCanvas", [
        { label: "Bone BMD %", data: [cur.bone_density_change_pct || 0, p30.bone_density_change_pct || 0, p90.bone_density_change_pct || 0, p180.bone_density_change_pct || 0], color: "#df9f15" },
        { label: "Muscle Mass %", data: [cur.muscle_mass_change_pct || 0, p30.muscle_mass_change_pct || 0, p90.muscle_mass_change_pct || 0, p180.muscle_mass_change_pct || 0], color: "#e5484d" },
        { label: "Plasma Vol %", data: [cur.plasma_volume_change_pct || 0, p30.plasma_volume_change_pct || 0, p90.plasma_volume_change_pct || 0, p180.plasma_volume_change_pct || 0], color: "#b4a5fa" },
      ], ["Now", "+30d", "+90d", "+180d"], -22, 6);
    }

    if (A && A.forecast && A.forecast.length) {
      drawLineSeries("forecastCanvas", [
        { label: "7-Day Forecast %", data: A.forecast, color: "#b4a5fa" },
      ], A.forecast.map((_, i) => `D+${i + 1}`), 0, 100);
    }

    if (state.history && state.history.length) {
      const pts = state.history;
      drawLineSeries("historyCanvas", [
        { label: "Health Risk %", data: pts.map((x) => x.overall_risk_pct ?? x.overall_risk ?? 0), color: "#29b377" },
      ], pts.map((_, i) => `#${i + 1}`), 0, 100);
    }

    drawHealthHistoryColumns();
  }

  function drawHealthHistoryColumns() {
    const canvas = document.getElementById("healthHistoryColCanvas");
    const box = setupCanvas(canvas, Number(canvas?.getAttribute("height")) || 255);
    if (!box) return;
    const { c, w, h } = box;
    const hist = state.history || [];

    if (!hist.length) {
      c.fillStyle = "rgba(158, 155, 174, 0.82)";
      c.font = "12.5px Inter, Segoe UI, sans-serif";
      c.fillText("No health snapshots recorded for this slot yet. Click 'Record Health Snapshot' above.", 22, h / 2);
      return;
    }

    const mode = state.historyColMode || "overall";
    const pad = { l: 44, r: 18, t: 30, b: 34 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const yMax = mode === "vitals" ? 160 : 100;

    // Gridlines & Y-axis ticks
    c.strokeStyle = "rgba(255, 255, 255, 0.065)";
    c.fillStyle = "rgba(158, 155, 174, 0.78)";
    c.font = "11px Consolas, monospace";
    c.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (ph * i) / 4;
      const val = yMax - (yMax * i) / 4;
      c.beginPath();
      c.moveTo(pad.l, y);
      c.lineTo(w - pad.r, y);
      c.stroke();
      c.fillText(val.toFixed(0), 8, y + 4);
    }

    // Risk threshold guide lines for percentage views
    if (mode !== "vitals") {
      const thresholds = [
        [40, "rgba(223, 159, 21, 0.35)", "40% Caution"],
        [70, "rgba(229, 72, 77, 0.38)", "70% High"],
      ];
      c.save();
      c.setLineDash([4, 4]);
      thresholds.forEach(([tv, col, lbl]) => {
        const ty = pad.t + ph * (1 - tv / yMax);
        c.strokeStyle = col;
        c.beginPath();
        c.moveTo(pad.l, ty);
        c.lineTo(w - pad.r, ty);
        c.stroke();
        c.fillStyle = col;
        c.font = "10px Consolas, monospace";
        c.fillText(lbl, w - pad.r - 68, ty - 4);
      });
      c.restore();
    }

    const drawLegend = (items) => {
      let lx = pad.l;
      c.font = "11px Consolas, monospace";
      items.forEach(([lbl, col]) => {
        c.fillStyle = col;
        c.fillRect(lx, 8, 10, 8);
        c.fillStyle = "#efeff4";
        c.fillText(lbl, lx + 14, 15);
        lx += c.measureText(lbl).width + 30;
      });
    };

    if (mode === "overall") {
      drawLegend([
        ["Nominal (<=40%)", "#29b377"],
        ["Elevated (41-70%)", "#df9f15"],
        ["High/Critical (>70%)", "#e5484d"],
      ]);
      const slice = hist.slice(-24);
      const startIdx = hist.length - slice.length;
      const slotW = pw / Math.max(1, slice.length);
      const barW = Math.max(10, Math.min(38, slotW * 0.62));

      slice.forEach((rec, i) => {
        const val = Math.max(0, Math.min(100, Number(rec.overall_risk_pct ?? 0)));
        const lv = rec.level || riskLevel(val);
        const col = lv === "black" ? "#e5484d" : (LEVEL_COLORS[lv] || "#29b377");
        const bh = Math.max(3, (val / yMax) * ph);
        const x = pad.l + i * slotW + (slotW - barW) / 2;
        const y = pad.t + ph - bh;

        c.fillStyle = col;
        c.globalAlpha = 0.86;
        c.fillRect(x, y, barW, bh);
        c.globalAlpha = 1.0;
        if (lv === "black") {
          c.fillStyle = "#f8fafc";
          c.fillRect(x, y, barW, 3);
        }

        if (slice.length <= 18 || i % 2 === 0 || i === slice.length - 1) {
          c.fillStyle = "#efeff4";
          c.font = "10px Consolas, monospace";
          const txt = `${val.toFixed(0)}%`;
          c.fillText(txt, x + Math.max(0, (barW - c.measureText(txt).width) / 2), Math.max(25, y - 5));
        }

        if (slice.length <= 14 || i % Math.ceil(slice.length / 10) === 0 || i === slice.length - 1) {
          const xLbl = `#${startIdx + i + 1}`;
          c.fillStyle = "rgba(158, 155, 174, 0.86)";
          c.font = "10.5px Consolas, monospace";
          c.fillText(xLbl, x + Math.max(-4, (barW - c.measureText(xLbl).width) / 2), h - 12);
        }
      });
    } else if (mode === "grouped") {
      drawLegend([
        ["Physiological %", "#29b377"],
        ["Voice & Mood %", "#9b87f5"],
        ["Habitat Hazard %", "#df9f15"],
      ]);
      const slice = hist.slice(-12);
      const startIdx = hist.length - slice.length;
      const slotW = pw / Math.max(1, slice.length);
      const groupW = Math.min(48, slotW * 0.76);
      const barW = Math.max(4, (groupW - 4) / 3);

      slice.forEach((rec, i) => {
        const vals = [
          [Number(rec.physiological_risk_pct ?? rec.overall_risk_pct ?? 0), "#29b377"],
          [Number(rec.voice_risk_pct ?? 0), "#9b87f5"],
          [Number(rec.env_hazard_pct ?? 0), "#df9f15"],
        ];
        const gx = pad.l + i * slotW + (slotW - groupW) / 2;
        vals.forEach(([v, col], j) => {
          const clamped = Math.max(0, Math.min(100, v));
          const bh = Math.max(2, (clamped / yMax) * ph);
          c.fillStyle = col;
          c.fillRect(gx + j * (barW + 2), pad.t + ph - bh, barW, bh);
        });
        const xLbl = `#${startIdx + i + 1}`;
        c.fillStyle = "rgba(158, 155, 174, 0.86)";
        c.font = "10.5px Consolas, monospace";
        c.fillText(xLbl, gx + Math.max(0, (groupW - c.measureText(xLbl).width) / 2), h - 12);
      });
    } else if (mode === "systems") {
      drawLegend([
        ["History Mean %", "#9b87f5"],
        ["Latest Snapshot %", "#29b377"],
      ]);
      const sysOrder = [
        ["cardiovascular", "Cardio"],
        ["respiratory", "Resp"],
        ["ecg", "ECG"],
        ["stress_thermal", "Stress"],
        ["sleep_activity", "Sleep"],
        ["bone_muscle", "Bone/Mus"],
        ["hydration_renal", "Renal"],
        ["nutrition", "Nutrit"],
        ["hemogram", "CBC"],
        ["biochemistry", "Biochem"],
      ];
      const latestSys = (hist[hist.length - 1] && hist[hist.length - 1].systems) || {};
      const slotW = pw / sysOrder.length;
      const groupW = Math.min(42, slotW * 0.72);
      const barW = Math.max(6, (groupW - 3) / 2);

      sysOrder.forEach(([cat, shortLbl], i) => {
        const samples = hist
          .map((r) => (r.systems && r.systems[cat] !== undefined ? Number(r.systems[cat]) : null))
          .filter((v) => v !== null && !Number.isNaN(v));
        const meanVal = samples.length ? samples.reduce((a, b) => a + b, 0) / samples.length : 0;
        const lastVal = latestSys[cat] !== undefined ? Number(latestSys[cat]) : meanVal;
        const gx = pad.l + i * slotW + (slotW - groupW) / 2;

        [[meanVal, "#9b87f5"], [lastVal, "#29b377"]].forEach(([v, col], j) => {
          const bh = Math.max(2, (Math.max(0, Math.min(100, v)) / yMax) * ph);
          c.fillStyle = col;
          c.fillRect(gx + j * (barW + 3), pad.t + ph - bh, barW, bh);
        });

        c.fillStyle = "rgba(158, 155, 174, 0.88)";
        c.font = "10px Consolas, monospace";
        c.fillText(shortLbl, gx + Math.max(-6, (groupW - c.measureText(shortLbl).width) / 2), h - 12);
      });
    } else if (mode === "vitals") {
      drawLegend([
        ["Heart Rate (bpm)", "#e5484d"],
        ["SpO2 (%)", "#b4a5fa"],
      ]);
      const slice = hist.slice(-14);
      const startIdx = hist.length - slice.length;
      const slotW = pw / Math.max(1, slice.length);
      const groupW = Math.min(42, slotW * 0.74);
      const barW = Math.max(5, (groupW - 3) / 2);

      slice.forEach((rec, i) => {
        const vit = rec.vitals || {};
        const hr = Number(vit.heart_rate ?? 0);
        const spo2 = Number(vit.spo2 ?? 0);
        const gx = pad.l + i * slotW + (slotW - groupW) / 2;
        [[hr, "#e5484d"], [spo2, "#b4a5fa"]].forEach(([v, col], j) => {
          const bh = Math.max(2, (Math.max(0, Math.min(yMax, v)) / yMax) * ph);
          c.fillStyle = col;
          c.fillRect(gx + j * (barW + 3), pad.t + ph - bh, barW, bh);
        });
        const xLbl = `#${startIdx + i + 1}`;
        c.fillStyle = "rgba(158, 155, 174, 0.86)";
        c.font = "10.5px Consolas, monospace";
        c.fillText(xLbl, gx + Math.max(0, (groupW - c.measureText(xLbl).width) / 2), h - 12);
      });
    }
  }

  /* -----------------------------------------------------------------------
     7. VOICE ACOUSTIC RECORDER (16 kHz PCM WAV) & UPLOAD DECODER
     ----------------------------------------------------------------------- */
  let mediaStream = null;
  let audioCtx = null;
  let scriptNode = null;
  let recordedChunks = [];
  let isRecording = false;
  let recTimerId = null;
  let recStartMs = 0;

  function resampleFloat32(samples, srcRate, dstRate) {
    if (srcRate === dstRate) return samples;
    const outLen = Math.max(1, Math.round((samples.length * dstRate) / srcRate));
    const out = new Float32Array(outLen);
    const ratio = (samples.length - 1) / Math.max(1, outLen - 1);
    for (let i = 0; i < outLen; i++) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const a = samples[idx] || 0;
      const b = samples[Math.min(samples.length - 1, idx + 1)] || 0;
      out[i] = a + frac * (b - a);
    }
    return out;
  }

  function encodeWav16k(floatSamples, sampleRate = 16000) {
    const pcm16 = new Int16Array(floatSamples.length);
    for (let i = 0; i < floatSamples.length; i++) {
      const s = Math.max(-1, Math.min(1, floatSamples[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    const byteLen = pcm16.length * 2;
    const buf = new ArrayBuffer(44 + byteLen);
    const view = new DataView(buf);
    const writeStr = (off, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + byteLen, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, byteLen, true);
    let offset = 44;
    for (let i = 0; i < pcm16.length; i++, offset += 2) {
      view.setInt16(offset, pcm16[i], true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  async function toggleVoiceRecording() {
    const recBtn = document.getElementById("voiceRecordBtn");
    const recLbl = document.getElementById("voiceRecordBtnLabel");
    const timerEl = document.getElementById("voiceRecTimer");

    if (!isRecording) {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const Ctx = window.AudioContext || window.webkitAudioContext;
        audioCtx = new Ctx();
        const source = audioCtx.createMediaStreamSource(mediaStream);
        scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
        recordedChunks = [];
        scriptNode.onaudioprocess = (ev) => {
          if (!isRecording) return;
          const ch = ev.inputBuffer.getChannelData(0);
          recordedChunks.push(new Float32Array(ch));
        };
        source.connect(scriptNode);
        scriptNode.connect(audioCtx.destination);
        isRecording = true;
        recStartMs = Date.now();
        recBtn.classList.add("recording");
        recLbl.textContent = t("recStop");
        timerEl.hidden = false;
        timerEl.textContent = "00:00";
        recTimerId = setInterval(() => {
          const sec = Math.floor((Date.now() - recStartMs) / 1000);
          const mm = String(Math.floor(sec / 60)).padStart(2, "0");
          const ss = String(sec % 60).padStart(2, "0");
          timerEl.textContent = `${mm}:${ss}`;
          if (sec >= 18) toggleVoiceRecording();
        }, 250);
      } catch (err) {
        showToast("Microphone unavailable: " + err.message);
      }
    } else {
      isRecording = false;
      clearInterval(recTimerId);
      recBtn.classList.remove("recording");
      recLbl.textContent = t("recStart");
      timerEl.hidden = true;

      const srcRate = audioCtx ? audioCtx.sampleRate : 44100;
      if (scriptNode) { try { scriptNode.disconnect(); } catch {} scriptNode = null; }
      if (audioCtx) { try { await audioCtx.close(); } catch {} audioCtx = null; }
      if (mediaStream) { mediaStream.getTracks().forEach((tr) => tr.stop()); mediaStream = null; }

      const totalLen = recordedChunks.reduce((s, a) => s + a.length, 0);
      if (totalLen < srcRate * 0.4) {
        showToast("Recording too short (speak for at least 1-2 seconds).");
        return;
      }
      const merged = new Float32Array(totalLen);
      let off = 0;
      for (const ch of recordedChunks) { merged.set(ch, off); off += ch.length; }
      const wav16 = encodeWav16k(resampleFloat32(merged, srcRate, 16000), 16000);

      const player = document.getElementById("voiceAudioPlayer");
      if (player) player.src = URL.createObjectURL(wav16);

      await uploadVoiceBlob(wav16, "live_voice_recording.wav");
    }
  }

  async function handleVoiceFileUpload(file) {
    if (!file) return;
    const lower = file.name.toLowerCase();
    if (lower.endsWith(".npy") || lower.endsWith(".csv")) {
      await uploadVoiceBlob(file, file.name);
      return;
    }
    try {
      const arrBuf = await file.arrayBuffer();
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const tmpCtx = new Ctx();
      const decoded = await tmpCtx.decodeAudioData(arrBuf.slice(0));
      const ch0 = decoded.getChannelData(0);
      const mono = new Float32Array(ch0.length);
      if (decoded.numberOfChannels > 1) {
        const ch1 = decoded.getChannelData(1);
        for (let i = 0; i < ch0.length; i++) mono[i] = 0.5 * (ch0[i] + ch1[i]);
      } else {
        mono.set(ch0);
      }
      const resampled = resampleFloat32(mono, decoded.sampleRate, 16000);
      await tmpCtx.close();
      const wavBlob = encodeWav16k(resampled, 16000);
      const player = document.getElementById("voiceAudioPlayer");
      if (player) player.src = URL.createObjectURL(wavBlob);
      await uploadVoiceBlob(wavBlob, file.name.replace(/\.[^.]+$/, "") + ".wav");
    } catch {
      const player = document.getElementById("voiceAudioPlayer");
      if (player) player.src = URL.createObjectURL(file);
      await uploadVoiceBlob(file, file.name);
    }
  }

  async function uploadVoiceBlob(blob, filename) {
    const fd = new FormData();
    fd.append("file", blob, filename);
    const res = await fetch(`/voice/analyze?lang=en&role=${state.userRole}`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      showToast("Voice analysis error: " + (err.detail || res.status));
      return;
    }
    state.voiceResult = await res.json();
    renderVoiceStudio();
    renderHUD();
    renderHome();
    drawVoiceCanvas();
    await runAssessment(true);
    showToast(`${t("voiceSynced")} (${state.voiceResult.voice_risk_pct.toFixed(1)}%)`);
  }

  async function loadVoiceSample(stateName, autoSync = false) {
    state.voiceActiveSample = stateName;
    document.querySelectorAll("#voiceSampleBtns .chip-btn").forEach((b) =>
      b.classList.toggle("active", b.dataset.vsample === stateName)
    );
    const res = await fetch(
      `/voice/sample/${encodeURIComponent(stateName)}?sex=${state.voiceSex}&lang=en&role=${state.userRole}`
    );
    if (!res.ok) return;
    const data = await res.json();
    state.voiceResult = data;
    if (data.audio_wav_base64) {
      const player = document.getElementById("voiceAudioPlayer");
      if (player) player.src = `data:audio/wav;base64,${data.audio_wav_base64}`;
    }
    renderVoiceStudio();
    renderHUD();
    renderHome();
    drawVoiceCanvas();
    if (autoSync) {
      await runAssessment(true);
      showToast(`${t("voiceSynced")} (${data.voice_risk_pct.toFixed(1)}%)`);
    }
  }

  function renderVoiceStudio() {
    const vr = state.voiceResult;
    const gaugeCard = document.getElementById("voiceRiskGaugeCard");
    const moodCard = document.getElementById("voiceMoodAndStateCard");
    const bmGrid = document.getElementById("voiceBiomarkersGrid");
    const metaLbl = document.getElementById("voiceClipMetaLabel");
    if (!gaugeCard || !moodCard || !bmGrid) return;

    if (!vr) {
      gaugeCard.innerHTML = `<p class="dim-note">Initializing voice acoustic telemetry...</p>`;
      moodCard.innerHTML = "";
      bmGrid.innerHTML = "";
      return;
    }

    if (metaLbl) {
      metaLbl.textContent = `${vr.record_name || "voice.wav"} · ${(vr.duration_s || 0).toFixed(1)}s · 16.0 kHz`;
    }

    const stTitle = vr.state_en || vr.state_key;
    const roleNote = state.userRole === "doctor" ? vr.clinical_note_en : vr.simple_note_en;

    gaugeCard.innerHTML = `
      <h3 class="card-title">
        <span>${t("voiceRiskLabel")}</span>
        <span class="status-chip">${vr.model_version || vr.source}</span>
      </h3>
      ${linearMeterHtml(
        vr.voice_risk_pct,
        vr.level,
        `<div><strong>${stTitle}</strong> (${(vr.confidence_pct || 0).toFixed(1)}% confidence)</div>
         <div class="mono-meta" style="margin-top:3px;">
           90% CI: ${vr.ci_low}% - ${vr.ci_high}% · Epistemic +/-${vr.epistemic_pct}%
         </div>`
      )}
      <div style="margin-top:12px;padding:9px 11px;background:rgba(255,255,255,0.025);border:1px solid var(--border);border-radius:3px;font-size:12px;line-height:1.5;color:var(--text-dim);">
        <div class="mono-meta" style="margin-bottom:3px;text-transform:uppercase;">
          ${state.userRole === "doctor" ? "Clinical Interpretation" : "Plain Summary"}
        </div>
        ${roleNote || ""}
      </div>
      <div style="margin-top:10px;display:flex;justify-content:flex-end;">
        <button type="button" class="btn primary small" id="applyVoiceToAIBtn">${t("applyVoiceBtn")}</button>
      </div>
    `;
    const applyBtn = document.getElementById("applyVoiceToAIBtn");
    if (applyBtn) {
      applyBtn.addEventListener("click", () => runAssessment(false));
    }

    const mood = vr.mood || {};
    const moodItems = [
      ["Affective Valence", mood.valence_pct || 0, "#29b377"],
      ["Autonomic Arousal", mood.arousal_pct || 0, "#9b87f5"],
      ["Vocal Stress Load", mood.stress_pct || 0, "#e5484d"],
      ["Cognitive Fatigue", mood.fatigue_pct || 0, "#df9f15"],
    ];

    const probs = (vr.probabilities || []).slice(0, 5);
    moodCard.innerHTML = `
      <h3 class="card-title"><span>${t("voiceMoodTitle")}</span></h3>
      <div>
        ${moodItems.map(([lbl, val, col]) => `
          <div class="voice-mood-row">
            <div class="voice-mood-head"><span>${lbl}</span><b>${Number(val).toFixed(1)}%</b></div>
            <div class="mini-bar"><span style="width:${Math.max(2, Math.min(100, val))}%;background:${col}"></span></div>
          </div>
        `).join("")}
      </div>
      <hr style="border:none;border-top:1px solid var(--border);margin:10px 0;">
      <div class="sub-label">${t("voiceStateProbs")}</div>
      <div>
        ${probs.map((p) => `
          <div class="voice-mood-row" style="margin-bottom:5px;">
            <div class="voice-mood-head"><span>${p.name_en}</span><b>${Number(p.pct).toFixed(1)}%</b></div>
            <div class="mini-bar"><span style="width:${Math.max(2, Math.min(100, p.pct))}%;background:#9b87f5"></span></div>
          </div>
        `).join("")}
      </div>
    `;

    const bm = vr.biomarkers || {};
    const bmCells = [
      ["F0 Pitch Mean", `${(bm.f0_mean_hz || 0).toFixed(1)} Hz`],
      ["F0 Variability (CV)", `${(bm.f0_cv || 0).toFixed(3)}`],
      ["Local Jitter", `${(bm.jitter_pct || 0).toFixed(2)} %`],
      ["Local Shimmer", `${(bm.shimmer_pct || 0).toFixed(2)} %`],
      ["Harmonics-to-Noise", `${(bm.hnr_db || 0).toFixed(1)} dB`],
      ["Speech Rate", `${(bm.speech_rate_sps || 0).toFixed(2)} syl/s`],
      ["Pause Ratio", `${((bm.pause_ratio || 0) * 100).toFixed(1)} %`],
      ["Alpha Spectral Tilt", `${(bm.alpha_ratio_db || 0).toFixed(1)} dB`],
    ];
    bmGrid.innerHTML = bmCells.map(([lbl, val]) => `
      <div class="voice-biomarker-cell">
        <div class="voice-bm-label">${lbl}</div>
        <div class="voice-bm-val">${val}</div>
      </div>
    `).join("");
  }

  /* -----------------------------------------------------------------------
     8. RENDER ALL SCREENS
     ----------------------------------------------------------------------- */
  function renderHUD() {
    const A = state.assessment;
    if (A && A.overall) {
      document.getElementById("hudHealthDot").className = `risk-dot ${A.overall.level}`;
      document.getElementById("hudHealthText").textContent = `${A.overall.risk_pct.toFixed(0)}%`;
    }
    const vr = state.voiceResult || (A && A.voice);
    if (vr && vr.voice_risk_pct !== undefined) {
      document.getElementById("hudVoiceDot").className = `risk-dot ${vr.level || riskLevel(vr.voice_risk_pct)}`;
      document.getElementById("hudVoiceText").textContent = `${Number(vr.voice_risk_pct).toFixed(0)}%`;
    }
    if (A && A.environment && A.environment.hazard_index_pct !== undefined) {
      document.getElementById("hudEnvDot").className = `risk-dot ${A.environment.level}`;
      document.getElementById("hudEnvText").textContent = `${A.environment.hazard_index_pct.toFixed(0)}%`;
    }
    if (state.gamification) {
      const g = state.gamification;
      document.getElementById("hudXpText").textContent = `Lv ${g.level} · ${g.xp} XP`;
    }
    const badge = document.getElementById("menuTasksBadge");
    if (badge) {
      if (state.pendingTasks.length > 0) {
        badge.textContent = String(state.pendingTasks.length);
        badge.hidden = false;
      } else {
        badge.hidden = true;
      }
    }
    const hBadge = document.getElementById("menuHistoryBadge");
    if (hBadge) {
      if (state.history && state.history.length > 0) {
        hBadge.textContent = String(state.history.length);
        hBadge.hidden = false;
      } else {
        hBadge.hidden = true;
      }
    }
  }

  function renderHome() {
    const A = state.assessment;
    const vr = state.voiceResult || (A && A.voice);
    const m = state.healthMeta;

    // 1. Top 4-Cell KPI Strip
    const kpiEl = document.getElementById("homeKpiStrip");
    if (kpiEl) {
      const oh = (A && A.overall) || { risk_pct: 0, level: "gray", ci_low: 0, ci_high: 0, n_at_risk: 0 };
      const env = (A && A.environment) || { hazard_index_pct: 0, level: "gray", n_hazards: 0, derived: {} };
      const vPct = vr ? Number(vr.voice_risk_pct || 0) : 0;
      const vLvl = vr ? (vr.level || riskLevel(vPct)) : "gray";
      const vState = vr ? (vr.state_en || vr.state_key || "Nominal") : "Not recorded";
      const g = state.gamification || { level: 1, xp: 0 };

      kpiEl.innerHTML = `
        <div class="kpi-cell" data-goto="overall-health">
          <div class="kpi-label"><span>Overall Health Risk</span><span class="risk-dot ${oh.level}"></span></div>
          <div class="kpi-val">${oh.risk_pct.toFixed(1)}%</div>
          <div class="kpi-meta">90% CI ${oh.ci_low}%-${oh.ci_high}% · ${oh.n_at_risk}/31 off-nominal</div>
        </div>
        <div class="kpi-cell" data-goto="mental-health">
          <div class="kpi-label"><span>Voice &amp; Mood Risk</span><span class="risk-dot ${vLvl}"></span></div>
          <div class="kpi-val">${vPct.toFixed(1)}%</div>
          <div class="kpi-meta">${vState}</div>
        </div>
        <div class="kpi-cell" data-goto="space-radiation">
          <div class="kpi-label"><span>Space Environment</span><span class="risk-dot ${env.level || "gray"}"></span></div>
          <div class="kpi-val">${(env.hazard_index_pct || 0).toFixed(1)}%</div>
          <div class="kpi-meta">Comm delay ${fmtCommDelay(env.derived)} · ${env.n_hazards || 0}/12 active</div>
        </div>
        <div class="kpi-cell" data-goto="rewards">
          <div class="kpi-label"><span>Pending Protocols</span><span class="mono-meta">Lv ${g.level}</span></div>
          <div class="kpi-val">${state.pendingTasks.length}</div>
          <div class="kpi-meta">${g.xp} XP · ${state.doneTasks.length} completed</div>
        </div>
      `;
      kpiEl.querySelectorAll("[data-goto]").forEach((el) =>
        el.addEventListener("click", () => showPage(el.dataset.goto))
      );
    }

    // 2. Overview Assessment Summary
    const homeBriefing = document.getElementById("homeBriefingBox");
    if (homeBriefing) {
      if (!A || !A.advisor) {
        homeBriefing.innerHTML = `<p class="dim-note">Running initial telemetry assessment...</p>`;
      } else {
        const adv = A.advisor;
        const recs = (adv.recommendations || []).map((r) => `<li>${r}</li>`).join("");
        const modeText = (adv.user_role || state.userRole) === "doctor" ? "Clinical Mode" : "Standard Mode";
        homeBriefing.innerHTML = `
          <div class="briefing-banner ${adv.escalate ? "escalate" : ""}">
            <div class="row-between">
              <span class="mono-meta">${adv.source || "edge"} · ${modeText}</span>
              ${adv.escalate ? `<span class="menu-badge" style="background:rgba(229,72,77,0.22);border-color:#e5484d;color:#fecdd3;">Medical Escalation</span>` : ""}
            </div>
            <p class="briefing-summary">${adv.summary || ""}</p>
            ${recs ? `<ul class="briefing-recs">${recs}</ul>` : ""}
          </div>
        `;
      }
    }

    // 3. Overview Organ Systems Grid
    const homeSys = document.getElementById("homeSystemsGrid");
    if (homeSys && A && A.systems) {
      homeSys.innerHTML = Object.entries(A.systems).map(([cat, s]) => `
        <div class="sys-card" data-goto="${CATEGORY_SCREEN_MAP[cat] || "biochemical-statistics"}">
          <div class="sys-top">
            <span>${s.label}</span>
            <span class="risk-dot ${s.level}"></span>
          </div>
          <div class="sys-val">${s.risk_pct.toFixed(0)}%</div>
          <div class="mini-bar"><span style="width:${Math.max(3, s.risk_pct)}%;background:${LEVEL_COLORS[s.level]}"></span></div>
        </div>
      `).join("");
      homeSys.querySelectorAll("[data-goto]").forEach((el) =>
        el.addEventListener("click", () => showPage(el.dataset.goto))
      );
    }

    // 4. Overview Voice Summary Card
    const homeVoice = document.getElementById("homeVoiceBox");
    if (homeVoice) {
      if (!vr) {
        homeVoice.innerHTML = `<p class="dim-note">No voice clip analyzed yet.</p>`;
      } else {
        const mood = vr.mood || {};
        const rows = [
          ["Valence", mood.valence_pct || 0, "#29b377"],
          ["Arousal", mood.arousal_pct || 0, "#9b87f5"],
          ["Stress", mood.stress_pct || 0, "#e5484d"],
          ["Fatigue", mood.fatigue_pct || 0, "#df9f15"],
        ];
        homeVoice.innerHTML = `
          <div class="row-between" style="margin-bottom:8px;">
            <div>
              <span class="risk-dot ${vr.level || riskLevel(vr.voice_risk_pct)}"></span>
              <strong style="margin-left:5px;">${vr.state_en || vr.state_key}</strong>
            </div>
            <span class="mono-meta">Risk ${Number(vr.voice_risk_pct || 0).toFixed(1)}%</span>
          </div>
          ${rows.map(([lbl, v, col]) => `
            <div class="voice-mood-row" style="margin-bottom:6px;">
              <div class="voice-mood-head"><span>${lbl}</span><b>${Number(v).toFixed(0)}%</b></div>
              <div class="mini-bar"><span style="width:${Math.max(2, Math.min(100, v))}%;background:${col}"></span></div>
            </div>
          `).join("")}
        `;
      }
    }

    // 5. Subsystem Specification Table
    const strip = document.getElementById("homeStatusStrip");
    if (strip) {
      const modelName = m?.model?.version || "AstrocyteX-Net-3.0";
      const voiceModel = m?.voice?.model_version || "AstrocyteX-Voice-1.0";
      const ecgMode = m?.ecg?.mode || "builtin-screening";
      const gemMode = m?.gemini?.configured ? "Gemini Structured Advisor" : "Offline Clinical Reasoner";
      strip.innerHTML = `
        <div class="spec-row"><span class="spec-key">Multimodal Backbone</span><span class="spec-val">${modelName}</span></div>
        <div class="spec-row"><span class="spec-key">Vocal Acoustic Model</span><span class="spec-val">${voiceModel}</span></div>
        <div class="spec-row"><span class="spec-key">Telemetry Channels</span><span class="spec-val">31 Physiological + 12 Environment</span></div>
        <div class="spec-row"><span class="spec-key">12-Lead ECG Engine</span><span class="spec-val">${ecgMode}</span></div>
        <div class="spec-row"><span class="spec-key">Decision Support</span><span class="spec-val">${gemMode}</span></div>
      `;
    }
  }

  function renderOverallHealth() {
    const A = state.assessment;
    if (!A) return;

    const oh = A.overall;
    const voiceLine = oh.voice_risk_pct !== undefined
      ? `<div class="mono-meta" style="margin-top:4px;">
           Physiological: ${oh.physiological_risk_pct}% · Vocal Risk: ${oh.voice_risk_pct}% (${oh.voice_state || "-"})
         </div>`
      : "";
    document.getElementById("overallHealthGaugeCard").innerHTML = `
      <h3 class="card-title"><span>${t("overallHealthTitle")}</span></h3>
      ${linearMeterHtml(oh.risk_pct, oh.level, `90% CI: <strong>${oh.ci_low}% - ${oh.ci_high}%</strong> · Off-nominal biomarkers: <strong>${oh.n_at_risk}</strong> / 31${voiceLine}`)}
    `;

    const env = A.environment || {};
    const upPts = env.uplift ? env.uplift.uplift_pts : 0;
    const upHtml = upPts > 0.3
      ? `<div class="uplift-pill">Environmental uplift: +${upPts.toFixed(1)} pts physiological risk</div>`
      : "";
    document.getElementById("envHazardGaugeCard").innerHTML = `
      <h3 class="card-title"><span>${t("envHazardTitle")}</span></h3>
      ${linearMeterHtml(env.hazard_index_pct || 0, env.level || "gray",
        `Active hazards: <strong>${env.n_hazards || 0}</strong> / 12 · Comm delay: <strong>${fmtCommDelay(env.derived)}</strong>${upHtml}`)}
    `;

    const adv = A.advisor || {};
    const recs = (adv.recommendations || []).map((r) => `<li>${r}</li>`).join("");
    const roleTag = (adv.user_role || state.userRole) === "doctor" ? "Clinical Mode" : "Standard Mode";
    document.getElementById("aiBriefingBox").innerHTML = `
      <div class="briefing-banner ${adv.escalate ? "escalate" : ""}">
        <div class="row-between">
          <strong>${t("aiBriefing")} (${adv.source || "edge"} · ${roleTag})</strong>
          ${adv.escalate ? `<span class="menu-badge" style="background:rgba(229,72,77,0.22);border-color:#e5484d;color:#fecdd3;">Medical Escalation</span>` : ""}
        </div>
        <p class="briefing-summary">${adv.summary || ""}</p>
        ${recs ? `<ul class="briefing-recs">${recs}</ul>` : ""}
      </div>
    `;

    const sysGrid = document.getElementById("organSystemsGrid");
    sysGrid.innerHTML = Object.entries(A.systems || {}).map(([cat, s]) => `
      <div class="sys-card" data-goto="${CATEGORY_SCREEN_MAP[cat] || "biochemical-statistics"}">
        <div class="sys-top">
          <span>${s.label}</span>
          <span class="risk-dot ${s.level}"></span>
        </div>
        <div class="sys-val">${s.risk_pct.toFixed(0)}%</div>
        <div class="mini-bar"><span style="width:${Math.max(3, s.risk_pct)}%;background:${LEVEL_COLORS[s.level]}"></span></div>
      </div>
    `).join("");
    sysGrid.querySelectorAll("[data-goto]").forEach((el) =>
      el.addEventListener("click", () => showPage(el.dataset.goto))
    );

    const ridgeGrid = document.getElementById("ridgeThemesGrid");
    ridgeGrid.innerHTML = Object.entries((env.themes) || {}).map(([_cat, s]) => `
      <div class="sys-card" data-goto="space-radiation">
        <div class="sys-top">
          <span>${s.label}</span>
          <span class="risk-dot ${s.level}"></span>
        </div>
        <div class="sys-val">${s.risk_pct.toFixed(0)}%</div>
        <div class="mini-bar"><span style="width:${Math.max(3, s.risk_pct)}%;background:${LEVEL_COLORS[s.level]}"></span></div>
      </div>
    `).join("");
    ridgeGrid.querySelectorAll("[data-goto]").forEach((el) =>
      el.addEventListener("click", () => showPage(el.dataset.goto))
    );

    const filterLabels = { all: "All (43)", risky: "Off-Nominal", health: "Physiological (31)", environment: "Environment (12)" };
    const fRow = document.getElementById("tableFilterRow");
    fRow.innerHTML = ["all", "risky", "health", "environment"].map((f) => `
      <button type="button" class="chip-btn ${state.tableFilter === f ? "active" : ""}" data-tf="${f}">${filterLabels[f]}</button>
    `).join("");
    fRow.querySelectorAll("[data-tf]").forEach((b) => b.addEventListener("click", () => {
      state.tableFilter = b.dataset.tf;
      renderOverallHealth();
    }));

    const rows = (A.parameters || []).filter((p) => {
      if (state.tableFilter === "risky") return ["yellow", "red", "black"].includes(p.level);
      if (state.tableFilter === "health") return p.kind !== "environment";
      if (state.tableFilter === "environment") return p.kind === "environment";
      return true;
    });

    document.getElementById("allParamsTableWrap").innerHTML = `
      <table class="ag-data-table">
        <thead>
          <tr>
            <th>${t("paramCol")}</th>
            <th>${t("valueCol")}</th>
            <th>${t("normalCol")}</th>
            <th>${t("ruleCol")}</th>
            <th>${t("modelCol")}</th>
            <th>${t("finalCol")}</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((p) => `
            <tr>
              <td><span class="risk-dot ${p.level}"></span> &nbsp;<strong>${p.name}</strong></td>
              <td style="font-family:var(--mono);">${p.value !== null ? p.value : "-"} ${p.unit || ""}</td>
              <td style="font-family:var(--mono);color:var(--text-muted);">${p.normal_range}</td>
              <td style="font-family:var(--mono);">${p.rule_risk_pct !== null ? p.rule_risk_pct.toFixed(1) + "%" : "-"}</td>
              <td style="font-family:var(--mono);">${p.model_risk_pct !== null ? p.model_risk_pct.toFixed(1) + "%" : "-"}</td>
              <td style="font-family:var(--mono);"><strong>${p.risk_pct !== null ? p.risk_pct.toFixed(1) + "%" : "-"}</strong>
                  ${p.ci_low !== null ? `<span class="mono-meta"> [${p.ci_low}-${p.ci_high}%]</span>` : ""}
              </td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  }

  function renderCategoryInputScreens() {
    buildParamGrid("mentalParamsGrid", ["sleep_hours", "gsr", "saliva_cortisol", "isolation_days", "habitable_volume_m3", "cabin_noise_dba"]);
    buildParamGrid("dietParamsGrid", ["fluid_intake_ml", "calories_kcal", "protein_g", "urine_sg", "urine_ph"]);
    buildParamGrid("envParamsGrid", [
      "radiation_dose_msv_d", "radiation_cumulative_msv",
      "isolation_days", "habitable_volume_m3",
      "earth_distance_mkm",
      "gravity_g", "g_transition_days",
      "cabin_co2_mmhg", "cabin_o2_kpa", "cabin_temp_c", "cabin_humidity_pct", "cabin_noise_dba",
    ]);
    buildParamGrid("exerciseParamsGrid", ["exercise_min", "gravity_g", "g_transition_days", "urine_calcium", "ctx_ngml", "vitamin_d_ngml"]);
    buildParamGrid("biochemParamsGrid", [
      "glucose_mgdl", "creatinine_mgdl", "sodium_mmol", "potassium_mmol", "calcium_mgdl", "crp_mgl",
      "ctx_ngml", "vitamin_d_ngml", "hemoglobin_gdl", "hematocrit_pct", "rbc_mil", "wbc_k", "platelets_k",
      "urine_sg", "urine_ph", "urine_calcium", "saliva_cortisol",
    ]);
    buildParamGrid("cardioParamsGrid", [
      "heart_rate", "spo2", "systolic_bp", "diastolic_bp", "resp_rate", "body_temp", "gsr", "ecg_qtc_ms", "ecg_abnormal_pct",
    ]);

    const symBox = document.getElementById("detectedSymptomsBox");
    const syms = (state.assessment && state.assessment.subjective_symptoms) || [];
    if (symBox) {
      symBox.innerHTML = syms.length
        ? `${t("detectedSymptoms")}: ` + syms.map((s) => `<span class="status-chip">${s}</span>`).join(" ")
        : "";
    }

    const env = (state.assessment && state.assessment.environment) || {};
    const der = env.derived || {};
    const cumDose = state.values.radiation_cumulative_msv || 0;
    const radCards = document.getElementById("envDerivedCards");
    if (radCards) {
      radCards.innerHTML = `
        <div class="glass-card">
          <div class="sub-label">${t("commDelay")}</div>
          <div class="kpi-val">${fmtCommDelay(der)}</div>
          <div class="dim-note" style="margin-top:4px;">Earth distance: ${state.values.earth_distance_mkm || 0} x 10^6 km</div>
        </div>
        <div class="glass-card">
          <div class="sub-label">${t("evacTime")}</div>
          <div class="kpi-val">${fmtEvacTime(der)}</div>
          <div class="dim-note" style="margin-top:4px;">Estimated return to definitive care</div>
        </div>
        <div class="glass-card">
          <div class="sub-label">${t("careerDose")}</div>
          <div class="kpi-val">${cumDose.toFixed(0)} / 600 mSv</div>
          <div class="mini-bar" style="margin-top:8px;"><span style="width:${Math.min(100, (cumDose / 600) * 100)}%;background:${cumDose > 350 ? "#e5484d" : "#29b377"}"></span></div>
        </div>
      `;
    }

    const hzBox = document.getElementById("hazardAttentionBars");
    if (hzBox && env.hazard_attention) {
      const entries = Object.entries(env.hazard_attention).sort((a, b) => b[1] - a[1]);
      hzBox.innerHTML = entries.map(([k, w]) => {
        const p = state.paramByKey[k];
        return `
          <div style="margin-bottom:7px;">
            <div style="display:flex;justify-content:space-between;font-size:12px;">
              <span>${p ? paramLabel(p) : k}</span>
              <b style="font-family:var(--mono);">${(w * 100).toFixed(1)}%</b>
            </div>
            <div class="mini-bar"><span style="width:${Math.min(100, w * 320)}%;background:#9b87f5"></span></div>
          </div>
        `;
      }).join("");
    }

    const odeRow = document.getElementById("odeMetricsRow");
    if (odeRow && state.assessment && state.assessment.physics && state.assessment.physics.projections) {
      const p180 = state.assessment.physics.projections["+180d"] || {};
      odeRow.innerHTML = `
        <div class="sys-card"><div class="sys-top"><span>+180d Bone Mineral Density</span></div><div class="sys-val">${(p180.bone_density_change_pct || 0).toFixed(2)}%</div></div>
        <div class="sys-card"><div class="sys-top"><span>+180d Lean Muscle Mass</span></div><div class="sys-val">${(p180.muscle_mass_change_pct || 0).toFixed(2)}%</div></div>
        <div class="sys-card"><div class="sys-top"><span>Cephalad Fluid Shift</span></div><div class="sys-val">${(p180.fluid_shift_L || 0).toFixed(2)} L</div></div>
        <div class="sys-card"><div class="sys-top"><span>Plasma Volume Delta</span></div><div class="sys-val">${(p180.plasma_volume_change_pct || 0).toFixed(1)}%</div></div>
      `;
    }
  }

  function renderProgressAndRewards() {
    const A = state.assessment;
    const modBox = document.getElementById("modalityBarsBox");
    if (modBox && A) {
      const mods = Object.entries(A.modality_attention || {});
      const drivers = A.trend_drivers || [];
      modBox.innerHTML = `
        <div style="margin-bottom:12px;">
          ${mods.map(([m, w]) => `
            <div style="margin-bottom:7px;">
              <div style="display:flex;justify-content:space-between;font-size:12px;">
                <span>${m}</span><span style="font-family:var(--mono);">${(w * 100).toFixed(1)}%</span>
              </div>
              <div class="mini-bar"><span style="width:${Math.min(100, w * 220)}%;background:#9b87f5"></span></div>
            </div>
          `).join("")}
        </div>
        <hr style="border:none;border-top:1px solid var(--border);margin:10px 0;">
        <div class="sub-label">Top Longitudinal Trend Drivers</div>
        <div>
          ${drivers.map((d) => `
            <div style="display:flex;justify-content:space-between;font-size:12px;padding:4px 0;">
              <span>${d.name}</span><b style="font-family:var(--mono);">${(d.weight * 100).toFixed(1)}%</b>
            </div>
          `).join("")}
        </div>
      `;
    }

    const g = state.gamification;
    const gHero = document.getElementById("gamificationHeroCard");
    if (g && gHero) {
      const startXp = g.level_xp ?? g.current_level_xp ?? 0;
      const span = Math.max(1, (g.next_level_xp || 60) - startXp);
      const pct = g.progress !== undefined
        ? Math.min(100, Math.max(0, g.progress * 100))
        : Math.min(100, Math.max(0, ((g.xp - startXp) / span) * 100));
      gHero.innerHTML = `
        <div class="row-between" style="margin-bottom:8px;">
          <div>
            <div style="font-size:15px;font-weight:600;">Level ${g.level} · ${g.rank}</div>
            <div class="mono-meta">${g.xp} XP (Next tier at ${g.next_level_xp} XP)</div>
          </div>
          <div class="row-wrap">
            <span class="status-chip">Streak: ${g.streak_days}d</span>
            <span class="status-chip">Completed: ${g.tasks_completed ?? g.total_completed ?? 0}</span>
          </div>
        </div>
        <div class="xp-bar-outer"><div class="xp-bar-inner" style="width:${pct.toFixed(1)}%"></div></div>
      `;

      document.getElementById("badgesGrid").innerHTML = (g.badges || []).map((b) => `
        <div class="badge-card ${b.earned ? "earned" : "locked"}">
          <div class="badge-ico">${b.icon}</div>
          <div class="badge-name">${b.name}</div>
          <div class="badge-desc">${b.description}</div>
        </div>
      `).join("");
    }

    const activeEl = document.getElementById("activeTasksList");
    if (activeEl) {
      if (!state.pendingTasks.length) {
        activeEl.innerHTML = `<p class="dim-note">${t("noTasksYet")}</p>`;
      } else {
        activeEl.innerHTML = state.pendingTasks.map((tk) => `
          <div class="task-item">
            <div class="task-info">
              <div class="task-title"><span class="risk-dot ${tk.severity_level}"></span> ${tk.title}</div>
              <div class="task-desc">${tk.description}</div>
              <div class="task-meta">${tk.duration_min} min · +${tk.points} XP · ${tk.tag || tk.parameter_key}</div>
            </div>
            <button type="button" class="btn success small" data-complete-task="${tk.id}">${t("completeBtn")} (+${tk.points} XP)</button>
          </div>
        `).join("");
        activeEl.querySelectorAll("[data-complete-task]").forEach((btn) =>
          btn.addEventListener("click", () => completeTask(Number(btn.dataset.completeTask)))
        );
      }
    }

    const doneEl = document.getElementById("doneTasksList");
    if (doneEl) {
      doneEl.innerHTML = !state.doneTasks.length
        ? `<p class="dim-note">No completed protocols in this session.</p>`
        : state.doneTasks.slice(0, 12).map((tk) => `
          <div class="task-item" style="opacity:0.72;">
            <div class="task-info">
              <div class="task-title">${tk.title}</div>
              <div class="task-meta">+${tk.points} XP · ${tk.completed_at ? tk.completed_at.slice(0, 16).replace("T", " ") : ""}</div>
            </div>
          </div>
        `).join("");
    }
  }

  function renderHealthHistory() {
    const kpiEl = document.getElementById("healthHistoryKpiStrip");
    const tbody = document.getElementById("healthHistoryTableBody");
    const hist = state.history || [];

    if (kpiEl) {
      if (!hist.length) {
        kpiEl.innerHTML = `
          <div class="kpi-cell"><div class="kpi-label"><span>Recorded Snapshots</span></div><div class="kpi-val">0</div><div class="kpi-meta">Slot ${escHtml(state.astronautId)}</div></div>
          <div class="kpi-cell"><div class="kpi-label"><span>Latest Health Risk</span></div><div class="kpi-val">--%</div><div class="kpi-meta">No record yet</div></div>
          <div class="kpi-cell"><div class="kpi-label"><span>Mean Recorded Risk</span></div><div class="kpi-val">--%</div><div class="kpi-meta">Chronological mean</div></div>
          <div class="kpi-cell"><div class="kpi-label"><span>Peak Recorded Risk</span></div><div class="kpi-val">--%</div><div class="kpi-meta">Maximum observed</div></div>
        `;
      } else {
        const latest = hist[hist.length - 1];
        const prev = hist.length > 1 ? hist[hist.length - 2] : null;
        const latestPct = Number(latest.overall_risk_pct ?? 0);
        const latestLv = latest.level || riskLevel(latestPct);
        const delta = prev ? latestPct - Number(prev.overall_risk_pct ?? 0) : 0;
        const deltaTxt = prev ? `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}% vs prior` : "Initial baseline";

        const risks = hist.map((h) => Number(h.overall_risk_pct ?? 0));
        const meanPct = risks.reduce((a, b) => a + b, 0) / risks.length;
        const nominalCount = hist.filter((h) => Number(h.overall_risk_pct ?? 0) <= 40).length;

        let peakRec = hist[0];
        hist.forEach((h) => {
          if (Number(h.overall_risk_pct ?? 0) >= Number(peakRec.overall_risk_pct ?? 0)) peakRec = h;
        });
        const peakPct = Number(peakRec.overall_risk_pct ?? 0);
        const peakLv = peakRec.level || riskLevel(peakPct);

        kpiEl.innerHTML = `
          <div class="kpi-cell">
            <div class="kpi-label"><span>Recorded Snapshots</span><span class="mono-meta">${escHtml(state.astronautId)}</span></div>
            <div class="kpi-val">${hist.length}</div>
            <div class="kpi-meta">${nominalCount}/${hist.length} within nominal band (&lt;=40%)</div>
          </div>
          <div class="kpi-cell">
            <div class="kpi-label"><span>Latest Health Risk</span><span class="risk-dot ${latestLv}"></span></div>
            <div class="kpi-val">${latestPct.toFixed(1)}%</div>
            <div class="kpi-meta">${deltaTxt} · Day ${Number(latest.mission_day || 0).toFixed(0)}</div>
          </div>
          <div class="kpi-cell">
            <div class="kpi-label"><span>Mean Recorded Risk</span><span class="risk-dot ${riskLevel(meanPct)}"></span></div>
            <div class="kpi-val">${meanPct.toFixed(1)}%</div>
            <div class="kpi-meta">Across ${hist.length} persisted snapshot(s)</div>
          </div>
          <div class="kpi-cell">
            <div class="kpi-label"><span>Peak Recorded Risk</span><span class="risk-dot ${peakLv}"></span></div>
            <div class="kpi-val">${peakPct.toFixed(1)}%</div>
            <div class="kpi-meta">${LEVEL_NAMES[peakLv] || "Baseline"} · Day ${Number(peakRec.mission_day || 0).toFixed(0)}</div>
          </div>
        `;
      }
    }

    if (!tbody) return;
    const indexed = hist.map((rec, idx) => ({ ...rec, seqNo: idx + 1 })).reverse();
    const q = (state.historySearch || "").trim().toLowerCase();
    const filtered = indexed.filter((rec) => {
      const pct = Number(rec.overall_risk_pct ?? 0);
      if (state.historyBandFilter === "elevated" && pct <= 40) return false;
      if (!q) return true;
      const lv = rec.level || riskLevel(pct);
      const flags = (rec.top_risks || []).map((r) => `${r.name} ${r.key}`).join(" ");
      const hay = `${rec.ts || ""} day ${rec.mission_day || ""} ${LEVEL_NAMES[lv] || lv} ${rec.subjective_log || ""} ${flags}`.toLowerCase();
      return hay.includes(q);
    });

    if (!filtered.length) {
      tbody.innerHTML = `
        <tr>
          <td colspan="9" class="history-empty">
            ${hist.length ? "No health history records match the active filter." : "No health snapshots recorded yet. Click 'Record Health Snapshot' above to store an entry."}
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = filtered.map((rec) => {
      const pct = Number(rec.overall_risk_pct ?? 0);
      const lv = rec.level || riskLevel(pct);
      const bandName = LEVEL_NAMES[lv] || "Baseline";
      const tsFmt = (rec.ts || "").slice(0, 19).replace("T", " ");
      const mDay = `Day ${Number(rec.mission_day ?? 0).toFixed(0)}`;

      const physPct = Number(rec.physiological_risk_pct ?? pct).toFixed(1);
      const vTxt = rec.voice_risk_pct !== null && rec.voice_risk_pct !== undefined
        ? `${Number(rec.voice_risk_pct).toFixed(0)}%`
        : "--";
      const eTxt = rec.env_hazard_pct !== null && rec.env_hazard_pct !== undefined
        ? `${Number(rec.env_hazard_pct).toFixed(0)}%`
        : "--";

      const vit = rec.vitals || {};
      const hrTxt = vit.heart_rate !== null && vit.heart_rate !== undefined ? `${Number(vit.heart_rate).toFixed(0)} bpm` : "--";
      const spo2Txt = vit.spo2 !== null && vit.spo2 !== undefined ? `${Number(vit.spo2).toFixed(1)}%` : "--";
      const bpTxt = vit.systolic_bp !== null && vit.systolic_bp !== undefined && vit.diastolic_bp !== null && vit.diastolic_bp !== undefined
        ? `${Number(vit.systolic_bp).toFixed(0)}/${Number(vit.diastolic_bp).toFixed(0)}`
        : "--";
      const slpTxt = vit.sleep_hours !== null && vit.sleep_hours !== undefined ? `${Number(vit.sleep_hours).toFixed(1)}h` : "--";

      const topRisks = Array.isArray(rec.top_risks) ? rec.top_risks : [];
      const flagsHtml = topRisks.length
        ? `<span class="mono-meta" style="margin-right:4px;">${rec.n_at_risk || topRisks.length}:</span>` +
          topRisks.map((r) => `<span class="hh-flag-tag">${escHtml(r.name)} ${Number(r.risk_pct || 0).toFixed(0)}%</span>`).join("")
        : `<span class="mono-meta">Nominal (0/31)</span>`;

      const noteTxt = rec.subjective_log ? escHtml(rec.subjective_log) : `<span class="mono-meta">--</span>`;
      const delBtn = rec.id !== undefined && rec.id !== null
        ? `<button type="button" class="btn small" data-del-hist="${rec.id}">Delete</button>`
        : "";

      return `
        <tr>
          <td class="mono-meta">#${String(rec.seqNo).padStart(2, "0")}</td>
          <td class="mono-meta">${escHtml(tsFmt)}</td>
          <td class="mono-meta">${escHtml(mDay)}</td>
          <td>
            <span class="hh-band-chip ${lv}">
              <span class="risk-dot ${lv}"></span>
              <strong>${pct.toFixed(1)}%</strong>
              <span>· ${bandName}</span>
            </span>
          </td>
          <td class="hh-sub-metrics">P ${physPct}% · V ${vTxt} · H ${eTxt}</td>
          <td class="hh-sub-metrics">${hrTxt} · ${spo2Txt} · ${bpTxt} · ${slpTxt}</td>
          <td>${flagsHtml}</td>
          <td class="hh-note-cell" title="${escHtml(rec.subjective_log || "")}">${noteTxt}</td>
          <td>${delBtn}</td>
        </tr>
      `;
    }).join("");

    tbody.querySelectorAll("[data-del-hist]").forEach((btn) => {
      btn.addEventListener("click", () => deleteHealthSnapshot(Number(btn.dataset.delHist)));
    });
  }

  function renderAllScreens() {
    renderHUD();
    renderHome();
    renderOverallHealth();
    renderVoiceStudio();
    renderCategoryInputScreens();
    renderProgressAndRewards();
    renderHealthHistory();
    redrawVisibleCanvases();
  }

  /* -----------------------------------------------------------------------
     9. API CALLS & INTERACTIVE WORKFLOWS
     ----------------------------------------------------------------------- */
  async function refreshTasksAndGamification() {
    const aid = encodeURIComponent(state.astronautId);
    const [pTasks, dTasks, gam, hist] = await Promise.all([
      fetch(`/astronauts/${aid}/tasks?status=pending&lang=en`).then((r) => r.json()),
      fetch(`/astronauts/${aid}/tasks?status=done&lang=en`).then((r) => r.json()),
      fetch(`/astronauts/${aid}/gamification`).then((r) => r.json()),
      fetch(`/astronauts/${aid}/history`).then((r) => r.json()),
    ]);
    state.pendingTasks = pTasks;
    state.doneTasks = dTasks;
    state.gamification = gam;
    state.history = hist;
  }

  async function runAssessment(silent = false) {
    const btn = document.getElementById("hudAssessBtn");
    btn.disabled = true;
    btn.textContent = t("runningAI");
    try {
      const aid = encodeURIComponent(state.astronautId);
      let voicePayload = null;
      if (state.voiceResult) {
        const { audio_wav_base64, waveform_preview, mel_spectrogram_preview, ...compactVoice } = state.voiceResult;
        voicePayload = compactVoice;
      }
      const body = {
        values: state.values,
        sensor_series: state.sensorSeries,
        subjective_log: state.subjectiveLog || "",
        mission_day: state.missionDay,
        ecg: state.ecgResult,
        voice: voicePayload,
        user_role: state.userRole,
      };
      const res = await fetch(`/astronauts/${aid}/assess?lang=en`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.assessment = await res.json();
      await refreshTasksAndGamification();
      renderAllScreens();
      if (!silent) {
        const nNew = (state.assessment.new_tasks || []).length;
        showToast(`Overall risk: ${state.assessment.overall.risk_pct}%` + (nNew ? ` · +${nNew} protocol(s)` : ""));
      }
    } catch (err) {
      console.error(err);
      if (!silent) showToast("Assessment error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = t("runAI");
    }
  }

  async function completeTask(taskId) {
    const aid = encodeURIComponent(state.astronautId);
    const res = await fetch(`/astronauts/${aid}/tasks/${taskId}/complete`, { method: "POST" });
    if (!res.ok) return;
    const out = await res.json();
    await refreshTasksAndGamification();
    renderAllScreens();
    let msg = `+${out.points_awarded} XP earned`;
    if (out.leveled_up) msg += ` · Level ${out.state.level} (${out.state.rank})`;
    if (out.new_badges && out.new_badges.length) {
      msg += " · Milestone: " + out.new_badges.map((b) => b.name).join(", ");
    }
    showToast(msg);
  }

  async function deleteHealthSnapshot(readingId) {
    const aid = encodeURIComponent(state.astronautId);
    const res = await fetch(`/astronauts/${aid}/history/${readingId}`, { method: "DELETE" });
    if (!res.ok) return;
    await refreshTasksAndGamification();
    renderAllScreens();
    showToast(`Deleted health snapshot #${readingId}.`);
  }

  async function clearHealthHistory() {
    if (!state.history || !state.history.length) {
      showToast("Health history is already empty.");
      return;
    }
    const aid = encodeURIComponent(state.astronautId);
    const res = await fetch(`/astronauts/${aid}/history`, { method: "DELETE" });
    if (!res.ok) return;
    const out = await res.json();
    await refreshTasksAndGamification();
    renderAllScreens();
    showToast(`Cleared ${out.cleared || 0} recorded health snapshot(s).`);
  }

  function exportHealthHistoryCsv() {
    const hist = state.history || [];
    if (!hist.length) {
      showToast("No recorded health history to export.");
      return;
    }
    const csvCell = (v) => {
      const s = String(v ?? "");
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const headers = [
      "record_no",
      "id",
      "timestamp_utc",
      "mission_day",
      "overall_risk_pct",
      "risk_band",
      "physiological_risk_pct",
      "voice_risk_pct",
      "habitat_hazard_pct",
      "n_off_nominal",
      "heart_rate_bpm",
      "spo2_pct",
      "systolic_bp_mmhg",
      "diastolic_bp_mmhg",
      "sleep_hours",
      "flagged_biomarkers",
      "note",
    ];
    const lines = [headers.join(",")];
    hist.forEach((r, idx) => {
      const vit = r.vitals || {};
      const flags = (r.top_risks || []).map((tItem) => `${tItem.name} (${tItem.risk_pct}%)`).join("; ");
      lines.push([
        idx + 1,
        r.id ?? "",
        r.ts ?? "",
        r.mission_day ?? 0,
        r.overall_risk_pct ?? 0,
        LEVEL_NAMES[r.level || riskLevel(r.overall_risk_pct)] || r.level || "",
        r.physiological_risk_pct ?? "",
        r.voice_risk_pct ?? "",
        r.env_hazard_pct ?? "",
        r.n_at_risk ?? 0,
        vit.heart_rate ?? "",
        vit.spo2 ?? "",
        vit.systolic_bp ?? "",
        vit.diastolic_bp ?? "",
        vit.sleep_hours ?? "",
        flags,
        r.subjective_log ?? "",
      ].map(csvCell).join(","));
    });
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `astrocytex_health_history_${state.astronautId}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    showToast(`Exported ${hist.length} health history record(s) to CSV.`);
  }

  async function applyMissionProfile(profileName) {
    state.profile = profileName;
    document.getElementById("hudProfile").value = profileName;
    const res = await fetch(`/environment/profiles?mission_day=${state.missionDay}`).then((r) => r.json());
    state.profiles = res.profiles;
    const prof = res.profiles[profileName];
    if (prof && prof.values) {
      Object.entries(prof.values).forEach(([k, v]) => setParamValue(k, v));
    }
    document.querySelectorAll("#profileChipsRow .chip-btn").forEach((b) =>
      b.classList.toggle("active", b.dataset.prof === profileName)
    );
    await runAssessment();
  }

  async function applyScenario(scenName) {
    state.scenario = scenName;
    document.getElementById("hudScenario").value = scenName;
    const res = await fetch(`/presets/scenario/${encodeURIComponent(scenName)}?lang=en`).then((r) => r.json());
    Object.entries(res.values || {}).forEach(([k, v]) => setParamValue(k, v));
    state.subjectiveLog = res.subjective_log || "";
    document.getElementById("subjLogInput").value = state.subjectiveLog;
    state.sensorSeries = res.sensor_series || state.sensorSeries;
    await runAssessment();
  }

  async function injectEnvironmentEvent(evName) {
    const res = await fetch("/environment/apply-event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values: state.values, event: evName, lang: "en" }),
    }).then((r) => r.json());
    Object.entries(res.values || {}).forEach(([k, v]) => setParamValue(k, v));
    if (res.log && !state.subjectiveLog.includes(res.log)) {
      state.subjectiveLog = (state.subjectiveLog + " " + res.log).trim();
      document.getElementById("subjLogInput").value = state.subjectiveLog;
    }
    showToast(`Simulated event: ${formatScenarioTitle(evName)}`);
    await runAssessment();
  }

  async function loadEcgSample(cls) {
    const res = await fetch(`/ecg/sample/${encodeURIComponent(cls)}`);
    if (!res.ok) return;
    state.ecgResult = await res.json();
    renderEcgBanner();
    drawECGCanvas();
  }

  function renderEcgBanner() {
    const ecg = state.ecgResult;
    const banner = document.getElementById("ecgResultBanner");
    const probsEl = document.getElementById("ecgProbBars");
    if (!ecg) { banner.innerHTML = ""; probsEl.innerHTML = ""; return; }
    banner.innerHTML = `
      <div class="briefing-banner row-between">
        <div>
          <div style="font-size:13.5px;font-weight:600;">${ecg.class_code} — ${ecg.class_name} (${ecg.source})</div>
          <div class="mono-meta" style="margin-top:3px;">
            HR ${(ecg.heart_rate || 0).toFixed(1)} bpm · QTc ${(ecg.qtc_ms || 0).toFixed(0)} ms ·
            Rhythm: ${ecg.rr_regularity} · Abnormal probability: ${(ecg.abnormal_probability_pct || 0).toFixed(1)}%
          </div>
        </div>
        <button type="button" class="btn success small" id="applyEcgNowBtn">${t("applyEcgBtn")}</button>
      </div>
    `;
    document.getElementById("applyEcgNowBtn").addEventListener("click", () => {
      Object.entries(ecg.parameter_values || {}).forEach(([k, v]) => setParamValue(k, v));
      runAssessment();
    });

    const rawProbs = Array.isArray(ecg.probabilities)
      ? ecg.probabilities.map((p) => [p.code ? `${p.code} (${p.name || ""})` : p.name, Number(p.pct || 0)])
      : Object.entries(ecg.probabilities || {}).map(([k, v]) => [k, Number(v) * 100]);
    const probs = rawProbs.slice(0, 6);
    probsEl.innerHTML = probs.length ? `<div class="grid-3">` + probs.map(([cName, pctVal]) => `
      <div class="sub-panel" style="padding:8px 10px;">
        <div style="display:flex;justify-content:space-between;font-size:12px;"><span>${cName}</span><span class="mono-meta">${pctVal.toFixed(1)}%</span></div>
        <div class="mini-bar"><span style="width:${Math.min(100, pctVal)}%;background:#9b87f5"></span></div>
      </div>
    `).join("") + `</div>` : "";
  }

  function appendSystemChatNotice(text) {
    const chatBox = document.getElementById("chatMessagesBox");
    if (!chatBox) return;
    const div = document.createElement("div");
    div.className = "chat-bubble ai";
    div.style.fontSize = "11.5px";
    div.style.color = "var(--text-dim)";
    div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  /* -----------------------------------------------------------------------
     10. INITIALIZATION
     ----------------------------------------------------------------------- */
  async function init() {
    const [healthMeta, reg, presets] = await Promise.all([
      fetch("/health").then((r) => r.json()),
      fetch("/parameters").then((r) => r.json()),
      fetch("/presets").then((r) => r.json()),
    ]);

    state.healthMeta = healthMeta;
    state.registry = reg.parameters;
    state.categories = reg.categories;
    reg.parameters.forEach((p) => { state.paramByKey[p.key] = p; });
    state.values = { ...presets.default_values };
    state.profiles = presets.profiles;
    state.events = presets.events;
    state.scenarios = presets.scenarios;
    state.sensorSeries = presets.initial_sensor_series;

    const issVals = presets.profiles.iss_leo?.values || {};
    Object.assign(state.values, issVals);

    // Topbar dropdowns
    const scenSel = document.getElementById("hudScenario");
    scenSel.innerHTML = state.scenarios.map((s) =>
      `<option value="${s}">Scenario: ${formatScenarioTitle(s)}</option>`
    ).join("");
    scenSel.addEventListener("change", () => applyScenario(scenSel.value));

    const profSel = document.getElementById("hudProfile");
    profSel.innerHTML = Object.entries(state.profiles).map(([k, v]) =>
      `<option value="${k}" ${k === state.profile ? "selected" : ""}>${v.en}</option>`
    ).join("");
    profSel.addEventListener("change", () => applyMissionProfile(profSel.value));

    document.getElementById("hudAstronaut").addEventListener("change", (e) => {
      state.astronautId = e.target.value;
      runAssessment(true);
    });
    document.getElementById("hudAssessBtn").addEventListener("click", () => runAssessment());
    document.getElementById("hudHealthPill").addEventListener("click", () => showPage("overall-health"));
    document.getElementById("hudVoicePill").addEventListener("click", () => showPage("mental-health"));
    document.getElementById("hudEnvPill").addEventListener("click", () => showPage("space-radiation"));
    document.getElementById("hudXpPill").addEventListener("click", () => showPage("rewards"));

    document.querySelectorAll("[data-run-ai]").forEach((b) =>
      b.addEventListener("click", () => runAssessment())
    );

    // Role buttons ("I am a doctor" / "I am not a doctor")
    document.querySelectorAll("[data-role-btn]").forEach((btn) => {
      btn.addEventListener("click", () => setUserRole(btn.dataset.roleBtn, true));
    });

    // Voice studio controls
    const vChip = document.getElementById("voiceModelStatusChip");
    if (vChip) {
      vChip.textContent = healthMeta.voice?.model_loaded
        ? `${healthMeta.voice.model_version} (MCC ${healthMeta.voice.validation_mcc ?? "ready"})`
        : "Acoustic Prior Ready";
    }
    const voicePresets = [
      ["calm_regulated", "Calm / Regulated"],
      ["cognitive_fatigue", "Cognitive Fatigue"],
      ["acute_stress", "Acute Stress"],
      ["low_mood_isolation", "Low Mood / Isolation"],
      ["dyspnea_hypoxic", "Dyspnea / Hypoxic Strain"],
    ];
    const vSampleRow = document.getElementById("voiceSampleBtns");
    vSampleRow.innerHTML = voicePresets.map(([key, enLbl]) => `
      <button type="button" class="chip-btn ${key === state.voiceActiveSample ? "active" : ""}" data-vsample="${key}">
        ${enLbl}
      </button>
    `).join("");
    vSampleRow.querySelectorAll("[data-vsample]").forEach((b) =>
      b.addEventListener("click", () => loadVoiceSample(b.dataset.vsample, true))
    );

    document.getElementById("voiceSexSelect").addEventListener("change", (e) => {
      state.voiceSex = e.target.value;
      loadVoiceSample(state.voiceActiveSample, true);
    });
    document.getElementById("voiceRecordBtn").addEventListener("click", toggleVoiceRecording);
    document.getElementById("voiceFileInput").addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) handleVoiceFileUpload(e.target.files[0]);
    });

    // Space Environment profile & event buttons
    const pRow = document.getElementById("profileChipsRow");
    pRow.innerHTML = Object.entries(state.profiles).map(([k, v]) =>
      `<button type="button" class="chip-btn ${k === state.profile ? "active" : ""}" data-prof="${k}">${v.en}</button>`
    ).join("");
    pRow.querySelectorAll("[data-prof]").forEach((b) =>
      b.addEventListener("click", () => applyMissionProfile(b.dataset.prof))
    );

    const eRow = document.getElementById("eventChipsRow");
    eRow.innerHTML = Object.entries(state.events).map(([k, v]) =>
      `<button type="button" class="chip-btn" data-ev="${k}">${v.en}</button>`
    ).join("");
    eRow.querySelectorAll("[data-ev]").forEach((b) =>
      b.addEventListener("click", () => injectEnvironmentEvent(b.dataset.ev))
    );

    const mdSlider = document.getElementById("missionDaySlider");
    mdSlider.addEventListener("change", () => {
      state.missionDay = Number(mdSlider.value);
      document.getElementById("missionDayBadge").textContent = `Day ${state.missionDay}`;
      applyMissionProfile(state.profile);
    });
    mdSlider.addEventListener("input", () => {
      document.getElementById("missionDayBadge").textContent = `Day ${mdSlider.value}`;
    });

    // Subjective log quick presets (100% English)
    const moodRow = document.getElementById("moodQuickChips");
    const moods = [
      "Mild headache and fatigue after shift",
      "Lightheaded upon standing, increased thirst",
      "Shortness of breath and palpitations",
      "Poor sleep quality and low concentration",
      "Rested, calm, and focused today",
    ];
    moodRow.innerHTML = moods.map((m) => `<button type="button" class="chip-btn" data-mood="${m}">${m}</button>`).join("");
    moodRow.querySelectorAll("[data-mood]").forEach((b) => b.addEventListener("click", () => {
      state.subjectiveLog = b.dataset.mood;
      document.getElementById("subjLogInput").value = state.subjectiveLog;
    }));
    document.getElementById("subjLogInput").addEventListener("input", (e) => {
      state.subjectiveLog = e.target.value;
    });

    // Exercise quick workout buttons
    document.querySelectorAll("[data-workout]").forEach((b) => b.addEventListener("click", () => {
      const delta = Number(b.dataset.workout);
      const next = delta < 0 ? 0 : (state.values.exercise_min || 0) + delta;
      setParamValue("exercise_min", next);
      runAssessment();
    }));

    // Telemetry simulation button
    document.getElementById("simTelemetryBtn").addEventListener("click", async () => {
      const res = await fetch("/telemetry/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values: state.values }),
      }).then((r) => r.json());
      state.sensorSeries = res.sensor_series;
      Object.entries(res.latest || {}).forEach(([k, v]) => setParamValue(k, v));
      redrawVisibleCanvases();
      showToast("24-hour sensor stream refreshed.");
    });

    // ECG sample buttons + upload
    const ecgClasses = ["NORMAL", "AFIB", "AFL", "LBBB", "RBBB", "MI", "PVC", "PAC", "AVB", "STTC"];
    document.getElementById("ecgModeBadge").textContent = `ECG: ${healthMeta.ecg?.mode || "ready"}`;
    const ecgRow = document.getElementById("ecgSampleBtns");
    ecgRow.innerHTML = ecgClasses.map((c) => `<button type="button" class="chip-btn" data-ecg="${c}">${c}</button>`).join("");
    ecgRow.querySelectorAll("[data-ecg]").forEach((b) => b.addEventListener("click", () => loadEcgSample(b.dataset.ecg)));

    document.getElementById("ecgFileInput").addEventListener("change", async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const fd = new FormData();
      fd.append("file", f);
      const res = await fetch("/ecg/analyze", { method: "POST", body: fd });
      if (res.ok) {
        state.ecgResult = await res.json();
        renderEcgBanner();
        drawECGCanvas();
        showToast(`ECG analyzed: ${state.ecgResult.class_code}`);
      }
    });

    // Role-Adaptive Advisor Chat
    document.getElementById("geminiStatusBadge").textContent = healthMeta.gemini?.configured ? "Gemini Online" : "Offline Reasoner";
    const chatBox = document.getElementById("chatMessagesBox");
    const addBubble = (who, text) => {
      const div = document.createElement("div");
      div.className = `chat-bubble ${who}`;
      div.textContent = text;
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
    };
    addBubble(
      "ai",
      "AstrocyteX medical advisor ready. Select 'I am a doctor' for clinical pathophysiology and quantitative biomarkers, or 'I am not a doctor' for plain-language guidance."
    );

    document.getElementById("chatForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const inp = document.getElementById("chatInput");
      const q = inp.value.trim();
      if (!q) return;
      inp.value = "";
      addBubble("user", q);
      const aid = encodeURIComponent(state.astronautId);
      const res = await fetch(`/astronauts/${aid}/chat?lang=en`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, role: state.userRole }),
      }).then((r) => r.json());
      const rBadge = res.role === "doctor" ? "Clinical Mode" : "Standard Mode";
      addBubble("ai", `[${res.source} · ${rBadge}] ${res.answer}`);
    });

    // Recorded Health History & Column Analytics controls
    const snapNoteInp = document.getElementById("healthSnapshotNoteInput");
    document.querySelectorAll("#historyNotePresetChips [data-hnote]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (snapNoteInp) snapNoteInp.value = btn.dataset.hnote;
      });
    });

    const recSnapBtn = document.getElementById("recordHealthSnapshotBtn");
    if (recSnapBtn) {
      recSnapBtn.addEventListener("click", async () => {
        recSnapBtn.disabled = true;
        const customNote = snapNoteInp ? snapNoteInp.value.trim() : "";
        const prevLog = state.subjectiveLog;
        if (customNote) {
          state.subjectiveLog = prevLog ? `${customNote} — ${prevLog}` : customNote;
        }
        try {
          await runAssessment(true);
          if (snapNoteInp) snapNoteInp.value = "";
          const count = (state.history || []).length;
          const pct = state.assessment?.overall?.risk_pct ?? 0;
          showToast(`Recorded health snapshot #${count} (${Number(pct).toFixed(1)}%)`);
        } finally {
          state.subjectiveLog = prevLog;
          recSnapBtn.disabled = false;
        }
      });
    }

    document.querySelectorAll("#historyColModeChips [data-hcol-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.historyColMode = btn.dataset.hcolMode;
        document.querySelectorAll("#historyColModeChips [data-hcol-mode]").forEach((b) =>
          b.classList.toggle("active", b.dataset.hcolMode === state.historyColMode)
        );
        drawHealthHistoryColumns();
      });
    });

    document.querySelectorAll("#historyBandFilterChips [data-hband]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.historyBandFilter = btn.dataset.hband;
        document.querySelectorAll("#historyBandFilterChips [data-hband]").forEach((b) =>
          b.classList.toggle("active", b.dataset.hband === state.historyBandFilter)
        );
        renderHealthHistory();
      });
    });

    const histSearchInp = document.getElementById("healthHistorySearch");
    if (histSearchInp) {
      histSearchInp.addEventListener("input", () => {
        state.historySearch = histSearchInp.value;
        renderHealthHistory();
      });
    }

    const expHistBtn = document.getElementById("exportHealthHistoryBtn");
    if (expHistBtn) {
      expHistBtn.addEventListener("click", exportHealthHistoryCsv);
    }

    const clrHistBtn = document.getElementById("clearHealthHistoryBtn");
    if (clrHistBtn) {
      clrHistBtn.addEventListener("click", clearHealthHistory);
    }

    applyStaticTranslations();
    showPage(location.hash.slice(1) || "home");
    await Promise.all([loadEcgSample("NORMAL"), loadVoiceSample("calm_regulated", false)]);
    await runAssessment(true);
  }

  init();
})();
