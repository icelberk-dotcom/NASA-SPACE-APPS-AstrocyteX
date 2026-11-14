/* ==========================================================================
   Astrocyte X · AstroGuard-X — Core SPA Engine (v2.0)
   Starfield physics · Router · Live 43-param risk · FastAPI client · Charts
   ========================================================================== */

(() => {
  /* -----------------------------------------------------------------------
     1. INTERACTIVE STARFIELD CANVAS (Mouse-repulsion + spring physics)
     ----------------------------------------------------------------------- */
  const starCanvas = document.getElementById("stars");
  const sCtx = starCanvas.getContext("2d");
  let stars = [];
  const mouse = { x: -9999, y: -9999 };
  const MOUSE_RADIUS = 95;
  const PUSH_STRENGTH = 0.85;
  const SPRING = 0.04;
  const FRICTION = 0.88;

  function createStars() {
    starCanvas.width = window.innerWidth;
    starCanvas.height = window.innerHeight;
    const count = Math.floor((starCanvas.width * starCanvas.height) / 4800);
    stars = Array.from({ length: count }, () => {
      const x = Math.random() * starCanvas.width;
      const y = Math.random() * starCanvas.height;
      return {
        x, y, homeX: x, homeY: y, vx: 0, vy: 0,
        radius: Math.random() * 1.45 + 0.3,
        alpha: Math.random(),
        speed: Math.random() * 0.015 + 0.003,
        direction: Math.random() < 0.5 ? 1 : -1,
        hue: Math.random() < 0.25 ? "205, 225, 255" : "255, 255, 255",
      };
    });
  }

  window.addEventListener("mousemove", (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
  window.addEventListener("mouseout", () => { mouse.x = -9999; mouse.y = -9999; });
  window.addEventListener("resize", () => { createStars(); redrawVisibleCanvases(); });

  function animateStars() {
    sCtx.clearRect(0, 0, starCanvas.width, starCanvas.height);
    for (const s of stars) {
      const dx = s.x - mouse.x;
      const dy = s.y - mouse.y;
      const dist = Math.hypot(dx, dy);
      if (dist < MOUSE_RADIUS && dist > 0) {
        const force = (1 - dist / MOUSE_RADIUS) * PUSH_STRENGTH;
        s.vx += (dx / dist) * force;
        s.vy += (dy / dist) * force;
      }
      s.vx += (s.homeX - s.x) * SPRING;
      s.vy += (s.homeY - s.y) * SPRING;
      s.vx *= FRICTION;
      s.vy *= FRICTION;
      s.x += s.vx;
      s.y += s.vy;

      s.alpha += s.speed * s.direction;
      if (s.alpha >= 1) { s.alpha = 1; s.direction = -1; }
      if (s.alpha <= 0.16) { s.alpha = 0.16; s.direction = 1; }

      sCtx.beginPath();
      sCtx.arc(s.x, s.y, s.radius, 0, Math.PI * 2);
      sCtx.fillStyle = `rgba(${s.hue}, ${s.alpha})`;
      sCtx.fill();
    }
    requestAnimationFrame(animateStars);
  }
  createStars();
  animateStars();

  /* -----------------------------------------------------------------------
     2. STATE & TRANSLATIONS
     ----------------------------------------------------------------------- */
  let currentLang = localStorage.getItem("language") || "tr";
  if (!window.TRANSLATIONS[currentLang]) currentLang = "tr";

  const state = {
    astronautId: "astro-1",
    missionDay: 90,
    profile: "iss_leo",
    scenario: "nominal",
    subjectiveLog: "",
    values: {},
    sensorSeries: null,
    ecgResult: null,
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
  };

  function t(key) {
    return (window.TRANSLATIONS[currentLang] && window.TRANSLATIONS[currentLang][key])
      || window.TRANSLATIONS.en[key]
      || key;
  }
  window.agTranslate = t;

  function apiLang() {
    return currentLang === "tr" ? "tr" : "en";
  }

  function paramLabel(p) {
    return currentLang === "tr" && p.name_tr ? p.name_tr : p.name;
  }

  function catLabel(catObj) {
    if (!catObj) return "";
    return currentLang === "tr" && catObj.tr ? catObj.tr : catObj.en;
  }

  function showToast(msg) {
    const box = document.getElementById("toastContainer");
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = msg;
    box.appendChild(el);
    setTimeout(() => el.remove(), 3400);
  }

  /* -----------------------------------------------------------------------
     3. PIECEWISE-LINEAR LIVE RISK ENGINE (Exact match to reference_ranges.py)
     ----------------------------------------------------------------------- */
  const LEVEL_COLORS = {
    gray: "#9aa0a6",
    green: "#2ecc71",
    yellow: "#f1c40f",
    red: "#e74c3c",
    black: "#ffffff",
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
    if (lo !== null && hi !== null) return `${lo} – ${hi}${u}`;
    if (lo !== null) return `≥ ${lo}${u}`;
    if (hi !== null) return `≤ ${hi}${u}`;
    return "—";
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
     4. MENU, ROUTING & LANGUAGE SELECTOR
     ----------------------------------------------------------------------- */
  const menu = document.getElementById("menu");
  const menuButton = document.getElementById("menuButton");
  const menuItems = document.querySelectorAll(".menu-list button[data-page]");

  function setMenuOpen(open) {
    menu.classList.toggle("open", open);
    menuButton.setAttribute("aria-expanded", String(open));
  }

  function showPage(pageId) {
    const id = pageId || "home";
    document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
    const target = document.getElementById(id) || document.getElementById("home");
    target.classList.add("active");

    menuItems.forEach((btn) => btn.classList.toggle("current", btn.dataset.page === id));
    document.querySelectorAll(".menu-group").forEach((group) => {
      const parentBtn = group.querySelector(":scope > button");
      const inside = parentBtn.dataset.page === id || group.querySelector(`.submenu button[data-page="${id}"]`);
      if (inside) {
        group.classList.add("expanded");
        parentBtn.setAttribute("aria-expanded", "true");
      }
    });
    setTimeout(redrawVisibleCanvases, 40);
  }
  window.agShowPage = showPage;

  menuButton.addEventListener("click", (e) => {
    e.stopPropagation();
    setLangOpen(false);
    setMenuOpen(!menu.classList.contains("open"));
  });

  menuItems.forEach((item) => {
    item.addEventListener("click", () => {
      const group = item.closest(".menu-group");
      const isGroupTab = group && item === group.querySelector(":scope > button");
      const wasExpanded = isGroupTab && group.classList.contains("expanded");
      showPage(item.dataset.page);
      history.replaceState(null, "", "#" + item.dataset.page);
      if (isGroupTab) {
        group.classList.toggle("expanded", !wasExpanded);
        item.setAttribute("aria-expanded", String(!wasExpanded));
        return;
      }
      setMenuOpen(false);
    });
  });

  const langBox = document.getElementById("lang");
  const langButton = document.getElementById("langButton");
  const langList = document.getElementById("langList");

  function setLangOpen(open) {
    langBox.classList.toggle("open", open);
    langButton.setAttribute("aria-expanded", String(open));
  }

  function applyLanguage(code) {
    if (!window.TRANSLATIONS[code]) code = "en";
    currentLang = code;
    document.documentElement.lang = code;

    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const txt = window.TRANSLATIONS[code][el.dataset.i18n];
      if (txt) el.textContent = txt;
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      const txt = window.TRANSLATIONS[code][el.dataset.i18nPlaceholder];
      if (txt) el.placeholder = txt;
    });

    langList.querySelectorAll("button").forEach((b) =>
      b.classList.toggle("current", b.dataset.lang === code)
    );
    localStorage.setItem("language", code);
    renderAllScreens();
    document.dispatchEvent(new CustomEvent("app-language-changed"));
  }

  window.LANGUAGES.forEach(({ code, name }) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.dataset.lang = code;
    btn.textContent = name;
    btn.addEventListener("click", () => {
      applyLanguage(code);
      setLangOpen(false);
      runAssessment(true);
    });
    li.appendChild(btn);
    langList.appendChild(li);
  });

  langButton.addEventListener("click", (e) => {
    e.stopPropagation();
    setMenuOpen(false);
    setLangOpen(!langBox.classList.contains("open"));
  });

  document.addEventListener("click", (e) => {
    if (!menu.contains(e.target)) setMenuOpen(false);
    if (!langBox.contains(e.target)) setLangOpen(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { setMenuOpen(false); setLangOpen(false); }
  });

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
          <span>${t("normalCol")}: ${normalRangeText(p)}</span>
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
     6. SVG GAUGE & HIGH-DPI CANVAS CHARTS
     ----------------------------------------------------------------------- */
  function svgRingGauge(pct, level, subtitleHtml) {
    const col = level === "black" ? "#ff5252" : (LEVEL_COLORS[level] || "#6ea8ff");
    const r = 46;
    const circ = 2 * Math.PI * r;
    const dash = (Math.max(0, Math.min(100, pct || 0)) / 100) * circ;
    return `
      <div class="gauge-wrap">
        <svg width="114" height="114" viewBox="0 0 114 114" style="flex-shrink:0;">
          <circle cx="57" cy="57" r="${r}" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="10"/>
          <circle cx="57" cy="57" r="${r}" fill="none" stroke="${col}" stroke-width="10"
            stroke-linecap="round" stroke-dasharray="${dash.toFixed(1)} ${circ.toFixed(1)}"
            transform="rotate(-90 57 57)" style="filter:drop-shadow(0 0 8px ${col});transition:stroke-dasharray 0.5s ease;"/>
          <text x="57" y="63" text-anchor="middle" fill="#fff" font-size="20" font-weight="800">${(pct || 0).toFixed(0)}%</text>
        </svg>
        <div class="gauge-meta">
          <div style="display:flex;align-items:center;gap:9px;">
            <span class="risk-dot lg ${level}"></span>
            <span class="gauge-big">${(pct || 0).toFixed(1)}%</span>
          </div>
          <div class="gauge-sub">${subtitleHtml}</div>
        </div>
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
    const pad = { l: 44, r: 18, t: 24, b: 28 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    // Grid lines
    c.strokeStyle = "rgba(175, 205, 255, 0.12)";
    c.fillStyle = "rgba(205, 222, 255, 0.68)";
    c.font = "11px Segoe UI, sans-serif";
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

    // Legend + curves
    let lx = pad.l;
    seriesList.forEach(({ label, data, color }) => {
      if (!data || !data.length) return;
      c.strokeStyle = color;
      c.lineWidth = 2.2;
      c.beginPath();
      data.forEach((v, idx) => {
        const x = pad.l + (pw * idx) / Math.max(1, data.length - 1);
        const y = pad.t + ph * (1 - (v - yMin) / Math.max(1e-6, yMax - yMin));
        if (idx === 0) c.moveTo(x, y); else c.lineTo(x, y);
      });
      c.stroke();

      c.fillStyle = color;
      c.fillRect(lx, 7, 12, 4);
      c.fillStyle = "#e6f0ff";
      c.fillText(label, lx + 16, 13);
      lx += c.measureText(label).width + 34;
    });
  }

  function drawECGCanvas() {
    const canvas = document.getElementById("ecgCanvas");
    const box = setupCanvas(canvas, 270);
    if (!box) return;
    const { c, w, h } = box;

    // Medical ECG grid
    c.strokeStyle = "rgba(255, 110, 130, 0.11)";
    c.lineWidth = 1;
    for (let x = 0; x < w; x += 16) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke(); }
    for (let y = 0; y < h; y += 16) { c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke(); }

    const ecg = state.ecgResult;
    const arr = (ecg && Array.isArray(ecg.lead_ii) && ecg.lead_ii.length) ? ecg.lead_ii : null;
    if (!arr) {
      c.fillStyle = "rgba(215, 230, 255, 0.65)";
      c.font = "13px Segoe UI, sans-serif";
      c.fillText("Select a clinical demo class above (e.g. AFIB, NORMAL, MI, PVC) or upload a 12-lead ECG file.", 24, h / 2);
      return;
    }

    // Top-half: Full Lead II diagnostic rhythm strip, Bottom-half: Zoomed QRS morphology beat window
    const strips = [
      { title: `Lead II — Full Rhythm Strip (${ecg.record_name || ecg.class_code})`, data: arr },
      { title: `Lead II — High-Resolution QRS-T Morphology Zoom`, data: arr.slice(0, Math.min(320, arr.length)) },
    ];
    const cellH = h / 2;
    strips.forEach((st, idx) => {
      const x0 = 16;
      const y0 = idx * cellH + 8;
      const cw = w - 32;
      const ch = cellH - 18;
      const mn = Math.min(...st.data), mx = Math.max(...st.data);
      const span = Math.max(0.2, mx - mn);

      c.fillStyle = "#9ec5ff";
      c.font = "bold 11.5px Consolas, monospace";
      c.fillText(st.title, x0 + 4, y0 + 13);

      c.strokeStyle = idx === 0 ? "#4df0a8" : "#6ec6ff";
      c.lineWidth = 1.75;
      c.beginPath();
      st.data.forEach((v, i) => {
        const x = x0 + (cw * i) / Math.max(1, st.data.length - 1);
        const y = y0 + ch - ((v - mn) / span) * (ch - 18) - 4;
        if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
      });
      c.stroke();
    });
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
        { label: "HR (bpm)", data: state.sensorSeries.heart_rate, color: "#ff6b81" },
        { label: "SpO₂ (%)", data: state.sensorSeries.spo2, color: "#4de1ff" },
        { label: "Sys BP", data: state.sensorSeries.systolic_bp, color: "#ffd54f" },
        { label: "Resp×4", data: (state.sensorSeries.resp_rate || []).map((x) => x * 4), color: "#69f0ae" },
      ], Array.from({ length: 96 }, (_, i) => `${Math.round((i - 95) * 15 / 60)}h`), 30, 185);
    }

    drawECGCanvas();

    const A = state.assessment;
    if (A && A.physics && A.physics.projections) {
      const cur = A.physics.current || {};
      const p30 = A.physics.projections["+30d"] || {};
      const p90 = A.physics.projections["+90d"] || {};
      const p180 = A.physics.projections["+180d"] || {};
      drawLineSeries("odeCanvas", [
        { label: "Bone BMD Δ%", data: [cur.bone_density_change_pct || 0, p30.bone_density_change_pct || 0, p90.bone_density_change_pct || 0, p180.bone_density_change_pct || 0], color: "#ffd54f" },
        { label: "Muscle Mass Δ%", data: [cur.muscle_mass_change_pct || 0, p30.muscle_mass_change_pct || 0, p90.muscle_mass_change_pct || 0, p180.muscle_mass_change_pct || 0], color: "#ff7a7a" },
        { label: "Plasma Vol Δ%", data: [cur.plasma_volume_change_pct || 0, p30.plasma_volume_change_pct || 0, p90.plasma_volume_change_pct || 0, p180.plasma_volume_change_pct || 0], color: "#6ea8ff" },
      ], ["Now", "+30d", "+90d", "+180d"], -22, 6);
    }

    if (A && A.forecast && A.forecast.length) {
      drawLineSeries("forecastCanvas", [
        { label: "TFT 7-Day Risk %", data: A.forecast, color: "#8ab4ff" },
      ], A.forecast.map((_, i) => `D+${i + 1}`), 0, 100);
    }

    if (state.history && state.history.length) {
      const pts = state.history;
      drawLineSeries("historyCanvas", [
        { label: "Overall Health Risk %", data: pts.map((x) => x.overall_risk_pct ?? x.overall_risk ?? 0), color: "#4df0a8" },
      ], pts.map((_, i) => `#${i + 1}`), 0, 100);
    }
  }

  /* -----------------------------------------------------------------------
     7. RENDER ALL SCREENS
     ----------------------------------------------------------------------- */
  function renderHUD() {
    const A = state.assessment;
    if (A && A.overall) {
      document.getElementById("hudHealthDot").className = `risk-dot ${A.overall.level}`;
      document.getElementById("hudHealthText").textContent = `${t("healthRisk")} ${A.overall.risk_pct.toFixed(0)}%`;
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
    if (state.pendingTasks.length > 0) {
      badge.textContent = String(state.pendingTasks.length);
      badge.hidden = false;
    } else {
      badge.hidden = true;
    }
  }

  function renderHome() {
    const strip = document.getElementById("homeStatusStrip");
    const m = state.healthMeta;
    const modelName = m?.model?.version || "astroguard-net-2.0";
    const ecgMode = m?.ecg?.mode || "ready";
    const gemMode = m?.gemini?.configured ? "Gemini Online" : "Offline Edge AI";
    strip.innerHTML = `
      <span class="status-chip">🧠 ${modelName} (MCC 0.979)</span>
      <span class="status-chip">🛰️ 43 Parameters (31 Bio + 12 NASA RIDGE)</span>
      <span class="status-chip">📈 ECG: ${ecgMode}</span>
      <span class="status-chip">🤖 ${gemMode}</span>
    `;

    const grid = document.getElementById("homeQuickGrid");
    const tiles = [
      { page: "overall-health", icon: "🛡️", title: t("overall-health"), desc: "43-parameter risk matrix, 10 organ systems & 5 RIDGE cards" },
      { page: "space-radiation", icon: "☢️", title: t("space-radiation"), desc: "ISS / Lunar / Mars profiles, SPE flares, CO₂ & counterfactual uplift" },
      { page: "cardiovascular-statistics", icon: "📈", title: t("cardiovascular-statistics"), desc: "24h Mamba-2 wearable stream & 12-lead TEKNOFEST V23 ECG" },
      { page: "diet", icon: "🥗", title: t("diet"), desc: "Space pantry stock bars, 0–100% meal portions, hydration & AI sync" },
      { page: "exercise-tracker", icon: "🦴", title: t("exercise-tracker"), desc: "ARED workouts & Neural ODE +30/+90/+180d bone/muscle projections" },
      { page: "rewards", icon: "🏆", title: t("rewards"), desc: "AI mitigation tasks, XP levels, streaks & 18 spaceflight badges" },
    ];
    grid.innerHTML = tiles.map((tl) => `
      <button type="button" class="home-tile" data-goto="${tl.page}">
        <div class="home-tile-icon">${tl.icon}</div>
        <div class="home-tile-title">${tl.title}</div>
        <div class="home-tile-desc">${tl.desc}</div>
      </button>
    `).join("");
    grid.querySelectorAll("[data-goto]").forEach((btn) =>
      btn.addEventListener("click", () => showPage(btn.dataset.goto))
    );
  }

  function renderOverallHealth() {
    const A = state.assessment;
    if (!A) return;

    const oh = A.overall;
    document.getElementById("overallHealthGaugeCard").innerHTML = `
      <h3 class="card-title"><span>${t("overallHealthTitle")}</span></h3>
      ${svgRingGauge(oh.risk_pct, oh.level, `90% CI: <b>${oh.ci_low}% – ${oh.ci_high}%</b> · At-risk biomarkers: <b>${oh.n_at_risk}</b> / 31`)}
    `;

    const env = A.environment || {};
    const upPts = env.uplift ? env.uplift.uplift_pts : 0;
    const upHtml = upPts > 0.3
      ? `<div class="uplift-pill">⚡ Space environment adds <b>+${upPts.toFixed(1)} pts</b> to physiological risk</div>`
      : "";
    document.getElementById("envHazardGaugeCard").innerHTML = `
      <h3 class="card-title"><span>${t("envHazardTitle")}</span></h3>
      ${svgRingGauge(env.hazard_index_pct || 0, env.level || "gray",
        `Active space hazards: <b>${env.n_hazards || 0}</b> / 12 · Comm delay: <b>${fmtCommDelay(env.derived)}</b>${upHtml}`)}
    `;

    const adv = A.advisor || {};
    const recs = (adv.recommendations || []).map((r) => `<li>${r}</li>`).join("");
    document.getElementById("aiBriefingBox").innerHTML = `
      <div class="briefing-banner ${adv.escalate ? "escalate" : ""}">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
          <strong>🩺 ${t("aiBriefing")} (${adv.source || "AI"})</strong>
          ${adv.escalate ? `<span class="menu-badge" style="background:rgba(231,76,60,0.35);border-color:#ff7a7a;color:#fff;">🚨 ESCALATE TO FLIGHT SURGEON</span>` : ""}
        </div>
        <p class="briefing-summary">${adv.summary || ""}</p>
        ${recs ? `<ul class="briefing-recs">${recs}</ul>` : ""}
      </div>
    `;

    // 10 Organ systems
    const sysGrid = document.getElementById("organSystemsGrid");
    sysGrid.innerHTML = Object.entries(A.systems || {}).map(([cat, s]) => `
      <div class="sys-card" data-goto="${CATEGORY_SCREEN_MAP[cat] || "biochemical-statistics"}">
        <div class="sys-top">
          <span>${s.icon} ${currentLang === "tr" ? s.label_tr : s.label}</span>
          <span class="risk-dot ${s.level}"></span>
        </div>
        <div class="sys-val">${s.risk_pct.toFixed(0)}%</div>
        <div class="mini-bar"><span style="width:${Math.max(3, s.risk_pct)}%;background:${LEVEL_COLORS[s.level]}"></span></div>
      </div>
    `).join("");
    sysGrid.querySelectorAll("[data-goto]").forEach((el) =>
      el.addEventListener("click", () => showPage(el.dataset.goto))
    );

    // 5 NASA RIDGE themes
    const ridgeGrid = document.getElementById("ridgeThemesGrid");
    ridgeGrid.innerHTML = Object.entries((env.themes) || {}).map(([cat, s]) => `
      <div class="sys-card" data-goto="space-radiation">
        <div class="sys-top">
          <span>${s.icon} ${currentLang === "tr" ? s.label_tr : s.label}</span>
          <span class="risk-dot ${s.level}"></span>
        </div>
        <div class="sys-val">${s.risk_pct.toFixed(0)}%</div>
        <div class="mini-bar"><span style="width:${Math.max(3, s.risk_pct)}%;background:${LEVEL_COLORS[s.level]}"></span></div>
      </div>
    `).join("");
    ridgeGrid.querySelectorAll("[data-goto]").forEach((el) =>
      el.addEventListener("click", () => showPage(el.dataset.goto))
    );

    // Filter chips + 43-parameter table
    const fRow = document.getElementById("tableFilterRow");
    fRow.innerHTML = ["all", "risky", "health", "environment"].map((f) => `
      <button type="button" class="chip-btn ${state.tableFilter === f ? "active" : ""}" data-tf="${f}">${f.toUpperCase()}</button>
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
              <td><span class="risk-dot ${p.level}"></span> &nbsp;<b>${currentLang === "tr" ? p.name_tr : p.name}</b></td>
              <td>${p.value !== null ? p.value : "—"} ${p.unit || ""}</td>
              <td style="color:rgba(205,222,255,0.72);">${p.normal_range}</td>
              <td>${p.rule_risk_pct !== null ? p.rule_risk_pct.toFixed(1) + "%" : "—"}</td>
              <td>${p.model_risk_pct !== null ? p.model_risk_pct.toFixed(1) + "%" : "—"}</td>
              <td><b>${p.risk_pct !== null ? p.risk_pct.toFixed(1) + "%" : "—"}</b>
                  ${p.ci_low !== null ? `<span style="font-size:0.76rem;color:rgba(200,220,255,0.65);"> [${p.ci_low}–${p.ci_high}%]</span>` : ""}
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

    // Detected subjective symptoms
    const symBox = document.getElementById("detectedSymptomsBox");
    const syms = (state.assessment && state.assessment.subjective_symptoms) || [];
    symBox.innerHTML = syms.length
      ? `${t("detectedSymptoms")}: ` + syms.map((s) => `<span class="status-chip">${s}</span>`).join(" ")
      : "";

    // Environment derived metrics + hazard attention
    const env = (state.assessment && state.assessment.environment) || {};
    const der = env.derived || {};
    const cumDose = state.values.radiation_cumulative_msv || 0;
    const radCards = document.getElementById("envDerivedCards");
    if (radCards) {
      radCards.innerHTML = `
        <div class="glass-card">
          <div style="font-size:0.84rem;color:#b8d4ff;">📡 ${t("commDelay")}</div>
          <div style="font-size:1.75rem;font-weight:800;margin-top:6px;">${fmtCommDelay(der)}</div>
          <div style="font-size:0.8rem;color:rgba(210,226,255,0.68);margin-top:4px;">Distance: ${state.values.earth_distance_mkm || 0} × 10⁶ km</div>
        </div>
        <div class="glass-card">
          <div style="font-size:0.84rem;color:#b8d4ff;">🚑 ${t("evacTime")}</div>
          <div style="font-size:1.75rem;font-weight:800;margin-top:6px;">${fmtEvacTime(der)}</div>
          <div style="font-size:0.8rem;color:rgba(210,226,255,0.68);margin-top:4px;">Definitive Earth medical care</div>
        </div>
        <div class="glass-card">
          <div style="font-size:0.84rem;color:#b8d4ff;">☢️ ${t("careerDose")}</div>
          <div style="font-size:1.75rem;font-weight:800;margin-top:6px;">${cumDose.toFixed(0)} / 600 mSv</div>
          <div class="mini-bar"><span style="width:${Math.min(100, (cumDose / 600) * 100)}%;background:${cumDose > 350 ? "#e74c3c" : "#2ecc71"}"></span></div>
        </div>
      `;
    }

    const hzBox = document.getElementById("hazardAttentionBars");
    if (hzBox && env.hazard_attention) {
      const entries = Object.entries(env.hazard_attention).sort((a, b) => b[1] - a[1]);
      hzBox.innerHTML = entries.map(([k, w]) => {
        const p = state.paramByKey[k];
        return `
          <div style="margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;font-size:0.84rem;">
              <span>${p ? paramLabel(p) : k}</span>
              <b>${(w * 100).toFixed(1)}%</b>
            </div>
            <div class="mini-bar"><span style="width:${Math.min(100, w * 320)}%;background:#6ea8ff"></span></div>
          </div>
        `;
      }).join("");
    }

    // Neural ODE metrics cards
    const odeRow = document.getElementById("odeMetricsRow");
    if (odeRow && state.assessment && state.assessment.physics && state.assessment.physics.projections) {
      const p180 = state.assessment.physics.projections["+180d"] || {};
      odeRow.innerHTML = `
        <div class="sys-card"><div class="sys-top">🦴 +180d Bone BMD</div><div class="sys-val">${(p180.bone_density_change_pct || 0).toFixed(2)}%</div></div>
        <div class="sys-card"><div class="sys-top">💪 +180d Muscle Mass</div><div class="sys-val">${(p180.muscle_mass_change_pct || 0).toFixed(2)}%</div></div>
        <div class="sys-card"><div class="sys-top">🧠 Cephalad Fluid</div><div class="sys-val">${(p180.fluid_shift_L || 0).toFixed(2)} L</div></div>
        <div class="sys-card"><div class="sys-top">🩸 Plasma Volume Δ</div><div class="sys-val">${(p180.plasma_volume_change_pct || 0).toFixed(1)}%</div></div>
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
        <div style="margin-bottom:14px;">
          ${mods.map(([m, w]) => `
            <div style="margin-bottom:7px;">
              <div style="display:flex;justify-content:space-between;font-size:0.84rem;">
                <span>Modality: <b>${m}</b></span><span>${(w * 100).toFixed(1)}%</span>
              </div>
              <div class="mini-bar"><span style="width:${Math.min(100, w * 220)}%;background:#8ab4ff"></span></div>
            </div>
          `).join("")}
        </div>
        <hr style="border:none;border-top:1px solid rgba(255,255,255,0.12);margin:12px 0;">
        <div>
          ${drivers.map((d) => `
            <div style="display:flex;justify-content:space-between;font-size:0.83rem;padding:4px 0;">
              <span>🔹 ${d.name}</span><b>${(d.weight * 100).toFixed(1)}%</b>
            </div>
          `).join("")}
        </div>
      `;
    }

    // Gamification hero + tasks + badges
    const g = state.gamification;
    const gHero = document.getElementById("gamificationHeroCard");
    if (g && gHero) {
      const startXp = g.level_xp ?? g.current_level_xp ?? 0;
      const span = Math.max(1, (g.next_level_xp || 60) - startXp);
      const pct = g.progress !== undefined ? Math.min(100, Math.max(0, g.progress * 100)) : Math.min(100, Math.max(0, ((g.xp - startXp) / span) * 100));
      gHero.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:10px;">
          <div>
            <div style="font-size:1.5rem;font-weight:800;">🎖️ Level ${g.level} — ${g.rank}</div>
            <div style="font-size:0.88rem;color:rgba(215,230,255,0.8);">${g.xp} XP (Next level at ${g.next_level_xp} XP)</div>
          </div>
          <div style="display:flex;gap:10px;">
            <span class="status-chip">🔥 Streak: <b>${g.streak_days}d</b></span>
            <span class="status-chip">✅ Completed: <b>${g.tasks_completed ?? g.total_completed ?? 0}</b></span>
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
        activeEl.innerHTML = `<p style="color:rgba(215,230,255,0.65);font-size:0.9rem;">${t("noTasksYet")}</p>`;
      } else {
        activeEl.innerHTML = state.pendingTasks.map((tk) => `
          <div class="task-item">
            <div class="task-info">
              <div class="task-title"><span class="risk-dot ${tk.severity_level}"></span> ${tk.title}</div>
              <div class="task-desc">${tk.description}</div>
              <div class="task-meta">⏱ ${tk.duration_min} min · 🎯 +${tk.points} XP · #${tk.tag || tk.parameter_key}</div>
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
        ? `<p style="color:rgba(215,230,255,0.55);font-size:0.88rem;">—</p>`
        : state.doneTasks.slice(0, 12).map((tk) => `
          <div class="task-item" style="opacity:0.78;">
            <div class="task-info">
              <div class="task-title">✅ ${tk.title}</div>
              <div class="task-meta">+${tk.points} XP · ${tk.completed_at ? tk.completed_at.slice(0, 16).replace("T", " ") : ""}</div>
            </div>
          </div>
        `).join("");
    }
  }

  function renderAllScreens() {
    renderHUD();
    renderHome();
    renderOverallHealth();
    renderCategoryInputScreens();
    renderProgressAndRewards();
    redrawVisibleCanvases();
  }

  /* -----------------------------------------------------------------------
     8. API CALLS & INTERACTIVE WORKFLOWS
     ----------------------------------------------------------------------- */
  async function refreshTasksAndGamification() {
    const aid = encodeURIComponent(state.astronautId);
    const [pTasks, dTasks, gam, hist] = await Promise.all([
      fetch(`/astronauts/${aid}/tasks?status=pending`).then((r) => r.json()),
      fetch(`/astronauts/${aid}/tasks?status=done`).then((r) => r.json()),
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
      const body = {
        values: state.values,
        sensor_series: state.sensorSeries,
        subjective_log: state.subjectiveLog || "",
        mission_day: state.missionDay,
        ecg: state.ecgResult,
      };
      const res = await fetch(`/astronauts/${aid}/assess?lang=${apiLang()}`, {
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
        showToast(`✅ Health Risk: ${state.assessment.overall.risk_pct}%` + (nNew ? ` · +${nNew} new mitigation tasks` : ""));
      }
    } catch (err) {
      console.error(err);
      if (!silent) showToast("⚠️ Assessment error: " + err.message);
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
    let msg = `🎉 +${out.points_awarded} XP!`;
    if (out.leveled_up) msg += ` 🚀 LEVEL UP → Lv ${out.state.level} (${out.state.rank})!`;
    if (out.new_badges && out.new_badges.length) {
      msg += " 🏅 " + out.new_badges.map((b) => `${b.icon} ${b.name}`).join(", ");
    }
    showToast(msg);
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
    const res = await fetch(`/presets/scenario/${encodeURIComponent(scenName)}?lang=${apiLang()}`).then((r) => r.json());
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
      body: JSON.stringify({ values: state.values, event: evName, lang: apiLang() }),
    }).then((r) => r.json());
    Object.entries(res.values || {}).forEach(([k, v]) => setParamValue(k, v));
    if (res.log && !state.subjectiveLog.includes(res.log)) {
      state.subjectiveLog = (state.subjectiveLog + " " + res.log).trim();
      document.getElementById("subjLogInput").value = state.subjectiveLog;
    }
    showToast(`⚡ Injected event: ${evName}`);
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
      <div class="briefing-banner" style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
        <div>
          <div style="font-size:1.08rem;font-weight:800;">💓 ${ecg.class_code} — ${ecg.class_name} (${ecg.source})</div>
          <div style="font-size:0.86rem;color:#cfe2ff;margin-top:3px;">
            HR: <b>${(ecg.heart_rate || 0).toFixed(1)} bpm</b> · QTc: <b>${(ecg.qtc_ms || 0).toFixed(0)} ms</b> ·
            Rhythm: <b>${ecg.rr_regularity}</b> · Abnormal prob: <b>${(ecg.abnormal_probability_pct || 0).toFixed(1)}%</b>
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
      <div style="background:rgba(255,255,255,0.05);padding:8px 10px;border-radius:8px;">
        <div style="display:flex;justify-content:space-between;font-size:0.82rem;"><b>${cName}</b><span>${pctVal.toFixed(1)}%</span></div>
        <div class="mini-bar"><span style="width:${Math.min(100, pctVal)}%;background:#4df0a8"></span></div>
      </div>
    `).join("") + `</div>` : "";
  }

  /* -----------------------------------------------------------------------
     9. BOOTSTRAP INITIALIZATION
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

    // Apply initial ISS profile environment values
    const issVals = presets.profiles.iss_leo?.values || {};
    Object.assign(state.values, issVals);

    // Populate HUD dropdowns
    const scenSel = document.getElementById("hudScenario");
    scenSel.innerHTML = state.scenarios.map((s) => `<option value="${s}">🩺 ${s.replace(/_/g, " ")}</option>`).join("");
    scenSel.addEventListener("change", () => applyScenario(scenSel.value));

    const profSel = document.getElementById("hudProfile");
    profSel.innerHTML = Object.entries(state.profiles).map(([k, v]) =>
      `<option value="${k}" ${k === state.profile ? "selected" : ""}>🚀 ${v.en}</option>`
    ).join("");
    profSel.addEventListener("change", () => applyMissionProfile(profSel.value));

    document.getElementById("hudAstronaut").addEventListener("change", (e) => {
      state.astronautId = e.target.value;
      runAssessment(true);
    });
    document.getElementById("hudAssessBtn").addEventListener("click", () => runAssessment());
    document.getElementById("hudHealthPill").addEventListener("click", () => showPage("overall-health"));
    document.getElementById("hudEnvPill").addEventListener("click", () => showPage("space-radiation"));
    document.getElementById("hudXpPill").addEventListener("click", () => showPage("rewards"));

    document.querySelectorAll("[data-run-ai]").forEach((b) =>
      b.addEventListener("click", () => runAssessment())
    );

    // Space Radiation profile & event chips
    const pRow = document.getElementById("profileChipsRow");
    pRow.innerHTML = Object.entries(state.profiles).map(([k, v]) =>
      `<button type="button" class="chip-btn ${k === state.profile ? "active" : ""}" data-prof="${k}">${v.en}</button>`
    ).join("");
    pRow.querySelectorAll("[data-prof]").forEach((b) =>
      b.addEventListener("click", () => applyMissionProfile(b.dataset.prof))
    );

    const eRow = document.getElementById("eventChipsRow");
    eRow.innerHTML = Object.entries(state.events).map(([k, v]) =>
      `<button type="button" class="chip-btn" data-ev="${k}">⚠️ ${v.en}</button>`
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

    // Subjective log & quick mood chips
    const moodRow = document.getElementById("moodQuickChips");
    const moods = [
      "Mild headache and fatigue after shift",
      "Baş ağrısı, yorgunluk ve uykusuzluk hissediyorum",
      "Feeling dizzy upon standing, very thirsty",
      "Shortness of breath and palpitations",
      "Feeling rested, calm and focused today",
    ];
    moodRow.innerHTML = moods.map((m) => `<button type="button" class="chip-btn" data-mood="${m}">💬 ${m.slice(0, 34)}…</button>`).join("");
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
      showToast("📡 24h Wearable Telemetry Updated");
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
        showToast(`📈 ECG Analysed: ${state.ecgResult.class_code}`);
      }
    });

    // AI Advisor Chat
    document.getElementById("geminiStatusBadge").textContent = healthMeta.gemini?.configured ? "Gemini Online" : "Offline Reasoner";
    const chatBox = document.getElementById("chatMessagesBox");
    const addBubble = (who, text) => {
      const div = document.createElement("div");
      div.className = `chat-bubble ${who}`;
      div.textContent = text;
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
    };
    addBubble("ai", "👨‍⚕️ Astrocyte X · AstroGuard-X Flight Surgeon ready. Ask anything about your biomarkers, space radiation, or mitigation tasks.");

    document.getElementById("chatForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const inp = document.getElementById("chatInput");
      const q = inp.value.trim();
      if (!q) return;
      inp.value = "";
      addBubble("user", q);
      const aid = encodeURIComponent(state.astronautId);
      const res = await fetch(`/astronauts/${aid}/chat?lang=${apiLang()}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      }).then((r) => r.json());
      addBubble("ai", `[${res.source}] ${res.answer}`);
    });

    applyLanguage(currentLang);
    showPage(location.hash.slice(1) || "home");
    await loadEcgSample("NORMAL");
    await runAssessment(true);
  }

  init();
})();
