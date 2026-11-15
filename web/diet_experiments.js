/* ==========================================================================
   AstrocyteX — Diet, Stock, Hydration, History & Experiments
   Preserves 100% of interactive mechanics + live model sync (zero emojis).
   ========================================================================== */

(() => {
  const t = (key) => window.agTranslate ? window.agTranslate(key) : key;

  /* ========================================================================
     1. EXPERIMENTS MODULE
     ======================================================================== */
  (() => {
    const STORAGE_KEY = "experiments";
    const SEEDED_KEY = "experiments_seeded_v2";

    const searchInput = document.getElementById("experimentSearch");
    const listEl = document.getElementById("experimentList");
    const addBtn = document.getElementById("addExperimentBtn");
    if (!listEl) return;

    const modal = document.getElementById("experimentModal");
    const form = document.getElementById("experimentForm");
    const modalTitle = document.getElementById("experimentModalTitle");
    const nameInput = document.getElementById("experimentName");
    const variablesInput = document.getElementById("experimentVariables");
    const resultsInput = document.getElementById("experimentResults");
    const detailsInput = document.getElementById("experimentDetails");
    const cancelBtn = document.getElementById("experimentCancel");

    function seedIfFirstVisit() {
      if (localStorage.getItem(SEEDED_KEY)) return;
      const existing = localStorage.getItem(STORAGE_KEY);
      if (!existing) {
        const demo = [
          {
            id: "exp-ared-bone",
            name: "ARED High-Load Resistance vs. CTX Bone Resorption",
            variables: "exercise_min (120 min/d), gravity_g (0.0 g), vitamin_d_ngml, ctx_ngml",
            results: "Daily 120 min ARED + T2 protocol attenuated serum CTX rise by 38% over 60 mission days.",
            details: "Tracks trabecular bone turnover biomarkers alongside Neural ODE +180d BMD projections.",
          },
          {
            id: "exp-co2-cognition",
            name: "Cabin CO₂ Partial Pressure & Circadian Sleep Architecture",
            variables: "cabin_co2_mmhg (2.2 -> 5.4 mmHg), sleep_hours, saliva_cortisol, resp_rate",
            results: "CO₂ > 4.5 mmHg increased nocturnal respiratory rate (+2.4 br/min) and morning headache logs.",
            details: "Evaluates closed-habitat CDRA scrubber efficiency and crew alertness.",
          },
          {
            id: "exp-spe-shielding",
            name: "Storm-Shelter Polyethylene Shielding Dosimetry",
            variables: "radiation_dose_msv_d, radiation_cumulative_msv, wbc_k, platelets_k",
            results: "Water-wall storm shelter reduced acute SPE dose rate by 64% during simulated solar flare.",
            details: "Monitors personal dosimeter telemetry and complete blood count stability.",
          },
        ];
        localStorage.setItem(STORAGE_KEY, JSON.stringify(demo));
      }
      localStorage.setItem(SEEDED_KEY, "1");
    }

    seedIfFirstVisit();
    let experiments = load();
    let editingId = null;

    function load() {
      try {
        const data = JSON.parse(localStorage.getItem(STORAGE_KEY));
        return Array.isArray(data) ? data : [];
      } catch {
        return [];
      }
    }

    function save() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(experiments));
    }

    function makeButton(className, key, onClick) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = className;
      btn.textContent = t(key);
      btn.addEventListener("click", onClick);
      return btn;
    }

    function render() {
      const query = searchInput.value.trim().toLowerCase();
      const shown = experiments.filter((exp) =>
        exp.name.toLowerCase().includes(query) ||
        (exp.variables && exp.variables.toLowerCase().includes(query))
      );

      listEl.replaceChildren();
      if (shown.length === 0) {
        const empty = document.createElement("li");
        empty.className = "experiment-empty";
        empty.textContent = experiments.length === 0 ? t("noExperiments") : t("noResults");
        listEl.appendChild(empty);
        return;
      }

      shown.forEach((exp) => {
        const li = document.createElement("li");
        li.className = "experiment-item";

        const info = document.createElement("div");
        info.className = "experiment-info";

        const name = document.createElement("div");
        name.className = "experiment-name";
        name.textContent = exp.name;
        info.appendChild(name);

        if (exp.variables) {
          const vars = document.createElement("div");
          vars.className = "experiment-meta";
          vars.textContent = `${t("variables")}: ${exp.variables}`;
          info.appendChild(vars);
        }
        if (exp.results) {
          const results = document.createElement("div");
          results.className = "experiment-meta experiment-details";
          results.textContent = `${t("results")}: ${exp.results}`;
          info.appendChild(results);
        }
        if (exp.details) {
          const details = document.createElement("div");
          details.className = "experiment-meta experiment-details";
          details.textContent = exp.details;
          info.appendChild(details);
        }

        const actions = document.createElement("div");
        actions.className = "experiment-actions";
        actions.append(
          makeButton("btn small", "editExperiment", () => openModal(exp.id)),
          makeButton("btn small danger", "deleteExperiment", () => {
            experiments = experiments.filter((e) => e.id !== exp.id);
            save();
            render();
          })
        );

        li.append(info, actions);
        listEl.appendChild(li);
      });
    }

    function openModal(id = null) {
      editingId = id;
      const exp = experiments.find((e) => e.id === id);
      modalTitle.dataset.i18n = exp ? "editExperiment" : "addExperiment";
      modalTitle.textContent = t(modalTitle.dataset.i18n);

      nameInput.value = exp ? exp.name : "";
      variablesInput.value = exp ? exp.variables : "";
      resultsInput.value = exp ? exp.results || "" : "";
      detailsInput.value = exp ? exp.details : "";
      nameInput.classList.remove("invalid");

      modal.hidden = false;
      nameInput.focus();
    }

    function closeModal() {
      modal.hidden = true;
      editingId = null;
    }

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const name = nameInput.value.trim();
      if (!name) {
        nameInput.classList.add("invalid");
        nameInput.focus();
        return;
      }
      const data = {
        name,
        variables: variablesInput.value.trim(),
        results: resultsInput.value.trim(),
        details: detailsInput.value.trim(),
      };
      const existing = experiments.find((exp) => exp.id === editingId);
      if (existing) {
        Object.assign(existing, data);
      } else {
        experiments.unshift({ id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7), ...data });
      }
      save();
      closeModal();
      render();
    });

    nameInput.addEventListener("input", () => nameInput.classList.remove("invalid"));
    cancelBtn.addEventListener("click", closeModal);
    addBtn.addEventListener("click", () => openModal());
    modal.addEventListener("mousedown", (e) => { if (e.target === modal) closeModal(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !modal.hidden) closeModal(); });
    searchInput.addEventListener("input", render);
    document.addEventListener("app-language-changed", render);
    render();
  })();

  /* ========================================================================
     2. DIET, FOOD STOCK, HYDRATION & AI BRIDGE MODULE
     ======================================================================== */
  (() => {
    const STORAGE_KEY = "foodItems";
    const MEALS_STORAGE_KEY = "mealPlans";
    const HYDRATION_STORAGE_KEY = "hydrationItems";
    const SEEDED_FOOD_KEY = "food_seeded_v2";

    const switcher = document.getElementById("dietSwitcher");
    if (!switcher) return;

    function roundAmount(n) {
      return Math.round((Number(n) + Number.EPSILON) * 100) / 100;
    }

    function parseAmountInput(str) {
      const trimmed = String(str || "").trim();
      const match = trimmed.match(/^(\d+(?:[.,]\d+)?)\s*(.*)$/);
      if (!match) return null;
      const value = parseFloat(match[1].replace(",", "."));
      if (!Number.isFinite(value) || value < 0) return null;
      return { value: roundAmount(value), unit: match[2].trim() };
    }

    function formatAmount(value, unit) {
      const num = Number.isFinite(Number(value)) ? roundAmount(Math.max(0, Number(value))) : 0;
      return unit ? `${num} ${unit}` : `${num}`;
    }

    function formatScaledNumber(num, useComma) {
      const safe = Math.max(0, Number(num) || 0);
      let rounded = Math.round((safe + Number.EPSILON) * 100) / 100;
      if (safe > 0 && rounded === 0) {
        rounded = Math.round((safe + Number.EPSILON) * 10000) / 10000;
      }
      const str = String(rounded);
      return useComma ? str.replace(".", ",") : str;
    }

    function scaleNutritionText(text, ratio) {
      const raw = String(text || "").trim();
      if (!raw) return "";
      const r = Number.isFinite(Number(ratio)) ? Math.max(0, Number(ratio)) : 0;
      return raw.replace(
        /(\p{L}+-\d+(?:[.,]\d+)?|\p{L}+\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)/gu,
        (full, identifierToken, numberToken) => {
          if (identifierToken) return identifierToken;
          const useComma = numberToken.includes(",");
          const val = parseFloat(numberToken.replace(",", "."));
          if (!Number.isFinite(val)) return numberToken;
          return formatScaledNumber(val * r, useComma);
        }
      );
    }

    function getItemAvailable(item) {
      return item && Number.isFinite(Number(item.amountValue)) ? roundAmount(Math.max(0, Number(item.amountValue))) : 0;
    }
    function getItemBaseAmount(item) {
      if (!item) return 1;
      const base = Number(item.baseAmountValue);
      if (Number.isFinite(base) && base > 0) return base;
      const avail = getItemAvailable(item);
      return avail > 0 ? avail : 1;
    }
    function getItemBaseNutrition(item) {
      if (!item) return "";
      return item.baseNutrition !== undefined ? item.baseNutrition : item.nutrition || "";
    }
    function getNutritionForAmount(item, portionAmount) {
      const baseNut = getItemBaseNutrition(item);
      if (!baseNut) return "";
      const baseAmt = getItemBaseAmount(item);
      const ratio = baseAmt > 0 ? Math.max(0, Number(portionAmount) || 0) / baseAmt : 0;
      return scaleNutritionText(baseNut, ratio);
    }
    function normalizeFoodItem(item) {
      const avail = getItemAvailable(item);
      const baseAmt = Number.isFinite(Number(item.baseAmountValue)) && Number(item.baseAmountValue) > 0
        ? Number(item.baseAmountValue) : (avail > 0 ? avail : 1);
      const baseNut = item.baseNutrition !== undefined ? item.baseNutrition : item.nutrition || "";
      return { ...item, amountValue: avail, amountUnit: item.amountUnit || "", baseAmountValue: baseAmt, baseNutrition: baseNut };
    }

    // Seed NASA ISS pantry on very first run so the stock colors & meal planner shine immediately
    if (!localStorage.getItem(SEEDED_FOOD_KEY)) {
      if (!localStorage.getItem(STORAGE_KEY)) {
        const pantry = [
          { id: "fd-1", name: "Thermostabilized Wild Salmon", code: "FD-001", nutrition: "920 kcal, 88 g protein, Omega-3 4.2 g", baseNutrition: "920 kcal, 88 g protein, Omega-3 4.2 g", amountValue: 400, baseAmountValue: 400, amountUnit: "g", consumedAmount: 0 },
          { id: "fd-2", name: "Freeze-Dried Quinoa & Chicken", code: "FD-002", nutrition: "1450 kcal, 95 g protein, 180 g carbs", baseNutrition: "1450 kcal, 95 g protein, 180 g carbs", amountValue: 600, baseAmountValue: 800, amountUnit: "g", consumedAmount: 200 },
          { id: "fd-3", name: "ISS Flour Tortilla Pack", code: "FD-003", nutrition: "1100 kcal, 30 g protein, 190 g carbs", baseNutrition: "1100 kcal, 30 g protein, 190 g carbs", amountValue: 4, baseAmountValue: 10, amountUnit: "pcs", consumedAmount: 6 },
          { id: "fd-4", name: "Rehydratable Berry Yogurt", code: "FD-004", nutrition: "480 kcal, 24 g protein, Vitamin D3 15 mcg", baseNutrition: "480 kcal, 24 g protein, Vitamin D3 15 mcg", amountValue: 90, baseAmountValue: 450, amountUnit: "g", consumedAmount: 360 },
          { id: "fd-5", name: "Space Peanut & Cocoa Energy Bar", code: "FD-005", nutrition: "780 kcal, 28 g protein, Iron 6 mg", baseNutrition: "780 kcal, 28 g protein, Iron 6 mg", amountValue: 300, baseAmountValue: 300, amountUnit: "g", consumedAmount: 0 },
        ];
        localStorage.setItem(STORAGE_KEY, JSON.stringify(pantry));
      }
      if (!localStorage.getItem(HYDRATION_STORAGE_KEY)) {
        localStorage.setItem(HYDRATION_STORAGE_KEY, JSON.stringify([
          { id: "hyd-1", volume: "1200 ml", supplements: "Electrolytes (Na/K), Vitamin D3" },
          { id: "hyd-2", volume: "1300 ml", supplements: "Potable ISS Recycled Water" },
        ]));
      }
      localStorage.setItem(SEEDED_FOOD_KEY, "1");
    }

    const switchButton = document.getElementById("dietSwitchButton");
    const switchLabel = document.getElementById("dietSwitchLabel");
    const options = switcher.querySelectorAll(".diet-switch-list button");
    const views = {
      food: { view: document.getElementById("dietFoodView"), labelKey: "foodItems" },
      hydration: { view: document.getElementById("dietHydrationView"), labelKey: "hydration" },
    };

    let pinnedOpen = false;
    function setDropdownOpen(open) {
      switcher.classList.toggle("open", open);
      switchButton.setAttribute("aria-expanded", String(open));
      if (!open) pinnedOpen = false;
    }
    function showView(name) {
      Object.entries(views).forEach(([key, { view }]) => view.classList.toggle("active", key === name));
      switchLabel.dataset.i18n = views[name].labelKey;
      switchLabel.textContent = t(views[name].labelKey);
      options.forEach((btn) => {
        const isCurrent = btn.dataset.view === name;
        btn.classList.toggle("current", isCurrent);
        btn.parentElement.hidden = isCurrent;
      });
    }

    switcher.addEventListener("mouseenter", () => setDropdownOpen(true));
    switcher.addEventListener("mouseleave", () => { if (!pinnedOpen) setDropdownOpen(false); });
    switchButton.addEventListener("click", (e) => {
      e.stopPropagation();
      if (pinnedOpen) setDropdownOpen(false);
      else { setDropdownOpen(true); pinnedOpen = true; }
    });
    options.forEach((btn) =>
      btn.addEventListener("click", () => {
        showView(btn.dataset.view);
        if (btn.dataset.view === "hydration") showHydrationSubpanel("list");
        setDropdownOpen(false);
      })
    );
    document.addEventListener("click", (e) => { if (!switcher.contains(e.target)) setDropdownOpen(false); });

    const searchInput = document.getElementById("foodSearch");
    const addFoodBtn = document.getElementById("addFoodBtn");
    const sidebarListEl = document.getElementById("foodSidebarList");
    const mealsPanel = document.getElementById("foodMealsPanel");
    const detailPanel = document.getElementById("foodDetailPanel");
    const detailContent = document.getElementById("foodDetailContent");
    const formPanel = document.getElementById("foodFormPanel");
    const mealTabs = document.querySelectorAll("#mealCategories .meal-tab");
    const mealCategoryPrompt = document.getElementById("mealCategoryPrompt");
    const mealCategoryBody = document.getElementById("mealCategoryBody");
    const mealAddForm = document.getElementById("mealAddForm");
    const mealFoodInput = document.getElementById("mealFoodInput");
    const mealSuggestions = document.getElementById("mealSuggestions");
    const mealFoodError = document.getElementById("mealFoodError");
    const mealFoodList = document.getElementById("mealFoodList");

    const mealAmountBox = document.getElementById("mealAmountBox");
    const mealPendingName = document.getElementById("mealPendingName");
    const mealPendingCode = document.getElementById("mealPendingCode");
    const mealAvailableText = document.getElementById("mealAvailableText");
    const mealOutOfStockMsg = document.getElementById("mealOutOfStockMsg");
    const mealAmountControls = document.getElementById("mealAmountControls");
    const mealAmountReadout = document.getElementById("mealAmountReadout");
    const mealAmountSlider = document.getElementById("mealAmountSlider");
    const mealAmountNumber = document.getElementById("mealAmountNumber");
    const mealAmountUnit = document.getElementById("mealAmountUnit");
    const mealScaleZero = document.getElementById("mealScaleZero");
    const mealScaleMax = document.getElementById("mealScaleMax");
    const mealAmountNutritionPreview = document.getElementById("mealAmountNutritionPreview");
    const mealAmountCancelBtn = document.getElementById("mealAmountCancelBtn");
    const mealAmountConfirmBtn = document.getElementById("mealAmountConfirmBtn");

    const formTitle = document.getElementById("foodFormTitle");
    const nameInput = document.getElementById("foodNameInput");
    const codeInput = document.getElementById("foodCodeInput");
    const nutritionInput = document.getElementById("foodNutritionInput");
    const amountInput = document.getElementById("foodAmountInput");
    const cancelBtn = document.getElementById("foodCancelBtn");

    let foodItems = loadFoodItems();
    let mealPlans = loadMealPlans();
    let hydrationItems = loadHydrationItems();
    let selectedId = null;
    let editingId = null;
    let activeMeal = "breakfast";
    let pendingFoodId = null;
    let pendingSelectedAmount = 0;

    function loadFoodItems() {
      try {
        const data = JSON.parse(localStorage.getItem(STORAGE_KEY));
        return Array.isArray(data) ? data.map(normalizeFoodItem) : [];
      } catch { return []; }
    }
    function saveFoodItems() { localStorage.setItem(STORAGE_KEY, JSON.stringify(foodItems)); }
    function loadMealPlans() {
      const empty = { breakfast: [], lunch: [], dinner: [], snack: [] };
      try {
        const data = JSON.parse(localStorage.getItem(MEALS_STORAGE_KEY));
        if (!data || typeof data !== "object") return empty;
        return {
          breakfast: Array.isArray(data.breakfast) ? data.breakfast : [],
          lunch: Array.isArray(data.lunch) ? data.lunch : [],
          dinner: Array.isArray(data.dinner) ? data.dinner : [],
          snack: Array.isArray(data.snack) ? data.snack : [],
        };
      } catch { return empty; }
    }
    function saveMealPlans() {
      localStorage.setItem(MEALS_STORAGE_KEY, JSON.stringify(mealPlans));
      updateDietAIBar();
    }
    function loadHydrationItems() {
      try {
        const data = JSON.parse(localStorage.getItem(HYDRATION_STORAGE_KEY));
        return Array.isArray(data) ? data : [];
      } catch { return []; }
    }
    function saveHydrationItems() {
      localStorage.setItem(HYDRATION_STORAGE_KEY, JSON.stringify(hydrationItems));
      updateDietAIBar();
    }

    /* ---- Live AI Macro Parser (kcal, protein_g, fluid_intake_ml) ---- */
    function computeDailyTotals() {
      let kcal = 0;
      let protein = 0;
      for (const key of ["breakfast", "lunch", "dinner", "snack"]) {
        for (const entry of mealPlans[key] || []) {
          const linked = entry.foodId ? foodItems.find((f) => f.id === entry.foodId) : null;
          const nut = linked && Number.isFinite(Number(entry.amountValue))
            ? getNutritionForAmount(linked, entry.amountValue)
            : (entry.nutrition || "");
          const km = nut.match(/(\d+(?:[.,]\d+)?)\s*(?:kcal|cal)/i);
          if (km) kcal += parseFloat(km[1].replace(",", "."));
          const pm = nut.match(/(\d+(?:[.,]\d+)?)\s*g\s*(?:protein|prot)/i) || nut.match(/(?:protein|prot)[:\s]*(\d+(?:[.,]\d+)?)/i);
          if (pm) protein += parseFloat(pm[1].replace(",", "."));
        }
      }
      let fluidMl = 0;
      for (const h of hydrationItems) {
        const raw = String(h.volume || "").trim();
        const m = raw.match(/(\d+(?:[.,]\d+)?)\s*(ml|mL|l|L)?/);
        if (m) {
          const v = parseFloat(m[1].replace(",", "."));
          const u = (m[2] || "ml").toLowerCase();
          fluidMl += u === "l" ? v * 1000 : v;
        }
      }
      return {
        calories_kcal: Math.round(kcal),
        protein_g: Math.round(protein * 10) / 10,
        fluid_intake_ml: Math.round(fluidMl),
      };
    }

    function updateDietAIBar() {
      const totals = computeDailyTotals();
      const summaryEl = document.getElementById("dietMacroSummary");
      if (!summaryEl) return;
      const dotHtml = (key, val) => window.agDotHtml ? window.agDotHtml(key, val) : "";
      const calUsed = totals.calories_kcal > 0 ? totals.calories_kcal : (window.agGetVal ? window.agGetVal("calories_kcal") : 2800);
      const proUsed = totals.protein_g > 0 ? totals.protein_g : (window.agGetVal ? window.agGetVal("protein_g") : 100);
      const flUsed = totals.fluid_intake_ml > 0 ? totals.fluid_intake_ml : (window.agGetVal ? window.agGetVal("fluid_intake_ml") : 2500);
      summaryEl.innerHTML = `
        <span class="status-chip">${dotHtml("calories_kcal", calUsed)} Energy <b>${calUsed}</b> kcal/d</span>
        <span class="status-chip">${dotHtml("protein_g", proUsed)} Protein <b>${proUsed}</b> g</span>
        <span class="status-chip">${dotHtml("fluid_intake_ml", flUsed)} Fluid <b>${flUsed}</b> mL</span>
      `;
    }

    window.AstroDietBridge = { computeDailyTotals, updateDietAIBar };

    const syncBtn = document.getElementById("syncDietToAIBtn");
    if (syncBtn) {
      syncBtn.addEventListener("click", () => {
        const totals = computeDailyTotals();
        if (window.agSetDietTotals) {
          window.agSetDietTotals(totals);
        }
        updateDietAIBar();
      });
    }

    function getMealUsedAmount(foodId) {
      let used = 0;
      for (const key of ["breakfast", "lunch", "dinner", "snack"]) {
        for (const entry of mealPlans[key] || []) {
          if (entry.foodId === foodId && Number.isFinite(Number(entry.amountValue))) used += Number(entry.amountValue);
        }
      }
      return roundAmount(used);
    }

    function getItemStockState(item) {
      const avail = getItemAvailable(item);
      const used = getMealUsedAmount(item.id);
      const consumed = roundAmount(Math.max(0, Number(item.consumedAmount) || 0));
      const initial = roundAmount(avail + used + consumed);
      if (avail <= 0) return { avail, initial, pct: 0, statusClass: "stock-none" };
      const pct = initial > 0 ? (avail / initial) * 100 : 100;
      const statusClass = pct > 75 ? "stock-green" : pct >= 50 ? "stock-yellow" : pct >= 25 ? "stock-orange" : "stock-red";
      return { avail, initial, pct: Math.round(pct), statusClass };
    }

    function makeId() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }
    function generateNextCode() {
      let n = foodItems.length + 1;
      let candidate = `FD-${String(n).padStart(3, "0")}`;
      const existingCodes = new Set(foodItems.map((f) => (f.code || "").toUpperCase()));
      while (existingCodes.has(candidate)) { n += 1; candidate = `FD-${String(n).padStart(3, "0")}`; }
      return candidate;
    }
    function makeActionBtn(className, key, onClick) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = className;
      btn.textContent = t(key);
      btn.addEventListener("click", onClick);
      return btn;
    }

    function showFoodSubpanel(mode, id = null) {
      mealsPanel.classList.toggle("active", mode === "meals");
      detailPanel.classList.toggle("active", mode === "detail");
      formPanel.classList.toggle("active", mode === "form");

      if (mode === "form") {
        editingId = id;
        const existing = foodItems.find((item) => item.id === id);
        formTitle.dataset.i18n = existing ? "editFoodItem" : "addFoodItem";
        formTitle.textContent = t(formTitle.dataset.i18n);
        if (existing) {
          const avail = getItemAvailable(existing);
          const refAmt = avail > 0 ? avail : getItemBaseAmount(existing);
          nameInput.value = existing.name;
          codeInput.value = existing.code;
          nutritionInput.value = avail > 0 ? getNutritionForAmount(existing, avail) : getItemBaseNutrition(existing);
          amountInput.value = formatAmount(refAmt, existing.amountUnit || "");
        } else {
          nameInput.value = ""; codeInput.value = ""; nutritionInput.value = ""; amountInput.value = "";
        }
        nameInput.classList.remove("invalid");
        amountInput.classList.remove("invalid");
      } else {
        editingId = null;
        if (mode === "detail") renderDetail();
        else { selectedId = null; renderMealCategory(); }
      }
      renderSidebarList();
    }

    function renderSidebarList() {
      const q = searchInput.value.trim().toLowerCase();
      const shown = !q ? foodItems : foodItems.filter((item) =>
        item.name.toLowerCase().includes(q) || (item.code && item.code.toLowerCase().includes(q))
      );
      sidebarListEl.replaceChildren();
      if (shown.length === 0) {
        const empty = document.createElement("li");
        empty.className = "food-sidebar-empty";
        empty.textContent = foodItems.length === 0 ? t("noFoodItems") : t("noFoodResults");
        sidebarListEl.appendChild(empty);
        return;
      }
      const isViewingDetail = detailPanel.classList.contains("active");
      shown.forEach((item) => {
        const { avail, pct, statusClass } = getItemStockState(item);
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `food-sidebar-item ${statusClass}` + (isViewingDetail && item.id === selectedId ? " selected" : "");

        const leftWrap = document.createElement("div");
        leftWrap.className = "food-sidebar-left";
        const nameRow = document.createElement("div");
        nameRow.className = "food-sidebar-name-row";
        const dot = document.createElement("span");
        dot.className = "food-stock-dot";
        const nameSpan = document.createElement("span");
        nameSpan.className = "food-sidebar-name";
        nameSpan.textContent = item.name;
        nameRow.append(dot, nameSpan);
        const stockSpan = document.createElement("span");
        stockSpan.className = "food-sidebar-amount";
        stockSpan.textContent = `${t("amount")}: ${formatAmount(avail, item.amountUnit || "")} (${pct}%)`;
        leftWrap.append(nameRow, stockSpan);
        btn.appendChild(leftWrap);

        if (item.code) {
          const codeSpan = document.createElement("span");
          codeSpan.className = "food-code-badge";
          codeSpan.textContent = item.code;
          btn.appendChild(codeSpan);
        }
        btn.addEventListener("click", () => {
          if (detailPanel.classList.contains("active") && selectedId === item.id) {
            selectedId = null; showFoodSubpanel("meals");
          } else {
            selectedId = item.id; showFoodSubpanel("detail");
          }
        });
        li.appendChild(btn);
        sidebarListEl.appendChild(li);
      });
    }

    function renderDetail() {
      detailContent.replaceChildren();
      const selected = foodItems.find((item) => item.id === selectedId);
      if (!selected) { showFoodSubpanel("meals"); return; }
      const avail = getItemAvailable(selected);

      const header = document.createElement("div");
      header.className = "food-detail-header";
      const titleWrap = document.createElement("div");
      titleWrap.className = "food-item-header";
      const title = document.createElement("h3");
      title.className = "food-detail-title";
      title.textContent = selected.name;
      titleWrap.appendChild(title);
      if (selected.code) {
        const badge = document.createElement("span");
        badge.className = "food-code-badge";
        badge.textContent = selected.code;
        titleWrap.appendChild(badge);
      }
      const actions = document.createElement("div");
      actions.className = "food-detail-actions";
      actions.append(
        makeActionBtn("btn small", "backToMeals", () => { selectedId = null; showFoodSubpanel("meals"); }),
        makeActionBtn("btn small", "editFoodItem", () => showFoodSubpanel("form", selected.id)),
        makeActionBtn("btn small danger", "deleteFoodItem", () => {
          foodItems = foodItems.filter((i) => i.id !== selected.id);
          selectedId = null; saveFoodItems(); showFoodSubpanel("meals");
        })
      );
      header.append(titleWrap, actions);

      const nutSec = document.createElement("div");
      nutSec.className = "food-detail-section";
      nutSec.innerHTML = `<h4 class="food-detail-label">${t("nutritionalValues")}</h4>`;
      const nutBody = document.createElement("div");
      nutBody.className = "food-detail-body";
      nutBody.textContent = getNutritionForAmount(selected, avail) || t("noNutritionalValues");
      nutSec.appendChild(nutBody);

      const amtSec = document.createElement("div");
      amtSec.className = "food-detail-section";
      amtSec.innerHTML = `<h4 class="food-detail-label">${t("amount")}</h4>`;
      const amtBody = document.createElement("div");
      amtBody.className = "food-detail-body";
      amtBody.textContent = formatAmount(avail, selected.amountUnit || "");
      amtSec.appendChild(amtBody);

      detailContent.append(header, nutSec, amtSec);
    }

    function hideMealError() { mealFoodError.hidden = true; mealFoodError.textContent = ""; }
    function showMealError(key) { mealFoodError.dataset.i18n = key; mealFoodError.textContent = t(key); mealFoodError.hidden = false; }
    function hideMealSuggestions() { mealSuggestions.hidden = true; mealSuggestions.replaceChildren(); }
    function updateSliderVisual(pct) { mealAmountSlider.style.setProperty("--fill", `${Math.max(0, Math.min(100, Number(pct) || 0))}%`); }

    function syncAmountReadout(foodObj, avail, unit) {
      const pct = avail > 0 ? Math.round((pendingSelectedAmount / avail) * 100) : 0;
      mealAmountReadout.textContent = `${formatAmount(pendingSelectedAmount, unit)} (${pct}%)`;
      mealAmountConfirmBtn.disabled = !(pendingSelectedAmount > 0 && pendingSelectedAmount <= avail);
      const previewText = getNutritionForAmount(foodObj, pendingSelectedAmount);
      if (previewText) {
        mealAmountNutritionPreview.textContent = `${t("nutritionalValues")}: ${previewText}`;
        mealAmountNutritionPreview.hidden = false;
      } else {
        mealAmountNutritionPreview.hidden = true;
      }
    }

    function openMealAmountPicker(foodObj) {
      if (!foodObj) return;
      pendingFoodId = foodObj.id;
      hideMealSuggestions();
      hideMealError();
      const avail = getItemAvailable(foodObj);
      const unit = foodObj.amountUnit || "";
      mealPendingName.textContent = foodObj.name;
      mealPendingCode.textContent = foodObj.code || "";
      mealPendingCode.hidden = !foodObj.code;
      mealAvailableText.textContent = `${t("availableStock")}: ${formatAmount(avail, unit)}`;

      if (avail <= 0) {
        mealOutOfStockMsg.hidden = false;
        mealAmountControls.hidden = true;
        mealAmountConfirmBtn.disabled = true;
        pendingSelectedAmount = 0;
      } else {
        mealOutOfStockMsg.hidden = true;
        mealAmountControls.hidden = false;
        pendingSelectedAmount = roundAmount(0.5 * avail) || avail;
        const actualPct = Math.round((pendingSelectedAmount / avail) * 100);
        mealAmountSlider.value = String(actualPct);
        updateSliderVisual(actualPct);
        mealAmountNumber.max = String(avail);
        mealAmountNumber.value = String(pendingSelectedAmount);
        mealAmountUnit.textContent = unit;
        mealScaleZero.textContent = `0% (${formatAmount(0, unit)})`;
        mealScaleMax.textContent = `100% (${formatAmount(avail, unit)})`;
        syncAmountReadout(foodObj, avail, unit);
      }
      mealAmountBox.hidden = false;
    }

    function closeMealAmountPicker() { pendingFoodId = null; pendingSelectedAmount = 0; mealAmountBox.hidden = true; }

    mealAmountSlider.addEventListener("input", () => {
      const foodObj = foodItems.find((f) => f.id === pendingFoodId);
      if (!foodObj) return;
      const avail = getItemAvailable(foodObj);
      const pct = Number(mealAmountSlider.value);
      pendingSelectedAmount = roundAmount((pct / 100) * avail);
      mealAmountNumber.value = String(pendingSelectedAmount);
      updateSliderVisual(pct);
      syncAmountReadout(foodObj, avail, foodObj.amountUnit || "");
    });

    mealAmountNumber.addEventListener("input", () => {
      const foodObj = foodItems.find((f) => f.id === pendingFoodId);
      if (!foodObj) return;
      const avail = getItemAvailable(foodObj);
      let val = parseFloat(String(mealAmountNumber.value).replace(",", "."));
      if (!Number.isFinite(val) || val < 0) val = 0;
      if (val > avail) { val = avail; mealAmountNumber.value = String(avail); }
      pendingSelectedAmount = roundAmount(val);
      const pct = avail > 0 ? (pendingSelectedAmount / avail) * 100 : 0;
      mealAmountSlider.value = String(Math.round(pct));
      updateSliderVisual(pct);
      syncAmountReadout(foodObj, avail, foodObj.amountUnit || "");
    });

    mealAmountCancelBtn.addEventListener("click", closeMealAmountPicker);
    mealAmountConfirmBtn.addEventListener("click", () => {
      if (!activeMeal || !pendingFoodId) return;
      const foodObj = foodItems.find((f) => f.id === pendingFoodId);
      if (!foodObj) { closeMealAmountPicker(); return; }
      const avail = getItemAvailable(foodObj);
      if (!(pendingSelectedAmount > 0 && pendingSelectedAmount <= avail)) return;
      const portionNutrition = getNutritionForAmount(foodObj, pendingSelectedAmount);
      foodObj.amountValue = roundAmount(Math.max(0, avail - pendingSelectedAmount));
      saveFoodItems();
      mealPlans[activeMeal].push({
        entryId: makeId(), foodId: foodObj.id, name: foodObj.name, code: foodObj.code || "",
        nutrition: portionNutrition, amountValue: pendingSelectedAmount, amountUnit: foodObj.amountUnit || "",
      });
      saveMealPlans();
      mealFoodInput.value = "";
      closeMealAmountPicker();
      renderSidebarList();
      renderMealList();
    });

    function selectMealCategory(mealKey) {
      activeMeal = mealKey;
      mealTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.meal === mealKey));
      mealCategoryPrompt.hidden = Boolean(mealKey);
      mealCategoryBody.hidden = !mealKey;
      hideMealSuggestions(); hideMealError(); closeMealAmountPicker();
      if (mealKey) { mealFoodInput.value = ""; renderMealList(); }
    }

    function renderMealCategory() {
      mealTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.meal === activeMeal));
      mealCategoryPrompt.hidden = Boolean(activeMeal);
      mealCategoryBody.hidden = !activeMeal;
      if (activeMeal) renderMealList();
    }

    function renderMealList() {
      mealFoodList.replaceChildren();
      if (!activeMeal) return;
      const entries = mealPlans[activeMeal] || [];
      if (entries.length === 0) {
        const empty = document.createElement("li");
        empty.className = "meal-empty";
        empty.textContent = t("noMealItems");
        mealFoodList.appendChild(empty);
        return;
      }
      entries.forEach((entry) => {
        const linked = entry.foodId ? foodItems.find((f) => f.id === entry.foodId) : null;
        const displayName = linked ? linked.name : entry.name;
        const displayCode = linked ? linked.code : entry.code;
        const displayUnit = entry.amountUnit !== undefined ? entry.amountUnit : (linked ? linked.amountUnit || "" : "");
        const displayNutrition = linked && Number.isFinite(Number(entry.amountValue))
          ? getNutritionForAmount(linked, entry.amountValue) : (entry.nutrition || "");

        const li = document.createElement("li");
        li.className = "meal-food-item";
        const info = document.createElement("div");
        info.className = "food-info";
        const header = document.createElement("div");
        header.className = "food-item-header";
        const nameEl = document.createElement("span");
        nameEl.className = "food-name";
        nameEl.textContent = displayName;
        header.appendChild(nameEl);
        if (displayCode) {
          const badge = document.createElement("span");
          badge.className = "food-code-badge";
          badge.textContent = displayCode;
          header.appendChild(badge);
        }
        if (Number.isFinite(Number(entry.amountValue))) {
          const amtBadge = document.createElement("span");
          amtBadge.className = "meal-used-amount-badge";
          amtBadge.textContent = `${t("amount")}: ${formatAmount(entry.amountValue, displayUnit)}`;
          header.appendChild(amtBadge);
        }
        info.appendChild(header);
        if (displayNutrition) {
          const nutEl = document.createElement("div");
          nutEl.className = "food-nutrition";
          nutEl.textContent = `${t("nutritionalValues")}: ${displayNutrition}`;
          info.appendChild(nutEl);
        }
        const actions = document.createElement("div");
        actions.appendChild(makeActionBtn("btn small danger", "removeFood", () => {
          if (entry.foodId && Number.isFinite(Number(entry.amountValue))) {
            const lk = foodItems.find((f) => f.id === entry.foodId);
            if (lk) { lk.amountValue = roundAmount(getItemAvailable(lk) + Number(entry.amountValue)); saveFoodItems(); }
          }
          mealPlans[activeMeal] = mealPlans[activeMeal].filter((e) => e.entryId !== entry.entryId);
          saveMealPlans(); renderSidebarList(); renderMealList();
        }));
        li.append(info, actions);
        mealFoodList.appendChild(li);
      });
    }

    mealTabs.forEach((tab) => tab.addEventListener("click", () => selectMealCategory(tab.dataset.meal)));
    mealFoodInput.addEventListener("input", () => {
      hideMealError();
      const q = mealFoodInput.value.trim().toLowerCase();
      if (!q || !foodItems.length) { hideMealSuggestions(); return; }
      const matches = foodItems.filter((f) => f.name.toLowerCase().includes(q) || (f.code && f.code.toLowerCase().includes(q)));
      if (!matches.length) { hideMealSuggestions(); return; }
      mealSuggestions.replaceChildren();
      matches.slice(0, 6).forEach((item) => {
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "meal-suggestion-btn";
        btn.innerHTML = `<span>${item.name} (${formatAmount(getItemAvailable(item), item.amountUnit || "")})</span>` +
          (item.code ? `<span class="food-code-badge">${item.code}</span>` : "");
        btn.addEventListener("click", () => { mealFoodInput.value = item.name; openMealAmountPicker(item); });
        li.appendChild(btn);
        mealSuggestions.appendChild(li);
      });
      mealSuggestions.hidden = false;
    });

    mealAddForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const raw = mealFoodInput.value.trim().toLowerCase();
      if (!raw) return;
      const matched = foodItems.find((f) => (f.code && f.code.toLowerCase() === raw) || f.name.toLowerCase() === raw)
        || foodItems.find((f) => (f.code && f.code.toLowerCase().includes(raw)) || f.name.toLowerCase().includes(raw));
      if (!matched) { showMealError("foodNotFound"); return; }
      openMealAmountPicker(matched);
    });

    searchInput.addEventListener("input", renderSidebarList);
    addFoodBtn.addEventListener("click", () => showFoodSubpanel("form"));
    cancelBtn.addEventListener("click", () => showFoodSubpanel(selectedId ? "detail" : "meals"));

    formPanel.addEventListener("submit", (e) => {
      e.preventDefault();
      const name = nameInput.value.trim();
      if (!name) { nameInput.classList.add("invalid"); return; }
      const parsedAmount = parseAmountInput(amountInput.value);
      if (!parsedAmount || (!editingId && parsedAmount.value <= 0)) { amountInput.classList.add("invalid"); return; }
      const rawCode = codeInput.value.trim();
      const nutrition = nutritionInput.value.trim();
      const existing = foodItems.find((item) => item.id === editingId);
      if (existing) {
        existing.name = name;
        existing.code = rawCode || existing.code || generateNextCode();
        if (parsedAmount.value !== getItemAvailable(existing)) existing.consumedAmount = 0;
        existing.amountValue = parsedAmount.value;
        existing.amountUnit = parsedAmount.unit;
        if (parsedAmount.value > 0) { existing.baseAmountValue = parsedAmount.value; existing.baseNutrition = nutrition; existing.nutrition = nutrition; }
        selectedId = existing.id; saveFoodItems(); showFoodSubpanel("detail");
      } else {
        foodItems.push({
          id: makeId(), name, code: rawCode || generateNextCode(), nutrition, baseNutrition: nutrition,
          amountValue: parsedAmount.value, baseAmountValue: parsedAmount.value, amountUnit: parsedAmount.unit,
        });
        saveFoodItems(); selectedId = null; showFoodSubpanel("meals");
      }
    });

    /* ---- Date & Save to Diet History ---- */
    const MEAL_KEYS = ["breakfast", "lunch", "dinner", "snack"];
    const HISTORY_STORAGE_KEY = "dietHistory";
    const DATE_STORAGE_KEY = "dietSelectedDate";
    const dietSaveBtn = document.getElementById("dietSaveBtn");
    const dietSaveStatus = document.getElementById("dietSaveStatus");
    const dietDateBtn = document.getElementById("dietDateBtn");
    const dietDateInput = document.getElementById("dietDateInput");
    const dietDateValue = document.getElementById("dietDateValue");

    const todayISO = () => new Date().toISOString().slice(0, 10);
    let selectedDate = localStorage.getItem(DATE_STORAGE_KEY) || todayISO();
    const renderDateButton = () => { dietDateValue.textContent = `: ${selectedDate}`; dietDateInput.value = selectedDate; };
    renderDateButton();

    dietDateBtn.addEventListener("click", () => {
      try { dietDateInput.showPicker(); } catch { dietDateInput.classList.add("fallback"); dietDateInput.focus(); }
    });
    dietDateInput.addEventListener("change", () => {
      const value = dietDateInput.value;
      dietDateInput.classList.remove("fallback");
      if (!value || value === selectedDate) return;
      MEAL_KEYS.forEach((k) => {
        (mealPlans[k] || []).forEach((entry) => {
          const lk = entry.foodId && foodItems.find((f) => f.id === entry.foodId);
          if (lk && Number.isFinite(Number(entry.amountValue))) lk.consumedAmount = roundAmount((Number(lk.consumedAmount) || 0) + Number(entry.amountValue));
        });
        mealPlans[k] = [];
      });
      saveFoodItems(); saveMealPlans();
      selectedDate = value; localStorage.setItem(DATE_STORAGE_KEY, selectedDate);
      renderDateButton(); renderSidebarList(); renderMealList();
    });

    dietSaveBtn.addEventListener("click", () => {
      if (!MEAL_KEYS.some((k) => (mealPlans[k] || []).length > 0)) {
        dietSaveStatus.textContent = t("nothingToSave"); dietSaveStatus.classList.add("error"); return;
      }
      const meals = {};
      MEAL_KEYS.forEach((k) => {
        meals[k] = (mealPlans[k] || []).map((e) => ({ name: e.name, code: e.code || "", amountValue: e.amountValue, amountUnit: e.amountUnit || "" }));
      });
      let history = [];
      try { history = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY)) || []; } catch { history = []; }
      const ex = history.find((h) => h.date === selectedDate);
      if (ex) ex.meals = meals; else history.push({ date: selectedDate, meals, rating: 5 });
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
      document.dispatchEvent(new CustomEvent("diet-history-changed"));
      dietSaveStatus.classList.remove("error");
      dietSaveStatus.textContent = t("savedForDate").replace("{date}", selectedDate);
    });

    /* ---- Hydration ---- */
    const addHydrationBtn = document.getElementById("addHydrationBtn");
    const hydrationListPanel = document.getElementById("hydrationListPanel");
    const hydrationFormPanel = document.getElementById("hydrationFormPanel");
    const hydrationListEl = document.getElementById("hydrationList");
    const hydrationFormTitle = document.getElementById("hydrationFormTitle");
    const volumeInput = document.getElementById("hydrationVolumeInput");
    const supplementsInput = document.getElementById("hydrationSupplementsInput");
    const hydrationCancelBtn = document.getElementById("hydrationCancelBtn");
    let editingHydrationId = null;

    function showHydrationSubpanel(mode, id = null) {
      const isForm = mode === "form";
      hydrationListPanel.classList.toggle("active", !isForm);
      hydrationFormPanel.classList.toggle("active", isForm);
      if (isForm) {
        editingHydrationId = id;
        const ex = hydrationItems.find((i) => i.id === id);
        hydrationFormTitle.textContent = t(ex ? "editHydrationData" : "addHydrationData");
        volumeInput.value = ex ? ex.volume : "";
        supplementsInput.value = ex ? ex.supplements : "";
      } else editingHydrationId = null;
    }

    function renderHydrationList() {
      hydrationListEl.replaceChildren();
      if (!hydrationItems.length) {
        const empty = document.createElement("li");
        empty.className = "hydration-empty";
        empty.textContent = t("noHydrationData");
        hydrationListEl.appendChild(empty);
        return;
      }
      hydrationItems.forEach((item) => {
        const li = document.createElement("li");
        li.className = "hydration-item";
        const info = document.createElement("div");
        info.className = "hydration-info";
        const vol = document.createElement("div");
        vol.className = "hydration-volume";
        vol.textContent = `${t("volumeOfWater")}: ${item.volume}`;
        info.appendChild(vol);
        if (item.supplements) {
          const sup = document.createElement("div");
          sup.className = "hydration-supplements";
          sup.textContent = `${t("addedSupplements")}: ${item.supplements}`;
          info.appendChild(sup);
        }
        const actions = document.createElement("div");
        actions.style.display = "flex"; actions.style.gap = "8px";
        actions.append(
          makeActionBtn("btn small", "editHydrationData", () => showHydrationSubpanel("form", item.id)),
          makeActionBtn("btn small danger", "deleteHydrationData", () => {
            hydrationItems = hydrationItems.filter((i) => i.id !== item.id);
            saveHydrationItems(); renderHydrationList();
          })
        );
        li.append(info, actions);
        hydrationListEl.appendChild(li);
      });
    }

    addHydrationBtn.addEventListener("click", () => showHydrationSubpanel("form"));
    hydrationCancelBtn.addEventListener("click", () => showHydrationSubpanel("list"));
    hydrationFormPanel.addEventListener("submit", (e) => {
      e.preventDefault();
      const volume = volumeInput.value.trim();
      if (!volume) return;
      const supplements = supplementsInput.value.trim();
      const ex = hydrationItems.find((i) => i.id === editingHydrationId);
      if (ex) { ex.volume = volume; ex.supplements = supplements; }
      else hydrationItems.push({ id: makeId(), volume, supplements });
      saveHydrationItems(); renderHydrationList(); showHydrationSubpanel("list");
    });

    document.addEventListener("app-language-changed", () => {
      renderSidebarList();
      if (detailPanel.classList.contains("active")) renderDetail();
      else renderMealCategory();
      renderHydrationList();
      updateDietAIBar();
    });

    showView("food");
    showFoodSubpanel("meals");
    renderHydrationList();
    setTimeout(updateDietAIBar, 120);
  })();

  /* ========================================================================
     3. DIET HISTORY MODULE
     ======================================================================== */
  (() => {
    const HISTORY_STORAGE_KEY = "dietHistory";
    const MEAL_KEYS = ["breakfast", "lunch", "dinner", "snack"];
    const body = document.getElementById("historyBody");
    if (!body) return;

    function loadHistory() {
      try { const d = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY)); return Array.isArray(d) ? d : []; }
      catch { return []; }
    }

    function render() {
      body.replaceChildren();
      const list = loadHistory().filter((i) => i && i.date).sort((a, b) => String(b.date).localeCompare(String(a.date)));
      if (!list.length) {
        const empty = document.createElement("div");
        empty.className = "history-empty";
        empty.textContent = t("noHistory");
        body.appendChild(empty);
        return;
      }
      list.forEach((item) => {
        const row = document.createElement("div");
        row.className = "history-row history-data-row";
        const dateCell = document.createElement("div");
        dateCell.className = "history-cell history-date";
        dateCell.textContent = item.date;
        row.appendChild(dateCell);

        MEAL_KEYS.forEach((key) => {
          const cell = document.createElement("div");
          cell.className = "history-cell";
          const entries = (item.meals && item.meals[key]) || [];
          if (!entries.length) {
            cell.innerHTML = `<span class="history-none">—</span>`;
          } else {
            const ul = document.createElement("ul");
            ul.className = "history-meal-list";
            entries.forEach((e) => {
              const li = document.createElement("li");
              if (e.amountValue) {
                const sp = document.createElement("span");
                sp.className = "history-amount";
                sp.textContent = e.amountUnit ? `${e.amountValue} ${e.amountUnit}` : String(e.amountValue);
                li.appendChild(sp);
              }
              const nm = document.createElement("span");
              nm.textContent = e.name;
              li.appendChild(nm);
              ul.appendChild(li);
            });
            cell.appendChild(ul);
          }
          row.appendChild(cell);
        });

        const rCell = document.createElement("div");
        rCell.className = "history-cell history-rating-cell";
        const rating = Number(item.rating) || 0;
        for (let i = 1; i <= 5; i++) {
          const star = document.createElement("button");
          star.type = "button";
          star.className = "history-star" + (i <= rating ? " filled" : "");
          star.textContent = String(i);
          star.addEventListener("click", () => {
            const all = loadHistory();
            const found = all.find((h) => h.date === item.date);
            if (found) { found.rating = rating === i ? 0 : i; localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(all)); render(); }
          });
          rCell.appendChild(star);
        }
        row.appendChild(rCell);
        body.appendChild(row);
      });
    }

    document.addEventListener("diet-history-changed", render);
    document.addEventListener("app-language-changed", render);
    render();
  })();
})();
