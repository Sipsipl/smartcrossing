/* Language support (German / English), dependency-free.
 *
 *  - German is the source text of the static HTML. Elements with data-i18n="key" get the English
 *    string of window.SC_I18N.en[key] (innerHTML) when the page is shown in English;
 *    data-i18n-attr="aria-label=key,content=key2" does the same for attributes.
 *  - Generated text (pages.js, brain.js, crossing.js, charts) uses SCX.t(key, {vars}); keys of
 *    the generated sentences live in both js/i18n-de.js and js/i18n-en.js.
 *  - Data from site/build.py carries English next to German (label_en, note_en, name_en ...);
 *    SCX.tl(obj, "label") picks the one for the current language.
 *  - The choice is remembered in localStorage ("sc-lang"); default = browser language (de-* -> German,
 *    else English; ?lang=de|en overrides and is the fallback without storage). Switching reloads the page (the simulations restart).
 */
(function () {
  "use strict";
  const SCX = (window.SCX = window.SCX || {});
  const DICT = window.SC_I18N || (window.SC_I18N = { de: {}, en: {} });
  const KEY = "sc-lang";

  let lang = null;
  const q = /[?&]lang=(de|en)\b/.exec(location.search); // fallback when storage is blocked
  if (q) lang = q[1];
  try { if (lang) localStorage.setItem(KEY, lang); } catch (e) { /* storage blocked */ }
  if (!lang) try { const s = localStorage.getItem(KEY); if (s === "de" || s === "en") lang = s; } catch (e) { /* storage blocked */ }
  if (!lang) lang = /^de\b/i.test((navigator.language || "").split("-")[0] || "") ? "de" : "en";
  SCX.lang = lang;
  document.documentElement.lang = lang;

  const fill = (s, vars) => String(s).replace(/\{(\w+)\}/g, (m, k) => (vars && vars[k] !== undefined ? vars[k] : m));
  const lookup = (key) => {
    const own = DICT[lang] && DICT[lang][key];
    return own !== undefined ? own : DICT.de[key] !== undefined ? DICT.de[key] : key;
  };
  /** Text for `key` in the current language, `{name}` placeholders filled from vars. */
  SCX.t = (key, vars) => fill(lookup(key), vars);
  /** obj[field] in the current language (build.py exports <field>_en next to the German <field>). */
  SCX.tl = (obj, field) => (obj && lang === "en" && obj[field + "_en"] != null ? obj[field + "_en"] : obj && obj[field]);

  // number formats: German decimal comma, English decimal point
  const EN = lang === "en";
  SCX.num = (v, d = 1) => (v === null || v === undefined ? "–" : EN ? v.toFixed(d) : v.toFixed(d).replace(".", ","));
  SCX.pct = (v) => (v === null || v === undefined ? "–" : EN ? `${Math.round(v * 100)}%` : `${Math.round(v * 100)} %`);
  SCX.pctInt = (v) => (EN ? `${v}%` : `${v} %`);
  SCX.group = (n) => n.toLocaleString(EN ? "en-US" : "de-DE");
  SCX.signed = (v, d = 2) => (v > 0 ? "+" : v < 0 ? "−" : "") + SCX.num(Math.abs(v), d);
  /** m/s -> km/h, rounded (speeds are always shown in km/h; the data stay in m/s). */
  SCX.kmh = (ms, d = 0) => SCX.num(ms * 3.6, d);

  /** A layout (build.py obs_layout) with the labels of the current language. */
  const layoutCache = new WeakMap();
  SCX.localizeLayout = (lay) => {
    if (!lay || !EN) return lay;
    if (!layoutCache.has(lay)) {
      layoutCache.set(lay, Object.assign({}, lay, {
        actions: lay.actions_en || lay.actions,
        groups: lay.groups.map((g) => Object.assign({}, g, {
          name: g.name_en || g.name,
          items: g.items.map((it) => Object.assign({}, it, { label: it.label_en || it.label })),
        })),
      }));
    }
    return layoutCache.get(lay);
  };

  // ------------------------------------------------------------------ static page texts
  function applyStatic() {
    const vars = { built: window.SC_DATA ? window.SC_DATA.built : "", repo: "https://github.com/Sipsipl/smartcrossing" };
    if (EN) {
      const en = DICT.en;
      document.querySelectorAll("[data-i18n]").forEach((el) => {
        const s = en[el.getAttribute("data-i18n")];
        if (s !== undefined) el.innerHTML = fill(s, vars);
      });
      document.querySelectorAll("[data-i18n-attr]").forEach((el) => {
        for (const pair of el.getAttribute("data-i18n-attr").split(",")) {
          const [attr, key] = pair.split("=");
          if (en[key] !== undefined) el.setAttribute(attr, fill(en[key], vars));
        }
      });
    }
    document.querySelectorAll("[data-lang]").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-lang") === lang));
      b.addEventListener("click", () => {
        const next = b.getAttribute("data-lang");
        if (next === lang) return;
        let stored = false;
        try { localStorage.setItem(KEY, next); stored = localStorage.getItem(KEY) === next; } catch (e) { /* storage blocked */ }
        if (stored) location.href = location.pathname + location.hash; // plain reload without ?lang=
        else location.search = "?lang=" + next;
      });
    });
  }
  applyStatic();
})();
