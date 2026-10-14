/* Page logic for every page of the site (selected by <body data-page>). */
(function () {
  "use strict";
  const SC = window.SmartCrossing;
  const SCX = window.SCX;
  const D = window.SC_DATA;
  const $ = SCX.$;
  const page = document.body.dataset.page;

  // ------------------------------------------------------------------ data helpers
  const METHOD_NAMES = { es: "Evolution", ppo: "PPO" };
  const pct = (v) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)} %`);
  const num = (v, d = 1) => (v === null || v === undefined ? "–" : v.toFixed(d).replace(".", ","));
  const pol = (id) => D.policies[id];
  const mainMethod = D.main || (D.best.ppo ? "ppo" : "es");
  const MAIN_ID = D.main_id || D.best[mainMethod];
  const isSetEntry = (e) => !!(e && e.policy && e.policy.arch && e.policy.arch.type === "set");
  const isNarrow = (entry) => !!(entry && entry.physics && entry.physics.W !== undefined && entry.physics.W < 5);

  function genLabel(entry) {
    if (!entry) return "";
    if (entry.id === "zufall") return "Start: zufälliges Gehirn";
    const unit = entry.method === "ppo" ? "Trainingsschritt" : "Generation";
    const where = isNarrow(entry) && !isNarrow(pol(MAIN_ID)) ? " · alte, schmale Kreuzung" : "";
    return (entry.tlGen !== undefined ? `${unit} ${entry.tlGen}` : `${entry.run_label || entry.run}`) + where;
  }

  /** Default agent types: the main brain (100 %), plus a second run and the old narrow-road
   *  brain for mixing (share 0). */
  function defaultTypes() {
    const types = [];
    const main = pol(MAIN_ID);
    if (main) types.push(SCX.makeType(main, "Aktuelles Gehirn", { share: 100 }));
    if (D.mix && pol(D.mix)) types.push(SCX.makeType(pol(D.mix), "Zweiter Lauf (anderer Zufall)", { share: 0 }));
    if (D.history && pol(D.history) && D.history !== MAIN_ID) types.push(SCX.makeType(pol(D.history), isNarrow(pol(D.history)) ? "Alt (schmale Kreuzung)" : "Älterer Stand", { share: 0 }));
    if (types.length < 2) {
      const om = mainMethod === "ppo" ? "es" : "ppo";
      if (D.best[om]) types.push(SCX.makeType(pol(D.best[om]), `${METHOD_NAMES[om]} (bestes Gehirn)`, { share: 0 }));
    }
    // all cars of one sim share the observation interface (lamp count, neighbour layout) of the first brain
    const key = (t) => SCX.IFACE_KEYS.map((k) => t.physics[k]).join("/");
    return types.filter((t) => key(t) === key(types[0])).slice(0, 3);
  }

  // ------------------------------------------------------------------ stage markup
  function buildStage(host, opts) {
    opts = opts || {};
    host.classList.add("stage");
    host.innerHTML = `
      ${opts.stats === false ? "" : `<div class="stats">
        <div class="stat"><div class="k">Durchsatz</div><div class="v"><span data-stat="thr">0</span><span class="u">/min</span></div></div>
        <div class="stat"><div class="k">Angekommen</div><div class="v" data-stat="exits">0</div></div>
        <div class="stat bad"><div class="k">Unfälle</div><div class="v" data-stat="crash">0</div></div>
        <div class="stat bad"><div class="k">Abgekommen</div><div class="v" data-stat="off">0</div></div>
        <div class="stat"><div class="k">Warten</div><div class="v"><span data-stat="queue">0</span></div></div>
      </div>`}
      <div class="canvas-box"><canvas aria-label="Die Kreuzung von oben"></canvas>${opts.hint ? `<div class="canvas-hint">${opts.hint}</div>` : ""}</div>
      <div class="transport">
        <button class="btn primary" data-act="play" type="button">Pause</button>
        <div class="seg" role="radiogroup" aria-label="Tempo"></div>
        <button class="btn" data-act="reset" type="button">Neu starten</button>
        <label class="toggle"><input type="checkbox" data-act="trails"> Fahrtwege</label>
        ${opts.follow ? `<label class="toggle"><input type="checkbox" data-act="follow"> Kamera folgt</label>` : ""}
        <span class="clock" data-act="clock"></span>
      </div>`;
    return host.querySelector("canvas");
  }

  function makeCrossing(host, opts) {
    const canvas = buildStage(host, opts);
    const cx = new SCX.Crossing(canvas, Object.assign({ types: opts.types || defaultTypes() }, opts));
    const tb = SCX.bindTransport(cx, host, opts.speeds);
    let last = 0;
    const refresh = (force) => {
      const now = performance.now();
      if (!force && now - last < 120) return;
      last = now;
      SCX.renderStats(cx, host);
      tb.update();
      if (cx.opts.onRefresh) cx.opts.onRefresh(cx);
    };
    cx.opts.onFrame = () => refresh(false);
    cx.onChange(() => refresh(true));
    const hint = host.querySelector(".canvas-hint");
    if (hint) {
      cx.opts.onPick = () => { hint.style.opacity = "0"; };
      setTimeout(() => { hint.style.opacity = "0"; }, 6000);
    }
    // small screens: let the camera follow the selected car so the sensors are readable
    const fol = host.querySelector("[data-act=follow]");
    if (fol && canvas.getBoundingClientRect().width < 520) { fol.checked = true; cx.follow = true; }
    refresh(true);
    // pause the loop while the canvas is off screen (saves battery on long pages)
    new IntersectionObserver((es) => { cx.visible = es[0].isIntersecting; }).observe(canvas);
    return cx;
  }

  function layoutWarning(host) {
    const msg = SCX.checkLayout();
    if (msg && host) host.insertAdjacentHTML("afterbegin", `<p class="note warn">${msg}</p>`);
    return !msg;
  }

  // ------------------------------------------------------------------ Start
  function initStart() {
    const types = defaultTypes();
    const cx = makeCrossing($("stage"), { types, speeds: [1, 2, 4, 8], speed: 1, select: false, warmup: 20 });
    window.__sc = cx;
    const c = cx.controls;
    const apply = () => {
      c.demand = +$("demand").value;
      let r = +$("pright").value, l = +$("pleft").value;
      if (r + l > 100) { l = 100 - r; $("pleft").value = l; }
      const s = 100 - r - l;
      c.turn = [r, s, l];
      c.signal = $("sigon").checked;
      c.noise = +$("noise").value;
      $("demand-o").textContent = `${num(c.demand * 240, 0)} Autos/min`;
      $("pright-o").textContent = `${r} %`;
      $("pleft-o").textContent = `${l} %`;
      $("pstraight").textContent = `Geradeaus: ${s} %`;
      $("noise-o").textContent = num(c.noise, 2);
      $("noise").disabled = !c.signal;
      cx.applyControls();
    };
    for (const id of ["demand", "pright", "pleft", "noise"]) $(id).addEventListener("input", apply);
    $("sigon").addEventListener("change", apply);
    apply();
    // agent types
    const box = $("types");
    const physKey = (t) => JSON.stringify(t.physics);
    types.forEach((t, i) => {
      const row = document.createElement("div");
      row.className = "type";
      row.innerHTML = `<span class="swatch" style="border-color:${SCX.TYPE_COLOURS[i]}"></span>
        <div class="tname" title="${t.id}">${t.name}</div>
        <div class="row2"><input type="range" min="0" max="100" step="1" value="${t.share}" aria-label="Anteil ${t.name}"><output></output>
        <label class="toggle"><input type="checkbox"> stumm</label></div>
        <div class="meta count" hidden>0 angekommen · 0 verunglückt</div>
        ${physKey(t) !== physKey(types[0]) ? `<div class="meta">trainiert mit anderer Fahrphysik; hier fährt es mit der Physik von „${types[0].name}“</div>` : ""}`;
      const rng = row.querySelector("input[type=range]"), cb = row.querySelector("input[type=checkbox]");
      rng.addEventListener("input", () => { t.share = +rng.value; showShares(); cx.applyControls(); });
      cb.addEventListener("change", () => { t.mute = cb.checked; });
      box.appendChild(row);
    });
    function showShares() {
      const tot = types.reduce((a, t) => a + t.share, 0);
      box.querySelectorAll(".type").forEach((row, i) => {
        row.querySelector("output").textContent = tot ? `${Math.round((100 * types[i].share) / tot)} %` : "0 %";
      });
    }
    showShares();
    // per-brain counts only when more than one brain is driving
    cx.opts.onRefresh = () => {
      const several = types.filter((t) => t.share > 0).length > 1;
      box.querySelectorAll(".type").forEach((row, i) => {
        const p = cx.perType[i], m = row.querySelector(".meta.count");
        m.hidden = !several;
        if (p && several) m.textContent = `${p.exits} angekommen · ${p.bad} verunglückt`;
      });
    };
    // honest status line
    const b = pol(MAIN_ID);
    const r15 = b && b.bench["0.15"], l15 = D.baselines.light["0.15"], v15 = D.baselines.reservation["0.15"];
    if (r15) {
      $("status").innerHTML = `Bei mittlerem Verkehr (${num(r15.demand_per_min, 0)} Autos pro Minute) schaffen es die besten gelernten Autos
        zu <b>${pct(r15.exit_ok_rate)}</b> heil ans Ziel – <b>${num(r15.throughput_per_min)}</b> Autos pro Minute.
        Eine Ampel schafft ${num(l15.throughput_per_min)} pro Minute, ein zentraler Verkehrsrechner ${num(v15.throughput_per_min)} – beide ohne einen einzigen Unfall.
        Die Autos lernen also noch.${prevNote()}
        <a href="gehirn.html">Zum Gehirn →</a> · <a href="vergleich.html">Zum Vergleich →</a>`;
    }
  }

  /** Crashes (+ off-road) per 1000 cars of a bench entry at demand d, or null. */
  const badPer1000 = (e, d) => (e && e.bench[d] ? e.bench[d].crash_cars_per_1000 : null);

  /** "17 statt 105" style comparison of main vs previous brain at demand d (bench data), or "". */
  function crashPair(d) {
    const a = badPer1000(D.previous && pol(D.previous), d), b = badPer1000(pol(MAIN_ID), d);
    return a === null || b === null ? "" : `${num(b, 0)} statt ${num(a, 0)}`;
  }

  /** What is new in the main brain vs the previous one (crash numbers from the bench data). */
  function prevNote() {
    const m = pol(MAIN_ID), mid = crashPair("0.15"), dense = crashPair("0.3");
    if (!mid) return "";
    const a = badPer1000(pol(D.previous), "0.15"), b = badPer1000(m, "0.15");
    const much = a > 0 && b < 0.5 * a ? "viel weniger" : b < a ? "weniger" : "nicht weniger";
    // bench values are means over several traffic seeds, but still a rounded comparison
    return isSetEntry(m)
      ? ` Neu beim aktuellen Gehirn: Es schaut auf bis zu ${m.policy.arch.K} Nachbarn, jeden mit demselben Mini-Netz, und eine „Aufmerksamkeit“ entscheidet, wer gerade zählt.
      Es baut ${much} Unfälle als das vorherige (in unserer Messung pro 1000 Autos: bei mittlerem Verkehr ${mid}${dense ? `, bei dichtem ${dense}` : ""}) –
      bei dichtem Verkehr kracht es mitten auf der Kreuzung aber noch, und manchmal verkeilen sich die Autos dort.`
      : ` Bei mittlerem Verkehr baut das aktuelle Gehirn ${much} Unfälle als das vorherige (in unserer Messung ${mid} pro 1000 Autos).`;
  }

  // ------------------------------------------------------------------ obs panel ("Was sieht ein Auto?")
  function fmtObs(it, v, compact) {
    if (it.unit === "bool") return v > 0.5 ? "ja" : "nein";
    if (compact) return it.unit === "m" || it.unit === "km/h" || it.unit === "%" ? num(v * it.scale, 0) : it.unit ? num(v * it.scale, 1) : num(v, 2);
    if (it.key.startsWith("ray_") && v >= 0.999) return `≥ ${num(it.scale, 0)} m`;
    const x = v * it.scale;
    if (it.unit === "m") return `${num(x)} m`;
    if (it.unit === "km/h") return `${num(x, 0)} km/h`;
    if (it.unit === "%") return `${num(x, 0)} %`;
    return num(v, 2);
  }

  function miniBar(v) {
    const c = Math.max(-1, Math.min(1, v));
    const left = c >= 0 ? 50 : 50 + c * 50, w = Math.abs(c) * 50;
    return `<span class="mini"><i style="left:${left}%;width:${w}%"></i></span>`;
  }

  /** Inputs a brain ignores (all weights 0: trained before the sensor existed, see policy_compat). */
  const unusedCache = new WeakMap();
  function unusedInputs(raw) {
    if (!raw || !raw.layers) return new Set(); // set encoders: no legacy (zero-weight) inputs
    if (!unusedCache.has(raw)) {
      const W = raw.layers[0].W, sc = raw.in_scale, off = new Set();
      W.forEach((row, i) => { if ((sc && sc[i] === 0) || row.every((w) => w === 0)) off.add(i); });
      unusedCache.set(raw, off);
    }
    return unusedCache.get(raw);
  }
  const UNUSED_TIP = "Dieses Gehirn nutzt den Wert nicht: es wurde trainiert, bevor es diesen Sensor gab.";

  function renderObsPanel(cx, el) {
    const n = cx.selIndex();
    if (n < 0) { el.innerHTML = ""; return; }
    const o = cx.obsRow(n), lay = SCX.layoutFor(cx.sim.cfg);
    if (!lay) { el.innerHTML = ""; return; }
    const ty = cx.typeOf(n), off = unusedInputs(ty && ty.raw);
    const own = lay.groups.filter((g) => g.sub !== "neighbour");
    const nbs = lay.groups.filter((g) => g.sub === "neighbour");
    let h = `<div class="obs-grid">`;
    for (const g of own) {
      h += `<div class="obs-group"><div class="gname">${g.name}<span class="n">${g.items.length} ${g.items.length === 1 ? "Zahl" : "Zahlen"}</span></div>`;
      for (const it of g.items) {
        h += off.has(it.i)
          ? `<div class="orow unused" title="${UNUSED_TIP}"><span class="lab">${it.label}</span><span class="val">${fmtObs(it, o[it.i])}</span><span class="small">ungenutzt</span></div>`
          : `<div class="orow"><span class="lab">${it.label}</span><span class="val">${fmtObs(it, o[it.i])}</span>${miniBar(o[it.i])}</div>`;
      }
      h += `</div>`;
    }
    h += `</div>`;
    // neighbours as a table: rows = features, columns = neighbour 1..K
    const nItems = nbs[0].items.length;
    h += `<div class="obs-group" style="margin-top:12px"><div class="gname">Nachbarn 1–${nbs.length}<span class="n">je ${nItems} Zahlen, nach Abstand sortiert</span></div>
      <table class="ntable"><thead><tr><th></th>${nbs.map((_, k) => `<th>${k + 1}</th>`).join("")}</tr></thead><tbody>`;
    const present = nbs.map((g) => o[g.items.find((it) => it.key.endsWith("_present")).i] > 0.5);
    for (let r = 0; r < nItems; r++) {
      const it0 = nbs[0].items[r];
      const lab = { fwd: "vor mir (m)", left: "links von mir (m)", same: "gleiche Richtung", cross: "kreuzt nach links", speed: "Tempo (km/h)", present: "da?", kappa: "lenkt links (%)", acc: "beschleunigt (m/s²)", brake: "Bremslicht", hazard: "Warnblinker (Wrack)" }[it0.key.split("_").slice(1).join("_")] || it0.label;
      const unused = off.has(it0.i);
      h += `<tr${unused ? ` class="unused" title="${UNUSED_TIP}"` : ""}><td>${lab}${unused ? " (ungenutzt)" : ""}</td>` + nbs.map((g, k) => {
        const it = g.items[r];
        return present[k] ? `<td title="${it.label}">${fmtObs(it, o[it.i], true)}</td>` : `<td class="none">–</td>`;
      }).join("") + `</tr>`;
    }
    const lampIdx = nbs.map((g) => g.items.filter((it) => /_lamp\d$/.test(it.key)).map((it) => it.i));
    h += `<tr><td>Lampenfarbe</td>` + nbs.map((_, k) => present[k]
      ? `<td><span class="lamp" style="background:${cx.sim.cfg.signal_enabled ? SCX.lampCSS(lampIdx[k].length, o[lampIdx[k][0]], o[lampIdx[k][1]]) : "rgb(90,90,96)"}"></span></td>`
      : `<td class="none">–</td>`).join("") + `</tr>`;
    h += `</tbody></table></div>`;
    el.innerHTML = h;
  }

  function initSehen() {
    layoutWarning($("panel"));
    const cx = makeCrossing($("stage"), {
      overlay: "sensors", autoSelect: true, speed: 0.5, speeds: [0.25, 0.5, 1, 2], follow: true,
      hint: "Tippe ein Auto an",
      onRefresh: (c) => { SCX.renderSelHead(c, $("selhead")); renderObsPanel(c, $("obs")); },
    });
    const lay = SCX.layoutFor(cx.sim.cfg) || D.layout;
    $("dim-count").textContent = lay.dim;
    document.querySelectorAll("[data-nbk]").forEach((e) => { e.textContent = lay.k_neighbours; });
    window.__sc = cx;
  }

  // ------------------------------------------------------------------ outputs ("Was kann ein Auto tun?")
  function initTun() {
    const cfg = SC.defaultConfig();
    const hist = [];
    let histUid = -1;
    const histCanvas = $("hist");
    const cols = () => [SCX.cssVar("--c1"), SCX.cssVar("--c2"), SCX.cssVar("--c3"), SCX.cssVar("--c4")];
    const cx = makeCrossing($("stage"), {
      overlay: "none", autoSelect: true, speed: 0.5, speeds: [0.25, 0.5, 1, 2], follow: true, hint: "Tippe ein Auto an",
      onStep: (c) => {
        const n = c.selIndex();
        if (n < 0) return;
        if (c.sim.uid[n] !== histUid) { hist.length = 0; histUid = c.sim.uid[n]; }
        hist.push(nextAction(c, n));
        if (hist.length > 100) hist.shift();
      },
      onRefresh: (c) => { SCX.renderSelHead(c, $("selhead")); renderOutputs(c); drawHist(); },
    });
    function nextAction(c, n) {
      const t = c.typeOf(n);
      const a = Array.from(SC.forward(t.policy, c.obs, n * SC.obsDim(c.sim.cfg)));
      if (t.mute) for (let q = 2; q < a.length; q++) a[q] = 0;
      return a.map((v) => Math.max(-1, Math.min(1, v)));
    }
    function bar(v, flip, colour, tick) {
      const p = flip ? -v : v;
      const left = p >= 0 ? 50 : 50 + p * 50, w = Math.abs(p) * 50;
      return `<div class="bar"><i style="left:${left}%;width:${w}%;${colour ? `background:${colour}` : ""}"></i>${tick !== undefined ? `<span class="tick" style="left:${50 + (flip ? -tick : tick) * 50}%"></span>` : ""}</div>`;
    }
    function renderOutputs(c) {
      const n = c.selIndex(), el = $("outs");
      if (n < 0) { el.innerHTML = ""; return; }
      const a = nextAction(c, n), sc = c.sim.cfg;
      const acc = a[0] > 0 ? a[0] * sc.acc_max : a[0] * sc.brake_max;
      const k = c.sim.kappa[n] / sc.kappa_max;
      const radius = Math.abs(a[1]) > 0.02 ? 1 / (Math.abs(a[1]) * sc.kappa_max) : null;
      const [c1, c2] = cols();
      const lampOn = sc.signal_enabled;
      el.innerHTML = `
        <div class="out"><div class="top-row"><b><i class="line-key" style="border-color:${c1}"></i> Gas / Bremse</b><span class="phys">${acc >= 0 ? "+" : "−"}${num(Math.abs(acc))} m/s² ${acc >= 0 ? "Gas" : "Bremse"}</span></div>
          ${bar(a[0], false, c1)}<div class="bar-ends"><span>voll bremsen</span><span>${num(a[0], 2)}</span><span>Vollgas</span></div></div>
        <div class="out"><div class="top-row"><b><i class="line-key" style="border-color:${c2}"></i> Lenkung</b><span class="phys">${radius ? `Kurve nach ${a[1] > 0 ? "links" : "rechts"}, Radius ${num(radius, 0)} m` : "geradeaus"}</span></div>
          ${bar(a[1], true, c2, sc.kappa_rate < 1e5 ? k : undefined)}<div class="bar-ends"><span>links</span><span>${num(a[1], 2)}</span><span>rechts</span></div></div>
        ${a.slice(2).map((v, q) => `<div class="out"><div class="top-row"><b><i class="line-key" style="border-color:${cols()[2 + q]}"></i> ${a.length > 3 ? `Lampe Kanal ${q + 1}` : "Lampe"}</b><span class="phys">${a.length > 3 ? (q === 0 ? "mehr Rot" : "mehr Blau") : "heller"} bei +1</span></div>
          ${bar(v, false, cols()[2 + q])}<div class="bar-ends"><span>−1</span><span>${num(v, 2)}</span><span>+1</span></div></div>`).join("")}
        <div class="lampbox"><span class="big" style="background:${lampOn ? SCX.lampCSS(a.length - 2, a[2], a[3]) : "rgb(90,90,96)"}"></span>
          <span>So leuchtet die Dachlampe${c.typeOf(n).mute ? " (stumm geschaltet)" : ""}${lampOn ? "" : " – die anderen sehen sie gerade nicht"}.</span></div>
        ${sc.kappa_rate < 1e5 ? `<p class="hint">Der schwarze Strich auf der Lenkung zeigt, wo das Lenkrad gerade steht: es dreht sich nur ${num(sc.kappa_rate / sc.kappa_max, 1)}-mal den vollen Weg pro Sekunde.</p>` : ""}`;
    }
    function drawHist() {
      const ctx = histCanvas.getContext("2d");
      const r = histCanvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      histCanvas.width = Math.round(r.width * dpr); histCanvas.height = Math.round(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const W = r.width, H = r.height;
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = SCX.cssVar("--line"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
      ctx.fillStyle = SCX.cssVar("--muted"); ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText("+1", 2, 10); ctx.fillText("−1", 2, H - 3); ctx.fillText("jetzt", W - 26, H / 2 - 4); ctx.fillText("vor 10 s", 22, H / 2 - 4);
      const c = cols();
      for (let q = 0; q < Math.min(4, hist.length ? hist[0].length : 0); q++) {
        ctx.strokeStyle = c[q]; ctx.lineWidth = 2; ctx.beginPath();
        hist.forEach((a, i) => {
          const x = W - (hist.length - 1 - i) * (W / 99), y = H / 2 - a[q] * (H / 2 - 4);
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        });
        ctx.stroke();
      }
    }
    // kinematics facts from the sim config
    const p = pol(MAIN_ID);
    const phys = Object.assign({}, cfg, p ? p.physics : {});
    $("kin").innerHTML = `
      <li>Höchstens <b>${num(phys.v_max * 3.6, 0)} km/h</b> (${num(phys.v_max, 0)} m/s) – wie in der Stadt.</li>
      <li>Beschleunigen mit bis zu <b>${num(phys.acc_max, 0)} m/s²</b> (von 0 auf 50 in etwa ${num(phys.v_max / phys.acc_max, 1)} s), bremsen mit bis zu <b>${num(phys.brake_max, 0)} m/s²</b> – das ist eine kräftige Vollbremsung.</li>
      <li>Engster Kreis: <b>${num(1 / phys.kappa_max, 0)} m Radius</b> – aber nur langsam: ${phys.a_lat_max > 0 ? `Die Reifen halten höchstens ${num(phys.a_lat_max, 0)} m/s² zur Seite, bei ${num(phys.v_max * 3.6, 0)} km/h ist der engste Kreis deshalb ${num(phys.v_max * phys.v_max / phys.a_lat_max, 0)} m groß.` : "(hier ohne Rutschgrenze)."}</li>
      <li>${phys.v_reverse > 0 ? `Rückwärtsgang: bis ${num(phys.v_reverse * 3.6, 0)} km/h (Gas &lt; 0 bremst erst, dann geht es rückwärts).` : "Rückwärts fahren geht nicht."}</li>
      ${cfg.kappa_rate ? `<li>Das Lenkrad dreht sich nicht beliebig schnell: von ganz links nach ganz rechts dauert es ${num((2 * cfg.kappa_max) / cfg.kappa_rate, 1)} s.${phys.kappa_rate > 1e5 ? " (Die bisher besten Evolutions-Autos wurden noch ohne diese Grenze trainiert und fahren hier deshalb mit sofort wirkender Lenkung.)" : ""}</li>` : ""}
      <li>Das Auto ist ${num(4.6, 1)} m lang und ${num(1.8, 1)} m breit; die Straße ist ${num(2 * (phys.W || cfg.W), 0)} m breit – ${(phys.W || cfg.W) >= 5 ? "Platz für zwei Autos nebeneinander pro Richtung" : "Platz für genau ein Auto pro Richtung"}.</li>`;
    window.__sc = cx;
  }

  // ------------------------------------------------------------------ brain
  function initGehirn() {
    layoutWarning($("brainnote"));
    const TIP0 = "Fahre mit der Maus über einen Kreis (oder tippe darauf).";
    const TIP_SET = "Fahre mit der Maus über einen Kreis, einen Nachbarn oder einen Balken (oder tippe darauf).";
    let net = null, setv = null, mismatch = false, attn = null;
    const tip = $("tip"), canvas = $("net"), setHost = $("setnet");
    const onHover = (txt) => { tip.textContent = txt || (!setHost.hidden ? TIP_SET : TIP0); };
    // brains to look at: the current one and the previous main brain (other structure: the classic MLP view)
    const picks = [{ id: MAIN_ID, name: "Aktuelles Gehirn" }];
    if (D.previous && pol(D.previous)) picks.push({ id: D.previous, name: "Vorheriges Gehirn" });
    let pick = 0;
    const cx = makeCrossing($("stage"), {
      types: [SCX.makeType(pol(picks[0].id), picks[0].name)],
      overlay: "attention", attention: () => attn,
      autoSelect: true, speed: 0.5, speeds: [0.25, 0.5, 1, 2], hint: "Tippe ein Auto an",
      onRefresh: (c) => {
        SCX.renderSelHead(c, $("selhead"));
        const n = c.selIndex();
        if (n < 0) { attn = null; return; }
        const t = c.typeOf(n), D_ = SC.obsDim(c.sim.cfg), lay = SCX.layoutFor(c.sim.cfg) || D.layout;
        const ref = SC.forward(t.policy, c.obs, n * D_);
        let out;
        if (t.policy.kind === "set") {
          canvas.style.display = "none"; setHost.hidden = false;
          if (!setv) setv = new SCX.SetView(setHost, lay, t.policy, { onHover, fmt: (it, v) => fmtObs(it, v), onFocus: () => c.draw() });
          else if (setv.pol !== t.policy) setv.setPolicy(t.policy, lay);
          const tr = SC.setTrace(t.policy, c.obs, n * D_);
          setv.update(tr, c.obsRow(n));
          attn = { w: setv.attention(), focus: setv.focus() };
          out = tr.out;
        } else {
          canvas.style.display = ""; setHost.hidden = true; attn = null;
          if (!net) net = new SCX.NetView(canvas, lay, t.raw, { onHover: (h) => onHover(h ? net.describe(h) : null) });
          else if (net.raw !== t.raw) { net.layout = lay; net.setPolicy(t.raw); }
          const trace = SCX.forwardTrace(t.raw, c.obs, n * D_);
          net.update(trace);
          if (net.hoverN) tip.textContent = net.describe(net.hoverN);
          out = trace[trace.length - 1];
        }
        // the drawn forward pass must equal the simulator's (sim.js) forward pass
        const diff = Math.max(...Array.from(out, (v, i) => Math.abs(v - ref[i])));
        if (diff > 1e-9 && !mismatch) {
          mismatch = true;
          console.error("brain forward differs from sim.js forward", diff);
          $("brainnote").insertAdjacentHTML("beforeend", `<p class="note warn">Achtung: Die Anzeige weicht vom Simulator ab (${diff}).</p>`);
        }
        window.__brainDiff = diff;
      },
    });
    function describeBrain() {
      const p = pol(picks[pick].id);
      if (!p) return;
      const set = isSetEntry(p);
      document.querySelectorAll("[data-set-only]").forEach((e) => { e.hidden = !set; });
      $("set-explain").hidden = !set;
      if (set) {
        const ar = p.policy.arch;
        $("brain-lead").innerHTML = `Jedes Auto wird von einem kleinen neuronalen Netz gesteuert. Das aktuelle ist in Teile gegliedert:
          die <b>eigenen Werte</b> (${ar.n_ego} Zahlen) und bis zu <b>${ar.K} Nachbarn</b> (je ${ar.F} Zahlen) → ein <b>Mini-Netz pro Nachbar</b>
          → <b>Aufmerksamkeit</b> („welcher Nachbar zählt gerade?“) → ${ar.hidden.join(" → ")} → ${ar.n_out} Ausgänge: Gas, Lenkung und Lampe.`;
        $("brain-params").textContent = p.policy.params.length.toLocaleString("de-DE");
        document.querySelectorAll("#set-explain [data-k]").forEach((e) => { e.textContent = ar.K; });
        document.querySelectorAll("#set-explain [data-heads]").forEach((e) => { e.textContent = ar.heads; });
      } else {
        const s = p.policy.sizes, params = s.slice(1).reduce((a, n, i) => a + s[i] * n + n, 0);
        $("brain-lead").innerHTML = `${pick > 0 ? "Das vorherige Gehirn ist ein klassisches neuronales Netz" : "Jedes Auto wird von einem kleinen neuronalen Netz gesteuert"}:
          <span id="brain-sizes">${s[0]} Eingänge → ${s.slice(1, -1).join(" → ")} → ${s[s.length - 1]} Ausgänge</span>.
          Links kommen die Sensorwerte rein, rechts kommen Gas, Lenkung und Lampe raus. Dazwischen: ${s.length - 2} Schichten „Neuronen“.`;
        $("brain-params").textContent = params.toLocaleString("de-DE");
      }
      onHover(null);
    }
    if (picks.length > 1) {
      const seg = $("brain-pick");
      seg.innerHTML = picks.map((p, i) => `<input type="radio" name="brain-pick" id="bp${i}" value="${i}"${i === 0 ? " checked" : ""}><label for="bp${i}">${p.name}</label>`).join("");
      seg.querySelectorAll("input").forEach((el) => el.addEventListener("change", () => {
        pick = +el.value;
        attn = null;
        cx.setTypes([SCX.makeType(pol(picks[pick].id), picks[pick].name)]);
        describeBrain();
      }));
      $("brain-pick-box").hidden = false;
    }
    describeBrain();
    window.__sc = cx;
  }

  // ------------------------------------------------------------------ learning
  // ------------------------------------------------------------------ reward scale (lernen)
  /** Draws the reward rules of the main brain (D.reward, from its training cfg) as a point scale + two example trips. */
  function drawReward() {
    const R = D.reward, host = $("reward");
    if (!R || !host) return;
    const f = (v, d = 2) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d).replace(".", ",");
    const IMPACT = 6;                                   // example impact speed (m/s)
    const crashAt = (v) => R.r_crash - R.r_crash_speed * v;
    const offAt = (v) => R.r_offroad - R.r_offroad_speed * v;
    const vTop = 10;
    const route = 2 * R.L;                              // straight through the crossing, entry to exit (m)
    const lo = Math.min(crashAt(vTop), offAt(vTop), R.r_wrong_exit, R.r_timeout), hi = Math.max(R.r_exit, R.r_progress * 100);
    const span = hi - lo, zero = (-lo / span) * 100;
    const x = (v) => (v / span) * 100;                  // value -> % of track width
    const bar = (a, b, cls) => {                        // bar from a to b (values)
      const l = Math.min(a, b), w = Math.abs(b - a);
      return `<i class="rw-bar ${cls}" style="left:${zero + x(l)}%;width:${x(w)}%"></i>`;
    };
    const row = (label, sub, bars, valTxt, sign) =>
      `<div class="rw-row"><div class="lbl">${label}${sub ? `<small>${sub}</small>` : ""}</div><div class="rw-track" style="--zero:${zero}%">${bars}</div><div class="val ${sign}">${valTxt}</div></div>`;
    const crashSame = R.r_crash === R.r_offroad && R.r_crash_speed === R.r_offroad_speed;
    const hard = (label, at) => row(label, `ab ${f(at(0), 1)}, bei Tempo ${vTop} m/s ${f(at(vTop), 1)}`,
      bar(0, at(0), "neg") + bar(at(0), at(vTop), "neg soft"), f(at(IMPACT), 1) + "*", "neg");
    const rows = [
      row("Richtige Ausfahrt erreicht", "", bar(0, R.r_exit, "pos"), f(R.r_exit, 1), "pos"),
      row("Dem Ziel näher kommen", `Lernhilfe: ${f(R.r_progress, 2)} je Meter`, bar(0, R.r_progress * 100, "help"), `${f(R.r_progress * 100, 1)} je 100 m`, "pos"),
      row("Zeit auf der Straße", "kleine Strafe je Sekunde", bar(0, -R.r_time * 10, "neg"), `${f(-R.r_time * 10, 1)} je 10 s`, "neg"),
      row("Falsche Ausfahrt", "", bar(0, R.r_wrong_exit, "neg"), f(R.r_wrong_exit, 1), "neg"),
      row("Zu lange unterwegs", `länger als ${Math.round(R.max_age)} s`, bar(0, R.r_timeout, "neg"), f(R.r_timeout, 1), "neg"),
      crashSame ? hard("Unfall oder von der Straße abgekommen", crashAt) : hard("Unfall", crashAt) + hard("Von der Straße abgekommen", offAt),
    ];
    // example trips (all numbers derived from the same rules)
    const tSec = route / R.spawn_speed;
    const trip = (title, lines) => {
      const tot = lines.reduce((a, l) => a + l[1], 0);
      return `<div class="card"><h3>${title}</h3>${lines.map((l) => `<div class="ln"><span>${l[0]}</span><span class="${l[1] >= 0 ? "pos" : "neg"}">${f(l[1])}</span></div>`).join("")}
        <div class="ln tot"><span>Summe</span><span class="${tot >= 0 ? "pos" : "neg"}">${f(tot)}</span></div></div>`;
    };
    const half = 0.5;
    const ex = trip("Eine normale Fahrt", [
      ["Unterwegs dem Ziel näher gekommen", R.r_progress * route],
      ["Richtige Ausfahrt erreicht", R.r_exit],
      [`Zeit (ca. ${Math.round(tSec)} s)`, -R.r_time * tSec],
    ]) + trip(`Unfall mit ${IMPACT} m/s`, [
      ["Halbe Strecke geschafft", R.r_progress * route * half],
      [`Unfall (Aufprall ${IMPACT} m/s)`, crashAt(IMPACT)],
      [`Zeit (ca. ${Math.round(tSec * half)} s)`, -R.r_time * tSec * half],
    ]);
    host.innerHTML = `${rows.join("")}<div class="rw-ex">${ex}</div>`;
    $("reward-note").innerHTML = `Grün = Pluspunkte, orange = Minuspunkte. *Beim Unfall zählt das Tempo des Aufpralls: je schneller, desto teurer (blasser Teil des Balkens). Bei einem Unfall zahlen <b>alle beteiligten Autos</b>. Die Zeitstrafe und die Strafe fürs Trödeln gibt es, weil Stehenbleiben sonst am sichersten wäre.`;
  }

  function initLernen() {
    drawReward();
    const methods = Object.keys(D.timelines).sort((a, b) => (a === "es" ? -1 : 1) - (b === "es" ? -1 : 1));
    let method = methods.includes(mainMethod) ? mainMethod : methods[0];
    let idx = 0, extra = null; // extra: a dead-end run shown instead of the timeline
    if (!method) { $("tt").innerHTML = `<p class="muted">Noch keine Trainingsdaten exportiert.</p>`; return; }
    const tabs = $("tt-methods");
    if (methods.length > 1) {
      for (const m of methods) {
        const b = document.createElement("button");
        b.type = "button"; b.className = "btn"; b.textContent = METHOD_NAMES[m];
        b.addEventListener("click", () => { method = m; extra = null; idx = D.timelines[m].length - 1; setupSlider(); show(); drawCurves(); });
        tabs.appendChild(b);
      }
    }
    const tl = () => D.timelines[method].map((p) => Object.assign({}, pol(p.id), { tlGen: p.gen }));
    idx = D.timelines[method].length - 1;
    const cx = makeCrossing($("stage"), {
      types: [SCX.makeType(pol(D.timelines[method][idx].id), "Zeitreise")], speeds: [1, 2, 4], speed: 1, select: false, seed: 3,
    });
    const slider = $("tt-range");
    function setupSlider() {
      const pts = tl();
      slider.max = pts.length - 1;
      slider.value = idx;
      const ticks = $("tt-ticks");
      ticks.innerHTML = "";
      pts.forEach((p, i) => {
        const b = document.createElement("button");
        b.type = "button"; b.textContent = p.id === "zufall" ? "0" : String(p.tlGen);
        b.title = genLabel(p);
        b.addEventListener("click", () => { idx = i; extra = null; slider.value = i; show(); });
        ticks.appendChild(b);
      });
      tabs.querySelectorAll("button").forEach((b, i) => b.classList.toggle("primary", methods[i] === method));
      setupOthers();
    }
    slider.addEventListener("input", () => { idx = +slider.value; extra = null; show(); });
    function show() {
      const p = extra ? Object.assign({}, pol(extra.id)) : tl()[idx];
      $("tt-ticks").querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(!extra && i === idx)));
      const chainAll = (D.chains[method] || []).filter((c) => ((D.curves[method] || {})[c.run] || []).length);
      const phase = chainAll.findIndex((c) => c.run === p.run);
      const chain = chainAll[phase];
      const note = extra ? extra.note : p.id === "zufall" ? "Vor dem Training: alle Gewichte sind Zufallszahlen. Die Autos fahren irgendwie los." : chain && chain.note;
      const b05 = p.bench["0.05"], b15 = p.bench["0.15"];
      $("tt-now").innerHTML = `<b>${extra ? (extra.label || extra.run) : genLabel(p)}</b>${chain && !extra ? ` <span class="muted small">(Abschnitt <span class="num-inline">${phase + 1}</span> der Lernkurve)</span>` : ""}<br>
        <span class="small">${note || ""}</span><br>
        <span class="small muted">Gemessen in 8 × 2 Minuten Simulation: bei wenig Verkehr kommen <b>${pct(b05 && b05.exit_ok_rate)}</b> der Autos an,
        bei mittlerem Verkehr <b>${pct(b15 && b15.exit_ok_rate)}</b> (${num(b15 && b15.throughput_per_min)} Autos/min).</span>`;
      cx.seed = 3; // same traffic for every checkpoint
      cx.setTypes([SCX.makeType(p, extra ? (extra.label || extra.run) : genLabel(p))]);
      drawCurves();
    }
    // dead-end runs
    const ow = $("tt-others");
    function setupOthers() {
      ow.innerHTML = "";
      const list = (D.others || []).filter((o) => (o.method || "es") === method);
      for (const o of list) {
        const b = document.createElement("button");
        b.type = "button"; b.className = "btn"; b.textContent = o.label || o.run;
        b.title = o.note || "";
        b.addEventListener("click", () => { extra = Object.assign({}, o, { note: o.note || "Ein anderer, früherer Trainingslauf." }); show(); });
        ow.appendChild(b);
      }
      ow.parentElement.style.display = list.length ? "" : "none";
    }

    function drawCurves() {
      const curves = D.curves[method] || {};
      const chain = (D.chains[method] || []).filter((c) => (curves[c.run] || []).length);
      if (!chain.length) { $("curves").innerHTML = `<p class="muted">Keine Logdaten.</p>`; return; }
      // phases = runs of the warm-start chain; each starts at the first logged point of its run
      // (iteration counts continue across warm starts) and is cut where the next one starts
      const starts = chain.map((c) => Math.min(...curves[c.run].map((p) => p.x)));
      const part = (i) => curves[chain[i].run].filter((p) => i + 1 >= chain.length || p.x < starts[i + 1]);
      const total = chain.reduce((a, _, i) => a + part(i).length, 0);
      const k = Math.max(1, Math.round(total / 80));
      const ser = (key) => chain.flatMap((_, i) => SCX.smooth(part(i).filter((p) => p[key] !== undefined).map((p) => [p.x, p[key]]), k));
      const phases = chain.map((c, i) => ({ x: starts[i], n: i + 1, title: c.step || c.note || c.label || c.run }));
      const WORLD = { narrow: "Alte, schmale Kreuzung (7 m Straßen)", wide: "Breite Kreuzung (14 m Straßen)" };
      const worlds = new Set(chain.map((c) => c.world).filter(Boolean));
      const bands = [];
      if (worlds.size > 1) chain.forEach((c, i) => {
        if (i === 0 || c.world !== chain[i - 1].world) bands.push({ x0: starts[i], x1: Infinity, label: { narrow: "schmale Kreuzung", wide: "breite Kreuzung" }[c.world] || "", world: c.world });
        else return;
        if (bands.length > 1) bands[bands.length - 2].x1 = starts[i];
      });
      const lastX = Math.max(...chain.flatMap((_, i) => part(i).map((p) => p.x)));
      bands.forEach((b) => { if (b.x1 === Infinity) b.x1 = lastX; });
      const leg = $("chain-legend");
      if (leg) leg.innerHTML = chain.map((c, i) => `${worlds.size > 1 && (i === 0 || c.world !== chain[i - 1].world) ? `<li class="world">${WORLD[c.world] || ""}</li>` : ""}<li><span class="num">${i + 1}</span><span>${c.step || c.note || c.label || c.run}</span></li>`).join("");
      const markers = [];
      if (!extra) {
        const cur = tl()[idx];
        if (cur) markers.push({ x: cur.tlGen, label: "▼ Zeitreise" });
      }
      SCX.lineChart($("curves"), {
        series: [
          { name: "angekommen", color: SCX.cssVar("--c1"), points: ser("exit"), tipMax: 20 },
          { name: "Unfall", color: SCX.cssVar("--c2"), points: ser("crash"), tipMax: 20 },
          { name: "von der Straße", color: SCX.cssVar("--c3"), points: ser("offroad"), tipMax: 20 },
        ],
        yMin: 0, yMax: 1, yFmt: (v) => `${Math.round(v * 100)} %`, xLabel: method === "ppo" ? "Trainingsschritt" : "Generation",
        xTipFmt: (x) => `${method === "ppo" ? "Schritt" : "Generation"} ${Math.round(x)}`,
        markers, phases, bands, directLabels: true, height: 300, aria: "Lernkurve: Anteil angekommener, verunglückter und abgekommener Autos, mit nummerierten Trainingsabschnitten",
      });
    }
    setupSlider();
    show();
    window.addEventListener("resize", debounce(drawCurves, 200));
    window.__sc = cx;
  }

  // ------------------------------------------------------------------ comparison
  /** Three canvases replaying the same traffic (pre-recorded by site/build.py) in sync. */
  function initSideBySide(host) {
    const R = window.SC_REPLAYS;
    if (!host) return;
    if (!R || !R.runs) { host.innerHTML = `<p class="muted">Keine Aufzeichnungen vorhanden.</p>`; return; }
    const DRIVERS = [["light", "Ampel"], ["reservation", "Verkehrsrechner"], ["policy", "Gelernte Autos"]].filter(([k]) => R.runs[k]);
    const LEVELS = ["wenig", "mittel", "viel"];
    const dems = R.demands.map(String);
    const world = Object.assign({ L: 50, W: 3.5, Rc: 6 }, R.world);
    host.innerHTML = `
      <div class="rp-ctl">
        <div class="seg" role="radiogroup" aria-label="Verkehr">${dems.map((d, i) => `<input type="radio" name="rp-dem" id="rp-d${i}" value="${d}"${i === 1 ? " checked" : ""}><label for="rp-d${i}">${LEVELS[i] || d}</label>`).join("")}</div>
        <button class="btn primary" type="button" data-act="play">Pause</button>
        <div class="seg" role="radiogroup" aria-label="Tempo">${[1, 2, 4].map((s) => `<input type="radio" name="rp-spd" id="rp-s${s}" value="${s}"${s === 1 ? " checked" : ""}><label for="rp-s${s}">${s}×</label>`).join("")}</div>
        <button class="btn" type="button" data-act="restart">Von vorn</button>
        <span class="clock" data-act="clock"></span>
      </div>
      <div class="rp-grid">${DRIVERS.map(([k, name]) => `
        <figure class="rp" data-k="${k}"><div class="rp-box"><canvas aria-label="${name}: Kreuzung von oben"></canvas></div>
          <figcaption><div class="rp-name">${name}</div>
          <div class="rp-stats"><span><b data-s="thr">–</b> Autos/min</span><span class="bad"><b data-s="bad">0</b> Unfälle</span></div></figcaption></figure>`).join("")}
      </div>`;
    const views = DRIVERS.map(([k]) => ({ k, fig: host.querySelector(`.rp[data-k=${k}]`) }));
    for (const v of views) v.view = new SCX.ReplayView(v.fig.querySelector("canvas"), world);
    let dem = dems[1] || dems[0], t = 0, speed = 1, running = true, lastTs = null, lastStats = 0, visible = true, dur = 0;
    const load = () => {
      for (const v of views) v.view.rep = SCX.decodeReplay(R.runs[v.k][dem], R.q);
      dur = Math.min(...views.map((v) => v.view.rep.steps * v.view.rep.dt));
      t = 0; render(true);
    };
    const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
    function render(stats) {
      for (const v of views) v.view.draw(t);
      if (!stats) return;
      for (const v of views) {
        const c = v.view.counts(t);
        v.fig.querySelector("[data-s=thr]").textContent = c.thr === null ? "–" : num(c.thr);
        v.fig.querySelector("[data-s=bad]").textContent = c.bad;
        v.fig.querySelector(".bad").classList.toggle("on", c.bad > 0);
      }
      host.querySelector("[data-act=clock]").textContent = `${mmss(t)} / ${mmss(dur)}`;
    }
    function frame(ts) {
      if (running && visible) {
        if (lastTs !== null) t += Math.min((ts - lastTs) / 1000, 0.1) * speed;
        if (t >= dur) t = 0; // loop
        lastTs = ts;
        const now = performance.now(), st = now - lastStats > 200;
        if (st) lastStats = now;
        render(st);
      } else lastTs = null;
      requestAnimationFrame(frame);
    }
    host.querySelectorAll("input[name=rp-dem]").forEach((el) => el.addEventListener("change", () => { dem = el.value; load(); }));
    host.querySelectorAll("input[name=rp-spd]").forEach((el) => el.addEventListener("change", () => { speed = +el.value; }));
    const play = host.querySelector("[data-act=play]");
    play.addEventListener("click", () => { running = !running; play.textContent = running ? "Pause" : "Weiter"; lastTs = null; });
    host.querySelector("[data-act=restart]").addEventListener("click", () => { t = 0; render(true); });
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(host);
    load();
    requestAnimationFrame(frame);
    window.__rp = { get t() { return t; }, views };
  }

  function initVergleich() {
    initSideBySide($("side"));
    // every brain is compared with Ampel / Verkehrsrechner on the crossing it was trained on
    const worldOf = (e) => (isNarrow(e) ? "narrow" : "wide");
    const WORLD_NAMES = { wide: "Breite Kreuzung (14 m Straßen)", narrow: "Alte, schmale Kreuzung (7 m Straßen)" };
    const byWorld = D.baselines_by_world || { [D.main_world || worldOf(pol(MAIN_ID))]: D.baselines };
    const mainWorld = MAIN_ID ? worldOf(pol(MAIN_ID)) : D.main_world || "wide";
    const brains = [];
    if (MAIN_ID) brains.push({ name: "Gelernte Autos", e: pol(MAIN_ID), color: "--c4" });
    if (D.previous && pol(D.previous)) brains.push({ name: "Vorheriges Gehirn", e: pol(D.previous), color: "--c3" });
    if (D.history && D.history !== MAIN_ID) brains.push({ name: worldOf(pol(D.history)) === mainWorld ? "Gelernt (älter)" : "Gelernt (PPO)", e: pol(D.history), color: "--c3" });
    if (D.best.es && D.best.es !== MAIN_ID) brains.push({ name: "Evolution", e: pol(D.best.es), color: "--c5" });
    const worlds = [mainWorld, ...Object.keys(byWorld).filter((w) => w !== mainWorld)];
    const groups = worlds.map((w) => {
      const base = byWorld[w];
      const rows = base ? [
        { name: "Ampel", b: base.light, color: "--c1", base: true },
        { name: "Verkehrsrechner", b: base.reservation, color: "--c2", base: true },
      ] : [];
      for (const r of brains) if (worldOf(r.e) === w) rows.push({ name: r.name, b: r.e.bench, color: r.color });
      return { w, rows };
    }).filter((g) => g.rows.some((r) => !r.base));
    const rows = (groups.find((g) => g.w === mainWorld) || groups[0]).rows; // chart: main crossing only
    const dem = D.demands.map(String);
    const demPerMin = (d) => {
      const r = D.baselines.reservation[d];
      return r ? r.demand_per_min : +d * 240;
    };
    const head = `<thead><tr><th>Verkehr →</th>${dem.map((d) => `<th>${num(demPerMin(d), 0)}/min</th>`).join("")}</tr></thead>`;
    const table = (fmt) => head + groups.map((g) => `<tbody>${groups.length > 1 ? `<tr class="grp"><th colspan="${dem.length + 1}">${WORLD_NAMES[g.w]}</th></tr>` : ""}`
      + g.rows.map((r) => `<tr><td>${r.name}</td>${dem.map((d) => `<td>${r.b && r.b[d] ? fmt(r.b[d]) : "–"}</td>`).join("")}</tr>`).join("") + "</tbody>").join("");
    $("t-thr").innerHTML = table((x) => num(x.throughput_per_min));
    $("t-ok").innerHTML = table((x) => `${pct(x.exit_ok_rate)} <span class="muted">· ${num(x.crash_cars_per_1000, 0)}</span>`);
    // "Ehrlich gesagt": numbers from the bench data, not hard-coded
    const mb = MAIN_ID && pol(MAIN_ID).bench, badPct = (d) => mb && mb[d] ? pct(mb[d].crash_cars_per_1000 / 1000) : null;
    if (badPct("0.15")) $("h-mid").textContent = `rund ${badPct("0.15")}`;
    if (badPct("0.05")) $("h-low").textContent = `rund ${badPct("0.05")}`;
    const pb = D.previous && pol(D.previous) && pol(D.previous).bench["0.15"];
    if (mb && mb["0.15"] && pb) {
      const pd = pol(D.previous).bench["0.3"], md = mb["0.3"];
      $("h-prev").textContent = ` Das vorherige Gehirn hatte bei mittlerem Verkehr noch ${num(pb.crash_cars_per_1000, 0)} Unfälle pro 1000 Autos, das aktuelle ${num(mb["0.15"].crash_cars_per_1000, 0)}`
        + (pd && md ? `, bei dichtem Verkehr ${num(pd.crash_cars_per_1000, 0)} gegen ${num(md.crash_cars_per_1000, 0)}.` : ".")
        + (isSetEntry(pol(MAIN_ID)) ? " Bei dichtem Verkehr verkeilen sich die Autos des aktuellen Gehirns außerdem manchmal mitten auf der Kreuzung." : "");
    }
    if ($("c-thr-note")) $("c-thr-note").textContent = groups.length > 1 ? `Diagramm: ${WORLD_NAMES[mainWorld].replace(/ \(.*/, "")}. In den Tabellen steht jedes Gehirn neben Ampel und Verkehrsrechner auf der Kreuzung, auf der es trainiert wurde.` : "";
    const draw = () => SCX.lineChart($("c-thr"), {
      series: [
        { name: "Nachfrage", color: SCX.cssVar("--c5"), dash: true, points: dem.map((d) => [demPerMin(d), demPerMin(d)]) },
        ...rows.map((r) => ({ name: r.name, color: SCX.cssVar(r.color), dots: true, points: dem.filter((d) => r.b[d]).map((d) => [demPerMin(d), r.b[d].throughput_per_min]) })),
      ],
      xLabel: "ankommende Autos pro Minute (alle 4 Zufahrten)", yLabel: "durchgekommen pro Minute",
      yFmt: (v, tip) => (tip ? num(v) : String(Math.round(v))), xFmt: (v) => String(Math.round(v)), xTipFmt: (v) => `${Math.round(v)} Autos/min kommen an`,
      xMin: 0, yMin: 0, directLabels: true, height: 300, aria: "Durchsatz abhängig von der Nachfrage",
    });
    draw();
    window.addEventListener("resize", debounce(draw, 200));
    $("bench-setup").textContent = `${D.bench_setup.envs} Simulationen × ${Math.round(D.bench_setup.seconds / 60)} Minuten je Wert, Mittel über ${D.bench_setup.seeds} Verkehrsläufe; gleicher Zufallsverkehr für alle (Stand ${D.built}).`;
  }

  function debounce(fn, ms) { let t; return () => { clearTimeout(t); t = setTimeout(fn, ms); }; }

  const init = { start: initStart, sehen: initSehen, tun: initTun, gehirn: initGehirn, lernen: initLernen, vergleich: initVergleich }[page];
  if (init) init();
})();
