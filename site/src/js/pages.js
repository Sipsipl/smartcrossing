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

  function genLabel(entry) {
    if (!entry) return "";
    if (entry.id === "zufall") return "Start: zufälliges Gehirn";
    const unit = entry.method === "ppo" ? "Trainingsschritt" : "Generation";
    return entry.tlGen !== undefined ? `${unit} ${entry.tlGen}` : `${entry.run}`;
  }

  /** Default agent types: best policy (+ the other method or a beginner, share 0). */
  function defaultTypes() {
    const types = [];
    const main = pol(D.best[mainMethod]);
    if (main) types.push(SCX.makeType(main, `${METHOD_NAMES[mainMethod]} (bestes Gehirn)`, { share: 100 }));
    const om = mainMethod === "ppo" ? "es" : "ppo";
    const other = D.best[om] ? pol(D.best[om]) : null;
    if (other) types.push(SCX.makeType(other, `${METHOD_NAMES[om]} (bestes Gehirn)`, { share: 0 }));
    else {
      const tl = (D.timelines[mainMethod] || []).filter((p) => p.id !== "zufall" && p.id !== D.best[mainMethod]);
      if (tl.length) types.push(SCX.makeType(pol(tl[0].id), `Anfänger (Gen. ${tl[0].gen})`, { share: 0 }));
    }
    return types;
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
        <label class="toggle"><input type="checkbox" data-act="trails"> Spuren</label>
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
      if (opts.onRefresh) opts.onRefresh(cx);
    };
    cx.opts.onFrame = () => refresh(false);
    cx.onChange(() => refresh(true));
    const hint = host.querySelector(".canvas-hint");
    if (hint) cx.opts.onPick = () => { hint.style.opacity = "0"; };
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
    const c = cx.controls;
    const apply = () => {
      c.demand = +$("demand").value;
      let r = +$("pright").value, l = +$("pleft").value;
      if (r + l > 100) { l = 100 - r; $("pleft").value = l; }
      const s = 100 - r - l;
      c.turn = [r, s, l];
      c.signal = $("sigon").checked;
      c.noise = +$("noise").value;
      $("demand-o").textContent = `${num(c.demand * 60, 0)} Autos/min`;
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
        <div class="meta">0 angekommen · 0 verunglückt</div>
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
    cx.opts.onRefresh = () => {
      box.querySelectorAll(".type").forEach((row, i) => {
        const p = cx.perType[i];
        if (p) row.querySelector(".meta").textContent = `${p.exits} angekommen · ${p.bad} verunglückt`;
      });
    };
    // honest status line
    const b = pol(D.best[mainMethod]);
    const r15 = b && b.bench["0.15"], l15 = D.baselines.light["0.15"], v15 = D.baselines.reservation["0.15"];
    if (r15) {
      $("status").innerHTML = `Bei mittlerem Verkehr (${num(r15.demand_per_min, 0)} Autos pro Minute) schaffen es die besten gelernten Autos
        zu <b>${pct(r15.exit_ok_rate)}</b> heil ans Ziel – <b>${num(r15.throughput_per_min)}</b> Autos pro Minute.
        Eine Ampel schafft ${num(l15.throughput_per_min)} pro Minute, ein zentraler Verkehrsrechner ${num(v15.throughput_per_min)} – beide ohne einen einzigen Unfall.
        Die Autos lernen also noch. <a href="vergleich.html">Zum Vergleich →</a>`;
    }
  }

  // ------------------------------------------------------------------ obs panel ("Was sieht ein Auto?")
  function fmtObs(it, v) {
    if (it.unit === "bool") return v > 0.5 ? "ja" : "nein";
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

  function renderObsPanel(cx, el) {
    const n = cx.selIndex();
    if (n < 0) { el.innerHTML = ""; return; }
    const o = cx.obsRow(n), lay = D.layout;
    const own = lay.groups.filter((g) => g.sub !== "neighbour");
    const nbs = lay.groups.filter((g) => g.sub === "neighbour");
    let h = `<div class="obs-grid">`;
    for (const g of own) {
      h += `<div class="obs-group"><div class="gname">${g.name}<span class="n">${g.items.length} ${g.items.length === 1 ? "Zahl" : "Zahlen"}</span></div>`;
      for (const it of g.items) {
        h += `<div class="orow"><span class="lab">${it.label}</span><span class="val">${fmtObs(it, o[it.i])}</span>${miniBar(o[it.i])}</div>`;
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
      const lab = nbs[0].items[r].label;
      h += `<tr><td>${lab}</td>` + nbs.map((g, k) => {
        const it = g.items[r];
        return present[k] ? `<td>${fmtObs(it, o[it.i])}</td>` : `<td class="none">–</td>`;
      }).join("") + `</tr>`;
    }
    const lampIdx = nbs.map((g) => g.items.filter((it) => /_lamp\d$/.test(it.key)).map((it) => it.i));
    h += `<tr><td>Lampenfarbe</td>` + nbs.map((_, k) => present[k]
      ? `<td><span class="lamp" style="background:${cx.sim.cfg.signal_enabled ? SCX.lampRGB(o[lampIdx[k][0]], o[lampIdx[k][1]]) : "rgb(90,90,96)"}"></span></td>`
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
    $("dim-count").textContent = D.layout.dim;
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
        ${[0, 1].map((q) => `<div class="out"><div class="top-row"><b><i class="line-key" style="border-color:${cols()[2 + q]}"></i> Lampe Kanal ${q + 1}</b><span class="phys">${q === 0 ? "mehr Rot" : "mehr Blau"} bei +1</span></div>
          ${bar(a[2 + q], false, q === 0 ? "rgb(220,60,40)" : "rgb(40,70,220)")}<div class="bar-ends"><span>−1</span><span>${num(a[2 + q], 2)}</span><span>+1</span></div></div>`).join("")}
        <div class="lampbox"><span class="big" style="background:${lampOn ? SCX.lampRGB(a[2], a[3]) : "rgb(90,90,96)"}"></span>
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
      for (let q = 0; q < 4; q++) {
        ctx.strokeStyle = c[q]; ctx.lineWidth = 2; ctx.beginPath();
        hist.forEach((a, i) => {
          const x = W - (hist.length - 1 - i) * (W / 99), y = H / 2 - a[q] * (H / 2 - 4);
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        });
        ctx.stroke();
      }
    }
    // kinematics facts from the sim config
    const p = pol(D.best[mainMethod]);
    const phys = Object.assign({}, cfg, p ? p.physics : {});
    $("kin").innerHTML = `
      <li>Höchstens <b>${num(phys.v_max * 3.6, 0)} km/h</b> (${num(phys.v_max, 0)} m/s) – wie in der Stadt.</li>
      <li>Beschleunigen mit bis zu <b>${num(phys.acc_max, 0)} m/s²</b> (von 0 auf 50 in etwa ${num(phys.v_max / phys.acc_max, 1)} s), bremsen mit bis zu <b>${num(phys.brake_max, 0)} m/s²</b> – das ist eine kräftige Vollbremsung.</li>
      <li>Engster Kreis: <b>${num(1 / phys.kappa_max, 0)} m Radius</b>. Rückwärts fahren geht nicht.</li>
      ${cfg.kappa_rate ? `<li>Das Lenkrad dreht sich nicht beliebig schnell: von ganz links nach ganz rechts dauert es ${num((2 * cfg.kappa_max) / cfg.kappa_rate, 1)} s.${phys.kappa_rate > 1e5 ? " (Die bisher besten Evolutions-Autos wurden noch ohne diese Grenze trainiert und fahren hier deshalb mit sofort wirkender Lenkung.)" : ""}</li>` : ""}
      <li>Das Auto ist ${num(4.6, 1)} m lang und ${num(1.8, 1)} m breit; die Straße ist ${num(2 * cfg.W, 0)} m breit – Platz für genau ein Auto pro Richtung.</li>`;
    window.__sc = cx;
  }

  // ------------------------------------------------------------------ brain
  function initGehirn() {
    layoutWarning($("brainnote"));
    let net = null, mismatch = false;
    const tip = $("tip");
    const cx = makeCrossing($("stage"), {
      overlay: "none", autoSelect: true, speed: 0.5, speeds: [0.25, 0.5, 1, 2], hint: "Tippe ein Auto an",
      onRefresh: (c) => {
        SCX.renderSelHead(c, $("selhead"));
        const n = c.selIndex();
        if (n < 0) return;
        const t = c.typeOf(n);
        if (!net || net.raw !== t.raw) {
          if (!net) net = new SCX.NetView($("net"), D.layout, t.raw, { onHover: (h) => { tip.textContent = h ? net.describe(h) : "Fahre mit der Maus über einen Kreis (oder tippe darauf)."; } });
          else net.setPolicy(t.raw);
        }
        const D_ = SC.obsDim(c.sim.cfg);
        const trace = SCX.forwardTrace(t.raw, c.obs, n * D_);
        // our forward must equal the simulator's (sim.js) forward pass
        const ref = SC.forward(t.policy, c.obs, n * D_);
        const out = trace[trace.length - 1];
        const diff = Math.max(...Array.from(out, (v, i) => Math.abs(v - ref[i])));
        if (diff > 1e-9 && !mismatch) {
          mismatch = true;
          console.error("brain forward differs from sim.js forward", diff);
          $("brainnote").insertAdjacentHTML("beforeend", `<p class="note warn">Achtung: Die Anzeige weicht vom Simulator ab (${diff}).</p>`);
        }
        window.__brainDiff = diff;
        net.update(trace);
        if (net.hoverN) tip.textContent = net.describe(net.hoverN);
      },
    });
    const p = pol(D.best[mainMethod]);
    if (p) {
      const s = p.policy.sizes, params = s.slice(1).reduce((a, n, i) => a + s[i] * n + n, 0);
      $("brain-sizes").textContent = `${s[0]} Eingänge → ${s[1]} → ${s[2]} → ${s[3]} Ausgänge`;
      $("brain-params").textContent = params.toLocaleString("de-DE");
    }
    window.__sc = cx;
  }

  // ------------------------------------------------------------------ learning
  function initLernen() {
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
    }
    slider.addEventListener("input", () => { idx = +slider.value; extra = null; show(); });
    function show() {
      const p = extra ? Object.assign({}, pol(extra.id)) : tl()[idx];
      $("tt-ticks").querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(!extra && i === idx)));
      const chain = (D.chains[method] || []).find((c) => c.run === p.run);
      const note = extra ? extra.note : p.id === "zufall" ? "Vor dem Training: alle Gewichte sind Zufallszahlen. Die Autos fahren irgendwie los." : chain && chain.note;
      const b05 = p.bench["0.05"], b15 = p.bench["0.15"];
      $("tt-now").innerHTML = `<b>${extra ? `Irrweg ${extra.run}` : genLabel(p)}</b>${p.run && !extra ? ` <span class="muted small">(Lauf ${p.run})</span>` : ""}<br>
        <span class="small">${note || ""}</span><br>
        <span class="small muted">Gemessen in 8 × 2 Minuten Simulation: bei wenig Verkehr kommen <b>${pct(b05 && b05.exit_ok_rate)}</b> der Autos an,
        bei mittlerem Verkehr <b>${pct(b15 && b15.exit_ok_rate)}</b> (${num(b15 && b15.throughput_per_min)} Autos/min).</span>`;
      cx.seed = 3; // same traffic for every checkpoint
      cx.setTypes([SCX.makeType(p, extra ? extra.run : genLabel(p))]);
      drawCurves();
    }
    // dead-end runs
    const ow = $("tt-others");
    for (const o of D.others || []) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "btn"; b.textContent = `Irrweg ${o.run}`;
      b.title = o.note || "";
      b.addEventListener("click", () => { extra = o; show(); });
      ow.appendChild(b);
    }
    if (!(D.others || []).length) ow.parentElement.style.display = "none";

    function drawCurves() {
      const curves = D.curves[method] || {};
      const runs = Object.keys(curves);
      const pts = runs.flatMap((r) => curves[r]);
      if (!pts.length) { $("curves").innerHTML = `<p class="muted">Keine Logdaten.</p>`; return; }
      const k = Math.max(1, Math.round(pts.length / 80));
      const ser = (key) => runs.flatMap((r) => SCX.smooth(curves[r].filter((p) => p[key] !== undefined).map((p) => [p.x, p[key]]), k));
      const chain = D.chains[method] || [];
      const markers = chain.filter((c) => c.offset > 0).map((c) => ({ x: c.offset, label: c.run }));
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
        markers, directLabels: true, height: 280, aria: "Lernkurve: Anteil angekommener, verunglückter und abgekommener Autos",
      });
    }
    setupSlider();
    show();
    window.addEventListener("resize", debounce(drawCurves, 200));
    window.__sc = cx;
  }

  // ------------------------------------------------------------------ comparison
  function initVergleich() {
    const rows = [
      { name: "Ampel", note: "Regeln + Spuren", b: D.baselines.light, color: "--c1" },
      { name: "Verkehrsrechner", note: "Regeln + Spuren + Funk", b: D.baselines.reservation, color: "--c2" },
    ];
    if (D.best.es) rows.push({ name: "Evolution (gelernt)", note: "keine Regeln", b: pol(D.best.es).bench, color: "--c3" });
    if (D.best.ppo) rows.push({ name: "PPO (gelernt)", note: "keine Regeln", b: pol(D.best.ppo).bench, color: "--c4" });
    const dem = D.demands.map(String);
    const demPerMin = (d) => {
      const r = D.baselines.reservation[d];
      return r ? r.demand_per_min : +d * 240;
    };
    const head = `<thead><tr><th>Verkehr →</th>${dem.map((d) => `<th>${num(demPerMin(d), 0)}/min</th>`).join("")}</tr></thead>`;
    $("t-thr").innerHTML = head + "<tbody>" + rows.map((r) => `<tr><td>${r.name}</td>${dem.map((d) => `<td>${r.b[d] ? num(r.b[d].throughput_per_min) : "–"}</td>`).join("")}</tr>`).join("") + "</tbody>";
    $("t-ok").innerHTML = head + "<tbody>" + rows.map((r) => `<tr><td>${r.name}</td>${dem.map((d) => `<td>${r.b[d] ? pct(r.b[d].exit_ok_rate) : "–"}</td>`).join("")}</tr>`).join("") + "</tbody>";
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
    $("bench-setup").textContent = `${D.bench_setup.envs} Simulationen × ${Math.round(D.bench_setup.seconds / 60)} Minuten je Wert, gleicher Zufallsverkehr für alle (Stand ${D.built}).`;
  }

  function debounce(fn, ms) { let t; return () => { clearTimeout(t); t = setTimeout(fn, ms); }; }

  const init = { start: initStart, sehen: initSehen, tun: initTun, gehirn: initGehirn, lernen: initLernen, vergleich: initVergleich }[page];
  if (init) init();
})();
