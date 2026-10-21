/* Page logic for every page of the site (selected by <body data-page>). */
(function () {
  "use strict";
  const SC = window.SmartCrossing;
  const SCX = window.SCX;
  const D = window.SC_DATA;
  const $ = SCX.$;
  const page = document.body.dataset.page;

  // ------------------------------------------------------------------ data helpers
  const tx = SCX.t, num = SCX.num, pct = SCX.pct;
  const METHOD_NAMES = { es: "Evolution", ppo: "PPO" };
  const pol = (id) => D.policies[id];
  const mainMethod = D.main || (D.best.ppo ? "ppo" : "es");
  const MAIN_ID = D.main_id || D.best[mainMethod];
  const isSetEntry = (e) => !!(e && e.policy && e.policy.arch && e.policy.arch.type === "set");
  const isNarrow = (entry) => !!(entry && entry.physics && entry.physics.W !== undefined && entry.physics.W < 5);

  function genLabel(entry) {
    if (!entry) return "";
    if (entry.id === "zufall") return tx("gen.random");
    const unit = tx(entry.method === "ppo" ? "gen.step" : "gen.generation");
    const where = isNarrow(entry) && !isNarrow(pol(MAIN_ID)) ? tx("gen.narrow") : "";
    return (entry.tlGen !== undefined ? `${unit} ${entry.tlGen}` : `${SCX.tl(entry, "run_label") || entry.run}`) + where;
  }

  /** Default agent types: the main brain (100 %), plus a second run and the old narrow-road
   *  brain for mixing (share 0). */
  function defaultTypes() {
    const types = [];
    const main = pol(MAIN_ID);
    if (main) types.push(SCX.makeType(main, tx("type.current"), { share: 100 }));
    if (D.mix && pol(D.mix)) types.push(SCX.makeType(pol(D.mix), tx("type.second"), { share: 0 }));
    if (D.history && pol(D.history) && D.history !== MAIN_ID) types.push(SCX.makeType(pol(D.history), tx(isNarrow(pol(D.history)) ? "type.oldnarrow" : "type.older"), { share: 0 }));
    if (types.length < 2) {
      const om = mainMethod === "ppo" ? "es" : "ppo";
      if (D.best[om]) types.push(SCX.makeType(pol(D.best[om]), tx("type.best", { m: METHOD_NAMES[om] }), { share: 0 }));
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
        <div class="stat"><div class="k">${tx("stat.thr")}</div><div class="v"><span data-stat="thr">0</span><span class="u">/min</span></div></div>
        <div class="stat"><div class="k">${tx("stat.exits")}</div><div class="v" data-stat="exits">0</div></div>
        <div class="stat bad"><div class="k">${tx("stat.crash")}</div><div class="v" data-stat="crash">0</div></div>
        <div class="stat bad"><div class="k">${tx("stat.off")}</div><div class="v" data-stat="off">0</div></div>
        <div class="stat"><div class="k">${tx("stat.queue")}</div><div class="v"><span data-stat="queue">0</span></div></div>
      </div>`}
      <div class="canvas-box"><canvas aria-label="${tx("stage.aria")}"></canvas>${opts.hint ? `<div class="canvas-hint">${opts.hint}</div>` : ""}</div>
      <div class="transport">
        <button class="btn primary" data-act="play" type="button">${tx("ui.pause")}</button>
        <div class="seg" role="radiogroup" aria-label="${tx("ui.speed")}"></div>
        <button class="btn" data-act="reset" type="button">${tx("ui.restart")}</button>
        <label class="toggle"><input type="checkbox" data-act="trails"> ${tx("ui.trails")}</label>
        ${opts.follow ? `<label class="toggle"><input type="checkbox" data-act="follow"> ${tx("ui.follow")}</label>` : ""}
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

  // ------------------------------------------------------------------ fundamental diagram
  const FD_DRIVERS = [["brain", "drv.policy", "--c4"], ["light", "drv.light", "--c1"], ["reservation", "drv.reservation", "--c2"]];
  const fdCloud = (key) => (D.fundamental && D.fundamental.cloud && D.fundamental.cloud[key]) || [];
  const fdPoints = (key) => ((D.fundamental && D.fundamental.drivers[key]) || []).map((p) => [p.density, p.throughput]);
  /** The point of a curve with the highest throughput ([density, throughput]). */
  const fdPeak = (pts) => pts.reduce((b, p) => (p[1] > b[1] ? p : b), pts[0]);

  /** Fundamental diagram options for SCX.lineChart: one curve per driver (from the origin: no cars, no flow), the
   *  maximum of each ringed. keys: which drivers; compact: small chart for the start page. */
  function fdChartOpts(keys, compact) {
    const used = FD_DRIVERS.filter(([k]) => keys.includes(k) && fdPoints(k).length);
    // compact chart: no label at the ring (it would collide with the live dot); the legend names the maximum
    const peaks = used.map(([k, name, col]) => {
      const pk = fdPeak(fdPoints(k));
      return { x: pk[0], y: pk[1], color: SCX.cssVar(col), label: compact ? null : tx("fd.peak", { n: num(pk[1], 0) }) };
    });
    const allPts = (k) => [...fdPoints(k), ...fdCloud(k)];
    return {
      clouds: used.filter(([k]) => fdCloud(k).length).map(([k, name, col]) => ({ color: SCX.cssVar(col), points: fdCloud(k), r: compact ? 2.2 : 2.4, opacity: compact ? 0.3 : 0.22 })),
      series: used.map(([k, name, col]) => ({ name: tx(name), color: SCX.cssVar(col), dots: !compact, points: [[0, 0], ...fdPoints(k)] })),
      peaks, xMin: 0, yMin: 0, yMax: compact ? Math.max(...used.flatMap(([k]) => allPts(k).map((p) => p[1]))) * 1.1 : undefined, xMax: compact ? Math.ceil(Math.max(...used.flatMap(([k]) => allPts(k).map((p) => p[0]))) * 1.05) : undefined,
      xLabel: tx("fd.xlabel"), yLabel: tx(compact ? "fd.ylabel" : "fd.ylabel.big"),
      yFmt: (v, tip) => (tip ? num(v) : String(Math.round(v))), xFmt: (v) => String(Math.round(v)),
      xTipFmt: (v) => tx("fd.xtip", { n: num(v, 1) }),
      directLabels: !compact, compact, height: compact ? 210 : 300, aria: tx(compact ? "fd.mini.aria" : "fd.aria"),
    };
  }

  /** Small fundamental diagram of the main brain with a dot for the running simulation, same definitions as the
   *  cloud behind the curve (results/fundamental_cloud.json): density = mean number of driving cars on the map
   *  over the last 60 s, throughput = exits of the last 60 s (shorter windows right after the start). The dot is moved by a CSS
   *  transform (transition) a few times per second; the chart itself is only redrawn on resize / theme change. */
  function fdLive(box, readout, cx) {
    if (!D.fundamental || !fdPoints("brain").length) { $("fd-live").hidden = true; return null; }
    let dot = null, trail = null, scales = null, ema = null, lastT = 0, lastUpd = 0;
    const carsHist = []; // [sim time, cars on the map] per update (4/s), the last 60 s
    const hist = []; // last positions (svg px), one per update (4/s): ~5 s of fading trail
    const NS = "http://www.w3.org/2000/svg";
    const place = (cars, thr) => {
      const { X, Y, m, W, H } = scales;
      const x = Math.min(Math.max(X(cars), m.l), W - m.r), y = Math.min(Math.max(Y(thr), m.t), H - m.b);
      dot.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
      dot.style.opacity = "1";
      const lab = dot.lastChild, right = x < W - m.r - 80;
      lab.setAttribute("x", right ? 11 : -11);
      lab.setAttribute("text-anchor", right ? "start" : "end");
      lab.setAttribute("y", y > m.t + 24 ? -11 : 20);
      hist.push([x, y]);
      if (hist.length > 21) hist.shift();
      trail.innerHTML = hist.slice(0, -1).map((p, i, a) => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="${(1.5 + 2 * i / a.length).toFixed(1)}" style="opacity:${(0.5 * (i + 1) / a.length).toFixed(2)}"/>`).join("");
    };
    const draw = () => {
      const o = fdChartOpts(["brain"], true);
      $("fd-max").textContent = tx("fd.max", { n: num(fdPeak(fdPoints("brain"))[1], 0) });
      const svg = SCX.lineChart(box, o);
      scales = svg.scales;
      trail = document.createElementNS(NS, "g");
      trail.setAttribute("class", "live-trail");
      svg.appendChild(trail);
      dot = document.createElementNS(NS, "g");
      dot.setAttribute("class", "live-dot");
      dot.innerHTML = '<circle r="9" class="halo"/><circle r="5.5" class="core"/><text class="live-label"></text>';
      dot.lastChild.textContent = tx("fd.me");
      dot.style.opacity = "0";
      svg.appendChild(dot);
      hist.length = 0;
      if (ema !== null) place(ema.cars, ema.thr);
    };
    const update = (now) => {
      if (now - lastUpd < 250 || !dot) return;
      lastUpd = now;
      const s = cx.statsSummary();
      if (s.t < lastT) { ema = null; carsHist.length = 0; } // simulation restarted
      lastT = s.t;
      carsHist.push([s.t, s.road]);
      while (carsHist.length > 1 && carsHist[0][0] < s.t - 60) carsHist.shift();
      if (s.t - carsHist[0][0] < 5) { readout.textContent = tx("fd.wait"); return; } // at least 5 s of sim time
      ema = { cars: carsHist.reduce((a, p) => a + p[1], 0) / carsHist.length, thr: s.throughput };
      readout.innerHTML = tx("fd.now", { cars: num(ema.cars, 0), thr: num(ema.thr, 0) });
      place(ema.cars, ema.thr);
    };
    draw();
    window.addEventListener("resize", debounce(draw, 200));
    return update;
  }

  /** Data-driven sentence below the big diagram: the capacity of each driver (from the measurement). */
  function fdCaps() {
    const [b, l, r] = ["brain", "light", "reservation"].map((k) => fdPoints(k));
    if (!b.length || !l.length || !r.length) return "";
    const pb = fdPeak(b), pl = fdPeak(l), pr = fdPeak(r), last = b[b.length - 1];
    const rl = r[r.length - 1], rprev = r[r.length - 2];
    return tx("fd.caps", { b: num(pb[1], 0), bd: num(pb[0], 0), l: num(pl[1], 0), ld: num(pl[0], 0), r: num(pr[1], 0), rd: num(pr[0], 0) })
      + (last[1] < 0.95 * pb[1] ? tx("fd.drop", { x: num(last[1], 0), p: pct(1 - last[1] / pb[1]) }) : "")
      + (pr === rl && rprev && rl[1] > 1.02 * rprev[1] ? tx("fd.rise") : "");
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
      $("demand-o").textContent = tx("start.demand", { n: num(c.demand * 240, 0) });
      $("pright-o").textContent = SCX.pctInt(r);
      $("pleft-o").textContent = SCX.pctInt(l);
      $("pstraight").textContent = tx("start.straight", { p: SCX.pctInt(s) });
      $("noise-o").textContent = num(c.noise, 2);
      $("noise").disabled = !c.signal;
      cx.applyControls();
    };
    // start value of the traffic slider: the main brain's capacity (its maximum measured throughput, set in build.py)
    const sd = D.fundamental && D.fundamental.start_demand;
    if (sd) {
      $("demand").value = sd;
      $("demand-opt").textContent = tx("start.demand-opt", { n: num(D.fundamental.max_throughput, 0) });
      $("demand-opt").hidden = false;
    }
    for (const id of ["demand", "pright", "pleft", "noise"]) $(id).addEventListener("input", apply);
    $("sigon").addEventListener("change", apply);
    apply();
    // agent types
    const box = $("types");
    const physKey = (t) => JSON.stringify(t.physics);
    types.forEach((ty, i) => {
      const row = document.createElement("div");
      row.className = "type";
      row.innerHTML = `<span class="swatch" style="border-color:${SCX.TYPE_COLOURS[i]}"></span>
        <div class="tname" title="${ty.id}">${ty.name}</div>
        <div class="row2"><input type="range" min="0" max="100" step="1" value="${ty.share}" aria-label="${tx("type.share", { name: ty.name })}"><output></output>
        <label class="toggle"><input type="checkbox"> ${tx("type.mute")}</label></div>
        <div class="meta count" hidden>${tx("type.count", { exits: 0, bad: 0 })}</div>
        ${physKey(ty) !== physKey(types[0]) ? `<div class="meta">${tx("type.physics", { name: types[0].name })}</div>` : ""}`;
      const rng = row.querySelector("input[type=range]"), cb = row.querySelector("input[type=checkbox]");
      rng.addEventListener("input", () => { ty.share = +rng.value; showShares(); cx.applyControls(); });
      cb.addEventListener("change", () => { ty.mute = cb.checked; });
      box.appendChild(row);
    });
    function showShares() {
      const tot = types.reduce((a, t) => a + t.share, 0);
      box.querySelectorAll(".type").forEach((row, i) => {
        row.querySelector("output").textContent = SCX.pctInt(tot ? Math.round((100 * types[i].share) / tot) : 0);
      });
    }
    showShares();
    // per-brain counts only when more than one brain is driving
    cx.opts.onRefresh = () => {
      const several = types.filter((t) => t.share > 0).length > 1;
      box.querySelectorAll(".type").forEach((row, i) => {
        const p = cx.perType[i], m = row.querySelector(".meta.count");
        m.hidden = !several;
        if (p && several) m.textContent = tx("type.count", { exits: p.exits, bad: p.bad });
      });
    };
    // live point on the fundamental diagram (updated with the stats refresh, throttled inside)
    const fd = fdLive($("fd-mini"), $("fd-now"), cx);
    if (fd) {
      const prev = cx.opts.onRefresh;
      cx.opts.onRefresh = (c) => { prev(c); fd(performance.now()); };
    }
    // honest status line
    const b = pol(MAIN_ID);
    const r15 = b && b.bench["0.15"], l15 = D.baselines.light["0.15"], v15 = D.baselines.reservation["0.15"];
    if (r15) {
      $("status").innerHTML = tx("start.status", {
        dem: num(r15.demand_per_min, 0), ok: pct(r15.exit_ok_rate), thr: num(r15.throughput_per_min),
        light: num(l15.throughput_per_min), ctrl: num(v15.throughput_per_min), prev: prevNote(),
      });
    }
  }

  /** Crashes (+ off-road) per 1000 cars of a bench entry at demand d, or null. */
  const badPer1000 = (e, d) => (e && e.bench[d] ? e.bench[d].crash_cars_per_1000 : null);

  /** "17 statt 105" style comparison of main vs previous brain at demand d (bench data), or "". */
  function crashPair(d) {
    const a = badPer1000(D.previous && pol(D.previous), d), b = badPer1000(pol(MAIN_ID), d);
    return a === null || b === null ? "" : tx("note.pair", { b: num(b, 0), a: num(a, 0) });
  }

  /** Mean throughput (cars/min) of a bench entry at demand d, or null. */
  const thrOf = (e, d) => (e && e.bench[d] ? e.bench[d].throughput_per_min : null);

  /** What is new in the main brain vs the previous one (crash numbers from the bench data). */
  function prevNote() {
    const m = pol(MAIN_ID), mid = crashPair("0.15"), dense = crashPair("0.3");
    const thr = thrOf(m, "0.5"), thrPrev = thrOf(D.previous && pol(D.previous), "0.5");
    if (!mid) return "";
    const a = badPer1000(pol(D.previous), "0.15"), b = badPer1000(m, "0.15");
    const much = tx(a > 0 && b < 0.5 * a ? "note.much" : b < a ? "note.fewer" : "note.notfewer");
    // bench values are means over several traffic seeds, but still a rounded comparison
    return isSetEntry(m)
      ? tx("note.set", {
        K: m.policy.arch.K, much, mid, dense: dense ? tx("note.dense", { dense }) : "",
        thr: num(thr, 0), thrprev: num(thrPrev, 0), dem: num(m.bench["0.5"].demand_per_min, 0),
      })
      : tx("note.mlp", { much, mid });
  }

  // ------------------------------------------------------------------ obs panel ("Was sieht ein Auto?")
  function fmtObs(it, v, compact) {
    if (it.unit === "bool") return tx(v > 0.5 ? "obs.yes" : "obs.no");
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
  const unusedTip = () => tx("obs.unusedtip");

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
      h += `<div class="obs-group"><div class="gname">${g.name}<span class="n">${tx(g.items.length === 1 ? "obs.number" : "obs.numbers", { n: g.items.length })}</span></div>`;
      for (const it of g.items) {
        h += off.has(it.i)
          ? `<div class="orow unused" title="${unusedTip()}"><span class="lab">${it.label}</span><span class="val">${fmtObs(it, o[it.i])}</span><span class="small">${tx("obs.unused")}</span></div>`
          : `<div class="orow"><span class="lab">${it.label}</span><span class="val">${fmtObs(it, o[it.i])}</span>${miniBar(o[it.i])}</div>`;
      }
      h += `</div>`;
    }
    h += `</div>`;
    // neighbours as a table: rows = features, columns = neighbour 1..K
    const nItems = nbs[0].items.length;
    h += `<div class="obs-group" style="margin-top:12px"><div class="gname">${tx("obs.nbs", { n: nbs.length })}<span class="n">${tx("obs.nbs.sub", { n: nItems })}</span></div>
      <table class="ntable"><thead><tr><th></th>${nbs.map((_, k) => `<th>${k + 1}</th>`).join("")}</tr></thead><tbody>`;
    const present = nbs.map((g) => o[g.items.find((it) => it.key.endsWith("_present")).i] > 0.5);
    for (let r = 0; r < nItems; r++) {
      const it0 = nbs[0].items[r];
      const nk = it0.key.split("_").slice(1).join("_");
      const lab = ["fwd", "left", "same", "cross", "speed", "present", "kappa", "acc", "brake", "hazard"].includes(nk) ? tx("obsrow." + nk) : it0.label;
      const unused = off.has(it0.i);
      h += `<tr${unused ? ` class="unused" title="${unusedTip()}"` : ""}><td>${lab}${unused ? ` (${tx("obs.unused")})` : ""}</td>` + nbs.map((g, k) => {
        const it = g.items[r];
        return present[k] ? `<td title="${it.label}">${fmtObs(it, o[it.i], true)}</td>` : `<td class="none">–</td>`;
      }).join("") + `</tr>`;
    }
    const lampIdx = nbs.map((g) => g.items.filter((it) => /_lamp\d$/.test(it.key)).map((it) => it.i));
    h += `<tr><td>${tx("obs.lampcolour")}</td>` + nbs.map((_, k) => present[k]
      ? `<td><span class="lamp" style="background:${cx.sim.cfg.signal_enabled ? SCX.lampCSS(lampIdx[k].length, o[lampIdx[k][0]], o[lampIdx[k][1]]) : "rgb(90,90,96)"}"></span></td>`
      : `<td class="none">–</td>`).join("") + `</tr>`;
    h += `</tbody></table></div>`;
    el.innerHTML = h;
  }

  function initSehen() {
    layoutWarning($("panel"));
    const cx = makeCrossing($("stage"), {
      overlay: "sensors", autoSelect: true, speed: 0.5, speeds: [0.25, 0.5, 1, 2], follow: true,
      hint: tx("hint.tap"),
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
      overlay: "none", autoSelect: true, speed: 0.5, speeds: [0.25, 0.5, 1, 2], follow: true, hint: tx("hint.tap"),
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
        <div class="out"><div class="top-row"><b><i class="line-key" style="border-color:${c1}"></i> ${tx("out.accbrake")}</b><span class="phys">${acc >= 0 ? "+" : "−"}${num(Math.abs(acc))} m/s² ${tx(acc >= 0 ? "out.acc" : "out.brake")}</span></div>
          ${bar(a[0], false, c1)}<div class="bar-ends"><span>${tx("out.fullbrake")}</span><span>${num(a[0], 2)}</span><span>${tx("out.fullacc")}</span></div></div>
        <div class="out"><div class="top-row"><b><i class="line-key" style="border-color:${c2}"></i> ${tx("out.steering")}</b><span class="phys">${radius ? tx("out.curve", { dir: tx(a[1] > 0 ? "man.left" : "man.right"), r: num(radius, 0) }) : tx("man.straight")}</span></div>
          ${bar(a[1], true, c2, sc.kappa_rate < 1e5 ? k : undefined)}<div class="bar-ends"><span>${tx("man.left")}</span><span>${num(a[1], 2)}</span><span>${tx("man.right")}</span></div></div>
        ${a.slice(2).map((v, q) => `<div class="out"><div class="top-row"><b><i class="line-key" style="border-color:${cols()[2 + q]}"></i> ${a.length > 3 ? tx("out.lampch", { n: q + 1 }) : tx("out.lamp")}</b><span class="phys">${tx("out.at1", { what: tx(a.length > 3 ? (q === 0 ? "out.morered" : "out.moreblue") : "out.brighter") })}</span></div>
          ${bar(v, false, cols()[2 + q])}<div class="bar-ends"><span>−1</span><span>${num(v, 2)}</span><span>+1</span></div></div>`).join("")}
        <div class="lampbox"><span class="big" style="background:${lampOn ? SCX.lampCSS(a.length - 2, a[2], a[3]) : "rgb(90,90,96)"}"></span>
          <span>${tx("out.lampbox")}${c.typeOf(n).mute ? tx("out.muted") : ""}${lampOn ? "" : tx("out.unseen")}.</span></div>
        ${sc.kappa_rate < 1e5 ? `<p class="hint">${tx("out.wheelhint", { n: num(sc.kappa_rate / sc.kappa_max, 1) })}</p>` : ""}`;
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
      ctx.fillText("+1", 2, 10); ctx.fillText("−1", 2, H - 3); ctx.textAlign = "right"; ctx.fillText(tx("hist.now"), W - 2, H / 2 - 4); ctx.textAlign = "left"; ctx.fillText(tx("hist.ago"), 22, H / 2 - 4);
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
      <li>${tx("kin.vmax", { kmh: num(phys.v_max * 3.6, 0) })}</li>
      <li>${tx("kin.acc", { acc: num(phys.acc_max, 0), kmh: num(phys.v_max * 3.6, 0), s: num(phys.v_max / phys.acc_max, 1), brake: num(phys.brake_max, 0) })}</li>
      <li>${tx("kin.circle", { r: num(1 / phys.kappa_max, 0) })} ${phys.a_lat_max > 0 ? tx("kin.grip", { lat: num(phys.a_lat_max, 0), kmh: num(phys.v_max * 3.6, 0), d: num(phys.v_max * phys.v_max / phys.a_lat_max, 0) }) : tx("kin.nogrip")}</li>
      <li>${phys.v_reverse > 0 ? tx("kin.reverse", { kmh: num(phys.v_reverse * 3.6, 0) }) : tx("kin.noreverse")}</li>
      ${cfg.kappa_rate ? `<li>${tx("kin.wheel", { s: num((2 * cfg.kappa_max) / cfg.kappa_rate, 1) })}${phys.kappa_rate > 1e5 ? " " + tx("kin.wheelold") : ""}</li>` : ""}
      <li>${tx("kin.size", { len: num(4.6, 1), wid: num(1.8, 1), road: num(2 * (phys.W || cfg.W), 0), room: tx((phys.W || cfg.W) >= 5 ? "kin.two" : "kin.one") })}</li>`;
    window.__sc = cx;
  }

  // ------------------------------------------------------------------ brain
  function initGehirn() {
    layoutWarning($("brainnote"));
    const TIP0 = tx("brain.tip");
    const TIP_SET = tx("brain.tipset");
    let net = null, setv = null, mismatch = false, attn = null;
    const tip = $("tip"), canvas = $("net"), setHost = $("setnet");
    const onHover = (txt) => { tip.textContent = txt || (!setHost.hidden ? TIP_SET : TIP0); };
    // brains to look at: the current one and the previous main brain (other structure: the classic MLP view)
    const picks = [{ id: MAIN_ID, name: tx("type.current") }];
    if (D.previous && pol(D.previous)) picks.push({ id: D.previous, name: tx("type.previous") });
    let pick = 0;
    const stamp = { n: -1, t: -1, uid: -1 };
    const cx = makeCrossing($("stage"), {
      types: [SCX.makeType(pol(picks[0].id), picks[0].name)],
      // the overlay is only drawn while the brain view belongs to the current sim step and car
      overlay: "attention", attention: (n, c) => (attn && setv && stamp.n === n && stamp.t === c.sim.t && stamp.uid === c.sim.uid[n] ? setv.lineInfo() : null),
      onTick: (ts) => { if (setv && !setHost.hidden) setv.animate(ts); },
      autoSelect: true, speed: 0.5, speeds: [0.25, 0.5, 1, 2], hint: tx("hint.tap"),
      onRefresh: (c) => { SCX.renderSelHead(c, $("selhead")); updateBrain(c); },
    });
    // the brain view is updated on every sim step (before drawing), not on the throttled panel refresh,
    // so its numbers, cards and the road overlay always belong to the state that is drawn
    cx.opts.onStepped = updateBrain;
    function updateBrain(c) {
        const n = c.selIndex();
        if (n < 0) { attn = null; return; }
        const ty = c.typeOf(n), D_ = SC.obsDim(c.sim.cfg), lay = SCX.layoutFor(c.sim.cfg) || SCX.localizeLayout(D.layout);
        const ref = SC.forward(ty.policy, c.obs, n * D_);
        let out;
        if (ty.policy.kind === "set") {
          canvas.style.display = "none"; setHost.hidden = false;
          if (!setv) setv = new SCX.SetView(setHost, lay, ty.policy, { onHover, fmt: (it, v) => fmtObs(it, v), onFocus: () => c.draw() });
          else if (setv.pol !== ty.policy) setv.setPolicy(ty.policy, lay);
          const tr = SC.setTrace(ty.policy, c.obs, n * D_);
          setv.update(tr, c.obsRow(n), c.neighbourUids(n, ty.policy.arch.K));
          attn = true;
          out = tr.out;
        } else {
          canvas.style.display = ""; setHost.hidden = true; attn = null;
          if (!net) net = new SCX.NetView(canvas, lay, ty.raw, { onHover: (h) => onHover(h ? net.describe(h) : null) });
          else if (nety.raw !== ty.raw) { net.layout = lay; net.setPolicy(ty.raw); }
          const trace = SCX.forwardTrace(ty.raw, c.obs, n * D_);
          net.update(trace);
          if (net.hoverN) tip.textContent = net.describe(net.hoverN);
          out = trace[trace.length - 1];
        }
        // the drawn forward pass must equal the simulator's (sim.js) forward pass
        const diff = Math.max(...Array.from(out, (v, i) => Math.abs(v - ref[i])));
        if (diff > 1e-9 && !mismatch) {
          mismatch = true;
          console.error("brain forward differs from sim.js forward", diff);
          $("brainnote").insertAdjacentHTML("beforeend", `<p class="note warn">${tx("brain.mismatch", { diff })}</p>`);
        }
        window.__brainDiff = diff;
        stamp.n = n; stamp.t = c.sim.t; stamp.uid = c.sim.uid[n];
    }
    function describeBrain() {
      const p = pol(picks[pick].id);
      if (!p) return;
      const set = isSetEntry(p);
      document.querySelectorAll("[data-set-only]").forEach((e) => { e.hidden = !set; });
      $("set-explain").hidden = !set;
      if (set) {
        const ar = p.policy.arch;
        $("brain-lead").innerHTML = tx("brain.lead.set", { ego: ar.n_ego, K: ar.K, F: ar.F, hidden: ar.hidden.join(" → "), out: ar.n_out });
        $("brain-params").textContent = SCX.group(p.policy.params.length);
        document.querySelectorAll("#set-explain [data-k]").forEach((e) => { e.textContent = ar.K; });
        document.querySelectorAll("#set-explain [data-heads]").forEach((e) => { e.textContent = ar.heads; });
      } else {
        const s = p.policy.sizes, params = s.slice(1).reduce((a, n, i) => a + s[i] * n + n, 0);
        $("brain-lead").innerHTML = tx("brain.lead.mlp", {
          intro: tx(pick > 0 ? "brain.intro.prev" : "brain.intro.cur"),
          sizes: tx("brain.sizes", { in: s[0], hidden: s.slice(1, -1).join(" → "), out: s[s.length - 1] }), n: s.length - 2 });
        $("brain-params").textContent = SCX.group(params);
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
    const bigBtn = $("brain-big"), lay = $("brain-layout");
    if (bigBtn && lay) bigBtn.addEventListener("click", () => {
      const big = lay.classList.toggle("big");
      bigBtn.textContent = tx(big ? "brain.small" : "brain.large");
      bigBtn.setAttribute("aria-pressed", String(big));
    });
    describeBrain();
    window.__sc = cx;
  }

  // ------------------------------------------------------------------ learning
  // ------------------------------------------------------------------ reward scale (lernen)
  /** Draws the reward rules of the main brain (D.reward, from its training cfg) as a point scale + two example trips. */
  function drawReward() {
    const R = D.reward, host = $("reward");
    if (!R || !host) return;
    const f = (v, d = 2) => SCX.signed(v, d);
    const IMPACT = 6;                                   // example impact speed (m/s)
    const kmh = (v) => Math.round(v * 3.6);             // speeds are shown in km/h
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
    const hard = (label, at) => row(label, tx("rw.hard", { a: f(at(0), 1), kmh: kmh(vTop), b: f(at(vTop), 1) }),
      bar(0, at(0), "neg") + bar(at(0), at(vTop), "neg soft"), f(at(IMPACT), 1) + "*", "neg");
    const rows = [
      row(tx("rw.exit"), "", bar(0, R.r_exit, "pos"), f(R.r_exit, 1), "pos"),
      row(tx("rw.progress"), tx("rw.progress.sub", { v: f(R.r_progress, 2) }), bar(0, R.r_progress * 100, "help"), tx("rw.per100", { v: f(R.r_progress * 100, 1) }), "pos"),
      row(tx("rw.time"), tx("rw.time.sub"), bar(0, -R.r_time * 10, "neg"), tx("rw.per10s", { v: f(-R.r_time * 10, 1) }), "neg"),
      row(tx("rw.wrong"), "", bar(0, R.r_wrong_exit, "neg"), f(R.r_wrong_exit, 1), "neg"),
      row(tx("rw.timeout"), tx("rw.timeout.sub", { n: Math.round(R.max_age) }), bar(0, R.r_timeout, "neg"), f(R.r_timeout, 1), "neg"),
      crashSame ? hard(tx("rw.crashoff"), crashAt) : hard(tx("rw.crash"), crashAt) + hard(tx("rw.off"), offAt),
    ];
    // example trips (all numbers derived from the same rules)
    const tSec = route / R.spawn_speed;
    const trip = (title, lines) => {
      const tot = lines.reduce((a, l) => a + l[1], 0);
      return `<div class="card"><h3>${title}</h3>${lines.map((l) => `<div class="ln"><span>${l[0]}</span><span class="${l[1] >= 0 ? "pos" : "neg"}">${f(l[1])}</span></div>`).join("")}
        <div class="ln tot"><span>${tx("rw.total")}</span><span class="${tot >= 0 ? "pos" : "neg"}">${f(tot)}</span></div></div>`;
    };
    const half = 0.5;
    const ex = trip(tx("rw.ex1"), [
      [tx("rw.ex.progress"), R.r_progress * route],
      [tx("rw.exit"), R.r_exit],
      [tx("rw.ex.time", { n: Math.round(tSec) }), -R.r_time * tSec],
    ]) + trip(tx("rw.ex2", { kmh: kmh(IMPACT) }), [
      [tx("rw.ex.half"), R.r_progress * route * half],
      [tx("rw.ex.crash", { kmh: kmh(IMPACT) }), crashAt(IMPACT)],
      [tx("rw.ex.time", { n: Math.round(tSec * half) }), -R.r_time * tSec * half],
    ]);
    host.innerHTML = `${rows.join("")}<div class="rw-ex">${ex}</div>`;
    $("reward-note").innerHTML = tx("rw.note");
  }

  function initLernen() {
    drawReward();
    const methods = Object.keys(D.timelines).sort((a, b) => (a === "es" ? -1 : 1) - (b === "es" ? -1 : 1));
    let method = methods.includes(mainMethod) ? mainMethod : methods[0];
    let idx = 0, extra = null; // extra: a dead-end run shown instead of the timeline
    if (!method) { $("tt").innerHTML = `<p class="muted">${tx("learn.nodata")}</p>`; return; }
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
      const note = extra ? SCX.tl(extra, "note") : p.id === "zufall" ? tx("learn.randomnote") : chain && SCX.tl(chain, "note");
      const b05 = p.bench["0.05"], b15 = p.bench["0.15"];
      $("tt-now").innerHTML = `<b>${extra ? (SCX.tl(extra, "label") || extra.run) : genLabel(p)}</b>${chain && !extra ? ` <span class="muted small">(${tx("learn.section", { n: `<span class="num-inline">${phase + 1}</span>` })})</span>` : ""}<br>
        <span class="small">${note || ""}</span><br>
        <span class="small muted">${tx("learn.measured", { low: pct(b05 && b05.exit_ok_rate), mid: pct(b15 && b15.exit_ok_rate), thr: num(b15 && b15.throughput_per_min) })}</span>`;
      cx.seed = 3; // same traffic for every checkpoint
      cx.setTypes([SCX.makeType(p, extra ? (SCX.tl(extra, "label") || extra.run) : genLabel(p))]);
      drawCurves();
    }
    // dead-end runs
    const ow = $("tt-others");
    function setupOthers() {
      ow.innerHTML = "";
      const list = (D.others || []).filter((o) => (o.method || "es") === method);
      for (const o of list) {
        const b = document.createElement("button");
        b.type = "button"; b.className = "btn"; b.textContent = SCX.tl(o, "label") || o.run;
        b.title = SCX.tl(o, "note") || "";
        b.addEventListener("click", () => { extra = Object.assign({}, o, { note: o.note || tx("learn.othernote"), note_en: o.note_en || tx("learn.othernote") }); show(); });
        ow.appendChild(b);
      }
      ow.parentElement.style.display = list.length ? "" : "none";
    }

    function drawCurves() {
      const curves = D.curves[method] || {};
      const chain = (D.chains[method] || []).filter((c) => (curves[c.run] || []).length);
      if (!chain.length) { $("curves").innerHTML = `<p class="muted">${tx("learn.nolog")}</p>`; return; }
      // phases = runs of the warm-start chain; each starts at the first logged point of its run
      // (iteration counts continue across warm starts) and is cut where the next one starts
      const starts = chain.map((c) => Math.min(...curves[c.run].map((p) => p.x)));
      const part = (i) => curves[chain[i].run].filter((p) => i + 1 >= chain.length || p.x < starts[i + 1]);
      const total = chain.reduce((a, _, i) => a + part(i).length, 0);
      const k = Math.max(1, Math.round(total / 80));
      const ser = (key) => chain.flatMap((_, i) => SCX.smooth(part(i).filter((p) => p[key] !== undefined).map((p) => [p.x, p[key]]), k));
      const phases = chain.map((c, i) => ({ x: starts[i], n: i + 1, title: SCX.tl(c, "step") || SCX.tl(c, "note") || SCX.tl(c, "label") || c.run }));
      const WORLD = { narrow: tx("world.narrow"), wide: tx("world.wide") };
      const worlds = new Set(chain.map((c) => c.world).filter(Boolean));
      const bands = [];
      if (worlds.size > 1) chain.forEach((c, i) => {
        if (i === 0 || c.world !== chain[i - 1].world) bands.push({ x0: starts[i], x1: Infinity, label: { narrow: tx("world.narrow.short"), wide: tx("world.wide.short") }[c.world] || "", world: c.world });
        else return;
        if (bands.length > 1) bands[bands.length - 2].x1 = starts[i];
      });
      const lastX = Math.max(...chain.flatMap((_, i) => part(i).map((p) => p.x)));
      bands.forEach((b) => { if (b.x1 === Infinity) b.x1 = lastX; });
      const leg = $("chain-legend");
      if (leg) leg.innerHTML = chain.map((c, i) => `${worlds.size > 1 && (i === 0 || c.world !== chain[i - 1].world) ? `<li class="world">${WORLD[c.world] || ""}</li>` : ""}<li><span class="num">${i + 1}</span><span>${SCX.tl(c, "step") || SCX.tl(c, "note") || SCX.tl(c, "label") || c.run}</span></li>`).join("");
      const markers = [];
      if (!extra) {
        const cur = tl()[idx];
        if (cur) markers.push({ x: cur.tlGen, label: tx("learn.marker") });
      }
      SCX.lineChart($("curves"), {
        series: [
          { name: tx("curve.exit"), color: SCX.cssVar("--c1"), points: ser("exit"), tipMax: 20 },
          { name: tx("curve.crash"), color: SCX.cssVar("--c2"), points: ser("crash"), tipMax: 20 },
          { name: tx("curve.off"), color: SCX.cssVar("--c3"), points: ser("offroad"), tipMax: 20 },
        ],
        yMin: 0, yMax: 1, yFmt: (v) => SCX.pctInt(Math.round(v * 100)), xLabel: tx(method === "ppo" ? "gen.step" : "gen.generation"),
        xTipFmt: (x) => `${tx(method === "ppo" ? "gen.step.short" : "gen.generation")} ${Math.round(x)}`,
        markers, phases, bands, directLabels: true, height: 300, aria: tx("curve.aria"),
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
    if (!R || !R.runs) { host.innerHTML = `<p class="muted">${tx("rp.none")}</p>`; return; }
    const DRIVERS = [["light", tx("drv.light")], ["reservation", tx("drv.reservation")], ["policy", tx("drv.policy")]].filter(([k]) => R.runs[k]);
    const LEVELS = [tx("rp.low"), tx("rp.mid"), tx("rp.high")];
    const dems = R.demands.map(String);
    const world = Object.assign({ L: 50, W: 3.5, Rc: 6 }, R.world);
    host.innerHTML = `
      <div class="rp-ctl">
        <div class="seg" role="radiogroup" aria-label="${tx("rp.traffic")}">${dems.map((d, i) => `<input type="radio" name="rp-dem" id="rp-d${i}" value="${d}"${i === 1 ? " checked" : ""}><label for="rp-d${i}">${LEVELS[i] || d}</label>`).join("")}</div>
        <button class="btn primary" type="button" data-act="play">${tx("ui.pause")}</button>
        <div class="seg" role="radiogroup" aria-label="${tx("ui.speed")}">${[1, 2, 4].map((s) => `<input type="radio" name="rp-spd" id="rp-s${s}" value="${s}"${s === 1 ? " checked" : ""}><label for="rp-s${s}">${s}×</label>`).join("")}</div>
        <button class="btn" type="button" data-act="restart">${tx("rp.restart")}</button>
        <span class="clock" data-act="clock"></span>
      </div>
      <div class="rp-grid">${DRIVERS.map(([k, name]) => `
        <figure class="rp" data-k="${k}"><div class="rp-box"><canvas aria-label="${tx("rp.canvas", { name })}"></canvas></div>
          <figcaption><div class="rp-name">${name}</div>
          <div class="rp-stats"><span><b data-s="thr">–</b> ${tx("rp.carsmin")}</span><span class="bad"><b data-s="bad">0</b> ${tx("rp.crashes")}</span></div></figcaption></figure>`).join("")}
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
    play.addEventListener("click", () => { running = !running; play.textContent = tx(running ? "ui.pause" : "ui.resume"); lastTs = null; });
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
    const WORLD_NAMES = { wide: tx("world.wide"), narrow: tx("world.narrow") };
    const byWorld = D.baselines_by_world || { [D.main_world || worldOf(pol(MAIN_ID))]: D.baselines };
    const mainWorld = MAIN_ID ? worldOf(pol(MAIN_ID)) : D.main_world || "wide";
    const brains = [];
    if (MAIN_ID) brains.push({ name: tx("drv.policy"), e: pol(MAIN_ID), color: "--c4" });
    if (D.previous && pol(D.previous)) brains.push({ name: tx("type.previous"), e: pol(D.previous), color: "--c3" });
    if (D.history && D.history !== MAIN_ID) brains.push({ name: tx(worldOf(pol(D.history)) === mainWorld ? "drv.older" : "drv.ppo"), e: pol(D.history), color: "--c3" });
    if (D.best.es && D.best.es !== MAIN_ID) brains.push({ name: METHOD_NAMES.es, e: pol(D.best.es), color: "--c5" });
    const worlds = [mainWorld, ...Object.keys(byWorld).filter((w) => w !== mainWorld)];
    const groups = worlds.map((w) => {
      const base = byWorld[w];
      const rows = base ? [
        { name: tx("drv.light"), b: base.light, color: "--c1", base: true },
        { name: tx("drv.reservation"), b: base.reservation, color: "--c2", base: true },
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
    const head = `<thead><tr><th>${tx("tbl.traffic")}</th>${dem.map((d) => `<th>${num(demPerMin(d), 0)}/min</th>`).join("")}</tr></thead>`;
    const table = (fmt) => head + groups.map((g) => `<tbody>${groups.length > 1 ? `<tr class="grp"><th colspan="${dem.length + 1}">${WORLD_NAMES[g.w]}</th></tr>` : ""}`
      + g.rows.map((r) => `<tr><td>${r.name}</td>${dem.map((d) => `<td>${r.b && r.b[d] ? fmt(r.b[d]) : "–"}</td>`).join("")}</tr>`).join("") + "</tbody>").join("");
    $("t-thr").innerHTML = table((x) => num(x.throughput_per_min));
    $("t-ok").innerHTML = table((x) => `${pct(x.exit_ok_rate)} <span class="muted">· ${num(x.crash_cars_per_1000, 0)}</span>`);
    // "Ehrlich gesagt": numbers from the bench data, not hard-coded
    const mb = MAIN_ID && pol(MAIN_ID).bench, badPct = (d) => mb && mb[d] ? num(mb[d].crash_cars_per_1000, 0) : null;
    if (badPct("0.15")) $("h-mid").textContent = tx("cmp.about", { v: badPct("0.15") });
    if (badPct("0.05")) $("h-low").textContent = tx("cmp.about", { v: badPct("0.05") });
    const pb = D.previous && pol(D.previous) && pol(D.previous).bench["0.15"];
    if (mb && mb["0.15"] && pb) {
      const pd = pol(D.previous).bench["0.3"], md = mb["0.3"];
      $("h-prev").textContent = " " + tx("cmp.prev", { prev: num(pb.crash_cars_per_1000, 0), cur: num(mb["0.15"].crash_cars_per_1000, 0) })
        + (pd && md ? tx("cmp.prevdense", { prev: num(pd.crash_cars_per_1000, 0), cur: num(md.crash_cars_per_1000, 0) }) : ".");
    }
    const m5 = mb && mb["0.5"], p5 = D.previous && pol(D.previous) && pol(D.previous).bench["0.5"];
    const fb = D.fundamental && D.fundamental.drivers.brain, f5 = fb && fb.find((q) => q.demand === 0.5);
    if (m5 && $("h-dcrash")) {
      $("h-ddem").textContent = num(m5.demand_per_min, 0);
      $("h-dcrash").textContent = num(m5.crash_cars_per_1000, 0);
      $("h-dthr").textContent = num(m5.throughput_per_min, 0);
      if (p5) $("h-dprev").textContent = tx("cmp.against", { v: num(p5.throughput_per_min, 0) });
      if (f5) $("h-djam").textContent = pct(f5.deadlock_time_frac);
    }
    if ($("c-thr-note")) $("c-thr-note").textContent = groups.length > 1 ? tx("cmp.chartnote", { world: WORLD_NAMES[mainWorld].replace(/ \(.*/, "") }) : "";
    const draw = () => SCX.lineChart($("c-thr"), {
      series: [
        { name: tx("chart.demand"), color: SCX.cssVar("--c5"), dash: true, points: dem.map((d) => [demPerMin(d), demPerMin(d)]) },
        ...rows.map((r) => ({ name: r.name, color: SCX.cssVar(r.color), dots: true, points: dem.filter((d) => r.b[d]).map((d) => [demPerMin(d), r.b[d].throughput_per_min]) })),
      ],
      xLabel: tx("chart.xlabel"), yLabel: tx("chart.ylabel"),
      yFmt: (v, tip) => (tip ? num(v) : String(Math.round(v))), xFmt: (v) => String(Math.round(v)), xTipFmt: (v) => tx("chart.xtip", { n: Math.round(v) }),
      xMin: 0, yMin: 0, directLabels: true, height: 300, aria: tx("chart.aria"),
    });
    draw();
    window.addEventListener("resize", debounce(draw, 200));
    if (D.fundamental && $("c-fd")) {
      const drawFd = () => SCX.lineChart($("c-fd"), fdChartOpts(["brain", "light", "reservation"], false));
      $("fd-caps").innerHTML = fdCaps();
      drawFd();
      window.addEventListener("resize", debounce(drawFd, 200));
    } else if ($("fundamental")) {
      for (const id of ["fundamental", "c-fd", "fd-caps"]) $(id).hidden = true;
    }
    $("bench-setup").textContent = tx("cmp.setup", { envs: D.bench_setup.envs, min: Math.round(D.bench_setup.seconds / 60), seeds: D.bench_setup.seeds, built: D.built });
  }

  function debounce(fn, ms) { let t; return () => { clearTimeout(t); t = setTimeout(fn, ms); }; }

  const init = { start: initStart, sehen: initSehen, tun: initTun, gehirn: initGehirn, lernen: initLernen, vergleich: initVergleich }[page];
  if (init) init();
})();
