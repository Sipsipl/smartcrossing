/* Smart Crossing viewer: UI, rendering and the simulation loop around sim.js. */
(function () {
  "use strict";
  const SC = window.SmartCrossing;
  const $ = (id) => document.getElementById(id);

  const MAN_COLOURS = { 1: "#4287f5", 2: "#f0f0f0", 3: "#f5a623" };
  const TYPE_COLOURS = ["#22d3ee", "#ef4fd0", "#a3e635"];
  const SCENE = { verge: "#3b5640", road: "#46484d", edge: "#8d938f", gate: "rgba(255,255,255,0.35)", text: "rgba(255,255,255,0.8)" };
  const MAX_TYPES = 3;
  const WINDOW_S = 60; // window for throughput and delay

  // ------------------------------------------------------------------ state
  const types = []; // {name, source, placeholder, policy (loaded), share, mute, exits, bad, error}
  for (const t of (window.SC_POLICIES || []).slice(0, MAX_TYPES)) addType(t);
  if (!types.length) types.push(null);
  while (types.length < MAX_TYPES) types.push(null);

  function addType(t, slot) {
    const entry = { name: t.name || "Policy", source: t.source || "", placeholder: !!t.placeholder, share: 50, mute: false };
    try {
      entry.policy = SC.loadPolicy(t.policy || t);
      const cfg = SC.defaultConfig();
      if (entry.policy.sizes[0] !== SC.obsDim(cfg) || entry.policy.sizes[entry.policy.sizes.length - 1] !== SC.actDim(cfg)) {
        throw new Error(`network expects ${entry.policy.sizes[0]} inputs; the simulator gives ${SC.obsDim(cfg)}`);
      }
    } catch (e) {
      entry.error = String(e.message || e);
      entry.policy = null;
    }
    if (slot === undefined) types.push(entry);
    else types[slot] = entry;
    return entry;
  }

  let seed = 1;
  let sim, crashMarks, trails, perType;
  let running = true, speed = 1, acc = 0, lastTs = null;

  function activeTypes() {
    return types.map((t, i) => (t && t.policy ? i : -1)).filter((i) => i >= 0);
  }

  function newSim() {
    const cfg = SC.defaultConfig();
    cfg.n_slots = 32;
    sim = new SC.Sim(cfg, seed);
    crashMarks = [];
    trails = new Map();
    perType = types.map(() => ({ exits: 0, bad: 0 }));
    applyControls();
  }

  function applyControls() {
    const cfg = sim.cfg;
    cfg.demand = +$("demand").value;
    let r = +$("pright").value, l = +$("pleft").value;
    if (r + l > 100) l = 100 - r; // keep shares consistent
    if (+$("pleft").value !== l) $("pleft").value = l;
    const s = 100 - r - l;
    cfg.turn_probs = [r, s, l].every((v) => v === 0) ? [0, 1, 0] : [r, s, l];
    cfg.signal_enabled = $("sigon").checked;
    cfg.signal_noise = +$("noise").value;
    $("demand-o").textContent = `${cfg.demand.toFixed(2)} cars/s`;
    $("pright-o").textContent = `${r}%`;
    $("pleft-o").textContent = `${l}%`;
    $("pstraight").textContent = `Straight: ${s}%`;
    $("noise-o").textContent = cfg.signal_noise.toFixed(2);
    $("noise").disabled = !cfg.signal_enabled;
    // population mix: groups index directly into `types`
    const probs = types.map((t) => (t && t.policy ? t.share : 0));
    const tot = probs.reduce((a, b) => a + b, 0);
    const act = activeTypes();
    sim.groupProbs = tot > 0 ? probs.map((p) => p / tot) : types.map((_, i) => (i === act[0] ? 1 : 0));
  }

  // ------------------------------------------------------------------ simulation
  function policiesForGroups() {
    // group g -> policy of type g; empty slots fall back to the first loaded type
    const act = activeTypes();
    const fallback = act.length ? types[act[0]] : null;
    return types.map((t) => (t && t.policy ? t : fallback));
  }

  function simStep() {
    const per = policiesForGroups();
    if (!per[0]) return;
    const obs = sim.observe();
    const a = SC.act(sim, per.map((t) => t.policy), per.map((t) => t.mute), obs);
    sim.step(a);
    for (const e of sim.events) {
      const g = sim.group[e.n];
      if (e.kind === "crash" || e.kind === "offroad") {
        crashMarks.push({ x: e.x, y: e.y, kind: e.kind, t: sim.t });
        if (perType[g]) perType[g].bad += 1;
      } else if (e.kind === "exit" && perType[g]) perType[g].exits += 1;
      trails.delete(e.uid);
    }
    if (crashMarks.length > 60) crashMarks.splice(0, crashMarks.length - 60);
    for (let n = 0; n < sim.N; n++) {
      if (!sim.alive[n]) continue;
      const id = sim.uid[n];
      let tr = trails.get(id);
      if (!tr) { tr = { man: sim.man[n], pts: [] }; trails.set(id, tr); }
      tr.pts.push(sim.x[n], sim.y[n]);
      if (tr.pts.length > 120) tr.pts.splice(0, 2);
    }
    if (trails.size > 200) {
      const live = new Set(Array.from(sim.uid).filter((u) => u >= 0));
      for (const k of trails.keys()) if (!live.has(k)) trails.delete(k);
    }
  }

  // ------------------------------------------------------------------ rendering
  const canvas = $("view");
  const ctx = canvas.getContext("2d");
  let px = 1; // css px per metre * dpr

  function resize() {
    const r = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    draw();
  }

  function draw() {
    if (!sim) return;
    const cfg = sim.cfg, { L, W, Rc } = cfg;
    const view = L + 1.5;
    const S = canvas.width;
    px = S / (2 * view);
    ctx.setTransform(px, 0, 0, -px, S / 2, S / 2); // metres, y up
    ctx.fillStyle = SCENE.verge;
    ctx.fillRect(-view, -view, 2 * view, 2 * view);
    // road: rounded corner fillets, then the two strips
    ctx.fillStyle = SCENE.road;
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
      ctx.fillRect(Math.min(sx * W, sx * (W + Rc)), Math.min(sy * W, sy * (W + Rc)), Rc, Rc);
    }
    ctx.fillStyle = SCENE.verge;
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
      ctx.beginPath(); ctx.arc(sx * (W + Rc), sy * (W + Rc), Rc, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.fillStyle = SCENE.road;
    ctx.fillRect(-L, -W, 2 * L, 2 * W);
    ctx.fillRect(-W, -L, 2 * W, 2 * L);
    // curb lines
    ctx.strokeStyle = SCENE.edge; ctx.lineWidth = 0.18;
    ctx.beginPath();
    for (let q = 0; q < 4; q++) {
      const [c, s] = SC.rot(q);
      const P = (u, v) => [c * u - s * v, s * u + c * v];
      // one quadrant corner (x>0, y>0 rotated): arm edge, arc, arm edge
      let p = P(L, W); ctx.moveTo(p[0], p[1]);
      p = P(W + Rc, W); ctx.lineTo(p[0], p[1]);
      const cc = P(W + Rc, W + Rc);
      const a0 = -Math.PI / 2 + q * Math.PI / 2;
      ctx.arc(cc[0], cc[1], Rc, a0, a0 - Math.PI / 2, true);
      p = P(W, L); ctx.lineTo(p[0], p[1]);
    }
    ctx.stroke();
    // exit gates
    ctx.strokeStyle = SCENE.gate; ctx.lineWidth = 0.25; ctx.setLineDash([0.8, 0.8]);
    ctx.beginPath();
    ctx.moveTo(-W, L); ctx.lineTo(W, L); ctx.moveTo(-W, -L); ctx.lineTo(W, -L);
    ctx.moveTo(L, -W); ctx.lineTo(L, W); ctx.moveTo(-L, -W); ctx.lineTo(-L, W);
    ctx.stroke(); ctx.setLineDash([]);

    if ($("trails").checked) {
      ctx.lineWidth = 0.25; ctx.globalAlpha = 0.45;
      for (const tr of trails.values()) {
        const p = tr.pts;
        if (p.length < 4) continue;
        ctx.strokeStyle = MAN_COLOURS[tr.man];
        ctx.beginPath(); ctx.moveTo(p[0], p[1]);
        for (let i = 2; i < p.length; i += 2) ctx.lineTo(p[i], p[i + 1]);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // cars
    const NS = cfg.n_signal;
    for (let n = 0; n < sim.N; n++) {
      if (!sim.alive[n]) continue;
      ctx.save();
      ctx.translate(sim.x[n], sim.y[n]);
      ctx.rotate(sim.th[n]);
      roundRect(-2.3, -0.9, 4.6, 1.8, 0.45);
      ctx.fillStyle = MAN_COLOURS[sim.man[n]];
      ctx.fill();
      ctx.lineWidth = 0.32;
      ctx.strokeStyle = TYPE_COLOURS[sim.group[n] % TYPE_COLOURS.length];
      ctx.stroke();
      // windscreen
      ctx.fillStyle = "rgba(20,24,28,0.55)";
      ctx.fillRect(0.7, -0.62, 0.6, 1.24);
      // roof lamp: signal (first two dims -> red / blue)
      if (NS > 0) {
        const s0 = sim.sig[n * NS], s1 = NS > 1 ? sim.sig[n * NS + 1] : 0;
        ctx.beginPath(); ctx.arc(-0.6, 0, 0.62, 0, 2 * Math.PI);
        ctx.fillStyle = cfg.signal_enabled
          ? `rgb(${Math.round(127 + 127 * s0)},40,${Math.round(127 + 127 * s1)})`
          : "rgb(90,90,96)";
        ctx.fill();
        ctx.lineWidth = 0.1; ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.stroke();
      }
      ctx.restore();
    }

    // crash / off-road markers (fade over 3 s)
    for (const m of crashMarks) {
      const age = sim.t - m.t;
      if (age > 3) continue;
      ctx.globalAlpha = 1 - age / 3;
      ctx.strokeStyle = m.kind === "crash" ? "#ff3b30" : "#ffd60a";
      ctx.lineWidth = 0.55;
      const r = 1.6;
      ctx.beginPath();
      ctx.moveTo(m.x - r, m.y - r); ctx.lineTo(m.x + r, m.y + r);
      ctx.moveTo(m.x - r, m.y + r); ctx.lineTo(m.x + r, m.y - r);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // text overlays in screen space
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const dpr = window.devicePixelRatio || 1;
    ctx.font = `${Math.round(12 * dpr)}px ui-monospace, Menlo, monospace`;
    ctx.fillStyle = SCENE.text;
    ctx.textBaseline = "middle";
    const toScreen = (x, y) => [S / 2 + x * px, S / 2 - y * px];
    // queues at the arm ends, on the verge to the right of each entry
    for (let k = 0; k < 4; k++) {
      const q = sim.queueLength(k);
      if (!q) continue;
      const [c, s] = SC.rot(k);
      const u = W + 2.2, v = -L + 4;
      const [sx, sy] = toScreen(c * u - s * v, s * u + c * v);
      ctx.textAlign = k === 0 || k === 3 ? "left" : "right";
      ctx.fillText(`+${q} waiting`, sx, sy);
    }
    // scale bar
    const bx = 12 * dpr, by = S - 14 * dpr, len = 10 * px;
    ctx.fillRect(bx, by, len, 2 * dpr);
    ctx.textAlign = "left";
    ctx.fillText("10 m", bx, by - 9 * dpr);
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // ------------------------------------------------------------------ stats
  function updateStats() {
    const st = sim.stats, t = sim.t;
    const win = Math.min(t, WINDOW_S);
    let n = 0, delay = 0;
    for (let i = st.travel.length - 1; i >= 0; i--) {
      const [te, age, ff] = st.travel[i];
      if (te < t - WINDOW_S) break;
      n += 1; delay += age - ff;
    }
    if (st.travel.length > 2000) st.travel.splice(0, st.travel.length - 1000);
    $("s-thr").textContent = win > 0 ? ((n / win) * 60).toFixed(1) : "0.0";
    $("s-crash").textContent = st.crashes;
    $("s-off").textContent = st.offroad;
    $("s-queue").textContent = sim.queueLength();
    $("s-delay").textContent = n ? (delay / n).toFixed(1) : "–";
    let road = 0;
    for (let i = 0; i < sim.N; i++) road += sim.alive[i];
    $("s-road").textContent = road;
    $("clock").textContent = `t = ${t.toFixed(1)} s`;
    types.forEach((ty, i) => {
      const el = document.querySelector(`#type-${i} .meta`);
      if (el && ty && ty.policy) el.textContent = `${perType[i].exits} exits · ${perType[i].bad} crashed/off-road`;
    });
  }

  // ------------------------------------------------------------------ agent type controls
  let loadSlot = 0;
  function renderTypes() {
    const box = $("types");
    box.textContent = "";
    types.forEach((t, i) => {
      const row = document.createElement("div");
      row.className = "type" + (t && t.policy ? "" : " empty");
      row.id = `type-${i}`;
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.borderColor = TYPE_COLOURS[i];
      const name = document.createElement("div");
      name.className = "tname";
      const loadBtn = document.createElement("button");
      loadBtn.type = "button"; loadBtn.className = "linkbtn";
      loadBtn.textContent = t ? "Replace" : "Load policy";
      loadBtn.addEventListener("click", () => { loadSlot = i; $("file").click(); });
      row.append(sw, name, loadBtn);
      if (!t) {
        name.textContent = "Empty slot";
      } else {
        name.textContent = t.name;
        name.title = t.source;
        if (t.placeholder) {
          const chip = document.createElement("span");
          chip.className = "chip"; chip.textContent = "random weights";
          name.append(chip);
        }
        if (t.error) {
          const err = document.createElement("div");
          err.className = "err"; err.textContent = `Not loaded: ${t.error}`;
          row.append(err);
        } else {
          const r2 = document.createElement("div");
          r2.className = "row2";
          const rng = document.createElement("input");
          rng.type = "range"; rng.min = "0"; rng.max = "100"; rng.step = "1"; rng.value = String(t.share);
          rng.id = `share-${i}`; rng.setAttribute("aria-label", `${t.name} share`);
          const out = document.createElement("output");
          const mute = document.createElement("label");
          mute.className = "toggle";
          const cb = document.createElement("input");
          cb.type = "checkbox"; cb.id = `mute-${i}`; cb.checked = t.mute;
          mute.append(cb, document.createTextNode("Mute"));
          const showShare = () => {
            const tot = types.reduce((a, x) => a + (x && x.policy ? x.share : 0), 0);
            for (let j = 0; j < types.length; j++) {
              const o = document.querySelector(`#type-${j} output`);
              if (o && types[j]) o.textContent = tot ? `${Math.round((100 * types[j].share) / tot)}%` : "0%";
            }
          };
          rng.addEventListener("input", () => { t.share = +rng.value; showShare(); applyControls(); });
          cb.addEventListener("change", () => { t.mute = cb.checked; });
          r2.append(rng, out, mute);
          const meta = document.createElement("div");
          meta.className = "meta"; meta.textContent = "0 exits · 0 crashed/off-road";
          row.append(r2, meta);
          queueMicrotask(showShare);
        }
      }
      box.append(row);
    });
  }

  $("file").addEventListener("change", async (ev) => {
    const f = ev.target.files && ev.target.files[0];
    ev.target.value = "";
    if (!f) return;
    let data;
    try { data = JSON.parse(await f.text()); } catch (e) { data = { error: "file is not valid JSON" }; }
    const name = data.name || f.name.replace(/\.json$/i, "");
    const entry = data.error
      ? { name, source: f.name, error: data.error, share: 50, mute: false, policy: null }
      : addType({ name, source: f.name, policy: data.policy || data }, loadSlot);
    types[loadSlot] = entry;
    if (!perType[loadSlot]) perType[loadSlot] = { exits: 0, bad: 0 };
    renderTypes();
    applyControls();
  });

  // ------------------------------------------------------------------ wiring
  for (const id of ["demand", "pright", "pleft", "noise"]) $(id).addEventListener("input", applyControls);
  $("sigon").addEventListener("change", applyControls);
  $("trails").addEventListener("change", draw);
  document.querySelectorAll('input[name="speed"]').forEach((el) =>
    el.addEventListener("change", () => { speed = +el.value; }));
  function setRunning(on) {
    running = on;
    $("play").textContent = running ? "Pause" : "Play";
    lastTs = null;
  }
  $("play").addEventListener("click", () => setRunning(!running));
  $("reset").addEventListener("click", () => { seed += 1; newSim(); updateStats(); draw(); });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLButtonElement)) {
      e.preventDefault(); setRunning(!running);
    }
  });

  function frame(ts) {
    if (running) {
      if (lastTs !== null) acc += Math.min((ts - lastTs) / 1000, 0.1) * speed;
      lastTs = ts;
      let steps = 0;
      while (acc >= sim.cfg.dt && steps < 40) { simStep(); acc -= sim.cfg.dt; steps += 1; }
      if (steps === 40) acc = 0;
      if (steps) { updateStats(); draw(); }
    }
    requestAnimationFrame(frame);
  }

  newSim();
  renderTypes();
  // warm up so the first frame already shows traffic
  for (let i = 0; i < 150; i++) simStep();
  new ResizeObserver(resize).observe(canvas);
  resize();
  updateStats();
  requestAnimationFrame(frame);
  window.__sc = { get sim() { return sim; }, types, step: simStep };
})();
