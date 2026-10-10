/* Live intersection widget for the website: runs sim.js (window.SmartCrossing) with
 * the exported policies, draws the scene on a canvas, lets the viewer tap a car and
 * overlays what that car senses. Ported from viewer/app.js (rendering) and extended.
 */
(function () {
  "use strict";
  const SC = window.SmartCrossing;
  const SCX = (window.SCX = window.SCX || {});

  const MAN_COLOURS = { 1: "#4287f5", 2: "#f0f0f0", 3: "#f5a623" };
  const MAN_NAMES = { 1: "rechts", 2: "geradeaus", 3: "links" };
  const TYPE_COLOURS = ["#22d3ee", "#ef4fd0", "#a3e635"];
  const SCENE = { verge: "#3b5640", road: "#46484d", edge: "#8d938f", gate: "rgba(255,255,255,0.35)", text: "rgba(255,255,255,0.85)" };
  const OVL = { ray: "#ffe066", hit: "#ff5d5d", nb: "#ffffff", goal: "#ff6bd5", vel: "#7fe3ff" };
  const WINDOW_S = 60;

  SCX.MAN_COLOURS = MAN_COLOURS;
  SCX.MAN_NAMES = MAN_NAMES;
  SCX.TYPE_COLOURS = TYPE_COLOURS;
  SCX.OVL = OVL;
  SCX.lampRGB = (s0, s1) => `rgb(${Math.round(127 + 127 * clamp1(s0))},40,${Math.round(127 + 127 * clamp1(s1))})`;
  function clamp1(v) { return Math.max(-1, Math.min(1, v || 0)); }

  /** Policy entry from SC_DATA.policies -> agent type for the widget. */
  SCX.makeType = function (entry, name, extra) {
    return Object.assign({
      name, id: entry.id, raw: entry.policy, policy: SC.loadPolicy(entry.policy),
      physics: entry.physics || {}, share: 50, mute: false,
    }, extra || {});
  };

  class Crossing {
    /**
     * opts: types [agent types], seed, warmup (s), overlay ("none" | "sensors"),
     *       select (tap to select), autoSelect (always keep a car selected),
     *       dimOthers, onStep(crossing), onFrame(crossing)
     */
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.opts = Object.assign({ overlay: "none", select: true, autoSelect: false, dimOthers: true, warmup: 15, seed: 1 }, opts);
      this.types = this.opts.types || [];
      this.seed = this.opts.seed;
      this.running = true;
      this.speed = this.opts.speed || 1;
      this.acc = 0;
      this.lastTs = null;
      this.showTrails = false;
      this.follow = false;
      this.cam = null; // {x, y, half}
      this.selUid = -1;
      this.fate = null; // {uid, kind, t}
      this.listeners = [];
      this.controls = { demand: 0.15, turn: [1, 1, 1], signal: true, noise: 0 };
      this.newSim();
      if (this.opts.select) this._bindClicks();
      new ResizeObserver(() => this.resize()).observe(canvas);
      this.resize();
      requestAnimationFrame((ts) => this._frame(ts));
    }

    // ---------------------------------------------------------------- setup
    newSim() {
      const cfg = SC.defaultConfig();
      cfg.n_slots = 32;
      this.sim = new SC.Sim(cfg, this.seed);
      this.crashMarks = [];
      this.trails = new Map();
      this.perType = this.types.map(() => ({ exits: 0, bad: 0 }));
      this.obs = null;
      this.selUid = -1;
      this.fate = null;
      this.applyControls();
      const n = Math.round(this.opts.warmup / cfg.dt);
      for (let i = 0; i < n; i++) this.step(true);
      if (this.opts.autoSelect) this.autoSelect();
      this.draw();
      this._emit();
    }

    reset() { this.seed += 1; this.newSim(); }

    setTypes(types, keepSim) {
      this.types = types;
      if (keepSim) {
        this.perType = types.map(() => ({ exits: 0, bad: 0 }));
        this.applyControls();
        this.obs = null;
      } else this.newSim();
    }

    /** Apply sliders + physics of the first agent type (all cars share one physics). */
    applyControls() {
      const c = this.controls, cfg = this.sim.cfg;
      cfg.demand = c.demand;
      cfg.turn_probs = c.turn.every((v) => v === 0) ? [0, 1, 0] : c.turn.slice();
      cfg.signal_enabled = c.signal;
      cfg.signal_noise = c.noise;
      const phys = (this.types[0] && this.types[0].physics) || {};
      for (const [k, v] of Object.entries(phys)) cfg[k] = v;
      const probs = this.types.map((t) => t.share);
      const tot = probs.reduce((a, b) => a + b, 0);
      this.sim.groupProbs = tot > 0 ? probs.map((p) => p / tot) : this.types.map((_, i) => (i === 0 ? 1 : 0));
      this.obs = null; // observation depends on signal settings
    }

    // ---------------------------------------------------------------- simulation
    step(silent) {
      if (!this.types.length) return;
      const sim = this.sim;
      const obs = this.obs || sim.observe();
      const pols = this.types.map((t) => t.policy), mute = this.types.map((t) => t.mute);
      const a = SC.act(sim, pols, mute, obs);
      sim.step(a);
      for (const e of sim.events) {
        const g = sim.group[e.n];
        if (e.kind === "crash" || e.kind === "offroad") {
          this.crashMarks.push({ x: e.x, y: e.y, kind: e.kind, t: sim.t });
          if (this.perType[g]) this.perType[g].bad += 1;
        } else if (e.kind === "exit" && this.perType[g]) this.perType[g].exits += 1;
        if (e.uid === this.selUid) { this.fate = { uid: e.uid, kind: e.kind, t: sim.t }; this.selUid = -1; }
        this.trails.delete(e.uid);
      }
      if (this.crashMarks.length > 60) this.crashMarks.splice(0, this.crashMarks.length - 60);
      for (let n = 0; n < sim.N; n++) {
        if (!sim.alive[n]) continue;
        const id = sim.uid[n];
        let tr = this.trails.get(id);
        if (!tr) { tr = { man: sim.man[n], pts: [] }; this.trails.set(id, tr); }
        tr.pts.push(sim.x[n], sim.y[n]);
        if (tr.pts.length > 160) tr.pts.splice(0, 2);
      }
      if (this.trails.size > 200) {
        const live = new Set(Array.from(sim.uid).filter((u) => u >= 0));
        for (const k of this.trails.keys()) if (!live.has(k)) this.trails.delete(k);
      }
      if (sim.stats.travel.length > 2000) sim.stats.travel.splice(0, sim.stats.travel.length - 1000);
      // observation of the new state: drawn as the sensor overlay and used for the next action
      this.obs = sim.observe();
      if (this.opts.autoSelect && this.selUid < 0 && (!this.fate || sim.t - this.fate.t > 1.5)) this.autoSelect();
      if (!silent && this.opts.onStep) this.opts.onStep(this);
    }

    // ---------------------------------------------------------------- selection
    selIndex() {
      if (this.selUid < 0) return -1;
      const sim = this.sim;
      for (let n = 0; n < sim.N; n++) if (sim.alive[n] && sim.uid[n] === this.selUid) return n;
      return -1;
    }

    select(n) {
      this.selUid = n >= 0 ? this.sim.uid[n] : -1;
      if (n >= 0) this.fate = null;
      this.draw();
      this._emit();
    }

    /** Pick the youngest car that already drove a bit (so it is followed for a while). */
    autoSelect() {
      const sim = this.sim;
      let best = -1;
      for (let n = 0; n < sim.N; n++) {
        if (!sim.alive[n] || sim.age[n] < 0.5) continue;
        if (best < 0 || sim.age[n] < sim.age[best]) best = n;
      }
      if (best >= 0) { this.selUid = sim.uid[best]; }
    }

    /** obs row of car n (Float64Array view). */
    obsRow(n) {
      if (!this.obs) this.obs = this.sim.observe();
      const D = SC.obsDim(this.sim.cfg);
      return this.obs.subarray(n * D, n * D + D);
    }

    typeOf(n) { return this.types[Math.min(this.sim.group[n], this.types.length - 1)]; }

    onChange(fn) { this.listeners.push(fn); }
    _emit() { for (const fn of this.listeners) fn(this); }

    _bindClicks() {
      const pick = (ev) => {
        const r = this.canvas.getBoundingClientRect();
        const { x: cx, y: cy, half } = this._camera();
        const mx = cx + ((ev.clientX - r.left) / r.width * 2 - 1) * half;
        const my = cy - ((ev.clientY - r.top) / r.height * 2 - 1) * half;
        const sim = this.sim;
        let best = -1, bd = Math.max(4.5, half * 0.07);
        for (let n = 0; n < sim.N; n++) {
          if (!sim.alive[n]) continue;
          const d = Math.hypot(sim.x[n] - mx, sim.y[n] - my);
          if (d < bd) { bd = d; best = n; }
        }
        if (best >= 0 || !this.opts.autoSelect) this.select(best);
        if (this.opts.onPick) this.opts.onPick(best);
      };
      this.canvas.addEventListener("click", pick);
    }

    // ---------------------------------------------------------------- loop
    setRunning(on) { this.running = on; this.lastTs = null; this._emit(); }

    _frame(ts) {
      if (this.running && this.visible !== false) {
        if (this.lastTs !== null) this.acc += Math.min((ts - this.lastTs) / 1000, 0.1) * this.speed;
        this.lastTs = ts;
        let steps = 0;
        while (this.acc >= this.sim.cfg.dt && steps < 40) { this.step(); this.acc -= this.sim.cfg.dt; steps += 1; }
        if (steps === 40) this.acc = 0;
        if (steps) { this.draw(); if (this.opts.onFrame) this.opts.onFrame(this); }
      } else this.lastTs = null;
      requestAnimationFrame((t) => this._frame(t));
    }

    // ---------------------------------------------------------------- rendering
    resize() {
      const r = this.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = Math.max(1, Math.round(r.width * dpr));
      this.canvas.height = Math.max(1, Math.round(r.height * dpr));
      this.draw();
    }

    _camera() {
      const L = this.sim.cfg.L, full = { x: 0, y: 0, half: L + 1.5 };
      const n = this.selIndex();
      if (!this.follow || n < 0) { this.cam = null; return full; }
      const half = 24;
      const lim = L + 1.5 - half;
      const tx = Math.max(-lim, Math.min(lim, this.sim.x[n])), ty = Math.max(-lim, Math.min(lim, this.sim.y[n]));
      if (!this.cam) this.cam = { x: tx, y: ty, half };
      this.cam.x += (tx - this.cam.x) * 0.25;
      this.cam.y += (ty - this.cam.y) * 0.25;
      return this.cam;
    }

    draw() {
      const sim = this.sim, ctx = this.ctx;
      if (!sim || !this.canvas.width) return;
      const cfg = sim.cfg, { L, W, Rc } = cfg;
      const S = this.canvas.width;
      const cam = this._camera();
      const px = S / (2 * cam.half);
      this.px = px;
      ctx.setTransform(px, 0, 0, -px, S / 2 - cam.x * px, S / 2 + cam.y * px); // metres, y up
      const view = L + 1.5;
      ctx.fillStyle = SCENE.verge;
      ctx.fillRect(-view - 30, -view - 30, 2 * view + 60, 2 * view + 60);
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
      ctx.strokeStyle = SCENE.edge; ctx.lineWidth = 0.18;
      ctx.beginPath();
      for (let q = 0; q < 4; q++) {
        const [c, s] = SC.rot(q);
        const P = (u, v) => [c * u - s * v, s * u + c * v];
        let p = P(L, W); ctx.moveTo(p[0], p[1]);
        p = P(W + Rc, W); ctx.lineTo(p[0], p[1]);
        const cc = P(W + Rc, W + Rc);
        const a0 = -Math.PI / 2 + q * Math.PI / 2;
        ctx.arc(cc[0], cc[1], Rc, a0, a0 - Math.PI / 2, true);
        p = P(W, L); ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
      ctx.strokeStyle = SCENE.gate; ctx.lineWidth = 0.25; ctx.setLineDash([0.8, 0.8]);
      ctx.beginPath();
      ctx.moveTo(-W, L); ctx.lineTo(W, L); ctx.moveTo(-W, -L); ctx.lineTo(W, -L);
      ctx.moveTo(L, -W); ctx.lineTo(L, W); ctx.moveTo(-L, -W); ctx.lineTo(-L, W);
      ctx.stroke(); ctx.setLineDash([]);

      const sel = this.selIndex();
      if (this.showTrails) {
        ctx.lineWidth = 0.25; ctx.globalAlpha = 0.45;
        for (const tr of this.trails.values()) {
          const p = tr.pts;
          if (p.length < 4) continue;
          ctx.strokeStyle = MAN_COLOURS[tr.man];
          ctx.beginPath(); ctx.moveTo(p[0], p[1]);
          for (let i = 2; i < p.length; i += 2) ctx.lineTo(p[i], p[i + 1]);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      if (sel >= 0) {
        // trail of the selected car
        const tr = this.trails.get(sim.uid[sel]);
        if (tr && tr.pts.length > 3) {
          ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 0.3; ctx.setLineDash([0.6, 0.6]);
          ctx.beginPath(); ctx.moveTo(tr.pts[0], tr.pts[1]);
          for (let i = 2; i < tr.pts.length; i += 2) ctx.lineTo(tr.pts[i], tr.pts[i + 1]);
          ctx.stroke(); ctx.setLineDash([]);
        }
      }

      const overlay = sel >= 0 && this.opts.overlay === "sensors";
      if (overlay) this._drawSensors(sel, true);
      const NS = cfg.n_signal;
      for (let n = 0; n < sim.N; n++) {
        if (!sim.alive[n]) continue;
        ctx.globalAlpha = sel >= 0 && n !== sel && this.opts.dimOthers && overlay ? 0.7 : 1;
        ctx.save();
        ctx.translate(sim.x[n], sim.y[n]);
        ctx.rotate(sim.th[n]);
        roundRect(ctx, -2.3, -0.9, 4.6, 1.8, 0.45);
        ctx.fillStyle = MAN_COLOURS[sim.man[n]];
        ctx.fill();
        ctx.lineWidth = 0.32;
        ctx.strokeStyle = TYPE_COLOURS[sim.group[n] % TYPE_COLOURS.length];
        ctx.stroke();
        ctx.fillStyle = "rgba(20,24,28,0.55)";
        ctx.fillRect(0.7, -0.62, 0.6, 1.24);
        if (NS > 0) {
          const s0 = sim.sig[n * NS], s1 = NS > 1 ? sim.sig[n * NS + 1] : 0;
          ctx.beginPath(); ctx.arc(-0.6, 0, 0.62, 0, 2 * Math.PI);
          ctx.fillStyle = cfg.signal_enabled ? SCX.lampRGB(s0, s1) : "rgb(90,90,96)";
          ctx.fill();
          ctx.lineWidth = 0.1; ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.stroke();
        }
        ctx.restore();
      }
      ctx.globalAlpha = 1;
      if (overlay) this._drawSensors(sel, false);
      if (sel >= 0) {
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 0.35;
        ctx.beginPath(); ctx.arc(sim.x[sel], sim.y[sel], 3.4, 0, 2 * Math.PI); ctx.stroke();
      }

      for (const m of this.crashMarks) {
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

      // screen-space text
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      const dpr = window.devicePixelRatio || 1;
      const fs = Math.round(Math.max(11, Math.min(13, S / dpr / 40)) * dpr);
      ctx.font = `${fs}px ui-monospace, Menlo, monospace`;
      ctx.fillStyle = SCENE.text;
      ctx.textBaseline = "middle";
      const toScreen = (x, y) => [S / 2 + (x - cam.x) * px, S / 2 - (y - cam.y) * px];
      for (let k = 0; k < 4; k++) {
        const q = sim.queueLength(k);
        if (!q) continue;
        const [c, s] = SC.rot(k);
        const u = W + 2.2, v = -L + 4;
        const [sx, sy] = toScreen(c * u - s * v, s * u + c * v);
        ctx.textAlign = k === 0 || k === 3 ? "left" : "right";
        ctx.fillText(`+${q} warten`, sx, sy);
      }
      const bx = 12 * dpr, by = S - 14 * dpr, len = 10 * px;
      ctx.fillRect(bx, by, len, 2 * dpr);
      ctx.textAlign = "left";
      ctx.fillText("10 m", bx, by - 9 * dpr);
    }

    /** Sensor overlay of car n, drawn from its observation vector (exactly what the
     *  network gets). under=true: wide elements below the cars; false: markers on top. */
    _drawSensors(n, under) {
      const sim = this.sim, ctx = this.ctx, cfg = sim.cfg;
      const lay = (window.SC_DATA && window.SC_DATA.layout) || null;
      const o = this.obsRow(n);
      const x = sim.x[n], y = sim.y[n], th = sim.th[n];
      const ct = Math.cos(th), st = Math.sin(th);
      const ego = (f, l) => [x + ct * f - st * l, y + st * f + ct * l];
      const own = SC.OBS_DIM, K = SC.K_NEIGHBOURS, blk = 6 + cfg.n_signal;
      const rayAngles = lay ? lay.ray_angles_deg : [-60, -30, 0, 30, 60];
      const rayRange = SC.RAY_RANGE || 20;
      const rayStart = SCX.keyIndex("ray_r60", 10);
      const gi = SCX.keyIndex("exit_fwd", 8);
      const lw = 1.6 / (this.px || 4) * (window.devicePixelRatio || 1);
      if (under) {
        // neighbour range
        ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = lw; ctx.setLineDash([1, 1]);
        ctx.beginPath(); ctx.arc(x, y, SC.NEIGHBOUR_RANGE, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
        // goal vector (to the centre of the exit gate)
        const [gx, gy] = ego(o[gi] * cfg.L, o[gi + 1] * cfg.L);
        ctx.strokeStyle = OVL.goal; ctx.lineWidth = lw * 1.2; ctx.setLineDash([1.2, 0.9]);
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(gx, gy); ctx.stroke(); ctx.setLineDash([]);
        flag(ctx, gx, gy, OVL.goal, lw);
        // rays
        ctx.strokeStyle = OVL.ray; ctx.lineWidth = lw;
        for (let r = 0; r < rayAngles.length; r++) {
          const d = o[rayStart + r] * rayRange;
          const a = (rayAngles[r] * Math.PI) / 180;
          const ex = x + Math.cos(th + a) * d, ey = y + Math.sin(th + a) * d;
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, ey); ctx.stroke();
        }
        // neighbour lines
        ctx.strokeStyle = OVL.nb; ctx.lineWidth = lw;
        for (let k = 0; k < K; k++) {
          const p = own + k * blk;
          if (!o[p + 5]) continue;
          const [nx, ny] = ego(o[p] * SC.NEIGHBOUR_RANGE, o[p + 1] * SC.NEIGHBOUR_RANGE);
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(nx, ny); ctx.stroke();
        }
        return;
      }
      // ray end points: hit (red dot) or nothing within range (open circle)
      for (let r = 0; r < rayAngles.length; r++) {
        const d = o[rayStart + r] * rayRange;
        const a = (rayAngles[r] * Math.PI) / 180;
        const ex = x + Math.cos(th + a) * d, ey = y + Math.sin(th + a) * d;
        ctx.beginPath(); ctx.arc(ex, ey, 0.75, 0, 2 * Math.PI);
        if (d < rayRange - 1e-6) { ctx.fillStyle = OVL.hit; ctx.fill(); }
        else { ctx.strokeStyle = OVL.ray; ctx.lineWidth = lw; ctx.stroke(); }
      }
      // neighbours: numbered badge filled with the lamp colour the car *receives*
      for (let k = 0; k < K; k++) {
        const p = own + k * blk;
        if (!o[p + 5]) continue;
        const [nx, ny] = ego(o[p] * SC.NEIGHBOUR_RANGE, o[p + 1] * SC.NEIGHBOUR_RANGE);
        const s0 = cfg.n_signal > 0 ? o[p + 6] : 0, s1 = cfg.n_signal > 1 ? o[p + 7] : 0;
        ctx.beginPath(); ctx.arc(nx, ny, 1.55, 0, 2 * Math.PI);
        ctx.fillStyle = cfg.signal_enabled ? SCX.lampRGB(s0, s1) : "rgb(90,90,96)";
        ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = lw; ctx.stroke();
        ctx.save(); ctx.translate(nx, ny); ctx.scale(1, -1);
        ctx.fillStyle = "#fff"; ctx.font = "bold 2px ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(k + 1), 0, 0.1);
        ctx.restore();
      }
      // velocity arrow: where the car would be in one second
      const v = sim.v[n];
      if (v > 0.3) {
        const [vx, vy] = ego(v, 0);
        ctx.strokeStyle = OVL.vel; ctx.fillStyle = OVL.vel; ctx.lineWidth = lw * 1.6;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(vx, vy); ctx.stroke();
        arrowHead(ctx, x, y, vx, vy, 1.4);
      }
    }

    // ---------------------------------------------------------------- stats
    statsSummary() {
      const sim = this.sim, st = sim.stats, t = sim.t;
      const win = Math.min(t, WINDOW_S);
      let n = 0, delay = 0;
      for (let i = st.travel.length - 1; i >= 0; i--) {
        const [te, age, ff] = st.travel[i];
        if (te < t - WINDOW_S) break;
        n += 1; delay += age - ff;
      }
      let road = 0;
      for (let i = 0; i < sim.N; i++) road += sim.alive[i];
      return {
        throughput: win > 0 ? (n / win) * 60 : 0, crashes: st.crashes, offroad: st.offroad,
        exits: st.exits, queue: sim.queueLength(), delay: n ? delay / n : null, road, t,
      };
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function arrowHead(ctx, x0, y0, x1, y1, s) {
    const a = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath();
    ctx.moveTo(x1 + Math.cos(a) * s * 0.6, y1 + Math.sin(a) * s * 0.6);
    ctx.lineTo(x1 + Math.cos(a + 2.5) * s, y1 + Math.sin(a + 2.5) * s);
    ctx.lineTo(x1 + Math.cos(a - 2.5) * s, y1 + Math.sin(a - 2.5) * s);
    ctx.closePath(); ctx.fill();
  }

  function flag(ctx, x, y, col, lw) {
    ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = lw * 1.4;
    ctx.beginPath(); ctx.arc(x, y, 1.1, 0, 2 * Math.PI); ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 2 * Math.PI); ctx.stroke();
  }

  SCX.Crossing = Crossing;

  /** Index of an observation entry by key (from the exported label list). */
  SCX.keyIndex = function (key, fallback) {
    const lay = window.SC_DATA && window.SC_DATA.layout;
    const i = lay ? lay.keys.indexOf(key) : -1;
    return i >= 0 ? i : fallback;
  };

  /** The labels were exported for one sim layout; refuse to mislabel a different one. */
  SCX.checkLayout = function () {
    const lay = window.SC_DATA && window.SC_DATA.layout;
    const dim = SC.obsDim(SC.defaultConfig());
    if (!lay || lay.dim !== dim || lay.own_dim !== SC.OBS_DIM) {
      const msg = `Beschriftung passt nicht zum Simulator (Labels: ${lay && lay.dim}, sim.js: ${dim}). Bitte site/build.py neu laufen lassen.`;
      console.error(msg);
      return msg;
    }
    return null;
  };

  // ------------------------------------------------------------------ shared UI helpers
  SCX.$ = (id) => document.getElementById(id);
  SCX.cssVar = (n, f) => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || f;

  /** Standard transport bar (play/pause, speed, reset, trails). */
  SCX.bindTransport = function (cx, root, speeds) {
    const play = root.querySelector("[data-act=play]");
    const reset = root.querySelector("[data-act=reset]");
    const trails = root.querySelector("[data-act=trails]");
    const follow = root.querySelector("[data-act=follow]");
    const clock = root.querySelector("[data-act=clock]");
    const seg = root.querySelector(".seg");
    if (seg) {
      const name = "spd" + Math.random().toString(36).slice(2, 7);
      seg.innerHTML = "";
      for (const s of speeds || [0.5, 1, 2, 4]) {
        const id = name + String(s).replace(".", "_");
        const inp = document.createElement("input");
        inp.type = "radio"; inp.name = name; inp.id = id; inp.value = s; inp.checked = s === cx.speed;
        const lab = document.createElement("label");
        lab.htmlFor = id; lab.textContent = (s === 0.5 ? "½" : s === 0.25 ? "¼" : s) + "×";
        inp.addEventListener("change", () => { cx.speed = s; });
        seg.append(inp, lab);
      }
    }
    const upd = () => { if (play) play.textContent = cx.running ? "Pause" : "Weiter"; };
    if (play) play.addEventListener("click", () => { cx.setRunning(!cx.running); upd(); });
    if (reset) reset.addEventListener("click", () => cx.reset());
    if (trails) trails.addEventListener("change", () => { cx.showTrails = trails.checked; cx.draw(); });
    if (follow) follow.addEventListener("change", () => { cx.follow = follow.checked; cx.draw(); });
    document.addEventListener("keydown", (e) => {
      if (e.code === "Space" && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLButtonElement)) {
        e.preventDefault(); cx.setRunning(!cx.running); upd();
      }
    });
    const tick = () => { if (clock) clock.textContent = `t = ${cx.sim.t.toFixed(1)} s`; };
    upd();
    return { update: () => { upd(); tick(); } };
  };

  /** Fill a .stats strip (elements with data-stat=...). */
  SCX.renderStats = function (cx, root) {
    const s = cx.statsSummary();
    const set = (k, v) => { const el = root.querySelector(`[data-stat=${k}]`); if (el) el.textContent = v; };
    set("thr", s.throughput.toFixed(1));
    set("exits", s.exits);
    set("crash", s.crashes);
    set("off", s.offroad);
    set("queue", s.queue);
    set("delay", s.delay === null ? "–" : s.delay.toFixed(1));
    set("road", s.road);
  };

  /** Tiny header shown above per-car panels. */
  SCX.renderSelHead = function (cx, el) {
    const n = cx.selIndex(), sim = cx.sim;
    if (n < 0) {
      const f = cx.fate;
      const txt = !f ? "Tippe ein Auto an, um es auszuwählen."
        : f.kind === "exit" ? "Dein Auto ist angekommen. Gleich wird ein neues ausgewählt."
          : f.kind === "crash" ? "Dein Auto hatte einen Unfall. Gleich wird ein neues ausgewählt."
            : f.kind === "offroad" ? "Dein Auto ist von der Straße abgekommen. Gleich wird ein neues ausgewählt."
              : f.kind === "wrong_exit" ? "Dein Auto hat die falsche Ausfahrt genommen."
                : "Dein Auto hat zu lange gebraucht und wurde entfernt.";
      el.innerHTML = `<div class="fate ${f && f.kind === "exit" ? "ok" : f ? "bad" : ""}">${txt}</div>`;
      return;
    }
    const ty = cx.typeOf(n);
    const g = sim.group[n] % TYPE_COLOURS.length;
    el.innerHTML = `<div class="sel-head"><span class="carbadge" style="background:${MAN_COLOURS[sim.man[n]]};border-color:${TYPE_COLOURS[g]}"></span>
      <b>Auto Nr. ${sim.uid[n]}</b><span class="muted small">will ${MAN_NAMES[sim.man[n]] === "geradeaus" ? "geradeaus" : MAN_NAMES[sim.man[n]] + " abbiegen"} · ${(sim.v[n] * 3.6).toFixed(0)} km/h · ${sim.age[n].toFixed(1)} s unterwegs${ty && cx.types.length > 1 ? " · Gehirn: " + ty.name : ""}</span></div>`;
  };
})();
