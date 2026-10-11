/* Live intersection widget for the website: runs sim.js (window.SmartCrossing) with
 * the exported policies, draws the scene on a canvas, lets the viewer tap a car and
 * overlays what that car senses.
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
  const TRAIL_FADE = 5; // s: trails of removed cars fade out over this time
  const MARK_FADE = 5; // s: crash / off-road markers fade out

  SCX.MAN_COLOURS = MAN_COLOURS;
  SCX.MAN_NAMES = MAN_NAMES;
  SCX.TYPE_COLOURS = TYPE_COLOURS;
  SCX.OVL = OVL;
  SCX.lampRGB = (s0, s1) => `rgb(${Math.round(127 + 127 * clamp1(s0))},40,${Math.round(127 + 127 * clamp1(s1))})`;
  /** One lamp channel: dark (-1) ... bright violet-white (+1) (smartcrossing/render.py). */
  SCX.lampRGB1 = (s) => {
    const t = (clamp1(s) + 1) / 2;
    return `rgb(${Math.round(40 + 185 * t)},${Math.round(40 + 130 * t)},${Math.round(50 + 205 * t)})`;
  };
  /** Roof-lamp colour for n lamp channels (1: brightness; 2: the old red / blue mix). */
  SCX.lampCSS = (n, s0, s1) => (n === 1 ? SCX.lampRGB1(s0) : SCX.lampRGB(s0, s1));
  /** Hazard lights of a wreck: on / off every 0.4 s of sim time. */
  SCX.hazardOn = (t) => Math.floor(t / 0.4 + 1e-6) % 2 === 0;
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
      Object.assign(cfg, this._iface());
      this.sim = new SC.Sim(cfg, this.seed);
      this.crashMarks = [];
      this.trails = new Map();
      this.prev = null; // state before the last sim step (for interpolated drawing)
      this.acc = 0;
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

    /** Observation / action interface of the first agent type (lamp count, neighbour layout): fixed
     *  for a sim (array sizes), so a change needs a new sim. Other types must share it. */
    _iface() {
      const phys = (this.types[0] && this.types[0].physics) || {}, d = SC.defaultConfig(), out = {};
      for (const k of ["n_signal", "brake_light"]) out[k] = phys[k] !== undefined ? phys[k] : d[k];
      return out;
    }

    setTypes(types, keepSim) {
      this.types = types;
      const want = this._iface();
      if (keepSim && (want.n_signal !== this.sim.cfg.n_signal || want.brake_light !== this.sim.cfg.brake_light)) keepSim = false;
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
      for (const [k, v] of Object.entries(phys)) if (k !== "n_signal" && k !== "brake_light") cfg[k] = v; // (interface: newSim)
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
      this._snap();
      sim.step(a);
      for (const e of sim.events) {
        const g = sim.group[e.n];
        if (e.kind === "crash" || e.kind === "offroad") {
          this.crashMarks.push({ x: e.x, y: e.y, kind: e.kind, t: sim.t });
          if (this.perType[g]) this.perType[g].bad += 1;
        } else if (e.kind === "exit" && this.perType[g]) this.perType[g].exits += 1;
        if (e.uid === this.selUid) { this.fate = { uid: e.uid, kind: e.kind, t: sim.t }; this.selUid = -1; }
        const tr = this.trails.get(e.uid);
        if (tr) tr.dead = sim.t; // keep it, fade out
      }
      if (this.crashMarks.length > 60) this.crashMarks.splice(0, this.crashMarks.length - 60);
      for (const [k, tr] of this.trails) if (tr.dead !== undefined && sim.t - tr.dead > TRAIL_FADE) this.trails.delete(k);
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
        for (const [k, tr] of this.trails) if (!live.has(k) && tr.dead === undefined) tr.dead = sim.t;
      }
      if (sim.stats.travel.length > 2000) sim.stats.travel.splice(0, sim.stats.travel.length - 1000);
      // observation of the new state: drawn as the sensor overlay and used for the next action
      this.obs = sim.observe();
      if (this.opts.autoSelect && this.selUid < 0 && (!this.fate || sim.t - this.fate.t > 1.5)) this.autoSelect();
      if (!silent && this.opts.onStep) this.opts.onStep(this);
    }

    // ---------------------------------------------------------------- interpolation
    /** Copy the state that the next sim step starts from. */
    _snap() {
      const s = this.sim, p = this.prev || (this.prev = {});
      p.uid = Int32Array.from(s.uid); p.x = Float64Array.from(s.x); p.y = Float64Array.from(s.y);
      p.th = Float64Array.from(s.th); p.sig = Float64Array.from(s.sig);
    }

    /** Car poses for drawing: between the previous and the current sim state, at the
     *  fraction of the step that has elapsed in real time (smooth at any frame rate).
     *  Cars that spawned in the last step are drawn where they are (no interpolation). */
    _interp() {
      const sim = this.sim, N = sim.N, NS = sim.cfg.n_signal, dt = sim.cfg.dt, p = this.prev;
      const a = Math.max(0, Math.min(1, this.acc / dt));
      let ip = this.ip;
      if (!ip || ip.x.length !== N) ip = this.ip = { x: new Float64Array(N), y: new Float64Array(N), th: new Float64Array(N), s0: new Float64Array(N), s1: new Float64Array(N) };
      const ok = p && p.uid.length === N;
      this.slotOf = new Map();
      for (let n = 0; n < N; n++) {
        if (!sim.alive[n] && !sim.wreck[n]) continue;
        this.slotOf.set(sim.uid[n], n);
        const s0 = NS > 0 ? sim.sig[n * NS] : 0, s1 = NS > 1 ? sim.sig[n * NS + 1] : 0;
        if (ok && p.uid[n] === sim.uid[n]) {
          ip.x[n] = p.x[n] + (sim.x[n] - p.x[n]) * a;
          ip.y[n] = p.y[n] + (sim.y[n] - p.y[n]) * a;
          ip.th[n] = p.th[n] + angDiff(sim.th[n], p.th[n]) * a;
          const q0 = NS > 0 ? p.sig[n * NS] : 0, q1 = NS > 1 ? p.sig[n * NS + 1] : 0;
          ip.s0[n] = q0 + (s0 - q0) * a; ip.s1[n] = q1 + (s1 - q1) * a;
        } else {
          ip.x[n] = sim.x[n]; ip.y[n] = sim.y[n]; ip.th[n] = sim.th[n]; ip.s0[n] = s0; ip.s1[n] = s1;
        }
      }
      this.tDraw = ok ? sim.t - (1 - a) * dt : sim.t; // time of the drawn state
      return ip;
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
        this.draw(); // every display frame: cars are interpolated between sim steps
        if (steps && this.opts.onFrame) this.opts.onFrame(this);
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
      const ip = this.ip || this._interp();
      const tx = Math.max(-lim, Math.min(lim, ip.x[n])), ty = Math.max(-lim, Math.min(lim, ip.y[n]));
      if (!this.cam) this.cam = { x: tx, y: ty, half };
      this.cam.x += (tx - this.cam.x) * 0.15;
      this.cam.y += (ty - this.cam.y) * 0.15;
      return this.cam;
    }

    draw() {
      const sim = this.sim, ctx = this.ctx;
      if (!sim || !this.canvas.width) return;
      const cfg = sim.cfg, { L, W } = cfg;
      const S = this.canvas.width;
      const ip = this._interp();
      const cam = this._camera();
      const px = S / (2 * cam.half);
      this.px = px;
      ctx.setTransform(px, 0, 0, -px, S / 2 - cam.x * px, S / 2 + cam.y * px); // metres, y up
      SCX.drawRoad(ctx, cfg);
      const tNow = this.tDraw;

      const sel = this.selIndex();
      const trailPath = (uid, tr) => {
        const p = tr.pts, n = tr.dead === undefined ? this.slotOf.get(uid) : undefined;
        const end = n === undefined ? p.length : p.length - 2; // live car: end at its drawn position
        ctx.beginPath(); ctx.moveTo(p[0], p[1]);
        for (let i = 2; i < end; i += 2) ctx.lineTo(p[i], p[i + 1]);
        if (n !== undefined) ctx.lineTo(ip.x[n], ip.y[n]);
      };
      const fade = (tr) => (tr.dead === undefined ? 1 : Math.max(0, Math.min(1, 1 - (tNow - tr.dead) / TRAIL_FADE)));
      if (this.showTrails) {
        ctx.lineWidth = 0.25;
        for (const [uid, tr] of this.trails) {
          const f = fade(tr);
          if (tr.pts.length < 4 || f <= 0) continue;
          ctx.globalAlpha = 0.45 * f;
          ctx.strokeStyle = MAN_COLOURS[tr.man];
          trailPath(uid, tr); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      // trail of the selected car (and of the one just lost, fading)
      for (const uid of [sel >= 0 ? sim.uid[sel] : -1, this.fate ? this.fate.uid : -1]) {
        const tr = uid >= 0 && this.trails.get(uid);
        if (!tr || tr.pts.length < 4) continue;
        ctx.globalAlpha = fade(tr);
        ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 0.3; ctx.setLineDash([0.6, 0.6]);
        trailPath(uid, tr); ctx.stroke(); ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }

      const overlay = sel >= 0 && this.opts.overlay === "sensors";
      if (overlay) this._drawSensors(sel, true);
      // crash / off-road marks under the cars (a wreck stays visible on top of its mark)
      for (const m of this.crashMarks) SCX.drawMark(ctx, m.x, m.y, m.kind, 1 - (tNow - m.t) / MARK_FADE);
      ctx.globalAlpha = 1;
      const NS = cfg.n_signal, hazard = SCX.hazardOn(tNow);
      for (let n = 0; n < sim.N; n++) {
        if (!sim.alive[n] && !sim.wreck[n]) continue;
        ctx.globalAlpha = sel >= 0 && n !== sel && this.opts.dimOthers && overlay ? 0.7 : 1;
        const frame = TYPE_COLOURS[sim.group[n] % TYPE_COLOURS.length];
        if (sim.wreck[n]) { SCX.drawCar(ctx, ip.x[n], ip.y[n], ip.th[n], sim.man[n], frame, null, false, { on: hazard }); continue; }
        const lamp = NS > 0 ? (cfg.signal_enabled ? SCX.lampCSS(NS, ip.s0[n], ip.s1[n]) : "rgb(90,90,96)") : null;
        SCX.drawCar(ctx, ip.x[n], ip.y[n], ip.th[n], sim.man[n], frame, lamp, sim.acc[n] < SCX.BRAKE_ACC);
      }
      ctx.globalAlpha = 1;
      if (overlay) this._drawSensors(sel, false);
      if (sel >= 0) {
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 0.35;
        ctx.beginPath(); ctx.arc(ip.x[sel], ip.y[sel], 3.4, 0, 2 * Math.PI); ctx.stroke();
      }


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
     *  network gets; old-layout brains get the same vector, their weights are mapped onto
     *  it). Indices and units come from the exported label metadata (SCX.obsMap).
     *  under=true: wide elements below the cars; false: markers on top. */
    _drawSensors(n, under) {
      const sim = this.sim, ctx = this.ctx, cfg = sim.cfg;
      const m = SCX.obsMap(cfg);
      if (!m) return;
      const o = this.obsRow(n);
      const ip = this.ip || this._interp();
      const x = ip.x[n], y = ip.y[n], th = ip.th[n];
      const ct = Math.cos(th), st = Math.sin(th);
      const ego = (f, l) => [x + ct * f - st * l, y + st * f + ct * l];
      const rayEnd = (r) => {
        const d = o[r.i] * r.scale, a = th + (r.deg * Math.PI) / 180;
        return [x + Math.cos(a) * d, y + Math.sin(a) * d, d >= r.scale - 1e-6];
      };
      const nbPos = (nb) => ego(o[nb.fwd.i] * nb.fwd.scale, o[nb.left.i] * nb.left.scale);
      const lw = 1.6 / (this.px || 4) * (window.devicePixelRatio || 1);
      if (under) {
        // perception range (neighbours further away are not seen)
        ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = lw; ctx.setLineDash([1, 1]);
        ctx.beginPath(); ctx.arc(x, y, cfg.neighbour_vis, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
        // goal vector (to the centre of the exit gate)
        const [gx, gy] = ego(o[m.exit[0].i] * m.exit[0].scale, o[m.exit[1].i] * m.exit[1].scale);
        ctx.strokeStyle = OVL.goal; ctx.lineWidth = lw * 1.2; ctx.setLineDash([1.2, 0.9]);
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(gx, gy); ctx.stroke(); ctx.setLineDash([]);
        flag(ctx, gx, gy, OVL.goal, lw);
        // rays
        ctx.strokeStyle = OVL.ray; ctx.lineWidth = lw;
        for (const r of m.rays) {
          const [ex, ey] = rayEnd(r);
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, ey); ctx.stroke();
        }
        // neighbour lines
        ctx.strokeStyle = OVL.nb; ctx.lineWidth = lw;
        for (const nb of m.nbs) {
          if (o[nb.present.i] < 0.5) continue;
          const [nx, ny] = nbPos(nb);
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(nx, ny); ctx.stroke();
        }
        return;
      }
      // ray end points: hit (red dot) or nothing within range (open circle)
      for (const r of m.rays) {
        const [ex, ey, free] = rayEnd(r);
        ctx.beginPath(); ctx.arc(ex, ey, 0.75, 0, 2 * Math.PI);
        if (!free) { ctx.fillStyle = OVL.hit; ctx.fill(); }
        else { ctx.strokeStyle = OVL.ray; ctx.lineWidth = lw; ctx.stroke(); }
      }
      // neighbours: numbered badge filled with the lamp colour the car *receives*
      m.nbs.forEach((nb, k) => {
        if (o[nb.present.i] < 0.5) return;
        const [nx, ny] = nbPos(nb);
        const s0 = nb.lamps.length > 0 ? o[nb.lamps[0].i] : 0, s1 = nb.lamps.length > 1 ? o[nb.lamps[1].i] : 0;
        ctx.beginPath(); ctx.arc(nx, ny, 1.55, 0, 2 * Math.PI);
        ctx.fillStyle = cfg.signal_enabled ? SCX.lampCSS(nb.lamps.length, s0, s1) : "rgb(90,90,96)";
        ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = lw; ctx.stroke();
        ctx.save(); ctx.translate(nx, ny); ctx.scale(1, -1);
        ctx.fillStyle = "#fff"; ctx.font = "bold 2px ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(k + 1), 0, 0.1);
        ctx.restore();
      });
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
      let road = 0, wrecks = 0;
      for (let i = 0; i < sim.N; i++) { road += sim.alive[i]; wrecks += sim.wreck[i]; }
      return {
        throughput: win > 0 ? (n / win) * 60 : 0, crashes: st.crashes, offroad: st.offroad,
        exits: st.exits, queue: sim.queueLength(), delay: n ? delay / n : null, road, wrecks, t,
      };
    }
  }

  function angDiff(a, b) { let d = a - b; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return d; }
  SCX.angDiff = angDiff;

  /** Verge, road, curbs and exit gates (canvas transform: metres, y up). */
  SCX.drawRoad = function (ctx, cfg) {
    const { L, W, Rc } = cfg;
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
  };

  SCX.BRAKE_ACC = SC.BRAKE_ACC; // m/s^2: brake lights on below this actual acceleration (sim.BRAKE_ACC)
  /** One car: body in the manoeuvre colour, frame in the brain colour, roof lamp (css colour or
   *  null), two brake lights at the rear (lit while braking). wreck = {on}: a crashed car (dimmed,
   *  hazard lights at the four corners, lit while on). */
  SCX.drawCar = function (ctx, x, y, th, man, frame, lamp, brake, wreck) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(th);
    roundRect(ctx, -2.3, -0.9, 4.6, 1.8, 0.45);
    ctx.fillStyle = MAN_COLOURS[man];
    ctx.fill();
    if (wreck) {
      ctx.fillStyle = "rgba(20,20,24,0.55)"; ctx.fill(); // dimmed body
      ctx.lineWidth = 0.3; ctx.strokeStyle = wreck.on ? "#ffab00" : "#1b1b1f"; ctx.stroke();
      for (const [cx, cy] of [[1.95, 0.6], [1.95, -0.6], [-1.95, 0.6], [-1.95, -0.6]]) {
        ctx.beginPath(); ctx.arc(cx, cy, 0.55, 0, 2 * Math.PI);
        if (wreck.on) { ctx.fillStyle = "#ffab00"; ctx.shadowColor = "#ffb300"; ctx.shadowBlur = 8; }
        else { ctx.fillStyle = "rgba(120,80,10,0.9)"; ctx.shadowBlur = 0; }
        ctx.fill();
      }
      ctx.shadowBlur = 0;
      ctx.restore();
      return;
    }
    if (frame) { ctx.lineWidth = 0.32; ctx.strokeStyle = frame; ctx.stroke(); }
    ctx.fillStyle = "rgba(20,24,28,0.55)";
    ctx.fillRect(0.7, -0.62, 0.6, 1.24);
    if (brake) {
      ctx.fillStyle = "#ff1f1f";
      ctx.shadowColor = "#ff2a2a"; ctx.shadowBlur = 6;
      ctx.fillRect(-2.32, -0.78, 0.42, 0.46);
      ctx.fillRect(-2.32, 0.32, 0.42, 0.46);
      ctx.shadowBlur = 0;
    } else {
      ctx.fillStyle = "rgba(120,20,24,0.85)";
      ctx.fillRect(-2.25, -0.7, 0.3, 0.36);
      ctx.fillRect(-2.25, 0.34, 0.3, 0.36);
    }
    if (lamp) {
      ctx.beginPath(); ctx.arc(-0.6, 0, 0.62, 0, 2 * Math.PI);
      ctx.fillStyle = lamp;
      ctx.fill();
      ctx.lineWidth = 0.1; ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.stroke();
    }
    ctx.restore();
  };

  /** Crash (red) / off-road (yellow) cross; alpha 0..1 (nothing drawn at <= 0). */
  SCX.drawMark = function (ctx, x, y, kind, alpha) {
    if (alpha <= 0) return;
    ctx.globalAlpha = Math.min(1, alpha);
    ctx.strokeStyle = kind === "crash" ? "#ff3b30" : "#ffd60a";
    ctx.lineWidth = 0.55;
    const r = 1.6;
    ctx.beginPath();
    ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
    ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r);
    ctx.stroke();
    ctx.globalAlpha = 1;
  };

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

  /** Name of an observation interface (site/obs_labels.py interface_key), e.g. "1L-brake". */
  SCX.ifaceKey = (cfg) => `${cfg.n_signal}L-${cfg.brake_light ? "brake" : "acc"}`;

  /** Label metadata (SC_DATA.layouts, one per observation interface) for a sim config; without
   *  cfg: the main brain's (SC_DATA.layout). */
  SCX.layoutFor = function (cfg) {
    const D = window.SC_DATA;
    if (!D) return null;
    if (!cfg) return D.layout;
    return (D.layouts && D.layouts[SCX.ifaceKey(cfg)]) || null;
  };

  /** Observation entry {i, scale, unit, ...} by feature key for a sim config (see layoutFor), or null. */
  SCX.obsItem = function (key, cfg) {
    const lay = SCX.layoutFor(cfg);
    if (!lay) return null;
    if (!lay._byKey) {
      lay._byKey = new Map();
      for (const g of lay.groups) for (const it of g.items) lay._byKey.set(it.key, it);
    }
    return lay._byKey.get(key) || null;
  };

  /** What the sensor overlay reads from the observation vector, by feature key (so a layout
   *  change in sim.py moves the indices with it): exit-gate vector, edge rays with their angles,
   *  and per neighbour its position, "present" flag and lamps. null (and an error) if the
   *  labels do not match the simulator. */
  const obsMapCache = new Map();
  SCX.obsMap = function (cfg) {
    const key = SCX.ifaceKey(cfg);
    if (obsMapCache.has(key)) return obsMapCache.get(key);
    const lay = SCX.layoutFor(cfg);
    let m = null;
    if (lay && !SCX.checkLayout(cfg)) {
      const it = (k) => SCX.obsItem(k, cfg);
      const rays = lay.keys.filter((k) => k.startsWith("ray_")).map((k, r) => Object.assign({ deg: lay.ray_angles_deg[r] }, it(k)));
      const nbs = [];
      for (let k = 1; k <= lay.k_neighbours; k++) {
        const lamps = [];
        for (let q = 1; q <= cfg.n_signal; q++) lamps.push(it(`n${k}_lamp${q}`));
        nbs.push({ fwd: it(`n${k}_fwd`), left: it(`n${k}_left`), present: it(`n${k}_present`), lamps });
      }
      m = { exit: [it("exit_fwd"), it("exit_left")], rays, nbs };
      const all = [...m.exit, ...rays.map((r) => r.i === undefined ? null : r), ...nbs.flatMap((nb) => [nb.fwd, nb.left, nb.present, ...nb.lamps])];
      if (rays.length !== lay.ray_angles_deg.length || all.some((x) => !x)) {
        console.error("sensor overlay: observation labels lack a feature it draws (site/obs_labels.py)");
        m = null;
      }
    }
    obsMapCache.set(key, m);
    return m;
  };

  /** The labels were exported per observation interface; refuse to mislabel a different one.
   *  cfg: the sim's config (default: the main brain's interface). */
  SCX.checkLayout = function (cfg) {
    const D = window.SC_DATA;
    if (!cfg) {
      const main = D && D.policies && D.policies[D.main_id];
      cfg = Object.assign(SC.defaultConfig(), main ? { n_signal: main.physics.n_signal, brake_light: main.physics.brake_light } : {});
    }
    const lay = SCX.layoutFor(cfg) || (cfg && SCX.layoutFor());
    const dim = SC.obsDim(cfg);
    if (!lay || lay.dim !== dim || lay.own_dim !== SC.OBS_DIM || (lay.key && lay.key !== SCX.ifaceKey(cfg))) {
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
