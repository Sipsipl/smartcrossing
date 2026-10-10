/* Smart Crossing: JS port of smartcrossing/sim.py (one environment, B = 1) and of
 * the MLP forward pass in smartcrossing/policy.py.
 *
 * Keep this file in lock-step with the Python code: tests/test_viewer_parity.py
 * runs observe()/step() on states exported from Python and compares the results.
 * Randomness (arrivals, signal noise) uses a seeded JS PRNG; it is not meant to
 * reproduce numpy's streams.
 *
 * Works as a browser script (global `SmartCrossing`) and as a CommonJS module.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.SmartCrossing = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const RIGHT = 1, STRAIGHT = 2, LEFT = 3;
  const DISC_OFFSETS = [-1.4, 0.0, 1.4];
  const DISC_R = 0.9;
  const SPAWN_LAT_MARGIN = 0.8;
  const N_RAYS = 5;
  const RAY_ANGLES = [-60.0, -30.0, 0.0, 30.0, 60.0].map((d) => d * (Math.PI / 180));
  const N_RAY_STEPS = 20; // RAY_STEPS = 1, 2, ..., 20 m
  const K_NEIGHBOURS = 4;
  const NEIGHBOUR_RANGE = 30.0;
  const OBS_DIM = 4 + 1 + 3 + 2 + N_RAYS;

  // Mirrors the SimConfig dataclass defaults (checked by the parity test).
  function defaultConfig() {
    return {
      L: 50.0, W: 3.5, Rc: 6.0, dt: 0.1,
      v_max: 14.0, acc_max: 3.0, brake_max: 7.0, kappa_max: 0.2,
      spawn_speed: 8.0, spawn_clearance: 7.0, spawn_headway: 1.5,
      max_age: 60.0, n_slots: 32, n_signal: 2, demand: 0.15,
      turn_probs: [1 / 3, 1 / 3, 1 / 3],
      signal_noise: 0.0, signal_enabled: true,
      r_exit: 1.0, r_wrong_exit: -1.0, r_crash: -2.0, r_offroad: -2.0,
      r_timeout: -1.0, r_progress: 0.01,
    };
  }

  const obsDim = (cfg) => OBS_DIM + K_NEIGHBOURS * (6 + cfg.n_signal);
  const actDim = (cfg) => 2 + cfg.n_signal;

  // ------------------------------------------------------------------ helpers
  /** rint(cos(k*90deg)), rint(sin(k*90deg)) for integer k. */
  function rot(k) {
    const m = ((k % 4) + 4) % 4;
    return [[1, 0], [0, 1], [-1, 0], [0, -1]][m];
  }

  function cornerDist(x, y, W, Rc) {
    return Math.hypot(Math.max(W + Rc - Math.abs(x), 0.0), Math.max(W + Rc - Math.abs(y), 0.0));
  }

  const offroadPoint = (x, y, W, Rc) => cornerDist(x, y, W, Rc) < Rc;

  /** numpy-style float remainder (result has the sign of the divisor). */
  function pyMod(a, b) {
    let r = a % b;
    if (r !== 0 && (r < 0) !== (b < 0)) r += b;
    return r;
  }

  function freeFlowTime(cfg, man) {
    const c = cfg.W + cfg.Rc;
    const armLen = 2 * (cfg.L - c) - 2.5;
    let len;
    if (man === STRAIGHT) len = 2 * cfg.L - 2.5;
    else if (man === RIGHT) len = armLen + (Math.PI / 2) * (c - cfg.W / 2);
    else len = armLen + (Math.PI / 2) * (c + cfg.W / 2);
    return len / cfg.v_max;
  }

  /** sfc32 PRNG seeded through splitmix32; returns floats in [0, 1). */
  function makeRng(seed) {
    let s = seed >>> 0;
    const sm = () => {
      s = (s + 0x9e3779b9) | 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) >>> 0;
    };
    let a = sm(), b = sm(), c = sm(), d = sm();
    const next = () => {
      a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
      let t = (a + b) | 0;
      a = b ^ (b >>> 9);
      b = (c + (c << 3)) | 0;
      c = (c << 21) | (c >>> 11);
      d = (d + 1) | 0;
      t = (t + d) | 0;
      c = (c + t) | 0;
      return (t >>> 0) / 4294967296;
    };
    let spare = null;
    next.normal = () => {
      if (spare !== null) { const r = spare; spare = null; return r; }
      let u = 0;
      while (u === 0) u = next();
      const v = next();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
    next.choice = (p) => {
      let tot = 0;
      for (const q of p) tot += q;
      let u = next() * tot;
      for (let i = 0; i < p.length; i++) { u -= p[i]; if (u < 0) return i; }
      return p.length - 1;
    };
    return next;
  }

  // ------------------------------------------------------------------ Sim
  class Sim {
    constructor(cfg, seed = 0) {
      this.cfg = Object.assign(defaultConfig(), cfg || {});
      this.rng = makeRng(seed);
      this.noiseRng = makeRng((seed ^ 0x5bd1e995) >>> 0);
      this.groupProbs = [1.0];
      this.reset();
    }

    reset() {
      const N = this.cfg.n_slots, S = this.cfg.n_signal;
      this.N = N;
      this.alive = new Uint8Array(N);
      this.x = new Float64Array(N);
      this.y = new Float64Array(N);
      this.th = new Float64Array(N);
      this.v = new Float64Array(N);
      this.kappa = new Float64Array(N);
      this.sig = new Float64Array(N * S);
      this.arm = new Int32Array(N);
      this.man = new Int32Array(N);
      this.age = new Float64Array(N);
      this.uid = new Int32Array(N).fill(-1);
      this.group = new Int32Array(N);
      this.prevDist = new Float64Array(N);
      this.queues = [[], [], [], []]; // FIFO of {lat, man, grp} per arm
      this.nextUid = 0;
      this.t = 0.0;
      this.events = [];
      this.stats = { arrived: 0, spawned: 0, exits: 0, wrong_exits: 0, crashes: 0, offroad: 0, timeouts: 0, travel: [] };
    }

    tgt(n) { return (this.arm[n] + this.man[n]) % 4; }

    gateXY(n) {
      const [c, s] = rot(this.tgt(n));
      return [s * this.cfg.L, -c * this.cfg.L];
    }

    distToGate(n) {
      const [gx, gy] = this.gateXY(n);
      return Math.hypot(this.x[n] - gx, this.y[n] - gy);
    }

    queueLength(k) { return k === undefined ? this.queues.reduce((a, q) => a + q.length, 0) : this.queues[k].length; }

    // ---------------------------------------------------------------- observe
    /** Returns a Float64Array of shape (N, obsDim) in row-major order. */
    observe() {
      const cfg = this.cfg, N = this.N, S = cfg.n_signal, L = cfg.L, W = cfg.W;
      const D = obsDim(cfg), blk = 6 + S;
      const obs = new Float64Array(N * D);
      const cand = [];
      for (let i = 0; i < N; i++) {
        if (!this.alive[i]) continue;
        const o = i * D;
        const x = this.x[i], y = this.y[i], th = this.th[i];
        const [c, s] = rot(-this.arm[i]);
        const cx = c * x - s * y, cy = s * x + c * y;
        const cth = th - this.arm[i] * (Math.PI / 2);
        obs[o] = cx / L;
        obs[o + 1] = cy / L;
        obs[o + 2] = Math.cos(cth);
        obs[o + 3] = Math.sin(cth);
        obs[o + 4] = this.v[i] / cfg.v_max;
        const m = this.man[i];
        obs[o + 5] = m === RIGHT ? 1 : 0;
        obs[o + 6] = m === STRAIGHT ? 1 : 0;
        obs[o + 7] = m === LEFT ? 1 : 0;
        const [gx, gy] = this.gateXY(i);
        const dx = gx - x, dy = gy - y;
        const ct = Math.cos(th), st = Math.sin(th);
        obs[o + 8] = (ct * dx + st * dy) / L;
        obs[o + 9] = (-st * dx + ct * dy) / L;
        for (let r = 0; r < N_RAYS; r++) {
          const ang = th + RAY_ANGLES[r];
          const ca = Math.cos(ang), sa = Math.sin(ang);
          let first = N_RAY_STEPS;
          for (let k = 0; k < N_RAY_STEPS; k++) {
            const step = k + 1;
            if (offroadPoint(x + ca * step, y + sa * step, W, cfg.Rc)) { first = k; break; }
          }
          obs[o + 10 + r] = first / N_RAY_STEPS;
        }
        // neighbours: nearest K alive cars within range
        cand.length = 0;
        for (let j = 0; j < N; j++) {
          if (j === i || !this.alive[j]) continue;
          const nx = this.x[j] - x, ny = this.y[j] - y;
          const d = Math.hypot(nx, ny);
          if (d < NEIGHBOUR_RANGE) cand.push([d, j, nx, ny]);
        }
        cand.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        for (let k = 0; k < Math.min(K_NEIGHBOURS, cand.length); k++) {
          const [, j, nx, ny] = cand[k];
          const p = o + OBS_DIM + k * blk;
          const nth = this.th[j] - th;
          obs[p] = (ct * nx + st * ny) / NEIGHBOUR_RANGE;
          obs[p + 1] = (-st * nx + ct * ny) / NEIGHBOUR_RANGE;
          obs[p + 2] = Math.cos(nth);
          obs[p + 3] = Math.sin(nth);
          obs[p + 4] = this.v[j] / cfg.v_max;
          obs[p + 5] = 1.0;
          for (let q = 0; q < S; q++) {
            let sv = 0.0;
            if (cfg.signal_enabled) {
              sv = this.sig[j * S + q];
              if (cfg.signal_noise > 0) sv += cfg.signal_noise * this.noiseRng.normal();
            }
            obs[p + 6 + q] = sv;
          }
        }
      }
      return obs;
    }

    // ---------------------------------------------------------------- step
    /** action: Float64Array (N, actDim) row-major in [-1, 1]. Returns the team reward. */
    step(action) {
      const cfg = this.cfg, N = this.N, S = cfg.n_signal, A = actDim(cfg), dt = cfg.dt;
      const clip = (a) => Math.min(Math.max(a, -1.0), 1.0);
      for (let n = 0; n < N; n++) {
        const a0 = clip(action[n * A]), a1 = clip(action[n * A + 1]);
        const alive = this.alive[n];
        const acc = a0 > 0 ? a0 * cfg.acc_max : a0 * cfg.brake_max;
        this.kappa[n] = a1 * cfg.kappa_max;
        for (let q = 0; q < S; q++) this.sig[n * S + q] = alive ? clip(action[n * A + 2 + q]) : 0.0;
        const v = Math.min(Math.max(this.v[n] + acc * dt, 0.0), cfg.v_max);
        const th = this.th[n] + v * this.kappa[n] * dt;
        if (alive) {
          this.x[n] = this.x[n] + v * Math.cos(th) * dt;
          this.y[n] = this.y[n] + v * Math.sin(th) * dt;
          this.th[n] = pyMod(th + Math.PI, 2 * Math.PI) - Math.PI;
          this.v[n] = v;
          this.age[n] += dt;
        } else {
          this.v[n] = 0.0;
        }
      }
      this.t += dt;
      this.events = [];

      let reward = 0.0;
      for (let n = 0; n < N; n++) {
        const d = this.distToGate(n);
        if (this.alive[n]) reward += cfg.r_progress * (this.prevDist[n] - d);
        this.prevDist[n] = d;
      }

      // exits
      const out = new Uint8Array(N);
      let anyOut = false;
      for (let n = 0; n < N; n++) {
        if (!this.alive[n] || !(Math.abs(this.x[n]) > cfg.L || Math.abs(this.y[n]) > cfg.L)) continue;
        out[n] = 1; anyOut = true;
        const cand = [-this.y[n], this.x[n], this.y[n], -this.x[n]];
        let ea = 0;
        for (let k = 1; k < 4; k++) if (cand[k] > cand[ea]) ea = k;
        if (ea === this.tgt(n)) {
          reward += cfg.r_exit;
          this.stats.exits += 1;
          this.stats.travel.push([this.t, this.age[n], freeFlowTime(cfg, this.man[n]), this.group[n]]);
          this.events.push({ kind: "exit", n, uid: this.uid[n], x: this.x[n], y: this.y[n] });
        } else {
          reward += cfg.r_wrong_exit;
          this.stats.wrong_exits += 1;
          this.events.push({ kind: "wrong_exit", n, uid: this.uid[n], x: this.x[n], y: this.y[n] });
        }
      }
      if (anyOut) this._remove(out);

      // off-road and car-car collisions
      const ox = new Float64Array(N * 3), oy = new Float64Array(N * 3);
      for (let n = 0; n < N; n++) {
        const ct = Math.cos(this.th[n]), st = Math.sin(this.th[n]);
        for (let k = 0; k < 3; k++) {
          ox[n * 3 + k] = this.x[n] + ct * DISC_OFFSETS[k];
          oy[n * 3 + k] = this.y[n] + st * DISC_OFFSETS[k];
        }
      }
      const off = new Uint8Array(N), col = new Uint8Array(N);
      const near2 = (2 * 1.4 + 2 * DISC_R) ** 2, hit2 = (2 * DISC_R) ** 2;
      for (let i = 0; i < N; i++) {
        if (!this.alive[i]) continue;
        for (let k = 0; k < 3; k++) {
          if (cornerDist(ox[i * 3 + k], oy[i * 3 + k], cfg.W, cfg.Rc) < cfg.Rc + DISC_R) { off[i] = 1; break; }
        }
        for (let j = 0; j < N && !col[i]; j++) {
          if (j === i || !this.alive[j]) continue;
          const dcx = this.x[i] - this.x[j], dcy = this.y[i] - this.y[j];
          if (!(dcx ** 2 + dcy ** 2 < near2)) continue;
          for (let a = 0; a < 3 && !col[i]; a++) {
            for (let b = 0; b < 3; b++) {
              const dx = ox[i * 3 + a] - ox[j * 3 + b], dy = oy[i * 3 + a] - oy[j * 3 + b];
              if (dx ** 2 + dy ** 2 < hit2) { col[i] = 1; break; }
            }
          }
        }
      }
      const bad = new Uint8Array(N);
      for (let n = 0; n < N; n++) {
        if (off[n]) {
          this.events.push({ kind: "offroad", n, uid: this.uid[n], x: this.x[n], y: this.y[n] });
          reward += cfg.r_offroad; this.stats.offroad += 1; bad[n] = 1;
        }
      }
      for (let n = 0; n < N; n++) {
        if (col[n] && !off[n]) {
          this.events.push({ kind: "crash", n, uid: this.uid[n], x: this.x[n], y: this.y[n] });
          reward += cfg.r_crash; this.stats.crashes += 1; bad[n] = 1;
        }
      }
      this._remove(bad);

      const tmo = new Uint8Array(N);
      for (let n = 0; n < N; n++) {
        if (this.alive[n] && this.age[n] > cfg.max_age) {
          tmo[n] = 1; reward += cfg.r_timeout; this.stats.timeouts += 1;
          this.events.push({ kind: "timeout", n, uid: this.uid[n], x: this.x[n], y: this.y[n] });
        }
      }
      this._remove(tmo);

      this._arrivals();
      return reward;
    }

    _remove(mask) {
      const S = this.cfg.n_signal;
      for (let n = 0; n < this.N; n++) {
        if (!mask[n]) continue;
        this.alive[n] = 0;
        this.v[n] = 0.0;
        for (let q = 0; q < S; q++) this.sig[n * S + q] = 0.0;
        this.age[n] = 0.0;
        this.uid[n] = -1;
        this.x[n] = 1e4;
        this.y[n] = 1e4;
      }
    }

    // ---------------------------------------------------------------- spawning
    _arrivals() {
      const cfg = this.cfg, N = this.N, rng = this.rng;
      if (cfg.demand > 0) {
        for (let k = 0; k < 4; k++) {
          const arrive = rng() < cfg.demand * cfg.dt;
          const lat = -(cfg.W - 1.2) + 2 * (cfg.W - 1.2) * rng();
          const man = 1 + rng.choice(cfg.turn_probs);
          const grp = rng.choice(this.groupProbs);
          if (arrive) {
            this.queues[k].push({ lat, man, grp });
            this.stats.arrived += 1;
          }
        }
      }
      // velocities of existing cars, taken before any spawn of this step
      const vx = new Float64Array(N), vy = new Float64Array(N);
      for (let n = 0; n < N; n++) {
        vx[n] = this.v[n] * Math.cos(this.th[n]);
        vy[n] = this.v[n] * Math.sin(this.th[n]);
      }
      for (let k = 0; k < 4; k++) {
        const q = this.queues[k];
        if (!q.length) continue;
        const [c, s] = rot(k);
        const u = q[0].lat;
        const sx = c * u - s * (-cfg.L + 2.5);
        const sy = s * u + c * (-cfg.L + 2.5);
        const h = Math.PI / 2 + k * (Math.PI / 2);
        const ch = Math.cos(h), sh = Math.sin(h);
        let clear = true, free = -1;
        for (let j = 0; j < N; j++) {
          if (!this.alive[j]) { if (free < 0) free = j; continue; }
          const dx = this.x[j] - sx, dy = this.y[j] - sy;
          const dist = Math.hypot(dx, dy);
          const dd = Math.max(dist, 1e-6);
          const rvx = cfg.spawn_speed * ch - vx[j], rvy = cfg.spawn_speed * sh - vy[j];
          let closing = Math.max((rvx * dx + rvy * dy) / dd, 0.0);
          const own = Math.max((cfg.spawn_speed * (ch * dx + sh * dy)) / dd, 0.0);
          const latOff = Math.abs(-sh * dx + ch * dy);
          if (!(latOff < 2 * DISC_R + SPAWN_LAT_MARGIN)) closing = own;
          if (!(dist > cfg.spawn_clearance + cfg.spawn_headway * closing)) { clear = false; break; }
        }
        if (!clear || free < 0) continue;
        const head = q.shift();
        const n = free;
        this.alive[n] = 1;
        this.x[n] = sx; this.y[n] = sy;
        this.th[n] = Math.atan2(sh, ch);
        this.v[n] = cfg.spawn_speed;
        this.kappa[n] = 0.0;
        this.arm[n] = k;
        this.man[n] = head.man;
        this.group[n] = head.grp;
        this.age[n] = 0.0;
        this.uid[n] = this.nextUid++;
        this.stats.spawned += 1;
        const [gx, gy] = this.gateXY(n);
        this.prevDist[n] = Math.hypot(sx - gx, sy - gy);
      }
    }

    // ---------------------------------------------------------------- state I/O (parity tests)
    setState(st) {
      const S = this.cfg.n_signal;
      const f64 = (a) => Float64Array.from(a);
      this.alive = Uint8Array.from(st.alive.map((b) => (b ? 1 : 0)));
      this.x = f64(st.x); this.y = f64(st.y); this.th = f64(st.th); this.v = f64(st.v);
      this.kappa = f64(st.kappa); this.age = f64(st.age); this.prevDist = f64(st.prev_dist);
      this.sig = f64(st.sig.flat()); if (this.sig.length !== this.N * S) throw new Error("sig shape");
      this.arm = Int32Array.from(st.arm); this.man = Int32Array.from(st.man);
      this.uid = Int32Array.from(st.uid); this.group = Int32Array.from(st.group);
      this.queues = st.queues.map((q) => q.map(([lat, man, grp]) => ({ lat, man, grp })));
      this.nextUid = st.next_uid; this.t = st.t;
      Object.assign(this.stats, st.stats || {});
    }

    getState() {
      const S = this.cfg.n_signal, arr = (a) => Array.from(a);
      const sig = [];
      for (let n = 0; n < this.N; n++) sig.push(arr(this.sig.subarray(n * S, n * S + S)));
      return {
        alive: arr(this.alive).map((b) => !!b), x: arr(this.x), y: arr(this.y), th: arr(this.th),
        v: arr(this.v), kappa: arr(this.kappa), age: arr(this.age), prev_dist: arr(this.prevDist), sig,
        arm: arr(this.arm), man: arr(this.man), uid: arr(this.uid), group: arr(this.group),
        queues: this.queues.map((q) => q.map((e) => [e.lat, e.man, e.grp])),
        next_uid: this.nextUid, t: this.t,
        stats: { ...this.stats, travel: this.stats.travel.map((r) => r.slice()) },
      };
    }
  }

  // ------------------------------------------------------------------ policy
  /** Normalises a policy JSON (policy.export_json format) for fast evaluation. */
  function loadPolicy(d) {
    if (!d || !Array.isArray(d.layers) || !d.layers.length) throw new Error("policy JSON has no layers");
    const layers = d.layers.map((l) => {
      const nin = l.W.length, nout = l.b.length;
      const W = new Float64Array(nin * nout);
      for (let i = 0; i < nin; i++) {
        if (l.W[i].length !== nout) throw new Error("policy layer shape mismatch");
        for (let o = 0; o < nout; o++) W[i * nout + o] = l.W[i][o];
      }
      return { nin, nout, W, b: Float64Array.from(l.b) };
    });
    for (let i = 1; i < layers.length; i++) {
      if (layers[i].nin !== layers[i - 1].nout) throw new Error("policy layers do not chain");
    }
    const sizes = d.sizes || [layers[0].nin, ...layers.map((l) => l.nout)];
    const inScale = d.in_scale ? Float64Array.from(d.in_scale) : null;
    if (inScale && inScale.length !== layers[0].nin) throw new Error("in_scale length mismatch");
    return { sizes, layers, inScale, cfg: d.cfg || {} };
  }

  /** tanh MLP on one observation row (obs[off .. off + nin)). */
  function forward(pol, obs, off = 0) {
    const l0 = pol.layers[0];
    let h = new Float64Array(l0.nin);
    for (let i = 0; i < l0.nin; i++) h[i] = pol.inScale ? obs[off + i] * pol.inScale[i] : obs[off + i];
    for (const l of pol.layers) {
      const out = new Float64Array(l.nout);
      for (let o = 0; o < l.nout; o++) {
        let acc = 0.0;
        for (let i = 0; i < l.nin; i++) acc += h[i] * l.W[i * l.nout + o];
        out[o] = Math.tanh(acc + l.b[o]);
      }
      h = out;
    }
    return h;
  }

  /** Actions for every car: group g uses policies[g] (the last one for g beyond the list);
   *  mute[g] forces that group's signal outputs to zero. Mirrors policy.PolicyDriver.act. */
  function act(sim, policies, mute, obs) {
    obs = obs || sim.observe();
    const D = obsDim(sim.cfg), A = actDim(sim.cfg);
    const out = new Float64Array(sim.N * A);
    for (let n = 0; n < sim.N; n++) {
      if (!sim.alive[n]) continue;
      const g = Math.min(sim.group[n], policies.length - 1);
      const a = forward(policies[g], obs, n * D);
      for (let q = 0; q < A; q++) out[n * A + q] = a[q];
      if (mute && mute[g]) for (let q = 2; q < A; q++) out[n * A + q] = 0.0;
    }
    return out;
  }

  return {
    RIGHT, STRAIGHT, LEFT, DISC_R, K_NEIGHBOURS, NEIGHBOUR_RANGE, OBS_DIM,
    defaultConfig, obsDim, actDim, rot, cornerDist, freeFlowTime, makeRng,
    Sim, loadPolicy, forward, act,
  };
});
