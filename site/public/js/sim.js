/* Smart Crossing: JS port of smartcrossing/sim.py (one environment, B = 1) and of
 * the MLP forward pass in smartcrossing/policy.py.
 *
 * Keep this file in lock-step with the Python code: tests/test_sim_js_parity.py
 * runs observe()/step() on states exported from Python and compares the results.
 * Randomness (arrivals, signal noise) uses a seeded JS PRNG; it is not meant to
 * reproduce numpy's streams.
 *
 * All numbers (SimConfig defaults, geometry and observation constants) come from
 * js/sim-config.js (globalThis.SC_SIM_CONFIG), which site/build.py generates from
 * sim.py (sim.js_constants); load it first.
 *
 * Works as a browser script (global `SmartCrossing`) and as a CommonJS module.
 */
(function (root, factory) {
  const api = factory(globalThis.SC_SIM_CONFIG);
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.SmartCrossing = api;
})(typeof self !== "undefined" ? self : this, function (C) {
  "use strict";
  if (!C) throw new Error("SC_SIM_CONFIG missing: load js/sim-config.js before sim.js");

  const { RIGHT, STRAIGHT, LEFT, DISC_OFFSETS, DISC_R, SPAWN_LAT_MARGIN, SPAWN_BACK, SPAWN_CURB_GAP, SPAWN_RIGHT_MIN, SPAWN_LAT_STEP, SPAWN_MAX_PER_STEP } = C;
  const { SPEED_TAU, N_RAYS, RAY_ANGLES, RAY_RANGE, CAR_LENGTH, CAR_WIDTH, HAZARD_PERIOD } = C;
  const { K_NEIGHBOURS, NEIGHBOUR_RANGE, POS_SCALE, GOAL_SCALE, NB_FEATURES, NB_FEATURES_LEGACY, BRAKE_ACC, LANE_WIDTH, OBS_DIM, OBS_RAYS, ARM_U } = C;
  // why a slot was cleared in the last step (sim.doneKind): { none: 0, exit: 1, ... }
  const DONE = Object.fromEntries(C.DONE_NAMES.map((name, i) => [name, i]));

  /** A fresh copy of the SimConfig dataclass defaults. */
  function defaultConfig() {
    return JSON.parse(JSON.stringify(C.defaults));
  }

  /** Features per neighbour before its lamps: brake light + hazard (cfg.brake_light) or the legacy acceleration. */
  const nbFeatures = (cfg) => (cfg.brake_light ? NB_FEATURES : NB_FEATURES_LEGACY);
  /** Neighbour slots in the observation (cfg.n_neighbours; 4 for every checkpoint before the set encoder). */
  const nNeighbours = (cfg) => cfg.n_neighbours || K_NEIGHBOURS;
  const obsDim = (cfg) => OBS_DIM + nNeighbours(cfg) * (nbFeatures(cfg) + cfg.n_signal);
  /** Steps a wreck stays (sim.wreck_steps). */
  const wreckSteps = (cfg) => Math.round(cfg.wreck_time / cfg.dt);

  /** Largest curvature the tyres allow at speed v (sim.grip_kappa). */
  function gripKappa(cfg, v) {
    if (!(cfg.a_lat_max > 0)) return cfg.kappa_max;
    return Math.min(cfg.kappa_max, cfg.a_lat_max / Math.max(v * v, 1e-9));
  }

  /** Lanes per direction (baselines / delay metric only) and lane-centre offsets (sim.lane_offsets). */
  function laneOffsets(cfg) {
    const n = Math.max(1, Math.round(cfg.W / LANE_WIDTH));
    const lw = cfg.W / n;
    return [cfg.W - lw / 2, lw / 2];
  }
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

  /** First t >= 0 at which the ray enters {u >= a, v >= b} (Infinity if never). */
  function quarterEntry(u, v, du, dv, a, b) {
    if (Math.abs(du) < 1e-12) du = 1e-12;
    if (Math.abs(dv) < 1e-12) dv = 1e-12;
    const t1 = (a - u) / du, t2 = (b - v) / dv;
    const lo = Math.max(du > 0 ? t1 : -Infinity, dv > 0 ? t2 : -Infinity, 0.0);
    const hi = Math.min(du < 0 ? t1 : Infinity, dv < 0 ? t2 : Infinity);
    return lo <= hi ? lo : Infinity;
  }

  /** First t >= 0 at which the ray (unit dir) enters the disc of radius R around the origin. */
  function discEntry(px, py, du, dv, R) {
    const bq = px * du + py * dv;
    const disc = bq * bq - (px * px + py * py - R * R);
    const sq = Math.sqrt(Math.max(disc, 0.0));
    const t0 = -bq - sq, t1 = -bq + sq;
    return disc >= 0 && t1 >= 0 ? Math.max(t0, 0.0) : Infinity;
  }

  /** Exact distance along a ray to the first off-road point, capped at tmax (sim.ray_edge_dist). */
  function rayEdgeDist(x, y, dx, dy, W, Rc, tmax = RAY_RANGE) {
    const c = W + Rc;
    let best = tmax;
    for (const sx of [1.0, -1.0]) {
      for (const sy of [1.0, -1.0]) {
        const u = sx * x, v = sy * y, du = sx * dx, dv = sy * dy;
        best = Math.min(best, quarterEntry(u, v, du, dv, W, c), quarterEntry(u, v, du, dv, c, W),
          discEntry(u - c, v - c, du, dv, Rc));
      }
    }
    return best;
  }

  /** Route-aware distance to the target gate (sim.route_dist). */
  function routeDist(cfg, arm, man, x, y) {
    const c0 = cfg.W + cfg.Rc;
    const [co, si] = rot(-arm);
    const cx = co * x - si * y, cy = si * x + co * y;
    const ut = ARM_U[man];
    const cand = [-cy, cx, cy, -cx];
    let a = 0;
    for (let k = 1; k < 4; k++) if (cand[k] > cand[a]) a = k;
    const ua = ARM_U[a];
    if (Math.abs(cx) <= c0 && Math.abs(cy) <= c0) return Math.hypot(cx - c0 * ut[0], cy - c0 * ut[1]) + cfg.L - c0;
    if (a === man) return cfg.L - (cx * ut[0] + cy * ut[1]);
    return Math.hypot(cx - c0 * ua[0], cy - c0 * ua[1]) + c0 * Math.hypot(ua[0] - ut[0], ua[1] - ut[1]) + cfg.L - c0;
  }

  /** numpy-style float remainder (result has the sign of the divisor). */
  function pyMod(a, b) {
    let r = a % b;
    if (r !== 0 && (r < 0) !== (b < 0)) r += b;
    return r;
  }

  function freeFlowTime(cfg, man) {
    if (man === STRAIGHT) return (2 * cfg.L - SPAWN_BACK) / cfg.v_max;
    const c = cfg.W + cfg.Rc;
    const armLen = 2 * (cfg.L - c) - SPAWN_BACK;
    const [offR, offL] = laneOffsets(cfg);
    const r = man === RIGHT ? c - offR : c + offL;
    const vArc = cfg.a_lat_max > 0 ? Math.min(cfg.v_max, Math.sqrt(cfg.a_lat_max * r)) : cfg.v_max;
    return armLen / cfg.v_max + ((Math.PI / 2) * r) / vArc;
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
      this.alive = new Uint8Array(N); // driving cars
      this.wreck = new Uint8Array(N); // crashed cars standing as obstacles (cfg.wreck_time)
      this.wreckLeft = new Int32Array(N); // steps until the wreck is removed
      this.x = new Float64Array(N);
      this.y = new Float64Array(N);
      this.th = new Float64Array(N);
      this.v = new Float64Array(N);
      this.kappa = new Float64Array(N); // actual curvature (rate and grip limited)
      this.acc = new Float64Array(N); // actual acceleration of the last step
      this.sig = new Float64Array(N * S);
      this.arm = new Int32Array(N);
      this.man = new Int32Array(N);
      this.age = new Float64Array(N);
      this.uid = new Int32Array(N).fill(-1);
      this.group = new Int32Array(N);
      this.prevDist = new Float64Array(N);
      this.carReward = new Float64Array(N); // per-step outputs (see sim.py step())
      this.doneKind = new Int8Array(N);
      this.doneUid = new Int32Array(N).fill(-1);
      this.queueCost = 0.0;
      // per-reset demand when cfg.demand_range is set (else cfg.demand is used live)
      this.demand = null;
      if (this.cfg.demand_range) {
        const [lo, hi] = this.cfg.demand_range;
        this.demand = lo + (hi - lo) * this.rng();
      }
      this.queues = [[], [], [], []]; // FIFO of {u, man, grp} per arm (u: lateral draw)
      this.nextUid = 0;
      this.t = 0.0;
      this.events = [];
      this.stats = { arrived: 0, spawned: 0, exits: 0, wrong_exits: 0, crashes: 0, wreck_hits: 0, offroad: 0, timeouts: 0, travel: [] };
    }

    tgt(n) { return (this.arm[n] + this.man[n]) % 4; }

    distToGate(n) { return routeDist(this.cfg, this.arm[n], this.man[n], this.x[n], this.y[n]); }

    queueLength(k) { return k === undefined ? this.queues.reduce((a, q) => a + q.length, 0) : this.queues[k].length; }

    // ---------------------------------------------------------------- observe
    /** Returns a Float64Array of shape (N, obsDim) in row-major order. */
    observe() {
      const cfg = this.cfg, N = this.N, S = cfg.n_signal, L = cfg.L, W = cfg.W;
      const NF = nbFeatures(cfg), D = obsDim(cfg), blk = NF + S;
      const obs = new Float64Array(N * D);
      const cand = [];
      for (let i = 0; i < N; i++) {
        if (!this.alive[i]) continue;
        const o = i * D;
        const x = this.x[i], y = this.y[i], th = this.th[i];
        const [c, s] = rot(-this.arm[i]);
        const cx = c * x - s * y, cy = s * x + c * y;
        const cth = th - this.arm[i] * (Math.PI / 2);
        obs[o] = cx / POS_SCALE;
        obs[o + 1] = cy / POS_SCALE;
        obs[o + 2] = Math.cos(cth);
        obs[o + 3] = Math.sin(cth);
        obs[o + 4] = this.v[i] / cfg.v_max;
        const m = this.man[i];
        obs[o + 5] = m === RIGHT ? 1 : 0;
        obs[o + 6] = m === STRAIGHT ? 1 : 0;
        obs[o + 7] = m === LEFT ? 1 : 0;
        const ut = ARM_U[this.tgt(i)], c0 = W + cfg.Rc;
        const dx = L * ut[0] - x, dy = L * ut[1] - y; // to the exit gate centre
        const mx = c0 * ut[0] - x, my = c0 * ut[1] - y; // to the target arm's mouth centre
        const ct = Math.cos(th), st = Math.sin(th);
        obs[o + 8] = (ct * dx + st * dy) / GOAL_SCALE;
        obs[o + 9] = (-st * dx + ct * dy) / GOAL_SCALE;
        obs[o + 10] = (ct * mx + st * my) / GOAL_SCALE;
        obs[o + 11] = (-st * mx + ct * my) / GOAL_SCALE;
        for (let r = 0; r < N_RAYS; r++) {
          const ang = th + RAY_ANGLES[r];
          const ca = Math.cos(ang), sa = Math.sin(ang);
          obs[o + OBS_RAYS + r] = rayEdgeDist(x, y, ca, sa, W, cfg.Rc) / RAY_RANGE;
        }
        obs[o + OBS_RAYS + N_RAYS] = this.kappa[i] / cfg.kappa_max;
        obs[o + OBS_RAYS + N_RAYS + 1] = gripKappa(cfg, this.v[i]) / cfg.kappa_max;
        // neighbours: nearest K cars (driving or wrecked) within range
        cand.length = 0;
        for (let j = 0; j < N; j++) {
          if (j === i || !(this.alive[j] || this.wreck[j])) continue;
          const nx = this.x[j] - x, ny = this.y[j] - y;
          const d = Math.hypot(nx, ny);
          if (d < cfg.neighbour_vis) cand.push([d, j, nx, ny]);
        }
        cand.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        for (let k = 0; k < Math.min(nNeighbours(cfg), cand.length); k++) {
          const [, j, nx, ny] = cand[k];
          const p = o + OBS_DIM + k * blk;
          const nth = this.th[j] - th;
          obs[p] = (ct * nx + st * ny) / NEIGHBOUR_RANGE;
          obs[p + 1] = (-st * nx + ct * ny) / NEIGHBOUR_RANGE;
          obs[p + 2] = Math.cos(nth);
          obs[p + 3] = Math.sin(nth);
          obs[p + 4] = this.v[j] / cfg.v_max;
          obs[p + 5] = 1.0;
          obs[p + 6] = this.kappa[j] / cfg.kappa_max;
          if (cfg.brake_light) {
            obs[p + 7] = this.acc[j] < BRAKE_ACC ? 1.0 : 0.0; // brake lights
            obs[p + 8] = this.wreck[j] ? 1.0 : 0.0; // hazard lights (a wreck)
          } else obs[p + 7] = this.acc[j] / cfg.brake_max; // legacy interface: acceleration
          for (let q = 0; q < S; q++) {
            let sv = 0.0;
            if (cfg.signal_enabled) {
              sv = this.sig[j * S + q];
              if (cfg.signal_noise > 0) sv += cfg.signal_noise * this.noiseRng.normal();
            }
            obs[p + NF + q] = sv;
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
        const v0 = this.v[n];
        let v;
        if (cfg.speed_control) {
          const acc = Math.min(Math.max(((a0 + 1.0) * 0.5 * cfg.v_max - v0) / SPEED_TAU, -cfg.brake_max), cfg.acc_max);
          v = Math.min(Math.max(v0 + acc * dt, 0.0), cfg.v_max); // no reverse in this interface
        } else {
          // pedal (sim.py): + accelerates forward / brakes a reversing car, - brakes a forward car /
          // accelerates backwards once stopped; never crosses 0 within one step
          const acc = a0 > 0 ? a0 * (v0 < 0 ? cfg.brake_max : cfg.acc_max) : a0 * (v0 > 0 ? cfg.brake_max : cfg.acc_max);
          v = Math.min(Math.max(v0 + acc * dt, -cfg.v_reverse), cfg.v_max);
          if ((v0 > 0 && v < 0) || (v0 < 0 && v > 0)) v = 0.0;
        }
        const dk = cfg.kappa_rate * dt;
        const kap = this.kappa[n] + Math.min(Math.max(a1 * cfg.kappa_max - this.kappa[n], -dk), dk);
        const kg = gripKappa(cfg, v); // tyre grip at the new speed (after the steering-rate limit)
        this.kappa[n] = alive ? Math.min(Math.max(kap, -kg), kg) : 0.0;
        for (let q = 0; q < S; q++) this.sig[n * S + q] = alive ? clip(action[n * A + 2 + q]) : 0.0;
        this.acc[n] = alive ? (v - this.v[n]) / dt : 0.0;
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
      const ws = wreckSteps(cfg);
      if (ws > 0) { // wrecks whose time is up disappear (before the collision check)
        const gone = new Uint8Array(N);
        let any = false;
        for (let n = 0; n < N; n++) {
          if (!this.wreck[n]) continue;
          this.wreckLeft[n] -= 1;
          if (this.wreckLeft[n] <= 0) { gone[n] = 1; any = true; }
        }
        if (any) this._remove(gone);
      }
      const cr = this.carReward, dk = this.doneKind;
      for (let n = 0; n < N; n++) {
        const d = this.distToGate(n);
        cr[n] = this.alive[n] ? -cfg.r_time * dt + cfg.r_progress * (this.prevDist[n] - d) : 0.0;
        this.prevDist[n] = d;
        dk[n] = 0;
        this.doneUid[n] = this.alive[n] ? this.uid[n] : -1;
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
          cr[n] += cfg.r_exit; dk[n] = DONE.exit;
          this.stats.exits += 1;
          this.stats.travel.push([this.t, this.age[n], freeFlowTime(cfg, this.man[n]), this.group[n]]);
          this.events.push({ kind: "exit", n, uid: this.uid[n], x: this.x[n], y: this.y[n] });
        } else {
          cr[n] += cfg.r_wrong_exit; dk[n] = DONE.wrong_exit;
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
      const off = new Uint8Array(N), col = new Uint8Array(N), vrel = new Float64Array(N), hitWreck = new Uint8Array(N);
      const near2 = (2 * DISC_OFFSETS[DISC_OFFSETS.length - 1] + 2 * DISC_R) ** 2, hit2 = (2 * DISC_R) ** 2;
      for (let i = 0; i < N; i++) {
        if (!this.alive[i]) continue;
        for (let k = 0; k < 3; k++) {
          if (cornerDist(ox[i * 3 + k], oy[i * 3 + k], cfg.W, cfg.Rc) < cfg.Rc + DISC_R) { off[i] = 1; break; }
        }
        for (let j = 0; j < N; j++) { // partners: driving cars and wrecks (wrecks are not hit again)
          if (j === i || !(this.alive[j] || this.wreck[j])) continue;
          const dcx = this.x[i] - this.x[j], dcy = this.y[i] - this.y[j];
          if (!(dcx ** 2 + dcy ** 2 < near2)) continue;
          let hit = false;
          for (let a = 0; a < 3 && !hit; a++) {
            for (let b = 0; b < 3; b++) {
              const dx = ox[i * 3 + a] - ox[j * 3 + b], dy = oy[i * 3 + a] - oy[j * 3 + b];
              if (dx ** 2 + dy ** 2 < hit2) { hit = true; break; }
            }
          }
          if (hit) {
            col[i] = 1; // impact speed = |relative velocity|, max over partners
            const rvx = this.v[i] * Math.cos(this.th[i]) - this.v[j] * Math.cos(this.th[j]);
            const rvy = this.v[i] * Math.sin(this.th[i]) - this.v[j] * Math.sin(this.th[j]);
            vrel[i] = Math.max(vrel[i], Math.hypot(rvx, rvy));
            if (this.wreck[j]) hitWreck[i] = 1;
          }
        }
      }
      const bad = new Uint8Array(N), crashed = new Uint8Array(N);
      for (let n = 0; n < N; n++) {
        if (off[n]) {
          this.events.push({ kind: "offroad", n, uid: this.uid[n], x: this.x[n], y: this.y[n] });
          cr[n] += cfg.r_offroad - cfg.r_offroad_speed * this.v[n]; dk[n] = DONE.offroad; this.stats.offroad += 1; bad[n] = 1;
        }
      }
      for (let n = 0; n < N; n++) {
        if (col[n] && !off[n]) {
          this.events.push({ kind: "crash", n, uid: this.uid[n], x: this.x[n], y: this.y[n] });
          cr[n] += cfg.r_crash - cfg.r_crash_speed * vrel[n]; dk[n] = DONE.crash; this.stats.crashes += 1;
          if (hitWreck[n]) this.stats.wreck_hits += 1;
          if (ws > 0) crashed[n] = 1; else bad[n] = 1;
        }
      }
      this._remove(bad); // off-road cars (and crashed ones without wrecks) vanish
      this._makeWreck(crashed, ws);

      const tmo = new Uint8Array(N);
      for (let n = 0; n < N; n++) {
        if (this.alive[n] && this.age[n] > cfg.max_age) {
          tmo[n] = 1; cr[n] += cfg.r_timeout; dk[n] = DONE.timeout; this.stats.timeouts += 1;
          this.events.push({ kind: "timeout", n, uid: this.uid[n], x: this.x[n], y: this.y[n] });
        }
      }
      this._remove(tmo);

      this._arrivals();
      let reward = 0.0;
      for (let n = 0; n < N; n++) reward += cr[n];
      this.queueCost = cfg.r_queue * dt * this.queueLength();
      return reward - this.queueCost;
    }

    /** Crashed cars become standing wrecks (speed 0, lamps off, hazard lights on; uid kept). */
    _makeWreck(mask, ws) {
      const S = this.cfg.n_signal;
      for (let n = 0; n < this.N; n++) {
        if (!mask[n]) continue;
        this.alive[n] = 0;
        this.wreck[n] = 1;
        this.wreckLeft[n] = ws;
        this.v[n] = 0.0;
        this.acc[n] = 0.0;
        this.kappa[n] = 0.0;
        for (let q = 0; q < S; q++) this.sig[n * S + q] = 0.0;
      }
    }

    _remove(mask) {
      const S = this.cfg.n_signal;
      for (let n = 0; n < this.N; n++) {
        if (!mask[n]) continue;
        this.alive[n] = 0;
        this.wreck[n] = 0;
        this.wreckLeft[n] = 0;
        this.v[n] = 0.0;
        this.acc[n] = 0.0;
        for (let q = 0; q < S; q++) this.sig[n * S + q] = 0.0;
        this.age[n] = 0.0;
        this.uid[n] = -1;
        this.x[n] = 1e4;
        this.y[n] = 1e4;
      }
    }

    // ---------------------------------------------------------------- spawning
    /** Candidate lateral spawn offsets (see sim.py spawn_lats). */
    spawnLats() {
      const cfg = this.cfg;
      const lo = cfg.spawn_right ? SPAWN_RIGHT_MIN : -(cfg.W - SPAWN_CURB_GAP);
      const hi = cfg.W - SPAWN_CURB_GAP;
      const m = Math.max(Math.round((hi - lo) / SPAWN_LAT_STEP), 1);
      const lats = new Float64Array(m + 1);
      for (let i = 0; i <= m; i++) lats[i] = lo + ((hi - lo) * i) / m;
      return lats;
    }

    /** Queue arrivals (u, manoeuvre, group drawn at arrival), then release queued cars at their
     *  own spot or, if blocked, a uniformly chosen clear one, several per arm and step (sim.py _arrivals). */
    _arrivals() {
      const cfg = this.cfg, N = this.N, rng = this.rng;
      const demand = this.demand === null ? cfg.demand : this.demand;
      if (demand > 0) {
        for (let k = 0; k < 4; k++) {
          const arrive = rng() < demand * cfg.dt;
          const u = rng();
          const man = 1 + rng.choice(cfg.turn_probs);
          const grp = rng.choice(this.groupProbs);
          if (arrive) {
            this.queues[k].push({ u, man, grp });
            this.stats.arrived += 1;
          }
        }
      }
      // velocities of existing cars (spawned cars are added as they enter)
      const vx = new Float64Array(N), vy = new Float64Array(N);
      for (let n = 0; n < N; n++) {
        vx[n] = this.v[n] * Math.cos(this.th[n]);
        vy[n] = this.v[n] * Math.sin(this.th[n]);
      }
      const lats = this.spawnLats(), M = lats.length;
      const back = -cfg.L + SPAWN_BACK;
      const clear = new Uint8Array(M);
      for (let k = 0; k < 4; k++) {
        const q = this.queues[k];
        const [c, s] = rot(k);
        const h = Math.PI / 2 + k * (Math.PI / 2);
        const hn = Math.atan2(Math.sin(h), Math.cos(h));
        const ch = Math.cos(h), sh = Math.sin(h);
        for (let r = 0; r < SPAWN_MAX_PER_STEP && q.length; r++) {
          let free = -1;
          for (let j = 0; j < N; j++) if (!this.alive[j] && !this.wreck[j]) { free = j; break; }
          if (free < 0) break;
          let cnt = 0;
          for (let i = 0; i < M; i++) {
            const sx = c * lats[i] - s * back, sy = s * lats[i] + c * back;
            let ok = 1;
            for (let j = 0; j < N; j++) { // wrecks block like standing cars
              if (!this.alive[j] && !this.wreck[j]) continue;
              const dx = this.x[j] - sx, dy = this.y[j] - sy;
              const dist = Math.hypot(dx, dy);
              const dd = Math.max(dist, 1e-6);
              const rvx = cfg.spawn_speed * ch - vx[j], rvy = cfg.spawn_speed * sh - vy[j];
              let closing = Math.max((rvx * dx + rvy * dy) / dd, 0.0);
              const own = Math.max((cfg.spawn_speed * (ch * dx + sh * dy)) / dd, 0.0);
              const latOff = Math.abs(-sh * dx + ch * dy);
              if (!(latOff < 2 * DISC_R + SPAWN_LAT_MARGIN)) closing = own;
              if (!(dist > cfg.spawn_clearance + cfg.spawn_headway * closing)) { ok = 0; break; }
            }
            clear[i] = ok;
            cnt += ok;
          }
          if (!cnt) break;
          const head = q.shift();
          // own spot (u across the range) if clear, else the clear spot of the same rank u
          let i = Math.floor(head.u * (M - 1) + 0.5);
          if (!clear[i]) {
            const pick = Math.min(Math.floor(head.u * cnt), cnt - 1);
            i = 0;
            for (let seen = -1; i < M; i++) if (clear[i] && ++seen === pick) break;
          }
          const sx = c * lats[i] - s * back, sy = s * lats[i] + c * back;
          const n = free;
          this.alive[n] = 1;
          this.x[n] = sx; this.y[n] = sy;
          this.th[n] = hn;
          this.v[n] = cfg.spawn_speed;
          vx[n] = cfg.spawn_speed * Math.cos(hn); vy[n] = cfg.spawn_speed * Math.sin(hn);
          this.kappa[n] = 0.0;
          this.acc[n] = 0.0;
          this.arm[n] = k;
          this.man[n] = head.man;
          this.group[n] = head.grp;
          this.age[n] = 0.0;
          this.uid[n] = this.nextUid++;
          this.stats.spawned += 1;
          this.prevDist[n] = routeDist(cfg, k, head.man, sx, sy);
        }
      }
    }

    // ---------------------------------------------------------------- state I/O (parity tests)
    setState(st) {
      const S = this.cfg.n_signal;
      const f64 = (a) => Float64Array.from(a);
      this.alive = Uint8Array.from(st.alive.map((b) => (b ? 1 : 0)));
      this.wreck = Uint8Array.from((st.wreck || st.alive.map(() => false)).map((b) => (b ? 1 : 0)));
      this.wreckLeft = Int32Array.from(st.wreck_left || st.alive.map(() => 0));
      this.x = f64(st.x); this.y = f64(st.y); this.th = f64(st.th); this.v = f64(st.v);
      this.kappa = f64(st.kappa); this.acc = f64(st.acc); this.age = f64(st.age); this.prevDist = f64(st.prev_dist);
      this.sig = f64(st.sig.flat()); if (this.sig.length !== this.N * S) throw new Error("sig shape");
      this.arm = Int32Array.from(st.arm); this.man = Int32Array.from(st.man);
      this.uid = Int32Array.from(st.uid); this.group = Int32Array.from(st.group);
      this.queues = st.queues.map((q) => q.map(([u, man, grp]) => ({ u, man, grp })));
      this.nextUid = st.next_uid; this.t = st.t;
      Object.assign(this.stats, st.stats || {});
    }

    getState() {
      const S = this.cfg.n_signal, arr = (a) => Array.from(a);
      const sig = [];
      for (let n = 0; n < this.N; n++) sig.push(arr(this.sig.subarray(n * S, n * S + S)));
      return {
        alive: arr(this.alive).map((b) => !!b), wreck: arr(this.wreck).map((b) => !!b), wreck_left: arr(this.wreckLeft), x: arr(this.x), y: arr(this.y), th: arr(this.th),
        v: arr(this.v), kappa: arr(this.kappa), acc: arr(this.acc), age: arr(this.age), prev_dist: arr(this.prevDist), sig,
        arm: arr(this.arm), man: arr(this.man), uid: arr(this.uid), group: arr(this.group),
        queues: this.queues.map((q) => q.map((e) => [e.u, e.man, e.grp])),
        next_uid: this.nextUid, t: this.t,
        stats: { ...this.stats, travel: this.stats.travel.map((r) => r.slice()) },
      };
    }
  }

  // ------------------------------------------------------------------ policy
  /** (name, shape) of the parameter blocks of a set encoder, in flat order (policy_set.param_shapes). */
  function setParamShapes(ar) {
    const out = [], dims = [ar.F, ...ar.phi];
    for (let i = 0; i + 1 < dims.length; i++) out.push([`phi${i}.W`, [dims[i], dims[i + 1]]], [`phi${i}.b`, [dims[i + 1]]]);
    const e = ar.phi[ar.phi.length - 1];
    if (ar.pool === "attn") {
      const H = ar.heads, qin = ar.n_ego + ar.n_extra;
      out.push(["q.W", [qin, H * ar.dk]], ["q.b", [H * ar.dk]], ["k.W", [e, H * ar.dk]], ["v.W", [e, H * ar.dv]],
        ["v.b", [H * ar.dv]]);
    }
    const pooled = ar.pool === "attn" ? ar.heads * ar.dv : e;
    const ts = [ar.n_ego + ar.n_extra + pooled, ...ar.hidden, ar.n_out];
    for (let i = 0; i + 1 < ts.length; i++) out.push([`trunk${i}.W`, [ts[i], ts[i + 1]]], [`trunk${i}.b`, [ts[i + 1]]]);
    return out;
  }

  /** Set-encoder policy (policy_set.py; export_json gives its arch + flat params). */
  function loadSetPolicy(d) {
    const ar = d.arch, flat = Float64Array.from(d.params), P = {};
    let i = 0;
    for (const [name, shape] of setParamShapes(ar)) {
      const n = shape.reduce((a, b) => a * b, 1);
      P[name] = flat.subarray(i, i + n);
      i += n;
    }
    if (i !== flat.length) throw new Error(`set policy: ${flat.length} params, arch needs ${i}`);
    const nin = ar.n_ego + ar.K * ar.F + ar.n_extra;
    const inScale = d.in_scale ? Float64Array.from(d.in_scale) : null;
    if (inScale && inScale.length !== nin) throw new Error("in_scale length mismatch");
    return { kind: "set", arch: ar, P, nin, sizes: d.sizes || [nin, ...ar.hidden, ar.n_out], inScale, cfg: d.cfg || {} };
  }

  /** y = x W + b for a row-major W (nin x nout); x is a plain array / typed array of length nin. */
  function affine(x, W, b, nin, nout) {
    const y = new Float64Array(nout);
    for (let o = 0; o < nout; o++) {
      let acc = 0.0;
      for (let i = 0; i < nin; i++) acc += x[i] * W[i * nout + o];
      y[o] = acc + (b ? b[o] : 0.0);
    }
    return y;
  }

  /** Set-encoder forward on one (scaled) input row x: linear output (policy_set.forward).
   *  tr (optional object) receives the intermediate values for the brain view: ego, present[K],
   *  phi[K][layer] (per-neighbour net activations), attn[H][K] (attention weights, 0 for absent neighbours),
   *  pooled, trunk (hidden activations) and the linear output. */
  function setForwardRow(pol, x, tr) {
    const ar = pol.arch, P = pol.P, E = ar.n_ego, K = ar.K, F = ar.F;
    const ego = new Float64Array(E + ar.n_extra);
    for (let i = 0; i < E; i++) ego[i] = x[i];
    for (let i = 0; i < ar.n_extra; i++) ego[E + i] = x[E + K * F + i];
    const hs = [], present = [], phiLayers = [];
    for (let k = 0; k < K; k++) {
      let h = x.subarray(E + k * F, E + (k + 1) * F);
      present.push(h[ar.present] !== 0);
      let nin = F;
      const ls = [];
      for (let l = 0; l < ar.phi.length; l++) {
        h = affine(h, P[`phi${l}.W`], P[`phi${l}.b`], nin, ar.phi[l]).map(Math.tanh);
        nin = ar.phi[l];
        if (tr) ls.push(h);
      }
      hs.push(h);
      phiLayers.push(ls);
    }
    const e = ar.phi[ar.phi.length - 1];
    let pooled;
    if (ar.pool === "attn") {
      const H = ar.heads, dk = ar.dk, dv = ar.dv;
      const q = affine(ego, P["q.W"], P["q.b"], ego.length, H * dk);
      const ks = hs.map((h) => affine(h, P["k.W"], null, e, H * dk));
      const vs = hs.map((h) => affine(h, P["v.W"], P["v.b"], e, H * dv));
      pooled = new Float64Array(H * dv);
      const sc = 1 / Math.sqrt(dk);
      if (tr) tr.attn = [];
      for (let hd = 0; hd < H; hd++) {
        const s = new Float64Array(K);
        let mx = -Infinity;
        for (let k = 0; k < K; k++) {
          let acc = 0.0;
          for (let d = 0; d < dk; d++) acc += q[hd * dk + d] * ks[k][hd * dk + d];
          s[k] = present[k] ? acc * sc : -1e9;
          mx = Math.max(mx, s[k]);
        }
        let sum = 0.0;
        for (let k = 0; k < K; k++) { s[k] = present[k] ? Math.exp(s[k] - mx) : 0.0; sum += s[k]; }
        sum = Math.max(sum, 1e-30);
        if (tr) tr.attn.push(Array.from(s, (v) => v / sum));
        for (let k = 0; k < K; k++) {
          if (!s[k]) continue;
          const w = s[k] / sum;
          for (let d = 0; d < dv; d++) pooled[hd * dv + d] += w * vs[k][hd * dv + d];
        }
      }
    } else {
      pooled = new Float64Array(e);
      let n = 0;
      if (ar.pool === "max") pooled.fill(-2.0);
      for (let k = 0; k < K; k++) {
        if (!present[k]) continue;
        n++;
        for (let d = 0; d < e; d++) pooled[d] = ar.pool === "max" ? Math.max(pooled[d], hs[k][d]) : pooled[d] + hs[k][d];
      }
      if (ar.pool === "max") { if (!n) pooled.fill(0.0); } else for (let d = 0; d < e; d++) pooled[d] /= Math.max(n, 1);
    }
    let z = new Float64Array(ego.length + pooled.length);
    z.set(ego, 0);
    z.set(pooled, ego.length);
    const ts = [z.length, ...ar.hidden, ar.n_out];
    if (tr) Object.assign(tr, { ego, present, phi: phiLayers, pooled, trunk: [] });
    for (let l = 0; l + 1 < ts.length; l++) {
      z = affine(z, P[`trunk${l}.W`], P[`trunk${l}.b`], ts[l], ts[l + 1]);
      if (l + 2 < ts.length) z = z.map(Math.tanh);
      if (tr && l + 2 < ts.length) tr.trunk.push(z);
    }
    if (tr) tr.linear = z;
    return z;
  }

  /** Set-encoder forward of one observation row with all intermediate values (see setForwardRow);
   *  trace.out equals forward(pol, obs, off). */
  function setTrace(pol, obs, off = 0) {
    const x = new Float64Array(pol.nin);
    for (let i = 0; i < pol.nin; i++) x[i] = pol.inScale ? obs[off + i] * pol.inScale[i] : obs[off + i];
    const tr = { x };
    tr.out = setForwardRow(pol, x, tr).map(Math.tanh);
    return tr;
  }

  /** Normalises a policy JSON (policy.export_json format) for fast evaluation. */
  function loadPolicy(d) {
    if (d && d.arch && d.arch.type === "set") return loadSetPolicy(d);
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

  /** Policy (tanh MLP, or set encoder + tanh) on one observation row (obs[off .. off + nin)). */
  function forward(pol, obs, off = 0) {
    if (pol.kind === "set") {
      const x = new Float64Array(pol.nin);
      for (let i = 0; i < pol.nin; i++) x[i] = pol.inScale ? obs[off + i] * pol.inScale[i] : obs[off + i];
      return setForwardRow(pol, x).map(Math.tanh);
    }
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
    RIGHT, STRAIGHT, LEFT, DISC_R, K_NEIGHBOURS, NEIGHBOUR_RANGE, OBS_DIM, NB_FEATURES, NB_FEATURES_LEGACY, BRAKE_ACC,
    POS_SCALE, GOAL_SCALE, RAY_RANGE, CAR_LENGTH, CAR_WIDTH, HAZARD_PERIOD,
    DONE, defaultConfig, obsDim, actDim, nbFeatures, nNeighbours, wreckSteps, rot, cornerDist, rayEdgeDist, routeDist, freeFlowTime,
    gripKappa, laneOffsets, makeRng,
    Sim, loadPolicy, forward, act, setTrace,
  };
});
