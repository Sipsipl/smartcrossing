/* Side-by-side replays ("Vergleich"): pre-recorded drives (site/build.py, window.SC_REPLAYS)
 * of several drivers in the same traffic, played back in sync with interpolation. */
(function () {
  "use strict";
  const SCX = (window.SCX = window.SCX || {});
  const KIND = { exit: 1, crash: 3, offroad: 4 };
  const MARK_FADE = 5;

  function b64(s, width) {
    const bin = atob(s), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return width === 2 ? new Int16Array(u8.buffer) : new Int8Array(u8.buffer);
  }

  /** Decode one recorded replay into per-car float arrays. */
  SCX.decodeReplay = function (r, q) {
    const F = r.lamps ? 5 : 3;
    const tracks = r.tracks.map(([t0, man, end, kind, x0, y0, th0, width, data]) => {
      const d = b64(data, width), len = d.length / F;
      const x = new Float32Array(len), y = new Float32Array(len), th = new Float32Array(len);
      const l0 = r.lamps ? new Float32Array(len) : null, l1 = r.lamps ? new Float32Array(len) : null;
      let qx = x0, qy = y0, qt = th0;
      for (let i = 0; i < len; i++) {
        qx += d[i * F]; qy += d[i * F + 1]; qt += d[i * F + 2];
        x[i] = qx * q.pos; y[i] = qy * q.pos; th[i] = qt * q.th;
        if (l0) { l0[i] = d[i * F + 3] / q.lamp; l1[i] = d[i * F + 4] / q.lamp; }
      }
      return { t0, man, end, kind, len, x, y, th, l0, l1 };
    });
    const qb = b64(r.queue, 1), queue = new Uint8Array(qb.buffer);
    const ends = (k) => tracks.filter((t) => t.kind === k && t.end >= 0).map((t) => t.end).sort((a, b) => a - b);
    return {
      dt: r.dt, steps: r.steps, lamps: r.lamps, tracks, queue,
      marks: r.marks.map(([k, kind, x, y]) => ({ k, kind: kind === 1 ? "crash" : "offroad", x: x * q.pos, y: y * q.pos })),
      exits: ends(KIND.exit), bad: ends(KIND.crash).concat(ends(KIND.offroad)).sort((a, b) => a - b),
    };
  };

  const countUpTo = (arr, k) => { let n = 0; while (n < arr.length && arr[n] <= k) n++; return n; };

  /** One canvas showing a decoded replay at time t (seconds). */
  class ReplayView {
    constructor(canvas, world) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.world = world; // {L, W, Rc}
      this.rep = null;
      this.t = 0;
      new ResizeObserver(() => this.resize()).observe(canvas);
      this.resize();
    }

    resize() {
      const r = this.canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      this.cssW = r.width;
      this.canvas.width = Math.max(1, Math.round(r.width * dpr));
      this.canvas.height = Math.max(1, Math.round(r.height * dpr));
      this.draw(this.t);
    }

    counts(t) {
      const r = this.rep;
      if (!r) return { thr: null, bad: 0, exits: 0 };
      const k = Math.floor(t / r.dt);
      const exits = countUpTo(r.exits, k);
      return { exits, bad: countUpTo(r.bad, k), thr: t >= 5 ? (exits / t) * 60 : null };
    }

    draw(t) {
      this.t = t;
      const r = this.rep, ctx = this.ctx, S = this.canvas.width;
      if (!r || !S) return;
      const { L, W } = this.world;
      const half = this.cssW < 260 ? Math.min(L + 1.5, 30) : L + 1.5; // small canvas: zoom on the box
      const px = S / (2 * half);
      ctx.setTransform(px, 0, 0, -px, S / 2, S / 2);
      SCX.drawRoad(ctx, this.world);
      const kf = t / r.dt, k = Math.floor(kf), a = kf - k;
      for (const tr of r.tracks) {
        const i = k - tr.t0;
        if (i < 0 || i >= tr.len) continue;
        let x = tr.x[i], y = tr.y[i], th = tr.th[i], lamp = null;
        let s0 = tr.l0 ? tr.l0[i] : 0, s1 = tr.l1 ? tr.l1[i] : 0;
        if (i + 1 < tr.len) {
          x += (tr.x[i + 1] - x) * a; y += (tr.y[i + 1] - y) * a; th += (tr.th[i + 1] - th) * a;
          if (tr.l0) { s0 += (tr.l0[i + 1] - s0) * a; s1 += (tr.l1[i + 1] - s1) * a; }
        }
        if (tr.l0) lamp = SCX.lampRGB(s0, s1);
        SCX.drawCar(ctx, x, y, th, tr.man, null, lamp);
      }
      for (const m of r.marks) {
        if (m.k > k) break;
        SCX.drawMark(ctx, m.x, m.y, m.kind, 1 - (t - m.k * r.dt) / MARK_FADE);
      }
      // waiting cars per arm
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      const dpr = window.devicePixelRatio || 1;
      ctx.font = `${Math.round(11 * dpr)}px ui-monospace, Menlo, monospace`;
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.textBaseline = "middle";
      const kk = Math.min(Math.max(k, 0), r.steps - 1);
      for (let arm = 0; arm < 4; arm++) {
        const q = r.queue[kk * 4 + arm];
        if (!q) continue;
        const [c, s] = window.SmartCrossing.rot(arm);
        const u = W + 1.5, v = -half + 5;
        const wx = c * u - s * v, wy = s * u + c * v;
        ctx.textAlign = arm === 0 || arm === 3 ? "left" : "right";
        ctx.fillText(`+${q}`, S / 2 + wx * px, S / 2 - wy * px);
      }
    }
  }
  SCX.ReplayView = ReplayView;
})();
