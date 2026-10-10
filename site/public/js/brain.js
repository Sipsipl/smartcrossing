/* "Das Gehirn": our own forward pass (mirrors smartcrossing/policy.py forward:
 * x = obs * in_scale, h = tanh(h W + b) per layer) that keeps every layer's
 * activations, and a light canvas drawing of the network. */
(function () {
  "use strict";
  const SCX = (window.SCX = window.SCX || {});

  /** raw: policy JSON (layers[{W[in][out], b[out]}], in_scale). Returns [x, h1, ..., out]. */
  SCX.forwardTrace = function (raw, obs, off = 0) {
    const n0 = raw.layers[0].W.length;
    const x = new Float64Array(n0);
    for (let i = 0; i < n0; i++) x[i] = obs[off + i] * (raw.in_scale ? raw.in_scale[i] : 1);
    const acts = [x];
    let h = x;
    for (const l of raw.layers) {
      const nout = l.b.length, out = new Float64Array(nout);
      for (let o = 0; o < nout; o++) {
        let s = l.b[o];
        for (let i = 0; i < h.length; i++) s += h[i] * l.W[i][o];
        out[o] = Math.tanh(s);
      }
      acts.push(out);
      h = out;
    }
    return acts;
  };

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  function hexToRgb(h) {
    const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(h);
    return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [128, 128, 128];
  }

  class NetView {
    /** layout: SC_DATA.layout; raw: policy JSON; edgesPer: strongest incoming edges per neuron */
    constructor(canvas, layout, raw, opts) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.layout = layout;
      this.opts = Object.assign({ edgesPer: 3, onHover: null }, opts);
      this.trace = null;
      this.setPolicy(raw);
      new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
      canvas.addEventListener("mousemove", (e) => this._hover(e));
      canvas.addEventListener("click", (e) => this._hover(e));
      canvas.addEventListener("mouseleave", () => { this.hoverN = null; this.draw(); if (this.opts.onHover) this.opts.onHover(null); });
    }

    setPolicy(raw) {
      this.raw = raw;
      this.sizes = [raw.layers[0].W.length, ...raw.layers.map((l) => l.b.length)];
      // strongest incoming weights per neuron (effective weights: input scale folded in)
      this.edges = raw.layers.map((l, li) => {
        const nin = l.W.length, nout = l.b.length, list = [];
        const per = li === raw.layers.length - 1 ? this.opts.edgesPer * 2 : this.opts.edgesPer;
        let wmax = 1e-9;
        for (let o = 0; o < nout; o++) {
          const cand = [];
          for (let i = 0; i < nin; i++) {
            const w = l.W[i][o] * (li === 0 && raw.in_scale ? raw.in_scale[i] : 1);
            cand.push([Math.abs(w), i, w]);
          }
          cand.sort((a, b) => b[0] - a[0]);
          for (const [aw, i, w] of cand.slice(0, per)) { list.push({ i, o, w }); wmax = Math.max(wmax, aw); }
        }
        for (const e of list) e.n = Math.abs(e.w) / wmax;
        return list;
      });
      this.resize();
    }

    resize() {
      const box = this.canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const W = Math.max(280, box.width);
      const narrow = W < 520;
      const groups = this.layout.groups;
      const row = narrow ? 11 : 12.5, gap = narrow ? 5 : 7;
      const H = Math.round(this.sizes[0] * row + (groups.length - 1) * gap + 30);
      this.canvas.style.height = H + "px";
      this.canvas.width = Math.round(W * dpr);
      this.canvas.height = Math.round(H * dpr);
      this.W = W; this.H = H; this.dpr = dpr; this.narrow = narrow;
      // node positions
      const labelW = narrow ? 92 : 170, outLabelW = narrow ? 64 : 120;
      const x0 = labelW, x3 = W - outLabelW;
      const xs = this.sizes.map((_, k) => x0 + ((x3 - x0) * k) / (this.sizes.length - 1));
      const pos = [];
      // inputs: grouped rows
      const inp = [];
      let y = 15;
      this.groupY = [];
      groups.forEach((g, gi) => {
        const yStart = y;
        for (let j = 0; j < g.items.length; j++) { inp.push([xs[0], y + row / 2]); y += row; }
        this.groupY.push({ name: g.name, y0: yStart, y1: y });
        if (gi < groups.length - 1) y += gap;
      });
      pos.push(inp);
      for (let k = 1; k < this.sizes.length; k++) {
        const n = this.sizes[k], list = [];
        const last = k === this.sizes.length - 1;
        const top = last ? H * 0.3 : 15, bot = last ? H * 0.7 : H - 15;
        for (let j = 0; j < n; j++) list.push([xs[k], n === 1 ? (top + bot) / 2 : top + ((bot - top) * j) / (n - 1)]);
        pos.push(list);
      }
      this.pos = pos;
      this.draw();
    }

    update(trace) { this.trace = trace; this.draw(); }

    colour(a, alpha) {
      const v = Math.max(-1, Math.min(1, a));
      const base = this.dark ? [70, 78, 74] : [214, 220, 214];
      const tgt = v >= 0 ? this.cPos : this.cNeg;
      const t = Math.abs(v);
      const c = base.map((b, i) => Math.round(b + (tgt[i] - b) * t));
      return `rgba(${c[0]},${c[1]},${c[2]},${alpha === undefined ? 1 : alpha})`;
    }

    draw() {
      const ctx = this.ctx, dpr = this.dpr || 1;
      if (!this.pos) return;
      this.cPos = hexToRgb(cssVar("--pos", "#2a7fd4"));
      this.cNeg = hexToRgb(cssVar("--neg", "#d9731a"));
      const bg = hexToRgb(cssVar("--bg", "#eeeeee"));
      this.dark = bg[0] + bg[1] + bg[2] < 3 * 110;
      const fg = cssVar("--fg", "#222"), muted = cssVar("--muted", "#666");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, this.W, this.H);
      const tr = this.trace;
      // edges
      for (let li = 0; li < this.edges.length; li++) {
        const A = this.pos[li], B = this.pos[li + 1];
        for (const e of this.edges[li]) {
          let alpha = 0.06 + 0.16 * e.n;
          let col = e.w >= 0 ? this.cPos : this.cNeg;
          if (tr) {
            // live: how much this edge pushes the target neuron right now
            const contrib = tr[li][e.i] * e.w;
            const strength = Math.min(1, Math.abs(contrib) / 1.0);
            alpha = 0.04 + 0.6 * strength * (0.4 + 0.6 * e.n);
            col = contrib >= 0 ? this.cPos : this.cNeg;
          }
          if (this.hoverN && !this._touches(this.hoverN, li, e)) alpha *= 0.25;
          ctx.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},${alpha.toFixed(3)})`;
          ctx.lineWidth = 0.6 + 1.6 * e.n;
          ctx.beginPath();
          ctx.moveTo(A[e.i][0], A[e.i][1]);
          ctx.lineTo(B[e.o][0], B[e.o][1]);
          ctx.stroke();
        }
      }
      // nodes
      for (let k = 0; k < this.pos.length; k++) {
        const r = k === 0 ? (this.narrow ? 3.6 : 4.3) : k === this.pos.length - 1 ? 9 : (this.narrow ? 4.2 : 5);
        this.pos[k].forEach(([x, y], j) => {
          const a = tr ? tr[k][j] : 0;
          ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
          ctx.fillStyle = this.colour(k === 0 ? Math.tanh(a) : a);
          ctx.fill();
          ctx.lineWidth = this.hoverN && this.hoverN[0] === k && this.hoverN[1] === j ? 2 : 0.8;
          ctx.strokeStyle = this.hoverN && this.hoverN[0] === k && this.hoverN[1] === j ? fg : "rgba(128,128,128,0.6)";
          ctx.stroke();
        });
      }
      // labels
      ctx.fillStyle = muted;
      ctx.font = `${this.narrow ? 11 : 12.5}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textBaseline = "middle"; ctx.textAlign = "right";
      const lx = this.pos[0][0][0] - 10;
      for (const g of this.groupY) {
        const name = g.name.replace("Abstand zum Fahrbahnrand", "Abstand Rand").replace("Weg zur Ausfahrt", this.narrow ? "zur Ausfahrt" : "Weg zur Ausfahrt");
        ctx.fillText(name, lx, (g.y0 + g.y1) / 2);
        if (g.y1 - g.y0 > 14) {
          ctx.strokeStyle = "rgba(128,128,128,0.45)"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(lx + 4, g.y0 + 2); ctx.lineTo(lx + 4, g.y1 - 2); ctx.stroke();
        }
      }
      ctx.textAlign = "left";
      const outs = this.pos[this.pos.length - 1];
      const names = this.layout.actions;
      ctx.fillStyle = fg;
      outs.forEach(([x, y], j) => {
        ctx.fillText((this.narrow ? (names[j] || "").replace("Gas / Bremse", "Gas") : names[j]) || `Ausgang ${j + 1}`, x + 14, y - (tr ? 7 : 0));
        if (tr) {
          ctx.fillStyle = muted;
          ctx.font = `${this.narrow ? 11 : 12}px ui-monospace, Menlo, monospace`;
          ctx.fillText((tr[tr.length - 1][j] >= 0 ? "+" : "") + tr[tr.length - 1][j].toFixed(2), x + 14, y + 8);
          ctx.font = `${this.narrow ? 11 : 12.5}px ui-sans-serif, system-ui, sans-serif`;
          ctx.fillStyle = fg;
        }
      });
      // column captions
      ctx.fillStyle = muted; ctx.textAlign = "center"; ctx.font = `11px ui-sans-serif, system-ui, sans-serif`;
      const caps = ["Eingänge", "Schicht 1", "Schicht 2", "Ausgänge"];
      this.pos.forEach((col, k) => ctx.fillText(this.narrow && k === 0 ? "" : caps[k] || "", col[0][0], this.H - 6));
    }

    _touches(h, li, e) {
      const [k, j] = h;
      return (k === li && e.i === j) || (k === li + 1 && e.o === j);
    }

    _hover(ev) {
      const r = this.canvas.getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      let best = null, bd = 12;
      this.pos.forEach((col, k) => col.forEach(([x, y], j) => {
        const d = Math.hypot(x - mx, y - my);
        if (d < bd) { bd = d; best = [k, j]; }
      }));
      const changed = JSON.stringify(best) !== JSON.stringify(this.hoverN);
      this.hoverN = best;
      if (changed) this.draw();
      if (this.opts.onHover) this.opts.onHover(best);
    }

    describe(h) {
      if (!h) return "";
      const [k, j] = h, tr = this.trace;
      const v = tr ? tr[k][j] : null;
      const val = v === null ? "" : ` = ${(v >= 0 ? "+" : "") + v.toFixed(2)}`;
      if (k === 0) {
        for (const g of this.layout.groups) for (const it of g.items) if (it.i === j) return `Eingang: ${g.name} · ${it.label}${val} (skaliert)`;
      }
      if (k === this.pos.length - 1) return `Ausgang: ${this.layout.actions[j]}${val}`;
      return `Neuron ${j + 1} in Schicht ${k}${val}`;
    }
  }

  SCX.NetView = NetView;
})();
