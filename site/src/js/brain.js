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
      const s = narrow ? 1 : Math.max(1, Math.min(1.7, W / 640));   // enlarged view: scale rows, dots and text
      this.s = s;
      const row = (narrow ? 11 : 12.5) * s, gap = (narrow ? 5 : 7) * s;
      const H = Math.round(this.sizes[0] * row + (groups.length - 1) * gap + 30 * s);
      this.canvas.style.height = H + "px";
      this.canvas.width = Math.round(W * dpr);
      this.canvas.height = Math.round(H * dpr);
      this.W = W; this.H = H; this.dpr = dpr; this.narrow = narrow;
      // node positions
      const labelW = narrow ? 92 : 170 * s, outLabelW = narrow ? 64 : 120 * s;
      const x0 = labelW, x3 = W - outLabelW;
      const xs = this.sizes.map((_, k) => x0 + ((x3 - x0) * k) / (this.sizes.length - 1));
      const pos = [];
      // inputs: grouped rows
      const inp = [];
      let y = 15 * s;
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
        const top = last ? H * 0.3 : 15 * s, bot = last ? H * 0.7 : H - 15 * s;
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
          ctx.lineWidth = (0.6 + 1.6 * e.n) * Math.sqrt(this.s || 1);
          ctx.beginPath();
          ctx.moveTo(A[e.i][0], A[e.i][1]);
          ctx.lineTo(B[e.o][0], B[e.o][1]);
          ctx.stroke();
        }
      }
      // nodes
      for (let k = 0; k < this.pos.length; k++) {
        const r = (k === 0 ? (this.narrow ? 3.6 : 4.3) : k === this.pos.length - 1 ? 9 : (this.narrow ? 4.2 : 5)) * (this.s || 1);
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
      const S = this.s || 1;
      const fnt = (n, nr, f) => `${(this.narrow ? nr : n) * S}px ${f || "ui-sans-serif, system-ui, sans-serif"}`;
      ctx.font = fnt(12.5, 11);
      ctx.textBaseline = "middle"; ctx.textAlign = "right";
      const lx = this.pos[0][0][0] - 10;
      for (const g of this.groupY) {
        const name = g.name.replace(SCX.t("brain.edge.long"), SCX.t("brain.edge.short")).replace(SCX.t("brain.exit.long"), this.narrow ? SCX.t("brain.exit.short") : SCX.t("brain.exit.long"));
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
        ctx.fillText((this.narrow ? (names[j] || "").replace(SCX.t("brain.out.accbrake"), SCX.t("brain.out.acc")) : names[j]) || SCX.t("brain.output", { n: j + 1 }), x + 14 * S, y - (tr ? 7 * S : 0));
        if (tr) {
          ctx.fillStyle = muted;
          ctx.font = fnt(12, 11, "ui-monospace, Menlo, monospace");
          ctx.fillText((tr[tr.length - 1][j] >= 0 ? "+" : "") + tr[tr.length - 1][j].toFixed(2), x + 14 * S, y + 8 * S);
          ctx.font = fnt(12.5, 11);
          ctx.fillStyle = fg;
        }
      });
      // column captions
      ctx.fillStyle = muted; ctx.textAlign = "center"; ctx.font = `${11 * S}px ui-sans-serif, system-ui, sans-serif`;
      const caps = [SCX.t("brain.cap.in"), SCX.t("brain.cap.l1"), SCX.t("brain.cap.l2"), SCX.t("brain.cap.out")];
      this.pos.forEach((col, k) => ctx.fillText(this.narrow && k === 0 ? "" : caps[k] || "", col[0][0], this.H - 6 * S));
    }

    _touches(h, li, e) {
      const [k, j] = h;
      return (k === li && e.i === j) || (k === li + 1 && e.o === j);
    }

    _hover(ev) {
      const r = this.canvas.getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      let best = null, bd = 12 * (this.s || 1);
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
        for (const g of this.layout.groups) for (const it of g.items) if (it.i === j) return SCX.t("brain.d.input", { name: g.name, label: it.label, val });
      }
      if (k === this.pos.length - 1) return SCX.t("brain.d.output", { name: this.layout.actions[j], val });
      return SCX.t("brain.d.neuron", { n: j + 1, layer: k, val });
    }
  }

  SCX.NetView = NetView;

  // ------------------------------------------------------------------ set-encoder view
  /* The set / attention brain (smartcrossing/policy_set.py) drawn as its real structure, top to
   * bottom: own values (ego) | up to K neighbour cards -> ONE shared mini-network (drawn once,
   * stacked like cards) -> attention per neighbour ("who matters now?", live bars + line widths)
   * -> mix (weighted sum) + own values (+ memory: the GRU state of a memory brain, a residual input of the first
   * layer) -> two layers -> outputs. Values come from the simulator's own forward pass (SmartCrossing.setTrace), so
   * the picture is exactly what drives the car. */
  const SVGNS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const nbShort = (key) => { const k = "nb." + key; const s = SCX.t(k); return s === k ? null : s; };  // short names of the neighbour features
  const sgn = (v) => (v >= 0 ? "+" : "−") + SCX.num(Math.abs(v), 2);

  /** Strongest `per` incoming weights per output neuron of a row-major W (nin x nout). */
  function topEdges(W, nin, nout, per, inScale) {
    const list = [];
    let wmax = 1e-9;
    for (let o = 0; o < nout; o++) {
      const cand = [];
      for (let i = 0; i < nin; i++) {
        const w = W[i * nout + o] * (inScale ? inScale[i] : 1);
        cand.push([Math.abs(w), i, w]);
      }
      cand.sort((a, b) => b[0] - a[0]);
      for (const [aw, i, w] of cand.slice(0, per)) { list.push({ i, o, w }); wmax = Math.max(wmax, aw); }
    }
    for (const e of list) e.n = Math.abs(e.w) / wmax;
    return list;
  }

  class SetView {
    /** host: an empty block element; layout: SC_DATA layout of the sim; pol: SmartCrossing.loadPolicy(set policy);
     *  opts.fmt(item, rawValue) formats an observation value, opts.onHover(text|null), opts.onFocus(k|null). */
    constructor(host, layout, pol, opts) {
      this.host = host;
      this.opts = Object.assign({ onHover: null, onFocus: null, fmt: null }, opts);
      this.hoverId = null; this.pinned = null; this.tr = null; this.obs = null; this.lastFocus = undefined;
      // Card identity: the set encoder is permutation-invariant (sum/attention over neighbours), so the
      // card a car sits on is free to choose. A car keeps its card (and number) while it stays among the
      // visible neighbours, instead of the distance-sorted slot order shuffling the numbers.
      this.cardUid = []; this.slotOfCard = []; this.cardOfSlot = [];
      this.sm = []; this.attTarget = null; this.animTs = null;
      this.setPolicy(pol, layout);
      new ResizeObserver(() => {
        const w = Math.round(host.getBoundingClientRect().width);
        if (w && Math.abs(w - this.W) > 2) this.build();
      }).observe(host);
      host.addEventListener("pointermove", (e) => this._hover(e.target));
      host.addEventListener("click", (e) => {
        const k = this._nbOf(this._idOf(e.target));
        this.pinned = k !== null && this.pinned !== k ? k : null;
        this._hover(e.target);
      });
      host.addEventListener("pointerleave", () => { this.hoverId = null; this.refresh(); if (this.opts.onHover) this.opts.onHover(null); });
    }

    setPolicy(pol, layout) {
      this.pol = pol; this.ar = pol.arch;
      if (layout) this.layout = layout;
      const ar = this.ar, P = pol.P, E = ar.n_ego, F = ar.F;
      const own = this.layout.groups.filter((g) => g.sub !== "neighbour");
      this.ownGroups = own.map((g) => g.items.length);
      this.ownItems = own.flatMap((g) => g.items.map((it) => Object.assign({ group: g.name }, it)));
      this.nbItems = this.layout.groups.filter((g) => g.sub === "neighbour").map((g) => g.items);
      const nbScale = pol.inScale ? pol.inScale.subarray(E, E + F) : null;
      const dims = [F, ...ar.phi], ts = [E + ar.n_extra + (ar.pool === "attn" ? ar.heads * ar.dv : ar.phi[ar.phi.length - 1]), ...ar.hidden, ar.n_out];
      this.phiEdges = ar.phi.map((_, l) => topEdges(P[`phi${l}.W`], dims[l], dims[l + 1], 2, l === 0 ? nbScale : null));
      // memory brains: the first layer reads [own values, mix, memory]; its weights = trunk0.W stacked on mem.out.W
      const Hm = (this.Hm = pol.mem || 0);
      let W0 = P["trunk0.W"];
      if (Hm) { W0 = new Float64Array((ts[0] + Hm) * ts[1]); W0.set(P["trunk0.W"], 0); W0.set(P["mem.out.W"], ts[0] * ts[1]); }
      this.trunkEdges = ts.slice(1).map((n, l) => topEdges(l === 0 ? W0 : P[`trunk${l}.W`], l === 0 ? ts[0] + Hm : ts[l], n, l === ts.length - 2 ? 6 : 2, null));
      this.nParams = pol.P ? Object.values(P).reduce((a, v) => a + v.length, 0) : 0;
      this.build();
    }

    /** Assign neighbour slots to cards by car id (uids[slot], -1 = empty): cars that stay keep their card,
     *  newcomers take a free card, cards of departed cars become free. Without ids: card = slot. */
    _assign(tr, uids) {
      const K = this.ar.K;
      const slotOfCard = new Array(K).fill(-1), cardOfSlot = new Array(K).fill(-1);
      if (tr && !uids) {
        for (let k = 0; k < K; k++) if (tr.present[k]) { slotOfCard[k] = k; cardOfSlot[k] = k; }
      } else if (tr) {
        const old = this.cardUid, now = new Map();
        for (let k = 0; k < K; k++) if (tr.present[k] && uids[k] >= 0) now.set(uids[k], k);
        const cardUid = new Array(K).fill(-1);
        for (let c = 0; c < K; c++) if (old[c] >= 0 && now.has(old[c])) cardUid[c] = old[c];
        const kept = new Set(cardUid);
        for (const u of now.keys()) {
          if (kept.has(u)) continue;
          const c = cardUid.indexOf(-1);           // always one free: at most K neighbours
          cardUid[c] = u; this.sm[c] = 0;          // a newcomer fades its bar in
        }
        this.cardUid = cardUid;
        cardUid.forEach((u, c) => { if (u >= 0) { slotOfCard[c] = now.get(u); cardOfSlot[now.get(u)] = c; } });
      }
      this.slotOfCard = slotOfCard; this.cardOfSlot = cardOfSlot;
    }

    /** Mean attention per card over the heads (0 for free cards); null before the first update. */
    attention() {
      const tr = this.tr;
      if (!tr || !tr.attn) return null;
      const K = this.ar.K, H = tr.attn.length, a = new Array(K).fill(0);
      for (let c = 0; c < K; c++) {
        const s = this.slotOfCard[c];
        if (s >= 0) for (const h of tr.attn) a[c] += h[s] / H;
      }
      return a;
    }

    /** For the road overlay, by neighbour slot: eased attention, card number and focus slot. */
    lineInfo() {
      const K = this.ar.K, w = new Array(K).fill(0), label = new Array(K).fill(0);
      for (let s = 0; s < K; s++) {
        const c = this.cardOfSlot[s];
        if (c >= 0) { w[s] = this.sm[c] || 0; label[s] = c + 1; }
      }
      const f = this.focus();
      return { w, label, focus: f === null ? null : this.slotOfCard[f] };
    }

    /** Display-only easing (about 150 ms) of bar heights and line thicknesses; call every frame. */
    animate(ts) {
      const dt = this.animTs === null ? 16 : Math.max(0, Math.min(100, ts - this.animTs));
      this.animTs = ts;
      const tg = this.attTarget;
      if (!tg) return;
      const f = 1 - Math.exp(-dt / 50);
      let moving = false;
      for (let c = 0; c < tg.length; c++) {
        const cur = this.sm[c] || 0, d = tg[c] - cur;
        if (Math.abs(d) > 5e-4) { this.sm[c] = cur + d * f; moving = true; } else this.sm[c] = tg[c];
      }
      if (moving) this._paintAtt();
    }

    _paintAtt() {
      const L = this.live, tr = this.tr, K = this.ar.K;
      if (!L || !tr) return;
      for (let c = 0; c < K; c++) {
        const a = this.sm[c] || 0, on = this.slotOfCard[c] >= 0;
        L.bars[c].setAttribute("height", (this.aH * a).toFixed(1));
        L.bars[c].setAttribute("y", (this.aBase - this.aH * a).toFixed(1));  // grows up from the baseline
        for (const l of [L.up[c], L.down[K + c]]) {
          l.style.strokeWidth = (0.6 + 7 * a).toFixed(2);
          l.style.strokeOpacity = on ? (0.2 + 0.8 * Math.min(1, a * 1.6)).toFixed(2) : "0.06";
        }
      }
    }

    /** Neighbour shown in the mini-network: hovered, else pinned, else the one with the most attention. */
    focus() {
      const h = this._nbOf(this.hoverId);
      if (h !== null) return h;
      if (this.pinned !== null) return this.pinned;
      const a = this.attention();
      if (!a) return null;
      let best = null;
      a.forEach((v, k) => { if (this.slotOfCard[k] >= 0 && (best === null || v > a[best])) best = k; });
      return best;
    }

    build() {
      const host = this.host, ar = this.ar, K = ar.K, F = ar.F, E = ar.n_ego;
      const hostW = Math.max(300, Math.round(host.getBoundingClientRect().width || 300));
      this.W = hostW;
      const W = Math.min(hostW, 700);                      // design width; wider hosts just scale the drawing up
      const shown = Math.min(hostW, 1000);                 // displayed width (enlarged view)
      const narrow = W < 460, fs = narrow ? 11 : 12;
      host.innerHTML = "";
      const svg = svgEl("svg", { width: shown, class: "setnet", role: "img", "aria-label": SCX.t("brain.set.aria") }, host);
      const gE = svgEl("g", {}, svg), gL = svgEl("g", {}, svg), gN = svgEl("g", {}, svg), gT = svgEl("g", {}, svg);
      const els = (this.els = {});
      const live = (this.live = { ego: [], nb: [], cards: [], phi: [], bars: [], pct: [], down: [], up: [], mix: [], mem: [], ego2: [], trunk: [], out: [], outTxt: [] });
      const text = (x, y, s, o = {}) => {
        const t = svgEl("text", { x, y, "font-size": o.size || fs, "text-anchor": o.anchor || "start", class: o.halo ? "halo" : "" }, gT);
        t.style.fill = o.fill || "var(--muted)";
        if (o.bold) t.setAttribute("font-weight", "600");
        t.textContent = s;
        return t;
      };
      const dot = (x, y, r, id, parent) => {
        const c = svgEl("circle", { cx: x, cy: y, r, "data-id": id }, parent || gN);
        c.style.stroke = "rgba(128,128,128,0.6)"; c.style.strokeWidth = "0.8";
        els[id] = c;
        return c;
      };
      const row = (n, x0, x1, y) => {
        const step = (x1 - x0) / n;
        return Array.from({ length: n }, (_, j) => [x0 + step * (j + 0.5), y]);
      };
      const edge = (A, B, e, parent) => {
        const l = svgEl("line", { x1: A[0], y1: A[1], x2: B[0], y2: B[1] }, parent || gE);
        l.style.stroke = e.w >= 0 ? "var(--pos)" : "var(--neg)";
        l.style.strokeOpacity = (0.07 + 0.22 * e.n).toFixed(3);
        l.style.strokeWidth = (0.5 + 1.1 * e.n).toFixed(2);
        return l;
      };
      const pad = 6, gut = 14, X0 = pad + gut, X1 = W - pad;
      const busX = pad + 4;
      // 1) own values
      text(X0, 13, SCX.t(narrow ? "brain.set.ego.short" : "brain.set.ego", { E }), { fill: "var(--fg)", bold: true });
      const nG = this.ownGroups.length, unit = (X1 - X0) / (E + 0.7 * (nG - 1));
      const rE = Math.min(5, unit * 0.36);
      let gx = X0, idx = 0;
      this.ownGroups.forEach((n) => {
        for (let j = 0; j < n; j++, idx++) live.ego.push(dot(gx + unit * (j + 0.5), 29, rE, `ego:${idx}`));
        gx += unit * (n + 0.7);
      });
      // 2) neighbour cards
      const cardTop = 62, colW = (X1 - X0) / K, cw = colW - (narrow ? 3 : 5), cellH = narrow ? 9 : 10;
      text(X0, cardTop - 7, SCX.t(narrow ? "brain.set.nb.short" : "brain.set.nb", { K, F }), { fill: "var(--fg)", bold: true });
      const rows = Math.ceil(F / 2), cardH = 16 + rows * cellH + 4;
      const colX = (k) => X0 + colW * (k + 0.5);
      for (let k = 0; k < K; k++) {
        const x = colX(k) - cw / 2;
        const g = svgEl("g", {}, gN);
        const r = svgEl("rect", { x, y: cardTop, width: cw, height: cardH, rx: 5, "data-id": `card:${k}` }, g);
        r.style.fill = "rgba(128,128,128,0.08)"; r.style.stroke = "rgba(128,128,128,0.55)";
        els[`card:${k}`] = r;
        const t = svgEl("text", { x: colX(k), y: cardTop + 12, "font-size": fs, "text-anchor": "middle", "data-id": `card:${k}` }, g);
        t.style.fill = "var(--fg)"; t.textContent = String(k + 1);
        const ds = [];
        for (let f = 0; f < F; f++) {
          const cx = colX(k) + (f % 2 ? 1 : -1) * Math.min(cw * 0.22, 9), cy = cardTop + 20 + Math.floor(f / 2) * cellH + cellH / 2;
          ds.push(dot(cx, cy, narrow ? 2.8 : 3.3, `nb:${k}:${f}`, g));
        }
        live.cards.push(g); live.nb.push(ds);
      }
      const cardBot = cardTop + cardH;
      // 3) the shared mini-network, drawn once (stacked cards behind it)
      const bx0 = X0 + (narrow ? 10 : 22), bx1 = X1 - (narrow ? 14 : 26), bTop = cardBot + 34;
      const bMid = (bx0 + bx1) / 2, rowsY = [bTop + 34, bTop + 64, bTop + 94], bBot = bTop + 106;
      for (const off of [8, 4]) {
        const s = svgEl("rect", { x: bx0 + off, y: bTop - off, width: bx1 - bx0, height: bBot - bTop, rx: 7 }, gE);
        s.style.fill = "var(--panel)"; s.style.stroke = "rgba(128,128,128,0.5)";
      }
      const box = svgEl("rect", { x: bx0, y: bTop, width: bx1 - bx0, height: bBot - bTop, rx: 7 }, gE);
      box.style.fill = "var(--panel)"; box.style.stroke = "rgba(128,128,128,0.75)";
      text(bx0 + 8, bTop + 15, SCX.t(narrow ? "brain.set.mini.short" : "brain.set.mini", { K }), { fill: "var(--fg)", bold: true });
      this.focusTxt = text(bx1 - 8, bTop + 15, "", { anchor: "end" });
      for (let k = 0; k < K; k++) {
        const l = svgEl("line", { x1: colX(k), y1: cardBot, x2: bMid + (colX(k) - bMid) * 0.45, y2: bTop - 9 }, gL);
        l.style.stroke = "var(--muted)"; l.style.strokeWidth = "1";
        live.down.push(l);
      }
      const phiPos = [row(F, bx0 + 10, bx1 - 10, rowsY[0]), ...ar.phi.map((n, l) => row(n, bx0 + 10, bx1 - 10, rowsY[Math.min(l + 1, 2)]))];
      this.phiEdges.forEach((list, l) => list.forEach((e) => edge(phiPos[l][e.i], phiPos[l + 1][e.o], e)));
      const rPhi = Math.min(3.4, ((bx1 - bx0 - 20) / Math.max(...ar.phi)) * 0.36);
      phiPos.forEach((ps, l) => live.phi.push(ps.map(([x, y], j) => dot(x, y, l === 0 ? (narrow ? 3.4 : 4.2) : rPhi, `phi:${l}:${j}`))));
      // 4) attention per neighbour
      const aTitle = bBot + 20, aTop = bBot + 30, aH = 44, aBase = aTop + aH;
      this.aH = aH; this.aBase = aBase;
      text(X0, aTitle, SCX.t(narrow ? "brain.set.att.short" : "brain.set.att"), { fill: "var(--fg)", bold: true, halo: true });
      const bw = Math.min(18, colW * 0.5);
      for (let k = 0; k < K; k++) {
        const l = svgEl("line", { x1: bMid + (colX(k) - bMid) * 0.45, y1: bBot, x2: colX(k), y2: aTop }, gL);
        l.style.stroke = "var(--c4)"; live.up.push(l);
        const hit = svgEl("rect", { x: colX(k) - colW / 2, y: aTop - 4, width: colW, height: aH + 20, "data-id": `att:${k}`, fill: "transparent" }, gN);
        els[`att:${k}`] = hit;
        const track = svgEl("rect", { x: colX(k) - bw / 2, y: aTop, width: bw, height: aH, rx: 2, "data-id": `att:${k}` }, gN);
        track.style.fill = "rgba(128,128,128,0.13)";
        const bar = svgEl("rect", { x: colX(k) - bw / 2, y: aBase, width: bw, height: 0, rx: 2, "data-id": `att:${k}` }, gN);
        bar.style.fill = "var(--c4)";
        live.bars.push(bar);
        live.pct.push(text(colX(k), aBase + 13, "–", { anchor: "middle", size: narrow ? 10 : 11 }));
      }
      // ego bus: own values go to the attention (the "question") and on to the combined layer
      const mixY = aBase + 52;
      const bus = svgEl("path", { d: `M${busX} 29 L${busX} ${mixY} L${X0} ${mixY}`, fill: "none" }, gE);
      bus.style.stroke = "var(--muted)"; bus.style.strokeDasharray = "3 3"; bus.style.strokeWidth = "1.2";
      const q = svgEl("path", { d: `M${busX} ${aTop + aH / 2} L${X0 + colW / 2 - bw / 2 - 3} ${aTop + aH / 2}`, fill: "none" }, gE);
      q.style.stroke = "var(--muted)"; q.style.strokeDasharray = "2 3";
      svgEl("path", { d: `M${busX} 29 L${X0 - 2} 29`, fill: "none" }, gE).style.stroke = "var(--muted)";
      // 5) combined: own values + mix (+ memory) -> trunk -> outputs
      const nMix = this.ar.pool === "attn" ? ar.heads * ar.dv : ar.phi[ar.phi.length - 1], Hm = this.Hm;
      const nIn = E + ar.n_extra + nMix, uIn = (X1 - X0) / (nIn + 2 + (Hm ? Hm + 2 : 0));
      const egoX1 = X0 + uIn * E, mixX0 = egoX1 + uIn * 2, mixX1 = Hm ? mixX0 + uIn * nMix : X1;
      const egoPos = row(E, X0, egoX1, mixY), mixPos = row(nMix, mixX0, mixX1, mixY);
      const memX0 = mixX1 + uIn * 2, memPos = Hm ? row(Hm, memX0, X1, mixY) : [];
      const mixMid = (mixX0 + mixX1) / 2;
      for (let k = 0; k < K; k++) {
        const l = svgEl("line", { x1: colX(k), y1: aBase + 18, x2: mixMid + (colX(k) - mixMid) * 0.3, y2: mixY - 6 }, gL);
        l.style.stroke = "var(--c4)"; live.down.push(l);
      }
      const rIn = Math.min(3.6, uIn * 0.4);
      egoPos.forEach(([x, y], j) => live.ego2.push(dot(x, y, rIn, `ego:${j}`)));
      mixPos.forEach(([x, y], j) => live.mix.push(dot(x, y, rIn, `mix:${j}`)));
      memPos.forEach(([x, y], j) => live.mem.push(dot(x, y, rIn, `mem:${j}`)));
      text((X0 + egoX1) / 2, mixY + 16, SCX.t(narrow ? "brain.set.me.short" : "brain.set.me"), { anchor: "middle", size: narrow ? 10 : 11 });
      text(mixMid, mixY + 16, SCX.t(narrow || Hm ? "brain.set.mix.short" : "brain.set.mix"), { anchor: "middle", size: narrow ? 10 : 11 });
      if (Hm) text((memX0 + X1) / 2, mixY + 16, SCX.t(narrow ? "brain.set.mem.short" : "brain.set.mem"), { anchor: "middle", size: narrow ? 10 : 11 });
      const trunkPos = [[...egoPos, ...mixPos, ...memPos]];
      let ty = mixY + 44;
      ar.hidden.forEach((n) => { trunkPos.push(row(n, X0, X1, ty)); ty += 34; });
      const outY = ty + 12, outPos = row(ar.n_out, X0, X1, outY);
      trunkPos.push(outPos);
      this.trunkEdges.forEach((list, l) => list.forEach((e) => edge(trunkPos[l][e.i], trunkPos[l + 1][e.o], e)));
      const rH = Math.min(3.4, ((X1 - X0) / Math.max(...ar.hidden)) * 0.38);
      ar.hidden.forEach((_, l) => live.trunk.push(trunkPos[l + 1].map(([x, y], j) => dot(x, y, rH, `tr:${l}:${j}`))));
      const names = this.layout.actions;
      outPos.forEach(([x, y], j) => {
        live.out.push(dot(x, y, 9, `out:${j}`));
        text(x, y + 24, names[j] || SCX.t("brain.output", { n: j + 1 }), { anchor: "middle", fill: "var(--fg)" });
        live.outTxt.push(text(x, y + 39, "", { anchor: "middle", size: narrow ? 10 : 11 }));
      });
      const H = outY + 46;
      svg.setAttribute("height", Math.round(H * shown / W));
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      this.lastFocus = undefined;
      this.refresh();
    }

    /** uids: ids of the cars in the neighbour slots (Crossing.neighbourUids); optional. */
    update(tr, obsRow, uids) { this.tr = tr; this.obs = obsRow; this._assign(tr, uids); this.refresh(); }

    refresh() {
      const L = this.live, tr = this.tr;
      if (!L) return;
      const cPos = hexToRgb(cssVar("--pos", "#2a7fd4")), cNeg = hexToRgb(cssVar("--neg", "#d9731a"));
      const bg = hexToRgb(cssVar("--bg", "#eeeeee")), dark = bg[0] + bg[1] + bg[2] < 3 * 110;
      const col = (a) => {
        const v = Math.max(-1, Math.min(1, a || 0)), base = dark ? [70, 78, 74] : [214, 220, 214], tgt = v >= 0 ? cPos : cNeg;
        const t = Math.abs(v), c = base.map((b, i) => Math.round(b + (tgt[i] - b) * t));
        return `rgb(${c[0]},${c[1]},${c[2]})`;
      };
      const fill = (el, v) => { el.style.fill = col(v); };
      const ar = this.ar, K = ar.K, F = ar.F, E = ar.n_ego;
      const att = this.attention(), foc = this.focus();
      if (tr) {
        L.ego.forEach((d, i) => fill(d, Math.tanh(tr.x[i])));
        L.ego2.forEach((d, i) => fill(d, Math.tanh(tr.x[i])));
        this.attTarget = att ? att.slice() : new Array(K).fill(0);
        for (let k = 0; k < K; k++) {
          const sl = this.slotOfCard[k], on = sl >= 0;
          L.cards[k].style.opacity = on ? "1" : "0.35";
          L.nb[k].forEach((d, f) => fill(d, on ? Math.tanh(tr.x[E + sl * F + f]) : 0));
          if (this.sm[k] === undefined) this.sm[k] = this.attTarget[k];
          L.pct[k].textContent = on ? `${Math.round(this.attTarget[k] * 100)} %` : "–";   // exact, not eased
          L.down[k].style.strokeOpacity = on ? "0.55" : "0.12";
          const card = this.els[`card:${k}`];
          card.style.stroke = k === foc ? "var(--c4)" : "rgba(128,128,128,0.55)";
          card.style.strokeWidth = k === foc ? "2.2" : "1";
        }
        this._paintAtt();
        const fsl = foc === null ? -1 : this.slotOfCard[foc], on = fsl >= 0;
        L.phi[0].forEach((d, f) => fill(d, on ? Math.tanh(tr.x[E + fsl * F + f]) : 0));
        for (let l = 1; l < L.phi.length; l++) L.phi[l].forEach((d, j) => fill(d, on ? tr.phi[fsl][l - 1][j] : 0));
        this.focusTxt.textContent = on ? SCX.t(this.W < 460 ? "brain.focus.short" : "brain.focus", { n: foc + 1 }) : SCX.t(this.W < 460 ? "brain.nofocus.short" : "brain.nofocus");
        L.mix.forEach((d, j) => fill(d, Math.tanh(tr.pooled[j])));
        L.mem.forEach((d, j) => fill(d, tr.mem ? tr.mem[j] : 0));
        L.trunk.forEach((ds, l) => ds.forEach((d, j) => fill(d, tr.trunk[l][j])));
        L.out.forEach((d, j) => fill(d, tr.out[j]));
        const word = [(v) => SCX.t(v >= 0 ? "brain.w.acc" : "brain.w.brake"), (v) => SCX.t(Math.abs(v) < 0.02 ? "man.straight" : v > 0 ? "man.left" : "man.right"), (v) => SCX.t(v >= 0 ? "brain.w.bright" : "brain.w.dark")];
        L.outTxt.forEach((t, j) => { t.textContent = `${sgn(tr.out[j])} · ${(word[j] || (() => ""))(tr.out[j])}`; });
      }
      for (const [id, el] of Object.entries(this.els)) {
        if (!id.startsWith("card:") && !id.startsWith("att:")) {
          el.style.stroke = id === this.hoverId ? "var(--fg)" : "rgba(128,128,128,0.6)";
          el.style.strokeWidth = id === this.hoverId ? "2" : "0.8";
        }
      }
      if (foc !== this.lastFocus) { this.lastFocus = foc; if (this.opts.onFocus) this.opts.onFocus(foc); }
      if (this.hoverId && this.opts.onHover) this.opts.onHover(this.describe(this.hoverId));
    }

    _idOf(t) { const e = t && t.closest ? t.closest("[data-id]") : null; return e ? e.getAttribute("data-id") : null; }
    _nbOf(id) {
      if (!id) return null;
      const m = /^(?:card|att|nb):(\d+)/.exec(id);
      return m ? +m[1] : null;
    }
    _hover(target) {
      const id = this._idOf(target);
      if (id === this.hoverId) return;
      this.hoverId = id;
      this.refresh();
      if (this.opts.onHover) this.opts.onHover(id ? this.describe(id) : null);
    }

    describe(id) {
      const tr = this.tr, [kind, a, b] = id.split(":"), ar = this.ar, E = ar.n_ego, F = ar.F;
      const raw = (i) => (this.obs ? this.obs[i] : null);
      const fmt = (it) => (this.opts.fmt && raw(it.i) !== null ? this.opts.fmt(it, raw(it.i)) : tr ? sgn(tr.x[it.i]) : "");
      const att = this.attention();
      if (kind === "ego") { const it = this.ownItems[+a]; return it ? SCX.t("brain.d.ego", { group: it.group, label: it.label, v: fmt(it) }) : ""; }
      const k = +a;
      if (kind === "nb" || kind === "card" || kind === "att") {
        if (tr && this.slotOfCard[k] < 0) return SCX.t("brain.d.nb.none", { n: k + 1 });
        const items = this.nbItems[k] || [];
        if (kind === "nb") {
          const it = items[+b];
          return it ? SCX.t("brain.d.nb.item", { n: k + 1, what: nbShort(it.key.split("_").slice(1).join("_")) || it.label, v: fmt(it) }) : "";
        }
        const by = (key) => items.find((it) => it.key.endsWith("_" + key));
        const pos = by("fwd") && by("left") ? SCX.t("brain.d.nb.pos", { fwd: fmt(by("fwd")), left: fmt(by("left")), speed: fmt(by("speed")) }) : "";
        const sl = this.slotOfCard[k], heads = tr && tr.attn && sl >= 0 ? SCX.t("brain.d.heads", { n: tr.attn.length, list: tr.attn.map((h) => SCX.pctInt(Math.round(h[sl] * 100))).join(", ") }) : "";
        return SCX.t("brain.d.nb.att", { n: k + 1, p: SCX.pctInt(att ? Math.round(att[k] * 100) : 0), heads, pos });
      }
      if (kind === "phi") {
        const f = this.focus(), fsl = f === null ? -1 : this.slotOfCard[f], v = tr && fsl >= 0 ? (+a === 0 ? Math.tanh(tr.x[E + fsl * F + +b]) : tr.phi[fsl][+a - 1][+b]) : null;
        const what = +a === 0 ? SCX.t("brain.d.phi.input", { what: nbShort((this.nbItems[0][+b] || { key: "" }).key.split("_").slice(1).join("_")) || +b + 1 }) : SCX.t("brain.d.layerneuron", { layer: a, n: +b + 1 });
        return SCX.t("brain.d.phi", { what, v: v !== null ? SCX.t("brain.d.phi.v", { v: sgn(v), n: f + 1 }) : "" });
      }
      if (kind === "mix") return SCX.t("brain.d.mix", { n: +a + 1, val: tr ? ` = ${sgn(tr.pooled[+a])}` : "" });
      if (kind === "mem") return SCX.t("brain.d.mem", { n: +a + 1, m: this.Hm, val: tr && tr.mem ? ` = ${sgn(tr.mem[+a])}` : "" });
      if (kind === "tr") return SCX.t("brain.d.layerneuron", { layer: +a + 1, n: +b + 1 }) + (tr ? ` = ${sgn(tr.trunk[+a][+b])}` : "");
      if (kind === "out") return SCX.t("brain.d.output", { name: this.layout.actions[+a], val: tr ? ` = ${sgn(tr.out[+a])}` : "" });
      return "";
    }
  }

  SCX.SetView = SetView;
})();
