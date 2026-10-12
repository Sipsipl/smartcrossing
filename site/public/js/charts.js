/* Minimal SVG line chart with legend, direct labels and a hover crosshair/tooltip. */
(function () {
  "use strict";
  const SCX = (window.SCX = window.SCX || {});
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }

  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1;
    const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  /**
   * box: container element. opts: {series:[{name, color (css), points:[[x,y]], dash, width, dots}],
   *   xLabel, yLabel, xMin, xMax, yMin, yMax, yFmt, xFmt, markers:[{x, label}], height, directLabels}
   */
  SCX.lineChart = function (box, opts) {
    box.innerHTML = "";
    box.classList.add("chart-wrap");
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    for (const s of opts.series) {
      const sp = document.createElement("span");
      sp.innerHTML = `<i class="line-key${s.dash ? " dash" : ""}" style="border-color:${s.color}"></i>${s.name}`;
      legend.appendChild(sp);
    }
    if (opts.series.length > 1) box.appendChild(legend);
    const W = Math.max(300, box.clientWidth || 600), H = opts.height || 260;
    const m = { l: 46, r: opts.directLabels ? 92 : 14, t: opts.phases && opts.phases.length ? 34 : 14, b: 40 };
    if (W < 420 && opts.directLabels) m.r = 14; // phone: legend only
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": opts.aria || opts.yLabel || "Diagramm" }, box);
    const all = opts.series.flatMap((s) => s.points);
    const xMin = opts.xMin ?? Math.min(...all.map((p) => p[0])), xMax = opts.xMax ?? Math.max(...all.map((p) => p[0]));
    const yMin = opts.yMin ?? 0, yMax = opts.yMax ?? Math.max(...all.map((p) => p[1])) * 1.08;
    const X = (x) => m.l + ((x - xMin) / (xMax - xMin || 1)) * (W - m.l - m.r);
    const Y = (y) => H - m.b - ((y - yMin) / (yMax - yMin || 1)) * (H - m.t - m.b);
    const yf = opts.yFmt || ((v) => String(v)), xf = opts.xFmt || ((v) => String(v));
    for (const t of niceTicks(yMin, yMax, 4)) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: t === yMin ? "axis" : "grid" }, svg);
      const tx = el("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, svg); tx.textContent = yf(t);
    }
    for (const t of opts.xTicks || niceTicks(xMin, xMax, W < 420 ? 4 : 6)) {
      const tx = el("text", { x: X(t), y: H - m.b + 16, "text-anchor": "middle" }, svg); tx.textContent = xf(t);
    }
    if (opts.xLabel) { const t = el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, svg); t.textContent = opts.xLabel; }
    if (opts.yLabel) { const t = el("text", { x: 4, y: 10, "text-anchor": "start" }, svg); t.textContent = opts.yLabel; }
    for (const mk of opts.markers || []) {
      el("line", { x1: X(mk.x), x2: X(mk.x), y1: m.t, y2: H - m.b, class: "sep" }, svg);
      if (mk.label) {
        const right = X(mk.x) > W - m.r - 70;
        const t = el("text", { x: X(mk.x) + (right ? -4 : 4), y: m.t + 10, class: "sep-label", "text-anchor": right ? "end" : "start" }, svg);
        t.textContent = mk.label;
      }
    }
    // labelled background bands (e.g. which crossing a stretch of training used)
    (opts.bands || []).forEach((b, i) => {
      const x0 = X(Math.max(b.x0, xMin)), x1 = X(Math.min(b.x1, xMax));
      if (i % 2 === 0) el("rect", { x: x0, y: m.t, width: Math.max(0, x1 - x0), height: H - m.t - m.b, class: "band" }, svg);
      const tx = el("text", { x: (x0 + x1) / 2, y: m.t + (H - m.t - m.b) * 0.48, "text-anchor": "middle", class: "band-label" }, svg);
      tx.textContent = b.label;
    });
    // numbered phase markers: dashed line at x, numbered circle above the plot
    // (circles that would overlap are pushed right, with a short leader line)
    let lastCx = -1e9;
    for (const ph of opts.phases || []) {
      const x = X(ph.x), cx = Math.max(x, lastCx + 19), cy = 14;
      lastCx = cx;
      if (ph.x > xMin) el("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, class: "sep" }, svg);
      const g = el("g", { class: "phase" }, svg);
      el("path", { d: `M${cx},${cy + 8} L${x},${m.t}`, class: "sep", fill: "none" }, g);
      el("circle", { cx, cy, r: 8.5 }, g);
      const tx = el("text", { x: cx, y: cy + 4, "text-anchor": "middle" }, g); tx.textContent = String(ph.n);
      if (ph.title) { const ti = el("title", {}, g); ti.textContent = ph.title; }
    }
    const lastYs = [];
    for (const s of opts.series) {
      if (!s.points.length) continue;
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash ? "5 4" : "none" }, svg);
      if (s.dots) for (const p of s.points) el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: s.color, stroke: "var(--panel)", "stroke-width": 2 }, svg);
      if (opts.directLabels && W >= 420) {
        const p = s.points[s.points.length - 1];
        let y = Y(p[1]) + 4;
        // move away from labels already placed until it overlaps none (a few passes for crowded ends)
        for (let pass = 0; pass < 6 && lastYs.some((o) => Math.abs(o - y) < 14); pass++) {
          for (const o of lastYs) if (Math.abs(o - y) < 14) y = o + 14 * Math.sign(y - o || 1);
        }
        lastYs.push(y);
        const t = el("text", { x: X(p[0]) + 8, y, "text-anchor": "start" }, svg); t.textContent = s.name;
      }
    }
    // hover: crosshair + tooltip with the nearest point of every series
    const tip = document.createElement("div");
    tip.className = "chart-tip";
    box.appendChild(tip);
    const cross = el("line", { y1: m.t, y2: H - m.b, class: "axis", "stroke-width": 1, visibility: "hidden" }, svg);
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    const move = (ev) => {
      const r = svg.getBoundingClientRect();
      const sx = ((ev.clientX - r.left) / r.width) * W;
      const xv = xMin + ((sx - m.l) / (W - m.l - m.r)) * (xMax - xMin);
      const rows = [];
      let xs = null;
      for (const s of opts.series) {
        if (!s.points.length) continue;
        let best = s.points[0];
        for (const p of s.points) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
        if (s.tipMax && Math.abs(best[0] - xv) > s.tipMax) continue;
        if (xs === null) xs = best[0];
        rows.push(`<div class="row"><i style="background:${s.color}"></i>${s.name}: <b>${yf(best[1], true)}</b></div>`);
      }
      if (!rows.length) return;
      cross.setAttribute("x1", X(xs)); cross.setAttribute("x2", X(xs)); cross.setAttribute("visibility", "visible");
      tip.innerHTML = `<div class="muted">${opts.xTipFmt ? opts.xTipFmt(xs) : xf(xs)}</div>` + rows.join("");
      tip.style.display = "block";
      const bx = box.getBoundingClientRect();
      const left = ev.clientX - bx.left + 12;
      tip.style.left = Math.min(left, bx.width - tip.offsetWidth - 4) + "px";
      tip.style.top = ev.clientY - bx.top - tip.offsetHeight - 10 + "px";
    };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
    return svg;
  };

  /** Moving average for noisy learning curves. */
  SCX.smooth = function (pts, k) {
    const out = [];
    for (let i = 0; i < pts.length; i++) {
      let s = 0, n = 0;
      for (let j = Math.max(0, i - k); j <= Math.min(pts.length - 1, i + k); j++) { s += pts[j][1]; n++; }
      out.push([pts[i][0], s / n]);
    }
    return out;
  };

})();
