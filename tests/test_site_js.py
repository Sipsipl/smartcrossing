"""Small pure helpers of the website's JavaScript, run with node: the live dot's measurement window
(js/fdwindow.js) against the Python definition of the cloud (deadlock.cloud_windows), and the percent format
SCX.pct (js/i18n.js) at its edges."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from smartcrossing import deadlock

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "site" / "src" / "js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def node(script: str, tmp_path, payload=None):
    """Run `script` with node (PAYLOAD = payload); returns the JSON it prints."""
    (tmp_path / "in.json").write_text(json.dumps(payload), encoding="utf-8")
    f = tmp_path / "t.js"
    f.write_text("const PAYLOAD = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));\n" + script,
                 encoding="utf-8")
    res = subprocess.run([NODE, str(f), str(tmp_path / "in.json")], capture_output=True, text=True, encoding="utf-8",
                         timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_live_dot_window_equals_the_cloud_windows(tmp_path):
    """Random per-step car counts and exits: the JS point at every cloud window end equals cloud_windows, and there
    is no point before the first full window after the warm-up."""
    dt, T = 0.1, 1500
    rng = np.random.default_rng(3)
    cars = rng.integers(0, 40, size=(T, 1)).astype(float)
    exits = np.concatenate([[0], np.cumsum(rng.random(T) < 0.03)])[:, None].astype(float)  # (T + 1, 1)
    py = deadlock.cloud_windows(cars, exits, dt)
    samples = [[round(i * dt, 10), float(cars[i, 0])] for i in range(T)]  # state before step i
    exit_times = [round((i + 1) * dt, 10) for i in range(T) for _ in range(int(exits[i + 1, 0] - exits[i, 0]))]
    ends = [t for t, *_ in py]
    js = node(f"""
const W = require({json.dumps(str(JS / "fdwindow.js"))});
const out = {{ consts: [W.WARMUP_S, W.WINDOW_S, W.firstAt()], at: PAYLOAD.ends.map((t) => W.point(PAYLOAD.samples, PAYLOAD.exits, t)),
  early: [89.9, 60, 0].map((t) => W.point(PAYLOAD.samples, PAYLOAD.exits, t)) }};
console.log(JSON.stringify(out));""", tmp_path, {"samples": samples, "exits": exit_times, "ends": ends})
    assert js["consts"] == [deadlock.WARMUP, deadlock.CLOUD_WINDOW_S, deadlock.WARMUP + deadlock.CLOUD_WINDOW_S]
    assert len(ends) == 7 and ends[0] == pytest.approx(90.0)
    for (t, dens, thr), p in zip(py, js["at"]):
        assert p["cars"] == pytest.approx(dens[0], abs=1e-9) and p["thr"] == pytest.approx(thr[0], abs=1e-9), t
    assert js["early"] == [None, None, None]


def test_pct_edges(tmp_path):
    """Whole percent, one truncated decimal from 99 % (never rounded up to 100), 100 only for exactly 1."""
    vals = [None, 0, 0.5, 0.984, 0.99, 0.999, 0.9995, 0.99999, 1]
    out = node(f"""
const vm = require("vm"), fs = require("fs");
const res = {{}};
for (const lang of ["de", "en"]) {{
  const ctx = {{ location: {{ search: "?lang=" + lang }}, localStorage: {{ setItem() {{}}, getItem() {{ return null; }} }},
    navigator: {{ language: "de-DE" }}, document: {{ documentElement: {{}}, querySelectorAll: () => [] }} }};
  ctx.window = ctx;
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync({json.dumps(str(JS / "i18n.js"))}, "utf8"), ctx);
  res[lang] = PAYLOAD.map((v) => ctx.SCX.pct(v));
}}
console.log(JSON.stringify(res));""", tmp_path, vals)
    assert out["de"] == ["–", "0 %", "50 %", "98 %", "99,0 %", "99,9 %", "99,9 %", "99,9 %", "100 %"]
    assert out["en"] == ["–", "0%", "50%", "98%", "99.0%", "99.9%", "99.9%", "99.9%", "100%"]


def test_compare_words_match_numbers(tmp_path):
    """SCX.compare: 'more'/'fewer' only for >= 25 % AND >= minAbs difference, else 'same' (symmetric)."""
    cases = [[1, 4, 1], [4, 6, 1], [1, 1.5, 1], [0, 0, 1], [30, 20, 1], [96.2, 86.7, 1], [20.5, 4.3, 1], [5, 5.9, 1]]
    out = node(f"""
const vm = require("vm"), fs = require("fs");
const ctx = {{ location: {{ search: "?lang=de" }}, localStorage: {{ setItem() {{}}, getItem() {{ return null; }} }},
  navigator: {{ language: "de-DE" }}, document: {{ documentElement: {{}}, querySelectorAll: () => [] }} }};
ctx.window = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync({json.dumps(str(JS / "i18n.js"))}, "utf8"), ctx);
console.log(JSON.stringify(PAYLOAD.map(([c, p, m]) => ctx.SCX.compare(c, p, m))));""", tmp_path, cases)
    assert out == ["fewer", "fewer", "same", "same", "more", "same", "more", "same"]
