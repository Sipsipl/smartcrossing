"""Model selection over the checkpoints of training runs (gate filter + quick traffic score):

    poetry run python -m smartcrossing.cli select --run runs/r5_K_s1 --run runs/r5_K_s2 --every 2 \
        --workers 4 --out results/x_select.jsonl [--copy-to results/x.npz]

Per checkpoint (ckpt_XXXX.npz; the first and last one are always kept): stop gate (gates.py) and the bench
at protocol.SELECT_DEMANDS x SELECTION_SEEDS x ENVS x SECONDS (a subset of the evalset protocol), via evalset.evalset.
Rule per run: among checkpoints with ``safe_rate_feasible >= gate_min`` take the one with the highest
throughput at 0.5 subject to crash+off/1000 at 0.3 <= ref * crash_tol + crash_slack (ref = the run's first
checkpoint, or ``--ref``); the best checkpoint without any constraint is reported for contrast.
Output JSONL: one ``"kind": "row"`` per checkpoint, one ``"kind": "selection"`` per run (+ overall).
"""

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from . import evalset
from .protocol import ENVS, SECONDS, SELECT_DEMANDS, SELECTION_SEEDS

GATE_MIN, CRASH_TOL, CRASH_SLACK = 0.95, 1.1, 1.0  # slack: absolute crash+off/1000 (so a ref of 0 is not a hard 0)


def checkpoints(run, every: int = 1) -> list[Path]:
    """ckpt_*.npz of a run dir sorted by iteration; every N-th, always including the first and the last."""
    cks = sorted(Path(run).glob("ckpt_*.npz"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    keep = {0, len(cks) - 1} | set(range(0, len(cks), max(every, 1)))
    return [c for i, c in enumerate(cks) if i in keep]


def pick(rows: list[dict], ref: dict, gate_min: float = GATE_MIN, crash_tol: float = CRASH_TOL,
         crash_slack: float = CRASH_SLACK) -> dict:
    """The selection rule on rows (see module doc). Returns {selected, unconstrained, crash_limit}."""
    limit = ref["crash"]["0.3"] * crash_tol + crash_slack
    ok = [r for r in rows if r["safe_rate_feasible"] >= gate_min and r["crash"]["0.3"] <= limit]
    best = lambda rs: max(rs, key=lambda r: r["thr"]["0.5"], default=None)  # noqa: E731
    return {"selected": best(ok), "unconstrained": best(rows), "crash_limit": round(limit, 2)}


def aggregate(lines: list[dict], specs: list[str]) -> list[dict]:
    """One row per checkpoint spec: gate rates, mean thr / crash per demand over the seeds."""
    rows = []
    for spec in specs:
        ls = [x for x in lines if x["driver"] == spec]
        g = next(x for x in ls if x["kind"] == "gate")
        bench = [x for x in ls if x["kind"] == "bench"]
        ds = sorted({x["demand"] for x in bench})

        def mean(key, d):
            return round(float(np.mean([x[key] for x in bench if x["demand"] == d])), 2)

        rows.append({"kind": "row", "ckpt": spec.removeprefix("policy:"),
                     "iter": int(m.group(1)) if (m := re.search(r"ckpt_(\d+)", spec)) else None,
                     "safe_rate_feasible": g["safe_rate_feasible"], "stop_rate_feasible": g["stop_rate_feasible"],
                     "thr": {str(d): mean("throughput_per_min", d) for d in ds},
                     "crash": {str(d): mean("crash_cars_per_1000", d) for d in ds}})
    return rows


def table(rows, marks=None) -> str:
    marks = marks or {}
    f = lambda v: f"{'-':>7}" if v is None else f"{v:>7.1f}"  # noqa: E731  (demand not measured)
    out = [f"{'ckpt':<28}{'gate':>6}{'stop':>6}{'thr.3':>7}{'thr.5':>7}{'cr.15':>7}{'cr.3':>7}{'cr.5':>7}"]
    for r in rows:
        name = Path(r["ckpt"]).parent.name + "/" + Path(r["ckpt"]).stem
        out.append(f"{name:<28}{r['safe_rate_feasible']:>6.2f}{r['stop_rate_feasible']:>6.2f}"
                   f"{f(r['thr'].get('0.3'))}{f(r['thr'].get('0.5'))}{f(r['crash'].get('0.15'))}"
                   f"{f(r['crash'].get('0.3'))}{f(r['crash'].get('0.5'))}  {marks.get(r['ckpt'], '')}")
    return "\n".join(out)


def select(runs, every: int = 1, gate_min: float = GATE_MIN, crash_tol: float = CRASH_TOL,
           crash_slack: float = CRASH_SLACK, ref: str | None = None, demands=SELECT_DEMANDS, seeds=SELECTION_SEEDS,
           envs: int = ENVS, seconds: float = SECONDS, workers: int = 1, out=None, copy_to=None,
           cfg_d: dict | None = None, log=print) -> dict:
    missing = {0.3, 0.5} - {float(d) for d in demands}
    if missing:  # the rule needs the crash rate at 0.3 and the throughput at 0.5
        raise SystemExit(f"select: the demands must include 0.3 and 0.5 (missing: {sorted(missing)})")
    per_run = {str(r): [str(c) for c in checkpoints(r, every)] for r in runs}
    if empty := [r for r, cs in per_run.items() if not cs]:
        raise SystemExit(f"select: no ckpt_*.npz in {empty}")
    t0 = time.time()
    specs = [f"policy:{c}" for cs in per_run.values() for c in cs]
    ref_spec = f"policy:{ref}" if ref else None
    if ref_spec and ref_spec not in specs:
        specs.append(ref_spec)
    lines = evalset.evalset(specs, None, cfg_d, demands=demands, seeds=seeds, envs=envs, seconds=seconds,
                            workers=workers, log=lambda *_: None)
    rows = {r["ckpt"]: r for r in aggregate(lines, specs)}
    out_lines, sels = [], {}
    for run, cs in per_run.items():
        rrows = [rows[c] for c in cs]
        sel = pick(rrows, rows[ref] if ref else rrows[0], gate_min, crash_tol, crash_slack)
        sels[run] = sel
        marks = {sel["unconstrained"]["ckpt"]: "<- best unconstrained"}
        if sel["selected"]:
            marks[sel["selected"]["ckpt"]] = "<- selected" + (" (= best unconstrained)" if marks.get(
                sel["selected"]["ckpt"]) else "")
        log(f"== {run}: gate >= {gate_min}, crash@0.3 <= {sel['crash_limit']} (ref {'given' if ref else 'first ckpt'})")
        log(table(rrows, marks))
        out_lines += [{**r, "run": run} for r in rrows]
        out_lines.append({"kind": "selection", "run": run, "gate_min": gate_min, "crash_tol": crash_tol,
                          "crash_slack": crash_slack, "crash_limit": sel["crash_limit"],
                          "reference": ref or rrows[0]["ckpt"], "selected": sel["selected"],
                          "unconstrained": sel["unconstrained"]})
    cands = [s["selected"] for s in sels.values() if s["selected"]]
    overall = max(cands, key=lambda r: r["thr"]["0.5"], default=None)
    out_lines.append({"kind": "overall", "selected": overall})
    log(f"overall selected: {overall['ckpt'] if overall else 'none (no checkpoint passes gate and crash limit)'}")
    if ref:
        log("reference:\n" + table([rows[ref]]))
        out_lines.append({**rows[ref], "run": "reference"})
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(x) + "\n" for x in out_lines)
    if copy_to and overall:
        Path(copy_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(overall["ckpt"], copy_to)
        log(f"copied {overall['ckpt']} -> {copy_to}")
    log(f"select: {len(specs)} checkpoints in {time.time() - t0:.0f} s")
    return {"selections": sels, "overall": overall, "rows": rows}
