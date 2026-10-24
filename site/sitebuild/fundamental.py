"""Measurements the site shows besides the bench: the stop gate of a brain (measured at build time) and the
fundamental diagram + scatter cloud (read from the deadlock study's committed results)."""

import json
import re

from smartcrossing.gates import stop_gate

from .bench import policy_driver
from .context import Context
from .discover import Brain

FUNDAMENTAL_DRIVERS = ("brain", "light", "reservation")  # curves of the fundamental diagram on the site
START_DEMAND_MAX = 0.5  # max of the start page demand slider
FUNDAMENTAL_FIELDS = ("demand", "density", "throughput", "density_veh_per_km", "demand_per_min", "deadlock_time_frac")


def gate_of(ctx: Context, cid: str | None) -> dict | None:
    """Stop-gate rates (feasible cases) of a checkpoint, measured here (smartcrossing/gates.py, seconds):
    {"safe": share that stop or pass without a crash, "stop": share that stop in front of a single obstacle,
    "block": share that stop in front of a full-width blockade (nothing can be passed)}.
    The checkpoint file in runs/ if it exists, else (snapshot brains: no checkpoint file) its weights in its
    own physics. None for an unknown checkpoint."""
    c = ctx.found.get(cid) if cid else None
    if c is None:
        return None
    if cid not in ctx.gates:
        if c["file"] and (ctx.root / c["file"]).is_file():
            summary, _ = stop_gate(f"policy:{ctx.root / c['file']}")
        else:
            payload = Brain(c["policy"]).payload()
            summary, _ = stop_gate(cid, c["physics"], driver=lambda sim: policy_driver(sim, payload))
        ctx.gates[cid] = {"safe": summary["safe_rate_feasible"], "stop": summary["stop_rate_feasible"],
                          "block": summary["block_stop_rate_feasible"], "cases": summary["feasible"]}
    return ctx.gates[cid]


def gate_text(ctx: Context, text: str | None, cid: str | None) -> str | None:
    """Fill {safe} / {stop} / {block} (whole percent of the stop gate of checkpoint cid) in a run text."""
    keys = ("safe", "stop", "block")
    if text is None or not any("{" + k + "}" in text for k in keys):
        return text
    g = gate_of(ctx, cid)
    for k in keys:  # checkpoint not available (e.g. in tests): no number rather than a made-up one
        text = text.replace("{" + k + "}", "?" if g is None else str(round(g[k] * 100)))
    return text


def fundamental_data(ctx: Context):
    """Fundamental diagram (throughput vs cars on the map per driver and demand) from the deadlock study's
    results/deadlock_fundamental.json, reduced to what the site plots, plus the start value of the start page's
    traffic slider; None if the file is missing."""
    path = ctx.root / "results" / "deadlock_fundamental.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    drivers = {k: [{f: p[f] for f in FUNDAMENTAL_FIELDS} for p in raw["drivers"][k]["points"]]
               for k in FUNDAMENTAL_DRIVERS if k in raw["drivers"]}
    out = {"drivers": drivers, "cloud": cloud_data(ctx), "setup": setup_of(raw, ctx)}
    if "brain" in drivers:  # start value of the traffic slider = the brain's capacity: its maximum mean throughput
        cap = max(p["throughput"] for p in drivers["brain"])
        out["max_throughput"] = round(cap, 1)
        out["start_demand"] = start_demand(cap)
    return out


def setup_of(raw: dict, ctx: Context) -> dict:
    """Measurement setup of the diagram for its footnote, read from the two result files' meta texts (envs, seconds per
    run, traffic seeds, warm-up, cloud window / step); fields that cannot be read are left out."""
    meta = raw.get("meta", {})
    out = {}
    m = re.search(r"envs \[(\d+)\] x seconds \[([\d.]+)\], traffic seeds \[([^\]]*)\]", meta.get("protocol", ""))
    if m:
        out.update(envs=int(m.group(1)), seconds=float(m.group(2)), seeds=len(m.group(3).split(",")))
    m = re.search(r"after a (\d+) s warm-up", meta.get("density", ""))
    if m:
        out["warmup"] = int(m.group(1))
    path = ctx.root / "results" / "fundamental_cloud.json"
    if path.exists():
        cm = json.loads(path.read_text(encoding="utf-8")).get("meta", {}).get("throughput", "")
        m = re.search(r"(\d+)-s sliding window, one every (\d+) s", cm)
        if m:
            out.update(window=int(m.group(1)), step=int(m.group(2)))
    return out


def start_demand(cap_per_min: float) -> float:
    """Arrivals /s/arm whose demand equals cap_per_min cars/min (x 240), on the slider's 0.01 grid, at most its
    maximum."""
    return min(round(cap_per_min / 240, 2), START_DEMAND_MAX)


def cloud_data(ctx: Context) -> dict:
    """Scatter cloud behind the fundamental diagram (single measured 60-s windows per driver, from
    results/fundamental_cloud.json: [density, throughput] pairs); {} if the file is missing."""
    path = ctx.root / "results" / "fundamental_cloud.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: [[p[0], p[1]] for p in v["points"]] for k, v in raw["drivers"].items() if k in FUNDAMENTAL_DRIVERS}
