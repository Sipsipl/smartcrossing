"""Stop gate (measurement only, docs/PLAN.md Roadmap E): one car alone drives straight towards a
standing obstacle on its path. Does it stop (or pass around it) without crashing?

Cases (one env each, all in one batched Sim): start speed (SPEEDS) x gap (GAPS: centre-to-centre
distance along the road, as runs/diag_rearend/solo_wreck.py; bumper gap = gap - CAR_LENGTH) x
position of the obstacle (POSITIONS: on the arm, at the box mouth, in the box) x obstacle type
(a wreck with hazard lights / a parked car without) x distractors (0 or N_DISTRACT wrecks on the
other half of the road beside the path, which fill neighbour slots).

Outcome per case (priority order): crash / offroad (at any time within `seconds`), stopped (came
to a standstill before the obstacle, no crash afterwards), passed (got past it without stopping and
without crash), other (none of these, e.g. still creeping). A case is `feasible` when full braking
from the first step stops the car in front of the obstacle.

  poetry run python -m smartcrossing.cli gate --driver policy:results/round1_x3.npz
"""

import itertools
import time

import numpy as np

from .sim import CAR_LENGTH, DISC_OFFSETS, DISC_R, DONE_CRASH, DONE_OFFROAD, STRAIGHT, Sim, SimConfig

SPEEDS = (6.0, 10.0, 14.0)  # m/s at the start
GAPS = (10.0, 20.0, 30.0)  # m, centre-to-centre distance car -> obstacle at the start
# obstacle centre (canonical frame: entry arm south, the car drives north on the right half)
POSITIONS = {"arm": -19.0, "near_box": -10.0, "in_box": 3.0}
OBST_HEADING = {"arm": np.pi / 2, "near_box": np.pi / 2, "in_box": 0.0}  # in the box: standing crosswise
TYPES = ("wreck", "parked")
N_DISTRACT = 3
STOP_V = 0.1  # m/s: standstill
OUTCOMES = ("stopped", "passed", "crash", "offroad", "other")


def cases():
    out = []
    for v0, gap, pos, kind, nd in itertools.product(SPEEDS, GAPS, POSITIONS, TYPES, (0, N_DISTRACT)):
        out.append(dict(v0=v0, gap=gap, pos=pos, kind=kind, distract=nd))
    return out


def braking_distance(cfg: SimConfig, v0: float) -> float:
    """Distance travelled with full braking from the first step (the sim's discrete kinematics)."""
    v = v0 - cfg.brake_max * cfg.dt * np.arange(1, int(np.ceil(v0 / (cfg.brake_max * cfg.dt))) + 1)
    return float(cfg.dt * np.maximum(v, 0.0).sum())


def contact_dist(pos: str) -> float:
    """Centre distance along the road at which the car touches the obstacle (disc model): aligned
    obstacle -> front disc meets rear disc; crosswise (in the box) -> front disc meets the side."""
    reach = DISC_OFFSETS[-1] + DISC_R  # front of the car
    return float(reach + (reach if OBST_HEADING[pos] == np.pi / 2 else DISC_R))


def feasible(cfg: SimConfig, v0: float, gap: float, pos: str) -> bool:
    """Full braking from the first step stops the car before it touches the obstacle."""
    return bool(braking_distance(cfg, v0) < gap - contact_dist(pos))


def setup(sim: Sim, cs: list[dict]):
    """Place car (slot 0), obstacle (slot 1) and distractors (slots 2..) of case b in env b."""
    cfg = sim.cfg
    lat = cfg.W / 2  # middle of the right half
    for b, c in enumerate(cs):
        yo = POSITIONS[c["pos"]]
        y0 = yo - c["gap"]
        sim.alive[b, 0] = True
        sim.x[b, 0], sim.y[b, 0], sim.th[b, 0], sim.v[b, 0] = lat, y0, np.pi / 2, c["v0"]
        sim.arm[b, 0], sim.man[b, 0], sim.uid[b, 0] = 0, STRAIGHT, b
        slots = [(1, lat, yo, OBST_HEADING[c["pos"]], c["kind"] == "parked")]
        slots += [(2 + k, -lat, max(y0 + 2.0 + 6.0 * k, -cfg.L + 2.0), -np.pi / 2, False) for k in range(c["distract"])]
        for n, x, y, th, parked in slots:
            sim.wreck[b, n] = sim.obstacle[b, n] = True
            sim.parked[b, n] = parked
            sim.wreck_left[b, n] = 10**6
            sim.x[b, n], sim.y[b, n], sim.th[b, n] = x, y, th
    sim._prev_dist = sim.dist_to_gate()


def stop_gate(spec: str, cfg_d: dict | None = None, seconds: float = 12.0, driver=None):
    """Run every case with driver `spec` (cli.make_driver) in the world cfg_d (traffic settings are
    replaced: no arrivals, no obstacle scenarios). ``driver``: optional factory sim -> driver used instead of
    cli.make_driver (``spec`` is then only the name in the summary), e.g. a brain without a checkpoint file.
    Returns (summary dict, per-case rows)."""
    from .cli import make_driver

    t0 = time.time()
    cs = cases()
    world = {**(cfg_d or {}), "demand": 0.0, "demand_range": None, "obstacle_envs": 0.0, "wreck_time": 1000.0,
             "n_slots": 2 + N_DISTRACT}
    sim = Sim(SimConfig(**world), len(cs), seed=0)
    drv = driver(sim) if driver else make_driver(spec, sim)  # (may apply the policy's interface / geometry to sim.cfg)
    setup(sim, cs)
    B = len(cs)
    out = np.full(B, "", dtype=object)
    stopped = np.zeros(B, bool)
    passed = np.zeros(B, bool)
    yo = np.array([POSITIONS[c["pos"]] for c in cs])
    for _ in range(int(round(seconds / sim.cfg.dt))):
        sim.step(drv.act())
        k = sim.done_kind[:, 0]
        live = out == ""
        out[live & (k == DONE_CRASH)] = "crash"
        out[live & (k == DONE_OFFROAD)] = "offroad"
        al = sim.alive[:, 0]
        before = al & (sim.y[:, 0] < yo)
        stopped |= (out == "") & before & ~passed & (np.abs(sim.v[:, 0]) < STOP_V)
        passed |= (out == "") & ((al & (sim.y[:, 0] > yo + CAR_LENGTH)) | (k > 0))
        if (out != "").all():
            break
    out[(out == "") & stopped] = "stopped"
    out[(out == "") & passed] = "passed"
    out[out == ""] = "other"
    rows = [dict(**c, outcome=o, feasible=feasible(sim.cfg, c["v0"], c["gap"], c["pos"])) for c, o in zip(cs, out)]
    return summarize_gate(rows, spec, time.time() - t0), rows


def _rates(rows):
    n = max(len(rows), 1)
    stop = sum(r["outcome"] == "stopped" for r in rows)
    safe = sum(r["outcome"] in ("stopped", "passed") for r in rows)
    return round(stop / n, 3), round(safe / n, 3)


def summarize_gate(rows, spec="", wall=0.0) -> dict:
    feas = [r for r in rows if r["feasible"]]
    s = {"driver": spec, "cases": len(rows), **{o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}}
    s["stop_rate"], s["safe_rate"] = _rates(rows)
    s["feasible"] = len(feas)
    s["stop_rate_feasible"], s["safe_rate_feasible"] = _rates(feas)
    for key, vals in (("pos", POSITIONS), ("kind", TYPES), ("distract", (0, N_DISTRACT)), ("v0", SPEEDS)):
        s[f"stop_by_{key}"] = {str(v): _rates([r for r in feas if r[key] == v])[0] for v in vals}
    s["wall_s"] = round(wall, 1)
    return s
